"""服务器端批量推理：用 LoRA 微调后的 Qwen3-32B 跑带标签数据，产出可本地归因的预测文件。

设计目标：一次开机把后续所有离线分析（丢分归因 / 温度扫描 / 自洽投票）需要的素材全部落盘，
避免为同一批数据二次开服务器。坏输出会显式重问并记录，绝不静默丢样本。

用法（服务器上）:
    # 1) 贪心解码，最快；先拿它做归因
    python src/infer_lora.py --data data/val.jsonl \
        --model /path/Qwen3-32B --adapter /path/lora-ckpt \
        --prompt-file /path/train_instruction.txt \
        --out preds/val_lora_greedy.jsonl --raw-dir runs/lora_greedy/raw

    # 2) 同一次开机里补 3 次高温采样（后台跑，供本地免费做投票/温度对比）
    for k in 1 2 3; do
      python src/infer_lora.py --data data/val.jsonl \
        --model /path/Qwen3-32B --adapter /path/lora-ckpt \
        --prompt-file /path/train_instruction.txt --temperature 0.8 --seed 20261012$k \
        --out preds/val_lora_s$k.jsonl --raw-dir runs/lora_s$k/raw
    done

--prompt-file 必须是你训练时逐字使用的那套 system/instruction，不能现场改词。
支持两种格式：JSON {"system": "...", "instruction": "..."}，或纯文本（整体当 instruction）。
"""

import argparse
import json
import os
import re
import sys
import time

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)
STANCES = {"support", "oppose", "neutral"}
DEFAULT_SYS = "You are a helpful assistant."


def done_path(out, sid):
    return f"{out}.{sid}.done"


def bad_path(out, sid):
    return f"{out}.{sid}.bad"


def is_settled(out, sid):
    return os.path.exists(done_path(out, sid)) or os.path.exists(bad_path(out, sid))


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_input(sample):
    """与 convert_to_alpaca.format_documents 保持逐字一致，否则训练/推理分布错位。"""
    parts = ["以下文档按时间顺序排列："]
    for index, doc in enumerate(sample["docs"], 1):
        text = str(doc.get("full_text") or "").strip()
        parts.append(
            f"<<<DOC {index} doc_id={doc.get('doc_id', '')} "
            f"doc_type={doc.get('doc_type', '')} publish_date={doc.get('publish_date', '')}>>>"
        )
        parts.append(text)
        parts.append("<<<END_DOC>>>")
    return "\n".join(parts)


def load_prompts(path):
    with open(path, encoding="utf-8") as fh:
        blob = fh.read()
    try:
        obj = json.loads(blob)
        return str(obj.get("system", "")), str(obj.get("instruction", ""))
    except json.JSONDecodeError:
        return "", blob.strip()


def parse_output(text):
    """模型原文 -> (payload, error)。容忍代码围栏与前后杂字。"""
    body = FENCE.sub("", text.strip()).strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None, "输出中找不到 JSON 对象"
    try:
        obj = json.loads(body[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"JSON 解析失败: {exc}"

    issues = obj.get("issue_list")
    if not isinstance(issues, list) or not issues:
        return None, "issue_list 缺失或为空"
    future = obj.get("future_argument")
    future = [str(f).strip() for f in future] if isinstance(future, list) else []

    clean, bad = [], []
    for it in issues:
        if not isinstance(it, dict):
            continue
        chain = it.get("argument_chain")
        if isinstance(chain, str):
            chain = [chain]
        name = str(it.get("issue_name") or "").strip()
        stance = str(it.get("stance") or "").strip()
        segs = [str(s).strip() for s in (chain or []) if str(s).strip()]
        if not name or stance not in STANCES or not segs:
            bad.append(f"name={name!r} stance={stance!r} chain={len(segs)}")
            continue
        clean.append({"issue_name": name, "stance": stance, "argument_chain": segs})
    if not clean:
        return None, f"issue_list 无有效元素（{bad[:2]}）"
    return {"issue_list": clean, "future_argument": future}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="带标签数据，如 data/val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="基座路径，或已合并 LoRA 的权重路径")
    ap.add_argument("--adapter", default="", help="LoRA adapter 目录；留空表示 --model 已合并")
    ap.add_argument("--prompt-file", required=True, help="训练时逐字使用的 system/instruction")
    ap.add_argument("--raw-dir", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--tensor-parallel", type=int, default=0, help="0=按可见 GPU 数自动")
    ap.add_argument("--repairs", type=int, default=2, help="坏输出重问次数")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # noqa: E402

    samples = read_jsonl(args.data)
    if args.limit:
        samples = samples[:args.limit]
    sys_prompt, instruction = load_prompts(args.prompt_file)
    todo = [s for s in samples if not is_settled(args.out, s["sample_id"])]

    # 先按字符数粗筛超长输入，避免 vLLM 直接报错中断整轮（宁可显式跳过也不静默截断）
    too_long = [s for s in todo if len(instruction) + len(format_input(s)) > args.max_model_len * 2]
    for s in too_long:
        print(f"[跳过] {s['sample_id']} 输入过长，需调大 --max-model-len", file=sys.stderr)
        open(bad_path(args.out, s["sample_id"]), "w").close()
    todo = [s for s in todo if s not in too_long]

    tp = args.tensor_parallel or max(1, (os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") + 1)
                                     if os.environ.get("CUDA_VISIBLE_DEVICES") else 1)
    kwargs = dict(model=args.model, tensor_parallel_size=tp, max_model_len=args.max_model_len,
                  dtype="bfloat16", trust_remote_code=True, gpu_memory_utilization=0.90)
    if args.adapter:
        from vllm.lora.request import LoRARequest  # noqa: E402
        kwargs["enable_lora"] = True
        lora = LoRARequest("task_adapter", 1, args.adapter)
    else:
        lora = None
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_new_tokens, seed=args.seed)

    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)

    def messages_for(sample, bad_text=None):
        user = f"{instruction}\n\n{format_input(sample)}" if instruction else format_input(sample)
        msgs = [{"role": "system", "content": sys_prompt or DEFAULT_SYS},
                {"role": "user", "content": user}]
        if bad_text is not None:
            msgs += [{"role": "assistant", "content": bad_text[:1500]},
                     {"role": "user", "content": "上一条输出不是合法 JSON 或字段缺失/取值非法。"
                                                 "请重新只输出完整合法的 JSON，不要解释、不要代码围栏。"}]
        return msgs

    problems, t0, tries, last_bad = [], time.time(), {}, {}
    remaining = list(todo)
    while remaining:
        progressed = False
        for i in range(0, len(remaining), args.chunk):
            batch = remaining[i:i + args.chunk]
            outs = llm.chat([messages_for(s, last_bad.get(s["sample_id"])) for s in batch],
                            params, lora_request=lora)
            progressed = True
            for s, out in zip(batch, outs):
                sid, text = s["sample_id"], out.outputs[0].text
                if args.raw_dir:
                    with open(os.path.join(args.raw_dir, f"{sid}.retry{tries.get(sid, 0)}.txt"),
                              "w", encoding="utf-8") as fh:
                        fh.write(text)
                payload, err = parse_output(text)
                if err and tries.get(sid, 0) < args.repairs:
                    tries[sid] = tries.get(sid, 0) + 1
                    last_bad[sid] = text
                    continue
                if err:
                    problems.append([sid, err])
                    open(bad_path(args.out, sid), "w").close()
                    continue
                issues, future = payload["issue_list"], payload["future_argument"]
                if len(future) != len(issues):
                    problems.append([sid, f"future 数量 {len(future)} != issues {len(issues)}"])
                    open(bad_path(args.out, sid), "w").close()
                    future = (future + [""] * len(issues))[:len(issues)]
                filler = "该议题后续走向仍有待观察，各方论证可能延续现有立场框架推进。"
                row = {"sample_id": sid, "issue_list": issues,
                       "future_argument": [f or filler for f in future]}
                with open(args.out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                open(done_path(args.out, sid), "w").close()
                tries.pop(sid, None)
                last_bad.pop(sid, None)
        remaining = [s for s in remaining if not is_settled(args.out, s["sample_id"])]
        if remaining and not progressed:
            break

    print(f"完成 {len(samples) - len(problems) - len(too_long)}/{len(samples)} 行 -> {args.out} "
          f"| 耗时 {time.time() - t0:.0f}s | 温度 {args.temperature}")
    if problems or too_long:
        print(f"[警告] 坏输出 {len(problems)} 条 / 超长跳过 {len(too_long)} 条（未静默丢样本）：")
        for sid, why in (problems + [[s["sample_id"], "输入超长"] for s in too_long])[:20]:
            print("  ", sid, why)
        with open(args.out + ".problems.json", "w", encoding="utf-8") as fh:
            json.dump(problems, fh, ensure_ascii=False, indent=2)
        sys.exit(2)


if __name__ == "__main__":
    main()
