#!/usr/bin/env python3
"""使用同一 Qwen3-32B + LoRA 审查 test.jsonl 和 result_mix.jsonl。

输出 result_review.jsonl；保留原预测作为安全回退；支持断点续跑。
依赖: vllm, transformers, torch
"""
import argparse
import json
import logging
import re
from pathlib import Path

LOG = logging.getLogger("review")

SYSTEM = """你是事件议题抽取与未来论点预测结果的严格审查员。
你的目标是纠正明确错误，而不是润色或重写答案。
请结合原始 docs 审查给出的预测：
1. 仅在有原文证据时修复重大议题遗漏、明显重复或错误合并；不强制议题数量。
2. 仅在原文明确证明立场错误时修正 support/neutral/oppose。
3. 保留正确的 issue_name 和 argument_chain；若调整议题，证据必须来自原文。
4. 仅在 future_argument 明显偏题、错位或与原文矛盾时修正；不要为增加具体性编造事实。
5. 如果没有明确错误，保持原预测不变。
6. issue_list 与 future_argument 必须一一对应，顺序一致。
只返回一个 JSON 对象，包含 need_revision（布尔值）、reason（简短中文说明）、issue_list、future_argument。
不要输出 Markdown、解释性前后缀或其他内容。"""


def read_jsonl(path):
    result = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc
            sid = obj.get("sample_id")
            if sid is None:
                raise ValueError(f"{path}:{line_no} 缺少 sample_id")
            if sid in result:
                raise ValueError(f"{path}:{line_no} 重复 sample_id={sid}")
            result[sid] = obj
    return result


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def parse_response(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        return None


def valid_prediction(obj):
    if not isinstance(obj, dict):
        return False
    issues, futures = obj.get("issue_list"), obj.get("future_argument")
    if not isinstance(issues, list) or not isinstance(futures, list):
        return False
    if len(issues) != len(futures) or not all(isinstance(x, str) and x.strip() for x in futures):
        return False
    for item in issues:
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("issue_name"), str) or not item["issue_name"].strip():
            return False
        if item.get("stance") not in {"support", "neutral", "oppose"}:
            return False
        chain = item.get("argument_chain")
        if not isinstance(chain, list) or not all(isinstance(s, str) and s.strip() for s in chain):
            return False
    return True


def make_messages(test_item, pred_item):
    docs = test_item.get("docs", [])
    original = {
        "issue_list": pred_item["issue_list"],
        "future_argument": pred_item["future_argument"],
    }
    user = (
        "【原始事件文档 docs】\n"
        + json.dumps(docs, ensure_ascii=False)
        + "\n\n【待审查预测】\n"
        + json.dumps(original, ensure_ascii=False)
        + "\n\n请审查。若无需修改，need_revision=false 并原样返回预测。"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def main():
    p = argparse.ArgumentParser(description="Qwen3-32B LoRA 二次审查已有预测")
    p.add_argument("--test", default="data/test.jsonl")
    p.add_argument("--input", default="result_mix.jsonl")
    p.add_argument("--output", default="result_review.jsonl")
    p.add_argument("--progress", default="result_review.progress.jsonl")
    p.add_argument("--changes", default="result_review.changes.jsonl")
    p.add_argument("--model", default="/root/eb-public/huggingface-models/Qwen/Qwen3-32B")
    p.add_argument("--adapter", default="outputs/qwen3-32b-qlora")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-input-tokens", type=int, default=13500)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=0, help="0=全部；非0时只处理前 N 条，用于试运行")
    p.add_argument("--overwrite", action="store_true", help="清空已有进度并重新审查")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if Path(args.output).resolve() == Path(args.input).resolve():
        p.error("--output 不得与 --input 相同")
    if Path(args.progress).resolve() == Path(args.input).resolve():
        p.error("--progress 不得与 --input 相同")
    test_data = read_jsonl(args.test)
    predictions = read_jsonl(args.input)
    missing = [sid for sid in predictions if sid not in test_data]
    if missing:
        raise ValueError(f"测试集中找不到 {len(missing)} 个预测 sample_id，例如 {missing[:5]}")
    for sid, pred in predictions.items():
        if not valid_prediction(pred):
            raise ValueError(f"原始预测 {sid} 格式不合法，请先检查")
    ids = list(predictions)
    if args.max_samples:
        ids = ids[:args.max_samples]
    if args.overwrite:
        for path in [args.progress, args.output, args.changes]:
            Path(path).unlink(missing_ok=True)
    done = read_jsonl(args.progress) if Path(args.progress).exists() else {}
    pending = [sid for sid in ids if sid not in done]
    LOG.info("预测 %d 条，已审查 %d 条，待审查 %d 条", len(ids), len(ids) - len(pending), len(pending))

    if pending:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enable_lora=True,
            max_lora_rank=128,
        )
        adapter = LoRARequest("review_adapter", 1, str(Path(args.adapter).resolve()))
        sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)

        for start in range(0, len(pending), args.batch_size):
            batch_ids = pending[start : start + args.batch_size]
            prompts = []
            for sid in batch_ids:
                messages = make_messages(test_data[sid], predictions[sid])
                prompt_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False,
                )
                # 保留完整 system 指令和预测末尾；过长时跳过，不做危险的头部截断
                if len(prompt_ids) + args.max_new_tokens > args.max_model_len or len(prompt_ids) > args.max_input_tokens:
                    prompts.append(None)
                    LOG.warning("%s 输入过长（%d tokens），保留原预测", sid, len(prompt_ids))
                else:
                    prompts.append({"prompt_token_ids": prompt_ids})
            valid_positions = [i for i, x in enumerate(prompts) if x is not None]
            generated = {}
            if valid_positions:
                requests = [prompts[i] for i in valid_positions]
                try:
                    outputs = llm.generate(requests, sampling_params=sampling, lora_request=adapter, use_tqdm=False)
                    for i, output in zip(valid_positions, outputs):
                        generated[i] = output.outputs[0].text
                except Exception:
                    LOG.exception("批次生成失败，回退为原预测")
            for i, sid in enumerate(batch_ids):
                original = predictions[sid]
                raw = generated.get(i, "")
                review = parse_response(raw) if raw else None
                accept = bool(
                    isinstance(review, dict)
                    and review.get("need_revision") is True
                    and valid_prediction(review)
                )
                # 若 need_revision=false，直接保留原始答案；解析失败也安全回退
                if accept:
                    revised = dict(original)
                    revised["issue_list"] = review["issue_list"]
                    revised["future_argument"] = review["future_argument"]
                    changed = (revised["issue_list"] != original["issue_list"] or
                               revised["future_argument"] != original["future_argument"])
                else:
                    revised, changed = original, False
                record = {
                    "sample_id": sid,
                    "prediction": revised,
                    "modified": changed,
                    "reason": str(review.get("reason", ""))[:500] if isinstance(review, dict) else "解析失败或未修改",
                    "review_valid": isinstance(review, dict) and valid_prediction(review),
                }
                append_jsonl(args.progress, record)
                done[sid] = record
            LOG.info("已完成 %d/%d，累计修改 %d", len([sid for sid in ids if sid in done]), len(ids),
                     sum(bool(done[sid]["modified"]) for sid in ids if sid in done))

    # 根据 input 的顺序重新构建最终文件，保证完整、无重复、字段一致
    with open(args.output, "w", encoding="utf-8") as out, open(args.changes, "w", encoding="utf-8") as changes:
        for sid in ids:
            record = done[sid]
            out.write(json.dumps(record["prediction"], ensure_ascii=False) + "\n")
            changes.write(json.dumps({
                "sample_id": sid,
                "modified": record["modified"],
                "reason": record.get("reason", ""),
                "review_valid": record.get("review_valid", False),
            }, ensure_ascii=False) + "\n")
    LOG.info("完成：%s（%d 条），修改 %d 条", args.output, len(ids),
             sum(bool(done[sid]["modified"]) for sid in ids))


if __name__ == "__main__":
    main()
