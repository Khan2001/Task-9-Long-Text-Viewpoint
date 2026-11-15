"""用在线兼容 API 做离线实验（仅限验证集/训练集研究，不得用于测试集最终提交）。

用法:
    export DASHSCOPE_API_KEY=sk-xxx
    python src/infer_api.py --in data/val.jsonl --out preds/val_api_v1.jsonl --run api_val_v1

DeepSeek:
    export DEEPSEEK_API_KEY=sk-xxx
    python src/infer_api.py --api-style deepseek --model deepseek-flash \
        --in data/val.jsonl --out preds/val_deepseek.jsonl --run deepseek_val
"""

import argparse
import importlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import assemble, build_input  # noqa: E402
from prompts import extract_prompt, future_prompt  # noqa: E402,F401  (默认模块，可被 --prompt-module 覆盖)
from retrieval import Retriever, extract_example, future_example  # noqa: E402

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-32b"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_lock = threading.Lock()
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
RETRIEVED = {}


def record_usage(usage):
    with _lock:
        USAGE["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = getattr(usage, k, None)
            if v:
                USAGE[k] += int(v)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv(path=os.path.join(ROOT, ".env")):
    """把项目根目录 .env 里的键值补进环境变量（已存在的不覆盖）。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))



def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(l) for l in fh if l.strip()]


class Client:
    def __init__(self, model=MODEL, temperature=0.0, top_p=0.8, max_tokens=2048, seed=20261005,
                 base_url=None, api_style="dashscope", json_mode=True):
        from openai import OpenAI

        load_dotenv()
        key_var = "DEEPSEEK_API_KEY" if api_style == "deepseek" else "DASHSCOPE_API_KEY"
        key = os.environ.get(key_var)
        default_url = DEEPSEEK_BASE_URL if api_style == "deepseek" else BASE_URL
        base_url_var = "DEEPSEEK_BASE_URL" if api_style == "deepseek" else "DASHSCOPE_BASE_URL"
        if api_style == "vllm":
            key = key or "EMPTY"  # vLLM 不校验密钥，但 OpenAI 客户端要求非空
        elif not key:
            sys.exit(f"缺少 {key_var}：在终端 export，或写入项目根目录 .env 一行 "
                     f"{key_var}=sk-xxx（.env 已被 .gitignore 忽略，勿提交）")
        self.cli = OpenAI(api_key=key, base_url=base_url or os.environ.get(base_url_var) or default_url,
                          timeout=600)
        self.model = model
        self.api_style = api_style
        self.json_mode = json_mode
        self.params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed}
        if api_style == "deepseek":
            self.params.pop("seed")  # DeepSeek Chat Completions 未提供 seed 参数

    def chat(self, system, user, json_mode=None, retries=4):
        json_mode = self.json_mode if json_mode is None else json_mode
        if self.api_style == "vllm":
            extra = {"chat_template_kwargs": {"enable_thinking": False}}
        elif self.api_style == "dashscope":
            extra = {"enable_thinking": False}
        else:
            extra = {"thinking": {"type": "disabled"}}
        kwargs = dict(model=self.model, messages=[{"role": "system", "content": system},
                                                 {"role": "user", "content": user}],
                      extra_body=extra, **self.params)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        for attempt in range(retries):
            try:
                r = self.cli.chat.completions.create(**kwargs)
                record_usage(getattr(r, "usage", None))
                return r.choices[0].message.content
            except Exception as exc:  # noqa: BLE001
                # vLLM 若未启用结构化输出后端，自动退回自由文本（parse_json_block 仍可解析）
                if json_mode and self.api_style == "vllm" and attempt == 0 \
                        and "response_format" in str(exc).lower():
                    kwargs.pop("response_format", None)
                    json_mode = False
                    continue
                if attempt == retries - 1:
                    return f"__ERROR__ {exc}"
                time.sleep(2 * (attempt + 1))
        return None


def cache_path(run_dir, stage, sid):
    d = os.path.join(run_dir, stage)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, sid + ".json")


def cached_call(run_dir, stage, sid, fn):
    p = cache_path(run_dir, stage, sid)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)["raw"]
    raw = fn()
    with _lock:
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"raw": raw}, fh, ensure_ascii=False)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", required=True, help="运行名，原始输出缓存在 runs/<run>/")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-issues", type=int, default=5)
    ap.add_argument("--max-groups", type=int, default=3, help="每议题最多合并成几段证据")
    ap.add_argument("--group-gap", type=int, default=1, help="相邻证据句允许跳过的句数")
    ap.add_argument("--max-chain-chars", type=int, default=0,
                    help="单议题证据字数上限，超限则退回离散原句；0=不限制（val 调参最优约 200）")
    ap.add_argument("--stage", choices=["extract", "both"], default="both")
    ap.add_argument("--examples", type=int, default=0, help="few-shot 条数（需配 --examples-from）")
    ap.add_argument("--examples-from", default="data/train.jsonl")
    ap.add_argument("--retrieval-k", type=int, default=0,
                    help="检索增强 few-shot 条数：从 train.jsonl 找最相似样本当范例，0 表示关闭")
    ap.add_argument("--retrieval-pool", type=int, default=50, help="检索候选池大小")
    ap.add_argument("--retrieval-min-sim", type=float, default=0.0, help="检索相似度下限，低于此值不作为范例")
    ap.add_argument("--examples-with-text", type=int, default=0,
                    help="检索范例附带的原文摘录字数（>0 时把范例原文前 N 字一起给模型，强化命名学习）；0=仅给议题名")
    ap.add_argument("--no-prefer-doc-type", action="store_true", help="检索时不优先同一 doc_type")
    ap.add_argument("--examples-file", default=None,
                    help="离线预计算的检索范例（prepare_examples.py 产物）；"
                         "适合没有 bge 依赖的远程推理环境")
    ap.add_argument("--base-url", default=None, help="OpenAI 兼容网关地址；缺省时按 --api-style 选择")
    ap.add_argument("--model", default=None, help="缺省时百炼用 qwen3-32b，DeepSeek 用 deepseek-flash")
    ap.add_argument("--api-style", choices=["dashscope", "vllm", "deepseek"], default="dashscope",
                    help="dashscope=百炼 Qwen；vllm=本地服务；deepseek=DeepSeek 官方 OpenAI 兼容接口")
    ap.add_argument("--no-json-mode", action="store_true", help="关闭 JSON 输出模式（vLLM 不支持时的兜底）")
    ap.add_argument("--prompt-module", default="prompts",
                    help="提示词模块名（src 下），如 prompts / prompts_new，用于 A/B 对照")
    args = ap.parse_args()
    args.model = args.model or ("deepseek-flash" if args.api_style == "deepseek" else MODEL)

    # 按 --prompt-module 动态绑定提示词构造函数（覆盖顶部默认导入）
    global extract_prompt, future_prompt
    pmod = importlib.import_module(args.prompt_module)
    extract_prompt, future_prompt = pmod.extract_prompt, pmod.future_prompt
    print(f"提示词模块: {args.prompt_module}")

    samples = read_jsonl(args.inp)
    if args.limit:
        samples = samples[:args.limit]
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    cli = Client(model=args.model, base_url=args.base_url, api_style=args.api_style,
                 json_mode=not args.no_json_mode)
    retriever = Retriever() if args.retrieval_k > 0 else None
    # 检索在主线程一次性算完：既避免多线程加载/编码模型的竞态，也省去重复编码。
    nbrs_by_id = {}
    precomputed = {}
    if args.examples_file:
        with open(args.examples_file, encoding="utf-8") as fh:
            precomputed = json.load(fh)
        for sid, item in precomputed.items():
            RETRIEVED[sid] = item.get("neighbors", [])
    if retriever is not None:
        retriever.warmup()
        for s in samples:
            doc0 = s["docs"][0]
            nbrs = retriever.neighbors(doc0["full_text"], doc0["doc_type"], k=args.retrieval_k,
                                       pool=args.retrieval_pool,
                                       prefer_doc_type=not args.no_prefer_doc_type,
                                       min_sim=args.retrieval_min_sim)
            nbrs_by_id[s["sample_id"]] = nbrs
            RETRIEVED[s["sample_id"]] = [
                {"sample_id": r["sample_id"], "doc_type": r["docs"][0]["doc_type"],
                 "sim": round(sim, 4)} for r, sim in nbrs]

    examples = []
    if args.examples:
        for s in read_jsonl(args.examples_from)[:args.examples]:
            examples.append({"issue_list": [
                {"issue_name": i["issue_name"], "stance": i["stance"],
                 "evidence_ids_hint": [x[:40] for x in i["argument_chain"]]}
                for i in s["issue_list"]]})

    reports = []

    def work(sample):
        doc, sents, numbered = build_input(sample)
        ex_examples, fu_examples = examples or None, None
        if retriever is not None:
            nbrs = nbrs_by_id[sample["sample_id"]]
            ex_examples = [extract_example(r, include_text=args.examples_with_text) for r, _ in nbrs]
            fu_examples = [future_example(r) for r, _ in nbrs]
        elif precomputed:
            item = precomputed.get(sample["sample_id"]) or {}
            ex_examples = item.get("extract") or None
            fu_examples = item.get("future") or None
        ex = extract_prompt(doc["doc_type"], doc["publish_date"], numbered, len(sents), ex_examples)
        raw_e = cached_call(run_dir, "extract", sample["sample_id"],
                            lambda: cli.chat(EXTRACT_SYS, ex))
        post = {"max_issues": args.max_issues, "max_groups": args.max_groups,
                "group_gap": args.group_gap, "max_chain_chars": args.max_chain_chars}
        if args.stage == "extract":
            return assemble(sample, raw_e, None, report=[], **post)[0]
        row, rep = assemble(sample, raw_e, None, report=[], **post)
        fu = future_prompt(doc["doc_type"], doc["full_text"], row["issue_list"], fu_examples)
        raw_f = cached_call(run_dir, "future", sample["sample_id"],
                            lambda: cli.chat(FUTURE_SYS, fu))
        row, rep = assemble(sample, raw_e, raw_f, report=rep, **post)
        with _lock:
            reports.extend(rep)
        return row

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(work, samples))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(reports) or "无告警")
    with open(os.path.join(run_dir, "usage.json"), "w", encoding="utf-8") as fh:
        json.dump(USAGE, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"cli": vars(args), "model": cli.model, "params": cli.params,
                   "prompt_file": f"src/{args.prompt_module}.py"}, fh, ensure_ascii=False, indent=2)
    if RETRIEVED:
        with open(os.path.join(run_dir, "retrieval.json"), "w", encoding="utf-8") as fh:
            json.dump(RETRIEVED, fh, ensure_ascii=False, indent=2)
    print(f"{args.stage} 完成: {len(rows)} 条 -> {args.out}  用时 {time.time()-t0:.0f}s  告警 {len(reports)} 条")
    print("token 用量: " + json.dumps(USAGE, ensure_ascii=False))
    print("评分: python src/metric.py --pred %s --gold %s" % (args.out, args.inp))


EXTRACT_SYS = (
    "你是谈判情报分析师，从给定长文本中挖掘观点三元组（议题名称、立场、原文证据句）。"
    "证据只能引用输入里带编号的原句编号，绝不改写或编造。只输出 JSON。"
)
FUTURE_SYS = (
    "你是谈判对手研判专家，基于给定文本与已抽取的议题链，预判各方后续可能表达的观点与论证方向。"
    "输出 JSON，future_argument 必须与议题数量相同且同序。"
)


if __name__ == "__main__":
    main()
