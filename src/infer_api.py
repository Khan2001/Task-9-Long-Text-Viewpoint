"""用百炼 DashScope 上的 Qwen3-32B 做离线实验（仅限验证集/训练集研究，不得用于测试集最终提交）。

用法:
    export DASHSCOPE_API_KEY=sk-xxx
    python src/infer_api.py --in data/val.jsonl --out preds/val_api_v1.jsonl --run api_val_v1
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import assemble, build_input  # noqa: E402
from prompts import extract_prompt, future_prompt  # noqa: E402

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-32b"
_lock = threading.Lock()
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


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
                 base_url=None):
        from openai import OpenAI

        load_dotenv()
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            sys.exit("缺少 DASHSCOPE_API_KEY：在终端 export，或写入项目根目录 .env 一行 "
                     "DASHSCOPE_API_KEY=sk-xxx（.env 已被 .gitignore 忽略，勿提交）")
        self.cli = OpenAI(api_key=key, base_url=base_url or os.environ.get("DASHSCOPE_BASE_URL") or BASE_URL,
                          timeout=180)
        self.model = model
        self.params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed}

    def chat(self, system, user, json_mode=True, retries=4):
        kwargs = dict(model=self.model, messages=[{"role": "system", "content": system},
                                                 {"role": "user", "content": user}],
                      extra_body={"enable_thinking": False}, **self.params)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        for attempt in range(retries):
            try:
                r = self.cli.chat.completions.create(**kwargs)
                record_usage(getattr(r, "usage", None))
                return r.choices[0].message.content
            except Exception as exc:  # noqa: BLE001
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
    ap.add_argument("--stage", choices=["extract", "both"], default="both")
    ap.add_argument("--examples", type=int, default=0, help="few-shot 条数（需配 --examples-from）")
    ap.add_argument("--examples-from", default="data/train.jsonl")
    ap.add_argument("--base-url", default=None, help="OpenAI 兼容网关地址，默认百炼 compatible-mode")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    samples = read_jsonl(args.inp)
    if args.limit:
        samples = samples[:args.limit]
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    cli = Client(model=args.model, base_url=args.base_url)

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
        ex = extract_prompt(doc["doc_type"], doc["publish_date"], numbered, len(sents), examples or None)
        raw_e = cached_call(run_dir, "extract", sample["sample_id"],
                            lambda: cli.chat(EXTRACT_SYS, ex))
        post = {"max_issues": args.max_issues, "max_groups": args.max_groups,
                "group_gap": args.group_gap}
        if args.stage == "extract":
            return assemble(sample, raw_e, None, report=[], **post)[0]
        row, rep = assemble(sample, raw_e, None, report=[], **post)
        fu = future_prompt(doc["doc_type"], doc["full_text"], row["issue_list"], examples or None)
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
                   "prompt_file": "src/prompts.py"}, fh, ensure_ascii=False, indent=2)
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
