"""离线预计算检索增强范例，供没有 sentence-transformers 的推理环境（如 vLLM 实例）使用。

本机跑一次，产出 examples.json；推理端用 infer_api.py --examples-file 读取，
这样实例上只需 openai 客户端，不必安装 bge 相关依赖。

用法:
    python src/prepare_examples.py --in data/test.jsonl --out examples_test.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import Retriever, extract_example, future_example  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--min-sim", type=float, default=0.75)
    args = ap.parse_args()

    rows = read_jsonl(args.inp)
    retriever = Retriever()
    retriever.warmup()

    out = {}
    hit = {"0": 0, "1": 0, "2": 0}
    for s in rows:
        doc = s["docs"][0]
        nbrs = retriever.neighbors(doc["full_text"], doc["doc_type"], k=args.k,
                                   pool=args.pool, min_sim=args.min_sim)
        hit[str(len(nbrs))] = hit.get(str(len(nbrs)), 0) + 1
        out[s["sample_id"]] = {
            "neighbors": [{"sample_id": r["sample_id"], "doc_type": r["docs"][0]["doc_type"],
                           "sim": round(sim, 4)} for r, sim in nbrs],
            "extract": [extract_example(r) for r, _ in nbrs],
            "future": [future_example(r) for r, _ in nbrs],
        }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"{len(rows)} 条 -> {args.out}；邻居条数分布 {hit}")


if __name__ == "__main__":
    main()
