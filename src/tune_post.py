"""零成本后处理调参：复用 runs/<run>/ 里缓存的模型原始输出，只改装配参数并打分。

用法:
    python src/tune_post.py --run api_v1 --gold data/val.jsonl \
        --max-issues 4,5 --max-groups 1,2,3
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metric import BertScorer, TripleMatcher, evaluate_with, read_jsonl  # noqa: E402
from pipeline import assemble  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cached_raw(run_dir, stage, sid):
    p = os.path.join(ROOT, "runs", run_dir, stage, sid + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)["raw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gold", default="data/val.jsonl")
    ap.add_argument("--max-issues", default="4,5")
    ap.add_argument("--max-groups", default="1,2,3")
    ap.add_argument("--group-gap", default="0,1")
    ap.add_argument("--pads", default="0")
    ap.add_argument("--max-chain-chars", default="0", help="单议题证据字数上限，逗号分隔；0=不限制")
    ap.add_argument("--save-prefix", default="preds/tune")
    args = ap.parse_args()

    samples = read_jsonl(args.gold)
    raw_e = {s["sample_id"]: cached_raw(args.run, "extract", s["sample_id"]) for s in samples}
    raw_f = {s["sample_id"]: cached_raw(args.run, "future", s["sample_id"]) for s in samples}
    have = [s for s in samples if raw_e[s["sample_id"]]]
    print(f"缓存可用样本 {len(have)}/{len(samples)}（缺缓存的样本不会参与本轮调参）")
    if not have:
        sys.exit(f"runs/{args.run}/extract 下没有缓存，先跑 infer_api.py")

    matcher, bs = TripleMatcher(threshold=0.7), BertScorer()
    results = []
    for mi in [int(x) for x in args.max_issues.split(",")]:
        for mg in [int(x) for x in args.max_groups.split(",")]:
            for gap in [int(x) for x in args.group_gap.split(",")]:
              for pad in [int(x) for x in args.pads.split(",")]:
                for mcc in [int(x) for x in args.max_chain_chars.split(",")]:
                    rows = [assemble(s, raw_e[s["sample_id"]], raw_f[s["sample_id"]],
                                     max_issues=mi, max_groups=mg, group_gap=gap, pad=pad,
                                     max_chain_chars=mcc)[0]
                            for s in have]
                    t0 = time.time()
                    overall = evaluate_with(matcher, bs, rows, samples)["overall"]
                    tag = f"i{mi}_g{mg}_p{gap}_e{pad}_c{mcc}"
                    out = os.path.join(ROOT, f"{args.save_prefix}_{args.run}_{tag}.jsonl")
                    with open(out, "w", encoding="utf-8") as fh:
                        for r in rows:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    results.append((tag, overall))
                    print(f"{tag}  Score {overall['Score']:.4f}  S_ext {overall['S_ext']:.4f}  "
                          f"S_pred {overall['S_pred']:.4f}  F1_ext {overall['F1_ext']:.4f}  "
                          f"alpha {overall['alpha']:.4f}  NC {overall['total_nc']}  "
                          f"pred {overall['total_pred']}  gold {overall['total_gold']}  "
                          f"[{time.time()-t0:.0f}s] -> {os.path.basename(out)}", flush=True)

    results.sort(key=lambda r: -r[1]["Score"])
    print("\n按总分排序:")
    for tag, o in results:
        print(f"  {tag}  {o['Score']:.4f}")
    print(f"\n最优: {results[0][0]}  Score {results[0][1]['Score']:.4f}")


if __name__ == "__main__":
    main()
