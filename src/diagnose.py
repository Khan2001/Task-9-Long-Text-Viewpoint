"""丢分归因：逐个参考三元组判断它为什么没被匹配上。"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metric import TripleMatcher, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--good", type=float, default=0.85, help="单侧相似度视为够用的下限")
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    preds = {p["sample_id"]: p for p in read_jsonl(args.pred)}
    golds = read_jsonl(args.gold)
    m = TripleMatcher(threshold=args.threshold)
    reason = Counter()
    examples = {}
    for g in golds:
        p = preds.get(g["sample_id"])
        if not p or not p["issue_list"]:
            reason["无预测"] += 1
            continue
        for gt in g["issue_list"]:
            cand = [m.pair_score(pt, gt) + (pt["stance"], pt["issue_name"])
                    for pt in p["issue_list"]]
            same = [c for c in cand if c[3] == gt["stance"]]
            if not same:
                best_any = max(cand, key=lambda c: c[0])
                if best_any[0] > args.threshold:
                    reason["语义很近但立场不符"] += 1
                    examples.setdefault("立场", []).append(
                        (g["sample_id"], gt["issue_name"], gt["stance"],
                         best_any[4], best_any[3], round(best_any[0], 3)))
                else:
                    reason["立场不符且语义也低"] += 1
                continue
            top = max(same, key=lambda c: c[0])
            score, sim_issue, sim_chain = top[0], top[1], top[2]
            if score > args.threshold:
                reason["匹配成功"] += 1
                continue
            if sim_issue >= args.good and sim_chain < args.good:
                key = "论据链拖累"
            elif sim_chain >= args.good and sim_issue < args.good:
                key = "议题名拖累"
            else:
                key = "议题名与论据链都偏低"
            reason[key] += 1
            examples.setdefault(key, []).append(
                (g["sample_id"], gt["issue_name"], top[4],
                 f"议题{sim_issue:.2f}/论据{sim_chain:.2f}"))

    total = sum(reason.values())
    print(f"参考三元组总数 {total}")
    for k, v in reason.most_common():
        print(f"  {k:12s} {v:5d}  {v/total:.1%}")
    for k in ("立场", "论据链拖累", "议题名拖累", "议题名与论据链都偏低"):
        rows = examples.get(k, [])
        if rows:
            print(f"\n--- {k} 示例 ---")
            for r in rows[:args.show]:
                print("  ", r)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"reason": dict(reason), "examples": examples}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
