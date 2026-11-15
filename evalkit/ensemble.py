"""自洽性融合（P1）：同一 adapter、同一基座采样 N 次后合并为一份提交。

规则允许"融合由同一 Qwen3-32B 基座产生的检查点或推理结果"，因此 N 次采样的投票是合规的。
合并逻辑：
  1) 跨采样按议题名余弦聚簇（贪心，按首次出现顺序建立簇代表）；
  2) 簇内 stance 取多数票 —— stance 是硬门控，投票是这里收益最大的一项；
  3) 簇代表取 medoid（与簇内其它成员平均相似度最高的那条），并沿用它所属采样的
     argument_chain 与 future_argument，保证三元组与预测自洽；
  4) 按票数降序、原顺序升序保留 top_k 个簇，避免多余预测压低 P 与 F1pred 分母。
"""

import argparse
import collections
import json
import os

import numpy as np

from metrics import Encoder, load_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def merge_sample(draws, enc, tau_name=0.85, top_k=5):
    items = []
    for di, d in enumerate(draws):
        for oi, it in enumerate(d["issue_list"]):
            fa = d.get("future_argument", [])
            items.append({"draw": di, "order": oi, "name": it["issue_name"],
                          "stance": it["stance"], "chain": list(it["argument_chain"]),
                          "future": fa[oi] if oi < len(fa) else ""})
    if not items:
        return None
    vecs = enc.encode([x["name"] for x in items])

    clusters = []
    for idx, item in enumerate(items):
        placed = False
        for cl in clusters:
            if float(vecs[idx] @ cl["centroid"]) >= tau_name:
                cl["members"].append(idx)
                cl["centroid"] = cl["centroid"] + vecs[idx]
                placed = True
                break
        if not placed:
            clusters.append({"members": [idx], "centroid": vecs[idx].copy()})

    rows = []
    for cl in clusters:
        members = cl["members"]
        support = len({items[i]["draw"] for i in members})
        stance_votes = collections.Counter(items[i]["stance"] for i in members)
        stance = stance_votes.most_common(1)[0][0]
        if len(members) == 1:
            rep = members[0]
        else:
            sub = vecs[members]
            sims = sub @ sub.T
            rep = members[int(np.argmax(sims.mean(1)))]
        first = min(items[i]["order"] for i in members)
        chain = items[rep]["chain"]
        if stance != items[rep]["stance"]:
            stance = stance  # 保留多数票立场，论据沿用 medoid
        rows.append({"issue_name": items[rep]["name"], "stance": stance,
                     "argument_chain": chain, "future_argument": items[rep]["future"],
                     "_support": support, "_order": first})
    rows.sort(key=lambda r: (-r["_support"], r["_order"]))
    keep = rows[:top_k]
    keep.sort(key=lambda r: r["_order"])
    return {"issue_list": [{"issue_name": r["issue_name"], "stance": r["stance"],
                            "argument_chain": r["argument_chain"]} for r in keep],
            "future_argument": [r["future_argument"] for r in keep],
            "_candidates": len(rows)}


def load_draws(paths):
    by_sid = collections.defaultdict(list)
    for p in paths:
        for row in load_jsonl(p):
            by_sid[row["sample_id"]].append(row)
    return by_sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", nargs="+", required=True, help="N 份采样预测文件")
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", help="输入集，用于 sample_id 排序")
    ap.add_argument("--tau-name", dest="tau_name", type=float, default=0.85)
    ap.add_argument("--top-k", dest="top_k", type=int, default=5)
    a = ap.parse_args()
    enc = Encoder(os.path.join(ROOT, "models", "bge-small-zh-v1.5"))
    by_sid = load_draws(a.draws)
    order = None
    if a.docs:
        order = [r["sample_id"] for r in load_jsonl(a.docs)]
        for sid in order:
            by_sid.setdefault(sid, [])
    out, extra = [], 0
    for sid in (order or list(by_sid)):
        draws = by_sid.get(sid, [])
        if not draws:
            continue
        merged = merge_sample(draws, enc, tau_name=a.tau_name, top_k=a.top_k)
        merged["sample_id"] = sid
        extra += merged.pop("_candidates") - len(merged["issue_list"])
        out.append(merged)
    with open(a.out, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("写出 %d 条 -> %s（丢弃候选三元组 %d 个）" % (len(out), a.out, extra))


if __name__ == "__main__":
    main()
