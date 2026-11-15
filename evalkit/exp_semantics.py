"""语义层实验：不依赖模型预测，直接在 train/val 上测三件事的上限。

A. 阈值标定：0.7 相对自然变化有多难通过（议题名 / 论据链各自的 cos 分布）。
B. 证据重选上限：若保留模型给的议题名与立场，只用 bge 从原文重选 top-k 句做论据链，
   三元组语义分能到多少、能过 0.7 的比例有多少（= chain 替换策略的天花板）。
C. 立场门控敏感度：stance 判错率每 +1%，F1ext 掉多少。
"""

import argparse
import json
import os
import random
import re

import numpy as np

from metrics import Encoder, evaluate, load_jsonl, embed_chains, TRIPLE_THRESHOLD
from postprocess import split_sentences

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score_pairs(enc, pairs_a, pairs_b):
    a = enc.encode(pairs_a)
    b = enc.encode(pairs_b)
    return np.array([float(a[i] @ b[i]) for i in range(len(pairs_a))])


def experiment_a(enc, gold, rng, n=400):
    names = [i["issue_name"] for s in gold for i in s["issue_list"]]
    sub = rng.sample(names, min(n, len(names)))
    same_doc = []          # 同一样本内不同议题（最接近"模型换了说法"的情形）
    for s in gold[:n]:
        il = s["issue_list"]
        for i in range(len(il)):
            for j in range(i + 1, len(il)):
                same_doc.append((il[i]["issue_name"], il[j]["issue_name"]))
    same_doc = rng.sample(same_doc, min(n, len(same_doc)))
    cross = [(rng.choice(sub), rng.choice(sub)) for _ in range(n)]
    cross = [(a, b) for a, b in cross if a != b]

    out = {}
    for label, pairs in (("same_sample_other_issue", same_doc), ("random_pair", cross)):
        sims = score_pairs(enc, [p[0] for p in pairs], [p[1] for p in pairs])
        out[label] = {"mean": round(float(sims.mean()), 3),
                      "p25": round(float(np.percentile(sims, 25)), 3),
                      "p75": round(float(np.percentile(sims, 75)), 3),
                      "share_over_0.55": round(float((sims > 0.55).mean()), 3)}
    # 论据链：gold 链 vs 同文档随机其它句
    off = []
    for s in gold[:200]:
        doc = "".join(d["full_text"] for d in s["docs"])
        sents = split_sentences(doc)
        golds = {a for i in s["issue_list"] for a in i["argument_chain"]}
        others = [x for x in sents if x not in golds and len(x) > 15]
        if len(others) < 2:
            continue
        for i in s["issue_list"][:2]:
            off.append(("；".join(i["argument_chain"]), "；".join(rng.sample(others, min(3, len(others))))))
    sims = score_pairs(enc, [x[0] for x in off], [x[1] for x in off])
    out["gold_chain_vs_random_sentences"] = {
        "n": len(off), "mean": round(float(sims.mean()), 3),
        "p25": round(float(np.percentile(sims, 25)), 3),
        "p75": round(float(np.percentile(sims, 75)), 3),
        "share_over_threshold_alone": round(float((0.4 + 0.6 * sims > TRIPLE_THRESHOLD).mean()), 3)}
    return out


def experiment_b(enc, gold, rng, n=150):
    """保留 gold 议题名，用检索重建论据链，测三元组分。"""
    cand, golds, seg_hit, seg_total = [], [], 0, 0
    for s in gold[:n]:
        doc = "".join(d["full_text"] for d in s["docs"])
        sents = split_sentences(doc)
        if len(sents) < 3:
            continue
        vecs = enc.encode(sents)
        q = enc.encode([i["issue_name"] for i in s["issue_list"]])
        for idx, item in enumerate(s["issue_list"]):
            order = np.argsort(-(vecs @ q[idx]))
            for k in (1, 2, 3):
                cand.append("；".join(sents[i] for i in order[:k]))
            golds.append("；".join(item["argument_chain"]))
            seg_total += 1
            if any("".join(item["argument_chain"]) == x for x in sents):
                seg_hit += 1
    cv, gv = enc.encode(cand), enc.encode(golds)
    sims = np.array([float(cv[i] @ gv[i]) for i in range(len(golds))]).reshape(-1, 3)
    out = {}
    for pos, k in enumerate((1, 2, 3)):
        arr = 0.4 + 0.6 * sims[:, pos]
        out["retrieve_top%d" % k] = {"mean_triple_score": round(float(arr.mean()), 3),
                                     "pass_0.7": round(float((arr > TRIPLE_THRESHOLD).mean()), 3)}
    out["n_issues"] = len(golds)
    out["gold_chain_equals_one_segment"] = round(seg_hit / max(1, seg_total), 3)
    return out


def experiment_c(gold, enc, rng, rates=(0.0, 0.05, 0.10, 0.20, 0.30)):
    rows = gold
    out = {}
    for r in rates:
        preds = []
        for s in rows[:200]:
            pl = []
            for i in s["issue_list"]:
                stance = i["stance"]
                if rng.random() < r:
                    alt = [x for x in ("support", "oppose", "neutral") if x != stance]
                    stance = rng.choice(alt)
                pl.append({"issue_name": i["issue_name"], "stance": stance,
                           "argument_chain": i["argument_chain"]})
            preds.append({"sample_id": s["sample_id"], "issue_list": pl,
                          "future_argument": s["future_argument"]})
        path = os.path.join(ROOT, "evalkit", "runs", "stance_%02d.jsonl" % int(r * 100))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        gold_path = path.replace("stance_", "stance_gold_")
        with open(gold_path, "w", encoding="utf-8") as f:
            for s in rows[:200]:
                f.write(json.dumps({k: s[k] for k in
                                    ("sample_id", "issue_list", "future_argument")},
                                   ensure_ascii=False) + "\n")
        res = evaluate(gold_path, path, encoder=enc)
        out["flip_%.0f%%" % (r * 100)] = {"F1ext": round(res["F1ext"], 4),
                                          "F1pred": round(res["F1pred"], 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--run-c", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    gold = load_jsonl(os.path.join(ROOT, "data", "%s.jsonl" % args.split))
    rng.shuffle(gold)
    enc = Encoder(os.path.join(ROOT, "models", "bge-small-zh-v1.5"))
    report = {"split": args.split}
    report["A_threshold"] = experiment_a(enc, gold, rng)
    report["B_chain_retrieval_ceiling"] = experiment_b(enc, gold, rng)
    if args.run_c:
        report["C_stance_sensitivity"] = experiment_c(gold, enc, rng)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
