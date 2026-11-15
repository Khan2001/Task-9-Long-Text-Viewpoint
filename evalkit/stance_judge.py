"""立场裁判：用官方 train 的 gold 三元组做检索投票，对 stance 给出独立第二意见。

动机：stance 是硬门控，且自洽性采样投票只能修"不稳定错误"，修不了"系统性偏差"
（例如模型习惯性把客观陈述判成 support）。检索裁判与 Qwen 完全独立，误差来源不同，
可以按规则合流。合规性：辅助模型（开源 bge）+ 官方训练数据，README 允许，需在复现材料披露。
"""

import json
import os
import sys

import numpy as np

from metrics import Encoder, load_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ("support", "oppose", "neutral")


def triple_key(name, chain):
    return name + "：" + "；".join(chain)


def build_index(enc, train, cache=None):
    if cache and os.path.exists(cache):
        vecs = np.load(cache)
        stances = [i["stance"] for s in train for i in s["issue_list"]]
        return vecs, stances
    keys, stances = [], []
    for s in train:
        for i in s["issue_list"]:
            keys.append(triple_key(i["issue_name"], i["argument_chain"]))
            stances.append(i["stance"])
    vecs = enc.encode(keys)
    if cache:
        np.save(cache, vecs)
    return vecs, np.array(stances)


def vote(enc, index_vecs, index_stances, queries, k=25, min_share=0.6):
    """返回每个 query 的裁判立场与票率；票率不足 min_share 时标记为 abstain。"""
    qv = enc.encode([triple_key(n, c) for n, c in queries])
    out = []
    idx = {l: np.where(index_stances == l)[0] for l in LABELS}
    for row in qv:
        sims = index_vecs @ row
        order = np.argsort(-sims)[:k]
        votes = np.array([index_stances[i] for i in order])
        counts = np.array([(votes == l).sum() for l in LABELS])
        best = int(np.argmax(counts))
        share = counts[best] / k
        out.append({"label": LABELS[best], "share": float(share),
                    "abstain": bool(share < min_share),
                    "dist": {l: int(counts[j]) for j, l in enumerate(LABELS)}})
    return out


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    enc = Encoder(os.path.join(ROOT, "models", "bge-small-zh-v1.5"))
    train = load_jsonl(os.path.join(ROOT, "data", "train.jsonl"))
    gold = load_jsonl(os.path.join(ROOT, "data", "%s.jsonl" % split))
    vecs, stances = build_index(enc, train,
                                cache=os.path.join(ROOT, "evalkit", "runs", "train_index.npy"))
    queries = [(i["issue_name"], i["argument_chain"]) for s in gold for i in s["issue_list"]]
    truth = [i["stance"] for s in gold for i in s["issue_list"]]
    votes = vote(enc, vecs, np.array(stances), queries)

    correct = sum(1 for v, t in zip(votes, truth) if v["label"] == t and not v["abstain"])
    active = sum(1 for v in votes if not v["abstain"])
    acc_all = sum(1 for v, t in zip(votes, truth) if v["label"] == t) / len(votes)
    print(json.dumps({
        "n_triples": len(votes),
        "judge_accuracy_all": round(acc_all, 3),
        "judge_accuracy_when_confident": round(correct / max(1, active), 3),
        "confident_coverage": round(active / len(votes), 3),
        "per_class_recall": {l: round(sum(1 for v, t in zip(votes, truth)
                                         if v["label"] == l and t == l) /
                                      max(1, sum(1 for t in truth if t == l)), 3) for l in LABELS},
        "baseline_always_support": round(sum(1 for t in truth if t == "support") / len(truth), 3),
        "gold_class_share": {l: round(sum(1 for t in truth if t == l) / len(truth), 3) for l in LABELS},
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
