"""从已有的 result.jsonl 生成候选提交文件（全程 CPU，不需要 GPU）。

变体设计成"一次只改一个变量"，可以直接拿线上 3 次/天的额度当 A/B 测量仪：
  v1  仅 anchor：预期 Δ≈0，用来验证本地评测器与线上口径一致；
  v2  anchor + 检索立场裁判覆盖（裁判高置信且与模型不一致时才改）；
  v3  anchor + 裁判覆盖 + 议题名检索规范化。
生成后一律跑结构校验，不合法就不许提交。
"""

import argparse
import json
import os

import numpy as np

from metrics import Encoder, load_jsonl
from postprocess import anchor_snippet, split_sentences
from stance_judge import build_index, vote, triple_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def name_canonicalize(enc, index_vecs, index_stances, train_triples, queries, tau=0.75):
    """议题名规范化：若检索到高度相似的官方训练三元组，用它的 gold 议题名替换。"""
    qv = enc.encode([n for n, _ in queries])
    names = [t["issue_name"] for t in train_triples]
    out = []
    for row in qv:
        sims = index_vecs @ row
        j = int(np.argmax(sims))
        out.append(names[j] if sims[j] >= tau else None)
    return out


def build(pred_path, docs_path, out_path, judge_override_min_share=0.8,
          use_judge=True, use_name_canon=False, override_classes=("neutral",)):
    enc = Encoder(os.path.join(ROOT, "models", "bge-small-zh-v1.5"))
    train = load_jsonl(os.path.join(ROOT, "data", "train.jsonl"))
    train_triples = [i for s in train for i in s["issue_list"]]
    index_vecs, index_stances = build_index(
        enc, train, cache=os.path.join(ROOT, "evalkit", "runs", "train_index.npy"))

    preds = load_jsonl(pred_path)
    docs = {r["sample_id"]: "".join(d["full_text"] for d in r["docs"])
            for r in load_jsonl(docs_path)}

    queries, meta = [], []
    for p in preds:
        for idx, item in enumerate(p["issue_list"]):
            queries.append((item["issue_name"], item["argument_chain"]))
            meta.append((p["sample_id"], idx))
    votes = vote(enc, index_vecs, np.array(index_stances), queries) if use_judge else [None] * len(queries)
    canon = name_canonicalize(enc, index_vecs, index_stances, train_triples, queries) \
        if use_name_canon else [None] * len(queries)

    by_sid = {}
    for (sid, idx), v, cn in zip(meta, votes, canon):
        by_sid.setdefault(sid, {})[idx] = (v, cn)

    out, n_flip, n_name = [], 0, 0
    for p in preds:
        doc = docs[p["sample_id"]]
        sents = split_sentences(doc)
        new_issues = []
        for idx, item in enumerate(p["issue_list"]):
            it = {"issue_name": item["issue_name"], "stance": item["stance"],
                  "argument_chain": [anchor_snippet(a, sents, doc) for a in item["argument_chain"]]}
            v, cn = by_sid[p["sample_id"]][idx]
            if v and not v["abstain"] and v["share"] >= judge_override_min_share \
                    and v["label"] in override_classes and v["label"] != it["stance"]:
                it["stance"] = v["label"]
                n_flip += 1
            if cn:
                it["issue_name"] = cn
                n_name += 1
            new_issues.append(it)
        out.append({"sample_id": p["sample_id"], "issue_list": new_issues,
                    "future_argument": p["future_argument"][:len(new_issues)]})

    validate(out, docs)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("写出 %d 条 -> %s（立场被裁判覆盖 %d 处，议题名替换 %d 处）"
          % (len(out), out_path, n_flip, n_name))


def validate(rows, docs):
    assert len(rows) == 300, "样本数 %d != 300" % len(rows)
    sids = [r["sample_id"] for r in rows]
    assert len(set(sids)) == 300, "sample_id 重复"
    for r in rows:
        assert len(r["issue_list"]) == len(r["future_argument"]), r["sample_id"]
        for it in r["issue_list"]:
            assert it["stance"] in ("support", "oppose", "neutral")
            assert isinstance(it["argument_chain"], list) and it["argument_chain"]
            assert len(it["issue_name"]) > 2
            for a in it["argument_chain"]:
                assert a.strip() in docs[r["sample_id"]], \
                    "论据非原文逐字: %s %s" % (r["sample_id"], a[:30])
    print("结构校验通过：300 条、长度对齐、立场合法、论据 100% 逐字可追溯")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(ROOT, "result.jsonl"))
    ap.add_argument("--docs", default=os.path.join(ROOT, "data", "test.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--override-min-share", type=float, default=0.8)
    ap.add_argument("--override-classes", default="neutral",
                    help="只允许裁判把这些类别覆盖上去（逗号分隔）；裁判在少数派议题上只有 0.335，默认仅放最可靠的 neutral")
    ap.add_argument("--name-canon", action="store_true")
    a = ap.parse_args()
    build(a.pred, a.docs, a.out, judge_override_min_share=a.override_min_share,
          use_judge=not a.no_judge, use_name_canon=a.name_canon,
          override_classes=tuple(x.strip() for x in a.override_classes.split(",")))
