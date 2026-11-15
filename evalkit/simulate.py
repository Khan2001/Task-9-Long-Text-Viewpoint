"""合成实验：把 train 的 gold 按可控噪声退化成"伪模型输出"，用来在无 GPU 情况下
预估 P1 各项后处理的收益方向与量级，并标定当前 F1ext≈0.745 所处的误差结构。

噪声对应真实失败模式：
  name_paraphrase   议题名换了说法（语义接近但表述不同）
  evidence_offsent  证据选了原文里"相关但不是标注那句"（逐字正确、语义错位，后处理救不回）
  evidence_imprecite 证据是从原文摘出来又不完整/带改写（逐字失败，anchor 能救）
  chain_short       论据链少给片段
  stance_flip       立场判错（硬门控，直接 0 分）
  issue_add/drop    多给/漏给议题
"""

import argparse
import json
import os
import random
import re

import numpy as np

from metrics import Encoder, evaluate, load_jsonl
from postprocess import postprocess_one, split_sentences

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAUSE = re.compile(r"(?<=[，,])")


def build_pseudo(gold, enc, args, rng):
    name_pool = rng.sample([i["issue_name"] for s in gold for i in s["issue_list"]],
                           args.name_pool)
    name_vecs = enc.encode(name_pool)
    pseudo = []
    for sample in gold[:args.n]:
        doc = "".join(d["full_text"] for d in sample["docs"])
        sents = split_sentences(doc)
        gold_snips = {a for i in sample["issue_list"] for a in i["argument_chain"]}
        issues, futures = [], []
        for item in sample["issue_list"]:
            if rng.random() < args.p_drop_issue:
                continue
            name = item["issue_name"]
            if rng.random() < args.p_name:
                nv = enc.encode([name])
                sims = (nv @ name_vecs.T)[0]
                cands = [(s, t) for s, t in zip(sims, name_pool)
                         if t != name and len(t) > 4]
                if cands:
                    name = max(cands)[1]
            chain = []
            for a in item["argument_chain"]:
                r = rng.random()
                if r < args.p_offsent:
                    others = [s for s in sents if s not in gold_snips and len(s) > 12]
                    chain.append(rng.choice(others) if others else a)
                elif r < args.p_offsent + args.p_imprecise:
                    parts = CLAUSE.split(a)
                    chain.append("".join(parts[:max(1, len(parts) - 1)])
                                 if len(parts) > 1 else a[:int(len(a) * 0.8)])
                else:
                    chain.append(a)
            if rng.random() < args.p_chain_short and len(chain) > 1:
                chain = chain[:1]
            stance = item["stance"]
            if rng.random() < args.p_stance:
                alt = [x for x in ("support", "oppose", "neutral") if x != stance]
                stance = rng.choice(alt)
            issues.append({"issue_name": name, "stance": stance,
                           "argument_chain": chain})
            futures.append(item_future(sample, item, sample["issue_list"].index(item)))
        if rng.random() < args.p_add_issue and issues:
            extra = json.loads(json.dumps(issues[-1], ensure_ascii=False))
            extra["issue_name"] = extra["issue_name"] + "合作"
            issues.append(extra)
            futures.append("将在既有共识基础上继续推进相关磋商，并保持沟通渠道畅通。")
        pseudo.append({"sample_id": sample["sample_id"], "issue_list": issues,
                       "future_argument": futures})
    return pseudo


def item_future(sample, item, idx):
    fa = sample.get("future_argument", [])
    return fa[idx] if idx < len(fa) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--name-pool", dest="name_pool", type=int, default=300)
    ap.add_argument("--p-name", dest="p_name", type=float, default=0.35)
    ap.add_argument("--p-offsent", dest="p_offsent", type=float, default=0.25)
    ap.add_argument("--p-imprecise", dest="p_imprecise", type=float, default=0.10)
    ap.add_argument("--p-chain-short", dest="p_chain_short", type=float, default=0.15)
    ap.add_argument("--p-stance", dest="p_stance", type=float, default=0.12)
    ap.add_argument("--p-drop-issue", dest="p_drop_issue", type=float, default=0.08)
    ap.add_argument("--p-add-issue", dest="p_add_issue", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dir", dest="outdir", default=os.path.join(ROOT, "evalkit", "runs"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gold = load_jsonl(os.path.join(ROOT, "data", "train.jsonl"))
    rng.shuffle(gold)
    enc = Encoder(os.path.join(ROOT, "models", "bge-small-zh-v1.5"))
    pseudo = build_pseudo(gold, enc, args, rng)

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "pseudo_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in pseudo:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    gold_path = os.path.join(args.outdir, "pseudo_gold.jsonl")
    with open(gold_path, "w", encoding="utf-8") as f:
        for row in gold[:args.n]:
            f.write(json.dumps({k: row[k] for k in ("sample_id", "issue_list",
                                                    "future_argument")},
                               ensure_ascii=False) + "\n")
    docmap = {r["sample_id"]: "".join(d["full_text"] for d in r["docs"])
              for r in gold[:args.n]}

    variants = {"raw": raw_path}
    for label, cfg in {
            "anchor": {"anchor": True},
            "anchor+trim112": {"anchor": True, "trim_future": 112},
            "anchor+merge0.92": {"anchor": True, "merge_threshold": 0.92},
            "trim112_only": {"anchor": False, "trim_future": 112},
    }.items():
        rows = [postprocess_one(json.loads(json.dumps(p, ensure_ascii=False)),
                                docmap[p["sample_id"]], enc,
                                anchor=cfg.get("anchor", False),
                                dedup_chain=cfg.get("anchor", False),
                                merge_threshold=cfg.get("merge_threshold"),
                                trim_future=cfg.get("trim_future")) for p in pseudo]
        path = os.path.join(args.outdir, "pseudo_%s.jsonl" % label)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        variants[label] = path

    out = {}
    for label, path in variants.items():
        res = evaluate(gold_path, path, encoder=enc)
        out[label] = {"F1ext": round(res["F1ext"], 4), "F1pred": round(res["F1pred"], 4),
                      "Score": round(res["Score"], 4)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
