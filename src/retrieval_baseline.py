"""检索式基线：不依赖大模型生成，用最近邻训练样本迁移议题结构与未来论点。

作用：给"免训练路线"一个可量化的下界，并检验 test 与 train 同场景改写可利用的程度。
用法:
    python src/retrieval_baseline.py --in data/val.jsonl --out preds/val_retr.jsonl --topk 3
"""

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
BGE_NAME = os.path.join(MODEL_CACHE, "bge-small-zh-v1.5")
TRAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train.jsonl")


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(BGE_NAME, cache_folder=MODEL_CACHE, device="cpu")


def embed(enc, texts, cache_key):
    import numpy as np

    cache = os.path.join(MODEL_CACHE, f"{cache_key}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    vecs = enc.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    vecs = vecs.astype("float32")
    np.save(cache, vecs)
    return vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max-issues", type=int, default=5)
    args = ap.parse_args()

    import numpy as np
    from textutil import merge_spans, sentence_texts, split_spans  # noqa: E402

    tr = read_jsonl(args.train)
    tgt = read_jsonl(args.inp)
    enc = encoder()
    tv = embed(enc, [s["docs"][0]["full_text"] for s in tr], "train_text_emb")
    sv = embed(enc, [s["docs"][0]["full_text"] for s in tgt],
               os.path.basename(args.inp).replace(".jsonl", "") + "_text_emb")

    rows, diag = [], []
    for i, s in enumerate(tgt):
        sims = tv @ sv[i]
        nbrs = np.argsort(-sims)[:args.topk]
        best = tr[int(nbrs[0])]
        text = s["docs"][0]["full_text"]
        spans = split_spans(text)
        sents = sentence_texts(text, spans)
        ev_texts = []
        for it in best["issue_list"][:args.max_issues]:
            ev_texts.extend(it["argument_chain"])
        ev_emb = enc.encode([e[:480] for e in ev_texts], normalize_embeddings=True, show_progress_bar=False)
        sent_emb = enc.encode([x[:480] for x in sents], normalize_embeddings=True, show_progress_bar=False)
        items = []
        cursor = 0
        for it in best["issue_list"][:args.max_issues]:
            k = len(it["argument_chain"])
            group = ev_emb[cursor:cursor + k]
            cursor += k
            scores = sent_emb @ group.T
            idxs = sorted({int(j) for j in scores.max(axis=1).argsort()[::-1][:k * 2]})
            chain = merge_spans(text, spans, idxs, max_groups=max(1, k))
            if not chain:
                chain = [sents[0]]
            items.append({"issue_name": it["issue_name"], "stance": it["stance"],
                          "argument_chain": chain[:3]})
        fut = best["future_argument"][:len(items)]
        while len(fut) < len(items):
            fut.append(f"围绕{items[len(fut)]['issue_name']}，各方将延续既有表态，推动机制化磋商与务实安排。")
        rows.append({"sample_id": s["sample_id"], "issue_list": items, "future_argument": fut})
        diag.append({"sample_id": s["sample_id"], "top_sim": float(sims[nbrs[0]]),
                     "neighbor": best["sample_id"]})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    sims = [d["top_sim"] for d in diag]
    print(f"{len(rows)} 条 -> {args.out}；最近邻相似度 mean={np.mean(sims):.3f} "
          f"p10={np.percentile(sims,10):.3f} p90={np.percentile(sims,90):.3f} "
          f">0.9 占比={np.mean([x>0.9 for x in sims]):.1%}")


if __name__ == "__main__":
    main()
