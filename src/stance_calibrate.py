"""立场校准后处理：用 train.jsonl 训练一个 bge 特征的小分类器，纠正基座模型判错的立场。

动机（见 diagnose 与实验结论）：官方指标要求 stance 严格相等才允许匹配，
而 Qwen3-32B 在"客观进展/机制化表述"上倾向给 support，标注却多为 neutral，
导致大量语义已经过阈值的候选被立场门槛直接丢弃。该模块仅改写 stance 字段，
不动 issue_name / argument_chain，属于可离线复现的辅助模块（需在复现材料中披露）。

用法:
    python src/stance_calibrate.py --pred preds/val_api_v2.jsonl --out preds/val_stance.jsonl
    python src/stance_calibrate.py --pred result.jsonl --out result_stance.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metric import BGE_NAME, MODEL_CACHE, chain_text, norm_issue  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data", "train.jsonl")


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def issue_texts(rows):
    names, chains, labels = [], [], []
    for s in rows:
        for it in s.get("issue_list") or []:
            names.append(norm_issue(it.get("issue_name")))
            chains.append(chain_text(it.get("argument_chain")))
            labels.append(it.get("stance"))
    return names, chains, labels


def embed(enc, texts, tag):
    import numpy as np

    path = os.path.join(MODEL_CACHE, tag + ".npy")
    if os.path.exists(path):
        return np.load(path)
    vecs = enc.encode(texts, batch_size=128, normalize_embeddings=True,
                      show_progress_bar=False).astype("float32")
    np.save(path, vecs)
    return vecs


def text_tag(prefix, texts):
    """用文本内容的摘要做缓存键，避免不同预测文件共用同一份嵌入缓存。"""
    h = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


def build_classifier(train_path=TRAIN, c=4.0):
    """在 train.jsonl 的（议题名, 证据链）上训练立场分类器，返回 (clf, encoder)。"""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    enc = SentenceTransformer(BGE_NAME, cache_folder=MODEL_CACHE, device="cpu")
    tr = read_jsonl(train_path)
    names, chains, labels = issue_texts(tr)
    trn = embed(enc, names, "stance_tr_name")
    trc = embed(enc, chains, "stance_tr_chain")
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=c)
    clf.fit(np.hstack([trn, trc]), labels)
    return clf, enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="待校准的预测 jsonl（含 issue_list[].stance）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--conf", type=float, default=0.7, help="分类器置信度阈值，低于此值保留原立场")
    ap.add_argument("--allow-oppose", action="store_true",
                    help="允许分类器把立场改成 oppose（默认关闭：oppose 仅占 3%，误判代价高）")
    args = ap.parse_args()

    import numpy as np

    clf, enc = build_classifier(args.train)
    rows = read_jsonl(args.pred)
    names, chains, _ = issue_texts(rows)
    feat = np.hstack([embed(enc, names, text_tag("stance_pred_name", names)),
                      embed(enc, chains, text_tag("stance_pred_chain", chains))])
    proba = clf.predict_proba(feat)
    classes = list(clf.classes_)
    arg = proba.argmax(1)
    conf = proba.max(1)

    changed = Counter()
    k = 0
    for s in rows:
        for it in s.get("issue_list") or []:
            label = str(classes[arg[k]])
            cf = float(conf[k])
            old = it.get("stance")
            k += 1
            if cf < args.conf:
                continue
            if not args.allow_oppose and label == "oppose":
                continue
            if label != old:
                changed[f"{old}->{label}"] += 1
                it["stance"] = label

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(changed.values())
    print(f"校准完成: {len(rows)} 条样本 / {k} 个议题，改写立场 {total} 个 (conf>={args.conf})")
    for key, v in changed.most_common():
        print(f"  {key:20s} {v}")
    print(f"输出 -> {args.out}")
    gold = os.path.join(ROOT, "data", "val.jsonl")
    print(f"评分: python src/metric.py --pred {args.out} --gold {gold}")


if __name__ == "__main__":
    main()
