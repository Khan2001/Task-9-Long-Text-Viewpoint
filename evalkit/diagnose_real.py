"""在真实模型输出（result.jsonl）上量化后处理的改动幅度，无需 gold。

输出：
  - 论据片段非逐字比例、被 anchor 改写的比例、改写前后 bge 余弦；
  - 近重复议题对数量（不同 merge 阈值下）；
  - future_argument 超长比例。
"""

import json
import os
import sys

import numpy as np

from metrics import Encoder, load_jsonl
from postprocess import anchor_snippet, split_sentences

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(pred_path, docs_path):
    preds = load_jsonl(pred_path)
    docs = {r["sample_id"]: "".join(d["full_text"] for d in r["docs"])
            for r in load_jsonl(docs_path)}
    enc = Encoder(os.path.join(ROOT, "models", "bge-small-zh-v1.5"))

    old, new, flags = [], [], []
    merged_pairs = {t: 0 for t in (0.88, 0.92, 0.95)}
    for p in preds:
        doc = docs[p["sample_id"]]
        sents = split_sentences(doc)
        for item in p["issue_list"]:
            for a in item["argument_chain"]:
                n = anchor_snippet(a, sents, doc)
                old.append(a)
                new.append(n)
                flags.append(n != a.strip())
        names = [i["issue_name"] for i in p["issue_list"]]
        v = enc.encode(names)
        s = v @ v.T
        for t in merged_pairs:
            cnt = sum(1 for i in range(len(names)) for j in range(i + 1, len(names))
                      if s[i, j] >= t)
            merged_pairs[t] += cnt

    vecs = enc.encode([x.strip() for x in old] + new)
    n_all = len(old)
    m = n_all
    cos_same = float(np.mean([float(vecs[i] @ vecs[i + m]) for i in range(n_all)]))
    changed = sum(flags)
    cos_changed = float(np.mean([float(vecs[i] @ vecs[i + m])
                                 for i, f in enumerate(flags) if f]) or 1.0)

    too_long = sum(1 for p in preds for f in p["future_argument"] if len(f) > 112)
    print(json.dumps({
        "snippets": n_all,
        "not_verbatim": sum(1 for p in preds for i in p["issue_list"]
                            for a in i["argument_chain"]
                            if a.strip() not in docs[p["sample_id"]]),
        "rewritten_by_anchor": changed,
        "rewrite_rate": round(changed / n_all, 4),
        "cos_before_after_all": round(cos_same, 4),
        "cos_before_after_rewritten": round(cos_changed, 4),
        "near_duplicate_issue_pairs": {str(k): v for k, v in merged_pairs.items()},
        "future_over_gold_p95": too_long,
        "future_total": sum(len(p["future_argument"]) for p in preds),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
