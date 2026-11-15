"""推理侧后处理（P1 路线）：不重训、只改输出。

包含四个独立开关，便于在本地评测器上逐项量化收益：
  1) anchor      把模型给出的论据片段回贴到原文真实句段，保证 100% 逐字可追溯；
  2) dedup_chain 回贴后消除同一样本内的重复片段；
  3) merge_issue 合并近重复议题（名称余弦过高且立场相同），避免多余预测压低 P 与 F1pred 分母；
  4) trim_future 按标注长度分布裁剪 future_argument（ROUGE-L 对过长输出同时罚 R 和 P）。
"""

import argparse
import collections
import difflib
import json
import os
import re

from metrics import Encoder, load_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_SPLIT = re.compile(r"(?<=[。！？；])|(?<=\n)")


def split_sentences(text):
    parts = [p for p in SENT_SPLIT.split(text) if p and p.strip()]
    # 合并过碎的片段（不足 4 字的并入相邻片段），避免句点/换行造成碎片
    merged = []
    for p in parts:
        if merged and len(p.strip()) < 4:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)
    return [m.strip() for m in merged if m.strip()]


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def anchor_snippet(snippet, sentences, doc, min_ratio=0.55):
    """返回原文中与 snippet 最接近的真实句段；找不到足够近的片段时退回原文子串修正。"""
    s = snippet.strip()
    if not s:
        return snippet
    if s in doc:
        return s

    best, best_score = None, 0.0
    n = len(sentences)
    for i in range(n):
        acc = ""
        for j in range(i, min(i + 4, n)):          # 片段最多跨 4 句
            acc = sentences[i] if j == i else acc + sentences[j]
            score = _ratio(s, acc)
            if score > best_score:
                best_score, best = score, acc
    if best is not None and best_score >= min_ratio:
        return best
    # 兜底：截取模型输出中确实存在于原文的最长子串
    for size in range(len(s), 8, -1):
        for start in range(0, len(s) - size + 1):
            piece = s[start:start + size].strip()
            if len(piece) >= 9 and piece in doc:
                return piece
    return snippet


def smooth_stance(issues, majority_min=3):
    """样本内立场平滑：stance 是硬门控，标注在同一文档内高度一致
    （doc-level 多数票可覆盖 87.7% 的议题）。仅当多数立场足够强时才拉齐，
    模拟实验显示 err>=20% 时 +3~4pt 立场准确率，err<=15% 时为负收益。"""
    if not issues:
        return issues
    counter = collections.Counter(i["stance"] for i in issues)
    top, count = counter.most_common(1)[0]
    if count < majority_min:
        return issues
    for i in issues:
        if i["stance"] != top:
            i["stance"] = top
    return issues


def postprocess_one(sample, doc, encoder, issue_vecs=None,
                    anchor=True, dedup_chain=True, merge_threshold=None,
                    trim_future=None, stance_majority=None):
    issues = sample["issue_list"]
    futures = list(sample.get("future_argument", []))
    sentences = split_sentences(doc)

    if anchor:
        for item in issues:
            item["argument_chain"] = [anchor_snippet(a, sentences, doc)
                                      for a in item["argument_chain"]]
    if dedup_chain:
        for item in issues:
            seen, kept = set(), []
            for a in item["argument_chain"]:
                key = a.strip()
                if key and key not in seen:
                    seen.add(key)
                    kept.append(a)
            item["argument_chain"] = kept

    if merge_threshold:
        names = [i["issue_name"] for i in issues]
        vecs = encoder.encode(names)
        sims = vecs @ vecs.T
        used, groups = set(), []
        for i in range(len(issues)):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(issues)):
                if j in used:
                    continue
                if (sims[i, j] >= merge_threshold
                        and issues[i]["stance"] == issues[j]["stance"]):
                    group.append(j)
                    used.add(j)
            used.add(i)
            groups.append(group)
        merged_issues, merged_futures = [], []
        for group in groups:
            keep = issues[group[0]]
            chain = []
            for idx in group:
                for a in issues[idx]["argument_chain"]:
                    if a not in chain:
                        chain.append(a)
            keep["argument_chain"] = chain[:3] if chain else keep["argument_chain"]
            merged_issues.append(keep)
            merged_futures.append(futures[group[0]] if group[0] < len(futures) else "")
        issues, futures = merged_issues, merged_futures

    if trim_future:
        cut = int(trim_future)
        out = []
        for f in futures:
            if len(f) <= cut:
                out.append(f)
                continue
            head = f[:cut]
            for sep in ("。", "；", "，"):
                pos = head.rfind(sep)
                if pos >= cut * 0.6:
                    head = head[:pos + (1 if sep == "。" else 0)]
                    break
            out.append(head)
        futures = out

    if stance_majority:
        issues = smooth_stance(issues, majority_min=int(stance_majority))

    result = {"sample_id": sample["sample_id"], "issue_list": issues,
              "future_argument": futures[:len(issues)]}
    while len(result["future_argument"]) < len(issues):
        result["future_argument"].append("")
    return result


def run(gold_or_pred_path, test_path, out_path, anchor=True, dedup_chain=True,
        merge_threshold=None, trim_future=None, stance_majority=None):
    encoder = Encoder(os.path.join(ROOT, "models", "bge-small-zh-v1.5"))
    preds = load_jsonl(gold_or_pred_path)
    docs = {r["sample_id"]: "".join(d["full_text"] for d in r["docs"])
            for r in load_jsonl(test_path)}
    out = []
    for p in preds:
        doc = docs.get(p["sample_id"], "")
        out.append(postprocess_one(p, doc, encoder, anchor=anchor,
                                   dedup_chain=dedup_chain,
                                   merge_threshold=merge_threshold,
                                   trim_future=trim_future,
                                   stance_majority=stance_majority))
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("写出 %d 条 -> %s" % (len(out), out_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="模型原始输出 jsonl")
    ap.add_argument("--docs", required=True, help="对应的输入集（train/val/test.jsonl）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--merge-threshold", type=float, default=None)
    ap.add_argument("--trim-future", type=int, default=None)
    ap.add_argument("--stance-majority", type=int, default=None,
                    help="样本内多数立场达到该票数时拉齐少数议题，如 3 或 4")
    a = ap.parse_args()
    run(a.pred, a.docs, a.out, anchor=not a.no_anchor, dedup_chain=not a.no_dedup,
        merge_threshold=a.merge_threshold, trim_future=a.trim_future,
        stance_majority=a.stance_majority)
