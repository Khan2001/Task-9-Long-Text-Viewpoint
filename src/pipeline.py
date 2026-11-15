"""样本 -> 模型输入 -> 合规 result.jsonl 的装配与后处理。"""

import re

from prompts import parse_json_block
from textutil import (align_quote, merge_spans, number_text, parse_indices,
                      sentence_texts, split_spans)

STANCE_MAP = {
    "support": "support", "supports": "support", "支持": "support", "肯定": "support", "积极": "support",
    "oppose": "oppose", "opposed": "oppose", "反对": "oppose", "批评": "oppose", "否定": "oppose",
    "neutral": "neutral", "中立": "neutral", "客观": "neutral", "中性": "neutral",
}
STANCE_FALLBACK = "support"  # 训练集占比 63%，无法判读时取先验众数


def clean_issue_name(name):
    s = re.sub(r"^子议题\d*[：:]\s*|^主议题[：:]\s*", "", str(name or "").strip())
    s = re.sub(r"[\"'“”《》\[\]（）()【】]", "", s).strip(" ，,。;；")
    return s[:20]


def norm_stance(raw):
    key = str(raw or "").strip().lower()
    for k, v in STANCE_MAP.items():
        if k in key:
            return v
    return STANCE_FALLBACK


def build_input(sample, numbered=True):
    """返回 (doc, 句偏移列表, 送模型的文本)。证据一律按偏移从原文切片。"""
    doc = sample["docs"][0]
    spans = split_spans(doc["full_text"])
    lines = sentence_texts(doc["full_text"], spans)
    text = number_text(lines) if numbered else doc["full_text"]
    return doc, spans, text


def example_from_gold(sample, max_issues=5):
    """把训练样本压成 few-shot 示范：只给议题名/立场/证据首句，不给编号。"""
    issues = []
    for it in sample["issue_list"][:max_issues]:
        issues.append({
            "issue_name": it["issue_name"],
            "stance": it["stance"],
            "evidence_ids": [1],
        })
    return {"issue_list": issues}


def assemble(sample, extract_raw, future_raw=None, max_issues=5, report=None, max_groups=3,
             group_gap=1, pad=0, max_chain_chars=0):
    doc, spans, _ = build_input(sample)
    text = doc["full_text"]
    n_sents = len(spans)
    report = report if report is not None else []
    parsed = extract_raw if isinstance(extract_raw, dict) else parse_json_block(extract_raw)
    if not parsed or not parsed.get("issue_list"):
        report.append(f"{sample['sample_id']}: 抽取输出无法解析，已回退为整篇单议题")
        parsed = {"issue_list": [{"issue_name": doc["doc_type"][:10] + "总体态势", "stance": "support",
                                 "evidence_ids": [1, 2]}]}
    items = []
    used = set()
    for it in parsed["issue_list"]:
        name = clean_issue_name(it.get("issue_name"))
        if not name:
            continue
        ids = [i for i in parse_indices(it.get("evidence_ids", it.get("argument_chain", [])), n_sents)
               if i not in used]
        if pad:  # 证据常与上下文同段，向前后各扩 pad 句提升论据链覆盖
            ids = sorted({i for x in ids for i in range(max(0, x - pad), min(n_sents, x + pad + 1))} - used)
        used.update(ids)
        chain = merge_spans(text, spans, ids, max_groups=max_groups, group_gap=group_gap)
        if max_chain_chars and sum(len(c) for c in chain) > max_chain_chars:
            # max_groups 合并会把首尾选中句之间的无关句一并纳入，导致证据过长稀释相似度；
            # 超限时退回只保留选中的离散原句（仍为原文逐字片段），仍超则按句截到上限。
            chain = merge_spans(text, spans, ids, max_groups=max(1, len(ids)), group_gap=0)
            if sum(len(c) for c in chain) > max_chain_chars:
                trimmed, total = [], 0
                for seg in chain:
                    if trimmed and total + len(seg) > max_chain_chars:
                        break
                    trimmed.append(seg)
                    total += len(seg)
                chain = trimmed or chain[:1]
        if not chain:
            quote = it.get("argument_chain")
            q = quote[0] if isinstance(quote, list) and quote else quote
            aligned = align_quote(re.sub(r"\[\d+\]", "", str(q or "")), text, spans) \
                or text[spans[0][0]:spans[0][1]].strip()
            chain = [aligned]
            report.append(f"{sample['sample_id']}/{name}: 无有效证据编号，回退近似原文句")
        items.append({"issue_name": name, "stance": norm_stance(it.get("stance")),
                      "argument_chain": chain})
    items = items[:max_issues]
    if not items:
        items = [{"issue_name": doc["doc_type"][:10] + "总体态势", "stance": STANCE_FALLBACK,
                  "argument_chain": [text[spans[0][0]:spans[0][1]].strip()]}]

    future = None
    if future_raw is not None:
        fp = future_raw if isinstance(future_raw, dict) else parse_json_block(future_raw)
        future = (fp or {}).get("future_argument")
    if future is not None:
        future = [re.sub(r"^\s*\d+[\.、)]\s*", "", str(x).strip()) for x in future if str(x).strip()]
        if len(future) < len(items):
            report.append(f"{sample['sample_id']}: 预测条数 {len(future)}<{len(items)}，已按议题名补写占位预测")
            while len(future) < len(items):
                it = items[len(future)]
                future.append(f"围绕{it['issue_name']}，后续各方将延续当前表态，"
                              f"推动相关机制化安排与务实合作落地，并以此稳定整体预期。")
        future = future[:len(items)]
    return {"sample_id": sample["sample_id"], "issue_list": items, "future_argument": future}, report
