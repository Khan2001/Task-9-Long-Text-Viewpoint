"""长文本切句（保留字符偏移）与原文片段还原。

设计要点：证据一律用 text[start:end] 从原文切片得到，保证逐字可核验，
不会因为拼接、去空格或去换行而偏离原文。
"""

import re

SENT_END = "。！？；"
_NUM = re.compile(r"^\[(\d+)\]\s*")


def split_spans(text):
    """切句并返回 (start, end) 偏移，end 含句末标点，起止均指向原文位置。"""
    spans = []
    start = 0
    for i, ch in enumerate(text):
        if ch in SENT_END:
            if text[start:i + 1].strip():
                spans.append((start, i + 1))
            start = i + 1
        elif ch == "\n":
            if text[start:i].strip():
                spans.append((start, i))
            start = i + 1
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    # 去掉每段首尾空白对应的偏移，仍保证切片是原文子串
    trimmed = []
    for a, b in spans:
        seg = text[a:b]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        if b - a - lead - trail > 0:
            trimmed.append((a + lead, b - trail))
    return trimmed


def sentence_texts(text, spans):
    return [text[a:b] for a, b in spans]


def number_text(lines, marker="[{i}]"):
    """给句子加编号，供模型按编号选证据，保证输出可逐字还原。"""
    return "\n".join(marker.format(i=i + 1) + " " + s for i, s in enumerate(lines))


def strip_number_prefix(line):
    m = _NUM.match(line.strip())
    return line.strip()[m.end():].strip() if m else line.strip()


def parse_indices(raw, n_sents):
    """宽松解析模型返回的句子编号，去重并升序，越界丢弃。"""
    out = []
    for tok in re.findall(r"\[?(\d{1,4})\]?", str(raw or "")):
        idx = int(tok) - 1
        if 0 <= idx < n_sents and idx not in out:
            out.append(idx)
    return sorted(out)


def merge_spans(text, spans, idxs, max_groups=3, group_gap=1):
    """把命中的句子编号合并成至多 max_groups 段原文连续切片。

    连续或近似相邻（间隔 <= group_gap 句）的命中会并成一段，
    段内容直接取原文 text[a:b]，因此一定是原文子串。
    """
    if not idxs:
        return []
    groups = []
    cur = [idxs[0]]
    for prev, idx in zip(idxs, idxs[1:]):
        if idx - prev <= group_gap + 1:
            cur.append(idx)
        else:
            groups.append(cur)
            cur = [idx]
    groups.append(cur)
    if len(groups) <= max_groups:
        return [text[spans[g[0]][0]:spans[g[-1]][1]].strip() for g in groups]
    # 组数超限时不做字符串拼接（会产生非原文片段），
    # 而是把相邻两组并成一个连续原文区间，切片始终是原文子串。
    while len(groups) > max_groups:
        widths = [spans[g[-1]][1] - spans[g[0]][0] for g in groups]
        j = min(range(len(groups) - 1),
                key=lambda k: widths[k] + widths[k + 1]
                + spans[groups[k + 1][0]][0] - spans[groups[k][-1]][1])
        groups = groups[:j] + [groups[j] + groups[j + 1]] + groups[j + 2:]
    return [text[spans[g[0]][0]:spans[g[-1]][1]].strip() for g in groups]


def align_quote(quote, text, spans, good_ratio=0.6):
    """模型自由引用时，回退到最相近的原文句子（保证可核验、可追溯）。"""
    from difflib import SequenceMatcher

    q = "".join(str(quote or "").split())
    if not q:
        return None
    best, best_r = None, 0.0
    for a, b in spans:
        seg = text[a:b]
        ns = "".join(seg.split())
        if not ns:
            continue
        if q and (q in ns or ns in q):
            return seg.strip()
        r = SequenceMatcher(None, q, ns).ratio()
        if r > best_r:
            best, best_r = seg.strip(), r
    return best if best_r >= good_ratio * 0.5 else None
