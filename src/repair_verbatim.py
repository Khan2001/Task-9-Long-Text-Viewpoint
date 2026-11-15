"""本地零成本修复：把提交件里非逐字的 argument_chain 片段吸附回原文连续片段。

背景：gold 的论据链 100% 是原文逐字段落，且以句子为边界（见 textutil.split_spans）；
LoRA 有约 1.6% 的片段只差标点/引号/个别字符。metric.py 的 chain_text 只去空白、保留标点，
因此这类差异会进入 bge 嵌入。吸附方向必为正：它把预测从"偏离原文"挪回"原文连续区间"，
也就是挪回 gold 的取值域。

只改 argument_chain，议题名/立场/future 一律不碰；吸附失败（改写幅度过大）保留原样并计入报告。
默认输出到 preds/，不覆盖 result.jsonl。

用法:
    .venv/bin/python src/repair_verbatim.py --pred result.jsonl --data data/test.jsonl \
        --out preds/result_verbatim.jsonl --dry-run
    .venv/bin/python src/check_submission.py --pred preds/result_verbatim.jsonl
"""

import argparse
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import read_jsonl  # noqa: E402
from textutil import split_spans  # noqa: E402


def ns(text):
    return re.sub(r"\s+", "", text or "")


def strip_map(text):
    """返回 (去空白串, ns 下标 -> 原串下标 的映射)。"""
    chars, idx = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            idx.append(i)
    return "".join(chars), idx


def sentence_at(spans, pos):
    for a, b in spans:
        if a <= pos <= b:
            return a, b
    return None


def snap(src, src_ns, idx, spans, seg_ns, min_keep=0.75, max_blow=1.6, pad=12):
    """把 seg 吸附到原文连续区间并尽量对齐句边界；不宜吸附时返回 None。"""
    if not seg_ns or seg_ns in src_ns:
        return None
    blocks = [b for b in difflib.SequenceMatcher(None, src_ns, seg_ns, autojunk=False)
              .get_matching_blocks() if b.size >= 2]
    if not blocks or sum(b.size for b in blocks) / len(seg_ns) < min_keep:
        return None
    a0 = blocks[0].a
    a1 = blocks[-1].a + blocks[-1].size
    if (a1 - a0) / len(seg_ns) > max_blow:      # 跨度被无关内容撑大 -> 退化为最长匹配块
        longest = max(blocks, key=lambda b: b.size)
        a0, a1 = longest.a, longest.a + longest.size
        if (a1 - a0) / len(seg_ns) > max_blow:
            return None
    p0, p1 = idx[a0], idx[a1 - 1] + 1
    left, right = sentence_at(spans, p0), sentence_at(spans, p1)
    if left and p0 - left[0] <= pad:            # 只在近旁补到句首，避免并入无关文字
        p0 = left[0]
    if right and right[1] - p1 <= pad:
        p1 = right[1]
    fixed = src[p0:p1].strip().lstrip("，、；：,;:（“《「『‘")
    if not fixed or ns(fixed) == seg_ns or ns(fixed) not in src_ns:
        return None
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-keep", type=float, default=0.75, help="片段字符需被原文覆盖的最低比例")
    ap.add_argument("--pad", type=int, default=12, help="向句首/句末扩展的最大字符数")
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    texts = {s["sample_id"]: s["docs"][0]["full_text"] for s in read_jsonl(args.data)}
    rows = read_jsonl(args.pred)
    log, failed = [], []

    for r in rows:
        src = texts.get(r["sample_id"])
        if src is None:
            continue
        src_ns, idx = strip_map(src)
        spans = split_spans(src)
        for it in r.get("issue_list") or []:
            new_chain = []
            for seg in it["argument_chain"]:
                seg_ns = ns(seg)
                if not seg_ns or seg_ns in src_ns:
                    new_chain.append(seg)
                    continue
                fixed = snap(src, src_ns, idx, spans, seg_ns, min_keep=args.min_keep, pad=args.pad)
                if fixed is None:
                    failed.append((r["sample_id"], it["issue_name"], seg[:50]))
                    new_chain.append(seg)
                    continue
                log.append((r["sample_id"], it["issue_name"], seg[:50], fixed[:50]))
                new_chain.append(fixed)
            it["argument_chain"] = new_chain

    left = sum(1 for r in rows for it in r["issue_list"] for s in it["argument_chain"]
               if ns(s) not in ns(texts.get(r["sample_id"], "")))
    total = sum(len(it["argument_chain"]) for r in rows for it in r["issue_list"])
    print(f"证据段总数 {total} | 吸附修复 {len(log)} 段 | 放弃修复 {len(failed)} 段 | "
          f"修复后仍非逐字 {left} 段")
    for sid, name, before, after in log[:args.show]:
        print(f"\n  {sid} / {name}\n    改前: {before}\n    改后: {after}")
    if failed:
        print("\n--- 改写幅度过大、未吸附（保留原样）---")
        for sid, name, seg in failed[:8]:
            print(f"  {sid} / {name}: {seg}")
    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out + ".repairs.json", "w", encoding="utf-8") as fh:
        json.dump({"repaired": log, "failed": failed}, fh, ensure_ascii=False, indent=2)
    print(f"\n已写出 {args.out}（result.jsonl 原文件未动；确认无误后再备份并覆盖）")


if __name__ == "__main__":
    main()
