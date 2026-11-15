"""分析 probe_stance.py 的输出：三变体立场准确率、塌缩指标、少数派议题上的召回，
并直接给出"用逐议题独立判断替换联合立场"的期望 ΔF1ext。

用法：python analyze_probe.py <probe_stance.jsonl>
"""

import collections
import json
import sys

import numpy as np

LABELS = ("support", "oppose", "neutral")


def per_issue(records, kind):
    out = []
    for r in records:
        if r["kind"] != kind:
            continue
        if kind == "J":
            for gold, pred in zip(r["gold"], r["pred"]):
                out.append((r["sample_id"], gold, pred))
        else:
            out.append((r["sample_id"], r["gold"], r["pred"]))
    return out


def acc(rows):
    ok = [r for r in rows if r[2] in LABELS]
    return (sum(1 for _, g, p in ok if g == p) / len(ok) if ok else 0.0, len(ok), len(rows))


def confusion(rows):
    m = collections.Counter((g, p) for _, g, p in rows if p in LABELS)
    table = {}
    for g in LABELS:
        tot = sum(m[(g, p)] for p in LABELS)
        table[g] = {p: m[(g, p)] for p in LABELS}
        table[g + "_recall"] = round(m[(g, g)] / tot, 3) if tot else None
    for p in LABELS:
        tot = sum(m[(g, p)] for g in LABELS)
        table[p + "_precision"] = round(m[(p, p)] / tot, 3) if tot else None
    return table


def collapse(rows):
    by_sid = collections.defaultdict(list)
    for sid, g, p in rows:
        by_sid[sid].append((g, p))
    pure_pred, maj_pred, minor_pred = [], [], []
    pure_gold, maj_gold, minor_gold = [], [], []
    for sid, lst in by_sid.items():
        ps = [p for _, p in lst if p in LABELS]
        gs = [g for g, _ in lst]
        if not ps:
            continue
        cp = collections.Counter(ps)
        cg = collections.Counter(gs)
        pure_pred.append(len(set(ps)) == 1)
        pure_gold.append(len(set(gs)) == 1)
        maj_pred.append(cp.most_common(1)[0][1] / len(ps))
        maj_gold.append(cg.most_common(1)[0][1] / len(gs))
        minor_pred.append(len(ps) - cp.most_common(1)[0][1])
        minor_gold.append(len(gs) - cg.most_common(1)[0][1])
    return {"pred_纯一样本占比": round(float(np.mean(pure_pred)), 3),
            "gold_纯一样本占比": round(float(np.mean(pure_gold)), 3),
            "pred_多数票占比": round(float(np.mean(maj_pred)), 3),
            "pred_少数派议题数": round(float(np.mean(minor_pred)), 2),
            "gold_少数派议题数": round(float(np.mean(minor_gold)), 2)}


def minority_split(gold_rows, rows):
    """gold 中属于样本内少数派的议题，各变体答对多少。"""
    by_sid = collections.defaultdict(list)
    for sid, g, _ in gold_rows:
        by_sid[sid].append(g)
    minor_flag = {}
    for sid, gs in by_sid.items():
        maj = collections.Counter(gs).most_common(1)[0][0]
        for i, g in enumerate(gs):
            minor_flag[(sid, i)] = (g != maj)
    stats = {}
    for lab, subset in (("全部议题", [1] * len(rows)),
                        ("多数派议题", [not minor_flag.get((sid, i), False)
                                       for i, (sid, _, _) in enumerate(rows)]),
                        ("少数派议题", [minor_flag.get((sid, i), False)
                                       for i, (sid, _, _) in enumerate(rows)])):
        idx = [k for k, keep in enumerate(subset) if keep]
        hit = sum(1 for k in idx if rows[k][2] in LABELS and rows[k][1] == rows[k][2])
        stats[lab] = {"n": len(idx), "acc": round(hit / len(idx), 3) if idx else None}
    return stats


def main(path):
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    gold_rows = per_issue(records, "I")          # 以 I 的逐议题记录作为 (sid, gold) 基准
    report = {"n_records": len(records)}
    for kind in ("I", "N", "J"):
        rows = per_issue(records, kind)
        if not rows:
            continue
        a, ok_n, tot = acc(rows)
        report["variant_" + kind] = {
            "准确率": round(a, 4), "有效解析": ok_n, "总请求": tot,
            "解析失败": tot - ok_n,
            "混淆与分类指标": confusion(rows),
            "塌缩指标": collapse(rows),
            "分多数派少数派": minority_split(gold_rows, rows),
        }
    if "variant_I" in report and "variant_J" in report:
        d = report["variant_I"]["准确率"] - report["variant_J"]["准确率"]
        report["结论"] = {
            "I_minus_J_stance_accuracy": round(d, 4),
            "若用 I 替换联合输出的立场，期望 ΔF1ext": round(0.98 * d, 4),
            "期望 ΔScore": round(0.8 * 0.98 * d, 4),
            "说明": "0.98 来自本地敏感度实验：F1ext ≈ 1 - 0.98×立场错误率（其余项全对时）",
        }
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
