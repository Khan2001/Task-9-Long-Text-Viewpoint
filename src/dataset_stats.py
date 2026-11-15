"""三份数据横向画像：data/train.jsonl、data/val.jsonl 与任意预测件（如 result.jsonl）。

用途：预测件没有标签，无法直接算分；但它的分布若偏离 gold 分布，就是可疑点。
本脚本把 输入侧/议题侧/证据侧/future 侧 的统计并排打出来，供人工判断"哪里不像 gold"。

用法:
    .venv/bin/python src/dataset_stats.py                          # 默认三份
    .venv/bin/python src/dataset_stats.py --preds result.jsonl preds/result_verbatim.jsonl
    .venv/bin/python src/dataset_stats.py --sections input,issue,chain,future
"""

import argparse
import json
import re
import statistics
from collections import Counter

HEAD_MARK = re.compile(r"^\s*[一二三四五六七八九十0-9]+[、.．）)]\s*\S{2,20}\s*$|"
                       r"^\s*[（(]\d+[)）]\s*\S{2,20}\s*$")


def ns(s):
    return re.sub(r"\s+", "", s or "")


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))] if vals else 0


def doc_texts(rows, fallback_path=None):
    """预测件本身不带正文，需要按 sample_id 从数据文件补，才能算输入侧与位置统计。"""
    if all(r.get("docs") for r in rows):
        return {r["sample_id"]: [x for x in r["docs"]] for r in rows}
    fb = {s["sample_id"]: s["docs"] for s in read_jsonl(fallback_path)}
    return {r["sample_id"]: fb.get(r["sample_id"], []) for r in rows}


def summarize(rows, has_label, docs=None):
    d = {"篇数": len(rows), "有标签": has_label}
    docs = docs or {}
    rows_docs = [(r, docs.get(r["sample_id"], [])) for r in rows]
    rows_docs = [(r, x) for r, x in rows_docs if x]
    # --- 输入侧 ---
    chars = [sum(len(x["full_text"]) for x in ds) for _, ds in rows_docs]
    d["docs/篇"] = round(statistics.mean([len(ds) for _, ds in rows_docs]), 2)
    d["文档字数 mean/p90/max"] = f"{int(statistics.mean(chars))}/{pct(chars, .9)}/{max(chars)}"
    dt = Counter(x.get("doc_type", "?") for _, ds in rows_docs for x in ds)
    d["doc_type 占比"] = {k: round(v / sum(dt.values()), 3) for k, v in dt.most_common(5)}
    heads = sum(1 for _, ds in rows_docs for x in ds if HEAD_MARK.search(x["full_text"]))
    d["带分节小标题的篇数占比"] = round(heads / max(1, sum(len(ds) for _, ds in rows_docs)), 3)

    # --- 议题侧 ---
    issues = [it for r in rows for it in r.get("issue_list") or []]
    per = Counter(len(r.get("issue_list") or []) for r in rows)
    d["议题数/篇 mean"] = round(len(issues) / max(1, len(rows)), 2)
    d["议题数分布"] = dict(sorted(per.items()))
    st = Counter(it.get("stance") for it in issues)
    tot = sum(st.values())
    d["立场占比"] = {k: round(v / tot, 3) for k, v in st.most_common()}
    nl = [len(ns(it["issue_name"])) for it in issues]
    d["议题名字数 mean/p90/max"] = f"{statistics.mean(nl):.1f}/{pct(nl, .9)}/{max(nl)}"
    d["议题名>14字占比"] = round(sum(1 for x in nl if x > 14) / len(nl), 3)
    d["议题名含『与/及/和』占比"] = round(
        sum(1 for it in issues if re.search(r"[与及和]", ns(it["issue_name"]))) / len(issues), 3)
    dupn = sum(len(r["issue_list"]) - len({ns(i["issue_name"]) for i in r["issue_list"]})
               for r in rows if r.get("issue_list"))
    d["同篇重名议题数"] = dupn

    # --- 证据侧 ---
    segc = Counter(len(it["argument_chain"]) for it in issues)
    d["每议题证据段数"] = {k: round(v / len(issues), 3) for k, v in sorted(segc.items())}
    allseg = [s for it in issues for s in it["argument_chain"]]
    d["证据段字数 mean/p90/max"] = (f"{statistics.mean([len(ns(s)) for s in allseg]):.0f}/"
                             f"{pct([len(ns(s)) for s in allseg], .9)}/{max(len(ns(s)) for s in allseg)}")
    tc = sorted(sum(len(ns(s)) for s in it["argument_chain"]) for it in issues)
    d["证据链总字数 mean/p90/max"] = f"{statistics.mean(tc):.0f}/{pct(tc, .9)}/{tc[-1]}"
    srcs = {r["sample_id"]: "".join(ns(x["full_text"]) for x in ds) for r, ds in rows_docs}
    verb = sum(1 for r in rows for it in r["issue_list"] for s in it["argument_chain"]
               if ns(s) in srcs.get(r["sample_id"], ""))
    d["证据逐字率"] = round(verb / len(allseg), 4)
    reuse = sum(len(c) - len({ns(s) for s in c}) for it in issues for c in [it["argument_chain"]])
    cross = 0
    for r in rows:
        seen = set()
        for it in r.get("issue_list") or []:
            for s in it["argument_chain"]:
                if ns(s) in seen:
                    cross += 1
                seen.add(ns(s))
    d["议题内重复段"] = reuse
    d["跨议题复用证据占比"] = round(cross / max(1, len(allseg)), 3)
    # 证据取自文档的哪一段（用首个匹配位置的相对分位）
    pos = []
    for r, ds in rows_docs:
        text = "".join(ns(x["full_text"]) for x in ds)
        if not text:
            continue
        for it in r.get("issue_list") or []:
            for s in it["argument_chain"]:
                i = text.find(ns(s))
                if i >= 0:
                    pos.append(i / len(text))
    if pos:
        b = Counter("前1/4" if p < .25 else "中" if p < .75 else "后1/4" for p in pos)
        d["证据位置分布"] = {k: round(v / sum(b.values()), 3)
                        for k, v in sorted(b.items(), key=lambda x: ["前1/4", "中", "后1/4"].index(x[0]))}

    # --- future 侧 ---
    fu = [f for r in rows for f in (r.get("future_argument") or [])]
    if fu:
        fl = [len(ns(f)) for f in fu]
        d["future 字数 mean/p90/max"] = f"{statistics.mean(fl):.0f}/{pct(fl, .9)}/{max(fl)}"
        d["future 条数/议题"] = round(len(fu) / max(1, len(issues)), 2)
        d["future 开头高频词"] = Counter(re.sub(r"\s", "", f)[:4] for f in fu).most_common(4)
        d["future 含预测词占比"] = round(
            sum(1 for f in fu if re.search(r"将|可能|预计|有望|进一步|未来|或", f)) / len(fu), 3)
        d["future 与自身证据重合度"] = round(
            sum(1 for r in rows for k, it in enumerate(r.get("issue_list") or [])
                if k < len(r.get("future_argument") or [])
                and ns(r["future_argument"][k])[:20] in "".join(ns(s) for s in it["argument_chain"]))
            / max(1, len(issues)), 3)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--preds", nargs="*", default=["result.jsonl"])
    ap.add_argument("--pred-data", default="data/test.jsonl", help="预测件缺正文时按 sample_id 补")
    ap.add_argument("--sections", default="input,issue,chain,future")
    args = ap.parse_args()

    cols = {}
    for path, labeled in ((args.train, True), (args.val, True)):
        rows = read_jsonl(path)
        cols[f"gold:{path.split('/')[-1].split('.')[0]}"] = (rows, doc_texts(rows))
    pred_rows = read_jsonl(args.pred_data)
    for path in args.preds:
        rows = read_jsonl(path)
        cols[f"pred:{path.split('/')[-1]}"] = (rows, doc_texts(rows, args.pred_data))

    stats = {k: summarize(v[0], k.startswith("gold"), v[1]) for k, v in cols.items()}
    names = list(stats)
    keys = list(stats[names[0]])
    print(f"{'指标':26s} " + " | ".join(f"{n:>22s}" for n in names))
    print("-" * (28 + 25 * len(names)))
    for k in keys:
        print(f"{k:26s} " + " | ".join(f"{str(stats[n].get(k, '-')):>22s}" for n in names))


if __name__ == "__main__":
    main()
