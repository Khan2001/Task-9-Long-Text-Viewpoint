"""在 val 上逐一验证后处理操作的收益（零成本，复用 preds/val_base.jsonl 与 gold）。

操作矩阵: stance 校准(conf 三档) / 重复议题删除或合并 / 证据裁到3段 /
议题裁到5个 / future 去掉抄证据的开头 / 逐字吸附修复, 以及各有利项的组合。

用法:
    .venv/bin/python src/test_post_ops.py --pred preds/val_base.jsonl --gold data/val.jsonl
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metric import BertScorer, TripleMatcher, chain_text, evaluate_with, norm_issue, read_jsonl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ns(s):
    return re.sub(r"\s+", "", s or "")


# ---------- 嵌入缓存（gold 侧重复编码浪费大） ----------

def cached_embed(matcher, texts, _cache={}):
    miss = [t for t in texts if t not in _cache]
    if miss:
        vecs = matcher._encoder().encode(miss, batch_size=128, normalize_embeddings=True,
                                         show_progress_bar=False).astype("float32")
        for t, v in zip(miss, vecs):
            _cache[t] = v
    return [_cache[t] for t in texts]


def patch_caches(matcher, bs):
    matcher.embed = lambda texts: cached_embed(matcher, list(texts))
    bs_cache = {}
    orig_score = bs.score

    def score(hyp, ref):
        key = (hyp, ref)
        if key not in bs_cache:
            bs_cache[key] = orig_score(hyp, ref)
        return bs_cache[key]
    bs.score = score


# ---------- 各后处理操作 ----------

def op_trim_seg3_keep_first(rows, **kw):
    n = 0
    for r in rows:
        for it in r["issue_list"]:
            if len(it["argument_chain"]) > 3:
                it["argument_chain"] = it["argument_chain"][:3]
                n += 1
    return rows, f"裁段议题数 {n}"


def op_trim_seg3_keep_longest(rows, **kw):
    n = 0
    for r in rows:
        for it in r["issue_list"]:
            if len(it["argument_chain"]) > 3:
                it["argument_chain"] = sorted(it["argument_chain"], key=len, reverse=True)[:3]
                n += 1
    return rows, f"裁段议题数 {n}"


def op_trim_issues5(rows, **kw):
    n = 0
    for r in rows:
        if len(r["issue_list"]) > 5:
            del r["issue_list"][5:]
            r["future_argument"] = (r.get("future_argument") or [])[:len(r["issue_list"])]
            n += 1
    return rows, f"裁议题篇数 {n}"


def find_dup_pairs(rows, enc, name_sim=0.80):
    """与 find_dup_issues.py 同口径: 名sim>=0.80 且证据重叠>=0.3, 或证据重叠>=0.8; 限同立场。"""
    hits = []
    for r in rows:
        ils = r["issue_list"]
        if len(ils) < 2:
            continue
        nm = [norm_issue(x["issue_name"]) for x in ils]
        vs = enc.encode(nm, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        for a in range(len(ils)):
            for b in range(a + 1, len(ils)):
                if ils[a]["stance"] != ils[b]["stance"]:
                    continue
                sim = float(vs[a] @ vs[b])
                sa = {ns(s) for s in ils[a]["argument_chain"]}
                sb = {ns(s) for s in ils[b]["argument_chain"]}
                jac = len(sa & sb) / max(1, len(sa | sb))
                if (sim >= name_sim and jac >= 0.3) or jac >= 0.8:
                    hits.append((r["sample_id"], a, b, round(sim, 3), round(jac, 2)))
    return hits


def op_dedup_delete(rows, enc, **kw):
    hits = find_dup_pairs(rows, enc)
    drop = {}
    for sid, a, b, *_ in hits:
        drop.setdefault(sid, set()).add(b)
    out = []
    for r in rows:
        ds = drop.get(r["sample_id"], set())
        if ds:
            r["issue_list"] = [it for k, it in enumerate(r["issue_list"]) if k not in ds]
            fa = r.get("future_argument") or []
            r["future_argument"] = [f for k, f in enumerate(fa) if k not in ds][:len(r["issue_list"])]
        out.append(r)
    return out, f"重复对 {len(hits)}, 删议题 {sum(len(v) for v in drop.values())}"


def op_dedup_merge(rows, enc, **kw):
    hits = find_dup_pairs(rows, enc)
    drop = {}
    for sid, a, b, *_ in hits:
        drop.setdefault(sid, {})[b] = a   # b 并入 a
    out = []
    for r in rows:
        mp = drop.get(r["sample_id"], {})
        if mp:
            for b in sorted(mp, reverse=True):
                a = mp[b]
                seen = {ns(s) for s in r["issue_list"][a]["argument_chain"]}
                for s in r["issue_list"][b]["argument_chain"]:
                    if ns(s) not in seen:
                        r["issue_list"][a]["argument_chain"].append(s)
                        seen.add(ns(s))
                r["issue_list"][a]["argument_chain"] = r["issue_list"][a]["argument_chain"][:3]
            r["issue_list"] = [it for k, it in enumerate(r["issue_list"]) if k not in mp]
            fa = r.get("future_argument") or []
            r["future_argument"] = [f for k, f in enumerate(fa) if k not in mp][:len(r["issue_list"])]
        out.append(r)
    return out, f"重复对 {len(hits)}, 合并议题 {sum(len(v) for v in drop.values())}"


def op_future_dedup(rows, **kw):
    n = 0
    for r in rows:
        fa = r.get("future_argument") or []
        for k, it in enumerate(r["issue_list"]):
            if k >= len(fa):
                continue
            f = fa[k]
            f_ns = ns(f)
            chain_ns = "".join(ns(s) for s in it["argument_chain"])
            lcp = 0
            for a in range(len(f_ns), 0, -1):
                if f_ns[:a] in chain_ns:
                    lcp = a
                    break
            if lcp >= 15:
                # ns 下标映射回原串下标
                idx = [i for i, ch in enumerate(f) if not ch.isspace()]
                if lcp < len(idx):
                    rest = f[idx[lcp]:].lstrip("，。；;,. ")
                    if len(ns(rest)) >= 10:
                        fa[k] = rest
                        n += 1
    return rows, f"改写 future {n} 条"


def op_verbatim(rows, texts, **kw):
    from repair_verbatim import snap, split_spans, strip_map
    fixed = failed = 0
    for r in rows:
        src = texts.get(r["sample_id"])
        if not src:
            continue
        src_ns, idx = strip_map(src)
        spans = split_spans(src)
        for it in r["issue_list"]:
            new_chain = []
            for seg in it["argument_chain"]:
                seg_ns = ns(seg)
                if not seg_ns or seg_ns in src_ns:
                    new_chain.append(seg)
                    continue
                f = snap(src, src_ns, idx, spans, seg_ns)
                if f is None:
                    failed += 1
                    new_chain.append(seg)
                else:
                    fixed += 1
                    new_chain.append(f)
            it["argument_chain"] = new_chain
    return rows, f"吸附 {fixed}, 放弃 {failed}"


def op_stance(rows, conf, _state={}, **kw):
    if "clf" not in _state:
        from stance_calibrate import build_classifier, embed, issue_texts, text_tag
        clf, enc = build_classifier()
        _state["clf"], _state["enc"] = clf, enc
        _state["fcns"] = [issue_texts, embed, text_tag]
    from stance_calibrate import embed, issue_texts, text_tag
    clf, enc = _state["clf"], _state["enc"]
    names, chains, _ = issue_texts(rows)
    feat = [None, None]
    for j, texts in enumerate([names, chains]):
        tag = text_tag(f"postval_name{j}", texts)
        p = os.path.join("models", tag + ".npy")
        if os.path.exists(p):
            import numpy as np
            feat[j] = __import__("numpy").load(p)
        else:
            feat[j] = embed(enc, texts, tag)
    import numpy as np
    X = np.hstack(feat)
    proba = clf.predict_proba(X)
    classes, arg, confv = list(clf.classes_), proba.argmax(1), proba.max(1)
    changed = 0
    k = 0
    for r in rows:
        for it in r["issue_list"]:
            lab, cf = str(classes[arg[k]]), float(confv[k])
            k += 1
            if cf >= conf and lab != "oppose" and lab != it["stance"]:
                it["stance"] = lab
                changed += 1
    return rows, f"改写立场 {changed} (conf>={conf})"


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="preds/val_base.jsonl")
    ap.add_argument("--gold", default="data/val.jsonl")
    ap.add_argument("--data", default="data/val.jsonl", help="逐字修复取正文的数据文件")
    args = ap.parse_args()

    base = read_jsonl(args.pred)
    gold = read_jsonl(args.gold)
    texts = {s["sample_id"]: s["docs"][0]["full_text"] for s in read_jsonl(args.data)}

    import copy
    matcher, bs = TripleMatcher(threshold=0.7), BertScorer()
    patch_caches(matcher, bs)

    def run_eval(tag, rows, note=""):
        res = evaluate_with(matcher, bs, rows, gold)["overall"]
        print(f"{tag:24s} Score {res['Score']:.4f}  F1_ext {res['F1_ext']:.4f}  "
              f"alpha {res['alpha']:.4f}  S_pred {res['S_pred']:.4f}  "
              f"pred {res['total_pred']} nc {res['total_nc']}   {note}", flush=True)
        return res

    def variant(tag, fn, *fargs, **kw):
        rows = copy.deepcopy(base)
        rows, note = fn(rows, *fargs, **kw)
        return run_eval(tag, rows, note), rows

    print("== 单项操作 ==")
    run_eval("baseline", copy.deepcopy(base))
    r_st06, rows_st06 = variant("stance_c06", op_stance, 0.6)
    r_st07, rows_st07 = variant("stance_c07", op_stance, 0.7)
    r_st08, rows_st08 = variant("stance_c08", op_stance, 0.8)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("models/bge-small-zh-v1.5", cache_folder="models", device="cpu")
    r_dd, rows_dd = variant("dup_delete", op_dedup_delete, enc)
    r_dm, rows_dm = variant("dup_merge", op_dedup_merge, enc)
    variant("trim_seg3_first", op_trim_seg3_keep_first)
    variant("trim_seg3_long", op_trim_seg3_keep_longest)
    variant("trim_issues5", op_trim_issues5)
    variant("future_dedup", op_future_dedup)
    r_vb, rows_vb = variant("verbatim", op_verbatim, texts)

    # 组合: 逐字修复 -> 裁段 -> 去重 -> future -> stance(最优conf)
    print("\n== 组合 ==")
    best_conf = max((("c06", r_st06), ("c07", r_st07), ("c08", r_st08)), key=lambda x: x[1]["Score"])[0]
    conf = {"c06": 0.6, "c07": 0.7, "c08": 0.8}[best_conf]
    best_dedup = max((("delete", r_dd, op_dedup_delete), ("merge", r_dm, op_dedup_merge)),
                     key=lambda x: x[1]["Score"])
    print(f"(组合采用 stance {best_conf}, 去重 {best_dedup[0]})")
    rows = copy.deepcopy(base)
    notes = []
    for fn, fargs in [(op_verbatim, (texts,)), (op_trim_seg3_keep_first, ()),
                      (best_dedup[2], (enc,)), (op_future_dedup, ()),
                      (op_stance, (conf,))]:
        rows, note = fn(rows, *fargs)
        notes.append(note)
    run_eval("combo", rows, " | ".join(notes))

    out = os.path.join(ROOT, "preds", "val_combo.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\ncombo 已写出 -> {out}")


if __name__ == "__main__":
    main()
