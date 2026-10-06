"""复刻赛题官方指标：观点抽取 S_ext 与未来观点预测 S_pred。

用法:
    python src/metric.py --pred preds/val_base.jsonl --gold data/val.jsonl
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

STANCES = ("support", "oppose", "neutral")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CACHE = os.path.join(ROOT, "models")
BGE_NAME = os.path.join(MODEL_CACHE, "bge-small-zh-v1.5")
BERT_NAME = os.path.join(MODEL_CACHE, "bert-base-chinese")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_issue(name):
    return "".join(str(name or "").split())


def chain_text(chain):
    parts = []
    for seg in chain or []:
        s = "".join(str(seg or "").split())
        if s:
            parts.append(s)
    return "\n".join(parts)


def future_text(text):
    return "".join(str(text or "").split())


class TripleMatcher:
    """议题 0.4 + 论据链 0.6 的 bge 余弦加权分，阈值过滤后做一对一最大权匹配。"""

    def __init__(self, threshold=0.7, w_issue=0.4, w_chain=0.6):
        self.threshold = threshold
        self.w_issue = w_issue
        self.w_chain = w_chain
        self._enc = None
        self._cache = {}

    def _encoder(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer

            self._enc = SentenceTransformer(BGE_NAME, cache_folder=MODEL_CACHE, device="cpu")
        return self._enc

    def embed(self, texts):
        missing = [t for t in texts if t and t not in self._cache]
        if missing:
            vecs = self._encoder().encode(
                list(OrderedDict.fromkeys(missing)),
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for t, v in zip(OrderedDict.fromkeys(missing), vecs):
                self._cache[t] = v
        return [self._cache.get(t) for t in texts]

    def cos(self, a, b):
        if a is None or b is None:
            return 0.0
        return float(max(0.0, min(1.0, float(a @ b))))

    def pair_score(self, pred, gold):
        pi, gi = norm_issue(pred["issue_name"]), norm_issue(gold["issue_name"])
        pc, gc = chain_text(pred["argument_chain"]), chain_text(gold["argument_chain"])
        v = self.embed([pi, gi, pc, gc])
        sim_issue = self.cos(v[0], v[1])
        sim_chain = self.cos(v[2], v[3])
        return self.w_issue * sim_issue + self.w_chain * sim_chain, sim_issue, sim_chain

    def match(self, preds, golds):
        """返回 [(pred_idx, gold_idx, score)]，立场不同的对不参与匹配。"""
        n, m = len(preds), len(golds)
        if n == 0 or m == 0:
            return []
        score = [[0.0] * m for _ in range(n)]
        ok = [[False] * m for _ in range(n)]
        for i, p in enumerate(preds):
            for j, g in enumerate(golds):
                if str(p.get("stance")) != str(g.get("stance")):
                    continue
                s, _, _ = self.pair_score(p, g)
                score[i][j] = s
                ok[i][j] = s > self.threshold
        pairs = _max_matching(n, m, score, ok)
        return [(i, j, score[i][j]) for i, j in pairs]


def _max_matching(n, m, score, ok):
    """先最大化匹配数、再最大化总分的一对一匹配（KM 形式实现）。"""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    big = 10.0
    cost = np.where(ok, big - np.array(score), big + 1.0)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if ok[i][j]]


class BertScorer:
    """bert-base-chinese 上的 BERTScore-F1（greedy 匹配，不做基线重标定）。"""

    def __init__(self):
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(BERT_NAME, cache_dir=MODEL_CACHE)
            self._model = AutoModel.from_pretrained(BERT_NAME, cache_dir=MODEL_CACHE, dtype=torch.float32)
            self._model.eval()
        return self._model

    def _emb(self, text):
        import torch

        model = self._load()
        enc = self._tok(text if text else "空", return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**enc).last_hidden_state[0]
        mask = enc["attention_mask"][0].bool()
        vec = out[mask]
        return torch.nn.functional.normalize(vec, dim=-1)

    def score(self, hyp, ref):
        import torch

        if not hyp or not ref:
            return 0.0
        h = self._emb(hyp)
        r = self._emb(ref)
        sim = h @ r.T
        p = float(sim.max(dim=1).values.mean())
        rc = float(sim.max(dim=0).values.mean())
        if p + rc <= 0:
            return 0.0
        return 2 * p * rc / (p + rc)


def rouge_l_f1(hyp, ref):
    """中文按字符切分的 ROUGE-L F1。"""
    x, y = list(hyp), list(ref)
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for a in x:
        cur = [0] * (len(y) + 1)
        for j, b in enumerate(y, 1):
            cur[j] = prev[j - 1] + 1 if a == b else max(cur[j - 1], prev[j])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec = lcs / len(x)
    rec = lcs / len(y)
    return 2 * prec * rec / (prec + rec)


def validate(rows, gold_ids=None, texts=None):
    errors = []
    seen = set()
    nonverbatim = 0
    for i, r in enumerate(rows):
        sid = r.get("sample_id")
        if not isinstance(sid, str) or not sid:
            errors.append(f"row {i}: sample_id 缺失或非字符串")
            continue
        if sid in seen:
            errors.append(f"{sid}: sample_id 重复")
        seen.add(sid)
        il = r.get("issue_list")
        fa = r.get("future_argument")
        if not isinstance(il, list) or not il:
            errors.append(f"{sid}: issue_list 必须是非空数组")
            continue
        if not isinstance(fa, list) or len(fa) != len(il):
            errors.append(f"{sid}: future_argument 数量 {len(fa) if isinstance(fa, list) else 'N/A'} != issue_list {len(il)}")
        for k, it in enumerate(il):
            if not isinstance(it, dict):
                errors.append(f"{sid}[{k}]: 议题元素必须是对象")
                continue
            if not norm_issue(it.get("issue_name")):
                errors.append(f"{sid}[{k}]: issue_name 为空")
            if it.get("stance") not in STANCES:
                errors.append(f"{sid}[{k}]: stance 非法 {it.get('stance')!r}")
            ac = it.get("argument_chain")
            if not isinstance(ac, list) or not ac or not all(isinstance(s, str) and s.strip() for s in ac):
                errors.append(f"{sid}[{k}]: argument_chain 必须是非空字符串数组")
            elif texts and sid in texts:
                src = "".join(texts[sid].split())
                for seg in ac:
                    if "".join(seg.split()) not in src:
                        nonverbatim += 1
        for k, s in enumerate(fa if isinstance(fa, list) else []):
            if not isinstance(s, str) or not s.strip():
                errors.append(f"{sid}[{k}]: future_argument 元素为空")
    if gold_ids is not None:
        missing = [s for s in gold_ids if s not in seen]
        extra = [s for s in seen if s not in set(gold_ids)]
        if missing:
            errors.append(f"缺失样本 {len(missing)} 条: {missing[:5]}")
        if extra:
            errors.append(f"多余样本 {len(extra)} 条: {extra[:5]}")
    if nonverbatim:
        errors.append(f"证据非原文逐字段落 {nonverbatim} 条（赛题要求可追溯可核验，会降低论据链相似度）")
    return errors


def evaluate_with(matcher, bs, pred_rows, gold_rows, verbose=False):
    gold_by_id = {g["sample_id"]: g for g in gold_rows}
    per_sample = []
    for p in pred_rows:
        g = gold_by_id.get(p["sample_id"])
        if g is None:
            continue
        preds = p.get("issue_list") or []
        golds = g.get("issue_list") or []
        pairs = matcher.match(preds, golds)
        nc, np_, ng = len(pairs), len(preds), len(golds)
        prec = nc / np_ if np_ else 0.0
        rec = nc / ng if ng else 0.0
        f1_ext = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        alphas = []
        for i, j, _ in pairs:
            pt, gt = preds[i], golds[j]
            h = norm_issue(pt["issue_name"]) + "。" + chain_text(pt["argument_chain"]).replace("\n", "。")
            r = norm_issue(gt["issue_name"]) + "。" + chain_text(gt["argument_chain"]).replace("\n", "。")
            alphas.append(bs.score(h, r))
        alpha = sum(alphas) / len(alphas) if alphas else 0.0
        s_ext = f1_ext * (0.7 + 0.3 * alpha)

        n_max = max(np_, ng)
        pf = p.get("future_argument") or []
        gf = g.get("future_argument") or []
        rsum, ssum = 0.0, 0.0
        for i, j, _ in pairs:
            x = future_text(pf[i] if i < len(pf) else "")
            y = future_text(gf[j] if j < len(gf) else "")
            rsum += rouge_l_f1(x, y)
            ssum += bs.score(x, y)
        f1_pred = rsum / n_max if n_max else 0.0
        s_sem = ssum / n_max if n_max else 0.0
        s_pred = 0.6 * f1_pred + 0.4 * s_sem
        rec = {
            "sample_id": p["sample_id"],
            "nc": nc, "n_pred": np_, "n_gold": ng,
            "f1_ext": f1_ext, "alpha": alpha, "s_ext": s_ext,
            "f1_pred": f1_pred, "s_semantic": s_sem, "s_pred": s_pred,
            "score": 0.8 * s_ext + 0.2 * s_pred,
        }
        per_sample.append(rec)
        if verbose:
            print(json.dumps(rec, ensure_ascii=False), file=sys.stderr)

    k = len(per_sample)
    if k == 0:
        return {"samples": 0, "matched_samples": [], "overall": {}}
    mean = lambda key: sum(r[key] for r in per_sample) / k
    overall = {
        "samples": k,
        "S_ext": mean("s_ext"),
        "S_pred": mean("s_pred"),
        "Score": mean("score"),
        "F1_ext": mean("f1_ext"),
        "alpha": mean("alpha"),
        "F1_pred": mean("f1_pred"),
        "S_semantic": mean("s_semantic"),
        "total_nc": sum(r["nc"] for r in per_sample),
        "total_pred": sum(r["n_pred"] for r in per_sample),
        "total_gold": sum(r["n_gold"] for r in per_sample),
    }
    return {"overall": overall, "samples": per_sample}


def evaluate(pred_rows, gold_rows, threshold=0.7, verbose=False):
    return evaluate_with(TripleMatcher(threshold=threshold), BertScorer(),
                         pred_rows, gold_rows, verbose=verbose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--per-sample", action="store_true")
    ap.add_argument("--out", help="明细写入 json 文件")
    args = ap.parse_args()

    pred_rows = read_jsonl(args.pred)
    gold_rows = read_jsonl(args.gold)
    errors = validate(pred_rows, gold_ids=[g["sample_id"] for g in gold_rows],
                      texts={g["sample_id"]: g["docs"][0]["full_text"] for g in gold_rows if g.get("docs")})
    if errors:
        print("格式校验失败:")
        for e in errors[:40]:
            print("  -", e)
        print(f"(共 {len(errors)} 项；无效样本会被官方计分规则剔除，请先修复)")
    res = evaluate(pred_rows, gold_rows, threshold=args.threshold, verbose=args.per_sample)
    print(json.dumps(res["overall"], ensure_ascii=False, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
