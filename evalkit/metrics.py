"""赛题9 官方指标本地复刻：F1ext（三元组语义匹配 + 最大二部图匹配）与 F1pred（ROUGE-L）。

评测口径按 README：
  - 立场严格相等，否则该三元组不参与匹配；
  - 议题名与论据链分别用 BAAI/bge-small-zh-v1.5 取向量余弦相似度，
    按 0.4 / 0.6 加权得三元组语义等价分，>0.7 视为可匹配；
  - 在满足阈值前提下做一对一最大匹配，NC 为匹配数，
    P=NC/Np, R=NC/Ng, F1ext=2PR/(P+R)；
  - F1pred：复用议题匹配关系，对匹配对 (i,j) 计算
    gold.future_argument[i] 与 pred.future_argument[j] 的字符级 ROUGE-L F1，
    求和后除以 N=max(Ng,Np)。
"""

import json
import os
from functools import lru_cache

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BGE_PATH = os.path.join(ROOT, "models", "bge-small-zh-v1.5")
BERT_PATH = os.path.join(ROOT, "models", "bert-base-chinese")

ISSUE_W, ARG_W = 0.4, 0.6
TRIPLE_THRESHOLD = 0.7
# 论据链聚合方式：concat=整链拼成一个向量（官方口径最可能的做法）；
# mean=逐片段编码后取均值向量；space=直接拼接不加分隔
CHAIN_MODE = os.environ.get("CHAIN_MODE", "concat")


class Encoder:
    def __init__(self, model_path=BGE_PATH, batch_size=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts):
        texts = [t if t.strip() else "空" for t in texts]
        vecs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=512, return_tensors="pt")
            out = self.model(**enc)
            last = enc["attention_mask"].sum(1) - 1          # bge 用 last-token pooling
            vec = out.last_hidden_state[torch.arange(len(batch)), last]
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            vecs.append(vec.cpu().numpy())
        return np.concatenate(vecs, axis=0)


def chain_text(chain, mode=CHAIN_MODE):
    if mode == "concat":
        return "；".join(chain)
    if mode in ("space", "mean"):
        return "".join(chain)
    raise ValueError(mode)


def embed_chains(encoder, chains):
    if CHAIN_MODE != "mean":
        return encoder.encode([chain_text(c) for c in chains])
    flat, spans = [], []
    for ch in chains:
        start = len(flat)
        flat.extend(ch if ch else ["空"])
        spans.append((start, len(flat)))
    vecs = encoder.encode(flat)
    out = np.stack([vecs[s:e].mean(0) for s, e in spans])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def align(gold, pred):
    """按 sample_id 对齐，返回 (id, gold_sample, pred_sample) 列表。"""
    pmap = {p["sample_id"]: p for p in pred}
    return [(g["sample_id"], g, pmap[g["sample_id"]]) for g in gold
            if g["sample_id"] in pmap]


def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l_f1(x, y):
    """字符级 ROUGE-L F1。x=预测，y=参考。"""
    lcs = lcs_len(x, y)
    if lcs == 0:
        return 0.0
    recall = lcs / len(y)
    precision = lcs / len(x) if x else 0.0
    return 2 * precision * recall / (precision + recall)


def maximum_matching(score):
    """在 score>threshold 的边上求最大基数匹配（scipy，支持非连通图）。"""
    n_g, n_p = score.shape
    if n_g == 0 or n_p == 0:
        return []
    adj = csr_matrix((score > TRIPLE_THRESHOLD).astype(np.int8))
    matched = maximum_bipartite_matching(adj, perm_type="column")
    return [(int(r), j) for j, r in enumerate(matched) if r >= 0]


def evaluate(gold_path, pred_path, encoder=None, verbose=False):
    encoder = encoder or Encoder()
    gold, pred = load_jsonl(gold_path), load_jsonl(pred_path)
    pairs = align(gold, pred)

    # 1) 编码：每个样本一组向量
    encoded = []
    for _, g, p in pairs:
        g_iss = encoder.encode([x["issue_name"] for x in g["issue_list"]])
        g_arg = embed_chains(encoder, [x["argument_chain"] for x in g["issue_list"]])
        p_iss = encoder.encode([x["issue_name"] for x in p["issue_list"]])
        p_arg = embed_chains(encoder, [x["argument_chain"] for x in p["issue_list"]])
        encoded.append((g_iss, g_arg, p_iss, p_arg))

    f1ext_list, f1pred_list, details = [], [], []
    diag = {"stance_blocked": 0, "near_miss": 0, "unmatched_gold": 0,
            "extra_pred": 0, "best_other": []}

    for enc_idx, (sid, g, p) in enumerate(pairs):
        ng, npred = len(g["issue_list"]), len(p["issue_list"])
        g_iss, g_arg, p_iss, p_arg = encoded[enc_idx]

        g_stance = [x["stance"] for x in g["issue_list"]]
        p_stance = [x["stance"] for x in p["issue_list"]]

        sem = 0.4 * (g_iss @ p_iss.T) + 0.6 * (g_arg @ p_arg.T)
        score = np.where(
            np.array([[gs == ps for ps in p_stance] for gs in g_stance]),
            sem, -1.0)

        matched = maximum_matching(score)
        nc = len(matched)
        precision = nc / npred if npred else 0.0
        recall = nc / ng if ng else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1ext_list.append(f1)

        # 诊断：立场正确但没过阈值的、以及被立场挡掉的高语义分对
        if verbose:
            for i in range(ng):
                for j in range(npred):
                    if sem[i, j] > TRIPLE_THRESHOLD and g_stance[i] != p_stance[j]:
                        diag["stance_blocked"] += 1
                        diag["best_other"].append((sid, i, j, round(float(sem[i, j]), 3),
                                                   g_stance[i], p_stance[j]))
            matched_g = {i for i, _ in matched}
            matched_p = {j for _, j in matched}
            diag["unmatched_gold"] += sum(1 for i in range(ng) if i not in matched_g)
            diag["extra_pred"] += sum(1 for j in range(npred) if j not in matched_p)
            for i in range(ng):
                if i in matched_g:
                    continue
                cand = [j for j in range(npred) if j not in matched_p]
                if cand:
                    diag["near_miss"] += 1

        # 2) F1pred：复用议题匹配关系
        nmax = max(ng, npred)
        total = 0.0
        gf, pf = g.get("future_argument", []), p.get("future_argument", [])
        for i, j in matched:
            if i < len(gf) and j < len(pf):
                total += rouge_l_f1(pf[j], gf[i])
        f1pred_list.append(total / nmax if nmax else 0.0)

        details.append({"sample_id": sid, "f1ext": round(f1, 4),
                        "f1pred": round(total / nmax, 4) if nmax else 0.0,
                        "nc": nc, "ng": ng, "np": npred,
                        "pairs": [(i, j, round(float(score[i, j]), 3)) for i, j in matched]})

    mean_ext = float(np.mean(f1ext_list))
    mean_pred = float(np.mean(f1pred_list))
    result = {
        "samples": len(pairs),
        "F1ext": mean_ext,
        "F1pred": mean_pred,
        "Score": 0.8 * mean_ext + 0.2 * mean_pred,
        "chain_mode": CHAIN_MODE,
        "micro_F1ext": None,
    }
    # micro（全量 NC 累加）作为参照
    tot_nc = sum(d["nc"] for d in details)
    tot_ng = sum(d["ng"] for d in details)
    tot_np = sum(d["np"] for d in details)
    mi = tot_nc / tot_np if tot_np else 0.0
    mr = tot_nc / tot_ng if tot_ng else 0.0
    result["micro_F1ext"] = 2 * mi * mr / (mi + mr) if mi + mr else 0.0
    if verbose:
        result["diag"] = {k: v for k, v in diag.items() if k != "best_other"}
        result["stance_blocked_examples"] = diag["best_other"][:20]
        result["details"] = details
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    res = evaluate(args.gold, args.pred, verbose=args.verbose)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "details"},
                     ensure_ascii=False, indent=1))
