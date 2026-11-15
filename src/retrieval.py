"""检索增强：用 bge 从 train.jsonl 里找出与当前文本最相似的人工标注样本，作为 few-shot 范例。

目的：让基座模型照着"人类标注的议题命名粒度"来起名，而不是自己发明上位大词。
训练集文档嵌入已缓存在 models/train_text_emb.npy（与 train.jsonl 行序一一对应）。

用法:
    from retrieval import Retriever, extract_example, future_example
    r = Retriever()
    for row, sim in r.neighbors(text, doc_type, k=2):
        ...
"""

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metric import BGE_NAME, MODEL_CACHE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data", "train.jsonl")
TRAIN_EMB = os.path.join(MODEL_CACHE, "train_text_emb.npy")


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def doc_of(row):
    return row["docs"][0]


class Retriever:
    def __init__(self, train_path=TRAIN, emb_path=TRAIN_EMB):
        self.rows = read_jsonl(train_path)
        self.emb_path = emb_path
        self._enc = None
        self._emb = None

    def _load(self):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if self._enc is None:
            self._enc = SentenceTransformer(BGE_NAME, cache_folder=MODEL_CACHE, device="cpu")
        if self._emb is None:
            if os.path.exists(self.emb_path):
                emb = np.load(self.emb_path)
                if emb.shape[0] != len(self.rows):
                    emb = None
            else:
                emb = None
            if emb is None:
                texts = [doc_of(r)["full_text"] for r in self.rows]
                emb = self._enc.encode(texts, batch_size=32, normalize_embeddings=True,
                                       show_progress_bar=True).astype("float32")
                np.save(self.emb_path, emb)
            self._emb = emb
        return self._enc, self._emb

    def encode(self, text):
        enc, _ = self._load()
        return enc.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]

    def warmup(self):
        """在主线程预先加载模型与嵌入，避免线程池里的初始化竞态。"""
        self._load()

    def neighbors(self, text, doc_type=None, k=3, pool=50, prefer_doc_type=True, min_sim=0.0):
        """返回 [(train_row, cos_sim)]。同一 doc_type 优先，相似度降序，低于 min_sim 的丢弃。"""
        import numpy as np

        _, emb = self._load()
        q = self.encode(text)
        sims = emb @ q
        idx = list(np.argsort(-sims)[:max(pool, k)])
        if prefer_doc_type and doc_type:
            same = [i for i in idx if doc_of(self.rows[i])["doc_type"] == doc_type]
            other = [i for i in idx if i not in set(same)]
            idx = same + other
        out = [(self.rows[i], float(sims[i])) for i in idx[:k]]
        return [(r, s) for r, s in out if s >= min_sim]


def extract_example(row, max_issues=5, excerpt=50, include_text=0):
    """把训练样本压成"抽取"范例：议题名 + 立场 + 证据摘录（仅示意风格，正式输出用编号）。

    include_text>0 时，额外附上范例原文的前若干字（text_excerpt），
    让模型看到"这段文本 → 这些议题名"的映射，而非孤立的命名，增强可学习性。
    """
    ex = {"issue_list": [
        {"issue_name": it["issue_name"], "stance": it["stance"],
         "evidence_ids_hint": [str(x)[:excerpt] for x in it.get("argument_chain", [])]}
        for it in row.get("issue_list", [])[:max_issues]]}
    if include_text:
        try:
            ex = {"text_excerpt": doc_of(row)["full_text"][:include_text], **ex}
        except (KeyError, IndexError, TypeError):
            pass
    return ex


def future_example(row, max_issues=5):
    """把训练样本压成"未来论点"范例：议题链 + 对应论断。"""
    issues = [{"issue_name": it["issue_name"], "stance": it["stance"]}
              for it in row.get("issue_list", [])[:max_issues]]
    future = [str(x) for x in row.get("future_argument", [])[:max_issues]]
    return {"issue_list": issues, "future_argument": future}
