"""检索器：稠密向量召回 + BM25 稀疏召回，RRF 融合，可选重排。

为什么做混合检索：
    纯向量召回对"专有名词 / 缩写 / 数字"（如 "mAP 41.2"、模型名 "ResNet-50"）不敏感，
    而 BM25 恰好擅长精确词匹配；两者用 RRF（Reciprocal Rank Fusion）融合，
    无需调参即可稳定优于任一单路召回。

RRF 公式：``score(d) = Σ_r 1 / (k + rank_r(d))``，k 默认 60。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..config import settings
from ..utils import get_logger, tokenize
from .embeddings import Embedder, get_embedder
from .vector_store import BaseVectorStore

logger = get_logger(__name__)

_RRF_K = 60


@dataclass
class RetrievedChunk:
    """一条检索结果。"""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_rank: int | None = None
    sparse_rank: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": round(self.score, 6),
            "metadata": self.metadata,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
        }


class BM25:
    """轻量 BM25（k1=1.5, b=0.75），中文按字、英文按词。"""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(doc) for doc in corpus]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = (sum(self.doc_len) / len(self.docs)) if self.docs else 0.0

        self.tf: list[Counter[str]] = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        out = np.zeros(len(self.docs), dtype=np.float32)
        if not q_terms or not self.docs:
            return out
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i] or 1
            total = 0.0
            for term in q_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                total += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            out[i] = total
        return out


class Retriever:
    """混合检索器：向量召回 + BM25 召回 + RRF 融合 + 可选重排。"""

    def __init__(
        self,
        store: BaseVectorStore,
        embedder: Embedder | None = None,
        top_k: int | None = None,
        hybrid: bool = True,
    ) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()
        self.top_k = top_k or settings.top_k
        self.hybrid = hybrid
        self._bm25: BM25 | None = None

    # ------------------------------------------------------------------ #
    @property
    def bm25(self) -> BM25 | None:
        """惰性构建 BM25（语料变化后需调用 ``invalidate``）。"""
        if not self.hybrid:
            return None
        if self._bm25 is None:
            corpus = [self._text_of(m) for m in self.store._metas]
            if corpus:
                self._bm25 = BM25(corpus)
        return self._bm25

    def invalidate(self) -> None:
        """索引内容变化后清空缓存。"""
        self._bm25 = None

    @staticmethod
    def _text_of(meta: dict[str, Any]) -> str:
        return str(meta.get("text") or meta.get("chunk") or "")

    # ------------------------------------------------------------------ #
    def _dense(self, query: str, top_k: int) -> list[tuple[int, float]]:
        vector = self.embedder.encode([query])
        hits = self.store.search(vector, top_k=top_k)
        return [(h["metadata"].get("_idx", -1), h["score"]) for h in hits]

    def _dense_with_index(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """向量检索并映射回全局下标（用于与 BM25 结果对齐）。"""
        vector = self.embedder.encode([query])
        results = self.store.search(vector, top_k=top_k)
        out: list[tuple[int, float]] = []
        for res in results:
            idx = res["metadata"].get("_idx")
            if idx is None:
                continue
            out.append((int(idx), float(res["score"])))
        return out

    def _sparse(self, query: str, top_k: int) -> list[tuple[int, float]]:
        bm25 = self.bm25
        if bm25 is None or len(bm25.docs) == 0:
            return []
        scores = bm25.scores(query)
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def _rerank(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """轻量重排：对查询与文档的重合度做小幅加权，抑制纯语义漂移。"""
        q_terms = set(tokenize(query))
        if not q_terms:
            return results
        for item in results:
            overlap = len(q_terms & set(tokenize(item.text))) / len(q_terms)
            item.score = round(item.score * (1.0 + 0.15 * overlap), 6)
        return sorted(results, key=lambda x: -x.score)

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int | None = None, rerank: bool = True) -> list[RetrievedChunk]:
        """执行检索，返回 Top-K 结果。"""
        top_k = top_k or self.top_k
        if len(self.store) == 0:
            return []

        candidates = max(top_k * 4, top_k)
        dense = self._dense_with_index(query, candidates)
        sparse = self._sparse(query, candidates)

        fused: dict[int, RetrievedChunk] = {}

        for rank, (idx, _score) in enumerate(dense):
            meta = dict(self.store._metas[idx])
            meta.pop("_idx", None)
            fused[idx] = RetrievedChunk(
                text=self._text_of(meta), score=1.0 / (_RRF_K + rank + 1), metadata=meta, dense_rank=rank
            )

        for rank, (idx, _score) in enumerate(sparse):
            contribution = 1.0 / (_RRF_K + rank + 1)
            if idx in fused:
                fused[idx].score += contribution
                fused[idx].sparse_rank = rank
            else:
                meta = dict(self.store._metas[idx])
                meta.pop("_idx", None)
                fused[idx] = RetrievedChunk(
                    text=self._text_of(meta), score=contribution, metadata=meta, sparse_rank=rank
                )

        results = sorted(fused.values(), key=lambda x: -x.score)[: top_k * 2]
        if rerank:
            results = self._rerank(query, results)
        return results[:top_k]

    def retrieve_as_dicts(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.retrieve(query, top_k)]


_QUERY_STOPWORDS = re.compile(r"^(的|了|是|和|与|在|对|请|帮|我|你)$")


def build_context(results: Sequence[RetrievedChunk], limit: int = 4000) -> str:
    """把检索结果拼成 LLM 可读的上下文（带编号引用，便于溯源）。"""
    parts: list[str] = []
    used = 0
    for i, item in enumerate(results, start=1):
        title = item.metadata.get("title") or item.metadata.get("paper_title") or item.metadata.get("source", "unknown")
        page = item.metadata.get("page")
        header = f"[{i}] {title}" + (f" (p.{page})" if page else "")
        block = f"{header}\n{item.text.strip()}"
        if used + len(block) > limit:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
