"""Embedding 后端：语义模型优先，哈希向量兜底。

设计动机：科研问答既有中文也有英文，理想方案是 BGE / bge-m3 这类语义模型。
但语义模型依赖 torch（体积大、需联网下载），在受限环境下会直接让整条 RAG 链路不可用。
因此这里提供两级实现，接口完全一致（``encode`` + ``dim`` + ``name``）：

    1. ``SentenceTransformerEmbedder`` —— 语义向量，检索质量高（生产推荐）
    2. ``HashingEmbedder``            —— 哈希 TF 向量，零依赖，保证链路可跑通

``EMBEDDING_BACKEND=auto`` 时自动探测并降级，并打印实际生效的后端，
便于在日志 / /health 中如实反映系统状态。
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..config import settings
from ..utils import get_logger, tokenize

logger = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Embedding 后端协议。"""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        ...


class HashingEmbedder:
    """零依赖哈希向量：unigram + bigram 特征，sublinear TF，L2 归一化。

    对中英文都有效（中文按字、英文按词），配合余弦相似度可做基础语义召回。
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @staticmethod
    def _features(text: str) -> list[str]:
        toks = tokenize(text)
        # bigram 弥补单字/单词丢失的语序信息
        return toks + [f"{a}\x1f{b}" for a, b in zip(toks, toks[1:])]

    def _bucket(self, feat: str) -> int:
        digest = hashlib.md5(feat.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            for feat in self._features(text or ""):
                bucket = self._bucket(feat)
                counts[bucket] = counts.get(bucket, 0) + 1
            for bucket, count in counts.items():
                out[row, bucket] = 1.0 + math.log(count)
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:
    """基于 sentence-transformers 的语义 Embedding（BGE / m3 等）。"""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


_embedder: Embedder | None = None


def get_embedder(backend: str | None = None, force_reload: bool = False) -> Embedder:
    """获取进程内单例 Embedder（自动降级）。"""
    global _embedder
    if _embedder is not None and not force_reload:
        return _embedder

    want = (backend or settings.embedding_backend or "auto").strip().lower()

    if want in {"hash", "hashing"}:
        resolved: Embedder = HashingEmbedder()
    elif want in {"st", "sentence-transformers", "sentence_transformers"}:
        resolved = SentenceTransformerEmbedder(settings.embedding_model)
    else:  # auto
        try:
            resolved = SentenceTransformerEmbedder(settings.embedding_model)
        except Exception as exc:  # noqa: BLE001 - 任何原因都降级，不让链路断掉
            logger.warning("语义 Embedding 不可用（%s），已降级为哈希向量", exc)
            resolved = HashingEmbedder()

    logger.info("Embedding 后端生效：%s (dim=%d)", resolved.name, resolved.dim)
    _embedder = resolved
    return resolved
