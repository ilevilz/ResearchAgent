"""向量索引：FAISS 优先，NumPy 精确检索兜底，对外接口完全一致。

生产环境使用 FAISS ``IndexFlatIP``（向量已归一化，内积等价余弦相似度）；
当 faiss 无可用 wheel（例如 Python 3.14 目前）时自动降级为 NumPy 矩阵乘法做
精确检索 —— 两者在返回结构与排序语义上完全一致，上层检索器无感知。

索引持久化：``<name>.vecs.npy`` 存向量、``<name>.meta.json`` 存元数据，
避免依赖 pickle，跨版本可读。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


def _try_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:  # noqa: BLE001
        return None


class BaseVectorStore:
    """向量索引基类。"""

    backend = "base"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._metas: list[dict[str, Any]] = []

    # ---- 子类实现 ----
    def _add_vectors(self, vectors: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def _search_vectors(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    # ---- 公共 API ----
    def __len__(self) -> int:
        return len(self._metas)

    def add(self, vectors: np.ndarray, metadatas: Sequence[dict[str, Any]]) -> int:
        """写入向量与元数据，返回写入条数。"""
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(metadatas):
            raise ValueError(f"向量数({vectors.shape[0]})与元数据数({len(metadatas)})不一致")
        if vectors.shape[0] == 0:
            return 0
        self._add_vectors(vectors)
        self._metas.extend(dict(m) for m in metadatas)
        return vectors.shape[0]

    def search(self, query: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Top-K 相似检索，返回 [{score, metadata, rank}]。"""
        if len(self._metas) == 0:
            return []
        query = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        top_k = min(top_k, len(self._metas))
        scores, indices = self._search_vectors(query, top_k)

        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self._metas):
                continue
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "metadata": self._metas[int(idx)],
                }
            )
        return results

    # ---- 持久化 ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path.with_suffix(".vecs.npy")), self._all_vectors())
        path.with_suffix(".meta.json").write_text(
            json.dumps(self._metas, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("索引已保存：%s（%d 条，backend=%s）", path, len(self._metas), self.backend)

    @classmethod
    def load(cls, path: str | Path) -> "BaseVectorStore":
        path = Path(path)
        vectors = np.load(str(path.with_suffix(".vecs.npy")))
        metas = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        store = create_vector_store(dim=int(vectors.shape[1]), backend="auto")
        if len(metas):
            store.add(vectors, metas)
        return store

    def _all_vectors(self) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class NumpyVectorStore(BaseVectorStore):
    """NumPy 精确检索（暴力内积），无任何外部依赖。"""

    backend = "numpy"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._matrix: np.ndarray | None = None

    def _add_vectors(self, vectors: np.ndarray) -> None:
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def _search_vectors(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._matrix is not None
        scores = self._matrix @ query[0]  # 归一化向量 -> 余弦相似度
        order = np.argsort(-scores)[:top_k]
        return scores[order].reshape(1, -1), order.reshape(1, -1)

    def _all_vectors(self) -> np.ndarray:
        return self._matrix if self._matrix is not None else np.zeros((0, self.dim), dtype=np.float32)


class FaissVectorStore(BaseVectorStore):
    """FAISS IndexFlatIP 精确检索。"""

    backend = "faiss"

    def __init__(self, dim: int, faiss_module: Any) -> None:
        super().__init__(dim)
        self._faiss = faiss_module
        self._index = faiss_module.IndexFlatIP(dim)

    def _add_vectors(self, vectors: np.ndarray) -> None:
        self._index.add(vectors)

    def _search_vectors(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._index.search(query, top_k)

    def _all_vectors(self) -> np.ndarray:
        return self._index.reconstruct_n(0, self._index.ntotal)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        path.with_suffix(".meta.json").write_text(
            json.dumps(self._metas, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("FAISS 索引已保存：%s（%d 条）", path, len(self._metas))

    @classmethod
    def load(cls, path: str | Path) -> "FaissVectorStore":  # pragma: no cover - 需 faiss 环境
        path = Path(path)
        faiss_module = _try_import_faiss()
        if faiss_module is None:
            raise RuntimeError("faiss 不可用，无法加载 FAISS 索引")
        index = faiss_module.read_index(str(path.with_suffix(".faiss")))
        metas = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        store = cls(dim=index.d, faiss_module=faiss_module)
        store._index = index
        store._metas = metas
        return store


def create_vector_store(dim: int, backend: str = "auto") -> BaseVectorStore:
    """工厂：后端可选 ``auto`` / ``faiss`` / ``numpy``。"""
    want = (backend or "auto").strip().lower()
    if want == "numpy":
        return NumpyVectorStore(dim)

    faiss_module = _try_import_faiss()
    if faiss_module is not None:
        return FaissVectorStore(dim, faiss_module)

    if want == "faiss":
        raise RuntimeError("请求 FAISS 后端，但当前环境未安装 faiss-cpu")
    logger.info("faiss 不可用，向量索引降级为 NumPy 精确检索")
    return NumpyVectorStore(dim)


def vector_index_exists(path: str | Path) -> bool:
    """检查磁盘上是否存在可加载的索引。"""
    path = Path(path)
    return path.with_suffix(".vecs.npy").exists() or path.with_suffix(".faiss").exists()


def load_vector_store(path: str | Path) -> BaseVectorStore:
    """按磁盘实际存在格式加载索引。"""
    path = Path(path)
    if path.with_suffix(".faiss").exists() and _try_import_faiss() is not None:
        return FaissVectorStore.load(path)
    return BaseVectorStore.load(path)
