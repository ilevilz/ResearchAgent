"""科研论文知识库：构建、持久化与检索的统一入口。

支持三种入库来源：
    1. 本地 PDF 目录（``data/papers/``）；
    2. 文本条目（如 arXiv 检索返回的 title + abstract）；
    3. 任意 ``ParsedDocument``。

向量写入时为每条元数据附加 ``_idx``（全局自增下标），
以便混合检索把 BM25 的稀疏结果与向量结果对齐到同一份语料。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import settings
from ..utils import get_logger
from .chunker import Chunk, chunk_pages, chunk_text
from .embeddings import Embedder, get_embedder
from .loader import ParsedDocument, load_documents
from .retriever import Retriever, RetrievedChunk, build_context
from .vector_store import BaseVectorStore, create_vector_store, load_vector_store, vector_index_exists

logger = get_logger(__name__)

DEFAULT_INDEX_NAME = "research_kb"


class KnowledgeBase:
    """论文知识库：负责索引生命周期与检索。"""

    def __init__(
        self,
        index_path: str | Path | None = None,
        embedder: Embedder | None = None,
        embedding_backend: str | None = None,
    ) -> None:
        self.index_path = Path(index_path or (settings.index_dir / DEFAULT_INDEX_NAME))
        self.embedder = embedder or get_embedder(embedding_backend)
        self.store: BaseVectorStore | None = None
        self.retriever: Retriever | None = None
        self.documents: list[dict[str, Any]] = []
        self._next_idx = 0

    # ------------------------------------------------------------------ #
    # 索引生命周期
    # ------------------------------------------------------------------ #
    def _ensure_store(self) -> BaseVectorStore:
        if self.store is None:
            self.store = create_vector_store(dim=self.embedder.dim, backend="auto")
        return self.store

    def _ensure_retriever(self) -> Retriever:
        if self.retriever is None:
            self.retriever = Retriever(self._ensure_store(), self.embedder)
        return self.retriever

    # ------------------------------------------------------------------ #
    # 入库
    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """把切分后的块写入向量索引。"""
        if not chunks:
            return 0
        store = self._ensure_store()
        texts = [c.text for c in chunks]
        vectors = self.embedder.encode(texts)

        metadatas: list[dict[str, Any]] = []
        for chunk in chunks:
            meta = {**chunk.metadata, "text": chunk.text, "chunk_id": chunk.chunk_id, "_idx": self._next_idx}
            self._next_idx += 1
            metadatas.append(meta)

        added = store.add(vectors, metadatas)
        if self.retriever is not None:
            self.retriever.invalidate()
        return added

    def add_document(self, doc: ParsedDocument, chunk_size: int | None = None, overlap: int | None = None) -> int:
        """把一个解析后的文档切分并入库。"""
        base_meta = {"source": doc.source, "title": doc.title, **doc.metadata}
        chunks = chunk_pages(doc.pages, chunk_size=chunk_size, overlap=overlap, base_metadata=base_meta)
        added = self.add_chunks(chunks)
        self.documents.append({**doc.summary(), "chunks": len(chunks)})
        logger.info("入库《%s》：%d 块", doc.title, added)
        return added

    def add_text_entry(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> int:
        """直接入库一段文本（如论文摘要）。"""
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, metadata=metadata or {})
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"t{self._next_idx + i:05d}"
        return self.add_chunks(chunks)

    def add_papers(self, papers: Iterable[dict[str, Any]]) -> int:
        """批量入库论文元数据（title + abstract），用于「检索即入库」场景。"""
        total = 0
        for paper in papers:
            title = paper.get("title") or paper.get("paper_title") or "untitled"
            abstract = paper.get("abstract") or paper.get("summary") or ""
            if not abstract.strip():
                continue
            body = f"标题：{title}\n作者：{', '.join(paper.get('authors', []) or [])}\n摘要：{abstract}"
            total += self.add_text_entry(
                body,
                metadata={
                    "source": paper.get("url") or paper.get("source") or "arxiv",
                    "title": title,
                    "paper_id": paper.get("paper_id") or paper.get("id", ""),
                    "published": paper.get("published", ""),
                    "kind": "paper_abstract",
                },
            )
            self.documents.append({"source": paper.get("url", ""), "title": title, "pages": 0, "chunks": 1})
        logger.info("入库论文摘要 %d 条", total)
        return total

    def build_from_papers_dir(self, directory: str | Path | None = None) -> int:
        """从本地 PDF 目录批量建库。"""
        directory = Path(directory or settings.papers_dir)
        docs = load_documents(directory)
        if not docs:
            logger.warning("目录 %s 下没有可解析的 PDF/文本", directory)
            return 0
        return sum(self.add_document(doc) for doc in docs)

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int | None = None, rerank: bool = True) -> list[RetrievedChunk]:
        return self._ensure_retriever().retrieve(query, top_k=top_k, rerank=rerank)

    def search_as_dicts(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.search(query, top_k)]

    def search_text(self, query: str, top_k: int | None = None, char_limit: int = 4000) -> tuple[str, list[dict[str, Any]]]:
        """检索并返回 (拼接后的上下文, 结构化结果)，供 Agent 工具使用。"""
        results = self.search(query, top_k)
        return build_context(results), [r.as_dict() for r in results]

    # ------------------------------------------------------------------ #
    # 持久化与状态
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        if self.store is None:
            logger.warning("索引为空，无需保存")
            return
        self.store.save(self.index_path)
        meta_file = self.index_path.with_suffix(".docs.json")
        meta_file.write_text(
            json.dumps(
                {
                    "documents": self.documents,
                    "next_idx": self._next_idx,
                    "embedder": self.embedder.name,
                    "dim": self.embedder.dim,
                    "backend": self.store.backend,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> bool:
        """从磁盘加载索引；不存在返回 False。"""
        if not vector_index_exists(self.index_path):
            return False
        self.store = load_vector_store(self.index_path)
        meta_file = self.index_path.with_suffix(".docs.json")
        if meta_file.exists():
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            self.documents = data.get("documents", [])
            self._next_idx = data.get("next_idx", len(self.store))
        else:
            self._next_idx = len(self.store)
        self.retriever = None  # 重建，确保 BM25 与最新语料一致
        logger.info("索引已加载：%s（%d 块，backend=%s）", self.index_path.name, len(self.store), self.store.backend)
        return True

    def stats(self) -> dict[str, Any]:
        store = self.store
        return {
            "num_chunks": len(store) if store else 0,
            "num_documents": len(self.documents),
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "vector_backend": store.backend if store else None,
            "index_path": str(self.index_path),
            "hybrid_retrieval": True,
        }


_kb_singleton: KnowledgeBase | None = None


def get_knowledge_base(auto_load: bool = True) -> KnowledgeBase:
    """获取进程内知识库单例（默认尝试加载已建好的索引）。"""
    global _kb_singleton
    if _kb_singleton is None:
        kb = KnowledgeBase()
        if auto_load:
            kb.load()
        _kb_singleton = kb
    return _kb_singleton
