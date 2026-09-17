"""文本切分：递归分隔 + 重叠窗口，保留页码与来源等元信息。

切分策略：
    1. 先按句末标点 / 换行把文本拆成原子句；
    2. 贪心地把句子打包进不超过 ``chunk_size`` 的块；
    3. 相邻块之间保留 ``chunk_overlap`` 字符的重叠，避免答案被切断；
    4. 对超长单句做硬切分兜底。

重叠是 RAG 召回质量的关键：没有重叠时，跨块的答案会在两块的边界处同时丢失。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..config import settings

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


@dataclass
class Chunk:
    """一个可检索的文本块。"""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def as_record(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "text": self.text, "metadata": self.metadata}


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s and s.strip()]


def _tail(text: str, overlap: int) -> str:
    """取尾部 overlap 个字符作为下一块的前缀。"""
    if overlap <= 0 or not text:
        return ""
    return text[-overlap:]


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """把一段文本切成带重叠的块。"""
    chunk_size = chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    overlap = min(overlap, max(chunk_size // 2, 0))

    text = (text or "").strip()
    if not text:
        return []

    sentences: list[str] = []
    for sentence in _split_sentences(text):
        sentences.extend(_hard_split(sentence, chunk_size) if len(sentence) > chunk_size else [sentence])

    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) + 1 > chunk_size:
            chunks.append(buffer.strip())
            buffer = _tail(buffer, overlap) + sentence
            # 重叠后仍超长（长句场景）时再切一刀
            while len(buffer) > chunk_size * 2:
                chunks.append(buffer[:chunk_size].strip())
                buffer = buffer[chunk_size:]
        else:
            buffer = f"{buffer} {sentence}".strip() if buffer else sentence

    if buffer.strip():
        chunks.append(buffer.strip())

    base_meta = dict(metadata or {})
    return [
        Chunk(text=chunk, metadata={**base_meta, "chunk_index": i})
        for i, chunk in enumerate(chunks)
        if chunk.strip()
    ]


def chunk_pages(
    pages: Iterable[Any],
    chunk_size: int | None = None,
    overlap: int | None = None,
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """按页切分（``pages`` 为带 ``.text`` / ``.page_number`` 的对象或字符串）。"""
    result: list[Chunk] = []
    for item in pages:
        if isinstance(item, str):
            text, page_no = item, 0
        else:
            text, page_no = getattr(item, "text", ""), getattr(item, "page_number", 0)

        meta = {**(base_metadata or {}), "page": page_no}
        result.extend(chunk_text(text, chunk_size, overlap, meta))

    # 统一分配全局自增 chunk_id
    for i, chunk in enumerate(result):
        chunk.chunk_id = f"c{i:05d}"
    return result
