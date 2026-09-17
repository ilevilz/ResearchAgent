"""PDF 工具：下载论文 PDF 并解析为结构化文本。

Agent 的典型链路：``search_papers`` 找到论文 -> ``download_pdf`` 取全文 ->
``parse_pdf`` 提取正文 -> 交给 LLM 做关键信息提取或指标抽取。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from ..config import settings
from ..rag.loader import parse_pdf as parse_pdf_file
from ..utils import get_logger, slugify
from .base import Tool, ToolError, tool

logger = get_logger(__name__)

_TIMEOUT = 90.0
_MAX_PDF_BYTES = 40 * 1024 * 1024


def download_pdf_impl(url: str, filename: str | None = None) -> dict[str, Any]:
    """下载 PDF 到 ``data/papers/``，返回本地路径与大小。"""
    if not url:
        raise ToolError("URL 为空")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError(f"不支持的 URL 协议：{parsed.scheme}")

    settings.ensure_dirs()
    name = filename or f"{slugify(Path(parsed.path).stem) or 'paper'}.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    target = settings.papers_dir / name

    if target.exists() and target.stat().st_size > 0:
        return {"path": str(target), "size_bytes": target.stat().st_size, "cached": True}

    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                logger.warning("Content-Type 非 PDF：%s", content_type)
            total = 0
            with target.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    total += len(chunk)
                    if total > _MAX_PDF_BYTES:
                        fh.close()
                        target.unlink(missing_ok=True)
                        raise ToolError(f"PDF 超过 {_MAX_PDF_BYTES // 1024 // 1024}MB 限制")

    return {"path": str(target), "size_bytes": target.stat().st_size, "cached": False}


def parse_pdf_impl(path: str, max_pages: int = 0, max_chars: int = 12000) -> dict[str, Any]:
    """解析本地 PDF，返回按页组织的纯文本。"""
    doc = parse_pdf_file(path)
    pages = doc.pages[: max_pages or None]

    collected: list[dict[str, Any]] = []
    used = 0
    for page in pages:
        text = page.text
        if used + len(text) > max_chars:
            text = text[: max(max_chars - used, 0)]
        if text:
            collected.append({"page": page.page_number, "text": text})
            used += len(text)
        if used >= max_chars:
            break

    return {
        "path": doc.source,
        "title": doc.title,
        "num_pages": doc.num_pages,
        "parser": doc.parser,
        "returned_pages": len(collected),
        "truncated": used >= max_chars,
        "pages": collected,
    }


@tool(
    name="download_pdf",
    description="根据论文 PDF 链接下载文件到本地，返回本地路径。已有缓存时直接复用，不重复下载。",
    parameters={
        "url": {"type": "string", "description": "PDF 直链，通常来自 search_papers 返回的 pdf_url"},
        "filename": {"type": "string", "description": "可选，保存的文件名（不含路径）"},
    },
    tags=["pdf"],
)
def download_pdf(url: str, filename: str | None = None) -> dict[str, Any]:
    """下载论文 PDF。"""
    return download_pdf_impl(url, filename)


@tool(
    name="parse_pdf",
    description=(
        "解析本地 PDF 文件，提取按页的纯文本内容。"
        "适用于阅读论文全文、提取方法/实验/指标等详细信息的场景。"
    ),
    parameters={
        "path": {"type": "string", "description": "本地 PDF 路径，可由 download_pdf 返回"},
        "max_pages": {"type": "integer", "description": "最多解析页数，0 表示全部（默认 0）"},
        "max_chars": {"type": "integer", "description": "返回文本的最大字符数，默认 12000，防止超长"},
    },
    tags=["pdf"],
)
def parse_pdf(path: str, max_pages: int = 0, max_chars: int = 12000) -> dict[str, Any]:
    """解析本地 PDF 为文本。"""
    return parse_pdf_impl(path, max_pages, max_chars)


def build_pdf_tools() -> list[Tool]:
    return [download_pdf, parse_pdf]
