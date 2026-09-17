"""知识库工具：把论文写入 RAG 索引，以及基于外部知识做检索问答。

这两个工具是 RAG 与 Agent 的接合点：
    - ``index_papers``        —— 把检索到的论文摘要 / 本地 PDF 灌入向量库；
    - ``search_knowledge_base`` —— 混合检索（向量 + BM25 + RRF）并返回可溯源的片段。
"""
from __future__ import annotations

from typing import Any

from ..rag.knowledge_base import get_knowledge_base
from ..rag.loader import parse_pdf as parse_pdf_file
from ..utils import get_logger
from .base import Tool, ToolError, clean_query, tool

logger = get_logger(__name__)


def index_papers_impl(papers: list[dict[str, Any]] | None = None, pdf_path: str | None = None, save: bool = True) -> dict[str, Any]:
    """把论文摘要或本地 PDF 写入知识库。"""
    kb = get_knowledge_base()
    added = 0

    if papers:
        added += kb.add_papers(papers)

    if pdf_path:
        doc = parse_pdf_file(pdf_path)
        added += kb.add_document(doc)

    if added == 0:
        raise ToolError("没有可入库的内容：请提供 papers 或 pdf_path")

    if save:
        kb.save()

    stats = kb.stats()
    return {"added_chunks": added, "index_stats": stats}


def search_kb_impl(query: str, top_k: int = 5) -> dict[str, Any]:
    """在知识库中做混合检索。"""
    query = clean_query(query)
    if not query:
        raise ToolError("查询为空")

    kb = get_knowledge_base()
    if not kb.stats()["num_chunks"]:
        raise ToolError("知识库为空，请先调用 index_papers 建库，或运行 `python scripts/build_kb.py`")

    context, results = kb.search_text(query, top_k=top_k)
    return {
        "query": query,
        "top_k": top_k,
        "num_results": len(results),
        "context": context,
        "results": [
            {
                "rank": r["rank"],
                "score": round(r["score"], 5),
                "title": r["metadata"].get("title", ""),
                "source": r["metadata"].get("source", ""),
                "page": r["metadata"].get("page"),
                "snippet": r["text"][:400],
            }
            for r in results
        ],
    }


@tool(
    name="index_papers",
    description=(
        "把论文写入 RAG 知识库（向量索引）。可传入 search_papers 返回的论文列表，"
        "或本地 PDF 路径。入库后即可用 search_knowledge_base 做基于外部知识的问答。"
    ),
    parameters={
        "papers": {
            "type": "array",
            "description": "论文对象列表，元素需包含 title 与 abstract 字段",
            "items": {"type": "object"},
        },
        "pdf_path": {"type": "string", "description": "可选，本地 PDF 路径"},
        "save": {"type": "boolean", "description": "是否持久化到磁盘，默认 true"},
    },
    tags=["rag", "index"],
)
def index_papers(papers: list[dict[str, Any]] | None = None, pdf_path: str | None = None, save: bool = True) -> dict[str, Any]:
    """把论文写入知识库。"""
    return index_papers_impl(papers, pdf_path, save)


@tool(
    name="search_knowledge_base",
    description=(
        "在科研论文知识库中做混合检索（向量召回 + BM25 + RRF 融合），返回最相关的原文片段及来源。"
        "适用于回答需要引用论文原文细节的问题，例如具体方法、数据集、实验指标数值。"
    ),
    parameters={
        "query": {"type": "string", "description": "检索问题或关键词"},
        "top_k": {"type": "integer", "description": "返回片段数量，默认 5"},
    },
    tags=["rag", "retrieve"],
)
def search_knowledge_base(query: str, top_k: int = 5) -> dict[str, Any]:
    """基于知识库检索论文片段。"""
    return search_kb_impl(query, top_k)


def build_kb_tools() -> list[Tool]:
    return [index_papers, search_knowledge_base]
