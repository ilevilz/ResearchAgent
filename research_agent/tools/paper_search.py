"""论文检索工具：arXiv + Semantic Scholar 双源检索。

两个数据源互补：
    - arXiv：预印本全文与摘要，覆盖快、无需 Key；
    - Semantic Scholar：补充引用量、发表venue等元数据，便于判断论文影响力。

任一源失败都降级为另一源的结果，保证检索工具本身不会整体失败。
"""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlencode

import httpx

from ..config import settings
from ..utils import get_logger, tokenize
from .base import Tool, ToolError, clean_query, tool

logger = get_logger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_TIMEOUT = 30.0


def _clean_text(text: str | None) -> str:
    return " ".join((text or "").split())


_RETRY_STATUS = {406, 408, 425, 429, 500, 502, 503, 504}

# 每主机最小请求间隔。arXiv 官方建议请求间隔 >= 3s，超频会被前置层以
# 406/429 拒绝；Semantic Scholar 无 Key 时限流更严。
_MIN_INTERVAL = {"export.arxiv.org": 3.0, "api.semanticscholar.org": 1.5}
_DEFAULT_INTERVAL = 0.5
_last_call_at: dict[str, float] = {}


def _throttle(url: str) -> None:
    """按主机限速，保证两次请求之间留够间隔。

    这是本模块最容易被忽略的一环：**重试若不带限速，等于变相加重限流**。
    之前的实现 4 次重试全部落在 ~8s 内，正好卡在对方的封禁窗口里，
    于是重试反而稳定地拿到 406。先限速再重试，成功率才有意义。
    """
    host = httpx.URL(url).host or ""
    interval = _MIN_INTERVAL.get(host, _DEFAULT_INTERVAL)
    elapsed = time.monotonic() - _last_call_at.get(host, 0.0)
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_call_at[host] = time.monotonic()


def _request_with_retry(
    url: str,
    *,
    retries: int = 4,
    base_delay: float = 3.0,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """带限速 + 指数退避的 GET。

    为什么必须有：arXiv / S2 都对突发请求限流。arXiv 超频后会**在源站封禁
    来源 IP**，此后所有新 URL 一律 406。这个现象极具迷惑性——已进 Cloudflare
    边缘缓存的 URL 仍返回 200（响应头 ``cf-cache-status: HIT``、``age`` 数千秒），
    于是「过一会儿又好了」其实是缓存命中，而非解封，很容易误判成随机抖动。
    验证方法：请求一个从未请求过的 URL，若它 406 而老 URL 200，即为源站封禁。

    限速是为了**不触发**封禁；退避是为了已触发时尽快恢复。二者缺一不可。
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _throttle(url)
            with httpx.Client(timeout=_TIMEOUT, follow_redirects=True, headers=headers or {}) as client:
                resp = client.get(url)
            if resp.status_code in _RETRY_STATUS:
                raise httpx.HTTPStatusError(
                    f"{resp.status_code} {resp.reason_phrase}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_error = exc
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning("检索请求失败（%s），%.1fs 后重试 %d/%d", exc, delay, attempt, retries)
                time.sleep(delay)
    raise last_error or RuntimeError("请求失败")


def _build_arxiv_url(search_query: str, **params: Any) -> str:
    """手工拼接 arXiv 查询 URL。

    目的：只对**用户输入部分**做转义，再用 ``replace`` 还原字段语法所需的冒号，
    确保 ``all:`` / ``ti:`` 这类前缀以裸冒号发送（``all%3Arag`` 不是合法的
    arXiv 字段语法）。

    注意：实测 406 与 URL 形态、User-Agent 均无关——四种 UA（含浏览器 UA）
    在同一时刻全部 406，几分钟后又全部 200。**根因是请求频率限流**，
    因此正确的对策是 ``_throttle()`` 限速而非改写 URL。
    """
    parts = [f"search_query={quote_plus(search_query).replace('%3A', ':')}"]
    parts.extend(f"{key}={quote_plus(str(value))}" for key, value in params.items())
    return f"{ARXIV_API}?{'&'.join(parts)}"


# ---------------------------------------------------------------------- #
# arXiv
# ---------------------------------------------------------------------- #
def search_arxiv(query: str, max_results: int = 5, sort_by: str = "relevance") -> list[dict[str, Any]]:
    """调用 arXiv Atom API 检索论文。"""
    sort_map = {"relevance": "relevance", "lastUpdatedDate": "lastUpdatedDate", "submittedDate": "submittedDate"}
    url = _build_arxiv_url(
        f"all:{query}",
        start=0,
        max_results=max(1, min(int(max_results), 30)),
        sortBy=sort_map.get(sort_by, "relevance"),
        sortOrder="descending",
    )
    headers = {"User-Agent": "ResearchAgent/0.1 (+https://github.com/)"}
    resp = _request_with_retry(url, headers=headers)

    root = ET.fromstring(resp.text)
    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        raw_id = _clean_text(entry.findtext("atom:id", default="", namespaces=_ATOM_NS))
        papers.append(
            {
                "paper_id": raw_id.rsplit("/", 1)[-1],
                "title": _clean_text(entry.findtext("atom:title", default="", namespaces=_ATOM_NS)),
                "abstract": _clean_text(entry.findtext("atom:summary", default="", namespaces=_ATOM_NS)),
                "authors": [
                    _clean_text(a.findtext("atom:name", default="", namespaces=_ATOM_NS))
                    for a in entry.findall("atom:author", _ATOM_NS)
                ],
                "published": _clean_text(entry.findtext("atom:published", default="", namespaces=_ATOM_NS)),
                "url": raw_id.replace("http://", "https://"),
                "pdf_url": next(
                    (
                        link.get("href", "")
                        for link in entry.findall("atom:link", _ATOM_NS)
                        if link.get("title") == "pdf"
                    ),
                    "",
                ),
                "source": "arxiv",
            }
        )
    return papers


# ---------------------------------------------------------------------- #
# Semantic Scholar
# ---------------------------------------------------------------------- #
def search_semantic_scholar(query: str, max_results: int = 5, retries: int = 3) -> list[dict[str, Any]]:
    """调用 Semantic Scholar Graph API。

    无 Key 时官方限流极严，实测几乎必返回 429；配置免费 Key 后为 1 req/s，
    是 arXiv 源站封禁 IP 时唯一可靠的在线兜底源。
    仍未成功则由上层降级为离线论文库。
    """
    params = {
        "query": query,
        "limit": max(1, min(int(max_results), 20)),
        "fields": "title,abstract,year,authors,url,venue,citationCount,externalIds",
    }
    url = f"{S2_API}?{urlencode(params)}"
    headers = {"x-api-key": settings.semantic_scholar_api_key} if settings.has_s2_key else None
    payload = _request_with_retry(url, retries=retries, base_delay=2.0, headers=headers).json()

    papers: list[dict[str, Any]] = []
    for item in payload.get("data", []) or []:
        external = item.get("externalIds") or {}
        papers.append(
            {
                "paper_id": external.get("ArXiv") or item.get("paperId", ""),
                "title": _clean_text(item.get("title")),
                "abstract": _clean_text(item.get("abstract")),
                "authors": [a.get("name", "") for a in (item.get("authors") or [])],
                "published": str(item.get("year") or ""),
                "url": item.get("url", ""),
                "venue": item.get("venue", ""),
                "citation_count": item.get("citationCount", 0),
                "source": "semantic_scholar",
            }
        )
    return papers


# ---------------------------------------------------------------------- #
# 融合
# ---------------------------------------------------------------------- #
def _merge(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按标题去重合并，保留信息更丰富的一条。"""
    merged: dict[str, dict[str, Any]] = {}
    for paper in papers:
        key = _clean_text(paper.get("title", "")).lower()[:120]
        if not key:
            continue
        if key in merged:
            existing = merged[key]
            for field, value in paper.items():
                if value and not existing.get(field):
                    existing[field] = value
                elif field == "citation_count" and value:
                    existing[field] = max(existing.get(field, 0), value)
        else:
            merged[key] = dict(paper)
    # 有引用数的排前面，其次按来源顺序
    return sorted(merged.values(), key=lambda p: -(p.get("citation_count") or 0))


def _offline_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """离线兜底检索：从内置论文库按关键词重合度排序。

    存在的意义：现场演示时网络不可控（WAF 拦截 / 断网 / 限流），
    但 Agent 的链路必须能完整跑通。兜底结果会带 ``source=offline_cache`` 标记，
    对使用者诚实可见，不会伪装成实时检索。
    """
    path = Path(__file__).with_name("offline_papers.json")
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    papers = data.get("papers", [])

    query_terms = set(tokenize(query))
    if not query_terms:
        return papers[:max_results]

    scored: list[tuple[float, dict[str, Any]]] = []
    for paper in papers:
        haystack = " ".join(
            [
                str(paper.get("title", "")),
                str(paper.get("abstract", "")),
                " ".join(paper.get("keywords", []) or []),
            ]
        )
        terms = set(tokenize(haystack))
        overlap = len(query_terms & terms) / max(len(query_terms), 1)
        if overlap > 0:
            scored.append((overlap, paper))

    scored.sort(key=lambda item: -item[0])
    return [dict(p) for _, p in scored[:max_results]]


def search_papers_impl(
    query: str,
    max_results: int = 5,
    source: str = "all",
    sort_by: str = "relevance",
    allow_offline_fallback: bool = True,
) -> dict[str, Any]:
    """多源论文检索的实际实现（供工具与 MCP 共用）。"""
    query = clean_query(query)
    if not query:
        raise ToolError("检索关键词为空")

    max_results = max(1, min(int(max_results), 20))
    collected: list[dict[str, Any]] = []
    errors: list[str] = []
    used_sources: list[str] = []

    want_arxiv = source in ("all", "arxiv")
    want_s2 = source in ("all", "semantic_scholar")

    if want_arxiv:
        try:
            collected.extend(search_arxiv(query, max_results, sort_by))
            used_sources.append("arxiv")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"arxiv: {exc}")
            logger.warning("arXiv 检索失败：%s", exc)

    if want_s2:
        try:
            collected.extend(search_semantic_scholar(query, max_results))
            used_sources.append("semantic_scholar")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"semantic_scholar: {exc}")
            logger.warning("Semantic Scholar 检索失败：%s", exc)

    papers = _merge(collected)[:max_results]

    # 全部在线源都失败 -> 降级到离线库，保证 Agent 链路不中断
    degraded = False
    if not papers and allow_offline_fallback:
        papers = _offline_search(query, max_results)
        if papers:
            degraded = True
            used_sources.append("offline_cache")
            logger.warning("在线检索源全部失败，已降级为离线论文库（%d 篇）", len(papers))

    if not papers:
        raise ToolError("所有检索源均失败 -> " + ("; ".join(errors) if errors else "无匹配结果"))

    return {
        "query": query,
        "count": len(papers),
        "sources": used_sources,
        "errors": errors,
        "degraded": degraded,
        "papers": papers,
    }


@tool(
    name="search_papers",
    description=(
        "按关键词检索科研论文，返回标题、作者、摘要、发表时间与链接。"
        "适用于需要查找相关文献、了解某方向研究现状的场景。"
    ),
    parameters={
        "query": {"type": "string", "description": "检索关键词，英文效果最佳，例如 'retrieval augmented generation'"},
        "max_results": {"type": "integer", "description": "返回论文数量，默认 5，最大 20"},
        "source": {
            "type": "string",
            "description": "检索源：all(默认) / arxiv / semantic_scholar",
            "enum": ["all", "arxiv", "semantic_scholar"],
        },
        "sort_by": {
            "type": "string",
            "description": "arXiv 排序方式：relevance(默认) / submittedDate",
            "enum": ["relevance", "submittedDate", "lastUpdatedDate"],
        },
    },
    tags=["search", "arxiv"],
)
def search_papers(query: str, max_results: int = 5, source: str = "all", sort_by: str = "relevance") -> dict[str, Any]:
    """按关键词检索科研论文（arXiv + Semantic Scholar）。"""
    return search_papers_impl(query, max_results, source, sort_by)


@tool(
    name="fetch_paper_detail",
    description="根据 arXiv ID 获取单篇论文的详细信息（标题、摘要、作者、发布时间、PDF 链接）。",
    parameters={
        "paper_id": {"type": "string", "description": "arXiv 论文 ID，例如 '2304.09542' 或 '2304.09542v1'"},
    },
    tags=["search", "arxiv"],
)
def fetch_paper_detail(paper_id: str) -> dict[str, Any]:
    """获取指定 arXiv 论文的详情。"""
    paper_id = clean_query(paper_id).split("/")[-1]
    url = f"{ARXIV_API}?{urlencode({'id_list': paper_id, 'max_results': 1})}"
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", _ATOM_NS)
    if not entries:
        raise ToolError(f"未找到论文：{paper_id}")
    entry = entries[0]
    return {
        "paper_id": paper_id,
        "title": _clean_text(entry.findtext("atom:title", default="", namespaces=_ATOM_NS)),
        "abstract": _clean_text(entry.findtext("atom:summary", default="", namespaces=_ATOM_NS)),
        "authors": [
            _clean_text(a.findtext("atom:name", default="", namespaces=_ATOM_NS))
            for a in entry.findall("atom:author", _ATOM_NS)
        ],
        "published": _clean_text(entry.findtext("atom:published", default="", namespaces=_ATOM_NS)),
        "url": _clean_text(entry.findtext("atom:id", default="", namespaces=_ATOM_NS)),
    }


def build_search_tools() -> list[Tool]:
    return [search_papers, fetch_paper_detail]
