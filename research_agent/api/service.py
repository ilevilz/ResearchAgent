"""服务层：把 Agent / RAG / 工具封装成与传输协议无关的用例接口。

为什么要多这一层：
    路由函数只做参数校验与序列化，真正的业务逻辑在这里。
    这样同一套逻辑可以被 FastAPI、CLI、评测脚本复用，
    也让 API 层可以脱离 HTTP 单独做单元测试。
"""
from __future__ import annotations

from typing import Any, Sequence

from .. import __version__
from ..agent.react_agent import AgentResult, ReActAgent
from ..config import settings
from ..core.llm_client import get_llm_client
from ..memory.session import get_memory_store
from ..rag.knowledge_base import get_knowledge_base
from ..tools.registry import get_default_registry
from ..utils import get_logger

logger = get_logger(__name__)


class AgentService:
    """Agent 服务门面。"""

    def __init__(self) -> None:
        self.registry = get_default_registry()
        self.memory_store = get_memory_store()

    # ------------------------------------------------------------------ #
    def chat(
        self,
        query: str,
        session_id: str | None = None,
        max_steps: int | None = None,
        reset_memory: bool = False,
    ) -> AgentResult:
        """执行一次 Agent 任务。"""
        memory = self.memory_store.get(session_id) if session_id else None
        agent = ReActAgent(
            registry=self.registry,
            max_steps=max_steps or settings.max_steps,
            memory=memory,
            verbose=False,
        )
        return agent.run(query, session_id=session_id, reset_memory=reset_memory)

    # ------------------------------------------------------------------ #
    def index(
        self,
        papers: list[dict[str, Any]] | None = None,
        pdf_paths: Sequence[str] | None = None,
        source_query: str | None = None,
        max_results: int = 5,
        save: bool = True,
    ) -> dict[str, Any]:
        """向知识库写入内容。"""
        from ..tools.kb_tools import index_papers_impl
        from ..tools.paper_search import search_papers_impl

        collected: list[dict[str, Any]] = list(papers or [])

        if source_query:
            found = search_papers_impl(source_query, max_results=max_results)
            collected.extend(found.get("papers", []))

        added = 0
        kb = get_knowledge_base()
        if collected:
            added += index_papers_impl(papers=collected, save=False)["added_chunks"]
        for path in pdf_paths or []:
            added += index_papers_impl(pdf_path=path, save=False)["added_chunks"]

        if added and save:
            kb.save()

        return {"added_chunks": added, "index_stats": kb.stats()}

    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """直接做知识库检索（不经过 Agent）。"""
        return get_knowledge_base().search_as_dicts(query, top_k=top_k)

    # ------------------------------------------------------------------ #
    def health(self, deep: bool = False) -> dict[str, Any]:
        """健康检查。``deep=True`` 时真实调用一次 LLM。"""
        kb = get_knowledge_base()
        payload: dict[str, Any] = {
            "status": "ok",
            "llm_configured": settings.has_llm_key,
            "llm": {
                "model": settings.deepseek_model,
                "base_url": settings.deepseek_base_url,
            },
            "knowledge_base": kb.stats(),
            "tools": self.registry.names(),
            "version": __version__,
        }

        if deep:
            check = get_llm_client().health_check()
            payload["llm"].update(check)
            if not check.get("ok"):
                payload["status"] = "degraded"
        elif not settings.has_llm_key:
            payload["status"] = "degraded"
            payload["llm"]["hint"] = "未配置 DEEPSEEK_API_KEY，Agent 无法执行"

        return payload

    def list_tools(self) -> list[dict[str, str]]:
        return self.registry.describe()

    def list_sessions(self) -> list[str]:
        return self.memory_store.all_sessions()

    def reset_session(self, session_id: str) -> None:
        self.memory_store.drop(session_id)


_service: AgentService | None = None


def get_service() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService()
    return _service
