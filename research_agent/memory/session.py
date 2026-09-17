"""会话记忆：短期对话历史 + 工作记忆（任务中沉淀的事实）。

分两层，解决不同问题：
    - **对话历史**：保留最近 N 轮 user/assistant 消息，支持多轮追问（"那它的数据集是什么？"）；
    - **工作记忆**：把本次任务中调用过的工具、命中的论文、抽取到的指标沉淀下来，
      既可以在多轮之间复用（避免重复检索），也能作为最终报告的素材。

为避免上下文无限增长，历史按轮数截断，工作记忆按条目数上限淘汰。
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..utils import get_logger, truncate

logger = get_logger(__name__)

MAX_TURNS = 8
MAX_FACTS = 40
MAX_PAPERS = 20


@dataclass
class Turn:
    """一轮对话。"""

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionMemory:
    """单个会话的记忆。"""

    def __init__(self, session_id: str | None = None, max_turns: int = MAX_TURNS) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.max_turns = max_turns
        self.turns: list[Turn] = []
        # 工作记忆
        self.papers: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []
        self.facts: list[str] = []
        self.tools_used: list[str] = []

    # ------------------------------------------------------------------ #
    # 对话历史
    # ------------------------------------------------------------------ #
    def add_user(self, content: str) -> None:
        self.turns.append(Turn(role="user", content=content))

    def add_assistant(self, content: str, **metadata: Any) -> None:
        self.turns.append(Turn(role="assistant", content=content, metadata=metadata))

    def recent_messages(self, limit: int | None = None) -> list[dict[str, str]]:
        """最近若干轮，转成 LLM 可用的 messages 片段。"""
        turns = self.turns[-(limit or self.max_turns * 2) :]
        return [{"role": t.role, "content": truncate(t.content, 2000)} for t in turns if t.content.strip()]

    # ------------------------------------------------------------------ #
    # 工作记忆
    # ------------------------------------------------------------------ #
    def record_tool_call(self, name: str, ok: bool = True) -> None:
        self.tools_used.append(name)

    def record_papers(self, papers: list[dict[str, Any]]) -> None:
        """沉淀命中的论文（按标题去重）。"""
        known = {p.get("title", "").lower() for p in self.papers}
        for paper in papers or []:
            title = (paper.get("title") or "").strip()
            if title and title.lower() not in known:
                self.papers.append({"title": title, "url": paper.get("url", ""), "year": paper.get("published", "")})
                known.add(title.lower())
        self.papers = self.papers[-MAX_PAPERS:]

    def record_metrics(self, metrics: list[dict[str, Any]]) -> None:
        self.metrics.extend(metrics or [])
        self.metrics = self.metrics[-MAX_FACTS:]

    def add_fact(self, fact: str) -> None:
        fact = (fact or "").strip()
        if fact and fact not in self.facts:
            self.facts.append(truncate(fact, 300))
            self.facts = self.facts[-MAX_FACTS:]

    def working_context(self, limit: int = 1500) -> str:
        """把工作记忆压成一段简短上下文，注入到下一轮。"""
        parts: list[str] = []
        if self.papers:
            titles = "; ".join(p["title"][:80] for p in self.papers[-5:])
            parts.append(f"已检索到的论文：{titles}")
        if self.metrics:
            parts.append(f"已抽取实验指标 {len(self.metrics)} 条")
        if self.facts:
            parts.append("已确认事实：" + "；".join(self.facts[-5:]))
        return truncate("\n".join(parts), limit) if parts else ""

    def reset(self) -> None:
        self.turns.clear()
        self.papers.clear()
        self.metrics.clear()
        self.facts.clear()
        self.tools_used.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": len(self.turns),
            "papers": len(self.papers),
            "metrics": len(self.metrics),
            "tools_called": len(self.tools_used),
            "unique_tools": sorted(set(self.tools_used)),
        }


class MemoryStore:
    """多会话记忆容器（线程安全）。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str | None = None) -> SessionMemory:
        if not session_id:
            return SessionMemory()
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionMemory(session_id)
            return self._sessions[session_id]

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def all_sessions(self) -> list[str]:
        return list(self._sessions)


_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store
