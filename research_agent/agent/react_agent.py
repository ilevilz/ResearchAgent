"""ReAct Agent：Task Planning -> Tool Calling -> Observation 循环。

实现说明（为什么用原生 Function Calling 而不是解析文本）：
    ReAct 的经典做法是让 LLM 输出 "Thought/Action/Action Input" 文本再正则解析，
    但这在生产里很脆：格式漂移、参数缺失、中文标点都会导致解析失败。
    这里保留 ReAct 的**推理-行动交替**结构，但把 Action 的选取交给 DeepSeek 的
    原生 Function Calling —— 由 API 保证参数是合法 JSON，可靠性显著更高，
    同时我们仍然完整记录 Thought / Action / Observation 轨迹，可解释性不打折。

收敛策略（三道保险）：
    1. LLM 主动停止调用工具 -> 直接产出最终回答；
    2. 达到 ``max_steps`` -> 用 FINAL_ANSWER_PROMPT 强制其基于已有 Observation 作答；
    3. 工具全部失败 -> 在最终回答里如实说明，而不是编造。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import settings
from ..core.llm_client import LLMError, get_llm_client
from ..memory.session import SessionMemory, get_memory_store
from ..tools.registry import ToolRegistry, get_default_registry
from ..utils import get_logger, timer, truncate
from .prompts import FINAL_ANSWER_PROMPT, REACT_SYSTEM_PROMPT

logger = get_logger(__name__)

StepCallback = Callable[["AgentStep"], None]


@dataclass
class AgentStep:
    """ReAct 的一步：Thought + Action + Observation。"""

    index: int
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    ok: bool = True
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
        }


@dataclass
class AgentResult:
    """一次完整任务的执行结果。"""

    query: str
    answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    success: bool = True
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    session_id: str = ""
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tools_used(self) -> list[str]:
        return [s.action for s in self.steps if s.action]

    def trace(self) -> list[dict[str, Any]]:
        return [s.as_dict() for s in self.steps]

    def as_dict(self, include_trace: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "answer": self.answer,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "steps": len(self.steps),
            "tools_used": self.tools_used,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "session_id": self.session_id,
            "error": self.error,
        }
        if include_trace:
            payload["trace"] = self.trace()
        return payload


class ReActAgent:
    """基于 ReAct 范式 + Function Calling 的科研分析 Agent。"""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        llm: Any = None,
        max_steps: int | None = None,
        memory: SessionMemory | None = None,
        verbose: bool = True,
        on_step: StepCallback | None = None,
    ) -> None:
        self.registry = registry or get_default_registry()
        self.llm = llm or get_llm_client()
        self.max_steps = max_steps or settings.max_steps
        self.memory = memory or SessionMemory()
        self.verbose = verbose
        self.on_step = on_step

    # ------------------------------------------------------------------ #
    # 内部工具方法
    # ------------------------------------------------------------------ #
    def _emit(self, step: AgentStep) -> None:
        if self.verbose:
            logger.info(
                "Step %d | Thought: %s | Action: %s(%s) | Obs: %s",
                step.index,
                truncate(step.thought, 120) or "-",
                step.action or "-",
                truncate(str(step.action_input), 100),
                truncate(step.observation, 160),
            )
        if self.on_step is not None:
            try:
                self.on_step(step)
            except Exception:  # noqa: BLE001 - 回调异常不应影响主流程
                logger.exception("on_step 回调异常")

    def _record_tool_output(self, action: str, payload: Any) -> None:
        """把工具结果沉淀进工作记忆，供后续步骤与最终报告复用。"""
        if not isinstance(payload, dict):
            return
        if action == "search_papers" and payload.get("papers"):
            self.memory.record_papers(payload["papers"])
        elif action == "extract_metrics" and payload.get("metrics"):
            self.memory.record_metrics(payload["metrics"])

    def _execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        result = self.registry.call(name, arguments)
        self.memory.record_tool_call(name, result.ok)
        if result.ok:
            self._record_tool_output(name, result.data)
        return result.to_observation(), result.ok

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def run(self, query: str, session_id: str | None = None, reset_memory: bool = False) -> AgentResult:
        """执行一次完整任务。"""
        if reset_memory:
            self.memory.reset()
        if session_id:
            self.memory = get_memory_store().get(session_id)

        started = time.perf_counter()
        result = AgentResult(query=query, session_id=self.memory.session_id)

        # ---- 组装初始消息 ----
        messages: list[dict[str, Any]] = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
        messages.extend(self.memory.recent_messages())

        working = self.memory.working_context()
        user_content = f"{query}\n\n[已知上下文]\n{working}" if working else query
        messages.append({"role": "user", "content": user_content})
        self.memory.add_user(query)

        tools = self.registry.schemas()
        final_answer = ""
        step_index = 0

        for step_index in range(1, self.max_steps + 1):
            try:
                with timer() as t:
                    response = self.llm.chat_with_tools(messages, tools)
            except LLMError as exc:
                result.success = False
                result.error = str(exc)
                final_answer = f"任务中断：LLM 调用失败（{exc}）。"
                logger.error("LLM 调用失败：%s", exc)
                break

            result.llm_calls += 1
            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens

            # ---- 无工具调用：模型认为可以回答了 ----
            if not response.tool_calls:
                final_answer = response.content or "（模型未返回内容）"
                self._emit(
                    AgentStep(
                        index=step_index,
                        thought=truncate(response.content, 200),
                        action="",
                        observation="任务完成，生成最终回答",
                        latency_ms=t["elapsed_ms"],
                    )
                )
                break

            thought = response.content or ""
            messages.append(
                {
                    "role": "assistant",
                    "content": thought,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }
                        for call in response.tool_calls
                    ],
                }
            )

            # ---- 依次执行本步的所有工具调用 ----
            for call in response.tool_calls:
                from ..utils import safe_json_loads

                arguments = safe_json_loads(call["arguments"], default={})
                if not isinstance(arguments, dict):
                    arguments = {}

                observation, ok = self._execute(call["name"], arguments)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": observation,
                    }
                )

                self._emit(
                    AgentStep(
                        index=step_index,
                        thought=thought,
                        action=call["name"],
                        action_input=arguments,
                        observation=observation,
                        ok=ok,
                        latency_ms=response.latency_ms,
                    )
                )
                result.steps.append(
                    AgentStep(
                        index=step_index,
                        thought=thought,
                        action=call["name"],
                        action_input=arguments,
                        observation=observation,
                        ok=ok,
                        latency_ms=t["elapsed_ms"],
                    )
                )
        else:
            # ---- 达到步数上限：强制收敛 ----
            logger.warning("达到最大步数 %d，强制生成最终回答", self.max_steps)
            messages.append({"role": "user", "content": FINAL_ANSWER_PROMPT.format(query=query)})
            try:
                response = self.llm.chat(messages, tools=None)
                result.llm_calls += 1
                result.prompt_tokens += response.prompt_tokens
                result.completion_tokens += response.completion_tokens
                final_answer = response.content or "（模型未返回内容）"
            except LLMError as exc:
                result.success = False
                result.error = str(exc)
                final_answer = "已达到工具调用上限，且生成最终回答失败。"

        result.answer = final_answer
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self.memory.add_assistant(final_answer)

        if not result.steps and not result.error:
            result.success = False
            result.error = "未执行任何工具调用"
        elif any(not s.ok for s in result.steps) and not final_answer:
            result.success = False

        return result

    # ------------------------------------------------------------------ #
    def available_tools(self) -> list[dict[str, str]]:
        return self.registry.describe()
