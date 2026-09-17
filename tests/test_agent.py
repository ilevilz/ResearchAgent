"""Agent 层单测：用打桩 LLM 验证 ReAct 循环，不消耗 API 额度。

覆盖三类行为：
    1. 正常收敛：Thought -> Action -> Observation -> 最终回答；
    2. 步数耗尽：必须强制收敛，而不是死循环；
    3. 工具报错：Agent 不崩溃，且错误如实出现在 Observation 里。
"""
from __future__ import annotations

from typing import Any

from research_agent.agent.react_agent import ReActAgent
from research_agent.core.llm_client import LLMResponse
from research_agent.tools.base import Tool
from research_agent.tools.registry import ToolRegistry


class StubLLM:
    """按预设脚本依次返回响应的打桩 LLM。"""

    def __init__(self, scripted: list[LLMResponse]) -> None:
        self.scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def _next(self) -> LLMResponse:
        return self.scripted.pop(0) if self.scripted else LLMResponse(content="(脚本耗尽)")

    def chat_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kw: Any) -> LLMResponse:
        self.calls.append({"n_messages": len(messages), "n_tools": len(tools)})
        return self._next()

    def chat(self, messages: list[dict[str, Any]], **kw: Any) -> LLMResponse:
        self.calls.append({"n_messages": len(messages), "final": True})
        return self._next()


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": arguments}


def _registry() -> ToolRegistry:
    def echo(text: str) -> dict[str, Any]:
        return {"echo": text}

    def explode() -> dict[str, Any]:
        raise RuntimeError("工具内部错误")

    return ToolRegistry([
        Tool("echo", "回显", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}, echo),
        Tool("explode", "抛错", {"type": "object", "properties": {}, "required": []}, explode),
    ])


def test_agent_completes_single_step_then_answers():
    llm = StubLLM([
        LLMResponse(content="需要先回显", tool_calls=[_tool_call("c1", "echo", '{"text": "hi"}')], latency_ms=10),
        LLMResponse(content="最终答案：已回显 hi", latency_ms=10),
    ])
    agent = ReActAgent(registry=_registry(), llm=llm, max_steps=4, verbose=False)
    result = agent.run("测试", reset_memory=True)

    assert result.success
    assert result.answer == "最终答案：已回显 hi"
    assert result.tools_used == ["echo"]
    assert len(result.steps) == 1
    assert "hi" in result.steps[0].observation
    assert result.llm_calls == 2


def test_agent_forces_convergence_at_max_steps():
    """模型一直要求调工具时，必须在 max_steps 后强制收敛。"""
    scripted = [
        LLMResponse(content="再查一次", tool_calls=[_tool_call(f"c{i}", "echo", '{"text": "x"}')], latency_ms=5)
        for i in range(3)
    ]
    scripted.append(LLMResponse(content="基于已有信息的最终回答", latency_ms=5))

    llm = StubLLM(scripted)
    agent = ReActAgent(registry=_registry(), llm=llm, max_steps=3, verbose=False)
    result = agent.run("无限循环测试", reset_memory=True)

    assert result.answer == "基于已有信息的最终回答"
    assert len(result.steps) == 3


def test_agent_survives_tool_exception():
    llm = StubLLM([
        LLMResponse(content="调用会抛错的工具", tool_calls=[_tool_call("c1", "explode", "{}")], latency_ms=5),
        LLMResponse(content="工具失败了，我如实告知用户。", latency_ms=5),
    ])
    agent = ReActAgent(registry=_registry(), llm=llm, max_steps=3, verbose=False)
    result = agent.run("触发异常", reset_memory=True)

    assert result.steps[0].ok is False
    assert "工具内部错误" in result.steps[0].observation
    assert result.answer  # 仍然给出回答，没有崩溃


def test_agent_handles_unknown_tool_gracefully():
    llm = StubLLM([
        LLMResponse(content="调用不存在的工具", tool_calls=[_tool_call("c1", "nope", "{}")], latency_ms=5),
        LLMResponse(content="该工具不可用。", latency_ms=5),
    ])
    agent = ReActAgent(registry=_registry(), llm=llm, max_steps=3, verbose=False)
    result = agent.run("未知工具", reset_memory=True)

    assert result.steps[0].ok is False
    assert "未知工具" in result.steps[0].observation


def test_memory_records_multi_turn():
    from research_agent.memory.session import SessionMemory

    memory = SessionMemory()
    llm = StubLLM([
        LLMResponse(content="回答1", latency_ms=1),
        LLMResponse(content="回答2", latency_ms=1),
    ])
    agent = ReActAgent(registry=_registry(), llm=llm, memory=memory, verbose=False)

    agent.run("第一个问题")
    agent.run("第二个问题")

    assert len(memory.turns) == 4  # 2 user + 2 assistant
    assert memory.stats()["turns"] == 4
