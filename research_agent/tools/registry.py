"""工具注册表：统一管理工具的生命周期与调用入口。

职责：
    - 注册 / 注销工具，导出 Function Calling schema；
    - 按名称安全调用（未知工具返回失败结果而非抛异常，保证 Agent 循环不被打断）；
    - 记录每次调用的耗时，供 Evaluation 的 Latency / Tool-Calling Accuracy 统计。
"""
from __future__ import annotations

from typing import Any, Iterable

from ..utils import get_logger
from .base import Tool, ToolResult

logger = get_logger(__name__)


class ToolRegistry:
    """工具集合。"""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self.call_history: list[dict[str, Any]] = []
        for item in tools or []:
            self.register(item)

    # ------------------------------------------------------------------ #
    def register(self, item: Tool, *, override: bool = True) -> None:
        if item.name in self._tools and not override:
            raise ValueError(f"工具已存在：{item.name}")
        self._tools[item.name] = item

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        """返回全部 ``Tool`` 对象（供 MCP Server 动态注册使用）。"""
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------ #
    def schemas(self) -> list[dict[str, Any]]:
        """导出全部工具的 Function Calling schema。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def describe(self) -> list[dict[str, str]]:
        """工具清单（供 CLI / API 展示）。"""
        return [{"name": t.name, "description": t.description, "tags": ",".join(t.tags)} for t in self._tools.values()]

    # ------------------------------------------------------------------ #
    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """按名调用工具，并记录调用历史。"""
        item = self.get(name)
        if item is None:
            result = ToolResult(
                tool=name,
                ok=False,
                error=f"未知工具 '{name}'，可用工具：{', '.join(self.names())}",
            )
        else:
            result = item.run(**(arguments or {}))

        self.call_history.append(
            {
                "tool": name,
                "args": arguments or {},
                "ok": result.ok,
                "latency_ms": result.latency_ms,
                "error": result.error,
            }
        )
        return result

    def reset_history(self) -> None:
        self.call_history.clear()


_DEFAULT_REGISTRY: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """构建（并缓存）内置工具注册表。"""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY

    from .data_analysis import build_analysis_tools
    from .experiment import build_experiment_tools
    from .kb_tools import build_kb_tools
    from .paper_search import build_search_tools
    from .pdf_parser import build_pdf_tools

    registry = ToolRegistry()
    for builder in (
        build_search_tools,
        build_pdf_tools,
        build_kb_tools,
        build_experiment_tools,
        build_analysis_tools,
    ):
        for item in builder():
            registry.register(item)

    logger.info("已注册 %d 个工具：%s", len(registry), ", ".join(registry.names()))
    _DEFAULT_REGISTRY = registry
    return registry
