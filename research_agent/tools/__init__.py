"""tools：Agent 可调用的原子工具集合。

每个工具都是「自描述」的：携带 OpenAI Function Calling 所需的 JSON Schema，
因此既可以被本地 ReAct Agent 直接调用，也可以由 MCP Server 原样暴露出去。
"""

from .base import Tool, ToolError, ToolResult, tool
from .registry import ToolRegistry, get_default_registry

__all__ = ["Tool", "ToolError", "ToolResult", "tool", "ToolRegistry", "get_default_registry"]
