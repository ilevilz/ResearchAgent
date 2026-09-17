"""mcp_client：MCP 客户端，把 MCP 工具适配为 Agent 可直接调用的本地工具。"""

from .client import MCPClient, get_mcp_registry

__all__ = ["MCPClient", "get_mcp_registry"]
