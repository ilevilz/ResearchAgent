"""Research MCP Server —— 把内部工具标准化暴露为 MCP Tools。

设计要点（对应简历「基于 MCP 构建 Research MCP Server，实现工具的标准化接入与扩展」）：

1. **单一事实来源**：工具只在 ``research_agent/tools/`` 定义一次，MCP Server
   直接遍历 ``ToolRegistry`` 动态注册，不重复声明 schema；
   因此「新增一个工具」= 写一个函数 + 加进 builder，本地 Agent 与 MCP 同时生效。
2. **协议标准**：走 MCP 官方 stdio 传输，任何支持 MCP 的宿主（Claude Desktop 等）
   都能直接接入，无需改动 Agent 代码。
3. **启动方式**：
       python -m research_agent.mcp_server.server
   或在 Claude Desktop 的配置中注册为 MCP Server。
"""
from __future__ import annotations

import sys

from ..tools.registry import get_default_registry
from ..utils import get_logger, setup_logging

logger = get_logger(__name__)

SERVER_NAME = "research-mcp-server"


def _create_server():
    """构建 MCP Server，兼容支持两代 SDK API。

    - mcp >= 2.0：``mcp.server.mcpserver.MCPServer``（FastMCP 已更名）
    - mcp <  2.0：``mcp.server.fastmcp.FastMCP``
    """
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer(SERVER_NAME), "MCPServer(2.x)"
    except ImportError:
        from mcp.server.fastmcp import FastMCP

        return FastMCP(SERVER_NAME), "FastMCP(1.x)"


def build_server():
    """创建 Server 实例并动态注册全部工具。"""
    server, flavor = _create_server()
    registry = get_default_registry()

    registered: list[str] = []
    for tool_obj in registry.tools():
        try:
            server.add_tool(tool_obj.func, name=tool_obj.name, description=tool_obj.description)
            registered.append(tool_obj.name)
        except Exception as exc:  # noqa: BLE001 - 单个工具注册失败不应拖垮整个 Server
            logger.warning("MCP 注册工具 %s 失败：%s", tool_obj.name, exc)

    logger.info(
        "Research MCP Server 就绪 [%s]，已注册 %d 个工具：%s",
        flavor, len(registered), ", ".join(registered),
    )
    return server


def main() -> None:
    """stdio 模式启动。"""
    setup_logging()
    # stdio 传输下 stdout 属于协议通道，日志必须走 stderr，否则会污染 JSON-RPC 消息
    import logging

    for handler in logging.getLogger().handlers:
        handler.stream = sys.stderr  # type: ignore[attr-defined]

    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
