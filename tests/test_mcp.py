"""MCP 层单测：inprocess 与 stdio 两条传输必须行为一致。

stdio 这条路径曾经静默失效（mcp 2.x 把 Tool.inputSchema 改成了 input_schema），
因为只测了 inprocess 所以一直没暴露。这里两条都测。
"""
from __future__ import annotations

import pytest

from research_agent.mcp_client.client import MCPClient, MCPClientError
from research_agent.tools.registry import get_default_registry

EXPECTED_TOOLS = {
    "search_papers", "fetch_paper_detail", "download_pdf", "parse_pdf",
    "index_papers", "search_knowledge_base", "extract_metrics",
    "compare_experiments", "analyze_data", "preview_table", "compute_statistics",
}


# ------------------------------------------------------------ inprocess
@pytest.fixture(scope="module")
def inprocess() -> MCPClient:
    client = MCPClient(transport="inprocess")
    yield client
    client.close()


def test_inprocess_lists_all_tools(inprocess: MCPClient):
    assert {t["name"] for t in inprocess.list_tools()} == EXPECTED_TOOLS


def test_inprocess_call_tool_returns_data(inprocess: MCPClient):
    out = inprocess.call_tool("compute_statistics", {"values": [1, 2, 3, 4, 100]})
    assert out["count"] == 5
    assert out["max"] == 100.0


def test_inprocess_raises_on_unknown_tool(inprocess: MCPClient):
    with pytest.raises(MCPClientError):
        inprocess.call_tool("no_such_tool", {})


def test_as_tools_produces_callable_local_tools(inprocess: MCPClient):
    adapted = {t.name: t for t in inprocess.as_tools()}
    assert set(adapted) == EXPECTED_TOOLS
    result = adapted["compute_statistics"].run(values=[2, 4, 6])
    assert result.ok and result.data["mean"] == 4.0


# ---------------------------------------------------------------- stdio
@pytest.fixture(scope="module")
def stdio() -> MCPClient:
    client = MCPClient(transport="stdio")
    try:
        client.connect()
    except Exception as exc:  # noqa: BLE001 - 子进程环境不可用时跳过而非误报失败
        pytest.skip(f"stdio 传输不可用：{exc}")
    yield client
    client.close()


def test_stdio_lists_all_tools_with_schemas(stdio: MCPClient):
    tools = stdio.list_tools()
    assert {t["name"] for t in tools} == EXPECTED_TOOLS
    # 回归点：schema 字段名在 mcp 2.x 改过，取值失败会得到空 dict
    assert all(t["inputSchema"] for t in tools), "MCP 工具 schema 为空，疑似字段名不兼容"
    assert all(t["inputSchema"].get("type") == "object" for t in tools)


def test_stdio_call_tool_roundtrip(stdio: MCPClient):
    out = stdio.call_tool("compute_statistics", {"values": [1, 2, 3, 4, 100]})
    assert out["count"] == 5
    assert out["max"] == 100.0


def test_stdio_matches_inprocess_tool_set(stdio: MCPClient, inprocess: MCPClient):
    """两条传输暴露的工具集合必须一致，否则 Agent 行为会随传输方式漂移。"""
    assert {t["name"] for t in stdio.list_tools()} == {t["name"] for t in inprocess.list_tools()}


# ------------------------------------------------------- 与本地注册表一致
def test_mcp_tools_cover_local_registry(inprocess: MCPClient):
    """MCP 暴露的工具应恰好覆盖本地注册表，防止「加了工具忘了注册 MCP」。"""
    assert {t["name"] for t in inprocess.list_tools()} == set(get_default_registry().names())
