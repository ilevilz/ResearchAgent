"""MCP 客户端：连接 Research MCP Server，并把远程工具适配为本地 ``Tool``。

两条传输路径，对外接口一致：

    - ``stdio``     —— 真实 MCP 协议（子进程 + JSON-RPC），演示协议互通；
    - ``inprocess`` —— 同进程直调，省去子进程开销，便于调试与单测。

``MCPClient`` 对外暴露的是**同步**接口，内部用独立线程跑事件循环，
这样 ReAct 主循环保持同步、易读，同时又能复用 MCP 官方的异步 SDK。
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, settings
from ..tools.base import Tool
from ..utils import dumps, get_logger, safe_json_loads

logger = get_logger(__name__)


class MCPClientError(RuntimeError):
    """MCP 交互失败。"""


class MCPClient:
    """同步外观的 MCP 客户端。"""

    def __init__(
        self,
        transport: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | Path | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.transport = (transport or settings.mcp_transport or "stdio").strip().lower()
        self.command = command or sys.executable
        self.args = args or ["-m", "research_agent.mcp_server.server"]
        self.cwd = Path(cwd or PROJECT_ROOT)
        self.timeout = timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._tools_cache: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def connect(self) -> "MCPClient":
        """建立连接（幂等）。"""
        if self.transport == "inprocess":
            logger.info("MCP 传输：inprocess（同进程直调）")
            return self
        if self._session is not None:
            return self

        self._thread = threading.Thread(target=self._run_loop, name="mcp-loop", daemon=True)
        self._thread.start()

        if not self._ready.wait(timeout=self.timeout):
            raise MCPClientError(f"MCP Server 连接超时（{self.timeout}s）：{self._error}")
        if self._error is not None:
            raise MCPClientError(f"MCP Server 启动失败：{self._error}")
        logger.info("MCP 传输：stdio（子进程 %s）", " ".join(self.args))
        return self

    def _run_loop(self) -> None:
        """后台线程：持有事件循环与 ClientSession 的整个生命周期。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()
        finally:
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001
                pass

    async def _main(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self.command, args=self.args, cwd=str(self.cwd))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                # 阻塞直到 close()
                while not self._stop.is_set():
                    await asyncio.sleep(0.1)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._session = None

    def __enter__(self) -> "MCPClient":
        return self.connect()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # 协议操作
    # ------------------------------------------------------------------ #
    def _submit(self, coro: Any) -> Any:
        if self._loop is None:
            raise MCPClientError("MCP 客户端未连接，请先调用 connect()")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise MCPClientError(f"MCP 调用失败：{exc}") from exc

    def list_tools(self, use_cache: bool = True) -> list[dict[str, Any]]:
        """获取 MCP Server 暴露的工具列表。"""
        if self.transport == "inprocess":
            from ..tools.registry import get_default_registry

            return [t.to_mcp_schema() for t in get_default_registry().tools()]

        if use_cache and self._tools_cache is not None:
            return self._tools_cache
        self.connect()

        async def _list() -> list[dict[str, Any]]:
            result = await self._session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    # mcp 2.x 的 Tool 是 pydantic 模型，字段名从 inputSchema 改为 input_schema；
                    # 这里同时兼容两个版本，避免升级 SDK 时静默失效。
                    "inputSchema": getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or {},
                }
                for t in result.tools
            ]

        self._tools_cache = self._submit(_list())
        return self._tools_cache

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """调用 MCP 工具，返回解析后的结果。"""
        arguments = arguments or {}

        if self.transport == "inprocess":
            from ..tools.registry import get_default_registry

            result = get_default_registry().call(name, arguments)
            if not result.ok:
                raise MCPClientError(result.error)
            return result.data

        self.connect()

        async def _call() -> Any:
            result = await self._session.call_tool(name, arguments)
            texts = [getattr(block, "text", "") for block in result.content if getattr(block, "text", None)]
            payload = "\n".join(texts)
            return safe_json_loads(payload, default=payload)

        return self._submit(_call())

    # ------------------------------------------------------------------ #
    # 适配为本地 Tool
    # ------------------------------------------------------------------ #
    def as_tools(self) -> list[Tool]:
        """把 MCP 工具包装成本地 ``Tool``，供 ReAct Agent 无感调用。"""
        adapted: list[Tool] = []
        for schema in self.list_tools():
            name = schema["name"]
            description = schema.get("description") or f"MCP 工具 {name}"
            parameters = schema.get("inputSchema") or {"type": "object", "properties": {}}

            def _make(tool_name: str):
                def _call(**kwargs: Any) -> Any:
                    return self.call_tool(tool_name, kwargs)

                _call.__name__ = tool_name
                return _call

            adapted.append(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    func=_make(name),
                    tags=["mcp"],
                )
            )
        return adapted


def get_mcp_registry(transport: str | None = None):
    """构建一个由 MCP Server 提供全部工具的 ``ToolRegistry``。

    这是「Agent 工具全部来自 MCP」的完整形态：
        registry = get_mcp_registry()   # stdio 或 inprocess
        agent = ReActAgent(registry=registry)
    """
    from ..tools.registry import ToolRegistry

    client = MCPClient(transport=transport)
    client.connect()
    registry = ToolRegistry(client.as_tools())
    logger.info("已从 MCP Server 接入 %d 个工具", len(registry))
    return registry, client


if __name__ == "__main__":  # pragma: no cover - 手动联通性自检
    with MCPClient() as mcp:
        tools = mcp.list_tools()
        print(f"MCP Server 暴露 {len(tools)} 个工具：")
        for item in tools:
            print(f"  - {item['name']}: {item['description'][:70]}")
        print("\n调用示例 search_papers('retrieval augmented generation', max_results=2)：")
        print(dumps(mcp.call_tool("search_papers", {"query": "retrieval augmented generation", "max_results": 2})[:800]))
