"""工具基础设施：统一的结果封装、可观测性与 Function Calling Schema 生成。

一个 ``Tool`` 同时具备三重身份，这是本项目「一次定义、多处复用」的关键：
    1. 本地函数  —— ReAct Agent 直接调用；
    2. JSON Schema —— 转成 DeepSeek 的 ``tools`` 参数（Function Calling）；
    3. MCP Tool  —— 由 Research MCP Server 原样暴露为标准 MCP 工具。
"""
from __future__ import annotations

import inspect
import re
import time
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Union, get_args, get_origin

# X | Y 语法在 3.10+ 才有独立的 types.UnionType
_UNION_TYPE = getattr(types, "UnionType", None)

from ..utils import dumps, get_logger, truncate

logger = get_logger(__name__)


class ToolError(RuntimeError):
    """工具执行失败（会被转换为 Observation 回传给 LLM，而不是中断循环）。"""


@dataclass
class ToolResult:
    """工具调用的结构化结果。"""

    tool: str
    ok: bool = True
    data: Any = None
    error: str = ""
    latency_ms: float = 0.0
    args: dict[str, Any] = field(default_factory=dict)

    def to_observation(self, limit: int = 2500) -> str:
        """序列化成给 LLM 阅读的 Observation 文本。

        统一在这里做截断，避免超长 PDF / 检索结果撑爆上下文窗口。
        """
        if not self.ok:
            return f"[工具 {self.tool} 执行失败] {self.error}"
        body = self.data if isinstance(self.data, str) else dumps(self.data, indent=None)
        return truncate(body, limit)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class Tool:
    """自描述工具对象。"""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
        tags: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.tags = tags or []

    # ---- 对外表示 ----
    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI / DeepSeek Function Calling 的 tools 元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_mcp_schema(self) -> dict[str, Any]:
        """转换为 MCP 工具定义。"""
        return {"name": self.name, "description": self.description, "inputSchema": self.parameters}

    # ---- 执行 ----
    def run(self, **kwargs: Any) -> ToolResult:
        """执行工具；任何异常都被收敛为 ``ok=False`` 的结果。

        计时用显式的时间戳而非 ``timer()`` 上下文管理器：
        后者在 ``finally`` 中写入 ``elapsed_ms``，早于 ``return`` 求值，
        会导致取值时 KeyError。
        """
        start = time.perf_counter()

        def _elapsed_ms() -> float:
            return round((time.perf_counter() - start) * 1000, 2)

        try:
            result = self.func(**kwargs)
            return ToolResult(tool=self.name, ok=True, data=result, latency_ms=_elapsed_ms(), args=kwargs)
        except TypeError as exc:
            # 参数错误单独区分，便于暴露 prompt / schema 不一致的问题
            logger.warning("工具 %s 参数错误：%s", self.name, exc)
            return ToolResult(
                tool=self.name, ok=False, error=f"参数错误：{exc}", latency_ms=_elapsed_ms(), args=kwargs
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("工具 %s 执行异常：%s: %s", self.name, type(exc).__name__, exc)
            return ToolResult(
                tool=self.name, ok=False, error=f"{type(exc).__name__}: {exc}", latency_ms=_elapsed_ms(), args=kwargs
            )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tool {self.name}>"


_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _annotation_to_json_type(annotation: Any) -> str | None:
    """把类型注解映射为 JSON Schema 的 type。

    需要同时处理 ``list[str]`` / ``dict[str, Any]`` / ``Optional[X]`` 这类
    泛型与联合类型——否则它们会被误判成 ``string``，导致 LLM 传错参数格式。
    """
    if annotation is inspect.Parameter.empty:
        return None

    # 必须先判 get_origin 再判 isinstance(annotation, type)：
    # 3.9+ 的 list[str] 是 types.GenericAlias，isinstance 同样为真，
    # 若先走 type 分支会被 _PY_TO_JSON 直接吞掉并返回 None，
    # 最终导致 Schema 里退化成 "string"，LLM 就会把数组传成字符串。
    origin = get_origin(annotation)
    if origin is not None:
        # Optional[X] / X | None -> 取非 None 的那一支
        if origin is Union or (_UNION_TYPE is not None and origin is _UNION_TYPE):
            args = [a for a in get_args(annotation) if a is not type(None)]
            return _annotation_to_json_type(args[0]) if args else None

        if origin in (list, set, tuple, frozenset):
            return "array"
        if origin in (dict,):
            return "object"
        return None

    if isinstance(annotation, type):
        return _PY_TO_JSON.get(annotation)
    return None


def _schema_from_signature(func: Callable[..., Any], overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """由函数签名推导 JSON Schema，并用 ``overrides`` 补充描述与默认值。"""
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        spec = dict(overrides.get(name, {}))
        json_type = spec.get("type") or _annotation_to_json_type(param.annotation) or "string"

        entry: dict[str, Any] = {"type": json_type}
        if "description" in spec:
            entry["description"] = spec["description"]
        if "enum" in spec:
            entry["enum"] = spec["enum"]
        if json_type == "array":
            item_spec = spec.get("items", {"type": "string"})
            entry["items"] = item_spec
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        else:
            required.append(name)
        properties[name] = entry

    return {"type": "object", "properties": properties, "required": required}


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, dict[str, Any]] | None = None,
    tags: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """装饰器：把普通函数登记为 ``Tool``。

    用法::

        @tool(
            name="search_papers",
            description="按关键词检索 arXiv 论文",
            parameters={"query": {"type": "string", "description": "检索关键词"}},
        )
        def search_papers(query: str, max_results: int = 5): ...
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        doc = inspect.getdoc(func) or ""
        tool_desc = description or doc.split("\n\n")[0].replace("\n", " ").strip()
        schema = _schema_from_signature(func, parameters or {})
        return Tool(
            name=tool_name,
            description=tool_desc,
            parameters=schema,
            func=func,
            tags=tags,
        )

    return decorator


def clean_query(text: str) -> str:
    """清洗检索词：去掉多余空白与标点。"""
    return re.sub(r"\s+", " ", str(text or "")).strip(" \t\n，。,.、:：;；")
