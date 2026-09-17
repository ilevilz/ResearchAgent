"""通用工具：日志、计时、文本与 JSON 容错处理。"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-26s | %(message)s"
_QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "openai", "mcp", "asyncio", "filelock")


def setup_logging(level: int = logging.INFO) -> None:
    """幂等初始化全局日志（重复调用不会叠加 handler）。"""
    root = logging.getLogger()
    if getattr(root, "_ra_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    root.handlers = [handler]
    root.setLevel(level)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    root._ra_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timer() -> Iterator[dict]:
    """计时上下文：``with timer() as t: ...``，结果写入 ``t["elapsed_ms"]``。"""
    box: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)


def truncate(text: Any, limit: int = 800, suffix: str = " …[已截断]") -> str:
    """安全截断，用于日志输出与 Observation 压缩。"""
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + suffix


def dumps(obj: Any, indent: int | None = None) -> str:
    """UTF-8 友好的 JSON 序列化（中文不转义）。"""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=indent, default=str)
    except (TypeError, ValueError):
        return str(obj)


def strip_code_fence(text: str) -> str:
    """剥离 LLM 输出常见的 ```json ... ``` 围栏。"""
    if not text:
        return ""
    t = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_+\-]*\s*\n?(.*?)\n?```$", t, re.DOTALL)
    return m.group(1).strip() if m else t


def safe_json_loads(text: str, default: Any = None) -> Any:
    """三级容错 JSON 解析：直解 -> 剥围栏 -> 正则截取首尾括号。"""
    if not text:
        return default
    fenced = strip_code_fence(text)
    for candidate in (fenced, text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    m = re.search(r"[\[{].*[\]}]", fenced, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return default


def normalize_ws(text: Any) -> str:
    """压缩空白字符，便于比较与统计。"""
    return re.sub(r"\s+", " ", str(text or "")).strip()


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """轻量分词：英文按单词、中文按单字（哈希向量与关键词召回共用）。"""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def slugify(text: str, max_len: int = 60) -> str:
    """生成安全的文件名片段。"""
    s = re.sub(r"[^a-zA-Z0-9一-鿿]+", "-", str(text or "").strip().lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return (s or "item")[:max_len]
