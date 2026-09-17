"""DeepSeek LLM 客户端。

统一封装 OpenAI 兼容接口，对外暴露两个能力：
    - ``chat(messages)``        -> 纯文本对话
    - ``chat_with_tools(...)``  -> 带 Function Calling 的对话

同时做了三件事：
    1. 自动重试（网络抖动 / 限流），指数退避；
    2. 统计 token 用量与耗时，供 Evaluation 的 Latency 维度使用；
    3. 可注入自定义 client，便于单元测试打桩。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import settings
from ..utils import get_logger, timer

logger = get_logger(__name__)


class SupportsChat(Protocol):
    """OpenAI SDK 客户端的最小协议（便于测试替身）。"""

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...  # pragma: no cover


@dataclass
class LLMResponse:
    """一次 LLM 调用的结构化结果。"""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """LLM 调用失败（重试耗尽后抛出）。"""


class DeepSeekClient:
    """DeepSeek Chat 客户端（OpenAI 兼容协议）。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.model = model or settings.deepseek_model
        self.max_retries = max_retries
        self._client = None
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    @property
    def client(self):
        """惰性构建 OpenAI SDK 客户端（无 Key 时不报错，调用时才失败）。"""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self._timeout,
                max_retries=0,  # 重试由本类统一控制，便于记录日志
            )
        return self._client

    def _call(self, messages: list[dict[str, Any]], tools: list[dict] | None = None, **overrides: Any) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": overrides.pop("model", self.model),
            "messages": messages,
            "temperature": overrides.pop("temperature", settings.llm_temperature),
            "max_tokens": overrides.pop("max_tokens", settings.llm_max_tokens),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = overrides.pop("tool_choice", "auto")

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with timer() as t:
                    resp = self.client.chat.completions.create(**payload)
                return self._parse(resp, t["elapsed_ms"])
            except Exception as exc:  # noqa: BLE001 - 需要兜住 SDK 的各种异常
                last_err = exc
                if attempt < self.max_retries:
                    backoff = 1.5 ** attempt
                    logger.warning("LLM 调用失败（第 %d/%d 次）：%s，%.1fs 后重试",
                                   attempt, self.max_retries, exc, backoff)
                    time.sleep(backoff)
        raise LLMError(f"LLM 调用失败，已重试 {self.max_retries} 次：{last_err}") from last_err

    @staticmethod
    def _parse(resp: Any, latency_ms: float) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[dict[str, Any]] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=(msg.content or "").strip(),
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            model=getattr(resp, "model", ""),
            raw=resp,
        )

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def chat(self, messages: list[dict[str, Any]], **overrides: Any) -> LLMResponse:
        """纯文本对话。"""
        return self._call(messages, tools=None, **overrides)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **overrides: Any,
    ) -> LLMResponse:
        """带 Function Calling 的对话。"""
        return self._call(messages, tools=tools, **overrides)

    def health_check(self) -> dict[str, Any]:
        """连通性自检：供 ``make doctor`` / API ``/health`` 使用。"""
        if not settings.has_llm_key:
            return {"ok": False, "reason": "未配置 DEEPSEEK_API_KEY"}
        try:
            resp = self.chat([{"role": "user", "content": "ping"}], max_tokens=8)
            return {"ok": True, "model": resp.model or self.model, "latency_ms": resp.latency_ms}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": str(exc)}


_default_client: DeepSeekClient | None = None


def get_llm_client() -> DeepSeekClient:
    """进程内单例，避免重复建立连接池。"""
    global _default_client
    if _default_client is None:
        _default_client = DeepSeekClient()
    return _default_client
