"""API 请求 / 响应模型（Pydantic）。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """一次 Agent 任务请求。"""

    query: str = Field(..., description="用户问题或任务描述", min_length=1)
    session_id: str | None = Field(None, description="会话 ID，用于多轮记忆；不传则无状态")
    max_steps: int | None = Field(None, description="最大工具调用步数，默认取服务端配置", ge=1, le=20)
    include_trace: bool = Field(True, description="是否返回 ReAct 推理轨迹")
    reset_memory: bool = Field(False, description="是否清空该会话历史后执行")


class ToolCallTrace(BaseModel):
    index: int
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    ok: bool = True
    latency_ms: float = 0.0


class ChatResponse(BaseModel):
    """Agent 任务结果。"""

    query: str
    answer: str
    success: bool
    session_id: str = ""
    steps: int = 0
    tools_used: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    total_tokens: int = 0
    llm_calls: int = 0
    error: str = ""
    trace: list[ToolCallTrace] = Field(default_factory=list)


class IndexRequest(BaseModel):
    """知识库入库请求。"""

    papers: list[dict[str, Any]] | None = Field(None, description="论文对象列表（含 title/abstract）")
    pdf_paths: list[str] | None = Field(None, description="本地 PDF 路径列表")
    source_query: str | None = Field(None, description="可选：先用该关键词检索 arXiv 再入库")
    max_results: int = Field(5, ge=1, le=20, description="检索入库时的论文数量")
    save: bool = Field(True, description="是否持久化索引")


class IndexResponse(BaseModel):
    added_chunks: int
    index_stats: dict[str, Any]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    num_results: int
    results: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    llm: dict[str, Any] = Field(default_factory=dict)
    knowledge_base: dict[str, Any] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    version: str = ""
