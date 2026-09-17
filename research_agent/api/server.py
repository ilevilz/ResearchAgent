"""FastAPI 服务：对外暴露 Agent / RAG 能力，并托管 Web Demo。

启动：``python -m research_agent.api.server`` 或 ``make serve``
文档：http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .. import __version__
from ..config import settings
from ..utils import get_logger, setup_logging
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
)
from .service import get_service

logger = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(
    title="ResearchAgent API",
    description="基于 ReAct + RAG + MCP 的科研智能分析 Agent",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    setup_logging()
    settings.ensure_dirs()
    logger.info("ResearchAgent API 启动 | model=%s | key=%s", settings.deepseek_model, settings.has_llm_key)


# ---------------------------------------------------------------------- #
# 基础
# ---------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> Any:
    """Web Demo 首页。"""
    page = WEB_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"service": "ResearchAgent", "version": __version__, "docs": "/docs"})


@app.get("/health", response_model=HealthResponse, summary="健康检查")
def health(deep: bool = False) -> Any:
    """服务健康状态。``deep=true`` 会真实调用一次 LLM（较慢）。"""
    return get_service().health(deep=deep)


@app.get("/tools", summary="工具清单")
def list_tools() -> Any:
    return {"count": len(get_service().list_tools()), "tools": get_service().list_tools()}


# ---------------------------------------------------------------------- #
# Agent
# ---------------------------------------------------------------------- #
@app.post("/chat", response_model=ChatResponse, summary="执行 Agent 任务")
def chat(request: ChatRequest) -> Any:
    """把任务交给 ReAct Agent 执行，返回答案与推理轨迹。"""
    # 未配置 Key 属于「服务未就绪」而非「服务出错」，先拦成 503：
    # 否则会一路冒泡成 500，把配置问题伪装成代码 bug。
    if not settings.has_llm_key:
        raise HTTPException(
            status_code=503,
            detail="未配置 DEEPSEEK_API_KEY，Agent 无法调用 LLM。请在 .env 中填写后重启服务。",
        )
    try:
        result = get_service().chat(
            query=request.query,
            session_id=request.session_id,
            max_steps=request.max_steps,
            reset_memory=request.reset_memory,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("执行 Agent 任务失败")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    payload = result.as_dict(include_trace=request.include_trace)
    if not request.include_trace:
        payload["trace"] = []
    return payload


# ---------------------------------------------------------------------- #
# RAG
# ---------------------------------------------------------------------- #
@app.post("/index", response_model=IndexResponse, summary="写入知识库")
def index_papers(request: IndexRequest) -> Any:
    """把论文（摘要或 PDF）写入向量知识库。"""
    if not any([request.papers, request.pdf_paths, request.source_query]):
        raise HTTPException(status_code=400, detail="papers / pdf_paths / source_query 至少提供一个")
    try:
        return get_service().index(
            papers=request.papers,
            pdf_paths=request.pdf_paths,
            source_query=request.source_query,
            max_results=request.max_results,
            save=request.save,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("入库失败")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/search", response_model=SearchResponse, summary="知识库检索")
def search(request: SearchRequest) -> Any:
    """混合检索（向量 + BM25 + RRF）。"""
    results = get_service().search(request.query, top_k=request.top_k)
    return {"query": request.query, "num_results": len(results), "results": results}


@app.get("/sessions", summary="会话列表")
def sessions() -> Any:
    return {"sessions": get_service().list_sessions()}


@app.delete("/sessions/{session_id}", summary="清空会话记忆")
def reset_session(session_id: str) -> Any:
    get_service().reset_session(session_id)
    return {"ok": True, "session_id": session_id}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
