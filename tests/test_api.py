"""API 层单测：契约与错误处理，不触发真实 LLM 调用。"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from research_agent.api.server import app  # noqa: E402
from research_agent.config import settings  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ------------------------------------------------------------------ 基础
def test_health_reports_configuration_honestly(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["version"]
    assert isinstance(body["tools"], list) and body["tools"]
    # 未配置 Key 时必须如实标记为 degraded，而不是假装健康
    if not settings.has_llm_key:
        assert body["llm_configured"] is False
        assert body["status"] == "degraded"


def test_root_serves_web_demo(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ResearchAgent" in resp.text


def test_openapi_schema_is_generated(client: TestClient):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) >= {"/chat", "/search", "/index", "/health", "/tools"}


# ------------------------------------------------------------------ 工具
def test_tools_endpoint_lists_registered_tools(client: TestClient):
    body = client.get("/tools").json()
    names = {t["name"] for t in body["tools"]}
    assert body["count"] == len(names)
    assert {"search_papers", "search_knowledge_base", "extract_metrics"} <= names


# ------------------------------------------------------------------ 检索
def test_search_rejects_empty_query(client: TestClient):
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_search_rejects_out_of_range_top_k(client: TestClient):
    assert client.post("/search", json={"query": "RAG", "top_k": 999}).status_code == 422


def test_search_returns_structured_hits(client: TestClient):
    resp = client.post("/search", json={"query": "ReAct 推理与行动", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "ReAct 推理与行动"
    assert body["num_results"] == len(body["results"])
    for hit in body["results"]:
        assert {"text", "score"} <= set(hit)


# ------------------------------------------------------------------ Agent
def test_chat_validates_payload(client: TestClient):
    assert client.post("/chat", json={}).status_code == 422
    assert client.post("/chat", json={"query": "hi", "max_steps": 0}).status_code == 422


def test_chat_returns_503_when_llm_not_configured(client: TestClient):
    """没有 Key 时应当是明确的 503，而不是 500 堆栈。"""
    if settings.has_llm_key:
        pytest.skip("本环境已配置 Key，跳过未配置分支")
    resp = client.post("/chat", json={"query": "你好"})
    assert resp.status_code == 503
    assert "DEEPSEEK_API_KEY" in resp.json()["detail"]


# ------------------------------------------------------------------ 会话
def test_session_reset_is_idempotent(client: TestClient):
    for _ in range(2):
        resp = client.delete("/sessions/pytest-demo")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


def test_sessions_endpoint_returns_list(client: TestClient):
    assert isinstance(client.get("/sessions").json()["sessions"], list)


# ------------------------------------------------------------------ 入库
def test_index_requires_at_least_one_source(client: TestClient):
    resp = client.post("/index", json={})
    assert resp.status_code == 400
    assert "至少提供一个" in resp.json()["detail"]
