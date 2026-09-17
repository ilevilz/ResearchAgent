"""工具层单测：不依赖网络与 LLM。"""
from __future__ import annotations

import pytest

from research_agent.tools.base import Tool, _annotation_to_json_type
from research_agent.tools.data_analysis import analyze_data_impl, compute_statistics_impl
from research_agent.tools.experiment import compare_experiments_impl, extract_metrics_impl
from research_agent.tools.registry import ToolRegistry, get_default_registry


def test_registry_has_all_tools():
    registry = get_default_registry()
    names = set(registry.names())
    expected = {
        "search_papers", "fetch_paper_detail", "download_pdf", "parse_pdf",
        "index_papers", "search_knowledge_base", "extract_metrics",
        "compare_experiments", "analyze_data", "preview_table", "compute_statistics",
    }
    assert expected <= names


def test_schemas_are_valid_openai_format():
    for tool in get_default_registry().tools():
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == tool.name
        assert schema["function"]["description"]
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["required"], list)


def test_array_type_inference():
    """list[...] 必须推断成 array，否则 LLM 会传字符串。"""
    assert _annotation_to_json_type(list[str]) == "array"
    assert _annotation_to_json_type(list[float]) == "array"
    assert _annotation_to_json_type(dict[str, int]) == "object"
    assert _annotation_to_json_type(str | None) == "string"
    assert _annotation_to_json_type(int) == "integer"


def test_tool_run_records_latency_on_success_and_failure():
    """回归测试：elapsed_ms 曾在 return 时未写入导致 KeyError。"""
    def ok(x: int) -> int:
        return x * 2

    def boom() -> None:
        raise ValueError("爆炸")

    ok_tool = Tool("ok", "d", {"type": "object", "properties": {}, "required": []}, ok)
    bad_tool = Tool("bad", "d", {"type": "object", "properties": {}, "required": []}, boom)

    good = ok_tool.run(x=3)
    assert good.ok and good.data == 6 and good.latency_ms >= 0

    bad = bad_tool.run()
    assert not bad.ok and "爆炸" in bad.error and bad.latency_ms >= 0


def test_unknown_tool_returns_error_not_raise():
    result = ToolRegistry().call("does_not_exist", {})
    assert not result.ok
    assert "未知工具" in result.error


def test_compute_statistics():
    out = compute_statistics_impl([1, 2, 3, 4, 5])
    assert out["count"] == 5
    assert out["mean"] == 3.0
    assert out["min"] == 1.0 and out["max"] == 5.0


def test_analyze_data_groupby():
    rows = [
        {"model": "A", "score": 80.0},
        {"model": "A", "score": 82.0},
        {"model": "B", "score": 90.0},
    ]
    out = analyze_data_impl(rows=rows, group_by="model", operation="groupby")
    assert out["group_by"] == "model"
    assert out["aggregations"]["score"]["mean"]["A"] == 81.0
    assert out["aggregations"]["score"]["mean"]["B"] == 90.0


def test_analyze_data_missing_column_raises():
    from research_agent.tools.base import ToolError

    with pytest.raises(ToolError):
        analyze_data_impl(rows=[{"a": 1}], columns=["nope"])


def test_extract_metrics_regex_fallback():
    """无 LLM 时正则兜底应能抽出指标。"""
    text = "BERT-base 在 SQuAD 上取得 F1 93.2%，RoBERTa-large 在 SQuAD 上 F1 94.6%。"
    out = extract_metrics_impl(text, source="t", use_llm=False)
    assert out["count"] >= 2
    values = {m["value"] for m in out["metrics"]}
    assert 93.2 in values and 94.6 in values


def test_compare_experiments():
    records = [
        {"model": "BERT", "dataset": "SQuAD", "metric": "f1", "value": 93.2},
        {"model": "DeBERTa", "dataset": "SQuAD", "metric": "f1", "value": 95.1},
    ]
    out = compare_experiments_impl(records)
    assert out["total_records"] == 2
    assert out["comparisons"][0]["best_model"] == "DeBERTa"


def test_throttle_enforces_min_interval_between_requests():
    """回归测试：重试若不限速，会变相加重对方限流（实测 arXiv 直接封 IP）。"""
    import time

    from research_agent.tools.paper_search import _DEFAULT_INTERVAL, _throttle

    url = "https://example.com/api"  # 用未登记主机，走 _DEFAULT_INTERVAL
    _throttle(url)  # 首次不等待
    start = time.monotonic()
    _throttle(url)  # 紧接着的第二次必须被延迟
    elapsed = time.monotonic() - start
    assert elapsed >= _DEFAULT_INTERVAL * 0.8, f"未限速，仅间隔 {elapsed:.3f}s"


def test_arxiv_and_s2_have_dedicated_throttle_intervals():
    from research_agent.tools.paper_search import _MIN_INTERVAL

    assert _MIN_INTERVAL["export.arxiv.org"] >= 3.0, "arXiv 官方建议间隔 >= 3s"
    assert _MIN_INTERVAL["api.semanticscholar.org"] >= 1.0


def test_offline_fallback_returns_papers():
    """断网兜底必须能返回结果，且如实标记来源。"""
    from research_agent.tools.paper_search import search_papers_impl

    out = search_papers_impl("ReAct agent reasoning", max_results=3, source="none")
    assert out["degraded"] is True
    assert "offline_cache" in out["sources"]
    assert out["count"] > 0
    assert all(p["source"] == "offline_cache" for p in out["papers"])
