"""评测层单测：指标计算与报告聚合。

重点是回归 `aggregate()` 的崩溃——它曾在整轮评测跑完后才抛 AttributeError，
导致 9 分钟的结果全部丢失，且单条用例的评分逻辑本身完全正常，
所以只有「聚合」这一步的测试才能拦住它。
"""
from __future__ import annotations

from research_agent.eval.metrics import CaseScore, aggregate


def _score(**kw) -> CaseScore:
    base = {"case_id": "c1", "category": "search", "success": True, "tool_f1": 1.0, "latency_ms": 100.0}
    base.update(kw)
    return CaseScore(**base)


def test_aggregate_does_not_crash_on_retrieval_cases():
    """回归测试：aggregate 曾读 CaseScore 上不存在的 expected_sources 而崩溃。"""
    report = aggregate([
        _score(case_id="a", retrieval_evaluated=True, retrieval_hit=True, retrieval_mrr=1.0),
        _score(case_id="b", retrieval_evaluated=True, retrieval_hit=False, retrieval_mrr=0.0),
        _score(case_id="c"),  # 无需检索的用例
    ])
    assert report.num_cases == 3
    assert report.retrieval_accuracy == 0.5
    assert report.retrieval_mrr == 0.5


def test_aggregate_excludes_non_retrieval_cases_from_retrieval_metrics():
    """不做检索的用例不能拉低检索准确率。"""
    report = aggregate([
        _score(case_id="a", retrieval_evaluated=True, retrieval_hit=True, retrieval_mrr=1.0),
        _score(case_id="b", retrieval_hit=False),
        _score(case_id="c", retrieval_hit=False),
    ])
    assert report.retrieval_accuracy == 1.0, "未做检索的用例被错误计入分母"


def test_aggregate_with_no_retrieval_cases_is_zero_not_error():
    report = aggregate([_score(case_id="a"), _score(case_id="b")])
    assert report.retrieval_accuracy == 0.0
    assert report.retrieval_mrr == 0.0


def test_aggregate_empty_input():
    report = aggregate([])
    assert report.num_cases == 0
    assert report.to_markdown()


def test_aggregate_computes_four_dimensions():
    report = aggregate([
        _score(case_id="a", success=True, tool_f1=1.0, latency_ms=100.0),
        _score(case_id="b", success=False, tool_f1=0.5, latency_ms=300.0),
    ])
    assert report.task_success_rate == 0.5
    assert report.tool_calling_accuracy == 0.75
    assert report.latency_mean_ms == 200.0
    assert report.latency_p50_ms == 200.0
    assert report.failed_cases == ["b"]


def test_report_markdown_contains_all_four_dimensions():
    md = aggregate([_score(case_id="a", latency_ms=120.0)]).to_markdown()
    for token in ("任务成功率", "工具调用", "检索", "延迟"):
        assert token in md, f"报告缺少维度：{token}"


def test_report_is_json_serializable():
    import json

    report = aggregate([_score(case_id="a", retrieval_evaluated=True, retrieval_hit=True)])
    payload = json.dumps(report.as_dict(), ensure_ascii=False)
    assert "task_success_rate" in payload
