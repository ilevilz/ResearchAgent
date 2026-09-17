"""评测指标：Task Success Rate / Tool Calling Accuracy / Retrieval Accuracy / Latency。

四个维度分别回答不同问题，缺一不可：

| 指标 | 回答的问题 | 计算方式 |
|------|-----------|---------|
| Task Success Rate    | Agent 到底能不能把活干完？ | 无异常 且 关键词命中率达标 |
| Tool Calling Accuracy| 工具选对了吗？选全了吗？多选了吗？ | 期望工具集合 vs 实际调用集合的 F1 |
| Retrieval Accuracy   | 检索回来的东西对吗？ | 期望来源命中率 + 排名倒数（MRR） |
| Latency              | 用户要等多久？ | 端到端耗时的 mean / p50 / p95 |

单独看任何一个都会误判：例如「成功率 100% 但工具准确率 30%」说明 Agent 靠
LLM 自身知识在硬答，没有真的用工具——这正是需要被暴露出来的问题。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Iterable, Sequence

SUCCESS_KEYWORD_THRESHOLD = 0.5  # 关键词命中率达到 50% 记为任务成功


@dataclass
class CaseScore:
    """单条用例的评分明细。"""

    case_id: str
    category: str
    success: bool = False
    # 工具调用
    tool_precision: float = 0.0
    tool_recall: float = 0.0
    tool_f1: float = 0.0
    expected_tools: list[str] = field(default_factory=list)
    actual_tools: list[str] = field(default_factory=list)
    # 检索
    # retrieval_evaluated 用于区分「检索了但没命中」与「本就无需检索」：
    # 只看 retrieval_hit 会把两者混为一谈，从而把检索准确率算低。
    retrieval_evaluated: bool = False
    retrieval_hit: bool = False
    retrieval_mrr: float = 0.0
    # 答案
    keyword_hit_rate: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    # 效率
    latency_ms: float = 0.0
    steps: int = 0
    total_tokens: int = 0
    error: str = ""
    answer_preview: str = ""

    @property
    def efficiency_ok(self) -> bool:
        """是否在步数预算内完成（用于发现「过度调用工具」）。"""
        return self.steps <= 6

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "success": self.success,
            "tool_precision": round(self.tool_precision, 4),
            "tool_recall": round(self.tool_recall, 4),
            "tool_f1": round(self.tool_f1, 4),
            "expected_tools": self.expected_tools,
            "actual_tools": self.actual_tools,
            "retrieval_evaluated": self.retrieval_evaluated,
            "retrieval_hit": self.retrieval_hit,
            "retrieval_mrr": round(self.retrieval_mrr, 4),
            "keyword_hit_rate": round(self.keyword_hit_rate, 4),
            "matched_keywords": self.matched_keywords,
            "latency_ms": self.latency_ms,
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "error": self.error,
            "answer_preview": self.answer_preview,
        }


def _f1(expected: Sequence[str], actual: Sequence[str]) -> tuple[float, float, float]:
    """计算工具调用的 precision / recall / F1（集合语义，忽略重复与顺序）。"""
    exp, act = set(expected), set(actual)
    if not exp and not act:
        return 1.0, 1.0, 1.0
    if not exp:
        # 不期望调用任何工具却调用了 -> 精确率惩罚
        return 0.0, 1.0, 0.0
    if not act:
        return 0.0, 0.0, 0.0
    tp = len(exp & act)
    precision = tp / len(act)
    recall = tp / len(exp)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _retrieval_score(expected_sources: Sequence[str], retrieved: Sequence[dict[str, Any]]) -> tuple[bool, float]:
    """检索命中判定 + MRR。

    ``retrieved`` 为按排名排序的检索结果，每项含 ``title`` / ``source`` / ``text``。
    """
    if not expected_sources:
        return False, 0.0
    needles = [s.lower() for s in expected_sources]
    for rank, item in enumerate(retrieved, start=1):
        # 检索结果的实际结构是 {rank, text, score, metadata:{title, source, ...}}，
        # source/title 在 metadata 里而非顶层。此前只在顶层找，永远取到空串，
        # 导致检索准确率恒为 0——是**度量写错**，不是检索不行。
        fields = dict(item)
        meta = item.get("metadata")
        if isinstance(meta, dict):
            for key, value in meta.items():
                fields.setdefault(key, value)
        haystack = " ".join(
            str(fields.get(k, "")) for k in ("title", "source", "text", "paper_title")
        ).lower()
        if any(n in haystack for n in needles):
            return True, 1.0 / rank
    return False, 0.0


def score_case(case: Any, result: Any, retrieved: Sequence[dict[str, Any]] | None = None) -> CaseScore:
    """对单条用例打分。"""
    score = CaseScore(
        case_id=case.case_id,
        category=case.category,
        expected_tools=list(case.expected_tools),
        actual_tools=list(getattr(result, "tools_used", []) or []),
    )

    # ---- 工具调用准确率 ----
    score.tool_precision, score.tool_recall, score.tool_f1 = _f1(score.expected_tools, score.actual_tools)

    # ---- 检索准确率 ----
    if case.expected_sources and retrieved:
        score.retrieval_evaluated = True
        score.retrieval_hit, score.retrieval_mrr = _retrieval_score(case.expected_sources, retrieved)

    # ---- 答案关键词命中 ----
    answer = (getattr(result, "answer", "") or "").lower()
    if case.expected_keywords:
        matched = [kw for kw in case.expected_keywords if kw.lower() in answer]
        score.matched_keywords = matched
        score.keyword_hit_rate = len(matched) / len(case.expected_keywords)
    else:
        # 无关键词要求时，只要有实质回答即算命中
        score.keyword_hit_rate = 1.0 if len(answer.strip()) > 20 else 0.0

    # ---- 效率 ----
    score.latency_ms = float(getattr(result, "latency_ms", 0.0) or 0.0)
    score.steps = len(getattr(result, "steps", []) or [])
    score.total_tokens = int(getattr(result, "total_tokens", 0) or 0)
    score.error = getattr(result, "error", "") or ""
    score.answer_preview = (getattr(result, "answer", "") or "")[:200]

    # ---- 任务成功判定 ----
    score.success = bool(
        getattr(result, "success", False)
        and not score.error
        and score.keyword_hit_rate >= SUCCESS_KEYWORD_THRESHOLD
    )
    return score


@dataclass
class EvalReport:
    """整体评测报告。"""

    num_cases: int = 0
    task_success_rate: float = 0.0
    tool_calling_accuracy: float = 0.0
    tool_precision: float = 0.0
    tool_recall: float = 0.0
    retrieval_accuracy: float = 0.0
    retrieval_mrr: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    avg_steps: float = 0.0
    avg_tokens: float = 0.0
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    cases: list[CaseScore] = field(default_factory=list)
    failed_cases: list[str] = field(default_factory=list)

    def as_dict(self, include_cases: bool = True) -> dict[str, Any]:
        payload = {
            "num_cases": self.num_cases,
            "task_success_rate": round(self.task_success_rate, 4),
            "tool_calling_accuracy": round(self.tool_calling_accuracy, 4),
            "tool_precision": round(self.tool_precision, 4),
            "tool_recall": round(self.tool_recall, 4),
            "retrieval_accuracy": round(self.retrieval_accuracy, 4),
            "retrieval_mrr": round(self.retrieval_mrr, 4),
            "latency_mean_ms": round(self.latency_mean_ms, 2),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "avg_steps": round(self.avg_steps, 2),
            "avg_tokens": round(self.avg_tokens, 1),
            "by_category": self.by_category,
            "failed_cases": self.failed_cases,
        }
        if include_cases:
            payload["cases"] = [c.as_dict() for c in self.cases]
        return payload

    def to_markdown(self) -> str:
        """生成可放进 README / 述职材料的 Markdown 报告。"""
        lines = [
            "# ResearchAgent 评测报告",
            "",
            f"- 用例总数：**{self.num_cases}**",
            f"- 任务成功率 (Task Success Rate)：**{self.task_success_rate:.1%}**",
            f"- 工具调用准确率 (Tool Calling F1)：**{self.tool_calling_accuracy:.1%}**"
            f"（Precision {self.tool_precision:.1%} / Recall {self.tool_recall:.1%}）",
            f"- 检索准确率 (Retrieval Accuracy)：**{self.retrieval_accuracy:.1%}**"
            f"（MRR {self.retrieval_mrr:.3f}）",
            f"- 端到端延迟：**{self.latency_mean_ms:.0f} ms**"
            f"（P50 {self.latency_p50_ms:.0f} ms / P95 {self.latency_p95_ms:.0f} ms）",
            f"- 平均步数：{self.avg_steps:.2f}　平均 Token：{self.avg_tokens:.0f}",
            "",
            "## 分类表现",
            "",
            "| 类别 | 用例数 | 成功率 | 工具 F1 | 平均延迟(ms) |",
            "|------|-------:|-------:|--------:|-------------:|",
        ]
        for name, stat in sorted(self.by_category.items()):
            lines.append(
                f"| {name} | {stat['count']} | {stat['success_rate']:.1%} | "
                f"{stat['tool_f1']:.1%} | {stat['latency_ms']:.0f} |"
            )

        lines += [
            "",
            "## 用例明细",
            "",
            "| 用例 | 类别 | 成功 | 工具 F1 | 期望工具 | 实际工具 | 延迟(ms) |",
            "|------|------|:----:|--------:|----------|----------|---------:|",
        ]
        for case in self.cases:
            lines.append(
                f"| {case.case_id} | {case.category} | {'✅' if case.success else '❌'} | "
                f"{case.tool_f1:.2f} | {', '.join(case.expected_tools) or '-'} | "
                f"{', '.join(case.actual_tools) or '-'} | {case.latency_ms:.0f} |"
            )

        if self.failed_cases:
            lines += ["", "## 失败用例", ""]
            for case in self.cases:
                if not case.success:
                    reason = case.error or f"关键词命中率 {case.keyword_hit_rate:.0%}"
                    lines.append(f"- **{case.case_id}**（{case.category}）：{reason}")
        return "\n".join(lines)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[k]


def aggregate(scores: Iterable[CaseScore]) -> EvalReport:
    """把单条评分汇总成整体报告。"""
    cases = list(scores)
    if not cases:
        return EvalReport()

    latencies = [c.latency_ms for c in cases]
    # 只统计真正做过检索的用例。这里曾误用 CaseScore 上不存在的
    # expected_sources 字段，导致整轮评测跑完后在聚合阶段崩溃、报告全丢。
    retrieval_cases = [c for c in cases if c.retrieval_evaluated]
    # 同理：没有期望工具集的用例（如 robustness）不计入工具 F1 均值。
    # 「未指定期望」不等于「不该调用任何工具」——面对不存在的论文，
    # Agent 去检索一次恰恰是正确行为，按 F1=0 计入会错误地拉低整体指标。
    tool_cases = [c for c in cases if c.expected_tools]

    by_category: dict[str, dict[str, Any]] = {}
    for case in cases:
        stat = by_category.setdefault(case.category, {"count": 0, "success": 0, "tool_f1": 0.0, "latency_ms": 0.0})
        stat["count"] += 1
        stat["success"] += int(case.success)
        stat["tool_f1"] += case.tool_f1
        stat["latency_ms"] += case.latency_ms
    for stat in by_category.values():
        stat["success_rate"] = stat["success"] / stat["count"]
        stat["tool_f1"] = stat["tool_f1"] / stat["count"]
        stat["latency_ms"] = stat["latency_ms"] / stat["count"]

    return EvalReport(
        num_cases=len(cases),
        task_success_rate=sum(c.success for c in cases) / len(cases),
        tool_calling_accuracy=mean([c.tool_f1 for c in tool_cases]) if tool_cases else 0.0,
        tool_precision=mean([c.tool_precision for c in tool_cases]) if tool_cases else 0.0,
        tool_recall=mean([c.tool_recall for c in tool_cases]) if tool_cases else 0.0,
        retrieval_accuracy=(sum(c.retrieval_hit for c in retrieval_cases) / len(retrieval_cases)) if retrieval_cases else 0.0,
        retrieval_mrr=mean([c.retrieval_mrr for c in retrieval_cases]) if retrieval_cases else 0.0,
        latency_mean_ms=mean(latencies),
        latency_p50_ms=median(latencies),
        latency_p95_ms=_percentile(latencies, 95),
        avg_steps=mean([c.steps for c in cases]),
        avg_tokens=mean([c.total_tokens for c in cases]),
        by_category=by_category,
        cases=cases,
        failed_cases=[c.case_id for c in cases if not c.success],
    )
