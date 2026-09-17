"""评测执行器：跑测试集、算四维指标、落盘报告。

检索指标单独测（不走 Agent）是有意为之：
把「检索组件」从「Agent 决策」中隔离出来，才能判断失败到底是
**检索没召回**还是**Agent 没去检索**——这是定位问题的关键区分。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ..agent.react_agent import ReActAgent
from ..config import settings
from ..utils import get_logger
from .dataset import EvalCase, load_dataset
from .metrics import EvalReport, aggregate, score_case

logger = get_logger(__name__)


@dataclass
class EvalRunner:
    """在测试集上评测 Agent。"""

    agent: ReActAgent | None = None
    dataset: list[EvalCase] = field(default_factory=list)
    verbose: bool = True
    retrieval_top_k: int = 5

    def __post_init__(self) -> None:
        if not self.dataset:
            self.dataset = load_dataset()

    # ------------------------------------------------------------------ #
    def _retrieval_probe(self, case: EvalCase) -> list[dict[str, Any]]:
        """直接对知识库做一次检索，用于独立评估检索组件。"""
        if not case.expected_sources:
            return []
        try:
            from ..rag.knowledge_base import get_knowledge_base

            kb = get_knowledge_base()
            if not kb.stats()["num_chunks"]:
                return []
            return kb.search_as_dicts(case.query, top_k=self.retrieval_top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("检索探针失败（%s）：%s", case.case_id, exc)
            return []

    # ------------------------------------------------------------------ #
    def run(
        self,
        case_ids: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
    ) -> EvalReport:
        """执行评测。"""
        cases = self.dataset
        if case_ids:
            cases = [c for c in cases if c.case_id in set(case_ids)]
        if categories:
            cases = [c for c in cases if c.category in set(categories)]

        if self.agent is None:
            self.agent = ReActAgent(verbose=False)

        scores = []
        started = time.perf_counter()

        for i, case in enumerate(cases, start=1):
            logger.info("[%d/%d] 评测 %s（%s）：%s", i, len(cases), case.case_id, case.category, case.query[:60])
            try:
                result = self.agent.run(case.query, reset_memory=True)
            except Exception as exc:  # noqa: BLE001 - 单条用例异常不应中断整轮评测
                logger.exception("用例 %s 执行异常", case.case_id)

                class _Failed:
                    answer = ""
                    success = False
                    tools_used: list[str] = []
                    latency_ms = 0.0
                    steps: list[Any] = []
                    total_tokens = 0
                    error = f"{type(exc).__name__}: {exc}"

                result = _Failed()

            score = score_case(case, result, self._retrieval_probe(case))
            scores.append(score)
            if self.verbose:
                logger.info(
                    "  -> 成功=%s 工具F1=%.2f 命中率=%.0f%% 延迟=%.0fms",
                    score.success, score.tool_f1, score.keyword_hit_rate * 100, score.latency_ms,
                )

        report = aggregate(scores)
        logger.info(
            "评测完成，用时 %.1fs | 成功率 %.1f%% | 工具F1 %.1f%% | 平均延迟 %.0fms",
            time.perf_counter() - started,
            report.task_success_rate * 100,
            report.tool_calling_accuracy * 100,
            report.latency_mean_ms,
        )
        return report


# ---------------------------------------------------------------------- #
def save_report(report: EvalReport, output_dir: str | Path | None = None) -> dict[str, str]:
    """把报告落盘为 JSON + Markdown，返回两个文件路径。"""
    output_dir = Path(output_dir or (settings.project_root / "eval" / "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = output_dir / f"report-{stamp}.json"
    md_path = output_dir / f"report-{stamp}.md"

    json_path.write_text(
        json.dumps(
            {"generated_at": stamp, "model": settings.deepseek_model, **report.as_dict()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    # 便于 README 引用的固定文件名
    (output_dir / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "latest.md").write_text(report.to_markdown(), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}


def run_evaluation(
    case_ids: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> EvalReport:
    """便捷入口：跑评测并保存报告。"""
    runner = EvalRunner(verbose=verbose)
    report = runner.run(case_ids=case_ids, categories=categories)
    paths = save_report(report, output_dir)
    logger.info("报告已保存：%s", paths["markdown"])
    return report
