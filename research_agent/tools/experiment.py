"""实验分析工具：从论文正文中抽取实验指标，并做跨论文对比。

``extract_metrics`` 采用「LLM 结构化抽取 + 正则兜底」双路设计：
    - 有 LLM 时，用 JSON Schema 约束输出，抽取质量高、可覆盖复杂表述；
    - 无 LLM 或输出不可解析时，用正则匹配 ``Model / Dataset / Metric = Value`` 模式，
      保证工具永远能返回结构，而不是直接失败。

抽取结果统一归一化为 ``{model, dataset, metric, value, unit, source}``，
因此可以直接喂给下游的 ``compare_experiments`` 做横向对比。
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ..core.llm_client import LLMError, get_llm_client
from ..utils import get_logger, safe_json_loads
from .base import Tool, ToolError, tool

logger = get_logger(__name__)

_EXTRACTION_PROMPT = """你是科研论文实验信息的抽取专家。请从下面的论文文本中抽取所有**实验指标**。

要求：
1. 只抽取文本中明确出现的数据，不要推测或补全；
2. 每条记录包含字段：model（方法/模型名）、dataset（数据集名）、metric（指标名，如 accuracy/F1/BLEU/mAP）、value（数值，浮点数）、unit（单位或%，可为空）、context（原文依据，不超过 60 字）；
3. 若某项缺失，用空字符串填充，但 value 必须有意义；
4. 严格输出 JSON 数组，不要输出任何解释文字。

输出格式示例：
[{"model": "BERT", "dataset": "SQuAD", "metric": "F1", "value": 93.2, "unit": "%", "context": "..."}]

论文文本：
---
{text}
---
"""

_METRIC_HINTS = (
    "accuracy", "acc", "f1", "precision", "recall", "bleu", "rouge", "map", "iou",
    "auc", "auc-roc", "psnr", "ssim", "wer", "perplexity", "ppl", "em", "mrr", "ndcg",
    "准确率", "精确率", "召回率", "f1值",
)

# 形如 "BERT on SQuAD: F1 = 93.2" / "准确率 95.3%" / "mAP@50 41.2"
_PATTERN_A = re.compile(
    r"(?P<model>[A-Za-z][\w\-\.\+]{1,40})?\s*(?:on|在)\s*(?P<dataset>[A-Za-z0-9\-_]{2,40})?"
    r"[^\n]{0,20}?(?P<metric>" + "|".join(_METRIC_HINTS) + r")[^\d%]{0,12}(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|)",
    re.IGNORECASE,
)
_PATTERN_B = re.compile(
    r"(?P<metric>" + "|".join(_METRIC_HINTS) + r")[^\d%]{0,12}(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|)",
    re.IGNORECASE,
)


def _regex_extract(text: str, source: str = "") -> list[dict[str, Any]]:
    """正则兜底抽取。"""
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()

    for match in _PATTERN_A.finditer(text or ""):
        value = float(match.group("value"))
        metric = (match.group("metric") or "").lower()
        key = (metric, value)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "model": (match.group("model") or "").strip(),
                "dataset": (match.group("dataset") or "").strip(),
                "metric": metric,
                "value": value,
                "unit": match.group("unit") or "",
                "context": match.group(0).strip()[:60],
                "source": source,
                "extractor": "regex",
            }
        )

    if not records:
        for match in _PATTERN_B.finditer(text or ""):
            value = float(match.group("value"))
            metric = (match.group("metric") or "").lower()
            key = (metric, value)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "model": "",
                    "dataset": "",
                    "metric": metric,
                    "value": value,
                    "unit": match.group("unit") or "",
                    "context": match.group(0).strip()[:60],
                    "source": source,
                    "extractor": "regex",
                }
            )
    return records


def _normalize(records: list[dict[str, Any]], source: str, extractor: str) -> list[dict[str, Any]]:
    """统一字段与类型，过滤无效记录。"""
    out: list[dict[str, Any]] = []
    for item in records or []:
        if not isinstance(item, dict):
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "model": str(item.get("model") or "").strip(),
                "dataset": str(item.get("dataset") or "").strip(),
                "metric": str(item.get("metric") or "").strip().lower(),
                "value": value,
                "unit": str(item.get("unit") or "").strip(),
                "context": str(item.get("context") or "").strip()[:80],
                "source": item.get("source") or source,
                "extractor": extractor,
            }
        )
    return out


def extract_metrics_impl(text: str, source: str = "", use_llm: bool = True) -> dict[str, Any]:
    """从文本中抽取实验指标。"""
    if not (text or "").strip():
        raise ToolError("待抽取的文本为空")

    text = text[:16000]
    metrics: list[dict[str, Any]] = []
    extractor = "regex"

    if use_llm:
        try:
            llm = get_llm_client()
            resp = llm.chat(
                [
                    {"role": "system", "content": "你是严谨的科研数据抽取助手，只输出 JSON。"},
                    {"role": "user", "content": _EXTRACTION_PROMPT.format(text=text)},
                ],
                temperature=0.0,
            )
            parsed = safe_json_loads(resp.content, default=None)
            if isinstance(parsed, dict):
                parsed = parsed.get("metrics") or parsed.get("data") or []
            normalized = _normalize(parsed if isinstance(parsed, list) else [], source, "llm")
            if normalized:
                metrics, extractor = normalized, "llm"
            else:
                logger.warning("LLM 抽取结果为空，转用正则兜底")
        except (LLMError, Exception) as exc:  # noqa: BLE001
            logger.warning("LLM 抽取失败（%s），转用正则兜底", exc)

    if not metrics:
        metrics = _normalize(_regex_extract(text, source), source, "regex")

    return {
        "source": source,
        "extractor": extractor,
        "count": len(metrics),
        "metrics": metrics,
    }


def compare_experiments_impl(records: list[dict[str, Any]], group_by: str = "metric") -> dict[str, Any]:
    """按指定维度分组对比多篇论文 / 多个方法的实验指标。"""
    if not records:
        raise ToolError("没有可对比的实验数据")

    key_field = group_by if group_by in {"metric", "dataset", "model"} else "metric"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        key = str(item.get(key_field) or "unknown").strip().lower()
        grouped[key].append(item)

    comparisons: list[dict[str, Any]] = []
    for key, items in grouped.items():
        values = [float(i.get("value", 0)) for i in items if isinstance(i.get("value"), (int, float))]
        if not values:
            continue
        best = max(items, key=lambda i: float(i.get("value", 0)))
        comparisons.append(
            {
                "group": key,
                "count": len(items),
                "best_value": round(max(values), 4),
                "worst_value": round(min(values), 4),
                "mean": round(sum(values) / len(values), 4),
                "best_model": best.get("model") or best.get("source") or "",
                "best_dataset": best.get("dataset", ""),
                "entries": [
                    {
                        "model": i.get("model", ""),
                        "dataset": i.get("dataset", ""),
                        "value": i.get("value"),
                        "unit": i.get("unit", ""),
                        "source": i.get("source", ""),
                    }
                    for i in sorted(items, key=lambda x: -float(x.get("value", 0)))[:10]
                ],
            }
        )

    comparisons.sort(key=lambda c: -c["count"])
    return {
        "group_by": key_field,
        "num_groups": len(comparisons),
        "total_records": len(records),
        "comparisons": comparisons,
    }


@tool(
    name="extract_metrics",
    description=(
        "从论文文本（全文、摘要或实验结果段落）中抽取实验指标，"
        "返回结构化的 {model, dataset, metric, value} 列表。"
        "适用于需要获取论文具体实验数值、构建对比表格的场景。"
    ),
    parameters={
        "text": {"type": "string", "description": "论文文本内容，可来自 parse_pdf 或 search_knowledge_base"},
        "source": {"type": "string", "description": "来源标识，如论文标题或路径，用于结果溯源"},
        "use_llm": {"type": "boolean", "description": "是否用 LLM 抽取（质量更高），默认 true；失败自动降级正则"},
    },
    tags=["experiment", "extraction"],
)
def extract_metrics(text: str, source: str = "", use_llm: bool = True) -> dict[str, Any]:
    """从论文文本中抽取实验指标。"""
    return extract_metrics_impl(text, source, use_llm)


@tool(
    name="compare_experiments",
    description=(
        "对多组实验指标做横向对比，按 metric / dataset / model 分组，"
        "输出每组的最优值、均值与排名。适用于'哪个方法在某数据集上最好'这类问题。"
    ),
    parameters={
        "records": {
            "type": "array",
            "description": "指标记录列表，通常来自 extract_metrics 的 metrics 字段",
            "items": {"type": "object"},
        },
        "group_by": {
            "type": "string",
            "description": "分组维度，默认 metric",
            "enum": ["metric", "dataset", "model"],
        },
    },
    tags=["experiment", "analysis"],
)
def compare_experiments(records: list[dict[str, Any]], group_by: str = "metric") -> dict[str, Any]:
    """对比多组实验指标。"""
    return compare_experiments_impl(records, group_by)


def build_experiment_tools() -> list[Tool]:
    return [extract_metrics, compare_experiments]
