"""数据分析工具：对 CSV / 内联表格做描述性统计与分组聚合。

面向「实验数据处理」场景：论文附录里的结果表、自建实验的日志表，
都可以直接丢进来做统计，再由 LLM 归纳结论。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import get_logger, safe_json_loads, truncate
from .base import Tool, ToolError, tool

logger = get_logger(__name__)

_MAX_ROWS = 20000


def _load_frame(data_path: str | None, rows: list[dict[str, Any]] | None, columns: list[str] | None):
    """把输入统一转成 DataFrame。"""
    import pandas as pd

    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise ToolError(f"文件不存在：{path}")
        if path.suffix.lower() in {".csv", ".txt"}:
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        elif path.suffix.lower() == ".json":
            frame = pd.read_json(path)
        else:
            raise ToolError(f"不支持的文件类型：{path.suffix}")
    elif rows:
        frame = pd.DataFrame(rows)
    else:
        raise ToolError("必须提供 data_path 或 rows 之一")

    if len(frame) > _MAX_ROWS:
        logger.warning("数据行数 %d 超过上限，截断为 %d 行", len(frame), _MAX_ROWS)
        frame = frame.head(_MAX_ROWS)

    if columns:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ToolError(f"列不存在：{missing}；可用列：{list(frame.columns)}")
        frame = frame[columns]

    return frame


def analyze_data_impl(
    data_path: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    columns: list[str] | None = None,
    group_by: str | None = None,
    operation: str = "summary",
) -> dict[str, Any]:
    """描述性统计 / 分组聚合 / 相关性分析。"""
    import pandas as pd

    frame = _load_frame(data_path, rows, columns)

    numeric = frame.select_dtypes(include="number")
    base: dict[str, Any] = {
        "num_rows": int(len(frame)),
        "num_columns": int(len(frame.columns)),
        "columns": list(frame.columns),
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
        "missing": {str(k): int(v) for k, v in frame.isna().sum().items() if int(v) > 0},
    }

    if operation == "correlation":
        if numeric.shape[1] < 2:
            raise ToolError("相关性分析至少需要 2 个数值列")
        corr = numeric.corr(numeric_only=True).round(4)
        base["correlation"] = safe_json_loads(corr.to_json(force_ascii=False), default={})
        return base

    if operation == "groupby":
        if not group_by:
            raise ToolError("groupby 操作需要指定 group_by 列")
        if group_by not in frame.columns:
            raise ToolError(f"分组列不存在：{group_by}")
        if numeric.empty:
            counts = frame.groupby(group_by).size().sort_values(ascending=False)
            base["group_counts"] = {str(k): int(v) for k, v in counts.items()}
            return base
        agg = frame.groupby(group_by)[list(numeric.columns)].agg(["mean", "std", "min", "max", "count"]).round(4)
        flat: dict[str, Any] = {}
        for col in numeric.columns:
            flat[str(col)] = {
                stat: {str(k): (None if pd.isna(v) else round(float(v), 4)) for k, v in agg[col][stat].items()}
                for stat in ("mean", "std", "min", "max", "count")
            }
        base["group_by"] = group_by
        base["groups"] = sorted(frame[group_by].dropna().astype(str).unique().tolist())[:50]
        base["aggregations"] = flat
        return base

    # 默认：描述性统计
    if not numeric.empty:
        desc = numeric.describe().round(4)
        base["describe"] = {
            str(col): {str(k): (None if pd.isna(v) else round(float(v), 4)) for k, v in desc[col].items()}
            for col in desc.columns
        }
        base["numeric_columns"] = list(numeric.columns)
    else:
        base["numeric_columns"] = []
        base["note"] = "数据中没有数值列，仅返回结构信息"

    return base


def summarize_table_impl(data_path: str | None = None, rows: list[dict[str, Any]] | None = None, max_rows: int = 10) -> dict[str, Any]:
    """快速预览表格内容（前 N 行 + 列信息），便于 LLM 判断数据结构。"""
    frame = _load_frame(data_path, rows, None)
    preview = frame.head(max_rows).fillna("").astype(str)
    return {
        "num_rows": int(len(frame)),
        "columns": list(frame.columns),
        "preview": preview.to_dict(orient="records"),
        "markdown": truncate(frame.head(max_rows).to_markdown(index=False), 3000) if len(frame) else "",
    }


def compute_statistics_impl(values: list[float]) -> dict[str, Any]:
    """对一组数值做统计。"""
    import numpy as np

    if not values:
        raise ToolError("数值列表为空")
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0,
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "median": round(float(np.median(arr)), 4),
        "q1": round(float(np.percentile(arr, 25)), 4),
        "q3": round(float(np.percentile(arr, 75)), 4),
    }


@tool(
    name="analyze_data",
    description=(
        "对实验数据做统计分析，支持三种操作：summary（描述性统计）、"
        "groupby（按列分组聚合）、correlation（相关性矩阵）。"
        "数据可来自 CSV/Excel/JSON 文件或直接传入行列表。"
    ),
    parameters={
        "data_path": {"type": "string", "description": "数据文件路径（.csv/.xlsx/.json），与 rows 二选一"},
        "rows": {"type": "array", "description": "内联数据行，每行为一个对象", "items": {"type": "object"}},
        "columns": {"type": "array", "description": "只分析指定列", "items": {"type": "string"}},
        "group_by": {"type": "string", "description": "groupby 操作时的分组列名"},
        "operation": {
            "type": "string",
            "description": "操作类型，默认 summary",
            "enum": ["summary", "groupby", "correlation"],
        },
    },
    tags=["analysis", "pandas"],
)
def analyze_data(
    data_path: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    columns: list[str] | None = None,
    group_by: str | None = None,
    operation: str = "summary",
) -> dict[str, Any]:
    """对数据做统计分析。"""
    return analyze_data_impl(data_path, rows, columns, group_by, operation)


@tool(
    name="preview_table",
    description="预览数据文件的前若干行与列结构，用于在分析前确认数据格式。",
    parameters={
        "data_path": {"type": "string", "description": "数据文件路径（.csv/.xlsx/.json）"},
        "rows": {"type": "array", "description": "内联数据行，与 data_path 二选一", "items": {"type": "object"}},
        "max_rows": {"type": "integer", "description": "预览行数，默认 10"},
    },
    tags=["analysis"],
)
def preview_table(
    data_path: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    max_rows: int = 10,
) -> dict[str, Any]:
    """预览表格数据。"""
    return summarize_table_impl(data_path, rows, max_rows)


@tool(
    name="compute_statistics",
    description="对一组数值做快速统计（均值、标准差、分位数、极值）。",
    parameters={
        "values": {"type": "array", "description": "数值列表", "items": {"type": "number"}},
    },
    tags=["analysis"],
)
def compute_statistics(values: list[float]) -> dict[str, Any]:
    """计算一组数值的统计量。"""
    return compute_statistics_impl(values)


def build_analysis_tools() -> list[Tool]:
    return [analyze_data, preview_table, compute_statistics]
