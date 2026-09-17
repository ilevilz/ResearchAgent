"""Agent Evaluation 测试集。

测试用例按**能力维度**分组，每组的评判标准不同，这样评测结果才能定位问题：

    - ``search``     论文检索：期望调用检索工具，且返回足够数量论文；
    - ``rag``        知识库问答：期望走 RAG 工具，且答案命中关键信息；
    - ``experiment`` 实验分析：期望抽取/对比指标，且数值正确；
    - ``analysis``   数据分析：期望调用统计工具；
    - ``multi_step`` 多步任务：期望组合多个工具（考察任务规划能力）；
    - ``robustness`` 异常与边界：期望优雅失败，不编造。

每个用例声明 ``expected_tools``（用于 Tool Calling Accuracy）与
``expected_keywords``（用于答案正确性的关键词命中判定）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """单条评测用例。"""

    case_id: str
    query: str
    category: str = "general"
    expected_tools: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    # 检索类用例：期望命中的来源关键词（用于 Retrieval Accuracy）
    expected_sources: list[str] = field(default_factory=list)
    max_steps: int = 6
    difficulty: str = "medium"
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "category": self.category,
            "expected_tools": self.expected_tools,
            "expected_keywords": self.expected_keywords,
            "expected_sources": self.expected_sources,
            "difficulty": self.difficulty,
        }


# ---------------------------------------------------------------------- #
# 内置测试集
# ---------------------------------------------------------------------- #
DEFAULT_DATASET: list[EvalCase] = [
    # ---- 论文检索 ----
    EvalCase(
        case_id="search-01",
        query="帮我找几篇关于 retrieval augmented generation 的论文",
        category="search",
        expected_tools=["search_papers"],
        expected_keywords=["检索", "论文", "RAG"],
        expected_sources=["arxiv"],
        difficulty="easy",
    ),
    EvalCase(
        case_id="search-02",
        query="找 3 篇 2023 年以后关于 Agent 工具调用（tool use / function calling）的论文",
        category="search",
        expected_tools=["search_papers"],
        expected_keywords=["工具", "论文"],
        difficulty="medium",
    ),
    EvalCase(
        case_id="search-03",
        query="搜索 chain-of-thought prompting 相关的经典工作",
        category="search",
        expected_tools=["search_papers"],
        expected_keywords=["CoT", "思维链", "chain"],
        difficulty="easy",
    ),
    # ---- 知识库问答（RAG）----
    EvalCase(
        case_id="rag-01",
        query="知识库里的论文主要研究了什么？请总结几条关键信息，并标注来源。",
        category="rag",
        expected_tools=["search_knowledge_base"],
        expected_keywords=["来源"],
        difficulty="medium",
        note="需要先建库：python scripts/build_kb.py",
    ),
    EvalCase(
        case_id="rag-02",
        query="根据知识库内容，这些论文用了哪些数据集或评测基准？",
        category="rag",
        expected_tools=["search_knowledge_base"],
        expected_keywords=[],
        difficulty="medium",
    ),
    # ---- 实验指标 ----
    EvalCase(
        case_id="exp-01",
        query=(
            "从下面这段实验描述中抽取指标："
            "BERT-base 在 SQuAD 上取得 F1 93.2%，在 MNLI 上准确率 86.7%；"
            "RoBERTa-large 在 SQuAD 上 F1 94.6%，MNLI 准确率 90.2%。"
        ),
        category="experiment",
        expected_tools=["extract_metrics"],
        expected_keywords=["93.2", "94.6"],
        difficulty="easy",
    ),
    EvalCase(
        case_id="exp-02",
        query=(
            "对比这些结果，哪个模型在 SQuAD 上 F1 最高："
            "BERT F1 93.2，RoBERTa F1 94.6，DeBERTa F1 95.1？"
        ),
        category="experiment",
        expected_tools=["extract_metrics", "compare_experiments"],
        expected_keywords=["DeBERTa", "95.1"],
        difficulty="medium",
    ),
    # ---- 数据分析 ----
    EvalCase(
        case_id="ana-01",
        query="对这组数值做统计：23.1, 24.5, 22.8, 25.0, 23.9, 24.2",
        category="analysis",
        expected_tools=["compute_statistics"],
        expected_keywords=["均值", "24"],
        difficulty="easy",
    ),
    EvalCase(
        case_id="ana-02",
        query=(
            "分析这组实验数据并给出结论："
            'rows=[{"model":"A","score":81.2},{"model":"A","score":82.0},'
            '{"model":"B","score":85.4},{"model":"B","score":84.9}]，'
            "按 model 分组统计 score 的均值。"
        ),
        category="analysis",
        expected_tools=["analyze_data"],
        expected_keywords=["A", "B"],
        difficulty="medium",
    ),
    # ---- 多步任务（任务规划）----
    EvalCase(
        case_id="multi-01",
        query=(
            "请检索 retrieval augmented generation 相关论文，"
            "把结果写入知识库，然后基于知识库总结这个方向的核心方法。"
        ),
        category="multi_step",
        expected_tools=["search_papers", "index_papers", "search_knowledge_base"],
        expected_keywords=["检索", "RAG"],
        max_steps=8,
        difficulty="hard",
    ),
    EvalCase(
        case_id="multi-02",
        query="找一篇关于 ReAct agent 的论文，获取它的摘要，并总结其核心贡献。",
        category="multi_step",
        expected_tools=["search_papers"],
        expected_keywords=["ReAct"],
        max_steps=8,
        difficulty="medium",
    ),
    # ---- 鲁棒性 ----
    EvalCase(
        case_id="robust-01",
        query="请帮我查询一篇完全不存在的论文，标题是 'ZZZQQQ Nonexistent Paper 9999'。",
        category="robustness",
        expected_tools=[],
        expected_keywords=[],
        difficulty="medium",
        note="应如实说明未找到，而不是编造内容",
    ),
    EvalCase(
        case_id="robust-02",
        query="分析 /nonexistent/path/data.csv 这个文件里的数据。",
        category="robustness",
        expected_tools=[],
        expected_keywords=[],
        difficulty="easy",
        note="应报告文件不存在",
    ),
]


def load_dataset(path: str | Path | None = None) -> list[EvalCase]:
    """从 JSON 文件加载测试集；不传则用内置默认集。"""
    if path is None:
        return list(DEFAULT_DATASET)

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            case_id=item.get("case_id") or f"case-{i:02d}",
            query=item["query"],
            category=item.get("category", "general"),
            expected_tools=item.get("expected_tools", []),
            expected_keywords=item.get("expected_keywords", []),
            expected_sources=item.get("expected_sources", []),
            max_steps=item.get("max_steps", 6),
            difficulty=item.get("difficulty", "medium"),
            note=item.get("note", ""),
        )
        for i, item in enumerate(data, start=1)
    ]
