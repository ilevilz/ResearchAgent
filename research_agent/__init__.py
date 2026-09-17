"""ResearchAgent：基于 ReAct + RAG + MCP 的科研智能分析 Agent。

模块划分：
    core/       LLM 客户端与底层基础设施
    rag/        PDF 解析、文本切分、Embedding、向量索引与 Top-K 检索
    tools/      原子工具（论文搜索 / PDF 解析 / 实验指标提取 / 数据分析）
    mcp_server/ Research MCP Server（工具的标准化封装，stdio 协议）
    mcp_client/ MCP 客户端（将 MCP 工具适配为 Agent 可调用的工具）
    agent/      ReAct 工作流（Thought -> Action -> Observation 循环）
    memory/     多轮会话记忆
    eval/       Agent Evaluation 测试集与四维指标
    api/        FastAPI 服务与 Web Demo
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
