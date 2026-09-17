"""ResearchAgent 命令行入口。

用法示例::

    python -m research_agent ask "找几篇 RAG 的论文"      # 单次任务
    python -m research_agent chat                        # 交互式多轮
    python -m research_agent index --query "RAG" -n 5    # 建库
    python -m research_agent search "attention"          # 只测检索
    python -m research_agent eval --category search      # 跑评测
    python -m research_agent tools                       # 查看工具
    python -m research_agent serve                       # 启动 API
    python -m research_agent mcp                         # 以 MCP Server 启动
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from .config import settings
from .utils import dumps, setup_logging, truncate

_BANNER = r"""
  ____                                _     _                    _
 |  _ \ ___  ___  ___  __ _ _ __ ___| |__ | |    __ _  __ _  ___| |_
 | |_) / _ \/ __|/ _ \/ _` | '__/ __| '_ \| |   / _` |/ _` |/ _ \ __|
 |  _ <  __/\__ \  __/ (_| | | | (__| | | | |__| (_| | (_| |  __/ |_
 |_| \_\___||___/\___|\__,_|_|  \___|_| |_|_____\__,_|\__, |\___|\__|
                                                      |___/
"""


def _print_result(result: Any, show_trace: bool = True) -> None:
    """终端里漂亮地打印一次 Agent 结果。"""
    print("\n" + "=" * 72)
    if show_trace and getattr(result, "steps", None):
        print("🔍 推理轨迹 (ReAct Trace)")
        print("=" * 72)
        for step in result.steps:
            icon = "✅" if step.ok else "❌"
            print(f"\n[Step {step.index}] {icon} Action: {step.action}")
            if step.thought:
                print(f"  💭 Thought    : {truncate(step.thought, 300)}")
            print(f"  📥 Input      : {truncate(dumps(step.action_input), 300)}")
            print(f"  📤 Observation: {truncate(step.observation, 400)}")
    print("\n" + "=" * 72)
    print("📄 最终回答")
    print("=" * 72)
    print(result.answer or "(空)")
    print("\n" + "-" * 72)
    print(
        f"状态：{'成功' if result.success else '失败'} | "
        f"步数：{len(result.steps)} | LLM 调用：{result.llm_calls} | "
        f"Token：{result.total_tokens} | 耗时：{result.latency_ms:.0f}ms"
    )
    if result.error:
        print(f"错误：{result.error}")
    print("-" * 72 + "\n")


# ---------------------------------------------------------------------- #
# 子命令
# ---------------------------------------------------------------------- #
def cmd_ask(args: argparse.Namespace) -> int:
    from .agent.react_agent import ReActAgent

    agent = ReActAgent(max_steps=args.max_steps, verbose=args.verbose)
    result = agent.run(args.query)
    _print_result(result, show_trace=not args.no_trace)
    return 0 if result.success else 1


def cmd_chat(args: argparse.Namespace) -> int:
    from .agent.react_agent import ReActAgent
    from .memory.session import get_memory_store

    session_id = args.session or "cli-session"
    memory = get_memory_store().get(session_id)
    agent = ReActAgent(max_steps=args.max_steps, memory=memory, verbose=args.verbose)

    print(_BANNER)
    print(f"ResearchAgent 交互模式（session={session_id}）")
    print("输入问题开始对话；命令：/reset 清空记忆，/tools 查看工具，/exit 退出\n")

    while True:
        try:
            query = input("👤 你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0

        if not query:
            continue
        if query in {"/exit", "/quit", "exit", "quit"}:
            print("再见。")
            return 0
        if query == "/reset":
            memory.reset()
            print("✅ 会话记忆已清空\n")
            continue
        if query == "/tools":
            for item in agent.available_tools():
                print(f"  - {item['name']}: {truncate(item['description'], 80)}")
            print()
            continue

        result = agent.run(query)
        _print_result(result, show_trace=not args.no_trace)


def cmd_index(args: argparse.Namespace) -> int:
    from .api.service import get_service

    service = get_service()
    outcome = service.index(
        source_query=args.query,
        pdf_paths=args.pdf or None,
        max_results=args.num,
        save=True,
    )
    print(f"\n✅ 入库完成：新增 {outcome['added_chunks']} 个文本块")
    print(dumps(outcome["index_stats"], indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .api.service import get_service

    results = get_service().search(args.query, top_k=args.top_k)
    if not results:
        print("知识库为空或没有命中。先执行：python -m research_agent index --query \"...\"")
        return 1
    print(f"\n检索到 {len(results)} 条结果：\n")
    for item in results:
        meta = item["metadata"]
        # rank 由 Retriever 统一赋值为 1 起的名次，此处不再 +1
        print(f"[{item['rank']}] score={item['score']:.4f} | {meta.get('title', 'unknown')}")
        print(f"    {truncate(item['text'], 200)}\n")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval.runner import run_evaluation

    report = run_evaluation(case_ids=args.case, categories=args.category, verbose=True)
    print("\n" + report.to_markdown())
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    from .tools.registry import get_default_registry

    registry = get_default_registry()
    print(f"\n已注册 {len(registry)} 个工具：\n")
    for item in registry.tools():
        print(f"■ {item.name}")
        print(f"  {truncate(item.description, 160)}")
        params = item.parameters.get("properties", {})
        if params:
            print(f"  参数：{', '.join(params)}")
        print()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.server import app

    print(f"启动 API 服务：http://127.0.0.1:{args.port}  （文档 /docs）")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server.server import main as mcp_main

    mcp_main()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """环境自检：依赖 / Key / 索引 / 工具 / LLM 连通性。"""
    import importlib
    import importlib.metadata as md

    print(_BANNER)
    print("环境自检\n" + "=" * 60)

    print("\n[1] 依赖")
    for name in ("numpy", "pandas", "pypdf", "openai", "httpx", "fastapi", "mcp", "faiss"):
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            print(f"  ⚠️  {name:<12} 未安装（部分功能会降级）")
            continue
        # 并非所有包都暴露 __version__（如 mcp 2.x），回退查发行版元数据
        version = getattr(mod, "__version__", "") or ""
        if not version:
            try:
                version = md.version(name)
            except Exception:  # noqa: BLE001 - 元数据缺失不影响自检
                version = ""
        print(f"  ✅ {name:<12} {version}")

    print("\n[2] DeepSeek 配置")
    print(f"  {'✅' if settings.has_llm_key else '❌'} API Key：{'已配置' if settings.has_llm_key else '未配置'}")
    print(f"  ℹ️  模型：{settings.deepseek_model}")

    print("\n[2.1] 论文检索源")
    print(f"  {'✅' if settings.has_s2_key else '⚠️ '} Semantic Scholar Key："
          f"{'已配置' if settings.has_s2_key else '未配置（将大概率 429，检索降级为离线库）'}")
    print("  ℹ️  arXiv：超频会封禁来源 IP，本模块已内置 3s 限速以避免触发")

    print("\n[3] 知识库")
    from .rag.knowledge_base import get_knowledge_base

    stats = get_knowledge_base().stats()
    print(f"  文本块：{stats['num_chunks']}　向量后端：{stats['vector_backend']}　Embedding：{stats['embedder']}")

    print("\n[4] 工具")
    from .tools.registry import get_default_registry

    print(f"  已注册 {len(get_default_registry())} 个：{', '.join(get_default_registry().names())}")

    print("\n[5] LLM 连通性")
    if settings.has_llm_key:
        from .core.llm_client import get_llm_client

        check = get_llm_client().health_check()
        print(f"  {'✅' if check.get('ok') else '❌'} {check}")
    else:
        print("  ⏭️  跳过（未配置 Key）")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent", description="ResearchAgent：基于 ReAct + RAG + MCP 的科研智能分析 Agent")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="执行单次任务")
    p_ask.add_argument("query", help="任务描述")
    p_ask.add_argument("--max-steps", type=int, default=None)
    p_ask.add_argument("--no-trace", action="store_true", help="不打印推理轨迹")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="交互式多轮对话")
    p_chat.add_argument("--session", default=None)
    p_chat.add_argument("--max-steps", type=int, default=None)
    p_chat.add_argument("--no-trace", action="store_true")
    p_chat.set_defaults(func=cmd_chat)

    p_index = sub.add_parser("index", help="建立 / 扩充知识库")
    p_index.add_argument("--query", help="先用该关键词检索 arXiv 再入库")
    p_index.add_argument("--pdf", nargs="*", help="本地 PDF 路径")
    p_index.add_argument("-n", "--num", type=int, default=5, help="检索论文数量")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="知识库检索")
    p_search.add_argument("query")
    p_search.add_argument("-k", "--top-k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="运行 Agent 评测")
    p_eval.add_argument("--case", nargs="*", help="指定用例 ID")
    p_eval.add_argument("--category", nargs="*", help="指定类别：search/rag/experiment/analysis/multi_step/robustness")
    p_eval.set_defaults(func=cmd_eval)

    sub.add_parser("tools", help="列出可用工具").set_defaults(func=cmd_tools)
    sub.add_parser("doctor", help="环境自检").set_defaults(func=cmd_doctor)

    p_serve = sub.add_parser("serve", help="启动 FastAPI 服务")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    sub.add_parser("mcp", help="以 MCP Server 方式启动").set_defaults(func=cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging()
    settings.ensure_dirs()
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ 执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
