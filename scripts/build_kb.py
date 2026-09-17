"""构建科研论文知识库。

三种用法::

    # 1) 按关键词检索 arXiv 并入库（最常用，演示前先跑这个）
    python scripts/build_kb.py --query "retrieval augmented generation" -n 8

    # 2) 把本地 PDF 目录全部入库
    python scripts/build_kb.py --pdf-dir data/papers

    # 3) 多个关键词批量建库
    python scripts/build_kb.py --query "ReAct agent" -n 5 --query "chain of thought" -n 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agent.api.service import get_service  # noqa: E402
from research_agent.rag.knowledge_base import get_knowledge_base  # noqa: E402
from research_agent.utils import dumps, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 ResearchAgent 论文知识库")
    parser.add_argument("--query", action="append", default=[], help="检索关键词，可重复传入")
    parser.add_argument("-n", "--num", type=int, default=8, help="每个关键词入库的论文数量")
    parser.add_argument("--pdf-dir", default=None, help="本地 PDF 目录，默认 data/papers")
    parser.add_argument("--no-save", action="store_true", help="只构建不落盘")
    args = parser.parse_args()

    setup_logging()
    service = get_service()
    kb = get_knowledge_base()

    total = 0

    for query in args.query:
        print(f"\n🔍 检索并入库：{query}（{args.num} 篇）")
        outcome = service.index(source_query=query, max_results=args.num, save=False)
        total += outcome["added_chunks"]
        print(f"   ✅ 新增 {outcome['added_chunks']} 块")

    if args.pdf_dir or (not args.query):
        pdf_dir = args.pdf_dir or str(kb.index_path.parent.parent / "papers")
        print(f"\n📂 解析本地 PDF 目录：{pdf_dir}")
        outcome = service.index(pdf_paths=None, save=False)
        # 目录批量入库单独走 loader
        from research_agent.rag.loader import load_documents

        docs = load_documents(pdf_dir)
        for doc in docs:
            total += kb.add_document(doc)
        print(f"   ✅ 解析 {len(docs)} 个文档")

    if total == 0:
        print("\n⚠️  没有任何内容入库。请传入 --query 或把 PDF 放到 data/papers/ 后重试。")
        return 1

    if not args.no_save:
        kb.save()
        print("\n💾 索引已保存")

    print("\n📊 知识库状态：")
    print(dumps(kb.stats(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
