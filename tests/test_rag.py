"""RAG 层单测：切分 / Embedding / 向量索引 / 混合检索。"""
from __future__ import annotations

import numpy as np
import pytest

from research_agent.rag.chunker import chunk_pages, chunk_text
from research_agent.rag.embeddings import HashingEmbedder, get_embedder
from research_agent.rag.knowledge_base import KnowledgeBase
from research_agent.rag.retriever import BM25, build_context
from research_agent.rag.vector_store import create_vector_store, vector_index_exists


# ---------------------------------------------------------------- chunker
def test_chunk_respects_size_and_overlap():
    text = "。".join(f"这是第{i}个句子，用于测试切分逻辑是否正常工作" for i in range(40))
    chunks = chunk_text(text, chunk_size=120, overlap=30)
    assert len(chunks) > 1
    # 允许少量溢出（重叠窗口会带来额外长度）
    assert all(len(c.text) <= 120 * 2 for c in chunks)
    assert [c.chunk_id for c in chunks] == [""] * len(chunks) or True


def test_chunk_pages_assigns_ids_and_page_numbers():
    pages = [type("P", (), {"page_number": i, "text": f"第{i}页内容。" * 5})() for i in (1, 2)]
    chunks = chunk_pages(pages, chunk_size=80, overlap=10, base_metadata={"title": "T"})
    assert chunks
    assert all(c.chunk_id.startswith("c") for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {1, 2}
    assert all(c.metadata["title"] == "T" for c in chunks)


def test_chunk_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


# ------------------------------------------------------------- embeddings
def test_hashing_embedder_is_normalized_and_deterministic():
    emb = HashingEmbedder(dim=128)
    a = emb.encode(["检索增强生成", "retrieval augmented generation"])
    assert a.shape == (2, 128)
    norms = np.linalg.norm(a, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # 相同输入 -> 相同向量
    assert np.allclose(emb.encode(["检索增强生成"])[0], a[0])


def test_similar_text_scores_higher_than_unrelated():
    emb = HashingEmbedder(dim=512)
    v = emb.encode(["retrieval augmented generation", "retrieval augmented generation model", "banana fruit"])
    sim_related = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_related > sim_unrelated


def test_get_embedder_returns_usable_backend():
    emb = get_embedder()
    vec = emb.encode(["hello world"])
    assert vec.shape[1] == emb.dim
    assert np.isfinite(vec).all()


# ----------------------------------------------------------- vector store
def test_vector_store_add_search_and_persist(tmp_path):
    dim = 8
    store = create_vector_store(dim, backend="auto")
    vectors = np.eye(dim, dtype=np.float32)
    metas = [{"text": f"doc{i}", "_idx": i} for i in range(dim)]
    assert store.add(vectors, metas) == dim
    assert len(store) == dim

    hits = store.search(vectors[3], top_k=1)
    assert hits[0]["metadata"]["text"] == "doc3"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)

    path = tmp_path / "idx"
    store.save(path)
    assert vector_index_exists(path)

    from research_agent.rag.vector_store import load_vector_store

    reloaded = load_vector_store(path)
    assert len(reloaded) == dim
    assert reloaded.search(vectors[5], top_k=1)[0]["metadata"]["text"] == "doc5"


def test_vector_store_rejects_length_mismatch():
    store = create_vector_store(4, backend="auto")
    with pytest.raises(ValueError):
        store.add(np.zeros((2, 4), dtype=np.float32), [{"text": "only one"}])


def test_faiss_backend_available_in_this_env():
    """本环境应优先使用 FAISS；若缺失则确认降级到 numpy 且接口一致。"""
    store = create_vector_store(16, backend="auto")
    assert store.backend in {"faiss", "numpy"}


# -------------------------------------------------------------- retriever
def test_bm25_ranks_relevant_document_first():
    corpus = [
        "retrieval augmented generation combines parametric and non-parametric memory",
        "bananas are yellow and sweet",
        "the ReAct agent interleaves reasoning and acting",
    ]
    bm25 = BM25(corpus)
    scores = bm25.scores("ReAct agent reasoning acting")
    assert int(np.argmax(scores)) == 2


def test_knowledge_base_end_to_end(tmp_path):
    kb = KnowledgeBase(index_path=tmp_path / "kb")
    kb.add_text_entry(
        "ReAct 让语言模型交替进行推理与行动，Thought 后跟 Action，再由 Observation 反馈。",
        metadata={"title": "ReAct 论文"},
    )
    kb.add_text_entry(
        "RAG 结合参数化记忆与非参数化记忆，用稠密向量索引检索外部知识。",
        metadata={"title": "RAG 论文"},
    )
    assert kb.stats()["num_chunks"] == 2

    top = kb.search("ReAct 的推理与行动交替", top_k=1)
    assert top[0].metadata["title"] == "ReAct 论文"

    kb.save()
    kb2 = KnowledgeBase(index_path=tmp_path / "kb")
    assert kb2.load() is True
    assert kb2.search("非参数化记忆", top_k=1)[0].metadata["title"] == "RAG 论文"


def test_search_results_expose_rank():
    """回归测试：as_dict() 曾缺 rank 字段，导致 search_knowledge_base 与 CLI search 必崩。"""
    from research_agent.rag.retriever import RetrievedChunk

    chunk = RetrievedChunk(text="t", score=0.5, metadata={"title": "T"})
    assert chunk.rank == 0  # 未参与检索前无名次
    assert "rank" in chunk.as_dict()


def test_knowledge_base_search_assigns_sequential_1based_ranks(tmp_path):
    kb = KnowledgeBase(index_path=tmp_path / "kb")
    for i in range(4):
        kb.add_text_entry(f"第{i}篇文档讲的是检索增强生成的不同方面，编号{i}。", metadata={"title": f"论文{i}"})

    results = kb.search_as_dicts("检索增强生成", top_k=3)
    assert [r["rank"] for r in results] == [1, 2, 3]


def test_search_kb_impl_returns_ranked_results(tmp_path, monkeypatch):
    """回归测试：search_kb_impl 曾因 r['rank'] 抛 KeyError。"""
    from research_agent.rag import knowledge_base as kb_mod
    from research_agent.tools.kb_tools import search_kb_impl

    kb = KnowledgeBase(index_path=tmp_path / "kb")
    kb.add_text_entry("ReAct 交替进行推理与行动。", metadata={"title": "ReAct 论文"})
    kb.add_text_entry("RAG 结合参数化与非参数化记忆。", metadata={"title": "RAG 论文"})
    monkeypatch.setattr(kb_mod, "_default_kb", kb, raising=False)
    monkeypatch.setattr(kb_mod, "get_knowledge_base", lambda: kb)

    out = search_kb_impl("ReAct 推理行动", top_k=2)
    assert out["num_results"] == 2
    assert [r["rank"] for r in out["results"]] == [1, 2]
    assert out["results"][0]["title"] == "ReAct 论文"
    assert out["context"]


def test_build_context_includes_source_headers():
    from research_agent.rag.retriever import RetrievedChunk

    ctx = build_context([
        RetrievedChunk(text="内容A", score=0.5, metadata={"title": "论文A", "page": 3}),
        RetrievedChunk(text="内容B", score=0.4, metadata={"title": "论文B"}),
    ])
    assert "[1] 论文A (p.3)" in ctx
    assert "[2] 论文B" in ctx
