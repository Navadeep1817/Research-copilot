"""Tests for retrieval strategies."""
import pytest
from unittest.mock import patch, MagicMock


def test_tokenize():
    from backend.retrieval.bm25_retriever import tokenize
    tokens = tokenize("Hello World, how are you?")
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert "hello" in tokens


def test_bm25_build_and_search():
    from backend.retrieval.bm25_retriever import build_index, bm25_search
    texts = [
        "The transformer architecture uses self-attention mechanisms.",
        "RAG systems combine retrieval with language model generation.",
        "Vector databases store embeddings for similarity search.",
    ]
    build_index(texts, [{"source": "test"}] * len(texts))
    results = bm25_search("transformer attention", top_k=2)
    assert len(results) >= 1
    assert results[0].strategy == "bm25"
    assert results[0].score > 0


def test_rrf_fusion():
    from backend.retrieval.hybrid_retriever import reciprocal_rank_fusion
    from backend.models.schemas import SourceDocument

    list1 = [SourceDocument(content="doc A text here", score=0.9, strategy="dense")]
    list2 = [SourceDocument(content="doc A text here", score=0.5, strategy="bm25")]

    fused = reciprocal_rank_fusion([list1, list2])
    assert len(fused) == 1
    assert fused[0].strategy == "hybrid_rrf"
    assert fused[0].score > 0


def test_recursive_split():
    from backend.ingestion.chunker import recursive_split
    long_text = "word " * 300
    chunks = recursive_split(long_text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 110  # allow small overflow due to overlap


def test_parent_child_split():
    from backend.ingestion.chunker import create_parent_child_chunks
    text = "paragraph " * 100
    parents, children_per_parent = create_parent_child_chunks(text)
    assert len(parents) >= 1
    for children in children_per_parent:
        assert all(len(c) <= 300 for c in children)
