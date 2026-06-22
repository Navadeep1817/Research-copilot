"""Tests for evaluation pipeline."""
import pytest


def test_evaluate_research_returns_dict():
    """evaluate_research should always return a dict with 4 keys."""
    from backend.evaluation.ragas_eval import evaluate_research
    # RAGAS may not be available in CI, but function should not raise
    result = evaluate_research(
        question="What is RAG?",
        answer="RAG is retrieval augmented generation.",
        contexts=["RAG combines retrieval with generation."],
    )
    assert isinstance(result, dict)
    assert "faithfulness" in result
    assert "answer_relevance" in result
    assert "context_precision" in result
    assert "context_recall" in result
