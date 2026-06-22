"""Tests for LangGraph agent nodes."""
import pytest
from unittest.mock import patch, MagicMock
from backend.models.schemas import SubQuestion, ResearchStatus


def _base_state():
    return {
        "query": "What is RAG and how does it work?",
        "session_id": "test-session",
        "retrieval_strategy": "hybrid",
        "sub_questions": [],
        "research_plan": "",
        "sources": [],
        "current_question_idx": 0,
        "retry_count": 0,
        "final_report": "",
        "critique_result": None,
        "evaluation": None,
        "status": ResearchStatus.PLANNING,
        "error": None,
        "messages": [],
    }


def test_critic_force_pass_after_max_retries():
    """Critic must force PASS when retry_count >= max_iterations."""
    from backend.agents.critic import critic_node
    state = _base_state()
    state["retry_count"] = 999  # exceed max
    state["sub_questions"] = [SubQuestion(question="What is RAG?", answer="RAG is...", answered=True)]

    with patch("backend.agents.critic.get_settings") as mock_settings:
        mock_settings.return_value.max_research_iterations = 3
        mock_settings.return_value.groq_model_name = "llama-3.1-70b-versatile"
        mock_settings.return_value.groq_api_key = "test"
        result = critic_node(state)

    assert result["critique_result"].passed is True
    assert result["status"] == ResearchStatus.SYNTHESIS


def test_should_continue_routing():
    from backend.agents.critic import should_continue
    state_pass = _base_state()
    state_pass["status"] = ResearchStatus.SYNTHESIS
    assert should_continue(state_pass) == "synthesizer"

    state_retry = _base_state()
    state_retry["status"] = ResearchStatus.RESEARCH
    assert should_continue(state_retry) == "researcher"


def test_researcher_router_routing():
    from backend.agents.graph import researcher_router
    state_more = _base_state()
    state_more["sub_questions"] = [SubQuestion(question="Q1"), SubQuestion(question="Q2")]
    state_more["current_question_idx"] = 1
    assert researcher_router(state_more) == "researcher"

    state_done = _base_state()
    state_done["sub_questions"] = [SubQuestion(question="Q1")]
    state_done["current_question_idx"] = 1
    assert researcher_router(state_done) == "critic"
