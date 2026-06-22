"""
critic.py — Critic Agent Node.

RESPONSIBILITY:
  Reviews all sub-question answers for quality.
  Checks: completeness, consistency, evidence grounding, gaps.
  Returns PASS (proceed to synthesizer) or RETRY (back to researcher).
  On RETRY, provides a list of specific gaps for targeted re-research.

WHY A CRITIC:
  Without critique, the synthesizer blindly accepts whatever the
  researcher produces — including hallucinations, contradictions, or
  incomplete answers. The critic acts as a quality gate.

  This is the "self-reflection" pattern in agentic AI systems.
  Systems like OpenAI's o1, AlphaCode, and many production agents
  use a generate->critique->refine loop.

RETRY LOGIC:
  The critic sets retry_count++ on each retry.
  The graph checks retry_count >= MAX_RESEARCH_ITERATIONS to prevent
  infinite loops (the critic could otherwise keep demanding retries).

INTERVIEW: "The critic implements the generate->evaluate->refine loop,
which is the key pattern that separates agentic AI from simple chains.
It also catches common RAG failures: hallucination (answer not grounded
in sources), incompleteness (sub-question not actually answered), and
contradiction (two sources say opposite things)."

COMMON MISTAKE: Letting the critic run indefinitely — always cap retries
(MAX_RESEARCH_ITERATIONS). Without a cap, a failure mode is an infinite
loop where the critic always finds something to complain about.

PROMPT DESIGN:
  The critic gets: original query + each sub-question + its answer + sources.
  It must justify RETRY with specific gap descriptions.
  Temperature=0 for consistent evaluation.
"""

import json
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import get_settings
from backend.agents.state import ResearchState
from backend.models.schemas import CritiqueResult, ResearchStatus

logger = logging.getLogger(__name__)

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a critical research reviewer. Evaluate the quality of research
conducted on a query.

Check for:
1. COMPLETENESS: Are all sub-questions answered with sufficient detail?
2. GROUNDING: Are answers supported by retrieved sources? (no hallucinations)
3. CONSISTENCY: Do answers contradict each other?
4. GAPS: Are there important aspects of the main query not covered?

Respond with ONLY valid JSON:
{{
  "passed": true/false,
  "feedback": "Overall assessment in 2-3 sentences",
  "gaps": ["specific gap 1", "specific gap 2"]  // empty list if passed=true
}}

Be strict but fair. Only PASS if the research genuinely covers the query well."""),
    ("human", """Main query: {query}

Research conducted:
{research_summary}

Evaluate this research:"""),
])


def _build_research_summary(state: ResearchState) -> str:
    """Format sub-questions and answers for the critic prompt."""
    parts = []
    for i, sq in enumerate(state.get("sub_questions", []), 1):
        source_titles = [s.title or s.source or "unknown" for s in sq.sources[:3]]
        parts.append(
            f"Sub-question {i}: {sq.question}\n"
            f"Answer: {sq.answer[:500] if sq.answer else 'NOT ANSWERED'}...\n"
            f"Sources used: {', '.join(source_titles) if source_titles else 'none'}"
        )
    return "\n\n".join(parts)


def critic_node(state: ResearchState) -> dict:
    """
    LangGraph node: evaluate research quality, decide PASS or RETRY.

    Input state fields used:  query, sub_questions, retry_count
    Output state fields set:  critique_result, retry_count, status,
                              current_question_idx (reset on retry)
    """
    settings = get_settings()
    retry_count = state.get("retry_count", 0)

    # Hard stop after max iterations to prevent infinite loops
    if retry_count >= settings.max_research_iterations:
        logger.warning("Max retry iterations (%d) reached, forcing PASS", retry_count)
        critique = CritiqueResult(
            passed=True,
            feedback=f"Forced pass after {retry_count} iterations.",
            retry_count=retry_count,
        )
        return {
            "critique_result": critique,
            "status": ResearchStatus.SYNTHESIS,
        }

    research_summary = _build_research_summary(state)
    logger.info("Critic evaluating research (retry_count=%d)", retry_count)

    try:
        llm = ChatGroq(
            model=settings.groq_model_name,
            temperature=0.0,
            api_key=settings.groq_api_key,
        )
        chain = CRITIC_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({
            "query": state["query"],
            "research_summary": research_summary,
        })
        raw = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)

        critique = CritiqueResult(
            passed=bool(data.get("passed", False)),
            feedback=data.get("feedback", ""),
            gaps=data.get("gaps", []),
            retry_count=retry_count,
        )

    except Exception as e:
        logger.error("Critic LLM call failed: %s", e)
        # On failure, pass anyway to avoid blocking the pipeline
        critique = CritiqueResult(
            passed=True,
            feedback=f"Critic evaluation failed ({e}), proceeding.",
            retry_count=retry_count,
        )

    if critique.passed:
        logger.info("Critic PASSED — proceeding to synthesis")
        return {
            "critique_result": critique,
            "status": ResearchStatus.SYNTHESIS,
        }
    else:
        logger.info("Critic RETRY — gaps: %s", critique.gaps)
        # Add gap questions as new sub-questions for re-research
        from backend.models.schemas import SubQuestion
        gap_questions = [SubQuestion(question=gap) for gap in critique.gaps[:2]]  # max 2 gap questions
        updated_sub_questions = list(state["sub_questions"]) + gap_questions

        return {
            "critique_result": critique,
            "sub_questions": updated_sub_questions,
            "current_question_idx": len(state["sub_questions"]),  # start from the new gap questions
            "retry_count": retry_count + 1,
            "status": ResearchStatus.RESEARCH,
        }


def should_continue(state: ResearchState) -> str:
    """
    Conditional edge function for LangGraph.
    Called after critic_node to decide next node.

    Returns: "researcher" to retry, "synthesizer" to proceed.
    """
    status = state.get("status", "")
    if status == ResearchStatus.SYNTHESIS:
        return "synthesizer"
    else:
        return "researcher"
