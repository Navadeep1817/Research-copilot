"""
state.py — ResearchState: the shared memory of the LangGraph pipeline.

WHY A TYPED STATE:
  LangGraph passes a single state dict between all agent nodes.
  Every node reads from and writes to this state.
  Using TypedDict gives us IDE autocomplete and type safety across nodes.

STATE DESIGN PRINCIPLES:
  1. Append-only lists (sources, sub_questions) — nodes add, never overwrite
  2. Counters for loop control (retry_count prevents infinite critic loops)
  3. Status string for the frontend streaming view
  4. Error field so the graph can fail gracefully without crashing

DATA FLOW THROUGH STATE:
  START
    -> planner sets: sub_questions, research_plan
    -> researcher appends: sources, fills sub_questions[i].answer
    -> critic sets: critique_result, may increment retry_count
    -> synthesizer sets: final_report
  END

INTERVIEW: "The state is the contract between agents. I used TypedDict
rather than a Pydantic model because LangGraph's reducers (Annotated
fields with add operator) work natively with TypedDict. Each node is a
pure function: (state) -> partial_state_update."

COMMON MISTAKE: Storing mutable objects (like a live DB connection) in
state — state must be serialisable for LangGraph's checkpointing.

REDUCERS: The Annotated[list, operator.add] pattern means when two nodes
return a list for the same field, LangGraph concatenates them instead of
overwriting. Critical for parallel sub-question research.
"""

from __future__ import annotations
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from backend.models.schemas import SubQuestion, SourceDocument, CritiqueResult, EvaluationScores


class ResearchState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────
    query:              str                  # original user query
    session_id:         str
    retrieval_strategy: str                  # which retrieval strategy to use

    # ── Planner output ────────────────────────────────────────────────────
    sub_questions:      list[SubQuestion]    # decomposed research questions
    research_plan:      str                  # planner's high-level plan

    # ── Researcher output ─────────────────────────────────────────────────
    # Annotated with operator.add = LangGraph concatenates lists across nodes
    sources:            Annotated[list[SourceDocument], operator.add]
    current_question_idx: int                # which sub-question we're on

    # ── Critic output ─────────────────────────────────────────────────────
    critique_result:    CritiqueResult | None
    retry_count:        int                  # number of critic-triggered retries

    # ── Synthesizer output ────────────────────────────────────────────────
    final_report:       str

    # ── Evaluation ────────────────────────────────────────────────────────
    evaluation:         EvaluationScores | None

    # ── Control ───────────────────────────────────────────────────────────
    status:             str                  # for streaming to frontend
    error:              str | None           # set on failure
    messages:           list[dict[str, Any]] # conversation history for memory
