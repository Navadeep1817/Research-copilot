"""
graph.py — LangGraph StateGraph definition (the beating heart).

THIS FILE DEFINES:
  - The ResearchState type that flows between nodes
  - All nodes (planner, researcher, critic, synthesizer)
  - All edges (fixed and conditional)
  - The compiled graph that FastAPI invokes

GRAPH STRUCTURE:
  START
    -> planner_node
    -> researcher_node  <─────────────────┐
    -> critic_node                        │
    -> [conditional edge]                 │
         if RETRY:  -> researcher_node ───┘  (loop back with new idx)
         if PASS:   -> synthesizer_node
    -> END

WHY LANGGRAPH OVER LANGCHAIN LCEL:
  LCEL is a linear DAG. Research requires CYCLES (critic -> researcher loop).
  LangGraph supports cycles, state persistence, streaming, and checkpointing.
  It also supports parallel fan-out (future: research sub-questions in parallel).

CONDITIONAL EDGES:
  critic_node returns {"status": "synthesizing"} or {"status": "researching"}.
  should_continue() reads the status and returns the next node name.
  LangGraph uses this string to route the graph.

CHECKPOINTING (production upgrade):
  Replace MemorySaver with SqliteSaver or PostgresSaver for persistent
  checkpointing. This allows resuming interrupted research sessions.

INTERVIEW: "I used LangGraph's StateGraph because the critic->researcher
retry loop cannot be expressed in a linear chain. The graph has a cycle,
and LangGraph is specifically designed for cyclic agent workflows. The
state is a TypedDict so every node has typed access to shared data."

COMMON MISTAKE: Not handling the researcher loop termination condition.
If current_question_idx is never incremented, the loop runs forever.
Our researcher_node increments the index every iteration.

STREAMING:
  graph.stream(state) yields partial state updates after each node.
  FastAPI uses this to stream agent progress to the frontend via WebSocket.
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from backend.agents.state import ResearchState
from backend.agents.planner import planner_node
from backend.agents.researcher import researcher_node
from backend.agents.critic import critic_node, should_continue
from backend.agents.synthesizer import synthesizer_node

logger = logging.getLogger(__name__)


def researcher_router(state: ResearchState) -> str:
    """
    After researcher_node: decide if we should keep researching
    (more sub-questions remain) or move to critique.
    """
    idx = state.get("current_question_idx", 0)
    total = len(state.get("sub_questions", []))

    if idx < total:
        return "researcher"   # more sub-questions to answer
    else:
        return "critic"       # all done, critique


def build_graph() -> StateGraph:
    """
    Construct and compile the research agent graph.

    Node execution order:
      planner -> researcher (loop) -> critic -> synthesizer

    Returns compiled graph ready for .invoke() or .stream().
    """
    builder = StateGraph(ResearchState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("planner",     planner_node)
    builder.add_node("researcher",  researcher_node)
    builder.add_node("critic",      critic_node)
    builder.add_node("synthesizer", synthesizer_node)

    # ── Fixed edges ───────────────────────────────────────────────────────
    builder.add_edge(START,       "planner")
    builder.add_edge("planner",   "researcher")

    # ── Conditional edge: after researcher ───────────────────────────────
    # Checks if more sub-questions remain or if it's time to critique
    builder.add_conditional_edges(
        "researcher",
        researcher_router,
        {
            "researcher": "researcher",  # loop back
            "critic":     "critic",      # move on
        },
    )

    # ── Conditional edge: after critic ───────────────────────────────────
    # should_continue() returns "researcher" (retry) or "synthesizer" (done)
    builder.add_conditional_edges(
        "critic",
        should_continue,
        {
            "researcher":  "researcher",   # retry with gap questions
            "synthesizer": "synthesizer",  # pass -> synthesize
        },
    )

    # ── Terminal edge ─────────────────────────────────────────────────────
    builder.add_edge("synthesizer", END)

    # ── Compile with in-memory checkpointing ─────────────────────────────
    # MemorySaver keeps state in RAM (single-process only).
    # For production: use SqliteSaver("./data/checkpoints.db")
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("Research graph compiled with nodes: planner, researcher, critic, synthesizer")
    return graph


# Module-level compiled graph (singleton)
research_graph = build_graph()
