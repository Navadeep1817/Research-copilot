"""
dependencies.py — FastAPI dependency injection container.

WHY DEPENDENCY INJECTION:
  FastAPI's Depends() system injects shared resources (DB, LLM, graph)
  into route handlers without global state. Each request gets a fresh
  DB connection, while expensive resources (graph, models) are singletons.

  This makes routes testable: inject a mock DB instead of a real one.

INTERVIEW: "I use FastAPI's dependency injection for three reasons:
1) Automatic cleanup (DB connections close after each request via yield),
2) Testability (swap real deps for mocks in pytest),
3) Shared singleton resources (the compiled LangGraph graph is built once
   and shared across all requests via lru_cache)."
"""

from functools import lru_cache
from backend.agents.graph import research_graph
from backend.db.database import get_db


def get_research_graph():
    """Return the compiled LangGraph graph (singleton)."""
    return research_graph


# get_db is already a generator (yield-based), re-export for clarity
__all__ = ["get_research_graph", "get_db"]
