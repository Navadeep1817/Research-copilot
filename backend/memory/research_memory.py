"""
research_memory.py — Per-session research evidence scratchpad.

WHY: During a multi-hop research session, the researcher collects
evidence across multiple sub-questions. This memory stores all
retrieved documents so the synthesizer can access them collectively.

DESIGN: Simple dict-backed store, keyed by session_id.
Each session has: list of sources, sub-question answers, metadata.

INTERVIEW: "Research memory is the working memory of the agent during
a single research session. It's separate from conversation memory
(which tracks the dialogue) and long-term memory (which persists
across sessions). This separation follows the cognitive science
distinction between working, episodic, and semantic memory."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from backend.models.schemas import SourceDocument, SubQuestion


@dataclass
class ResearchSession:
    session_id:    str
    query:         str
    sources:       list[SourceDocument] = field(default_factory=list)
    sub_questions: list[SubQuestion]    = field(default_factory=list)
    report:        str                  = ""

    def add_sources(self, sources: list[SourceDocument]) -> None:
        """Add sources, deduplicating by content prefix."""
        existing = {s.content[:100] for s in self.sources}
        for s in sources:
            if s.content[:100] not in existing:
                self.sources.append(s)
                existing.add(s.content[:100])

    def get_all_context(self) -> str:
        """Return all collected evidence as a single string."""
        parts = [f"[{s.title or s.source}]\n{s.content}" for s in self.sources[:20]]
        return "\n\n---\n\n".join(parts)


_sessions: dict[str, ResearchSession] = {}


def get_research_session(session_id: str, query: str = "") -> ResearchSession:
    if session_id not in _sessions:
        _sessions[session_id] = ResearchSession(session_id=session_id, query=query)
    return _sessions[session_id]


def clear_research_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
