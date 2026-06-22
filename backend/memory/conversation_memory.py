"""
conversation_memory.py — Sliding window conversation buffer.

WHY: The LLM has no memory between calls. We maintain a list of
{role, content} messages and prepend them to every LLM prompt so
the agent can refer to previous exchanges.

WINDOW BUFFER: We keep only the last N message pairs (default: 10).
This prevents the context window from filling up on long sessions.

INTERVIEW: "Conversation memory is the simplest form of agent memory.
It's a bounded sliding window of (human, assistant) turns. For longer
sessions, you'd summarise older turns with an LLM (summary memory) or
embed them into a vector store for semantic retrieval (vector memory)."

ALTERNATIVES:
  - Summary memory: LLM compresses older turns into a running summary
  - Entity memory: extract named entities and track their attributes
  - Vector memory: embed turns, retrieve relevant ones by similarity
"""

from collections import deque
from typing import Any


class ConversationMemory:
    """Sliding window conversation history buffer."""

    def __init__(self, max_messages: int = 20):
        # Store as deque for O(1) left-pop when window is full
        self._messages: deque[dict[str, str]] = deque(maxlen=max_messages)

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        """Return message history as a list (oldest first)."""
        return list(self._messages)

    def get_context_string(self) -> str:
        """Format history as a plain text block for prompt injection."""
        parts = []
        for msg in self._messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")
        return "\n".join(parts)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


# Per-session memory store: session_id -> ConversationMemory
_session_memories: dict[str, ConversationMemory] = {}


def get_conversation_memory(session_id: str) -> ConversationMemory:
    """Get or create a ConversationMemory for a session."""
    if session_id not in _session_memories:
        _session_memories[session_id] = ConversationMemory()
    return _session_memories[session_id]


def clear_session_memory(session_id: str) -> None:
    """Clear memory for a specific session."""
    if session_id in _session_memories:
        del _session_memories[session_id]
