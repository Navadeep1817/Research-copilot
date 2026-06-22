"""long_term_memory.py — SQLite-backed persistence. Uses absolute db path."""

import json
import logging
from datetime import datetime
import aiosqlite
from backend.config import get_settings

logger = logging.getLogger(__name__)


async def store_memory(key: str, value: str | dict) -> None:
    settings = get_settings()
    val = json.dumps(value) if isinstance(value, dict) else value
    async with aiosqlite.connect(str(settings.db_path)) as db:
        await db.execute(
            """INSERT INTO long_term_memory (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, val, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def retrieve_memory(key: str) -> str | None:
    settings = get_settings()
    async with aiosqlite.connect(str(settings.db_path)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT value FROM long_term_memory WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
    return row["value"] if row else None


async def list_memory_keys(prefix: str = "") -> list[str]:
    settings = get_settings()
    async with aiosqlite.connect(str(settings.db_path)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT key FROM long_term_memory WHERE key LIKE ?", (f"{prefix}%",)
        ) as cur:
            rows = await cur.fetchall()
    return [r["key"] for r in rows]


async def store_research_summary(session_id: str, query: str, summary: str) -> None:
    await store_memory(f"research:{session_id}", json.dumps({"query": query, "summary": summary}))
