from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nmafc.schemas.memory import MemoryStateUpdate
from nmafc.storage.cold_base import ColdStorageBase


class ColdStorage(ColdStorageBase):
    """Append-only SQLite event log (Cold ROM).

    Stores 100% of raw state-change events as an immutable audit trail.
    Supports full-text search via FTS5 for fallback retrieval.
    """

    def __init__(self, db_path: str, agent_id: str = "default", conversation_id: str = "default") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._agent_id = agent_id
        self._conversation_id = conversation_id
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL DEFAULT 'default',
                conversation_id TEXT NOT NULL DEFAULT 'default',
                timestamp TEXT NOT NULL,
                turn INTEGER NOT NULL,
                entity_name TEXT NOT NULL,
                fact_content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                overrides_entity TEXT,
                is_active INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_entity_name
                ON memory_event_log(entity_name);

            CREATE INDEX IF NOT EXISTS idx_is_active
                ON memory_event_log(is_active);

            CREATE INDEX IF NOT EXISTS idx_agent_conversation
                ON memory_event_log(agent_id, conversation_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(entity_name, fact_content, content=memory_event_log, content_rowid=id);

            CREATE TRIGGER IF NOT EXISTS memory_fts_insert AFTER INSERT ON memory_event_log
            BEGIN
                INSERT INTO memory_fts(rowid, entity_name, fact_content)
                VALUES (new.id, new.entity_name, new.fact_content);
            END;
        """)
        self._conn.commit()

    def append_event(self, update: MemoryStateUpdate, turn: int) -> int:
        cursor = self._conn.execute(
            """INSERT INTO memory_event_log
               (agent_id, conversation_id, timestamp, turn, entity_name, fact_content, memory_type, overrides_entity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._agent_id,
                self._conversation_id,
                datetime.now(timezone.utc).isoformat(),
                turn,
                update.entity_name,
                update.fact_content,
                update.memory_type.value,
                update.overrides_entity,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def mark_inactive(self, event_id: int) -> None:
        self._conn.execute(
            "UPDATE memory_event_log SET is_active = 0 WHERE id = ?",
            (event_id,),
        )
        self._conn.commit()

    def get_active_events(self) -> list[dict[str, Any]]:
        cursor = self._conn.execute(
            """SELECT * FROM memory_event_log
               WHERE agent_id = ? AND conversation_id = ? AND is_active = 1
               ORDER BY turn ASC, id ASC""",
            (self._agent_id, self._conversation_id),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_events_for_entity(self, entity_name: str) -> list[dict[str, Any]]:
        cursor = self._conn.execute(
            """SELECT * FROM memory_event_log
               WHERE agent_id = ? AND conversation_id = ? AND entity_name = ? AND is_active = 1
               ORDER BY turn ASC""",
            (self._agent_id, self._conversation_id, entity_name),
        )
        return [dict(row) for row in cursor.fetchall()]

    def keyword_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []
        cursor = self._conn.execute(
            """SELECT mel.* FROM memory_fts
               JOIN memory_event_log mel ON memory_fts.rowid = mel.id
               WHERE memory_fts MATCH ? AND mel.agent_id = ? AND mel.conversation_id = ?
                 AND mel.is_active = 1
               ORDER BY rank
               LIMIT ?""",
            (sanitized, self._agent_id, self._conversation_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a raw user query into valid FTS5 syntax.

        Splits into words, removes non-alphanumeric tokens,
        and joins with OR for broad matching.
        """
        import re
        words = re.findall(r"[a-zA-Z0-9]+", query)
        if not words:
            return ""
        return " OR ".join(f'"{w}"' for w in words)

    def count_active(self) -> int:
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM memory_event_log WHERE agent_id = ? AND conversation_id = ? AND is_active = 1",
            (self._agent_id, self._conversation_id),
        )
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def count_total(self) -> int:
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM memory_event_log WHERE agent_id = ? AND conversation_id = ?",
            (self._agent_id, self._conversation_id),
        )
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()
