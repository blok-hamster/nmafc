from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nmafc.schemas.events import EventType, MemoryEvent


class EventLog:
    """SQLite-backed cognitive event log for the Web UI.

    Stores structured events (weight updates, overrides, pruning,
    consolidation, LTP, retrieval) for visualization, audit, and
    real-time monitoring.

    Backward-compatible: engine functions accept an optional
    ``event_logger`` parameter. When None, no events are logged.
    """

    def __init__(self, db_path: str, agent_id: str = "default", conversation_id: str = "default") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._agent_id = agent_id
        self._conversation_id = conversation_id
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                agent_id TEXT NOT NULL DEFAULT 'default',
                conversation_id TEXT NOT NULL DEFAULT 'default',
                turn INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                record_id TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                old_weight REAL,
                new_weight REAL,
                old_memory_type TEXT,
                new_memory_type TEXT,
                suppressed_by TEXT,
                old_k INTEGER,
                new_k INTEGER,
                retrieval_score REAL,
                hops INTEGER,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_me_agent_conv
                ON memory_events(agent_id, conversation_id);

            CREATE INDEX IF NOT EXISTS idx_me_type
                ON memory_events(event_type);

            CREATE INDEX IF NOT EXISTS idx_me_turn
                ON memory_events(turn);

            CREATE INDEX IF NOT EXISTS idx_me_record
                ON memory_events(record_id);

            CREATE INDEX IF NOT EXISTS idx_me_entity
                ON memory_events(entity_name);
        """)
        self._conn.commit()

    def log(self, event: MemoryEvent) -> int:
        """Insert a cognitive event and return its row ID."""
        event.agent_id = self._agent_id
        event.conversation_id = self._conversation_id
        cursor = self._conn.execute(
            """INSERT INTO memory_events
               (event_type, agent_id, conversation_id, turn, timestamp,
                record_id, entity_name, old_weight, new_weight,
                old_memory_type, new_memory_type, suppressed_by,
                old_k, new_k, retrieval_score, hops, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_type.value,
                event.agent_id,
                event.conversation_id,
                event.turn,
                event.timestamp.isoformat(),
                event.record_id,
                event.entity_name,
                event.old_weight,
                event.new_weight,
                event.old_memory_type,
                event.new_memory_type,
                event.suppressed_by,
                event.old_k,
                event.new_k,
                event.retrieval_score,
                event.hops,
                json.dumps(event.metadata) if event.metadata else None,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def log_many(self, events: list[MemoryEvent]) -> list[int]:
        """Batch-insert cognitive events. Returns list of row IDs."""
        ids = []
        for event in events:
            ids.append(self.log(event))
        return ids

    def query(
        self,
        *,
        turn_from: int | None = None,
        turn_to: int | None = None,
        event_types: list[EventType] | None = None,
        entity_name: str | None = None,
        record_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEvent]:
        """Query events with flexible filters."""
        clauses = ["agent_id = ?", "conversation_id = ?"]
        params: list[Any] = [self._agent_id, self._conversation_id]

        if turn_from is not None:
            clauses.append("turn >= ?")
            params.append(turn_from)
        if turn_to is not None:
            clauses.append("turn <= ?")
            params.append(turn_to)
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            clauses.append(f"event_type IN ({placeholders})")
            params.extend([et.value for et in event_types])
        if entity_name is not None:
            clauses.append("entity_name = ?")
            params.append(entity_name)
        if record_id is not None:
            clauses.append("record_id = ?")
            params.append(record_id)

        where = " AND ".join(clauses)
        params.extend([limit, offset])

        cursor = self._conn.execute(
            f"""SELECT * FROM memory_events
                WHERE {where}
                ORDER BY turn DESC, id DESC
                LIMIT ? OFFSET ?""",
            params,
        )
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_timeline(self, limit: int = 100) -> list[dict[str, Any]]:
        """Aggregated event counts per turn, grouped by event type."""
        cursor = self._conn.execute(
            """SELECT turn, event_type, COUNT(*) as count
               FROM memory_events
               WHERE agent_id = ? AND conversation_id = ?
               GROUP BY turn, event_type
               ORDER BY turn DESC
               LIMIT ?""",
            (self._agent_id, self._conversation_id, limit * 10),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_entity_history(self, entity_name: str, limit: int = 50) -> list[MemoryEvent]:
        """All events for a specific entity, newest first."""
        cursor = self._conn.execute(
            """SELECT * FROM memory_events
               WHERE agent_id = ? AND conversation_id = ? AND entity_name = ?
               ORDER BY turn DESC, id DESC
               LIMIT ?""",
            (self._agent_id, self._conversation_id, entity_name, limit),
        )
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def count(self) -> int:
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM memory_events WHERE agent_id = ? AND conversation_id = ?",
            (self._agent_id, self._conversation_id),
        )
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def count_by_type(self) -> dict[str, int]:
        cursor = self._conn.execute(
            """SELECT event_type, COUNT(*) as count
               FROM memory_events
               WHERE agent_id = ? AND conversation_id = ?
               GROUP BY event_type""",
            (self._agent_id, self._conversation_id),
        )
        return {row["event_type"]: row["count"] for row in cursor.fetchall()}

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
        metadata = {}
        raw = row["metadata"]
        if raw:
            try:
                metadata = json.loads(raw)
            except (TypeError, ValueError):
                metadata = {}
        return MemoryEvent(
            event_type=EventType(row["event_type"]),
            agent_id=row["agent_id"],
            conversation_id=row["conversation_id"],
            turn=row["turn"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            record_id=row["record_id"],
            entity_name=row["entity_name"],
            old_weight=row["old_weight"],
            new_weight=row["new_weight"],
            old_memory_type=row["old_memory_type"],
            new_memory_type=row["new_memory_type"],
            suppressed_by=row["suppressed_by"],
            old_k=row["old_k"],
            new_k=row["new_k"],
            retrieval_score=row["retrieval_score"],
            hops=row["hops"],
            metadata=metadata,
        )
