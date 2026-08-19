from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from nmafc.schemas.memory import MemoryStateUpdate
from nmafc.storage.cold_base import ColdStorageBase


class PostgresColdStorage(ColdStorageBase):
    """Append-only PostgreSQL event log (Cold ROM) for remote/scaled deployments.

    Drop-in replacement for ColdStorage (SQLite) when you need:
    - Multi-instance access (serverless, horizontal scaling)
    - Remote persistence (managed Postgres: Supabase, Neon, RDS, etc.)
    - Full-text search via PostgreSQL tsvector

    Usage:
        cold = PostgresColdStorage("postgresql://user:pass@host:5432/nmafc")
    """

    def __init__(self, dsn: str, agent_id: str = "default", conversation_id: str = "default") -> None:
        try:
            import psycopg2  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Install psycopg2 for PostgreSQL support: pip install psycopg2-binary"
            ) from e

        self._agent_id = agent_id
        self._conversation_id = conversation_id
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._create_tables()

    def _create_tables(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_event_log (
                    id SERIAL PRIMARY KEY,
                    agent_id TEXT NOT NULL DEFAULT 'default',
                    conversation_id TEXT NOT NULL DEFAULT 'default',
                    timestamp TIMESTAMPTZ NOT NULL,
                    turn INTEGER NOT NULL,
                    entity_name TEXT NOT NULL,
                    fact_content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    overrides_entity TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    search_vector TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', entity_name || ' ' || fact_content)
                    ) STORED
                );

                CREATE INDEX IF NOT EXISTS idx_mel_entity_name
                    ON memory_event_log(entity_name);

                CREATE INDEX IF NOT EXISTS idx_mel_is_active
                    ON memory_event_log(is_active);

                CREATE INDEX IF NOT EXISTS idx_mel_agent_conversation
                    ON memory_event_log(agent_id, conversation_id);

                CREATE INDEX IF NOT EXISTS idx_mel_search_vector
                    ON memory_event_log USING GIN(search_vector);
            """)
        self._conn.commit()

    def append_event(
        self,
        update: MemoryStateUpdate,
        turn: int,
        embedding: list[float] | None = None,
    ) -> int:
        """Append one immutable event.

        `embedding` is accepted for interface parity with the SQLite backend and
        currently discarded: dense retrieval over the archive needs a vector
        column, and adding one here means requiring the pgvector extension,
        which is a deployment decision rather than a code change. Until then
        this backend inherits ColdStorageBase.semantic_search, which returns
        nothing, so the router falls back to keyword search alone -- the
        behaviour every backend had before dense fallback existed.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO memory_event_log
                   (agent_id, conversation_id, timestamp, turn, entity_name, fact_content, memory_type, overrides_entity)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    self._agent_id,
                    self._conversation_id,
                    datetime.now(timezone.utc),
                    turn,
                    update.entity_name,
                    update.fact_content,
                    update.memory_type.value,
                    update.overrides_entity,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("INSERT RETURNING did not return a row")
            event_id: int = row[0]
        self._conn.commit()
        return event_id

    def mark_inactive(self, event_id: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE memory_event_log SET is_active = FALSE WHERE id = %s AND agent_id = %s AND conversation_id = %s",
                (event_id, self._agent_id, self._conversation_id),
            )
        self._conn.commit()

    def get_active_events(self) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT id, timestamp, turn, entity_name, fact_content,
                          memory_type, overrides_entity, is_active
                   FROM memory_event_log
                   WHERE agent_id = %s AND conversation_id = %s AND is_active = TRUE
                   ORDER BY turn ASC, id ASC""",
                (self._agent_id, self._conversation_id),
            )
            if cur.description is None:
                return []
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_events_for_entity(self, entity_name: str) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT id, timestamp, turn, entity_name, fact_content,
                          memory_type, overrides_entity, is_active
                   FROM memory_event_log
                   WHERE agent_id = %s AND conversation_id = %s AND entity_name = %s AND is_active = TRUE
                   ORDER BY turn ASC""",
                (self._agent_id, self._conversation_id, entity_name),
            )
            if cur.description is None:
                return []
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def keyword_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        sanitized = self._sanitize_query(query)
        if not sanitized:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT id, timestamp, turn, entity_name, fact_content,
                          memory_type, overrides_entity, is_active
                   FROM memory_event_log
                   WHERE agent_id = %s AND conversation_id = %s
                     AND search_vector @@ plainto_tsquery('english', %s)
                     AND is_active = TRUE
                   ORDER BY ts_rank(search_vector, plainto_tsquery('english', %s)) DESC
                   LIMIT %s""",
                (self._agent_id, self._conversation_id, sanitized, sanitized, limit),
            )
            if cur.description is None:
                return []
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    @staticmethod
    def _sanitize_query(query: str) -> str:
        import re
        words = re.findall(r"[a-zA-Z0-9]+", query)
        return " ".join(words) if words else ""

    def count_active(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM memory_event_log WHERE agent_id = %s AND conversation_id = %s AND is_active = TRUE",
                (self._agent_id, self._conversation_id),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def count_total(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM memory_event_log WHERE agent_id = %s AND conversation_id = %s",
                (self._agent_id, self._conversation_id),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()
