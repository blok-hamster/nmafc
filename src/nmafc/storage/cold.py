from __future__ import annotations

import array
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nmafc.schemas.memory import MemoryStateUpdate
from nmafc.storage.cold_base import ColdStorageBase


class ColdStorage(ColdStorageBase):
    """Append-only SQLite event log (Cold ROM).

    Stores 100% of raw state-change events as an immutable audit trail.
    Retrieval is hybrid: FTS5 keyword search plus dense vector search over the
    same embeddings Hot RAM already holds.

    Why the archive carries vectors at all
    --------------------------------------
    Cold ROM is the only complete record -- Hot RAM prunes, Cold ROM never
    does -- but keyword matching could only find facts that reuse the question's
    words. A question asking about someone's aunt cannot keyword-match a fact
    stored as "mother's sister", so the archive held the answer and could not
    reach it. That is precisely the retrieval a dense vector handles, and it is
    what the fallback existed to provide.

    The embeddings cost nothing extra. The wrapper already computes one vector
    per fact to write into Hot RAM, and simply hands the same vector here on the
    way past, so the archive gains semantic search for zero additional
    embedding calls. Vectors are optional: rows written before this existed have
    NULL and are still reachable by keyword.
    """

    def __init__(self, db_path: str, agent_id: str = "default", conversation_id: str = "default") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._agent_id = agent_id
        self._conversation_id = conversation_id
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        # (ids, matrix) for this tenant's embedded rows, rebuilt lazily and
        # dropped on every append. Cold ROM is append-only and a conversation
        # holds a few hundred to a few thousand facts, so a brute-force scan
        # over an in-memory matrix is both exact and fast enough that an index
        # would add a dependency and an approximation for no measurable gain.
        self._vector_cache: tuple[list[int], Any] | None = None

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
                is_active INTEGER DEFAULT 1,
                related_entities TEXT,
                embedding BLOB
            );

            CREATE INDEX IF NOT EXISTS idx_entity_name
                ON memory_event_log(entity_name);

            CREATE INDEX IF NOT EXISTS idx_is_active
                ON memory_event_log(is_active);

            CREATE INDEX IF NOT EXISTS idx_agent_conversation
                ON memory_event_log(agent_id, conversation_id);

            -- When each turn actually happened, in whatever wording the caller
            -- uses. One row per turn rather than per fact: the date belongs to
            -- the conversation, not to each thing said in it, and a turn
            -- routinely produces six facts that would otherwise carry six
            -- copies of one string.
            --
            -- This exists because turn numbers were reaching the model in place
            -- of dates. Facts were presented as "(Valid: turn 220 - present)",
            -- and an ordinal is not a date, so answers came back saying "it was
            -- mentioned in turn 220 but no specific date is given" -- which is
            -- a correct reading of what it was shown.
            CREATE TABLE IF NOT EXISTS turn_timestamps (
                agent_id TEXT NOT NULL DEFAULT 'default',
                conversation_id TEXT NOT NULL DEFAULT 'default',
                turn INTEGER NOT NULL,
                occurred_at TEXT NOT NULL,
                PRIMARY KEY (agent_id, conversation_id, turn)
            );

            -- What was actually said, kept verbatim, one row per turn.
            --
            -- Extraction is lossy and nothing else in this store survives it:
            -- the hot vector and the event log both hold only the summarised
            -- fact, so once a turn is processed its wording is gone. Measured
            -- over 1,540 questions, restoring the source text behind the five
            -- best-ranked facts moved strict answer reachability from 52.7% to
            -- 59.4%, and on the temporal category from 50.5% to 59.8%, because
            -- the dataset's own phrasing ("two weekends before 17 July 2023")
            -- lives in the utterance while the fact carries a date the
            -- extractor resolved for itself.
            --
            -- Deliberately outside memory_fts and carrying no embedding. Facts
            -- are what decays and what gets ranked; this is evidence, reachable
            -- only through a fact that already won a slot, by its turn number.
            -- Indexing it would put raw turns into the candidate pool where
            -- they would compete with facts for the retrieval budget, which is
            -- the one way storing it could cost anything.
            CREATE TABLE IF NOT EXISTS turn_text (
                agent_id TEXT NOT NULL DEFAULT 'default',
                conversation_id TEXT NOT NULL DEFAULT 'default',
                turn INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (agent_id, conversation_id, turn)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(entity_name, fact_content, content=memory_event_log, content_rowid=id);

            CREATE TRIGGER IF NOT EXISTS memory_fts_insert AFTER INSERT ON memory_event_log
            BEGIN
                INSERT INTO memory_fts(rowid, entity_name, fact_content)
                VALUES (new.id, new.entity_name, new.fact_content);
            END;
        """)
        # Databases written before related_entities and embedding existed are
        # still valid archives, and the archive is the one store that must never
        # be rebuilt from scratch. Add the columns in place; the old rows keep
        # NULL and stay keyword-reachable.
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(memory_event_log)")}
        for column, decl in (
            ("related_entities", "TEXT"),
            ("embedding", "BLOB"),
            ("valid_at", "INTEGER"),
            ("invalid_at", "INTEGER"),
        ):
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE memory_event_log ADD COLUMN {column} {decl}"
                )
        self._conn.commit()

    def append_event(
        self,
        update: MemoryStateUpdate,
        turn: int,
        embedding: list[float] | None = None,
        valid_at: int | None = None,
    ) -> int:
        """Append one immutable event, optionally with its dense vector.

        `embedding` is the same vector the caller is about to write into Hot RAM.
        Passing it here is what makes semantic_search possible; omitting it
        leaves the row keyword-only, which is what every pre-existing archive
        and every caller that has no embedder will get.
        """
        cursor = self._conn.execute(
            """INSERT INTO memory_event_log
               (agent_id, conversation_id, timestamp, turn, entity_name, fact_content,
                memory_type, overrides_entity, related_entities, embedding, valid_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._agent_id,
                self._conversation_id,
                datetime.now(timezone.utc).isoformat(),
                turn,
                update.entity_name,
                update.fact_content,
                update.memory_type.value,
                update.overrides_entity,
                json.dumps(list(update.related_entities)),
                self._pack(embedding),
                valid_at,
            ),
        )
        self._conn.commit()
        self._vector_cache = None
        return cursor.lastrowid  # type: ignore[return-value]

    def record_turn_timestamp(self, turn: int, occurred_at: str) -> None:
        """Note when a turn happened, so its facts can be dated at answer time.

        Last write wins. Re-ingesting a turn or backfilling a finished store
        should correct the date rather than fail on the primary key.
        """
        self._conn.execute(
            """INSERT INTO turn_timestamps (agent_id, conversation_id, turn, occurred_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(agent_id, conversation_id, turn)
               DO UPDATE SET occurred_at = excluded.occurred_at""",
            (self._agent_id, self._conversation_id, turn, occurred_at),
        )
        self._conn.commit()

    def turn_timestamps(self) -> dict[int, str]:
        """Every dated turn for this tenant, as turn -> date."""
        rows = self._conn.execute(
            "SELECT turn, occurred_at FROM turn_timestamps "
            "WHERE agent_id = ? AND conversation_id = ?",
            (self._agent_id, self._conversation_id),
        ).fetchall()
        return {row["turn"]: row["occurred_at"] for row in rows}

    def record_turn_text(self, turn: int, text: str) -> None:
        """Keep a turn's wording, so facts extracted from it stay recoverable.

        Last write wins, matching `record_turn_timestamp`: re-ingesting a turn
        or backfilling a finished store should correct the text rather than
        fail on the primary key.
        """
        self._conn.execute(
            """INSERT INTO turn_text (agent_id, conversation_id, turn, text)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(agent_id, conversation_id, turn)
               DO UPDATE SET text = excluded.text""",
            (self._agent_id, self._conversation_id, turn, text),
        )
        self._conn.commit()

    def text_for_turns(self, turns: list[int]) -> dict[int, str]:
        """The wording of specific turns, as turn -> text.

        Takes the turns wanted rather than returning the lot. A conversation
        holds hundreds of turns and a prompt hydrates a handful, so reading the
        whole table per query would be the expensive way to answer a question
        the primary key already answers.
        """
        wanted = sorted({t for t in turns if t})
        if not wanted:
            return {}
        placeholders = ",".join("?" for _ in wanted)
        rows = self._conn.execute(
            f"""SELECT turn, text FROM turn_text
                WHERE agent_id = ? AND conversation_id = ?
                  AND turn IN ({placeholders})""",
            (self._agent_id, self._conversation_id, *wanted),
        ).fetchall()
        return {row["turn"]: row["text"] for row in rows}

    @staticmethod
    def _pack(embedding: list[float] | None) -> bytes | None:
        """Float32 little-endian bytes. Half the size of float64 and lossless
        against what the providers return, which is float32 to begin with."""
        if not embedding:
            return None
        return array.array("f", embedding).tobytes()

    def mark_inactive(self, event_id: int) -> None:
        self._conn.execute(
            "UPDATE memory_event_log SET is_active = 0 WHERE id = ? AND agent_id = ? AND conversation_id = ?",
            (event_id, self._agent_id, self._conversation_id),
        )
        self._conn.commit()
        # The vector matrix only holds active rows, so retiring one invalidates
        # it. Without this a superseded fact keeps being returned by semantic
        # search after it has been withdrawn.
        self._vector_cache = None

    def invalidate_facts(self, facts: list[tuple[str, str, int]]) -> int:
        """Record in the archive that a fact has been superseded.

        Hot RAM already drops invalidated records from search, but the router
        queries the archive on every retrieval, so without this the superseded
        fact is pulled straight back into the prompt and the invalidation is
        invisible to the answer. Measured over 101 belief-update questions: with
        Hot-only invalidation the stale value still reached the prompt 64 times
        out of 101, one MORE than with no invalidation at all.

        Matched on (entity_name, fact_content) because the archive autoincrements
        its own primary key and does not carry the Hot record's UUID. Rows
        already carrying an `invalid_at` are left alone, so the earliest
        supersession stands.

        Distinct from `mark_inactive`, and deliberately so. A pruned fact is
        weak but still true and stays reachable; a superseded fact is wrong.
        """
        if not facts:
            return 0
        changed = 0
        for entity_name, fact_content, turn in facts:
            cursor = self._conn.execute(
                """UPDATE memory_event_log SET invalid_at = ?
                   WHERE agent_id = ? AND conversation_id = ?
                     AND entity_name = ? AND fact_content = ?
                     AND invalid_at IS NULL""",
                (turn, self._agent_id, self._conversation_id,
                 entity_name, fact_content),
            )
            changed += cursor.rowcount
        self._conn.commit()
        # Same reasoning as mark_inactive: the matrix is built from the rows
        # that survive the filter, so it is stale the moment one is withdrawn.
        self._vector_cache = None
        return changed

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
                 AND mel.is_active = 1 AND mel.invalid_at IS NULL
               ORDER BY rank
               LIMIT ?""",
            (sanitized, self._agent_id, self._conversation_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def semantic_search(
        self, query_embedding: list[float], limit: int = 20
    ) -> list[dict[str, Any]]:
        """Rank archived events by cosine similarity to the query vector.

        Returns rows in descending similarity with a `score` key added, so a
        caller can threshold on relevance rather than taking a fixed top-N of
        possibly unrelated facts. Rows stored without an embedding are invisible
        here and remain reachable only by keyword_search; the router runs both
        and merges, so nothing is lost either way.
        """
        if not query_embedding:
            return []

        ids, matrix = self._load_vectors()
        if not ids:
            return []

        import numpy as np

        q = np.asarray(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            return []
        # Row norms are folded in at load time, so this is one matrix-vector
        # product per query rather than a per-row normalisation.
        scores = matrix @ (q / norm)

        top = np.argsort(-scores)[: max(0, limit)]
        chosen = [(ids[int(i)], float(scores[int(i)])) for i in top]
        if not chosen:
            return []

        placeholders = ",".join("?" for _ in chosen)
        cursor = self._conn.execute(
            f"""SELECT * FROM memory_event_log
                WHERE id IN ({placeholders}) AND agent_id = ? AND conversation_id = ?
                  AND is_active = 1 AND invalid_at IS NULL""",
            [i for i, _ in chosen] + [self._agent_id, self._conversation_id],
        )
        by_id = {row["id"]: dict(row) for row in cursor.fetchall()}

        results = []
        for event_id, score in chosen:
            row = by_id.get(event_id)
            if row is not None:
                row["score"] = score
                results.append(row)
        return results

    def _load_vectors(self):
        """Build (ids, L2-normalised matrix) for this tenant's embedded rows."""
        if self._vector_cache is not None:
            return self._vector_cache

        import numpy as np

        cursor = self._conn.execute(
            """SELECT id, embedding FROM memory_event_log
               WHERE agent_id = ? AND conversation_id = ?
                 AND is_active = 1 AND invalid_at IS NULL AND embedding IS NOT NULL
               ORDER BY id ASC""",
            (self._agent_id, self._conversation_id),
        )
        ids: list[int] = []
        vectors: list[Any] = []
        width = None
        for row in cursor.fetchall():
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            # A dimension change mid-archive means the embedding model was
            # swapped. Those vectors are not comparable, so keep the first
            # width seen and let the odd ones out fall back to keyword search
            # rather than silently scoring nonsense.
            if width is None:
                width = vec.shape[0]
            elif vec.shape[0] != width:
                continue
            ids.append(int(row["id"]))
            vectors.append(vec)

        if not ids:
            self._vector_cache = ([], None)
            return self._vector_cache

        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._vector_cache = (ids, matrix / norms)
        return self._vector_cache

    def get_events_for_entities(
        self, entity_names: list[str], limit: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch archived events for a set of entity names, case-insensitively.

        This is what lets spreading activation continue into Cold ROM. Hot RAM
        traversal stops at whatever it still holds, so a link pointing at a
        pruned fact used to be a dead end; the archive still has that fact.
        """
        names = [n.lower() for n in entity_names if n]
        if not names:
            return []
        placeholders = ",".join("?" for _ in names)
        cursor = self._conn.execute(
            f"""SELECT * FROM memory_event_log
                WHERE agent_id = ? AND conversation_id = ? AND is_active = 1
                  AND LOWER(entity_name) IN ({placeholders})
                ORDER BY turn ASC, id ASC
                LIMIT ?""",
            [self._agent_id, self._conversation_id] + names + [limit],
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
