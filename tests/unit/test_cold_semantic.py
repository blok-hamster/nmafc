"""Cold ROM as a searchable archive, not just a keyword index.

Cold ROM is the only complete record — Hot RAM prunes, Cold ROM never does —
but it could previously only be reached by shared words. A question about an
aunt could not match a fact filed as "mother's sister", so the archive held the
answer and had no way to return it.

These tests pin three things: that the vectors get stored at all, that dense
search actually ranks by meaning rather than by word overlap, and that an
archive written before any of this existed still opens and still answers.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from nmafc.schemas.memory import MemoryStateUpdate, MemoryType
from nmafc.storage.cold import ColdStorage


def update(entity: str, fact: str = "some fact", related=None) -> MemoryStateUpdate:
    return MemoryStateUpdate(
        entity_name=entity,
        fact_content=fact,
        memory_type=MemoryType.ACTIVE_CONTEXT,
        related_entities=related or [],
    )


@pytest.fixture
def cold(tmp_path: Path) -> ColdStorage:
    store = ColdStorage(str(tmp_path / "cold.db"))
    yield store
    store.close()


class TestVectorsAreStored:
    def test_embedding_round_trips(self, cold: ColdStorage):
        cold.append_event(update("a"), turn=1, embedding=[1.0, 0.0, 0.0])
        hits = cold.semantic_search([1.0, 0.0, 0.0], limit=5)
        assert [h["entity_name"] for h in hits] == ["a"]
        assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_related_entities_round_trip(self, cold: ColdStorage):
        cold.append_event(update("a", related=["b", "c"]), turn=1, embedding=[1.0, 0.0])
        rows = cold.get_events_for_entities(["a"])
        assert rows[0]["related_entities"] == '["b", "c"]'

    def test_events_without_embeddings_still_append(self, cold: ColdStorage):
        cold.append_event(update("a"), turn=1)
        assert cold.count_total() == 1
        assert cold.semantic_search([1.0, 0.0], limit=5) == []

    def test_unembedded_rows_stay_keyword_reachable(self, cold: ColdStorage):
        """The fallback runs both searches, so a vectorless row is not lost."""
        cold.append_event(update("a", "the platypus is nocturnal"), turn=1)
        assert cold.keyword_search("platypus")


class TestSemanticRanking:
    def test_ranks_by_direction_not_word_overlap(self, cold: ColdStorage):
        cold.append_event(update("near"), turn=1, embedding=[1.0, 0.1, 0.0])
        cold.append_event(update("far"), turn=2, embedding=[0.0, 0.0, 1.0])
        hits = cold.semantic_search([1.0, 0.0, 0.0], limit=2)
        assert [h["entity_name"] for h in hits] == ["near", "far"]
        assert hits[0]["score"] > hits[1]["score"]

    def test_magnitude_does_not_beat_direction(self, cold: ColdStorage):
        """Cosine, not dot product: a long vector pointing away must not win."""
        cold.append_event(update("aligned"), turn=1, embedding=[1.0, 0.0])
        cold.append_event(update("loud"), turn=2, embedding=[0.0, 50.0])
        hits = cold.semantic_search([1.0, 0.0], limit=2)
        assert hits[0]["entity_name"] == "aligned"

    def test_limit_is_respected(self, cold: ColdStorage):
        for i in range(10):
            cold.append_event(update(f"e{i}"), turn=i, embedding=[1.0, i / 10])
        assert len(cold.semantic_search([1.0, 0.0], limit=3)) == 3

    def test_retired_events_disappear(self, cold: ColdStorage):
        """A superseded fact must not keep surfacing after being withdrawn."""
        event_id = cold.append_event(update("a"), turn=1, embedding=[1.0, 0.0])
        assert cold.semantic_search([1.0, 0.0], limit=5)
        cold.mark_inactive(event_id)
        assert cold.semantic_search([1.0, 0.0], limit=5) == []

    def test_new_events_are_visible_immediately(self, cold: ColdStorage):
        """The vector matrix is cached, so appends must invalidate it."""
        cold.append_event(update("first"), turn=1, embedding=[1.0, 0.0])
        cold.semantic_search([1.0, 0.0], limit=5)  # warm the cache
        cold.append_event(update("second"), turn=2, embedding=[1.0, 0.0])
        assert len(cold.semantic_search([1.0, 0.0], limit=5)) == 2

    def test_empty_query_returns_nothing(self, cold: ColdStorage):
        cold.append_event(update("a"), turn=1, embedding=[1.0, 0.0])
        assert cold.semantic_search([], limit=5) == []

    def test_zero_query_vector_returns_nothing(self, cold: ColdStorage):
        """A zero vector has no direction, so cosine is undefined, not zero."""
        cold.append_event(update("a"), turn=1, embedding=[1.0, 0.0])
        assert cold.semantic_search([0.0, 0.0], limit=5) == []

    def test_mismatched_dimensions_are_skipped_not_scored(self, cold: ColdStorage):
        """A mid-archive model swap makes old vectors incomparable, not wrong."""
        cold.append_event(update("old"), turn=1, embedding=[1.0, 0.0])
        cold.append_event(update("new"), turn=2, embedding=[1.0, 0.0, 0.0, 0.0])
        hits = cold.semantic_search([1.0, 0.0], limit=5)
        assert [h["entity_name"] for h in hits] == ["old"]


class TestGraphExpansion:
    def test_finds_events_by_entity_name(self, cold: ColdStorage):
        cold.append_event(update("alice"), turn=1, embedding=[1.0, 0.0])
        cold.append_event(update("bob"), turn=2, embedding=[0.0, 1.0])
        rows = cold.get_events_for_entities(["bob"])
        assert [r["entity_name"] for r in rows] == ["bob"]

    def test_matching_is_case_insensitive(self, cold: ColdStorage):
        """Links are lowercased everywhere else; the archive must agree."""
        cold.append_event(update("Alice"), turn=1)
        assert cold.get_events_for_entities(["ALICE"])

    def test_reaches_facts_that_hot_ram_would_have_pruned(self, cold: ColdStorage):
        """The point of the whole feature: a link to a pruned fact is not a dead end."""
        cold.append_event(update("kept", related=["dropped"]), turn=1)
        cold.append_event(update("dropped", "the fact Hot RAM lost"), turn=2)
        rows = cold.get_events_for_entities(["dropped"])
        assert rows[0]["fact_content"] == "the fact Hot RAM lost"

    def test_empty_input_is_not_a_full_table_scan(self, cold: ColdStorage):
        cold.append_event(update("a"), turn=1)
        assert cold.get_events_for_entities([]) == []

    def test_retired_events_are_excluded(self, cold: ColdStorage):
        event_id = cold.append_event(update("a"), turn=1)
        cold.mark_inactive(event_id)
        assert cold.get_events_for_entities(["a"]) == []


class TestBackwardCompatibility:
    def test_archive_without_the_new_columns_is_migrated_in_place(self, tmp_path: Path):
        """Cold ROM is the audit trail; it must never need rebuilding to be read.

        The schema below is the previous release's verbatim -- FTS table and
        insert trigger included, since those have always existed -- minus the
        two columns added for dense retrieval. That is the exact shape of every
        archive already on disk.
        """
        path = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE memory_event_log (
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

            CREATE VIRTUAL TABLE memory_fts
                USING fts5(entity_name, fact_content, content=memory_event_log, content_rowid=id);

            CREATE TRIGGER memory_fts_insert AFTER INSERT ON memory_event_log
            BEGIN
                INSERT INTO memory_fts(rowid, entity_name, fact_content)
                VALUES (new.id, new.entity_name, new.fact_content);
            END;

            INSERT INTO memory_event_log
                (timestamp, turn, entity_name, fact_content, memory_type)
            VALUES ('2020-01-01T00:00:00', 1, 'old_fact', 'written long ago', 'CoreAnchor');
        """)
        conn.commit()
        conn.close()

        store = ColdStorage(path)
        try:
            assert store.count_total() == 1
            # The pre-existing row has no vector and must not break dense search.
            assert store.semantic_search([1.0, 0.0], limit=5) == []
            # It is still reachable the way it always was.
            assert store.keyword_search("written")
            # And new rows can carry vectors alongside it.
            store.append_event(update("new_fact"), turn=2, embedding=[1.0, 0.0])
            hits = store.semantic_search([1.0, 0.0], limit=5)
            assert [h["entity_name"] for h in hits] == ["new_fact"]
        finally:
            store.close()
