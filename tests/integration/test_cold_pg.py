"""Integration tests for PostgresColdStorage against a real PostgreSQL.

Skipped unless a DSN is provided via ``NMAFC_TEST_PG_DSN`` (CI wires an
ephemeral ``postgres`` service container and sets it; local runs default to
skipped so nothing here needs credentials).

The suite uses a unique agent_id per module session and unique
conversation_ids per test, so it can run against a shared database without
clobbering other tenants.
"""

from __future__ import annotations

import os
import uuid

import pytest

from nmafc.schemas.memory import MemoryStateUpdate, MemoryType
from nmafc.storage.cold_pg import PostgresColdStorage

PG_DSN = os.environ.get("NMAFC_TEST_PG_DSN", "")

pytestmark = pytest.mark.skipif(
    not PG_DSN, reason="NMAFC_TEST_PG_DSN not set (needs a PostgreSQL server)"
)

AGENT_ID = f"pgtest-{uuid.uuid4().hex[:8]}"


def _update(
    entity: str, fact: str, mtype: MemoryType = MemoryType.ACTIVE_CONTEXT
) -> MemoryStateUpdate:
    return MemoryStateUpdate(
        entity_name=entity, fact_content=fact, memory_type=mtype
    )


@pytest.fixture()
def cold() -> PostgresColdStorage:
    store = PostgresColdStorage(
        PG_DSN, agent_id=AGENT_ID, conversation_id=f"conv-{uuid.uuid4().hex[:8]}"
    )
    yield store
    store.close()


class TestAppendAndRead:
    def test_append_returns_sequential_ids(self, cold: PostgresColdStorage):
        id1 = cold.append_event(_update("alice", "Works at Initech"), turn=1)
        id2 = cold.append_event(_update("bob", "Owns a red car"), turn=2)
        assert id2 == id1 + 1

    def test_get_active_events_returns_appended(
        self, cold: PostgresColdStorage
    ):
        cold.append_event(_update("alice", "Works at Initech"), turn=1)
        cold.append_event(_update("bob", "Owns a red car"), turn=2)
        events = cold.get_active_events()
        contents = {e["fact_content"] for e in events}
        assert {"Works at Initech", "Owns a red car"} <= contents

    def test_counts(self, cold: PostgresColdStorage):
        before_active = cold.count_active()
        before_total = cold.count_total()
        cold.append_event(_update("alice", "Works at Initech"), turn=1)
        assert cold.count_active() == before_active + 1
        assert cold.count_total() == before_total + 1


class TestMarkInactiveAndInvalidate:
    def test_mark_inactive_hides_from_active(self, cold: PostgresColdStorage):
        event_id = cold.append_event(_update("alice", "Will be hidden"), turn=1)
        assert cold.count_total() >= 1
        cold.mark_inactive(event_id)
        contents = {e["fact_content"] for e in cold.get_active_events()}
        assert "Will be hidden" not in contents

    def test_invalidate_facts_sets_invalid_at(self, cold: PostgresColdStorage):
        cold.append_event(_update("alice", "Lives in Vertigo City"), turn=1)
        n = cold.invalidate_facts([("alice", "Lives in Vertigo City", 5)])
        assert n == 1
        contents = {e["fact_content"] for e in cold.get_active_events()}
        assert "Lives in Vertigo City" not in contents
        invalidated = [
            e for e in cold.get_events_for_entity("alice")
            if e["fact_content"] == "Lives in Vertigo City"
        ]
        # get_events_for_entity filters invalid rows out entirely (it mirrors
        # the active filter), so the row is unreachable even for history here.
        assert invalidated == []

    def test_invalidate_twice_counts_once(self, cold: PostgresColdStorage):
        cold.append_event(_update("alice", "One invalidation only"), turn=1)
        assert cold.invalidate_facts([("alice", "One invalidation only", 2)]) == 1
        assert cold.invalidate_facts([("alice", "One invalidation only", 3)]) == 0


class TestKeywordSearch:
    def test_finds_by_content_word(self, cold: PostgresColdStorage):
        cold.append_event(_update("alice", "Drives a red convertible"), turn=1)
        hits = cold.keyword_search("convertible")
        assert any("red convertible" in h["fact_content"] for h in hits)

    def test_scoped_to_conversation(self, cold: PostgresColdStorage):
        cold.append_event(_update("alice", "Secret phrase zephyr"), turn=1)
        other = PostgresColdStorage(
            PG_DSN, agent_id=cold._agent_id, conversation_id="other-conv"
        )
        try:
            hits = other.keyword_search("zephyr")
            assert hits == []
        finally:
            other.close()


class TestTurnClock:
    def test_max_turn_tracks_appends(self, cold: PostgresColdStorage):
        cold.append_event(_update("alice", "Turn tracking"), turn=7)
        cold.append_event(_update("bob", "Turn tracking two"), turn=9)
        assert cold.max_turn() >= 9
