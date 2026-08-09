import tempfile
from pathlib import Path

import pytest

from nmafc.schemas.memory import MemoryStateUpdate, MemoryType
from nmafc.storage.cold import ColdStorage


@pytest.fixture
def cold() -> ColdStorage:
    tmp = tempfile.mkdtemp()
    db_path = str(Path(tmp) / "test_cold.db")
    return ColdStorage(db_path)


class TestAppendAndRetrieve:
    def test_append_returns_id(self, cold: ColdStorage):
        update = MemoryStateUpdate(
            entity_name="medication_morning",
            fact_content="Takes aspirin at 9 AM",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        )
        event_id = cold.append_event(update, turn=1)
        assert event_id >= 1

    def test_append_sequential_ids(self, cold: ColdStorage):
        u1 = MemoryStateUpdate(
            entity_name="a", fact_content="fact a", memory_type=MemoryType.CORE_ANCHOR
        )
        u2 = MemoryStateUpdate(
            entity_name="b", fact_content="fact b", memory_type=MemoryType.EPHEMERAL_STATE
        )
        id1 = cold.append_event(u1, turn=1)
        id2 = cold.append_event(u2, turn=2)
        assert id2 == id1 + 1

    def test_get_active_events_ordered_by_turn(self, cold: ColdStorage):
        for i in range(5):
            cold.append_event(
                MemoryStateUpdate(
                    entity_name=f"e{i}",
                    fact_content=f"fact {i}",
                    memory_type=MemoryType.ACTIVE_CONTEXT,
                ),
                turn=5 - i,
            )
        events = cold.get_active_events()
        turns = [e["turn"] for e in events]
        assert turns == sorted(turns)

    def test_get_events_for_entity(self, cold: ColdStorage):
        cold.append_event(
            MemoryStateUpdate(
                entity_name="user_name",
                fact_content="Alice",
                memory_type=MemoryType.CORE_ANCHOR,
            ),
            turn=1,
        )
        cold.append_event(
            MemoryStateUpdate(
                entity_name="other",
                fact_content="irrelevant",
                memory_type=MemoryType.EPHEMERAL_STATE,
            ),
            turn=2,
        )
        results = cold.get_events_for_entity("user_name")
        assert len(results) == 1
        assert results[0]["fact_content"] == "Alice"


class TestMarkInactive:
    def test_mark_inactive_excludes_from_active(self, cold: ColdStorage):
        event_id = cold.append_event(
            MemoryStateUpdate(
                entity_name="temp",
                fact_content="temporary",
                memory_type=MemoryType.EPHEMERAL_STATE,
            ),
            turn=1,
        )
        assert cold.count_active() == 1
        cold.mark_inactive(event_id)
        assert cold.count_active() == 0

    def test_mark_inactive_preserves_total_count(self, cold: ColdStorage):
        event_id = cold.append_event(
            MemoryStateUpdate(
                entity_name="temp",
                fact_content="temporary",
                memory_type=MemoryType.EPHEMERAL_STATE,
            ),
            turn=1,
        )
        cold.mark_inactive(event_id)
        assert cold.count_total() == 1


class TestKeywordSearch:
    def test_finds_matching_content(self, cold: ColdStorage):
        cold.append_event(
            MemoryStateUpdate(
                entity_name="medication",
                fact_content="Takes aspirin every morning for headaches",
                memory_type=MemoryType.ACTIVE_CONTEXT,
            ),
            turn=1,
        )
        cold.append_event(
            MemoryStateUpdate(
                entity_name="hobby",
                fact_content="Enjoys painting landscapes on weekends",
                memory_type=MemoryType.EPHEMERAL_STATE,
            ),
            turn=2,
        )
        results = cold.keyword_search("aspirin")
        assert len(results) == 1
        assert results[0]["entity_name"] == "medication"

    def test_excludes_inactive_from_search(self, cold: ColdStorage):
        event_id = cold.append_event(
            MemoryStateUpdate(
                entity_name="old",
                fact_content="aspirin dosage was wrong",
                memory_type=MemoryType.ACTIVE_CONTEXT,
            ),
            turn=1,
        )
        cold.mark_inactive(event_id)
        results = cold.keyword_search("aspirin")
        assert len(results) == 0

    def test_respects_limit(self, cold: ColdStorage):
        for i in range(10):
            cold.append_event(
                MemoryStateUpdate(
                    entity_name=f"item_{i}",
                    fact_content=f"aspirin fact number {i}",
                    memory_type=MemoryType.ACTIVE_CONTEXT,
                ),
                turn=i,
            )
        results = cold.keyword_search("aspirin", limit=3)
        assert len(results) == 3
