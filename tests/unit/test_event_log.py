import tempfile
from datetime import datetime, timezone

import pytest

from nmafc.engine.decay import decay_all
from nmafc.engine.pruning import apply_suppression, create_suppression_event, prune_cycle
from nmafc.engine.reinforcement import batch_reinforce, create_ltp_events
from nmafc.schemas.events import EventType, MemoryEvent
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType
from nmafc.storage.event_log import EventLog


def make_record(entity: str, weight: float = 1.0, **kwargs) -> MemoryRecord:
    defaults = {
        "entity_name": entity,
        "fact_content": f"Fact about {entity}",
        "memory_type": MemoryType.ACTIVE_CONTEXT,
        "weight": weight,
        "consolidation_index": 0,
        "created_at_turn": 1,
        "last_reinforced_turn": 1,
    }
    defaults.update(kwargs)
    return MemoryRecord(**defaults)


@pytest.fixture
def event_log(tmp_path) -> EventLog:
    db_path = str(tmp_path / "test_events.db")
    return EventLog(db_path, agent_id="test_agent", conversation_id="test_conv")


class TestEventLogTableCreation:
    def test_creates_table(self, event_log: EventLog):
        count = event_log.count()
        assert count == 0

    def test_creates_idempotent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        log1 = EventLog(db_path)
        log1.close()
        log2 = EventLog(db_path)
        assert log2.count() == 0
        log2.close()


class TestEventLogLogging:
    def test_log_single_event(self, event_log: EventLog):
        event = MemoryEvent(
            event_type=EventType.WEIGHT_UPDATE,
            turn=1,
            record_id="rec-1",
            entity_name="user_allergy",
            old_weight=1.0,
            new_weight=0.85,
        )
        row_id = event_log.log(event)
        assert row_id is not None
        assert row_id > 0
        assert event_log.count() == 1

    def test_log_sets_agent_and_conversation(self, event_log: EventLog):
        event = MemoryEvent(
            event_type=EventType.LTP,
            turn=1,
            record_id="rec-1",
            entity_name="user_name",
        )
        event_log.log(event)
        events = event_log.query()
        assert len(events) == 1
        assert events[0].agent_id == "test_agent"
        assert events[0].conversation_id == "test_conv"

    def test_log_many(self, event_log: EventLog):
        events = [
            MemoryEvent(
                event_type=EventType.WEIGHT_UPDATE,
                turn=i,
                record_id=f"rec-{i}",
                entity_name=f"entity_{i}",
            )
            for i in range(5)
        ]
        ids = event_log.log_many(events)
        assert len(ids) == 5
        assert event_log.count() == 5


class TestEventLogQuery:
    def _populate(self, event_log: EventLog):
        event_log.log(MemoryEvent(
            event_type=EventType.WEIGHT_UPDATE, turn=1,
            record_id="r1", entity_name="allergy",
        ))
        event_log.log(MemoryEvent(
            event_type=EventType.SUPPRESSION, turn=1,
            record_id="r2", entity_name="medication",
            suppressed_by="r1",
        ))
        event_log.log(MemoryEvent(
            event_type=EventType.WEIGHT_UPDATE, turn=2,
            record_id="r1", entity_name="allergy",
        ))
        event_log.log(MemoryEvent(
            event_type=EventType.PRUNE, turn=3,
            record_id="r3", entity_name="mood",
        ))

    def test_query_all(self, event_log: EventLog):
        self._populate(event_log)
        results = event_log.query(limit=100)
        assert len(results) == 4

    def test_query_by_turn_range(self, event_log: EventLog):
        self._populate(event_log)
        results = event_log.query(turn_from=2, turn_to=2)
        assert len(results) == 1
        assert results[0].turn == 2

    def test_query_by_event_type(self, event_log: EventLog):
        self._populate(event_log)
        results = event_log.query(event_types=[EventType.SUPPRESSION])
        assert len(results) == 1
        assert results[0].event_type == EventType.SUPPRESSION

    def test_query_by_entity(self, event_log: EventLog):
        self._populate(event_log)
        results = event_log.query(entity_name="allergy")
        assert len(results) == 2

    def test_query_by_record_id(self, event_log: EventLog):
        self._populate(event_log)
        results = event_log.query(record_id="r3")
        assert len(results) == 1
        assert results[0].entity_name == "mood"

    def test_query_with_limit(self, event_log: EventLog):
        self._populate(event_log)
        results = event_log.query(limit=2)
        assert len(results) == 2

    def test_query_with_offset(self, event_log: EventLog):
        self._populate(event_log)
        all_events = event_log.query(limit=10)
        offset_events = event_log.query(limit=10, offset=2)
        assert len(offset_events) == len(all_events) - 2

    def test_query_combined_filters(self, event_log: EventLog):
        self._populate(event_log)
        results = event_log.query(
            turn_from=1, turn_to=1,
            event_types=[EventType.WEIGHT_UPDATE],
        )
        assert len(results) == 1


class TestEventLogTimeline:
    def test_timeline_empty(self, event_log: EventLog):
        timeline = event_log.get_timeline()
        assert timeline == []

    def test_timeline_aggregation(self, event_log: EventLog):
        event_log.log(MemoryEvent(
            event_type=EventType.WEIGHT_UPDATE, turn=1,
            record_id="r1", entity_name="a",
        ))
        event_log.log(MemoryEvent(
            event_type=EventType.WEIGHT_UPDATE, turn=1,
            record_id="r2", entity_name="b",
        ))
        event_log.log(MemoryEvent(
            event_type=EventType.LTP, turn=1,
            record_id="r1", entity_name="a",
        ))
        timeline = event_log.get_timeline()
        assert len(timeline) == 2
        types = {row["event_type"]: row["count"] for row in timeline}
        assert types["weight_update"] == 2
        assert types["ltp"] == 1


class TestEventLogEntityHistory:
    def test_entity_history(self, event_log: EventLog):
        event_log.log(MemoryEvent(
            event_type=EventType.WEIGHT_UPDATE, turn=1,
            record_id="r1", entity_name="allergy",
        ))
        event_log.log(MemoryEvent(
            event_type=EventType.SUPPRESSION, turn=2,
            record_id="r2", entity_name="medication",
        ))
        event_log.log(MemoryEvent(
            event_type=EventType.WEIGHT_UPDATE, turn=3,
            record_id="r1", entity_name="allergy",
        ))
        history = event_log.get_entity_history("allergy")
        assert len(history) == 2
        assert all(e.entity_name == "allergy" for e in history)


class TestEventLogCountByType:
    def test_count_by_type(self, event_log: EventLog):
        event_log.log(MemoryEvent(
            event_type=EventType.WEIGHT_UPDATE, turn=1,
            record_id="r1", entity_name="a",
        ))
        event_log.log(MemoryEvent(
            event_type=EventType.WEIGHT_UPDATE, turn=1,
            record_id="r2", entity_name="b",
        ))
        event_log.log(MemoryEvent(
            event_type=EventType.PRUNE, turn=2,
            record_id="r3", entity_name="c",
        ))
        counts = event_log.count_by_type()
        assert counts["weight_update"] == 2
        assert counts["prune"] == 1


class TestEventSchema:
    def test_all_event_types(self):
        for et in EventType:
            event = MemoryEvent(
                event_type=et,
                turn=1,
                record_id="r1",
                entity_name="test",
            )
            assert event.event_type == et

    def test_optional_fields_default_none(self):
        event = MemoryEvent(
            event_type=EventType.WEIGHT_UPDATE,
            turn=1,
            record_id="r1",
            entity_name="test",
        )
        assert event.old_weight is None
        assert event.new_weight is None
        assert event.suppressed_by is None
        assert event.old_k is None
        assert event.new_k is None
        assert event.retrieval_score is None
        assert event.hops is None
        assert event.metadata == {}

    def test_full_event(self):
        event = MemoryEvent(
            event_type=EventType.SUPPRESSION,
            turn=5,
            record_id="rec-abc",
            entity_name="user_allergy",
            old_weight=0.85,
            new_weight=0.085,
            suppressed_by="rec-xyz",
            old_memory_type="ActiveContext",
            metadata={"reason": "contradiction"},
        )
        assert event.old_weight == 0.85
        assert event.new_weight == 0.085
        assert event.suppressed_by == "rec-xyz"
        assert event.metadata["reason"] == "contradiction"


class TestEngineEventIntegration:
    def test_decay_all_emits_events(self, event_log: EventLog):
        records = [
            make_record("allergy", weight=1.0),
            make_record("mood", weight=0.5, memory_type=MemoryType.EPHEMERAL_STATE),
        ]
        config = DecayConfig()
        decay_all(records, current_turn=5, config=config, event_logger=event_log)
        events = event_log.query(event_types=[EventType.WEIGHT_UPDATE])
        assert len(events) >= 1
        for e in events:
            assert e.old_weight is not None
            assert e.new_weight is not None

    def test_prune_cycle_emits_events(self, event_log: EventLog):
        from nmafc.storage.cold import ColdStorage
        from nmafc.storage.config import StorageConfig
        from nmafc.storage.hot import HotStorage
        import tempfile, os

        embed_dim = 8
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(
                hot_uri=os.path.join(tmpdir, "hot"),
                cold_uri=os.path.join(tmpdir, "cold.db"),
                embedding_dim=embed_dim,
            )
            hot = HotStorage(config)
            cold = ColdStorage(config.cold_uri)

            rec = make_record("decayed_thing", weight=0.05)
            from nmafc.engine.decay import compute_weight
            hot.upsert(rec, [0.1] * embed_dim)

            pruned = prune_cycle(
                hot, cold, w_prune=0.1, current_turn=10,
                event_logger=event_log,
            )
            events = event_log.query(event_types=[EventType.PRUNE])
            assert len(events) == pruned
            assert events[0].entity_name == "decayed_thing"

            cold.close()

    def test_suppression_event_creation(self):
        rec = make_record("old_fact", weight=0.8, memory_type=MemoryType.ACTIVE_CONTEXT)
        suppressed = apply_suppression(rec, gamma=0.1)
        event = create_suppression_event(
            rec, suppressed.weight, suppressed_by="new_fact", turn=5,
        )
        assert event.event_type == EventType.SUPPRESSION
        assert event.old_weight == 0.8
        assert event.new_weight == pytest.approx(0.08)
        assert event.suppressed_by == "new_fact"

    def test_ltp_events_creation(self):
        records = [make_record("fact1"), make_record("fact2")]
        reinforced = batch_reinforce(records, current_turn=3)
        events = create_ltp_events(records, reinforced, current_turn=3)
        assert len(events) == 2
        for e in events:
            assert e.event_type == EventType.LTP
            assert e.old_k == 0
            assert e.new_k == 1
