import tempfile
from pathlib import Path

import pytest

from nmafc.engine.rollback import invalidate_event, rebuild_hot_from_cold
from nmafc.schemas.memory import DecayConfig, MemoryStateUpdate, MemoryType
from nmafc.storage.cold import ColdStorage
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage

EMBED_DIM = 8


def fake_embed(text: str) -> list[float]:
    seed = sum(ord(c) for c in text) % 100 / 100.0
    return [seed + i * 0.01 for i in range(EMBED_DIM)]


@pytest.fixture
def storage():
    tmp = tempfile.mkdtemp()
    hot_config = StorageConfig(
        hot_uri=str(Path(tmp) / "lance"),
        embedding_dim=EMBED_DIM,
    )
    hot = HotStorage(hot_config)
    cold = ColdStorage(str(Path(tmp) / "cold.db"))
    return hot, cold


@pytest.fixture
def config() -> DecayConfig:
    return DecayConfig()


class TestRebuildHotFromCold:
    def test_rebuilds_from_events(self, storage, config: DecayConfig):
        hot, cold = storage

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
                entity_name="medication",
                fact_content="Aspirin at 9AM",
                memory_type=MemoryType.ACTIVE_CONTEXT,
            ),
            turn=2,
        )
        cold.append_event(
            MemoryStateUpdate(
                entity_name="mood",
                fact_content="Happy",
                memory_type=MemoryType.EPHEMERAL_STATE,
            ),
            turn=3,
        )

        restored = rebuild_hot_from_cold(cold, hot, fake_embed, config, up_to_turn=10)
        # Core anchor survives, active context might survive depending on decay,
        # ephemeral likely pruned after 7 turns
        assert restored >= 1  # At least the core anchor
        assert hot.count() >= 1

    def test_respects_up_to_turn(self, storage, config: DecayConfig):
        hot, cold = storage

        cold.append_event(
            MemoryStateUpdate(
                entity_name="early",
                fact_content="Early fact",
                memory_type=MemoryType.CORE_ANCHOR,
            ),
            turn=1,
        )
        cold.append_event(
            MemoryStateUpdate(
                entity_name="late",
                fact_content="Late fact",
                memory_type=MemoryType.CORE_ANCHOR,
            ),
            turn=10,
        )

        restored = rebuild_hot_from_cold(cold, hot, fake_embed, config, up_to_turn=5)
        assert restored == 1
        records = hot.get_by_entity("early")
        assert len(records) == 1
        assert hot.get_by_entity("late") == []

    def test_clears_existing_hot(self, storage, config: DecayConfig):
        hot, cold = storage

        from nmafc.schemas.memory import MemoryRecord

        hot.upsert(
            MemoryRecord(
                entity_name="old",
                fact_content="stale",
                memory_type=MemoryType.ACTIVE_CONTEXT,
            ),
            fake_embed("stale"),
        )
        assert hot.count() == 1

        cold.append_event(
            MemoryStateUpdate(
                entity_name="new",
                fact_content="fresh",
                memory_type=MemoryType.CORE_ANCHOR,
            ),
            turn=1,
        )

        rebuild_hot_from_cold(cold, hot, fake_embed, config, up_to_turn=5)
        assert hot.get_by_entity("old") == []
        assert len(hot.get_by_entity("new")) == 1

    def test_skips_inactive_events(self, storage, config: DecayConfig):
        hot, cold = storage

        event_id = cold.append_event(
            MemoryStateUpdate(
                entity_name="bad",
                fact_content="hallucinated",
                memory_type=MemoryType.CORE_ANCHOR,
            ),
            turn=1,
        )
        cold.mark_inactive(event_id)
        cold.append_event(
            MemoryStateUpdate(
                entity_name="good",
                fact_content="correct",
                memory_type=MemoryType.CORE_ANCHOR,
            ),
            turn=2,
        )

        restored = rebuild_hot_from_cold(cold, hot, fake_embed, config, up_to_turn=5)
        assert restored == 1
        assert hot.get_by_entity("bad") == []
        assert len(hot.get_by_entity("good")) == 1


class TestInvalidateEvent:
    def test_marks_cold_inactive_and_removes_from_hot(self, storage):
        hot, cold = storage

        event_id = cold.append_event(
            MemoryStateUpdate(
                entity_name="target",
                fact_content="to be invalidated",
                memory_type=MemoryType.ACTIVE_CONTEXT,
            ),
            turn=1,
        )

        from nmafc.schemas.memory import MemoryRecord

        hot.upsert(
            MemoryRecord(
                entity_name="target",
                fact_content="to be invalidated",
                memory_type=MemoryType.ACTIVE_CONTEXT,
            ),
            fake_embed("to be invalidated"),
        )
        assert hot.count() == 1

        invalidate_event(cold, hot, event_id, "target")
        assert cold.count_active() == 0
        assert hot.count() == 0
