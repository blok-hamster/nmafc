import tempfile
from pathlib import Path

import pytest

from nmafc.engine.pruning import (
    apply_suppression,
    detect_override,
    identify_prunable,
    prune_cycle,
)
from nmafc.schemas.memory import MemoryRecord, MemoryStateUpdate, MemoryType
from nmafc.storage.cold import ColdStorage
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage

EMBED_DIM = 8


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


def make_embedding(seed: float = 0.1) -> list[float]:
    return [seed + i * 0.01 for i in range(EMBED_DIM)]


class TestDetectOverride:
    def test_explicit_override(self):
        existing = [make_record("medication_morning")]
        update = MemoryStateUpdate(
            entity_name="medication_morning_new",
            fact_content="Takes at 11 AM now",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            overrides_entity="medication_morning",
        )
        targets = detect_override(update, existing)
        assert len(targets) == 1
        assert targets[0].entity_name == "medication_morning"

    def test_implicit_override_same_entity(self):
        existing = [make_record("medication_morning")]
        update = MemoryStateUpdate(
            entity_name="medication_morning",
            fact_content="Takes at 11 AM now",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        )
        targets = detect_override(update, existing)
        assert len(targets) == 1

    def test_no_override_different_entity(self):
        existing = [make_record("hobby")]
        update = MemoryStateUpdate(
            entity_name="medication_morning",
            fact_content="Takes aspirin",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        )
        targets = detect_override(update, existing)
        assert len(targets) == 0

    def test_multiple_overrides(self):
        existing = [
            make_record("medication_morning"),
            make_record("medication_morning"),
        ]
        update = MemoryStateUpdate(
            entity_name="medication_morning",
            fact_content="New schedule",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        )
        targets = detect_override(update, existing)
        assert len(targets) == 2


class TestApplySuppression:
    def test_reduces_weight(self):
        record = make_record("test", weight=1.0)
        suppressed = apply_suppression(record, gamma=0.1)
        assert abs(suppressed.weight - 0.1) < 0.001

    def test_at_threshold(self):
        record = make_record("test", weight=1.0)
        suppressed = apply_suppression(record, gamma=0.1)
        assert suppressed.weight <= 0.1

    def test_below_threshold(self):
        record = make_record("test", weight=0.5)
        suppressed = apply_suppression(record, gamma=0.1)
        assert suppressed.weight == pytest.approx(0.05)

    def test_does_not_mutate_original(self):
        record = make_record("test", weight=1.0)
        apply_suppression(record, gamma=0.1)
        assert record.weight == 1.0


class TestIdentifyPrunable:
    def test_finds_below_threshold(self):
        records = [
            make_record("a", weight=0.5),
            make_record("b", weight=0.05),
            make_record("c", weight=0.09),
            make_record("d", weight=0.15),
        ]
        prunable = identify_prunable(records, w_prune=0.1)
        assert len(prunable) == 2

    def test_empty_when_all_above(self):
        records = [make_record("a", weight=0.5), make_record("b", weight=1.0)]
        prunable = identify_prunable(records, w_prune=0.1)
        assert prunable == []

    def test_empty_list(self):
        assert identify_prunable([], w_prune=0.1) == []


class TestPruneCycle:
    @pytest.fixture
    def storage(self):
        tmp = tempfile.mkdtemp()
        hot_config = StorageConfig(
            hot_uri=str(Path(tmp) / "lance"),
            embedding_dim=EMBED_DIM,
        )
        hot = HotStorage(hot_config)
        cold = ColdStorage(str(Path(tmp) / "cold.db"))
        return hot, cold

    def test_prunes_below_threshold(self, storage):
        """ActiveContext below threshold is invalidated (not deleted); EphemeralState is deleted."""
        hot, cold = storage
        r_low = make_record("low", weight=0.05)
        r_high = make_record("high", weight=0.8)
        hot.upsert(r_low, make_embedding(0.1))
        hot.upsert(r_high, make_embedding(0.2))
        assert hot.count() == 2

        pruned = prune_cycle(hot, cold, w_prune=0.1, current_turn=5)
        assert pruned == 1
        # ActiveContext is invalidated not deleted — still in store but with invalid_at set
        assert hot.count() == 2
        invalidated = hot.get_record(r_low.id)
        assert invalidated is not None
        assert invalidated.invalid_at == 5

    def test_prunes_ephemeral_below_threshold(self, storage):
        """EphemeralState below threshold is physically deleted."""
        hot, cold = storage
        r_low = make_record("low", weight=0.05, memory_type=MemoryType.EPHEMERAL_STATE)
        r_high = make_record("high", weight=0.8)
        hot.upsert(r_low, make_embedding(0.1))
        hot.upsert(r_high, make_embedding(0.2))
        assert hot.count() == 2

        pruned = prune_cycle(hot, cold, w_prune=0.1, current_turn=5)
        assert pruned == 1
        assert hot.count() == 1

    def test_prunes_nothing_when_all_healthy(self, storage):
        hot, cold = storage
        hot.upsert(make_record("a", weight=0.5), make_embedding(0.1))
        hot.upsert(make_record("b", weight=1.0), make_embedding(0.2))
        pruned = prune_cycle(hot, cold, w_prune=0.1, current_turn=5)
        assert pruned == 0
        assert hot.count() == 2
