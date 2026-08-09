import math

import pytest

from nmafc.engine.decay import (
    compute_alpha,
    compute_lambda,
    compute_weight,
    decay_all,
    decay_record,
)
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType


@pytest.fixture
def config() -> DecayConfig:
    return DecayConfig()


class TestComputeAlpha:
    def test_k_zero_returns_one(self, config: DecayConfig):
        assert compute_alpha(0, config.eta) == 1.0

    def test_decreases_monotonically(self, config: DecayConfig):
        values = [compute_alpha(k, config.eta) for k in range(10)]
        for i in range(1, len(values)):
            assert values[i] < values[i - 1]

    def test_approaches_zero(self, config: DecayConfig):
        assert compute_alpha(50, config.eta) < 0.001


class TestComputeLambda:
    def test_core_anchor_always_zero(self, config: DecayConfig):
        for k in range(20):
            assert compute_lambda(MemoryType.CORE_ANCHOR, k, config) == 0.0

    def test_ephemeral_at_k0(self, config: DecayConfig):
        lam = compute_lambda(MemoryType.EPHEMERAL_STATE, 0, config)
        assert lam == config.lambda_ephemeral

    def test_active_context_with_consolidation(self, config: DecayConfig):
        lam_k0 = compute_lambda(MemoryType.ACTIVE_CONTEXT, 0, config)
        lam_k5 = compute_lambda(MemoryType.ACTIVE_CONTEXT, 5, config)
        assert lam_k5 < lam_k0


class TestComputeWeight:
    def test_zero_lambda_no_decay(self):
        assert compute_weight(1.0, 0.0, 100) == 1.0

    def test_zero_delta_no_decay(self):
        assert compute_weight(0.8, 0.5, 0) == 0.8

    def test_exponential_decay(self):
        w = compute_weight(1.0, 0.69, 1)
        assert abs(w - math.exp(-0.69)) < 0.001

    def test_weight_decreases_with_time(self):
        w1 = compute_weight(1.0, 0.1, 5)
        w2 = compute_weight(1.0, 0.1, 10)
        assert w2 < w1


class TestDecayRecord:
    def test_core_anchor_never_decays(self, config: DecayConfig):
        record = MemoryRecord(
            entity_name="identity",
            fact_content="User is Alice",
            memory_type=MemoryType.CORE_ANCHOR,
            weight=1.0,
            last_reinforced_turn=0,
        )
        assert decay_record(record, current_turn=1000, config=config) == 1.0

    def test_ephemeral_decays_fast(self, config: DecayConfig):
        record = MemoryRecord(
            entity_name="mood",
            fact_content="User is happy",
            memory_type=MemoryType.EPHEMERAL_STATE,
            weight=1.0,
            consolidation_index=0,
            last_reinforced_turn=0,
        )
        w = decay_record(record, current_turn=4, config=config)
        assert w < config.w_prune  # e^{-0.69*4} ≈ 0.063, below 0.1 threshold

    def test_active_context_moderate_decay(self, config: DecayConfig):
        record = MemoryRecord(
            entity_name="schedule",
            fact_content="Meeting at 3PM",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0,
            consolidation_index=0,
            last_reinforced_turn=0,
        )
        w = decay_record(record, current_turn=10, config=config)
        expected = math.exp(-0.05 * 10)
        assert abs(w - expected) < 0.001

    def test_high_consolidation_resists_decay(self, config: DecayConfig):
        record_k0 = MemoryRecord(
            entity_name="fact",
            fact_content="Important",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0,
            consolidation_index=0,
            last_reinforced_turn=0,
        )
        record_k10 = record_k0.model_copy(update={"consolidation_index": 10})
        w_k0 = decay_record(record_k0, current_turn=20, config=config)
        w_k10 = decay_record(record_k10, current_turn=20, config=config)
        assert w_k10 > w_k0

    def test_no_decay_when_current_equals_reinforced(self, config: DecayConfig):
        record = MemoryRecord(
            entity_name="x",
            fact_content="y",
            memory_type=MemoryType.EPHEMERAL_STATE,
            weight=0.8,
            last_reinforced_turn=5,
        )
        assert decay_record(record, current_turn=5, config=config) == 0.8


class TestDecayAll:
    def test_skips_core_anchors(self, config: DecayConfig):
        records = [
            MemoryRecord(
                entity_name="anchor",
                fact_content="fact",
                memory_type=MemoryType.CORE_ANCHOR,
                weight=1.0,
                last_reinforced_turn=0,
            ),
            MemoryRecord(
                entity_name="active",
                fact_content="fact",
                memory_type=MemoryType.ACTIVE_CONTEXT,
                weight=1.0,
                last_reinforced_turn=0,
            ),
        ]
        results = decay_all(records, current_turn=5, config=config)
        ids = [r[0] for r in results]
        assert records[0].id not in ids
        assert records[1].id in ids

    def test_returns_correct_weights(self, config: DecayConfig):
        record = MemoryRecord(
            entity_name="test",
            fact_content="fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0,
            consolidation_index=0,
            last_reinforced_turn=0,
        )
        results = decay_all([record], current_turn=10, config=config)
        expected = math.exp(-0.05 * 10)
        assert abs(results[0][1] - expected) < 0.001
