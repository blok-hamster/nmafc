import pytest
from pydantic import ValidationError

from nmafc.schemas.memory import (
    DecayConfig,
    MemoryRecord,
    MemoryStateUpdate,
    MemoryType,
    SearchResult,
    UnifiedMemoryPayload,
)


class TestMemoryType:
    def test_values(self):
        assert MemoryType.CORE_ANCHOR == "CoreAnchor"
        assert MemoryType.ACTIVE_CONTEXT == "ActiveContext"
        assert MemoryType.EPHEMERAL_STATE == "EphemeralState"

    def test_from_string(self):
        assert MemoryType("CoreAnchor") == MemoryType.CORE_ANCHOR


class TestMemoryStateUpdate:
    def test_valid_update(self):
        update = MemoryStateUpdate(
            entity_name="medication_morning",
            fact_content="User takes aspirin at 9 AM",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        )
        assert update.entity_name == "medication_morning"
        assert update.overrides_entity is None

    def test_with_override(self):
        update = MemoryStateUpdate(
            entity_name="medication_morning",
            fact_content="User takes aspirin at 11 AM",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            overrides_entity="medication_morning",
        )
        assert update.overrides_entity == "medication_morning"

    def test_empty_entity_name_rejected(self):
        with pytest.raises(ValidationError):
            MemoryStateUpdate(
                entity_name="",
                fact_content="Some fact",
                memory_type=MemoryType.EPHEMERAL_STATE,
            )

    def test_empty_fact_content_rejected(self):
        with pytest.raises(ValidationError):
            MemoryStateUpdate(
                entity_name="entity",
                fact_content="",
                memory_type=MemoryType.EPHEMERAL_STATE,
            )

    def test_invalid_memory_type_rejected(self):
        with pytest.raises(ValidationError):
            MemoryStateUpdate(
                entity_name="entity",
                fact_content="fact",
                memory_type="InvalidType",  # type: ignore[arg-type]
            )


class TestUnifiedMemoryPayload:
    def test_empty_payload(self):
        payload = UnifiedMemoryPayload()
        assert payload.updates == []

    def test_payload_with_updates(self):
        payload = UnifiedMemoryPayload(
            updates=[
                MemoryStateUpdate(
                    entity_name="user_name",
                    fact_content="User is named Alice",
                    memory_type=MemoryType.CORE_ANCHOR,
                ),
                MemoryStateUpdate(
                    entity_name="user_mood",
                    fact_content="User seems happy today",
                    memory_type=MemoryType.EPHEMERAL_STATE,
                ),
            ]
        )
        assert len(payload.updates) == 2


class TestMemoryRecord:
    def test_defaults(self):
        record = MemoryRecord(
            entity_name="test",
            fact_content="test fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        )
        assert record.weight == 1.0
        assert record.consolidation_index == 0
        assert record.created_at_turn == 0
        assert record.last_reinforced_turn == 0
        assert record.is_active is True
        assert len(record.id) > 0

    def test_weight_bounds(self):
        with pytest.raises(ValidationError):
            MemoryRecord(
                entity_name="test",
                fact_content="fact",
                memory_type=MemoryType.CORE_ANCHOR,
                weight=1.5,
            )
        with pytest.raises(ValidationError):
            MemoryRecord(
                entity_name="test",
                fact_content="fact",
                memory_type=MemoryType.CORE_ANCHOR,
                weight=-0.1,
            )

    def test_unique_ids(self):
        r1 = MemoryRecord(
            entity_name="a", fact_content="f", memory_type=MemoryType.CORE_ANCHOR
        )
        r2 = MemoryRecord(
            entity_name="b", fact_content="g", memory_type=MemoryType.CORE_ANCHOR
        )
        assert r1.id != r2.id


class TestSearchResult:
    def test_valid(self):
        record = MemoryRecord(
            entity_name="test",
            fact_content="fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        )
        result = SearchResult(record=record, score=0.85)
        assert result.score == 0.85

    def test_score_bounds(self):
        record = MemoryRecord(
            entity_name="test",
            fact_content="fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        )
        with pytest.raises(ValidationError):
            SearchResult(record=record, score=1.5)


class TestDecayConfig:
    def test_defaults(self):
        config = DecayConfig()
        assert config.lambda_core_anchor == 0.0
        assert config.lambda_active_context == 0.05
        assert config.lambda_ephemeral == 0.69
        assert config.eta == 0.15
        assert config.gamma == 0.1
        assert config.w_prune == 0.1
        assert config.theta == 0.75
        assert config.top_k == 10

    def test_get_lambda_base(self):
        config = DecayConfig()
        assert config.get_lambda_base(MemoryType.CORE_ANCHOR) == 0.0
        assert config.get_lambda_base(MemoryType.ACTIVE_CONTEXT) == 0.05
        assert config.get_lambda_base(MemoryType.EPHEMERAL_STATE) == 0.69

    def test_invalid_eta_rejected(self):
        with pytest.raises(ValidationError):
            DecayConfig(eta=0.0)

    def test_invalid_gamma_rejected(self):
        with pytest.raises(ValidationError):
            DecayConfig(gamma=1.5)
