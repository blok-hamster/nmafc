from nmafc.engine.reinforcement import batch_reinforce, reinforce
from nmafc.schemas.memory import MemoryRecord, MemoryType


class TestReinforce:
    def test_resets_weight_to_one(self):
        record = MemoryRecord(
            entity_name="test",
            fact_content="fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=0.3,
            consolidation_index=2,
            last_reinforced_turn=5,
        )
        result = reinforce(record, current_turn=10)
        assert result.weight == 1.0

    def test_increments_consolidation_index(self):
        record = MemoryRecord(
            entity_name="test",
            fact_content="fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            consolidation_index=3,
            last_reinforced_turn=5,
        )
        result = reinforce(record, current_turn=10)
        assert result.consolidation_index == 4

    def test_updates_last_reinforced_turn(self):
        record = MemoryRecord(
            entity_name="test",
            fact_content="fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            last_reinforced_turn=5,
        )
        result = reinforce(record, current_turn=15)
        assert result.last_reinforced_turn == 15

    def test_does_not_mutate_original(self):
        record = MemoryRecord(
            entity_name="test",
            fact_content="fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=0.5,
            consolidation_index=1,
            last_reinforced_turn=3,
        )
        reinforce(record, current_turn=10)
        assert record.weight == 0.5
        assert record.consolidation_index == 1
        assert record.last_reinforced_turn == 3

    def test_preserves_id(self):
        record = MemoryRecord(
            entity_name="test",
            fact_content="fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        )
        result = reinforce(record, current_turn=5)
        assert result.id == record.id

    def test_multiple_reinforcements_compound(self):
        record = MemoryRecord(
            entity_name="test",
            fact_content="fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            consolidation_index=0,
        )
        for turn in range(1, 6):
            record = reinforce(record, current_turn=turn)
        assert record.consolidation_index == 5
        assert record.weight == 1.0
        assert record.last_reinforced_turn == 5


class TestBatchReinforce:
    def test_reinforces_all(self):
        records = [
            MemoryRecord(
                entity_name=f"e{i}",
                fact_content=f"f{i}",
                memory_type=MemoryType.ACTIVE_CONTEXT,
                weight=0.5,
                consolidation_index=i,
                last_reinforced_turn=0,
            )
            for i in range(3)
        ]
        results = batch_reinforce(records, current_turn=10)
        assert len(results) == 3
        for r in results:
            assert r.weight == 1.0
            assert r.last_reinforced_turn == 10

    def test_empty_list(self):
        assert batch_reinforce([], current_turn=5) == []
