import math
import tempfile
from pathlib import Path

import pytest

from nmafc.schemas.memory import MemoryRecord, MemoryType
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage

EMBED_DIM = 8


@pytest.fixture
def hot() -> HotStorage:
    tmp = tempfile.mkdtemp()
    config = StorageConfig(
        hot_uri=str(Path(tmp) / "test_lance"),
        embedding_dim=EMBED_DIM,
    )
    return HotStorage(config)


def make_embedding(seed: float = 0.1) -> list[float]:
    return [seed + i * 0.01 for i in range(EMBED_DIM)]


def make_record(entity: str = "test_entity", **kwargs) -> MemoryRecord:
    defaults = {
        "entity_name": entity,
        "fact_content": f"Fact about {entity}",
        "memory_type": MemoryType.ACTIVE_CONTEXT,
        "weight": 1.0,
        "consolidation_index": 0,
        "created_at_turn": 1,
        "last_reinforced_turn": 1,
    }
    defaults.update(kwargs)
    return MemoryRecord(**defaults)


class TestUpsertAndCount:
    def test_upsert_adds_record(self, hot: HotStorage):
        record = make_record()
        hot.upsert(record, make_embedding())
        assert hot.count() == 1

    def test_upsert_same_id_replaces(self, hot: HotStorage):
        record = make_record()
        hot.upsert(record, make_embedding(0.1))
        hot.upsert(record, make_embedding(0.5))
        assert hot.count() == 1

    def test_multiple_records(self, hot: HotStorage):
        for i in range(5):
            r = make_record(entity=f"entity_{i}")
            hot.upsert(r, make_embedding(seed=i * 0.1))
        assert hot.count() == 5


class TestSearch:
    def test_search_returns_results(self, hot: HotStorage):
        record = make_record()
        emb = make_embedding(0.1)
        hot.upsert(record, emb)
        results = hot.search(emb, top_k=5)
        assert len(results) == 1
        assert results[0].record.entity_name == "test_entity"
        assert results[0].score > 0.9

    def test_search_respects_top_k(self, hot: HotStorage):
        for i in range(10):
            r = make_record(entity=f"entity_{i}")
            hot.upsert(r, make_embedding(seed=i * 0.05))
        results = hot.search(make_embedding(0.1), top_k=3)
        assert len(results) == 3

    def test_search_empty_table(self, hot: HotStorage):
        results = hot.search(make_embedding(), top_k=5)
        assert results == []


class TestSearchScoreIsCosineSimilarity:
    """`score` is consumed as a similarity in [0, 1] and compared to DecayConfig.theta.

    LanceDB defaults to L2, under which `1 - distance` is not a similarity at
    all. These pin the metric by asserting the exact analytic cosine value: a
    query at 45 degrees to a stored vector scores 0.707 under cosine but 0.0
    under the L2 default, so a silent regression to L2 fails loudly here rather
    than by making every retrieval threshold unreachable in production.
    """

    def unit(self, *values: float) -> list[float]:
        return list(values) + [0.0] * (EMBED_DIM - len(values))

    def test_identical_vector_scores_one(self, hot: HotStorage):
        emb = self.unit(1.0)
        hot.upsert(make_record(), emb)
        results = hot.search(emb, top_k=1)
        assert results[0].score == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_vector_scores_zero(self, hot: HotStorage):
        hot.upsert(make_record(), self.unit(1.0))
        results = hot.search(self.unit(0.0, 1.0), top_k=1)
        assert results[0].score == pytest.approx(0.0, abs=1e-4)

    def test_partial_overlap_matches_analytic_cosine(self, hot: HotStorage):
        hot.upsert(make_record(), self.unit(1.0))
        results = hot.search(self.unit(1.0, 1.0), top_k=1)
        assert results[0].score == pytest.approx(0.70710678, abs=1e-4)

    def test_magnitude_does_not_change_the_score(self, hot: HotStorage):
        """Cosine ignores length; L2 does not. Embedders do not always normalise."""
        hot.upsert(make_record(), self.unit(1.0))
        short = hot.search(self.unit(1.0, 1.0), top_k=1)[0].score
        long = hot.search(self.unit(9.0, 9.0), top_k=1)[0].score
        assert short == pytest.approx(long, abs=1e-4)


class TestGetByEntity:
    def test_finds_matching_entity(self, hot: HotStorage):
        hot.upsert(make_record(entity="medication"), make_embedding(0.1))
        hot.upsert(make_record(entity="hobby"), make_embedding(0.2))
        results = hot.get_by_entity("medication")
        assert len(results) == 1
        assert results[0].entity_name == "medication"

    def test_returns_empty_for_missing(self, hot: HotStorage):
        hot.upsert(make_record(entity="exists"), make_embedding())
        results = hot.get_by_entity("missing")
        assert results == []


class TestDelete:
    def test_delete_removes_record(self, hot: HotStorage):
        record = make_record()
        hot.upsert(record, make_embedding())
        assert hot.count() == 1
        hot.delete(record.id)
        assert hot.count() == 0

    def test_delete_nonexistent_is_safe(self, hot: HotStorage):
        hot.delete("nonexistent-id")


class TestUpdateWeight:
    def test_updates_weight(self, hot: HotStorage):
        record = make_record(weight=1.0)
        hot.upsert(record, make_embedding())
        hot.update_weight(record.id, 0.5)
        updated = hot.get_record(record.id)
        assert updated is not None
        assert abs(updated.weight - 0.5) < 0.01


class TestApplyWeightUpdates:
    def test_applies_weights(self, hot: HotStorage):
        records = [make_record(entity=f"e{i}") for i in range(3)]
        for i, rec in enumerate(records):
            hot.upsert(rec, make_embedding(0.1 + i * 0.05))

        hot.apply_weight_updates([(rec.id, 0.5) for rec in records])

        for rec in records:
            got = hot.get_record(rec.id)
            assert got is not None
            assert abs(got.weight - 0.5) < 1e-6

    def test_decay_turn_advances_the_clock(self, hot: HotStorage):
        record = make_record(last_reinforced_turn=1)
        hot.upsert(record, make_embedding())

        hot.apply_weight_updates([(record.id, 0.9)], decay_turn=7)

        got = hot.get_record(record.id)
        assert got is not None
        assert got.last_reinforced_turn == 7

    def test_clock_is_left_alone_without_decay_turn(self, hot: HotStorage):
        """Suppression and other non-decay writes must not reset the decay clock."""
        record = make_record(last_reinforced_turn=1)
        hot.upsert(record, make_embedding())

        hot.apply_weight_updates([(record.id, 0.4)])

        got = hot.get_record(record.id)
        assert got is not None
        assert got.last_reinforced_turn == 1

    def test_repeated_passes_match_the_closed_form(self, hot: HotStorage):
        """Decay over n turns must equal w0 * e^{-lambda*n}, not compound.

        The stored weight is read back as w0 on the next pass. Without the clock
        advancing, turn n applies the whole elapsed span to a weight that has
        already absorbed n-1 turns of it, giving e^{-lambda*n(n+1)/2}: an
        ActiveContext record hits the 0.1 prune threshold at turn 10 rather than
        turn 46. This pins the arithmetic end to end through storage, which is
        where the double-count lived; decay_record() alone always looked right.
        """
        from nmafc.engine.decay import decay_all
        from nmafc.schemas.memory import DecayConfig

        config = DecayConfig()
        turns = 12
        record = make_record(
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0,
            consolidation_index=0,
            created_at_turn=0,
            last_reinforced_turn=0,
        )
        hot.upsert(record, make_embedding())

        for turn in range(1, turns + 1):
            updates = decay_all(hot.get_all_mutable(), turn, config)
            hot.apply_weight_updates(updates, decay_turn=turn)

        got = hot.get_record(record.id)
        assert got is not None
        expected = math.exp(-config.lambda_active_context * turns)
        assert got.weight == pytest.approx(expected, rel=1e-6)

    def test_core_anchors_keep_their_clock(self, hot: HotStorage):
        """decay_all skips anchors, so the pass must never stamp one."""
        from nmafc.engine.decay import decay_all
        from nmafc.schemas.memory import DecayConfig

        anchor = make_record(
            entity="identity",
            memory_type=MemoryType.CORE_ANCHOR,
            last_reinforced_turn=1,
        )
        hot.upsert(anchor, make_embedding())

        hot.apply_weight_updates(
            decay_all(hot.get_all_mutable(), 9, DecayConfig()), decay_turn=9
        )

        got = hot.get_record(anchor.id)
        assert got is not None
        assert got.weight == 1.0
        assert got.last_reinforced_turn == 1


class TestUpdateReinforcement:
    def test_updates_k_and_turn(self, hot: HotStorage):
        record = make_record(consolidation_index=2, last_reinforced_turn=5)
        hot.upsert(record, make_embedding())
        hot.update_reinforcement(record.id, new_k=3, turn=10)
        updated = hot.get_record(record.id)
        assert updated is not None
        assert updated.consolidation_index == 3
        assert updated.last_reinforced_turn == 10
        assert updated.weight == 1.0


class TestApplyReinforcements:
    def test_batch_matches_sequential_loop(self, hot: HotStorage):
        """The batched path must be indistinguishable from the per-record loop.

        Retrieval reinforces every record Spreading Activation surfaces, so this
        runs on the hot path for every question; a divergence here would silently
        corrupt weights rather than fail loudly.
        """
        records = [
            make_record(entity=f"e{i}", consolidation_index=i, last_reinforced_turn=1)
            for i in range(5)
        ]
        for i, rec in enumerate(records):
            hot.upsert(rec, make_embedding(0.1 + i * 0.05))

        updates = [(rec.id, i + 10) for i, rec in enumerate(records)]
        hot.apply_reinforcements(updates, turn=42)

        for rec, (_, new_k) in zip(records, updates):
            got = hot.get_record(rec.id)
            assert got is not None
            assert got.consolidation_index == new_k
            assert got.last_reinforced_turn == 42
            assert got.weight == 1.0

    def test_leaves_untouched_records_alone(self, hot: HotStorage):
        target = make_record(entity="target", consolidation_index=0)
        other = make_record(entity="other", consolidation_index=7, weight=0.4)
        hot.upsert(target, make_embedding(0.1))
        hot.upsert(other, make_embedding(0.9))

        hot.apply_reinforcements([(target.id, 3)], turn=20)

        untouched = hot.get_record(other.id)
        assert untouched is not None
        assert untouched.consolidation_index == 7
        assert abs(untouched.weight - 0.4) < 0.01
        assert untouched.last_reinforced_turn == 1

    def test_empty_updates_is_a_noop(self, hot: HotStorage):
        record = make_record(consolidation_index=2)
        hot.upsert(record, make_embedding())
        hot.apply_reinforcements([], turn=99)
        got = hot.get_record(record.id)
        assert got is not None
        assert got.consolidation_index == 2
        assert got.last_reinforced_turn == 1


class TestGetAllMutable:
    def test_excludes_core_anchors(self, hot: HotStorage):
        hot.upsert(
            make_record(entity="anchor", memory_type=MemoryType.CORE_ANCHOR),
            make_embedding(0.1),
        )
        hot.upsert(
            make_record(entity="active", memory_type=MemoryType.ACTIVE_CONTEXT),
            make_embedding(0.2),
        )
        hot.upsert(
            make_record(entity="ephemeral", memory_type=MemoryType.EPHEMERAL_STATE),
            make_embedding(0.3),
        )
        mutable = hot.get_all_mutable()
        types = {r.memory_type for r in mutable}
        assert MemoryType.CORE_ANCHOR not in types
        assert len(mutable) == 2


class TestClear:
    def test_clear_removes_all(self, hot: HotStorage):
        for i in range(3):
            hot.upsert(make_record(entity=f"e{i}"), make_embedding(i * 0.1))
        assert hot.count() == 3
        hot.clear()
        assert hot.count() == 0
