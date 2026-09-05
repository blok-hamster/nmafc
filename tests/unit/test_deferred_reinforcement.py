"""LTP writeback is buffered, and the buffer cannot quietly swallow a write.

Reinforcement rewrites every surviving record through a delete and an add. On
an append-only store that appends a table version per query, so the cost is
paid on the read path and grows with the run: measured over 240 retrievals
against a copy of a finished LoCoMo store, immediate writeback ran at a mean of
501 ms and rising (420 ms over the first forty queries, 561 ms over the last
eighty) while deferred ran at 300 ms and flat.

Deferring is therefore the default, which makes the failure modes of the buffer
part of the shipped behaviour rather than a benchmark-only concern. Three of
them matter and are covered here: a buffered write must survive to the store, a
record retrieved twice before a flush must end up with the same consolidation
index the immediate path would have given it, and a caller that only ever
retrieves must not accumulate an unbounded buffer that is never written.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nmafc.integration.base import EmbeddingProvider
from nmafc.integration.query_router import QueryRouter
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType
from nmafc.storage.cold import ColdStorage
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage

EMBED_DIM = 3


class FixedEmbedder(EmbeddingProvider):
    """Always on-topic, so every retrieval reinforces the same record."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def stores():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        config = StorageConfig(
            hot_uri=str(Path(tmpdir) / "hot"),
            cold_uri=str(Path(tmpdir) / "cold.db"),
            embedding_dim=EMBED_DIM,
        )
        hot = HotStorage(config)
        cold = ColdStorage(config.cold_uri)
        yield hot, cold
        cold.close()


def seed(hot: HotStorage, count: int = 1) -> list[str]:
    ids = []
    for index in range(count):
        record = MemoryRecord(
            entity_name=f"fact_{index}",
            fact_content=f"Fact number {index}",
            memory_type=MemoryType.CORE_ANCHOR,
        )
        hot.upsert(record, [1.0, 0.0, 0.0])
        ids.append(record.id)
    return ids


def build(hot, cold, **overrides) -> QueryRouter:
    return QueryRouter(
        hot, cold, FixedEmbedder(),
        DecayConfig(top_k=10, rerank_top_k=10, always_search_cold=False,
                    **overrides),
    )


def index_of(hot: HotStorage, record_id: str) -> int:
    return next(r.consolidation_index
                for r in hot.get_all() if r.id == record_id)


def test_deferral_is_on_by_default():
    """The shipped default is the one the latency figures were taken at."""
    assert DecayConfig().defer_reinforcement_writes is True


@pytest.mark.asyncio
async def test_buffered_write_is_invisible_until_flushed(stores):
    hot, cold = stores
    record_id = seed(hot)[0]
    router = build(hot, cold)

    await router.retrieve("anything", current_turn=4)
    assert index_of(hot, record_id) == 0, "buffered write reached the store early"

    assert router.flush_reinforcements() == 1
    assert index_of(hot, record_id) == 1


@pytest.mark.asyncio
async def test_repeat_retrievals_match_the_immediate_path(stores):
    """Three reads before a flush must leave k = 3, not k = 1 written thrice.

    The buffer is the authority on a record's index while it holds one. Reading
    the index off the store instead would hand out 1 every time, because the
    store still says 0 until the flush lands.
    """
    hot, cold = stores
    record_id = seed(hot)[0]
    router = build(hot, cold)

    for turn in (4, 5, 6):
        await router.retrieve("anything", current_turn=turn)
    router.flush_reinforcements()
    assert index_of(hot, record_id) == 3


@pytest.mark.asyncio
async def test_immediate_path_still_writes_per_query(stores):
    """The ablation has to be a real ablation."""
    hot, cold = stores
    record_id = seed(hot)[0]
    router = build(hot, cold, defer_reinforcement_writes=False)

    await router.retrieve("anything", current_turn=4)
    assert index_of(hot, record_id) == 1
    assert router.flush_reinforcements() == 0


@pytest.mark.asyncio
async def test_buffer_flushes_itself_at_the_limit(stores):
    """A caller that only ever retrieves still gets its reinforcement written.

    Decay, maintain() and close() all flush, but nothing obliges a caller to
    reach any of them. Without the bound the buffer would grow for the life of
    the process and hold writes that never land.
    """
    hot, cold = stores
    ids = seed(hot, count=6)
    router = build(hot, cold, reinforcement_buffer_limit=4)

    await router.retrieve("anything", current_turn=4)

    assert router._pending_reinforcements == {}, "buffer was not flushed at the limit"
    assert all(index_of(hot, record_id) == 1 for record_id in ids)
