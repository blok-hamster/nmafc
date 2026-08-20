"""Cold ROM fallback in QueryRouter.

The fallback fires when Hot RAM's best cosine similarity falls below theta. It
previously fetched the keyword results and then discarded them unless the vector
search returned literally nothing, which on a populated store never happens — so
the archive was searched on every weak query and consulted on none of them.
"""

import tempfile
from pathlib import Path

import pytest

from nmafc.integration.base import EmbeddingProvider
from nmafc.integration.query_router import QueryRouter
from nmafc.schemas.memory import (
    DecayConfig,
    MemoryRecord,
    MemoryStateUpdate,
    MemoryType,
)
from nmafc.storage.cold import ColdStorage
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage

EMBED_DIM = 3


class FixedEmbedder(EmbeddingProvider):
    """Returns one caller-chosen vector, so tests set the similarity directly."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vector for _ in texts]


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


def seed(hot: HotStorage, cold: ColdStorage) -> None:
    """One fact in Hot RAM, a different one reachable only through Cold ROM."""
    hot.upsert(
        MemoryRecord(
            entity_name="hot_fact",
            fact_content="Lives in Berlin",
            memory_type=MemoryType.CORE_ANCHOR,
        ),
        [1.0, 0.0, 0.0],
    )
    cold.append_event(
        MemoryStateUpdate(
            entity_name="archived_fact",
            fact_content="Studied marine biology at university",
            memory_type=MemoryType.CORE_ANCHOR,
        ),
        turn=1,
    )


@pytest.mark.asyncio
async def test_weak_hit_reaches_cold_rom(stores):
    """A non-empty but off-topic vector result must still trigger the archive."""
    hot, cold = stores
    seed(hot, cold)

    # Orthogonal to the stored vector: score 0.0, below theta, but not empty.
    router = QueryRouter(
        hot, cold, FixedEmbedder([0.0, 1.0, 0.0]), DecayConfig(theta=0.75, top_k=5)
    )
    results = await router.retrieve("marine biology", current_turn=2)

    entities = {r.entity_name for r in results}
    assert "archived_fact" in entities


@pytest.mark.asyncio
async def test_strong_hit_skips_cold_rom_legacy_mode(stores):
    """With always_search_cold=False, a strong Hot hit skips the archive (legacy theta gate)."""
    hot, cold = stores
    seed(hot, cold)

    router = QueryRouter(
        hot, cold, FixedEmbedder([1.0, 0.0, 0.0]),
        DecayConfig(theta=0.75, top_k=5, always_search_cold=False),
    )
    results = await router.retrieve("marine biology", current_turn=2)

    entities = {r.entity_name for r in results}
    assert entities == {"hot_fact"}


@pytest.mark.asyncio
async def test_always_search_cold_includes_archive(stores):
    """With always_search_cold=True (default), archive is always consulted."""
    hot, cold = stores
    seed(hot, cold)

    router = QueryRouter(
        hot, cold, FixedEmbedder([1.0, 0.0, 0.0]),
        DecayConfig(theta=0.75, top_k=5, always_search_cold=True),
    )
    results = await router.retrieve("marine biology", current_turn=2)

    entities = {r.entity_name for r in results}
    assert "hot_fact" in entities
    assert "archived_fact" in entities


@pytest.mark.asyncio
async def test_cold_results_do_not_duplicate_hot_hits(stores):
    """The same entity in both tiers must appear once, not twice."""
    hot, cold = stores
    seed(hot, cold)
    cold.append_event(
        MemoryStateUpdate(
            entity_name="hot_fact",
            fact_content="Lives in Berlin",
            memory_type=MemoryType.CORE_ANCHOR,
        ),
        turn=1,
    )

    router = QueryRouter(
        hot, cold, FixedEmbedder([0.0, 1.0, 0.0]), DecayConfig(theta=0.75, top_k=5)
    )
    results = await router.retrieve("Berlin", current_turn=2)

    names = [r.entity_name for r in results]
    assert names.count("hot_fact") == 1


@pytest.mark.asyncio
async def test_cold_results_are_not_reinforced(stores):
    """Reading the archive must not resurrect a memory into the working set.

    Cold ROM records carry no Hot RAM id, so reinforcing them would either fail
    silently or write a phantom row; either way, retrieval from cold storage is
    not evidence that a fact belongs in Hot RAM.
    """
    hot, cold = stores
    seed(hot, cold)
    before = hot.count()

    router = QueryRouter(
        hot, cold, FixedEmbedder([0.0, 1.0, 0.0]), DecayConfig(theta=0.75, top_k=5)
    )
    await router.retrieve("marine biology", current_turn=2)

    assert hot.count() == before
    assert hot.get_by_entity("archived_fact") == []
