import tempfile
from pathlib import Path

import pytest

from nmafc.integration.query_router import QueryRouter
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType
from nmafc.storage.cold import ColdStorage
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage


class MockEmbedder:
    async def embed_single(self, text: str) -> list[float]:
        # Return deterministic mock embeddings
        if "spouse" in text.lower():
            return [1.0, 0.0, 0.0]
        elif "brother" in text.lower():
            return [0.0, 1.0, 0.0]
        else:
            return [0.5, 0.5, 0.0]


@pytest.mark.asyncio
async def test_spreading_activation_2_hops():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage_config = StorageConfig(
            hot_uri=str(Path(tmpdir) / "hot"),
            cold_uri=str(Path(tmpdir) / "cold.db"),
            embedding_dim=3,
        )
        hot = HotStorage(storage_config)
        cold = ColdStorage(storage_config.cold_uri)
        embedder = MockEmbedder()
        decay_config = DecayConfig(max_hops=2, top_k=1, theta=0.0)

        router = QueryRouter(hot, cold, embedder, decay_config)

        # Store 2-hop relational chain: Me -> Spouse (James) -> Brother (David)
        rec1 = MemoryRecord(
            entity_name="spouse_james",
            fact_content="Married to James",
            memory_type=MemoryType.CORE_ANCHOR,
            related_entities=["brother_david"],
        )
        rec2 = MemoryRecord(
            entity_name="brother_david",
            fact_content="James's brother David is an Airline Pilot",
            memory_type=MemoryType.CORE_ANCHOR,
            related_entities=["job_pilot"],
        )
        rec3 = MemoryRecord(
            entity_name="job_pilot",
            fact_content="Works as Chief Airline Pilot at Delta",
            memory_type=MemoryType.CORE_ANCHOR,
            related_entities=[],
        )

        hot.upsert(rec1, [1.0, 0.0, 0.0])   # Vector hit for "spouse" query
        hot.upsert(rec2, [0.0, 1.0, 0.0])   # Hop 1 hit via related_entities
        hot.upsert(rec3, [0.0, 0.0, 1.0])   # Hop 2 hit via related_entities

        # Query "spouse" -> Vector search returns Hop 0 (rec1), Spreading
        # Activation traverses to rec2 (Hop 1) and rec3 (Hop 2)
        results = await router.retrieve("What is my spouse's info?", current_turn=1)

        retrieved_entities = {r.entity_name for r in results}
        assert "spouse_james" in retrieved_entities
        assert "brother_david" in retrieved_entities
        assert "job_pilot" in retrieved_entities

        cold.close()


@pytest.mark.asyncio
async def test_spreading_activation_reverse_from_seed():
    """A link pointing at a vector-hit seed is a graph neighbour of that seed.

    Regression: traversal used to seed the frontier only from outbound links,
    so a record whose related_entities named a direct hit was invisible from
    the hit's side -- the entity graph drew the edge but BFS never crossed it.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage_config = StorageConfig(
            hot_uri=str(Path(tmpdir) / "hot"),
            cold_uri=str(Path(tmpdir) / "cold.db"),
            embedding_dim=3,
        )
        hot = HotStorage(storage_config)
        cold = ColdStorage(storage_config.cold_uri)
        embedder = MockEmbedder()
        decay_config = DecayConfig(max_hops=2, top_k=1, theta=0.0)

        router = QueryRouter(hot, cold, embedder, decay_config)

        rec_seed = MemoryRecord(
            entity_name="spouse_james",
            fact_content="Married to James",
            memory_type=MemoryType.CORE_ANCHOR,
            related_entities=[],
        )
        rec_pointing = MemoryRecord(
            entity_name="pilot_project",
            fact_content="James and David built a hobby plane",
            memory_type=MemoryType.CORE_ANCHOR,
            related_entities=["spouse_james"],
        )
        hot.upsert(rec_seed, [1.0, 0.0, 0.0])     # vector hit for "spouse"
        hot.upsert(rec_pointing, [0.0, 1.0, 0.0])  # not a vector hit

        # Only spouse_james is a top-k=1 hit; pilot_project must arrive via
        # the reverse lookup on the seed entity itself.
        results = await router.retrieve("What is my spouse's info?", current_turn=1)

        retrieved_entities = {r.entity_name for r in results}
        assert "spouse_james" in retrieved_entities
        assert "pilot_project" in retrieved_entities

        cold.close()
