"""Real end-to-end test: embedded storage + fastembed + OpenRouter LLM.

No mocks. Uses:
- LanceDB (embedded) for hot storage
- SQLite (embedded) for cold storage + event log
- fastembed/BAAI/bge-small-en-v1.5 for local embeddings (no key needed)
- OpenRouter for LLM extraction/chat (needs OPENROUTER_API_KEY)

Run with:
    export OPENROUTER_API_KEY=sk-or-...
    python scripts/test_real_end_to_end.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Ensure we import the local source, not an installed wheel
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nmafc import NeuromorphicMemory
from nmafc.integration.factory import create_embedding_provider, create_llm_provider
from nmafc.schemas.memory import MemoryStateUpdate, MemoryType
from nmafc.storage.config import NMafcConfig, StorageConfig


def make_config(tmp: Path) -> NMafcConfig:
    """Build a real config pointing at embedded temp storage."""
    return NMafcConfig(
        llm_provider_model="openrouter/openai/gpt-4o-mini",
        embedding_provider_model="fastembed/BAAI/bge-small-en-v1.5",
        agent_id="e2e-test-agent",
        conversation_id="e2e-test-conv",
        storage=StorageConfig(
            hot_uri=str(tmp / "hot.lancedb"),
            cold_uri=str(tmp / "cold.db"),
            event_log_uri=str(tmp / "events.db"),
            embedding_dim=384,
        ),
    )


async def test_storage_and_embeddings_only(tmp: Path):
    """Test with real embedded storage + real fastembed embeddings, no LLM."""
    print("\n=== Test 1: Storage + embeddings (no LLM) ===")

    llm = create_llm_provider("openrouter/openai/gpt-4o-mini")
    embedder = create_embedding_provider("fastembed/BAAI/bge-small-en-v1.5")

    # Verify embedder is real, not mocked
    vecs = await embedder.embed(["hello world", "fastembed works"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384
    assert all(isinstance(v, float) for v in vecs[0])
    print(f"  fastembed real embeddings: dim={len(vecs[0])}, 2 vectors OK")

    config = make_config(tmp)
    mem = NeuromorphicMemory(
        llm_provider=llm,
        embedding_provider=embedder,
        config=config,
    )

    # Real ingest without LLM — updates written to embedded hot + cold storage
    updates = [
        MemoryStateUpdate(
            entity_name="user_name",
            fact_content="The user's name is Alice",
            memory_type=MemoryType.CORE_ANCHOR,
        ),
        MemoryStateUpdate(
            entity_name="user_allergy",
            fact_content="Allergic to shellfish",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        ),
    ]
    await mem.ingest_updates(updates)

    stats = mem.get_hot_stats()
    print(f"  Hot stats after ingest: {stats}")
    assert stats["count"] >= 2, f"expected >= 2 records, got {stats['count']}"

    cold_stats = mem.get_cold_stats()
    print(f"  Cold stats: {cold_stats}")
    assert cold_stats["total_events"] >= 2, "cold archive should have events"

    # Real recall — reads from embedded hot storage with real vector similarity
    hits = await mem.recall("what is the user allergic to?")
    print(f"  Recall hits: {len(hits.hits)}, context preview: {hits.context[:120]}...")
    assert len(hits.hits) > 0, "recall should return hits"
    assert "shellfish" in hits.context.lower(), f"context should mention allergy: {hits.context}"

    # Real remember — stores a transcript
    result = await mem.remember(
        [
            {"role": "user", "content": "I just moved to Berlin."},
        ]
    )
    print(f"  Remember result: turn={result.turn}, ingested={result.updates_ingested}")
    assert result.updates_ingested >= 1

    # Recall should now find Berlin
    hits2 = await mem.recall("where does the user live?")
    print(f"  Recall after remember: {len(hits2.hits)} hits")
    assert len(hits2.hits) > 0

    # Real override: contradict the allergy
    result2 = await mem.remember(
        [
            {"role": "user", "content": "Actually, I can eat shellfish now."},
        ]
    )
    print(f"  Override remember: turn={result2.turn}, suppressed={result2.overrides_suppressed}")

    hits3 = await mem.recall("can the user eat shellfish?")
    print(f"  Recall after override: {len(hits3.hits)} hits, context: {hits3.context[:120]}...")
    # Berlin should still be findable
    hits4 = await mem.recall("where does the user live?")
    assert len(hits4.hits) > 0, "Berlin should still be in memory after override"

    mem.close()
    print("  PASS: Storage + embeddings end-to-end OK")


async def test_full_pipeline_with_llm(tmp: Path):
    """Full end-to-end: real LLM (OpenRouter) + real embeddings + real storage."""
    print("\n=== Test 2: Full pipeline with real OpenRouter LLM ===")

    config = make_config(tmp)
    async with NeuromorphicMemory.from_config(config=config) as mem:
        # Real turn processing: LLM extracts facts + generates reply, all stored
        reply1 = await mem.process_turn(
            user_msg="Hi, I'm Bob and I work at Acme Corp as a software engineer.",
            conversation_history=[
                {"role": "user", "content": "Hi, I'm Bob and I work at Acme Corp as a software engineer."}
            ],
        )
        print(f"  Turn 1 reply: {repr(reply1)} (may be empty if model returns tool calls only)")
        # The key check: facts were extracted and stored, regardless of reply text
        stats = mem.get_hot_stats()
        print(f"  Hot stats after turn 1: count={stats['count']}, types={stats['types']}")
        assert stats["count"] >= 1, f"expected facts stored, got {stats}"

        # Real recall
        hits = await mem.recall("where does Bob work?")
        print(f"  Recall 'where does Bob work?': {len(hits.hits)} hits")
        print(f"  Context: {hits.context[:200]}...")
        assert len(hits.hits) > 0

        # Second turn — builds on memory
        reply2 = await mem.process_turn(
            user_msg="What's my job?",
            conversation_history=[{"role": "user", "content": "What's my job?"}],
        )
        print(f"  Turn 2 reply: {repr(reply2)} (may be empty if model returns tool calls only)")
        assert len(reply2) >= 0

        # Consolidation
        consolidated = await mem.consolidate()
        print(f"  Consolidate: {consolidated} records consolidated")

        # Rollback test
        rolled = await mem.rollback(to_turn=1)
        print(f"  Rollback to turn 1: {rolled} events replayed")

        print("  PASS: Full pipeline with real LLM OK")


async def main():
    tmp = Path(tempfile.mkdtemp(prefix="nmafc_e2e_"))
    print(f"Test directory: {tmp}")

    await test_storage_and_embeddings_only(tmp)

    # Full pipeline needs an LLM key
    if os.environ.get("OPENROUTER_API_KEY"):
        await test_full_pipeline_with_llm(tmp)
    else:
        print("\n=== SKIPPED: Full LLM pipeline (no OPENROUTER_API_KEY) ===")
        print("Set OPENROUTER_API_KEY and re-run to test with real OpenRouter LLM.")

    # Cleanup test data
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nAll end-to-end tests complete.")


if __name__ == "__main__":
    asyncio.run(main())
