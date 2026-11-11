"""Minimal NMAFC example — 10 lines to get started.

Shows the pattern we validate end-to-end: explicit storage config, local
embeddings with fastembed (no key needed), and the `async with` context
manager on the constructor's return value.

Prerequisites:
    pip install "nmafc[llm,fastembed]"
    export OPENAI_API_KEY=sk-...   # only needed for the LLM route

Run:
    python examples/minimal.py
"""

import asyncio
import tempfile
from pathlib import Path

from nmafc import NeuromorphicMemory
from nmafc.storage.config import NMafcConfig, StorageConfig


async def main():
    # Point storage at a temp dir and match the vector width to the embedding
    # model (fastembed BGE-small = 384 dims). This mirrors the end-to-end
    # test in scripts/test_real_end_to_end.py.
    tmp = Path(tempfile.mkdtemp(prefix="nmafc_example_"))
    config = NMafcConfig(
        llm_provider_model="openai/gpt-4o-mini",
        embedding_provider_model="fastembed/BAAI/bge-small-en-v1.5",
        agent_id="example-agent",
        conversation_id="example-conv",
        storage=StorageConfig(
            hot_uri=str(tmp / "hot.lancedb"),
            cold_uri=str(tmp / "cold.db"),
            event_log_uri=str(tmp / "events.db"),
            embedding_dim=384,
        ),
    )

    # `from_config` is synchronous — no `await`. Use `async with` on the
    # returned instance.
    async with NeuromorphicMemory.from_config(config=config) as mem:
        # Turn 1: Store a fact
        response = await mem.process_turn(
            "My name is Alice and I'm a software engineer."
        )
        print(f"Turn 1: {response}\n")

        # Turn 2: Memory remembers
        response = await mem.process_turn(
            "What's my name and job?"
        )
        print(f"Turn 2: {response}\n")

        # Check memory state
        stats = mem.get_hot_stats()
        print(f"Records: {stats['count']}, Types: {stats['types']}")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
