"""Synchronous usage example — no async required.

Demonstrates the SyncNeuromorphicMemory wrapper for scripts,
notebooks, and CLI tools that don't use async/await.

Uses fastembed for key-free local embeddings and points storage at a temp
directory, mirroring the end-to-end test structure.

Prerequisites:
    pip install "nmafc[llm,fastembed]"
    export OPENAI_API_KEY=sk-...   # only needed for the LLM route

Run:
    python examples/sync_usage.py
"""

import tempfile
from pathlib import Path

from nmafc import SyncNeuromorphicMemory
from nmafc.storage.config import NMafcConfig, StorageConfig


def main():
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

    with SyncNeuromorphicMemory.from_config(config=config) as mem:
        # Process turns — no async needed
        response = mem.process_turn_sync(
            "I just started a new job at Acme Corp."
        )
        print(f"Turn 1: {response}\n")

        response = mem.process_turn_sync(
            "Where do I work?"
        )
        print(f"Turn 2: {response}\n")

        # Introspection
        stats = mem.get_hot_stats()
        print(f"Records: {stats['count']}")
        print(f"Types: {stats['types']}")
        print(f"Avg weight: {stats['avg_weight']:.4f}")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
