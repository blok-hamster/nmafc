"""Synchronous usage example — no async required.

Demonstrates the SyncNeuromorphicMemory wrapper for scripts,
notebooks, and CLI tools that don't use async/await.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/sync_usage.py
"""

from nmafc.wrapper import SyncNeuromorphicMemory
from nmafc.schemas.memory import MemoryStateUpdate


def main():
    with SyncNeuromorphicMemory.from_config() as mem:
        # Process turns — no async needed
        response = mem.process_turn_sync(
            "I just started a new job at Acme Corp."
        )
        print(f"Turn 1: {response}\n")

        response = mem.process_turn_sync(
            "Where do I work?"
        )
        print(f"Turn 2: {response}\n")

        # Manual ingestion
        mem.ingest_updates_sync([
            MemoryStateUpdate(
                entity_name="company",
                fact_content="Works at Acme Corp as a senior engineer",
                memory_type="CoreAnchor",
            )
        ])

        # Introspection
        stats = mem.get_hot_stats()
        print(f"Records: {stats['count']}")
        print(f"Types: {stats['types']}")
        print(f"Avg weight: {stats['avg_weight']:.4f}")


if __name__ == "__main__":
    main()
