"""Manual ingestion example — bootstrap memory without an LLM.

Demonstrates:
    - Injecting facts with ingest_updates()
    - Querying memory state
    - Watching decay over turns
    - Override detection and suppression
    - Rollback to a previous turn

Run:
    export OPENAI_API_KEY=sk-...
    python examples/manual_ingestion.py
"""

import asyncio
from nmafc.wrapper import NeuromorphicMemory
from nmafc.schemas.memory import MemoryStateUpdate, MemoryType


async def main():
    async with await NeuromorphicMemory.from_config() as mem:
        # ── Bootstrap facts ──
        print("=== Injecting initial facts ===")
        await mem.ingest_updates([
            MemoryStateUpdate(
                entity_name="user_name",
                fact_content="User is named Charlie",
                memory_type=MemoryType.CORE_ANCHOR,
            ),
            MemoryStateUpdate(
                entity_name="user_location",
                fact_content="Lives in Paris",
                memory_type=MemoryType.ACTIVE_CONTEXT,
                related_entities=["user_name"],
            ),
            MemoryStateUpdate(
                entity_name="user_mood",
                fact_content="Feeling tired today",
                memory_type=MemoryType.EPHEMERAL_STATE,
            ),
        ])
        print(f"Turn {mem.current_turn}: {mem.get_hot_stats()}")

        # ── Override detection ──
        print("\n=== Override: Paris → Berlin ===")
        await mem.ingest_updates([
            MemoryStateUpdate(
                entity_name="user_location",
                fact_content="Moved to Berlin last week",
                memory_type=MemoryType.ACTIVE_CONTEXT,
                overrides_entity="user_location",
                related_entities=["user_name"],
            ),
        ])
        print(f"Turn {mem.current_turn}: {mem.get_hot_stats()}")

        # ── Watch decay over several turns ──
        print("\n=== Decay simulation (5 empty turns) ===")
        for _ in range(5):
            await mem.ingest_updates([])  # triggers decay without new facts
        stats = mem.get_hot_stats()
        print(f"Turn {mem.current_turn}: avg_weight={stats['avg_weight']:.4f}")

        # ── Inspect records ──
        print("\n=== Hot RAM Records ===")
        for r in mem._hot.get_all():
            print(f"  [{r.memory_type.value:16s}] {r.entity_name:20s} "
                  f"weight={r.weight:.4f} fact={r.fact_content}")

        # ── Events ──
        print(f"\n=== Events ({mem.get_event_stats()['total_events']} total) ===")
        for ev in mem.get_events(limit=10):
            print(f"  Turn {ev.turn}: {ev.event_type.value} → {ev.entity_name}")

        # ── Rollback ──
        print("\n=== Rollback to turn 2 ===")
        restored = await mem.rollback(to_turn=2)
        print(f"Restored {restored} records. Current turn: {mem.current_turn}")


if __name__ == "__main__":
    asyncio.run(main())
