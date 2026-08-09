from __future__ import annotations

from typing import Callable

from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryStateUpdate, MemoryType
from nmafc.engine.decay import decay_record
from nmafc.storage.cold import ColdStorage
from nmafc.storage.hot import HotStorage


def rebuild_hot_from_cold(
    cold: ColdStorage,
    hot: HotStorage,
    embed_fn: Callable[[str], list[float]],
    config: DecayConfig,
    up_to_turn: int,
) -> int:
    """Wipe Hot RAM and rebuild it by replaying the Cold ROM event log.

    Replays all active events up to the specified turn, applying decay
    math to compute final weights. Returns the number of records restored.
    """
    hot.clear()

    events = cold.get_active_events()
    restored = 0

    for event in events:
        if event["turn"] > up_to_turn:
            continue

        memory_type = MemoryType(event["memory_type"])
        record = MemoryRecord(
            entity_name=event["entity_name"],
            fact_content=event["fact_content"],
            memory_type=memory_type,
            weight=1.0,
            consolidation_index=0,
            created_at_turn=event["turn"],
            last_reinforced_turn=event["turn"],
        )

        new_weight = decay_record(record, up_to_turn, config)
        if new_weight < config.w_prune:
            continue

        record = record.model_copy(update={"weight": new_weight})
        embedding = embed_fn(event["fact_content"])
        hot.upsert(record, embedding)
        restored += 1

    return restored


def invalidate_event(
    cold: ColdStorage,
    hot: HotStorage,
    event_id: int,
    entity_name: str,
) -> None:
    """Invalidate a specific event and remove its vector from Hot RAM.

    Marks the event inactive in Cold ROM and deletes any matching
    vector from Hot RAM by entity name.
    """
    cold.mark_inactive(event_id)

    records = hot.get_by_entity(entity_name)
    for record in records:
        hot.delete(record.id)
