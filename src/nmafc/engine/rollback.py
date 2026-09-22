from __future__ import annotations

import json
from typing import Callable

from nmafc.engine.decay import decay_record
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType
from nmafc.storage.cold_base import ColdStorageBase
from nmafc.storage.hot import HotStorage


def rebuild_hot_from_cold(
    cold: ColdStorageBase,
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
            valid_at=event.get("valid_at"),
            related_entities=_links_from_event(event),
        )

        new_weight = decay_record(record, up_to_turn, config)
        if new_weight < config.w_prune:
            continue

        # decay_record reads record.weight as w0 and computes the decay over the
        # elapsed turns since last_reinforced_turn. The weight has absorbed all
        # of that decay, so the persisted record must advance last_reinforced_turn
        # to up_to_turn or the next decay pass charges the same span again and
        # every rebuilt record is decayed twice -- see the contract documented on
        # decay_record and HotStorage.apply_weight_updates.
        record = record.model_copy(
            update={"weight": new_weight, "last_reinforced_turn": up_to_turn}
        )
        embedding = embed_fn(event["fact_content"])
        hot.upsert(record, embedding)
        restored += 1

    return restored


def _links_from_event(event: dict) -> list[str]:
    """Reconstruct related_entities from a Cold ROM event row.

    The SQLite backend serialises the list with json.dumps; the Postgres backend
    lacks the column entirely and yields nothing, which is the correct shape for
    a backend that never captured links.
    """
    raw = event.get("related_entities")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(link) for link in raw]
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


def invalidate_event(
    cold: ColdStorageBase,
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
