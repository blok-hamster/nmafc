from __future__ import annotations

from typing import TYPE_CHECKING

from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryStateUpdate
from nmafc.storage.cold_base import ColdStorageBase
from nmafc.storage.hot import HotStorage

if TYPE_CHECKING:
    from nmafc.storage.event_log import EventLog


def detect_override(
    new_update: MemoryStateUpdate,
    existing: list[MemoryRecord],
) -> list[MemoryRecord]:
    """Find existing records that the new update contradicts.

    Detection rules:
    1. Explicit: new_update.overrides_entity matches record.entity_name
    2. Implicit: same entity_name (the new fact replaces the old one)
    """
    targets = []
    for record in existing:
        if new_update.overrides_entity and record.entity_name == new_update.overrides_entity:
            targets.append(record)
        elif record.entity_name == new_update.entity_name:
            targets.append(record)
    return targets


def apply_suppression(record: MemoryRecord, gamma: float) -> MemoryRecord:
    """Apply synaptic suppression multiplier to a contradicted record.

    Legacy path — kept for backward compatibility with tests and benchmarks
    that explicitly set exclude_invalidated=False.
    """
    new_weight = record.weight * gamma
    return record.model_copy(update={"weight": new_weight})


def invalidate_record(record: MemoryRecord, current_turn: int) -> MemoryRecord:
    """Mark a contradicted record as temporally invalidated.

    The record's weight is frozen (not multiplied by gamma) and invalid_at is
    set. The record remains in Hot RAM but is excluded from default searches.
    """
    return record.model_copy(update={"invalid_at": current_turn})


def create_suppression_event(
    old_record: MemoryRecord,
    new_weight: float,
    suppressed_by: str,
    turn: int,
) -> MemoryEvent:
    """Create a SUPPRESSION event for a contradicted record.

    Call this after apply_suppression to log the override for the Web UI.
    """
    from nmafc.schemas.events import EventType, MemoryEvent

    return MemoryEvent(
        event_type=EventType.SUPPRESSION,
        turn=turn,
        record_id=old_record.id,
        entity_name=old_record.entity_name,
        old_weight=old_record.weight,
        new_weight=new_weight,
        suppressed_by=suppressed_by,
        old_memory_type=old_record.memory_type.value,
    )


def identify_prunable(records: list[MemoryRecord], w_prune: float) -> list[str]:
    """Return IDs of records whose weight has fallen below the prune threshold."""
    return [r.id for r in records if r.weight <= w_prune]


def prune_cycle(
    hot: HotStorage,
    cold: ColdStorageBase,
    w_prune: float,
    current_turn: int,
    event_logger: EventLog | None = None,
) -> int:
    """Evict or invalidate below-threshold records from Hot RAM.

    Behavior depends on memory type:
    - EphemeralState below w_prune: physically deleted (genuine expiry)
    - ActiveContext below w_prune: temporally invalidated (invalid_at set,
      excluded from default search but retained for temporal queries)
    - Already-invalidated records below 0.01: physically deleted to prevent
      indefinite accumulation

    Returns the count of records removed or invalidated.

    When `event_logger` is provided, PRUNE events are emitted for each
    affected record.
    """
    from nmafc.schemas.memory import MemoryType

    all_records = hot.get_all()

    delete_ids: list[str] = []
    invalidate_updates: list[tuple[str, int]] = []

    for rec in all_records:
        if rec.weight > w_prune:
            if rec.invalid_at is not None and rec.weight < 0.01:
                delete_ids.append(rec.id)
            continue

        if rec.invalid_at is not None:
            delete_ids.append(rec.id)
        elif rec.memory_type == MemoryType.EPHEMERAL_STATE:
            delete_ids.append(rec.id)
        else:
            invalidate_updates.append((rec.id, current_turn))

    if event_logger is not None and (delete_ids or invalidate_updates):
        from nmafc.schemas.events import EventType, MemoryEvent

        affected = set(delete_ids) | {uid for uid, _ in invalidate_updates}
        for rec in all_records:
            if rec.id in affected:
                event_logger.log(
                    MemoryEvent(
                        event_type=EventType.PRUNE,
                        turn=current_turn,
                        record_id=rec.id,
                        entity_name=rec.entity_name,
                        old_weight=rec.weight,
                        old_memory_type=rec.memory_type.value,
                    )
                )

    hot.delete_many(delete_ids)
    hot.set_invalid_at_many(invalidate_updates)

    return len(delete_ids) + len(invalidate_updates)
