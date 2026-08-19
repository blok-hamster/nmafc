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
    """Apply synaptic suppression multiplier to a contradicted record."""
    new_weight = record.weight * gamma
    return record.model_copy(update={"weight": new_weight})


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
    """Evict all below-threshold records from Hot RAM.

    Records are physically deleted from LanceDB but remain
    in the Cold ROM as inactive events (preserving the audit trail).
    Returns the count of pruned records.

    When `event_logger` is provided, PRUNE events are emitted for each
    evicted record before deletion.
    """
    all_records = hot.get_all()
    prunable_ids = identify_prunable(all_records, w_prune)

    if event_logger is not None and prunable_ids:
        from nmafc.schemas.events import EventType, MemoryEvent

        prunable_set = set(prunable_ids)
        for rec in all_records:
            if rec.id in prunable_set:
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

    hot.delete_many(prunable_ids)

    return len(prunable_ids)
