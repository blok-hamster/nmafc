from __future__ import annotations

from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryStateUpdate
from nmafc.storage.cold import ColdStorage
from nmafc.storage.hot import HotStorage


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


def identify_prunable(records: list[MemoryRecord], w_prune: float) -> list[str]:
    """Return IDs of records whose weight has fallen below the prune threshold."""
    return [r.id for r in records if r.weight <= w_prune]


def prune_cycle(
    hot: HotStorage,
    cold: ColdStorage,
    w_prune: float,
    current_turn: int,
) -> int:
    """Evict all below-threshold records from Hot RAM.

    Records are physically deleted from LanceDB but remain
    in the Cold ROM as inactive events (preserving the audit trail).
    Returns the count of pruned records.
    """
    all_records = hot.get_all()
    prunable_ids = identify_prunable(all_records, w_prune)

    for record_id in prunable_ids:
        hot.delete(record_id)

    return len(prunable_ids)
