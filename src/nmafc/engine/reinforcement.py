from __future__ import annotations

from nmafc.schemas.memory import MemoryRecord


def reinforce(record: MemoryRecord, current_turn: int) -> MemoryRecord:
    """Apply Long-Term Potentiation (LTP) to a retrieved memory.

    Resets weight to 1.0, increments consolidation index,
    and updates the last reinforced turn.
    """
    return record.model_copy(
        update={
            "weight": 1.0,
            "consolidation_index": record.consolidation_index + 1,
            "last_reinforced_turn": current_turn,
        }
    )


def batch_reinforce(
    records: list[MemoryRecord], current_turn: int
) -> list[MemoryRecord]:
    """Apply LTP to a batch of retrieved memories."""
    return [reinforce(r, current_turn) for r in records]


def create_ltp_events(
    records: list[MemoryRecord],
    reinforced: list[MemoryRecord],
    current_turn: int,
) -> list:
    """Create LTP events for reinforced records.

    Call this after batch_reinforce to log the LTP events for the Web UI.
    Returns list of MemoryEvent objects (not yet persisted).
    """
    from nmafc.schemas.events import EventType, MemoryEvent

    events = []
    for old, new in zip(records, reinforced):
        if new.consolidation_index > old.consolidation_index:
            events.append(
                MemoryEvent(
                    event_type=EventType.LTP,
                    turn=current_turn,
                    record_id=old.id,
                    entity_name=old.entity_name,
                    old_weight=old.weight,
                    new_weight=new.weight,
                    old_k=old.consolidation_index,
                    new_k=new.consolidation_index,
                )
            )
    return events
