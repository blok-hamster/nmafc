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
