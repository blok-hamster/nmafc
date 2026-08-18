from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of cognitive events emitted by the memory engine."""

    WEIGHT_UPDATE = "weight_update"
    OVERRIDE = "override"
    SUPPRESSION = "suppression"
    PRUNE = "prune"
    CONSOLIDATION = "consolidation"
    LTP = "ltp"
    RETRIEVAL = "retrieval"


class MemoryEvent(BaseModel):
    """A structured cognitive event emitted by the memory engine.

    Captures every significant state change for audit, visualization,
    and real-time monitoring via the Web UI.
    """

    event_type: EventType
    agent_id: str = "default"
    conversation_id: str = "default"
    turn: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    record_id: str
    entity_name: str

    # Weight tracking
    old_weight: float | None = None
    new_weight: float | None = None

    # Type change tracking (consolidation)
    old_memory_type: str | None = None
    new_memory_type: str | None = None

    # Override tracking
    suppressed_by: str | None = None

    # LTP tracking
    old_k: int | None = None
    new_k: int | None = None

    # Retrieval tracking
    retrieval_score: float | None = None
    hops: int | None = None

    # Catch-all for future extensions
    metadata: dict[str, Any] = Field(default_factory=dict)
