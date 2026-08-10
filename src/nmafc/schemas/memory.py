from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    CORE_ANCHOR = "CoreAnchor"
    ACTIVE_CONTEXT = "ActiveContext"
    EPHEMERAL_STATE = "EphemeralState"


class MemoryStateUpdate(BaseModel):
    """Structured state change extracted from a conversation turn via LLM tool calling."""

    entity_name: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the entity being updated (e.g. 'user_allergy', 'blood_pressure_medication')",
    )
    fact_content: str = Field(
        ...,
        min_length=1,
        description="The factual content extracted from the conversation.",
    )
    memory_type: MemoryType = Field(
        ...,
        description="Classification tier determining decay behavior.",
    )
    overrides_entity: Optional[str] = Field(
        default=None,
        description="Entity name of an existing memory this update contradicts/replaces.",
    )
    related_entities: list[str] = Field(
        default_factory=list,
        description="List of related entity names linked to this fact for graph spreading activation.",
    )


class UnifiedMemoryPayload(BaseModel):
    """Container for all state updates extracted from a single conversation turn."""

    updates: list[MemoryStateUpdate] = Field(
        default_factory=list,
        description="List of structured state changes extracted from the user's turn.",
    )


class MemoryRecord(BaseModel):
    """Internal representation of a memory vector stored in Hot RAM (LanceDB)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_name: str
    fact_content: str
    memory_type: MemoryType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    consolidation_index: int = Field(default=0, ge=0)
    created_at_turn: int = Field(default=0, ge=0)
    last_reinforced_turn: int = Field(default=0, ge=0)
    is_active: bool = Field(default=True)
    related_entities: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """A memory record returned from vector search with its similarity score."""

    record: MemoryRecord
    score: float = Field(ge=0.0, le=1.0)
    hops: int = Field(default=0, ge=0, description="Graph traversal hop distance (0 = direct vector hit)")



class DecayConfig(BaseModel):
    """All tunable hyperparameters for the cognitive decay engine."""

    lambda_core_anchor: float = Field(default=0.0, ge=0.0)
    lambda_active_context: float = Field(default=0.05, ge=0.0)
    lambda_ephemeral: float = Field(default=0.69, ge=0.0)
    eta: float = Field(default=0.15, gt=0.0, description="Consolidation constant")
    gamma: float = Field(default=0.1, ge=0.0, le=1.0, description="Suppression multiplier")
    w_prune: float = Field(default=0.1, ge=0.0, le=1.0, description="Eviction threshold")
    theta: float = Field(default=0.75, ge=0.0, le=1.0, description="Retrieval similarity threshold")
    top_k: int = Field(default=10, gt=0)
    fallback_keyword_limit: int = Field(default=20, gt=0)
    max_hops: int = Field(default=2, ge=0, description="Max graph traversal depth for Spreading Activation")
    auto_consolidate_turns: int = Field(default=5, ge=0, description="Interval in turns for automatic REM consolidation")

    def get_lambda_base(self, memory_type: MemoryType) -> float:
        match memory_type:
            case MemoryType.CORE_ANCHOR:
                return self.lambda_core_anchor
            case MemoryType.ACTIVE_CONTEXT:
                return self.lambda_active_context
            case MemoryType.EPHEMERAL_STATE:
                return self.lambda_ephemeral
