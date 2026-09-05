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
    valid_at: Optional[str] = Field(
        default=None,
        description="When the fact became true — a date string or relative reference resolved downstream.",
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
    valid_at: Optional[int] = Field(
        default=None,
        description="Turn when the fact became true. Falls back to created_at_turn if None.",
    )
    # The date the extractor read off the conversation, kept as it wrote it.
    #
    # MemoryStateUpdate.valid_at is a string, and the extraction prompt tells the
    # model to fill it with "the ISO date string or the relative expression;
    # downstream code resolves it". Nothing resolved it: the wrapper wrote the
    # current turn number into the integer valid_at and dropped the string, so
    # the only dates that ever survived were the ones the extractor happened to
    # repeat inside fact_content. That is the same failure as the unresolved
    # link names -- a prompt promising a downstream step that was never built.
    #
    # Stored as written rather than parsed into a date. "last summer" and
    # "around March 2023" are the shapes that actually turn up, and a parser
    # would have to either invent precision or discard them; the model reading
    # the context handles both fine.
    valid_at_text: Optional[str] = Field(
        default=None,
        description="Calendar date the fact refers to, in the extractor's own wording.",
    )
    invalid_at: Optional[int] = Field(
        default=None,
        description="Turn when the fact was superseded. None means still valid.",
    )


class SearchResult(BaseModel):
    """A memory record returned from vector search with its similarity score."""

    record: MemoryRecord
    score: float = Field(ge=0.0, le=1.0)
    hops: int = Field(default=0, ge=0, description="Graph traversal hop distance (0 = direct vector hit)")


class SearchCandidate(BaseModel):
    """Internal candidate for reranking — carries source provenance and rank."""

    record: MemoryRecord
    score: Optional[float] = None
    source: str = Field(description="Origin: hot_vector, cold_semantic, cold_keyword, bfs_hot, bfs_cold")
    rank_in_source: int = Field(default=0, ge=0)
    hop_distance: int = Field(default=0, ge=0)



class DecayConfig(BaseModel):
    """All tunable hyperparameters for the cognitive decay engine."""

    lambda_core_anchor: float = Field(default=0.0, ge=0.0)
    lambda_active_context: float = Field(default=0.05, ge=0.0)
    lambda_ephemeral: float = Field(default=0.69, ge=0.0)
    eta: float = Field(default=0.15, gt=0.0, description="Consolidation constant")
    gamma: float = Field(default=0.1, ge=0.0, le=1.0, description="Suppression multiplier")
    w_prune: float = Field(default=0.1, ge=0.0, le=1.0, description="Eviction threshold")
    # Resolve every extracted link onto an entity that actually exists, by name
    # overlap, dropping the ones that match nothing.
    #
    # Of the 11,902 links written across the ten-store LoCoMo run, 38.8% named
    # an entity that was never stored -- near misses like
    # "melanie_lake_sunrise_painting_2022" for the stored
    # "melanie_lake_sunrise_painting". 78% of them match a real entity at an
    # overlap of 0.5 or better. This is the dominant cause of the sparse graph
    # that Spreading Activation has to traverse, and the extraction prompt
    # already promises the behaviour ("Names matching no stored entity are
    # discarded") that nothing in the pipeline implemented.
    resolve_link_targets: bool = Field(
        default=True,
        description="Match extracted link names to stored entities; drop unmatchable links",
    )
    link_match_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum token overlap for a written link name to count as naming an entity",
    )
    # Reroute a pruned record's inbound links onto whatever it pointed at,
    # instead of leaving them to be swept up as dead pointers.
    #
    # Off by default, on measurement rather than on principle. The mechanism is
    # sound and the unit tests show it doing exactly what it claims, but pruned
    # entities are only 5.0% of all link targets, and replaying it over the
    # finished stores moved usable links per fact from 0.92 to 0.93. Forgetting
    # is not what disconnected this graph; see resolve_link_targets, which
    # addresses the 38.8% that is. Kept because it costs nothing when there is
    # nothing to reroute, and because a workload that prunes more heavily than
    # LoCoMo would see more from it.
    rewire_pruned_links: bool = Field(
        default=False,
        description="Reroute links around pruned records instead of severing them",
    )
    rewire_max_links: int = Field(
        default=8,
        ge=0,
        description="Cap on a record's links after rewiring; direct links are never displaced",
    )
    rewire_max_depth: int = Field(
        default=3,
        ge=1,
        description="How many consecutive pruned records a rewiring walk may pass through",
    )
    # Cosine similarity below which Hot RAM is judged to hold nothing on-topic
    # and the Cold ROM keyword fallback fires. Derived from the separation
    # between answerable and unanswerable queries, measured on two populated
    # LoCoMo stores (410 and 330 records, text-embedding-3-small) by scoring each
    # store against its own questions and against another conversation's:
    #
    #             on-topic top-1        off-topic top-1
    #   conv-26   0.493 - 0.867         0.201 - 0.472
    #   conv-30   0.403 - 0.873         0.177 - 0.481
    #
    # The two populations barely overlap, and 0.45 sits in the gap: it wrongly
    # falls back on 0-2% of answerable queries while catching 97-98% of
    # unanswerable ones. It is deliberately biased toward the on-topic side,
    # because a spurious fallback injects fallback_keyword_limit BM25 rows into
    # a context Hot RAM had already answered correctly.
    #
    # The previous default of 0.75 predates the search metric being fixed to
    # cosine; under the old L2 arithmetic no hit ever scored above 0, so the
    # value was unreachable and untested. Against real scores it fires on 58-65%
    # of questions, which makes the fallback the default path rather than a
    # fallback. Chosen from score distributions only -- never from answer keys.
    theta: float = Field(default=0.45, ge=0.0, le=1.0, description="Retrieval similarity threshold")
    top_k: int = Field(default=10, gt=0)
    fallback_keyword_limit: int = Field(default=20, gt=0)
    max_hops: int = Field(default=2, ge=0, description="Max graph traversal depth for Spreading Activation")
    # Whether the Cold ROM fallback searches by meaning (dense vectors over the
    # archive, plus one hop of link expansion inside it) or by shared words
    # alone. False reproduces keyword-only fallback exactly, so this is the
    # ablation control for archive retrieval -- the same role beta = 0 plays for
    # clustering decay. It exists because dense archive fallback shipped
    # alongside two extractor changes and could not afterwards be told apart
    # from them; anything that cannot be switched off cannot be attributed.
    cold_semantic_fallback: bool = Field(
        default=True,
        description="Search Cold ROM by meaning and links, not keywords alone",
    )
    # Protection a fact earns from sitting in a densely interlinked
    # neighbourhood: lambda is scaled by (1 - beta * C), where C is the local
    # clustering coefficient of the fact's entity. beta = 0 disables the
    # mechanism and reproduces type-and-consolidation decay exactly, so it
    # doubles as the ablation control.
    #
    # Clustering, not degree. Lin et al. (2026, Science 393, eaee7004) found
    # that hippocampal memory survived the elimination of most synapses, and
    # that what survived was *clustered* connectivity -- synapses grouped within
    # 5um on a dendrite, sitting on shared multi-synaptic boutons. Their
    # anaesthesia control is the reason degree is the wrong measure: synapse
    # density recovered to naive levels while the memory stayed impaired, so a
    # raw count of connections predicted nothing. Counting related_entities
    # would reproduce exactly the measure their control rules out; the
    # clustering coefficient asks the question they actually answered, which is
    # whether a node's neighbours are connected to each other.
    #
    # The mapping from spatial clustering on a dendrite to graph clustering over
    # entity names is an analogy. No claim of derivation is made.
    beta: float = Field(
        default=0.0, ge=0.0, lt=1.0,
        description="Clustering protection strength for decay (0 = disabled)",
    )
    auto_consolidate_turns: int = Field(default=5, ge=0, description="Interval in turns for automatic REM consolidation")

    # --- Unified search & reranking (Phase A/C/D) ---
    always_search_cold: bool = Field(
        default=True,
        description="Search Cold ROM in parallel with Hot, ignoring the theta gate.",
    )
    # Buffer LTP writebacks in memory and commit them when the caller asks,
    # instead of on every retrieval.
    #
    # Reinforcement rewrites each surviving record through a delete + add, which
    # on an append-only store means a new table version per query. Measured at
    # ~650 ms per retrieval against a store of 494 records -- an order of
    # magnitude more than the search (32 ms), the graph traversal (48 ms) and
    # the archive scan (0.7 ms) put together, and the largest cost this system
    # controls.
    #
    # Off by default because it is not free of consequence: a buffered
    # reinforcement is invisible to any retrieval before the flush, so with
    # `weight_signal` above zero a deferred run can rank differently from an
    # immediate one. At `weight_signal = 0` (the default) weight does not enter
    # ranking and the two are equivalent, which is the regime the benchmark runs
    # in. Callers that turn this on must flush before any decay pass, since
    # decay reads the weight and the consolidation index this would be holding.
    defer_reinforcement_writes: bool = Field(
        default=False,
        description="Buffer LTP writebacks until flush_reinforcements() is called",
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF constant k (higher = less weight to top ranks)")
    rerank_top_k: int = Field(default=20, gt=0, description="Max records surviving reranking into prompt")
    # Measured over 250 prompts at rerank_top_k=40: the validity suffix was 21%
    # of the context, and 10,000 of 10,000 suffixes ended "- present" because
    # nothing in the store had been invalidated. Rendering the span only when a
    # fact has an end date returns that budget to the facts themselves.
    compact_validity: bool = Field(
        default=False,
        description="Render validity as '(date)' rather than '(Valid: date - present)'",
    )
    # Extraction is lossy, and on 1,540 LoCoMo questions the loss is
    # measurable: with the top five facts' source turns restored, strict answer
    # reachability rose from 52.7% to 59.4% overall and from 50.5% to 59.8% on
    # temporal questions, where the gold is phrased relatively ("two weekends
    # before 17 July 2023") and the fact carries a date the extractor resolved
    # on its own. The answers followed: 64.4% -> 66.1% over 1,538 questions,
    # p = 0.021, at a cost of about 384 tokens.
    #
    # On by default at five, because five is what was measured and a library
    # whose default configuration is not the configuration its numbers were
    # taken from is one that misreports itself. Set to 0 for the ablation.
    # Stores written before turn_text existed have nothing to hydrate from and
    # degrade silently to facts alone, which is the old behaviour exactly.
    hydrate_top_k: int = Field(
        default=5,
        ge=0,
        description="Attach the source turns behind this many top-ranked facts",
    )
    recency_boost: float = Field(default=0.0, ge=0.0, description="Additive RRF boost for recent records")
    weight_signal: float = Field(default=0.0, ge=0.0, description="Additive RRF boost proportional to record weight")
    exclude_invalidated: bool = Field(
        default=True,
        description="Exclude records with invalid_at set from normal search. When False, they are deprioritized by reranker instead.",
    )

    def get_lambda_base(self, memory_type: MemoryType) -> float:
        match memory_type:
            case MemoryType.CORE_ANCHOR:
                return self.lambda_core_anchor
            case MemoryType.ACTIVE_CONTEXT:
                return self.lambda_active_context
            case MemoryType.EPHEMERAL_STATE:
                return self.lambda_ephemeral
