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
    # On by default. Measured over 240 retrievals against a copy of a finished
    # LoCoMo store, immediate writeback costs a mean of 501 ms per retrieval
    # against 300 ms deferred, and the gap widens as the run goes on -- 420 ms
    # over the first forty queries and 561 ms over the last eighty, because the
    # tombstoned fragments accumulate. Deferred is flat across the same span
    # (307, 287, 308, 294). About 260 ms of both figures is the embedding call,
    # so what this removes is roughly six times the rest of the local work put
    # together, and it is the only part that grows.
    #
    # It is not free of consequence: a buffered reinforcement is invisible to
    # any retrieval before the flush, so with `weight_signal` above zero a
    # deferred run can rank differently from an immediate one. At
    # `weight_signal = 0` (the default) weight does not enter ranking and the
    # two are equivalent. Set False to restore per-query writeback.
    #
    # Nothing is lost by leaving it on. Decay, `maintain()` and `close()` all
    # flush first, and `reinforcement_buffer_limit` bounds a caller that reaches
    # none of them.
    defer_reinforcement_writes: bool = Field(
        default=True,
        description="Buffer LTP writebacks until flush_reinforcements() is called",
    )
    reinforcement_buffer_limit: int = Field(
        default=256,
        gt=0,
        description="Flush buffered LTP writebacks once this many records are held",
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF constant k (higher = less weight to top ranks)")
    rerank_top_k: int = Field(default=20, gt=0, description="Max records surviving reranking into prompt")
    # Drops three things that cost tokens without carrying information: the
    # word "Valid:", the "- present" suffix, and the clock time on a session
    # timestamp. Measured over 200 prompts on the finished LoCoMo stores, the
    # validity span was 227 tokens of a 1,124-token context (20.2%), all 4,000
    # suffixes in the sample ended "- present" because nothing in those stores
    # had been invalidated, and the clock time was 11.2 characters of a 25.8-
    # character date on a benchmark where no question asks the hour. Compact
    # renders 982 tokens against 1,124: 142 fewer, 12.6%.
    #
    # No information leaves the prompt. The date survives in full and an end
    # date is still rendered when a fact has one, which is the only case where
    # the range is doing any work.
    #
    # On by default on a paired A/B over 1,537 questions, both arms reading the
    # same stores with the same retrieval and the same judge: 66.23% verbose
    # against 66.10% compact, 37 fixed and 39 broken, McNemar p = 0.91, at
    # 1,137 tokens against 997. The honest reading of that is not "identical"
    # but "any real effect is inside +/-1.2 points", and 12.4% of the context
    # is a certain saving against an unmeasurable cost. Set False to restore
    # the full span, which is what the 66.95% headline run rendered.
    compact_validity: bool = Field(
        default=True,
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
    # How many facts are written into the prompt, when that should differ from
    # how many are retrieved and hydrated. Unset means all of them, which is the
    # behaviour every measurement before this field was added; 0 means the
    # prompt carries source turns and no fact list at all.
    #
    # It exists because the context pays twice for the same information. A fact
    # is a summary of a turn, so a prompt carrying twenty facts and the twenty
    # turns behind them spends about 619 tokens restating what the other 1,250
    # already say. The turns are the half worth keeping: extraction strips the
    # qualifier a question turns on -- gripping, next month, two weeks before 11
    # August -- and the verbatim turn still has it. Without this field the two
    # cannot be separated, because `_source_turns` hydrates out of the same list
    # `format_context` renders, so trimming facts trims the source with them.
    #
    # Ranking is untouched. The facts dropped here are the lowest-ranked ones,
    # they were retrieved and they still decide which turns get hydrated; they
    # are only not restated.
    context_facts_top_k: int | None = Field(
        default=None,
        ge=0,
        description="Render only this many top-ranked facts (None = all retrieved)",
    )
    # Pattern separation, in the dentate gyrus sense: make similar traces less
    # similar before they compete, so that near-identical memories do not blur
    # into one another. This store has the opposite property. Extraction writes a
    # fresh summary per turn, the same standing fact gets restated across a
    # conversation, and the rendered fact list was measured to be almost entirely
    # duplication.
    #
    # The evidence that this costs accuracy and not merely tokens: cutting the
    # rendered list from twenty facts to six fixed 17 open-domain questions that
    # were previously wrong, and inspection of those 17 showed the fix was the
    # removal of a competing summary the model had been answering from. Cutting
    # by rank is a blunt instrument for that -- it discards distinct facts and
    # duplicates alike, and the same cut broke 26 questions elsewhere. This
    # removes the duplicates specifically and keeps the distinct facts.
    #
    # Overlap is lexical, over the same content words `cue_words` already
    # extracts for hydration. No embedding call, so it costs nothing per query
    # and works identically for Cold ROM records, which carry no vector.
    # Containment rather than Jaccard: "Caroline plays saxophone" is wholly
    # contained in "Caroline plays saxophone in a jazz quartet", and it is
    # precisely that redundant restatement Jaccard would score as only half
    # similar and keep.
    #
    # Applied to the rendered facts alone. `_source_turns` still reads the full
    # ranked list, so hydration is byte-identical to what was measured and the
    # context can only get smaller, never larger.
    #
    # None disables it, which is the behaviour every result to date was measured
    # under.
    fact_overlap_max: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Drop a fact whose content words are this contained in a "
                    "better-ranked fact's (None = keep every fact)",
    )
    # Hydration currently restores whole turns, and a turn is an episode: a
    # session header, then both speakers' lines. The answer is usually one
    # clause of one line. At about 50 tokens a turn and 20 turns hydrated, that
    # is roughly 1,250 tokens to deliver a few dozen useful ones.
    #
    # Brains do not replay episodes to recall a detail. The hippocampal index
    # points at a cortical pattern and reinstates the part the cue calls for.
    # These two fields are that idea made cheap, and neither costs an API call.
    #
    # `hydrate_lines` keeps the N speaker lines of each turn that best match the
    # cue -- the question, plus the facts that were extracted from that very
    # turn, which is the closest thing the store has to a pointer at the right
    # fragment. Order within the turn is preserved, so what survives still reads
    # as dialogue.
    hydrate_lines: int | None = Field(
        default=None,
        ge=1,
        description="Keep only the N best-matching speaker lines of each "
                    "hydrated turn (None = the whole turn)",
    )
    # Recall is not uniform, and applying one granularity to forty turns throws
    # away the ranking that retrieval just spent its effort producing. A strong
    # cue reinstates rich detail; a weak one reinstates a fragment. So the
    # best-ranked turns come back whole and the tail comes back as single lines,
    # which is what lets the window be wide and the bill small at the same time.
    #
    # Counted by rank, not by date. The turns are printed in the order they were
    # said, because that is what makes them readable, but which ones get the
    # full treatment is decided by where their facts finished in the ranking.
    hydrate_full_turns: int = Field(
        default=0,
        ge=0,
        description="Top-ranked turns returned whole before `hydrate_lines` "
                    "applies to the rest",
    )
    # `[Session - 1:56 pm on 8 May, 2023]` is repeated verbatim on every turn of
    # a session, so twenty hydrated turns can carry the same header a dozen
    # times. Printing it once per run of turns that share it is lossless: the
    # turns are emitted in order, so a header still applies to everything under
    # it until the next one.
    dedupe_source_headers: bool = Field(
        default=False,
        description="Print a repeated session header once per run of turns",
    )
    # Separate the facts the other signals cannot tell apart, using the turns
    # they came from. Reranking fuses vector rank, keyword rank and graph
    # distance, none of which sees the question's wording, so two facts stored
    # as "Evan is reading X" and "Evan is reading Y" are identical to it even
    # when the question says which one he found gripping -- the word the
    # extractor dropped is still sitting in the turn.
    #
    # An additive boost rather than a sixth fused list, and the difference is
    # the whole design. Tried as a list it was worth as much as topping vector
    # search, because RRF weights every list alike; screened free, that bought
    # three of 69 open-domain losses and reshuffled the facts of 54 of 60
    # questions already answered correctly. The diagnosis was near-ties, so the
    # fix has to be sized like a tie-break. Scale: two facts adjacent in one
    # fused list differ by about 0.0003, and belonging to a whole extra list is
    # worth about 0.016. A weight in between moves a fact past its near-copy
    # and leaves a fact that won on merit where it was.
    #
    # Zero by default: it reorders results that have already been measured.
    source_grounding: float = Field(
        default=0.0, ge=0.0,
        description="Additive RRF boost proportional to how well a fact's "
                    "source turn matches the question",
    )
    # Which turns get hydrated is, as shipped, decided entirely by fact rank:
    # `_source_turns` takes `records[:hydrate_top_k]` and hydrates whatever turns
    # those facts came from. The question is already used to choose *lines within*
    # a turn, and never to choose *which turns*.
    #
    # That is backwards for the one bucket that decides open-domain. 47 of 116
    # open-domain losses had the answer in the prompt and lost anyway, because
    # extraction dropped the qualifier the question turns on -- `gripping`, `two
    # weeks before 11 August`, `next month` -- leaving several stored facts that
    # match the question equally. The dropped word is still in the turn. So the
    # turn worth hydrating is the one that answers the question, and fact rank
    # cannot know which that is, because nothing upstream of it has read the
    # question's wording either.
    #
    # This widens the pool the turns are chosen *from* without widening how many
    # are chosen: the same number of turns is hydrated, picked by BM25 of the
    # question against each turn instead of by the rank of the facts beneath it.
    # Token cost is therefore the same turn count, and the screen measures the
    # width rather than assuming it, because turns are not all the same length.
    #
    # Zero by default, which is the shipped behaviour exactly.
    hydrate_pool: int = Field(
        default=0, ge=0,
        description="Choose hydrated turns by question match from the turns of "
                    "this many top facts (0 = by fact rank alone)",
    )
    # `hydrate_pool` widens the choice to the turns behind more retrieved facts.
    # This widens it past retrieval entirely: the top `hydrate_scan` turns of
    # the whole conversation by question match, whether or not any fact of
    # theirs was retrieved.
    #
    # The distinction is the point. A turn whose facts all rank below the cut is
    # unreachable under `hydrate_pool` at any setting, because the pool is drawn
    # from the retrieved list and nothing nominates it. That is most of what is
    # left of the open-domain gap: of 55 losses, 36 have no content word of the
    # gold anywhere in the rendered context, and the missing word is typically
    # one concrete noun -- "Hoodies" where we answered "clothing", "tree pose"
    # where we answered "Dancer Pose".
    #
    # It costs no tokens and takes no retrieval slot. The number of hydrated
    # turns is still `hydrate_top_k`; only which turns changes, and the turn
    # fact rank was surest about is still reserved.
    hydrate_scan: int = Field(
        default=0, ge=0,
        description="Also consider the top N turns of the whole conversation "
                    "by question match, past what retrieval returned "
                    "(0 = only turns behind retrieved facts)",
    )
    # `hydrate_scan` searches every turn in the store, and in a store holding
    # more than one conversation that is too wide. Asked which novel Evan finds
    # gripping, four of five hydrated turns came from other people entirely --
    # two strangers discussing houses, another a screenplay, another a different
    # novel -- each matching on "gripping" or "novel" and none of them about
    # Evan. Most of the source budget went on distractors, and the model, shown
    # a confident wrong candidate first, stopped looking.
    #
    # The fix is to bound the search by what retrieval already believes. The
    # turns of the top few facts say roughly where in the store the answer lives;
    # turns far outside that range are matching on a word rather than on the
    # subject. Measured on 90 open-domain losses: the span of the top 5 facts
    # widened by 80 turns excludes 36% of hydrated turns while still containing
    # the gold turn in 98% of the questions where the gold is in a turn at all.
    #
    # This is inference from retrieval's own output, not knowledge of which
    # conversation a question came from, so it holds on a store whose sources
    # were never labelled -- which is the case here, since the merge flattened
    # ten conversations onto one conversation id.
    #
    # Zero disables it, which is the shipped behaviour exactly.
    scan_span_facts: int = Field(
        default=0, ge=0,
        description="Bound scanned turns to the turn range of this many top "
                    "facts (0 = scan the whole store)",
    )
    scan_span_margin: int = Field(
        default=0, ge=0,
        description="Widen the scan span by this many turns at each end",
    )
    # The scan ranks turns by words, and words are why it works and why it
    # stops working. It was built to recover a concrete noun the extractor
    # generalised away -- "Hoodies" where we answered "clothing" -- and BM25 is
    # exactly the thing that rewards a rare word. It is blind to the failure
    # that is left, where the question and the turn holding the answer say the
    # same thing in different words.
    #
    # Measured on the 106 open-domain losses whose answer is in a turn, looking
    # at the eight turns the scan takes: words find that turn 50.0% of the time,
    # meaning 65.1%, and whichever of the two is better 68.9%. Asked how John
    # feels while surfing, the turn saying "super exciting and free-feeling" is
    # 834th by words and 5th by meaning.
    #
    # Both halves carry something, hence a blend rather than a swap, fused on
    # rank because a BM25 score and a cosine share no scale. This weights the
    # meaning half; 0 is the shipped behaviour and needs no index.
    scan_semantic: float = Field(
        default=0.0, ge=0.0,
        description="Weight of meaning against words when ranking scanned "
                    "turns (0 = words alone, 1 = equal)",
    )
    # A turn with no matching word is not taken, because spending a hydration
    # slot on an arbitrary turn is worse than leaving it with the fact that
    # earned it. Meaning needs its own version of that test: a cosine is never
    # zero, so without a floor every turn in the store would look like a match.
    scan_semantic_floor: float = Field(
        default=0.0, ge=0.0,
        description="Minimum similarity for a turn with no matching word to be "
                    "taken on meaning alone (0 = never)",
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
