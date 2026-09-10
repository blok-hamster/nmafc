from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from nmafc.engine.reinforcement import reinforce
from nmafc.integration.base import EmbeddingProvider
from nmafc.integration.grounding import source_scores
from nmafc.integration.quantities import disagrees, quantity_cues
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType, SearchCandidate
from nmafc.storage.cold_base import ColdStorageBase
from nmafc.storage.hot import HotStorage

if TYPE_CHECKING:
    from nmafc.storage.event_log import EventLog

# Leading clock time on a session timestamp: "7:18 pm on 27 May, 2023".
CLOCK_PREFIX = re.compile(r"^\d{1,2}:\d{2}\s*[ap]\.?m\.?\s+on\s+", re.I)

WORD = re.compile(r"[a-z0-9']+")
# Deliberately small. A long stop list starts deciding which content words
# matter, and the cue here is already narrow -- one question and the handful of
# facts drawn from one turn -- so the words worth dropping are only the ones
# that appear in every question ever asked.
STOPWORDS = frozenset(
    "a an and are as at be been but by can did do does for from had has have "
    "he her him his how i if in into is it its me my not of on or our she that "
    "the their them then there they this to was we were what when where which "
    "who why will with would you your".split()
)


def cue_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower())
            if len(w) > 2 and w not in STOPWORDS}


def separate_facts(records: list[MemoryRecord],
                   threshold: float) -> list[MemoryRecord]:
    """Ranked facts with the restatements of earlier ones removed.

    Pattern separation over the rendered list. Two facts are treated as the same
    fact when the content words of one are `threshold` contained in the other's
    -- containment, not Jaccard, because the redundant case is a short summary
    wholly restated inside a longer one, and Jaccard scores that pair as barely
    similar on the strength of the extra words alone.

    Which of the pair survives is decided by specificity, not by rank. The more
    specific fact says everything the vaguer one says and more, so keeping the
    vaguer one because it happened to rank higher would discard information to
    preserve an ordering. The survivor inherits the better of the two positions,
    so ranking order is otherwise untouched and nothing is promoted past a fact
    it lost to.

    Facts with no content words at all -- an extraction that produced only
    stopwords -- are kept and compared against nothing, because a zero-length cue
    is contained in everything and would otherwise delete the whole list.

    Quantities are checked separately and can veto a merge, because the content
    words cannot see them. `cue_words` keeps tokens longer than two characters,
    so "He paid 42 dollars" and "He paid 47 dollars" both reduce to
    `{paid, dollars}`: containment is 1.0, and without the veto one of the two
    amounts is deleted as a restatement of the other, with the survivor decided
    by which happened to rank higher. Two facts that state different numbers are
    two facts.
    """
    kept: list[MemoryRecord] = []
    cues: list[set[str]] = []
    for record in records:
        cue = cue_words(record.fact_content)
        if not cue:
            kept.append(record)
            cues.append(cue)
            continue
        for i, other in enumerate(cues):
            if not other:
                continue
            shared = len(cue & other)
            if shared / min(len(cue), len(other)) < threshold:
                continue
            if disagrees(record.fact_content, kept[i].fact_content):
                continue
            # Same fact. Keep whichever carries more, at the earlier position.
            if len(cue) > len(other):
                kept[i] = record
                cues[i] = cue
            break
        else:
            kept.append(record)
            cues.append(cue)
    return kept


def split_turn(text: str) -> tuple[str | None, list[str]]:
    """A turn as its session header and its speaker lines.

    The header is the bracketed timestamp the ingester writes above every turn.
    A turn with no header is all body, which is what a store written by an older
    ingester looks like.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if lines and lines[0].lstrip().startswith("["):
        return lines[0], lines[1:]
    return None, lines


def best_lines(body: list[str], cue: set[str], keep: int,
               quantities: set[str] | None = None) -> list[str]:
    """The `keep` lines that share the most cue words, in the order they were said.

    Ties go to the earlier line. That is not arbitrary: in this transcript the
    first line of a turn is the one that raises the subject and the second is
    usually a reaction to it, so when neither matches the cue better, the one
    that introduced the topic is the more likely to carry the answer.

    A shared quantity outranks any number of shared words. A number is the most
    specific thing a line can share with the question -- several lines of a turn
    will talk about rent, but one of them says what the rent was -- and the word
    cue cannot see it, because `cue_words` drops anything two characters or
    shorter and so loses every number below 100. Passing no quantities restores
    the word-only ordering exactly.
    """
    if len(body) <= keep:
        return body
    numbers = quantities or set()

    def rank(i: int) -> tuple[int, int, int]:
        line = body[i]
        shared = len(numbers & quantity_cues(line)) if numbers else 0
        return (-shared, -len(cue & cue_words(line)), i)

    order = sorted(range(len(body)), key=rank)
    return [body[i] for i in sorted(order[:keep])]


class QueryRouter:
    """Routes retrieval queries through Hot RAM with Cold ROM fallback.

    1. Embeds the query
    2. Searches Hot RAM (LanceDB) for top_k results
    3. If best score < theta, falls back to Cold ROM keyword search
    4. Applies LTP reinforcement to retrieved records
    """

    def __init__(
        self,
        hot: HotStorage,
        cold: ColdStorageBase,
        embedder: EmbeddingProvider,
        config: DecayConfig,
    ) -> None:
        self._hot = hot
        self._cold = cold
        self._embedder = embedder
        self._config = config
        # Reinforcements held back when defer_reinforcement_writes is on, keyed
        # by record id so a record retrieved twice before a flush ends up with
        # the higher consolidation index rather than two queued writes.
        self._pending_reinforcements: dict[str, int] = {}
        self._pending_turn = 0
        # turn -> date, read once from the archive. Loaded lazily because a
        # store that was never given dates should not pay a query per retrieval
        # to rediscover that, and refreshed by reset_turn_dates() when ingestion
        # adds turns after the router has already answered something.
        self._turn_dates: dict[int, str] | None = None
        # question -> the vector `retrieve` embedded it to, so the rendering
        # pass can rank turns by meaning without paying for the embedding twice.
        self._query_vectors: dict[str, list[float]] = {}

    def reset_turn_dates(self) -> None:
        """Forget the cached turn dates, so newly ingested turns are picked up."""
        self._turn_dates = None

    @staticmethod
    def _day(stamp: str) -> str:
        """A session timestamp with its clock time removed, if it has one.

        Leaves anything else exactly as it came in: the extractor's own wording
        ("last summer"), a turn number fallback, or a date that never carried a
        time. Only the one shape it recognises is edited.
        """
        return CLOCK_PREFIX.sub("", stamp)

    def _date_for(self, turn: int | None) -> str | None:
        """The calendar date of a turn, or None if this store has no dates.

        Falls back to the nearest earlier dated turn. Dates are recorded per
        turn but a conversation is dated per session, so an undated turn sits
        inside the session opened by the last dated one -- which is a better
        answer than none, and never invents a date for a store that has no
        dates at all.
        """
        if turn is None:
            return None
        if self._turn_dates is None:
            getter = getattr(self._cold, "turn_timestamps", None)
            self._turn_dates = getter() if callable(getter) else {}
        if not self._turn_dates:
            return None
        if turn in self._turn_dates:
            return self._turn_dates[turn]
        earlier = [t for t in self._turn_dates if t <= turn]
        return self._turn_dates[max(earlier)] if earlier else None

    def flush_reinforcements(self) -> int:
        """Commit buffered LTP writebacks and report how many records moved.

        A no-op unless `defer_reinforcement_writes` is set, since without it
        nothing is ever buffered. Callers that defer must call this before any
        decay pass and before closing the store: decay reads the weight and the
        consolidation index that the buffer is holding, and an unflushed buffer
        is simply lost when the process ends.
        """
        if not self._pending_reinforcements:
            return 0
        updates = list(self._pending_reinforcements.items())
        self._hot.apply_reinforcements(updates, turn=self._pending_turn)
        self._pending_reinforcements = {}
        return len(updates)

    async def retrieve(
        self,
        query: str,
        current_turn: int,
        event_logger: EventLog | None = None,
    ) -> list[MemoryRecord]:
        """Retrieve relevant memories using unified parallel search + reranking.

        1. Searches Hot RAM (vector, top_k) and Cold ROM (semantic + keyword) in parallel.
        2. Traverses graph pointers (related_entities) up to max_hops in Hot RAM.
        3. Expands one hop into the Cold ROM archive from cold-only seeds.
        4. Reranks all candidates via Reciprocal Rank Fusion (RRF).
        5. Applies LTP reinforcement to Hot-sourced records that survive reranking.

        When `event_logger` is provided, RETRIEVAL events are emitted for each
        record retrieved, capturing the score and hop distance.
        """
        from nmafc.engine.reranking import rerank

        query_embedding = await self._embedder.embed_single(query)
        # Kept so `format_context` can rank turns by meaning. That runs later,
        # synchronously, and from a different caller, so it cannot embed the
        # question itself and re-embedding here would be a second paid call for
        # a vector we are already holding. Keyed by the question, so a lookup
        # either finds the vector for that exact question or finds nothing and
        # the ranking falls back to words; there is no key under which it can
        # return the wrong vector. Bounded because a long session must not
        # accumulate one vector per question ever asked.
        self._query_vectors[query] = query_embedding
        while len(self._query_vectors) > 8:
            self._query_vectors.pop(next(iter(self._query_vectors)))
        exclude_inv = self._config.exclude_invalidated

        # --- Step 1: Parallel search across both tiers ---
        vector_hits = self._hot.search(
            query_embedding,
            top_k=self._config.top_k,
            exclude_invalidated=exclude_inv,
        )

        cold_records: list[MemoryRecord] = []
        if self._config.always_search_cold:
            cold_records = self._search_cold(query_embedding, query)
        elif not vector_hits or vector_hits[0].score < self._config.theta:
            cold_records = self._search_cold(query_embedding, query)

        # --- Step 2: Build candidate lists with source tags ---
        candidates: list[SearchCandidate] = []
        visited_ids: set[str] = set()
        visited_entities: set[str] = set()
        record_meta: dict[str, tuple[float | None, int]] = {}

        # Hot RAM vector hits (Hop 0)
        frontier_entities: set[str] = set()
        for rank, hit in enumerate(vector_hits):
            rec = hit.record
            if rec.id not in visited_ids:
                visited_ids.add(rec.id)
                visited_entities.add(rec.entity_name.lower())
                candidates.append(SearchCandidate(
                    record=rec, score=hit.score, source="hot_vector",
                    rank_in_source=rank, hop_distance=0,
                ))
                record_meta[rec.id] = (hit.score, 0)
                for rel in rec.related_entities:
                    frontier_entities.add(rel.lower())

        # Cold ROM results
        for rank, rec in enumerate(cold_records):
            candidates.append(SearchCandidate(
                record=rec, score=None, source="cold_semantic",
                rank_in_source=rank, hop_distance=0,
            ))

        # --- Step 3: BFS expansion in Hot RAM ---
        current_hop = 0
        max_hops = self._config.max_hops

        while current_hop < max_hops and frontier_entities:
            current_hop += 1
            unvisited = [e for e in frontier_entities if e not in visited_entities]
            if not unvisited:
                break

            neighbors = self._hot.get_by_entities(unvisited, exclude_invalidated=exclude_inv)
            frontier_entities = set()

            for rec in neighbors:
                if rec.id not in visited_ids:
                    visited_ids.add(rec.id)
                    visited_entities.add(rec.entity_name.lower())
                    candidates.append(SearchCandidate(
                        record=rec, score=None, source="bfs_hot",
                        rank_in_source=0, hop_distance=current_hop,
                    ))
                    record_meta[rec.id] = (None, current_hop)
                    for rel in rec.related_entities:
                        if rel.lower() not in visited_entities:
                            frontier_entities.add(rel.lower())

        # --- Step 4: Cold ROM one-hop graph expansion ---
        if cold_records:
            seen_entities = {rec.entity_name.lower() for rec in
                            [c.record for c in candidates]}
            cold_only = [c.record for c in candidates if c.source.startswith("cold")]
            expanded = self._expand_cold_graph(cold_only, seen_entities)
            for rank, rec in enumerate(expanded):
                candidates.append(SearchCandidate(
                    record=rec, score=None, source="bfs_cold",
                    rank_in_source=rank, hop_distance=1,
                ))

        # --- Step 4b: Score the evidence, not just the summaries ---
        grounding = (self._grounded(query, candidates)
                     if self._config.source_grounding else None)

        # --- Step 5: Rerank all candidates ---
        final_records = rerank(candidates, self._config, current_turn,
                               grounding=grounding)

        # --- Step 6: LTP reinforcement (Hot-sourced records only) ---
        hot_ids = visited_ids
        reinforcements: list[tuple[str, int]] = []
        for rec in final_records:
            if rec.id in hot_ids and rec.memory_type != MemoryType.EPHEMERAL_STATE:
                reinforced = reinforce(rec, current_turn)
                reinforcements.append(
                    (reinforced.id, reinforced.consolidation_index)
                )

        if self._config.defer_reinforcement_writes:
            # Buffer instead of writing, and count from the buffer rather than
            # from `reinforcements`. Every index in that list was computed from
            # a record just read out of the store, and while writes are held
            # back the store still says k = 0, so a record retrieved three times
            # would be handed k = 1 three times over. Once a record is buffered
            # the buffer is the authority on its index, so each further
            # retrieval adds one to what is already held -- matching the
            # immediate path, where the increment reads the value the previous
            # retrieval wrote.
            #
            # The turn stamp is the latest retrieval's, which is what the
            # immediate path would have written for the records touched last;
            # earlier buffered records get a stamp later than the retrieval that
            # earned it, and that is the accuracy this trade gives up. It is
            # read only by decay, so flushing before any decay pass bounds the
            # difference by the flush interval.
            for record_id, new_k in reinforcements:
                held = self._pending_reinforcements.get(record_id)
                self._pending_reinforcements[record_id] = (
                    new_k if held is None else held + 1
                )
            self._pending_turn = max(self._pending_turn, current_turn)
            # A caller that only ever retrieves reaches none of the three
            # flush points -- decay, maintain, close -- so without a bound the
            # buffer grows for the life of the process and the reinforcement it
            # holds is never visible to anything. The bound turns "written on
            # every query" into "written every few dozen queries", which is
            # where the saving comes from, while keeping the buffer finite and
            # its staleness capped at a known number of records.
            if len(self._pending_reinforcements) >= self._config.reinforcement_buffer_limit:
                self.flush_reinforcements()
        else:
            self._hot.apply_reinforcements(reinforcements, turn=current_turn)

        # Emit RETRIEVAL events
        if event_logger is not None:
            from nmafc.schemas.events import EventType, MemoryEvent

            for rec in final_records:
                score, hops = record_meta.get(rec.id, (None, 0))
                event_logger.log(
                    MemoryEvent(
                        event_type=EventType.RETRIEVAL,
                        turn=current_turn,
                        record_id=rec.id,
                        entity_name=rec.entity_name,
                        retrieval_score=score,
                        hops=hops,
                    )
                )

        return final_records

    @staticmethod
    def _cold_row_to_record(row: dict) -> MemoryRecord:
        """Rebuild a MemoryRecord from an archived event row.

        Archived rows carry no Hot RAM id, and none is invented: these records
        are read-only passengers in the result and must never be reinforced or
        written back, or reading the archive would resurrect pruned memories
        into the working set.
        """
        related = row.get("related_entities")
        if isinstance(related, str):
            try:
                related = json.loads(related)
            except (TypeError, ValueError):
                related = []
        return MemoryRecord(
            entity_name=row["entity_name"],
            fact_content=row["fact_content"],
            memory_type=row["memory_type"],
            created_at_turn=row["turn"],
            last_reinforced_turn=row["turn"],
            related_entities=list(related or []),
        )

    def _search_cold(
        self, query_embedding: list[float], query: str
    ) -> list[MemoryRecord]:
        """Hybrid archive retrieval: dense first, keyword filling the remainder.

        `fallback_keyword_limit` is treated as the total budget for archive
        contributions rather than as a per-method limit. Running two searches
        and taking a full quota from each would double what the fallback adds to
        the prompt, and a measured A/B on Hot RAM's own graph traversal showed
        3.75x the context buying no accuracy -- more retrieved text is not
        reliably better, so the budget stays where it was.

        Dense results are taken first because they are ranked by meaning and
        keyword rank is not comparable to a cosine score, so there is no honest
        way to interleave the two by score.
        """
        budget = self._config.fallback_keyword_limit
        if budget <= 0:
            return []

        records: list[MemoryRecord] = []
        seen: set[str] = set()

        if self._config.cold_semantic_fallback:
            for row in self._cold.semantic_search(query_embedding, limit=budget):
                key = row["entity_name"].lower()
                if key not in seen:
                    seen.add(key)
                    records.append(self._cold_row_to_record(row))

        if len(records) < budget:
            for row in self._cold.keyword_search(query, limit=budget):
                key = row["entity_name"].lower()
                if key not in seen:
                    seen.add(key)
                    records.append(self._cold_row_to_record(row))
                    if len(records) >= budget:
                        break

        return records[:budget]

    def _expand_cold_graph(
        self, records: list[MemoryRecord], already_seen: set[str]
    ) -> list[MemoryRecord]:
        """Follow one hop of links from archive hits, into the archive.

        Hot RAM traversal stops at whatever Hot RAM still holds, so a link
        pointing at a pruned fact is a dead end there. The archive still has it.

        One hop, not max_hops, and only on the fallback path. The paired A/B on
        Hot RAM measured deeper traversal costing 3.75x context for no accuracy
        gain, and this path is already the expensive branch; spending the same
        way here would reproduce that result rather than learn from it. Gating
        it on the fallback is also the intended shape of the design -- Cold ROM
        does the thorough, RAG-priced search precisely when Hot RAM came up
        empty, and stays out of the way when it did not.
        """
        if not self._config.cold_semantic_fallback:
            return []

        wanted = {
            rel.lower()
            for rec in records
            for rel in rec.related_entities
            if rel.lower() not in already_seen
        }
        if not wanted:
            return []

        found: list[MemoryRecord] = []
        for row in self._cold.get_events_for_entities(
            sorted(wanted), limit=self._config.fallback_keyword_limit
        ):
            key = row["entity_name"].lower()
            if key not in already_seen:
                already_seen.add(key)
                found.append(self._cold_row_to_record(row))
        return found

    def format_context(self, records: list[MemoryRecord],
                       query: str | None = None) -> str:
        """Format retrieved memories using Zep-style structured fact presentation.

        Facts are presented as discrete items with temporal validity ranges.
        This gives the LLM scannable, atomic facts rather than a narrative list.

        Validity is shown as a calendar date wherever the store knows one, and
        as a turn number only where it does not. The distinction is not
        cosmetic. Presented as "(Valid: turn 220 - present)", a dated question
        is unanswerable however good the retrieval was, and the run's own
        answers show the model saying so: "it was mentioned in turn 220 and is
        valid from turn 220 onward, but no specific date". Twenty answers in
        that run quoted a turn number back to the user as if it were a date.

        A per-fact date recorded by the extractor wins over the date of the turn
        that mentioned it, because facts are routinely stated after the event
        ("last year we drove to the coast"). Turn dates are the fallback, and
        turn numbers the fallback of last resort.
        """
        if not records:
            return ""

        # The facts written into the prompt can be fewer than the facts
        # retrieved. `_source_turns` below still reads the full list, so
        # trimming here buys source depth instead of restating it. See
        # `context_facts_top_k`.
        #
        # Separation runs before the trim, which is the whole point of it. Run
        # afterwards it would only shorten a list of six near-copies to three;
        # run first, the six slots are filled with six facts that differ.
        distinct = records
        if self._config.fact_overlap_max is not None:
            distinct = separate_facts(records, self._config.fact_overlap_max)
        limit = self._config.context_facts_top_k
        shown = distinct if limit is None else distinct[:limit]

        lines = ["<FACTS>"] if shown else []
        for r in shown:
            valid_from = (
                r.valid_at_text
                or self._date_for(r.valid_at or r.created_at_turn)
                or f"turn {r.valid_at or r.created_at_turn}"
            )
            valid_to = None
            if r.invalid_at:
                valid_to = self._date_for(r.invalid_at) or f"turn {r.invalid_at}"

            if self._config.compact_validity:
                # "Valid:" and "- present" are the same on every line that
                # carries them, so they cost tokens without telling the model
                # anything it could not assume. An end date is only rendered
                # when the fact actually has one, which is the only case where
                # the range is doing work.
                #
                # The clock time goes with them. A session timestamp reads
                # "7:18 pm on 27 May, 2023" and the hour is 11.2 characters of
                # the 25.8, on a benchmark where no question asks what time of
                # day anything happened. Stripped only from the rendered
                # validity: the SOURCE block keeps its headers as stored,
                # because verbatim that has been edited is not verbatim.
                span = (
                    f"{self._day(valid_from)} to {self._day(valid_to)}"
                    if valid_to
                    else self._day(valid_from)
                )
                lines.append(f"{r.fact_content} ({span})")
            else:
                lines.append(
                    f"{r.fact_content} (Valid: {valid_from} - {valid_to or 'present'})"
                )
        if shown:
            lines.append("</FACTS>")

        hydrated = self._source_turns(records, query)
        if hydrated:
            if lines:
                lines.append("")
            lines.append("<SOURCE>")
            lines.extend(hydrated)
            lines.append("</SOURCE>")

        return "\n".join(lines)

    def _grounded(self, query: str,
                  candidates: list[SearchCandidate]) -> dict[str, float]:
        """How well each candidate's source turn answers the question, 0 to 1.

        Everything before this ranks the fact, which is a summary written at
        ingestion by a model that could not know what would later be asked. When
        a question turns on a word the summary dropped, no amount of reranking
        recovers it, because the word is not in anything being ranked.

        Keyed by entity name because that is what the fusion scores, and scaled
        so the best-matching turn scores 1.0. Scaling rather than using the raw
        BM25 total keeps the boost inside a known range whatever the question's
        length, which is what makes a single configured weight mean the same
        thing on every query.

        No turn is ever added as a candidate. A turn is read only because a fact
        from it already won a slot, so the retrieval budget is spent on exactly
        what it was spent on before; only the order changes.
        """
        reader = getattr(self._cold, "text_for_turns", None)
        if reader is None:
            return {}
        by_turn: dict[int, list[SearchCandidate]] = {}
        for c in candidates:
            if c.record.created_at_turn:
                by_turn.setdefault(c.record.created_at_turn, []).append(c)
        if len(by_turn) < 2:
            # Nothing to separate, and scoring one turn against itself would
            # hand a full boost to whatever happened to be the only entrant.
            return {}

        texts = reader(sorted(by_turn))
        scores = source_scores(query, {t: texts.get(t, "") for t in by_turn})
        top = max(scores.values(), default=0.0)
        if top <= 0.0:
            return {}

        out: dict[str, float] = {}
        for turn, group in by_turn.items():
            share = scores.get(turn, 0.0) / top
            if share <= 0.0:
                continue
            for c in group:
                # An entity can hold facts from several turns. The best of them
                # wins, because the boost answers "is this entity evidenced by
                # a turn that matches the question", and one turn that does is
                # enough.
                key = c.record.entity_name.lower()
                out[key] = max(out.get(key, 0.0), share)
        return out

    def _scan_span(self, records: list[MemoryRecord]) -> tuple[int, int] | None:
        """The turn range the answer is likely to be in, or None for no bound.

        Retrieval has already decided roughly where in the store this question
        belongs; the turns behind its best facts are that decision expressed as
        a number. Bounding the turn scan to that range is what stops a stranger's
        dialogue from winning a hydration slot on a shared rare word.

        Only the top `scan_span_facts` are used. The tail of the retrieved list
        is exactly where off-subject facts sit, so a span drawn from all of them
        is nearly the whole store and bounds nothing.
        """
        k = self._config.scan_span_facts
        if k <= 0:
            return None
        turns = [r.created_at_turn for r in records if r.created_at_turn][:k]
        if not turns:
            return None
        margin = self._config.scan_span_margin
        return min(turns) - margin, max(turns) + margin

    def _scanned_turns(self, query: str, have: list[int], scan: int,
                       span: tuple[int, int] | None = None) -> list[int]:
        """The best-matching turns of the whole conversation that retrieval missed.

        Everything else in this file ranks facts, and hydration then reads the
        turns behind the facts that won. That leaves one thing unreachable: a
        turn whose facts all ranked below the cut. No amount of `hydrate_pool`
        recovers it, because the pool is drawn from the retrieved list and
        nothing in the retrieved list points at it.

        The evidence for building this is the shape of what open-domain still
        gets wrong. Of 55 remaining losses, 36 have no content word of the gold
        anywhere in the rendered context, and reading them the missing word is
        usually one concrete noun the extractor generalised away: the gold is
        "Hoodies" and we answer "clothing", the gold is "tree pose" and we
        answer "Dancer Pose". The noun is in the turn. It is not in any fact,
        so it cannot be retrieved, so the turn is never nominated.

        **This does not put turns into the retrieval budget**, which is the
        objection the store's own schema comment raises against indexing turn
        text, and it is a fair objection. Turns returned here are candidates for
        hydration only, a budget of `hydrate_top_k` that is already being spent;
        the caller still takes exactly the number of turns it would have taken,
        and no fact loses a slot. The cost is a BM25 pass over a few hundred
        short strings, which is why it is capped: a question that matches
        nothing gets nothing, and `scan` bounds how many of the ones it does
        match are even allowed to compete.

        Scored against the whole conversation rather than against the pool, so
        document frequency is computed over every turn there is. That is the
        better estimate of a term's rarity and it is what makes "Hoodies"
        outrank a turn that merely repeats the question's common words.

        `span` bounds which turns may be *taken*, and deliberately not which are
        scored. Rarity is a property of the store, so narrowing the corpus would
        make a word common in one conversation look rare, which is the opposite
        of the discrimination this exists for.
        """
        if scan <= 0:
            return []
        reader = getattr(self._cold, "all_turn_text", None)
        if reader is None:
            return []
        texts = reader()
        if not texts:
            return []
        seen = set(have)
        scores = source_scores(query, texts)
        lo, hi = span if span else (None, None)
        eligible = [t for t in texts
                    if t not in seen and (lo is None or lo <= t <= hi)]
        ranked = sorted(eligible, key=lambda t: (-scores.get(t, 0.0), t))
        sims: dict[int, float] = {}
        qv = self._query_vectors.get(query)
        if qv is not None and self._config.scan_semantic > 0:
            ranked, sims = self._fuse_semantic(qv, eligible, ranked)
        # A turn no ranker has an opinion about is not a preference. Taking it
        # would spend a hydration slot on an arbitrary turn, which is strictly
        # worse than leaving the slot with the fact that earned it.
        #
        # With meaning in play the test cannot stay "shares a word", because the
        # turns this exists to reach are the ones that share no word: the turn
        # answering how John feels while surfing says "super exciting and
        # free-feeling" and is 834th by word overlap. So a turn qualifies on
        # either signal, and `scan_semantic_floor` is what stops the second one
        # from qualifying everything -- cosine is never zero, so without a floor
        # every turn in the store would pass.
        floor = self._config.scan_semantic_floor
        return [t for t in ranked[:scan]
                if scores.get(t, 0.0) > 0.0 or sims.get(t, 0.0) >= floor > 0.0]

    def _fuse_semantic(self, query_vector: list[float], eligible: list[int],
                       by_words: list[int],
                       ) -> tuple[list[int], dict[int, float]]:
        """Blend the word ranking with a meaning ranking, best first.

        Fused on rank rather than on score, because a BM25 score and a cosine
        are not on one scale and nothing in the store says how to put them
        there. Reciprocal rank fusion is the same device the retrieval side
        already uses for the same reason, so the two halves of this system
        combine evidence the same way.

        `scan_semantic` weights the meaning half. At 0 this is never called; at
        1 the two halves count equally. Measured on the 106 open-domain losses
        whose answer sits in a turn, inside the eight turns the scan takes,
        words find it 50.0% of the time, meaning 65.1%, and whichever is better
        68.9% -- so there is something in each and the blend is worth having
        rather than a straight swap.

        Returns the similarities alongside the order because the caller needs
        them to apply its floor, and computing them twice would double the only
        expensive step here.
        """
        vectors = getattr(self._cold, "all_turn_vectors", None)
        vecs = vectors() if vectors else {}
        if not vecs:
            return by_words, {}
        # Provider vectors are unit length, so the dot product is the cosine.
        sims = {t: sum(a * b for a, b in zip(query_vector, v))
                for t in eligible if (v := vecs.get(t))}
        if not sims:
            return by_words, {}
        by_meaning = sorted(sims, key=lambda t: (-sims[t], t))
        w = self._config.scan_semantic
        k = self._config.rrf_k or 60
        word_rank = {t: i for i, t in enumerate(by_words)}
        mean_rank = {t: i for i, t in enumerate(by_meaning)}
        # A turn missing from one ranking is not penalised into last place; it
        # simply scores nothing from that half, which is what "no opinion"
        # should mean.
        def fused(t: int) -> float:
            s = 0.0
            if t in word_rank:
                s += 1.0 / (k + word_rank[t] + 1)
            if t in mean_rank:
                s += w / (k + mean_rank[t] + 1)
            return s

        return sorted(eligible, key=lambda t: (-fused(t), t)), sims

    def _turns_by_question(self, query: str | None,
                           records: list[MemoryRecord],
                           want: int) -> list[int]:
        """The `want` turns that best match the question, best first.

        Empty whenever the shipped behaviour should stand: the feature off, no
        question, nothing to widen to, or a question the scorer has no opinion
        about. An empty return is the caller's signal to hydrate by fact rank as
        before, so every path that cannot improve on it leaves it alone.

        `want` is passed in rather than decided here, and that is the whole
        discipline of this function. It is the number of distinct turns fact
        rank would have hydrated anyway, so this changes *which* turns are read
        and never *how many*. A version that returned its own best `n` would be
        buying accuracy with tokens, which on a budget of 1,000 against RAG's
        1,430 is not a trade available to us.
        """
        pool_size = self._config.hydrate_pool
        scan = self._config.hydrate_scan
        if (not pool_size and not scan) or not query or want <= 0:
            return []
        pool = list(dict.fromkeys(
            r.created_at_turn for r in records[:pool_size or len(records)]
            if r.created_at_turn))
        span = self._scan_span(records)
        if span:
            # The pool is bounded as well as the scan. A retrieved fact from
            # another conversation is off-subject for the same reason a scanned
            # turn from one is, and it arrives with rank behind it, which makes
            # it likelier to win a slot rather than less. The first turn is kept
            # regardless: it is what defines the span, so excluding it would be
            # incoherent, and it is the turn fact rank was surest about.
            keep_first = pool[:1]
            pool = keep_first + [t for t in pool[1:] if span[0] <= t <= span[1]]
        pool += self._scanned_turns(query, pool, scan, span)
        # Nothing to choose from beyond what rank already picked.
        if len(pool) <= want:
            return []
        reader = getattr(self._cold, "text_for_turns", None)
        if reader is None:
            return []
        texts = reader(sorted(pool))
        scores = source_scores(query, {t: texts.get(t, "") for t in pool})
        # All-zero means the question shares no distinguishing word with any
        # turn, which is not a preference for the first `want` of them.
        if not any(scores.values()):
            return []
        # Fact rank breaks ties, so a question that cannot separate two turns
        # falls back to the order that was measured rather than to dict order.
        rank = {t: i for i, t in enumerate(pool)}
        best = sorted(pool, key=lambda t: (-scores.get(t, 0.0), rank[t]))

        # The top-ranked turn is never dropped, and the reason is in the paired
        # run's regressions. Four of five breaks were the same shape: a gold
        # spread over two turns, where A answered "showed him his work and gave
        # him advice" and B answered "showed him his work". Choosing every turn
        # by question match concentrates hydration on the single best turn and
        # drops the one holding the rest of the answer, which is a good trade
        # for a question turning on one qualifier and a bad one for a question
        # whose answer is a list.
        #
        # Keeping the first rank turn costs at most one of `want` slots and
        # makes this a refinement of the measured behaviour rather than a
        # replacement for it: the turn fact rank was surest about survives, and
        # the question spends what is left.
        keep = pool[:1]
        chosen = keep + [t for t in best if t not in keep]
        return chosen[:want]

    def _source_turns(self, records: list[MemoryRecord],
                      query: str | None = None) -> list[str]:
        """The verbatim turns behind the best-ranked facts.

        Facts are a summary of what was said, and the summary is what decays and
        what gets ranked. This restores the wording for the few facts that won
        the top of the ranking, which is the only place it is worth the tokens:
        the source of the fifth-ranked fact is evidence, the source of the
        twentieth is noise with a paragraph attached.

        Deduplicated by turn rather than by fact. One exchange routinely yields
        six facts, so hydrating five facts is usually three or four turns, and
        billing per fact would pay for the same text repeatedly.
        """
        budget = self._config.hydrate_top_k
        if not budget:
            return []
        # The turn the fact was extracted from, not `valid_at`. They usually
        # agree, but `valid_at` is when the fact became true and a fact stated
        # long after the event carries an earlier one, which would hydrate a
        # turn that never mentioned it. This is also the field the measurement
        # used, so the shipped behaviour is the one that was measured.
        kept = records[:budget]
        # Chronological for printing, best-ranked first for deciding how much
        # of each turn survives. `dict.fromkeys` keeps the ranking order and
        # drops repeats, which matters because one exchange yields six facts.
        ranked = list(dict.fromkeys(
            r.created_at_turn for r in kept if r.created_at_turn))
        chosen = self._turns_by_question(query, records, len(ranked))
        if chosen:
            ranked = chosen
            # The cues below are drawn from the facts of the turns being
            # hydrated, so they have to follow the turns rather than the rank.
            wanted = set(ranked)
            kept = [r for r in records if r.created_at_turn in wanted]
        turns = sorted(ranked)
        texts = self._cold.text_for_turns(turns)

        keep = self._config.hydrate_lines
        if keep is None and not self._config.dedupe_source_headers:
            return [texts[t] for t in turns if texts.get(t)]
        whole = set(ranked[: self._config.hydrate_full_turns])

        # The cue for a turn is the question plus the facts extracted from that
        # turn. The facts alone are not enough -- extraction is what dropped the
        # qualifier the question turns on, so a fact-only cue would reliably
        # keep the line the fact already covers and discard the one that
        # completes it. The question alone is not enough either, because a
        # one-word question shares almost nothing with anything.
        asked = cue_words(query or "")
        # The same cue, in numbers. Kept apart from the words rather than merged
        # into them so that `best_lines` can rank a shared quantity above any
        # number of shared words, which is what a question asking how much or
        # how many turns on.
        asked_numbers = quantity_cues(query or "")
        by_turn: dict[int, set[str]] = {}
        numbers_by_turn: dict[int, set[str]] = {}
        for r in kept:
            if r.created_at_turn:
                by_turn.setdefault(r.created_at_turn, set()).update(
                    cue_words(r.fact_content))
                numbers_by_turn.setdefault(r.created_at_turn, set()).update(
                    quantity_cues(r.fact_content))

        out: list[str] = []
        last_header: str | None = None
        for turn in turns:
            text = texts.get(turn)
            if not text:
                continue
            header, body = split_turn(text)
            if keep is not None and turn not in whole:
                body = best_lines(body, asked | by_turn.get(turn, set()), keep,
                                  asked_numbers | numbers_by_turn.get(turn, set()))
            if header and (not self._config.dedupe_source_headers
                           or header != last_header):
                out.append(header)
            if header:
                last_header = header
            out.extend(body)
        return out
