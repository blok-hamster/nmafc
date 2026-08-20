from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nmafc.engine.reinforcement import reinforce
from nmafc.integration.base import EmbeddingProvider
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType, SearchCandidate
from nmafc.storage.cold_base import ColdStorageBase
from nmafc.storage.hot import HotStorage

if TYPE_CHECKING:
    from nmafc.storage.event_log import EventLog


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

        # --- Step 5: Rerank all candidates ---
        final_records = rerank(candidates, self._config, current_turn)

        # --- Step 6: LTP reinforcement (Hot-sourced records only) ---
        hot_ids = visited_ids
        reinforcements: list[tuple[str, int]] = []
        for rec in final_records:
            if rec.id in hot_ids and rec.memory_type != MemoryType.EPHEMERAL_STATE:
                reinforced = reinforce(rec, current_turn)
                reinforcements.append(
                    (reinforced.id, reinforced.consolidation_index)
                )

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

    def format_context(self, records: list[MemoryRecord]) -> str:
        """Format retrieved memories using Zep-style structured fact presentation.

        Facts are presented as discrete items with temporal validity ranges.
        This gives the LLM scannable, atomic facts rather than a narrative list.
        """
        if not records:
            return ""

        lines = ["<FACTS>"]
        for r in records:
            valid_from = f"turn {r.valid_at}" if r.valid_at else f"turn {r.created_at_turn}"
            valid_to = f"turn {r.invalid_at}" if r.invalid_at else "present"
            lines.append(f"{r.fact_content} (Valid: {valid_from} - {valid_to})")
        lines.append("</FACTS>")

        return "\n".join(lines)
