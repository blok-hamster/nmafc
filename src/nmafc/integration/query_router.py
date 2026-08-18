from __future__ import annotations

import json

from nmafc.engine.reinforcement import reinforce
from nmafc.integration.base import EmbeddingProvider
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType
from nmafc.storage.cold_base import ColdStorageBase
from nmafc.storage.hot import HotStorage


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
    ) -> list[MemoryRecord]:
        """Retrieve relevant memories for the given query using Spreading Activation.

        1. Performs top_k vector similarity search in Hot RAM (Hop 0).
        2. Traverses graph pointers (related_entities) up to max_hops (Default: 2-hops).
        3. Falls back to Cold ROM keyword search if initial vector hits are weak.
        4. Applies spaced repetition (LTP reinforcement) to all retrieved records.
        """
        query_embedding = await self._embedder.embed_single(query)

        vector_hits = self._hot.search(query_embedding, top_k=self._config.top_k)

        # Cold ROM fallback. theta is compared against a cosine similarity (see
        # HotStorage.search), so a weak best hit means Hot RAM holds nothing
        # on-topic and the archive is worth consulting.
        #
        # These records are merged into the result at the end rather than
        # replacing the Hot RAM hits. The previous version fetched them and then
        # threw them away unless vector_hits was completely empty, which on a
        # populated store never happens -- so the fallback did the keyword search
        # on every weak query and used the answer on none of them.
        #
        # The fallback is hybrid. Keyword search alone could only reach facts
        # that reuse the question's words, so a question about someone's aunt
        # could not match a fact filed as "mother's sister" even though the
        # archive held it -- and the archive is the only complete record, since
        # Hot RAM prunes and Cold ROM never does. Dense search over the same
        # embeddings Hot RAM uses closes that gap; keyword search is kept
        # alongside it because it still wins on rare exact tokens (names, dates,
        # drug names) that embeddings blur together, and because rows archived
        # before vectors were stored have no embedding to match on.
        cold_records: list[MemoryRecord] = []
        if not vector_hits or vector_hits[0].score < self._config.theta:
            cold_records = self._search_cold(query_embedding, query)

        # Spreading Activation Graph Traversal
        visited_ids: set[str] = set()
        visited_entities: set[str] = set()
        active_records: list[MemoryRecord] = []

        # Hop 0: Vector Search Hits
        frontier_entities: set[str] = set()
        for hit in vector_hits:
            rec = hit.record
            if rec.id not in visited_ids:
                visited_ids.add(rec.id)
                visited_entities.add(rec.entity_name.lower())
                active_records.append(rec)
                for rel in rec.related_entities:
                    frontier_entities.add(rel.lower())

        # Hop 1 to max_hops Spreading Activation
        current_hop = 0
        max_hops = getattr(self._config, "max_hops", 2)

        while current_hop < max_hops and frontier_entities:
            current_hop += 1
            unvisited = [e for e in frontier_entities if e not in visited_entities]
            if not unvisited:
                break

            neighbors = self._hot.get_by_entities(unvisited)
            frontier_entities = set()

            for rec in neighbors:
                if rec.id not in visited_ids:
                    visited_ids.add(rec.id)
                    visited_entities.add(rec.entity_name.lower())
                    active_records.append(rec)
                    for rel in rec.related_entities:
                        if rel.lower() not in visited_entities:
                            frontier_entities.add(rel.lower())

        # Apply Long-Term Potentiation (LTP) Reinforcement to retrieved records.
        # The writes are collected and applied in one batch: Spreading Activation
        # routinely surfaces dozens of records per question, and reinforcing them
        # one at a time cost a scan, a delete and an add each.
        final_records: list[MemoryRecord] = []
        reinforcements: list[tuple[str, int]] = []
        for rec in active_records:
            if rec.weight < self._config.w_prune:
                continue

            final_records.append(rec)

            if rec.memory_type != MemoryType.EPHEMERAL_STATE:
                reinforced = reinforce(rec, current_turn)
                reinforcements.append(
                    (reinforced.id, reinforced.consolidation_index)
                )

        self._hot.apply_reinforcements(reinforcements, turn=current_turn)

        # Cold ROM records live outside Hot RAM, so they carry no id there and
        # are not reinforced -- reading the archive must not resurrect a memory
        # into the working set. Entities Hot RAM already surfaced are skipped, so
        # the fallback only ever adds facts the vector search missed.
        if cold_records:
            seen = {rec.entity_name.lower() for rec in final_records}
            merged: list[MemoryRecord] = []
            for rec in cold_records:
                key = rec.entity_name.lower()
                if key not in seen:
                    seen.add(key)
                    merged.append(rec)
            final_records.extend(merged)
            # Links out of the archive hits, followed inside the archive. Only
            # from what actually made it into the result, so a fact filtered out
            # as a duplicate does not still drag its neighbours in.
            final_records.extend(self._expand_cold_graph(merged, seen))

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
        """Format retrieved memories as a context string for the LLM."""
        if not records:
            return ""

        lines = []
        for r in records:
            lines.append(f"- [{r.memory_type.value}] {r.entity_name}: {r.fact_content}")
        return "\n".join(lines)
