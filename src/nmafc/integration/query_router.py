from __future__ import annotations

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

        if not vector_hits or vector_hits[0].score < self._config.theta:
            cold_results = self._cold.keyword_search(
                query, limit=self._config.fallback_keyword_limit
            )
            if cold_results and not vector_hits:
                return [
                    MemoryRecord(
                        entity_name=r["entity_name"],
                        fact_content=r["fact_content"],
                        memory_type=r["memory_type"],
                        created_at_turn=r["turn"],
                        last_reinforced_turn=r["turn"],
                    )
                    for r in cold_results
                ]

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

        # Apply Long-Term Potentiation (LTP) Reinforcement to retrieved records
        final_records: list[MemoryRecord] = []
        for rec in active_records:
            if rec.weight < self._config.w_prune:
                continue

            final_records.append(rec)

            if rec.memory_type != MemoryType.EPHEMERAL_STATE:
                reinforced = reinforce(rec, current_turn)
                self._hot.update_reinforcement(
                    reinforced.id,
                    new_k=reinforced.consolidation_index,
                    turn=current_turn,
                )

        return final_records

    def format_context(self, records: list[MemoryRecord]) -> str:
        """Format retrieved memories as a context string for the LLM."""
        if not records:
            return ""

        lines = []
        for r in records:
            lines.append(f"- [{r.memory_type.value}] {r.entity_name}: {r.fact_content}")
        return "\n".join(lines)
