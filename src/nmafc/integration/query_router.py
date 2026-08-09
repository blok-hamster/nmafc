from __future__ import annotations

from nmafc.engine.reinforcement import reinforce
from nmafc.integration.base import EmbeddingProvider
from nmafc.schemas.memory import DecayConfig, MemoryRecord, SearchResult
from nmafc.storage.cold import ColdStorage
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
        cold: ColdStorage,
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
        """Retrieve relevant memories for the given query.

        Applies spaced repetition (LTP) to all retrieved records as a side effect.
        """
        query_embedding = await self._embedder.embed_single(query)

        results = self._hot.search(query_embedding, top_k=self._config.top_k)

        if not results or results[0].score < self._config.theta:
            cold_results = self._cold.keyword_search(
                query, limit=self._config.fallback_keyword_limit
            )
            if cold_results and not results:
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

        for result in results:
            reinforced = reinforce(result.record, current_turn)
            self._hot.update_reinforcement(
                reinforced.id,
                new_k=reinforced.consolidation_index,
                turn=current_turn,
            )

        return [r.record for r in results]

    def format_context(self, records: list[MemoryRecord]) -> str:
        """Format retrieved memories as a context string for the LLM."""
        if not records:
            return ""

        lines = []
        for r in records:
            lines.append(f"- [{r.memory_type.value}] {r.entity_name}: {r.fact_content}")
        return "\n".join(lines)
