from __future__ import annotations

from pathlib import Path

from nmafc.engine.decay import decay_all
from nmafc.engine.pruning import apply_suppression, detect_override, prune_cycle
from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.integration.extractor import StateExtractor
from nmafc.integration.query_router import QueryRouter
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryStateUpdate, UnifiedMemoryPayload
from nmafc.storage.cold import ColdStorage
from nmafc.storage.config import NMafcConfig
from nmafc.storage.hot import HotStorage


class NeuromorphicMemory:
    """Top-level neuromorphic memory wrapper for LLM agents.

    Orchestrates real-time state extraction, dual-track storage,
    cognitive decay, spaced repetition, and active pruning.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        config: NMafcConfig | None = None,
        config_path: str | Path = "configs/default.toml",
    ) -> None:
        if config is None:
            config = NMafcConfig.from_env_or_toml(config_path)

        self._config = config
        self._decay_config = config.decay
        self._current_turn: int = 0

        self._hot = HotStorage(config.storage)
        self._cold = ColdStorage(config.storage.cold_uri)
        self._embedder = embedding_provider
        self._extractor = StateExtractor(llm_provider)
        self._router = QueryRouter(
            self._hot, self._cold, self._embedder, self._decay_config
        )

    @classmethod
    def from_config(
        cls,
        config: NMafcConfig | None = None,
        config_path: str | Path = "configs/default.toml",
    ) -> NeuromorphicMemory:
        """Create a NeuromorphicMemory instance using provider_model strings from config.

        Automatically constructs LLM and embedding providers based on the
        llm_provider_model and embedding_provider_model config fields.

        Supports: openai, anthropic, groq, openrouter, together, ollama, lmstudio, vllm.
        """
        from nmafc.integration.factory import create_embedding_provider, create_llm_provider

        if config is None:
            config = NMafcConfig.from_env_or_toml(config_path)

        llm = create_llm_provider(config.llm_provider_model)
        embedder = create_embedding_provider(config.embedding_provider_model)
        return cls(llm_provider=llm, embedding_provider=embedder, config=config)

    @property
    def current_turn(self) -> int:
        return self._current_turn

    async def process_turn(
        self,
        user_msg: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Process a single conversation turn through the full neuromorphic pipeline.

        1. Increment turn counter
        2. Retrieve relevant context from Hot RAM
        3. Generate response + extract state updates via LLM
        4. Process each update: log to Cold ROM, detect overrides, upsert to Hot RAM
        5. Run decay on all mutable records
        6. Prune below-threshold records

        Returns the assistant's response text.
        """
        self._current_turn += 1
        history = conversation_history or []

        retrieved = await self._router.retrieve(user_msg, self._current_turn)
        memory_context = self._router.format_context(retrieved)

        response_text, payload = await self._extractor.extract(
            user_msg=user_msg,
            context=history,
            memory_context=memory_context if memory_context else None,
        )

        await self._process_updates(payload)

        self._run_decay()

        prune_cycle(self._hot, self._cold, self._decay_config.w_prune, self._current_turn)

        return response_text

    async def ingest_updates(self, updates: list[MemoryStateUpdate]) -> None:
        """Manually ingest memory updates without an LLM call.

        Useful for benchmarking and testing.
        """
        self._current_turn += 1
        await self._process_updates(UnifiedMemoryPayload(updates=updates))
        self._run_decay()
        prune_cycle(self._hot, self._cold, self._decay_config.w_prune, self._current_turn)

    async def _process_updates(self, payload: UnifiedMemoryPayload) -> None:
        """Process extracted memory updates: log, suppress overrides, upsert."""
        for update in payload.updates:
            self._cold.append_event(update, self._current_turn)

            existing = self._hot.get_by_entity(update.entity_name)
            if update.overrides_entity:
                existing += self._hot.get_by_entity(update.overrides_entity)

            overrides = detect_override(update, existing)
            for old_record in overrides:
                suppressed = apply_suppression(old_record, self._decay_config.gamma)
                self._hot.update_weight(old_record.id, suppressed.weight)

            embedding = await self._embedder.embed_single(update.fact_content)
            record = MemoryRecord(
                entity_name=update.entity_name,
                fact_content=update.fact_content,
                memory_type=update.memory_type,
                weight=1.0,
                consolidation_index=0,
                created_at_turn=self._current_turn,
                last_reinforced_turn=self._current_turn,
            )
            self._hot.upsert(record, embedding)

    def _run_decay(self) -> None:
        """Apply decay to all mutable records in Hot RAM."""
        mutable = self._hot.get_all_mutable()
        weight_updates = decay_all(mutable, self._current_turn, self._decay_config)
        for record_id, new_weight in weight_updates:
            self._hot.update_weight(record_id, new_weight)

    async def rollback(self, to_turn: int) -> int:
        """Rebuild Hot RAM state from Cold ROM up to the specified turn."""
        from nmafc.engine.rollback import rebuild_hot_from_cold

        async def _embed(text: str) -> list[float]:
            return await self._embedder.embed_single(text)

        def sync_embed(text: str) -> list[float]:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, _embed(text)).result()
            return asyncio.run(_embed(text))

        restored = rebuild_hot_from_cold(
            self._cold, self._hot, sync_embed, self._decay_config, to_turn
        )
        self._current_turn = to_turn
        return restored

    def get_hot_stats(self) -> dict:
        """Get statistics about the current Hot RAM state."""
        all_records = self._hot.get_all()
        if not all_records:
            return {"count": 0, "avg_weight": 0.0, "types": {}}

        weights = [r.weight for r in all_records]
        type_counts: dict[str, int] = {}
        for r in all_records:
            key = r.memory_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "count": len(all_records),
            "avg_weight": sum(weights) / len(weights),
            "types": type_counts,
        }

    def get_cold_stats(self) -> dict:
        """Get statistics about the Cold ROM event log."""
        return {
            "total_events": self._cold.count_total(),
            "active_events": self._cold.count_active(),
        }

    def close(self) -> None:
        """Close storage resources and connections."""
        self._cold.close()

