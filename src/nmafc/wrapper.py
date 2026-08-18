from __future__ import annotations

import os
from pathlib import Path

from nmafc.engine.consolidation import MemoryConsolidator
from nmafc.engine.decay import build_entity_graph, decay_all
from nmafc.engine.pruning import apply_suppression, detect_override, prune_cycle
from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.integration.extractor import StateExtractor
from nmafc.integration.query_router import QueryRouter
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryStateUpdate, UnifiedMemoryPayload
from nmafc.storage.cold_base import ColdStorageBase
from nmafc.storage.config import NMafcConfig
from nmafc.storage.hot import HotStorage

# Seconds to wait on the embedding-dimension probe before falling back to the
# configured dimension. Bounded because the probe can block indefinitely.
EMBED_PROBE_TIMEOUT = float(os.environ.get("NMAFC_EMBED_PROBE_TIMEOUT", "30"))


class NeuromorphicMemory:
    """Top-level neuromorphic memory wrapper for LLM agents.

    Orchestrates real-time state extraction, dual-track storage,
    cognitive decay, spaced repetition, graph spreading activation,
    and active pruning.
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

        # Auto-detect embedding dimension if using default or un-synced config.
        #
        # Set NMAFC_EMBEDDING_DIM to skip the probe entirely. Prefer that when
        # constructing instances concurrently: the probe below runs the async
        # provider on a *second* event loop, and if that provider's HTTP client
        # was already used on the calling loop its connection pool is bound
        # there, so the probe blocks until the timeout expires.
        if not os.environ.get("NMAFC_EMBEDDING_DIM"):
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    # Not a `with` block: its __exit__ shuts the pool down with
                    # wait=True, which would re-block for exactly as long as the
                    # timeout was meant to avoid.
                    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        sample_vec = pool.submit(
                            asyncio.run, embedding_provider.embed_single("test")
                        ).result(timeout=EMBED_PROBE_TIMEOUT)
                    finally:
                        pool.shutdown(wait=False)
                else:
                    sample_vec = asyncio.run(embedding_provider.embed_single("test"))
                if sample_vec and len(sample_vec) > 0:
                    config.storage.embedding_dim = len(sample_vec)
            except Exception:
                # Falls back to the configured dimension rather than hanging.
                pass

        self._config = config
        self._decay_config = config.decay
        self._current_turn: int = 0

        self._hot = HotStorage(config.storage)
        if config.storage.cold_is_postgres:
            from nmafc.storage.cold_pg import PostgresColdStorage
            self._cold: ColdStorageBase = PostgresColdStorage(
                config.storage.cold_uri,
                agent_id=config.storage.agent_id,
                conversation_id=config.storage.conversation_id,
            )
        else:
            from nmafc.storage.cold import ColdStorage
            self._cold = ColdStorage(
                config.storage.cold_uri,
                agent_id=config.storage.agent_id,
                conversation_id=config.storage.conversation_id,
            )
        self._embedder = embedding_provider
        self._extractor = StateExtractor(llm_provider)
        self._router = QueryRouter(
            self._hot, self._cold, self._embedder, self._decay_config
        )
        self._consolidator = MemoryConsolidator(
            self._hot, self._cold, self._decay_config
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
        2. Retrieve relevant context from Hot RAM (using Spreading Activation)
        3. Generate response + extract state updates via LLM
        4. Process each update: log to Cold ROM, detect overrides, upsert to Hot RAM
        5. Run decay on all mutable records
        6. Prune below-threshold records
        7. Trigger auto-consolidation if interval threshold reached

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

        auto_interval = getattr(self._decay_config, "auto_consolidate_turns", 5)
        if auto_interval > 0 and self._current_turn % auto_interval == 0:
            self._consolidator.consolidate(self._current_turn)

        return response_text

    async def ingest_updates(self, updates: list[MemoryStateUpdate]) -> None:
        """Manually ingest memory updates without an LLM call.

        Useful for benchmarking and testing.
        """
        self._current_turn += 1
        await self._process_updates(UnifiedMemoryPayload(updates=updates))
        self._run_decay()
        prune_cycle(self._hot, self._cold, self._decay_config.w_prune, self._current_turn)

        auto_interval = getattr(self._decay_config, "auto_consolidate_turns", 5)
        if auto_interval > 0 and self._current_turn % auto_interval == 0:
            self._consolidator.consolidate(self._current_turn)

    async def consolidate(self) -> int:
        """Manually invoke REM sleep consolidation pass over Hot RAM."""
        return self._consolidator.consolidate(self._current_turn)

    async def _process_updates(self, payload: UnifiedMemoryPayload) -> None:
        """Process extracted memory updates: log, suppress overrides, upsert."""
        updates = payload.updates
        if not updates:
            return

        # Embed the whole turn in one request. Every provider's embed() takes a
        # list and sends it as a single HTTP body (Azure and OpenAI batch up to
        # 2048 inputs at a time), but this loop used to call embed_single() per
        # fact, which is embed([one_text]) -- so a turn extracting six facts
        # paid six sequential network round-trips where one would do. That
        # serialised latency, not token cost, is what put ingestion at 8.9s per
        # turn against RAG's 3.2s while using a third of the tokens.
        #
        # Only the embedding is hoisted. The rest of the loop must stay
        # sequential: override detection reads Hot RAM state that earlier
        # iterations have already written, so two updates naming the same entity
        # depend on running in order. Embeddings have no such dependency --
        # each is a function of its own fact_content alone.
        embeddings = await self._embedder.embed([u.fact_content for u in updates])
        if len(embeddings) != len(updates):
            # Silent misalignment would store facts against other facts' vectors
            # and corrupt every subsequent retrieval, so fail loudly instead.
            raise RuntimeError(
                f"Embedding provider returned {len(embeddings)} vectors "
                f"for {len(updates)} facts"
            )

        for update, embedding in zip(updates, embeddings):
            # The archive gets the same vector Hot RAM is about to store, which
            # is what lets Cold ROM answer by meaning rather than by shared
            # words. It is free: the embedding has already been paid for above.
            self._cold.append_event(update, self._current_turn, embedding)

            existing = self._hot.get_by_entity(update.entity_name)
            if update.overrides_entity:
                existing += self._hot.get_by_entity(update.overrides_entity)

            overrides = detect_override(update, existing)
            for old_record in overrides:
                suppressed = apply_suppression(old_record, self._decay_config.gamma)
                self._hot.update_weight(old_record.id, suppressed.weight)

            record = MemoryRecord(
                entity_name=update.entity_name,
                fact_content=update.fact_content,
                memory_type=update.memory_type,
                weight=1.0,
                consolidation_index=0,
                created_at_turn=self._current_turn,
                last_reinforced_turn=self._current_turn,
                related_entities=list(update.related_entities),
            )
            self._hot.upsert(record, embedding)

    def _run_decay(self) -> None:
        """Apply decay to all mutable records in Hot RAM."""
        mutable = self._hot.get_all_mutable()

        # The graph spans every record, anchors included: anchors do not decay
        # but they are still nodes, and a fact linked to one sits in a denser
        # neighbourhood for it. Skipped entirely at beta = 0, where clustering
        # has no effect and get_all() would be a table scan for nothing.
        graph = (
            build_entity_graph(self._hot.get_all())
            if self._decay_config.beta > 0.0
            else None
        )

        weight_updates = decay_all(
            mutable, self._current_turn, self._decay_config, graph
        )
        # decay_turn moves the clock forward with the weight, so the next pass
        # decays the stored value by one turn rather than by the whole elapsed
        # span again. See HotStorage.apply_weight_updates.
        self._hot.apply_weight_updates(weight_updates, decay_turn=self._current_turn)

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


