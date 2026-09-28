from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from nmafc.engine.consolidation import MemoryConsolidator
from nmafc.engine.decay import build_entity_graph, decay_all
from nmafc.engine.linking import resolve_link_targets
from nmafc.engine.pruning import (
    create_suppression_event,
    detect_override,
    prune_cycle,
)
from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.integration.extractor import StateExtractor
from nmafc.integration.query_router import QueryRouter
from nmafc.schemas.memory import (
    MemoryHit,
    MemoryRecord,
    MemoryStateUpdate,
    RecallResult,
    RememberResult,
    UnifiedMemoryPayload,
)
from nmafc.storage.cold_base import ColdStorageBase
from nmafc.storage.config import NMafcConfig
from nmafc.storage.event_log import EventLog
from nmafc.storage.hot import HotStorage

# Seconds to wait on the embedding-dimension probe before falling back to the
# configured dimension. Bounded because the probe can block indefinitely.
EMBED_PROBE_TIMEOUT = float(os.environ.get("NMAFC_EMBED_PROBE_TIMEOUT", "30"))


def sanitize_entity_name(name: str) -> str:
    """Normalize an entity name for safe storage and consistent graph lookups.

    Lowercases, strips whitespace, replaces apostrophes and special chars
    with underscores, and collapses runs of underscores. Prevents SQL injection
    via entity names containing quotes (e.g. "caroline's friendship").
    """
    name = name.strip().lower()
    name = name.replace("'", "_").replace('"', "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


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
        self._event_log = EventLog(
            config.storage.event_log_uri,
            agent_id=config.storage.agent_id,
            conversation_id=config.storage.conversation_id,
        )

        # Resume the turn clock where the last session left it. Cold ROM is
        # append-only and the turn counter lives only in memory, so without this
        # a reopen restarts at 0: every existing record looks newborn, decay
        # resets, reinforcements stamp early turns onto old records, and the
        # archive rows ahead of the counter are judged "future" and hidden.
        # The event log records a turn on every WEIGHT_UPDATE pass, but RETRIEVAL
        # events and the event-less ingest path mean it is not a reliable high
        # water mark; the archive's own max turn is.
        self._current_turn: int = 0
        try:
            self._current_turn = max(0, self._cold.max_turn())
        except Exception:
            # A backend or archive that cannot answer should not break a reopen.
            pass

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

    @classmethod
    def from_env(
        cls, config_path: str | Path | None = None,
    ) -> NeuromorphicMemory:
        """Build a fully-configured instance from environment variables.

        Pass `config_path` to load defaults from a TOML file first; when None,
        only environment variables are read, with library defaults for
        anything unset.
        """
        config = NMafcConfig.from_env_or_toml(config_path)
        return cls.from_config(config=config)

    @classmethod
    def from_openai(cls) -> NeuromorphicMemory:
        """Convenience constructor wired to OpenAI with no config file needed.

        Uses `gpt-4o-mini` for extraction and `text-embedding-3-small` for
        embeddings; both read `OPENAI_API_KEY` from the environment.
        """
        config = NMafcConfig.from_env_or_toml(None)
        config.llm_provider_model = "openai/gpt-4o-mini"
        config.embedding_provider_model = "openai/text-embedding-3-small"
        return cls.from_config(config=config)

    @property
    def current_turn(self) -> int:
        return self._current_turn

    async def process_turn(
        self,
        user_msg: str,
        conversation_history: list[dict] | None = None,
        occurred_at: str | None = None,
    ) -> str:
        """Process a single conversation turn through the full neuromorphic pipeline.

        1. Increment turn counter
        2. Retrieve relevant context from Hot RAM (using Spreading Activation)
        3. Generate response + extract state updates via LLM
        4. Process each update: log to Cold ROM, detect overrides, upsert to Hot RAM
        5. Run decay on all mutable records
        6. Prune below-threshold records
        7. Trigger auto-consolidation if interval threshold reached

        `occurred_at` is when this turn happened, in whatever wording should be
        read back to the model ("15 May 2023"). Recorded before retrieval so
        that the turn being ingested can already be dated.

        Returns the assistant's response text.
        """
        self._current_turn += 1
        self._record_turn_date(occurred_at)
        self._record_turn_text(user_msg)
        history = conversation_history or []

        retrieved = await self._router.retrieve(
            user_msg, self._current_turn, event_logger=self._event_log
        )
        # `user_msg`, not just the records. With `hydrate_lines` set, the
        # source block keeps the N lines of each turn that best match the cue,
        # and without the cue that choice is made from the retrieved facts
        # alone -- which is to say, blind to what was actually asked.
        memory_context = self._router.format_context(retrieved, user_msg)

        response_text, payload = await self._extractor.extract(
            user_msg=user_msg,
            context=history,
            memory_context=memory_context if memory_context else None,
        )

        await self._process_updates(payload)

        self._run_decay()

        prune_cycle(
            self._hot, self._cold, self._decay_config.w_prune,
            self._current_turn, event_logger=self._event_log,
            config=self._decay_config,
        )

        auto_interval = getattr(self._decay_config, "auto_consolidate_turns", 5)
        if auto_interval > 0 and self._current_turn % auto_interval == 0:
            self._consolidator.consolidate(
                self._current_turn, event_logger=self._event_log,
            )

        return response_text

    def _resolve_tenant(
        self,
        agent_id: str | None,
        conversation_id: str | None,
    ) -> tuple[str, str]:
        """Confirm requested tenant scope matches the store this instance holds.

        Hot RAM, Cold ROM and the event log are all opened against one
        agent/conversation at construction, so accepting a different scope here
        would label results with a tenant they were not read from. Defaults to
        the configured scope; a caller wanting another tenant builds another
        instance (see `nmafc.web.deps.get_memory`).
        """
        bound_agent = self._config.storage.agent_id
        bound_conv = self._config.storage.conversation_id
        agent = agent_id if agent_id is not None else bound_agent
        conv = conversation_id if conversation_id is not None else bound_conv
        if (agent, conv) != (bound_agent, bound_conv):
            raise ValueError(
                "This memory instance is bound to agent_id="
                f"{bound_agent!r}, conversation_id={bound_conv!r}; it cannot "
                f"serve agent_id={agent!r}, conversation_id={conv!r}. "
                "Build a NeuromorphicMemory per tenant instead "
                "(see nmafc.web.deps.get_memory)."
            )
        return agent, conv

    async def recall(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        top_k: int | None = None,
        turn: int | None = None,
    ) -> RecallResult:
        """Retrieve relevant memories for a query without changing anything.

        1. Validate tenant scope against this instance's store.
        2. Rerank cached facts for relevance (read-only retrieval).
        3. Format a bounded context string plus the raw scored hits.

        The retrieval neither writes nor buffers reinforcement and emits no
        retrieval events: a recall never mutates the store it reads. Nothing is
        ingested, decayed, pruned, or consolidated here.
        """
        agent, conv = self._resolve_tenant(agent_id, conversation_id)
        at_turn = turn if turn is not None else self._current_turn

        records, provenance = await self._router.retrieve_readonly(
            query, at_turn, top_k=top_k,
        )
        context = self._router.format_context(records, query)

        hits = [
            MemoryHit(
                id=rec.id,
                entity_name=rec.entity_name,
                fact_content=rec.fact_content,
                memory_type=rec.memory_type,
                score=score,
                source=source,
                hop_distance=hops,
            )
            for rec, (score, source, hops) in zip(records, provenance)
        ]

        return RecallResult(
            query=query,
            context=context,
            hits=hits,
            token_estimate=max(1, len(context) // 4) if context else 0,
            turn=at_turn,
            agent_id=agent,
            conversation_id=conv,
        )

    def _split_transcript(
        self, messages: list[dict]
    ) -> tuple[str, list[dict]] | None:
        """Split a chat transcript into the turn to store and prior context.

        The most recent user message is the turn; everything before it is
        history passed to the extractor alongside it. Returns None when the
        transcript holds no user message, which callers treat as a no-op rather
        than an error: an empty, aborted, or assistant-only transcript carries
        nothing to remember and is not the caller's mistake.
        """
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                prior = list(messages[:idx])
                user_msg = str(messages[idx].get("content") or "")
                if not user_msg.strip():
                    return None
                return user_msg, prior
        return None

    async def remember(
        self,
        messages: list[dict],
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        occurred_at: str | None = None,
    ) -> RememberResult:
        """Store what a transcript said, without generating a reply.

        Runs the extractor in extract-only mode, then the same ingest path as
        a chat turn: log to Cold ROM, detect and suppress overrides, upsert to
        Hot RAM, decay, prune, and consolidate on schedule. No response text is
        produced -- answering is the caller's job -- and an unusable transcript
        returns an empty result rather than raising.
        """
        agent, conv = self._resolve_tenant(agent_id, conversation_id)
        split = self._split_transcript(messages)
        if split is None:
            return RememberResult(
                turn=self._current_turn,
                agent_id=agent,
                conversation_id=conv,
            )

        user_msg, history = split

        # Retrieve read-only: what a past turn said must not reinforce the
        # records already there before it is even stored.
        records, _ = await self._router.retrieve_readonly(
            user_msg, self._current_turn + 1,
        )
        memory_context = self._router.format_context(records, user_msg)

        _, payload = await self._extractor.extract(
            user_msg=user_msg,
            context=history,
            memory_context=memory_context or None,
            generate_response=False,
        )

        if not payload.updates:
            # Nothing worth storing: no turn is consumed, so a later remember
            # or chat turn is not stamped with a turn that never happened.
            return RememberResult(
                turn=self._current_turn,
                agent_id=agent,
                conversation_id=conv,
            )

        self._current_turn += 1
        self._record_turn_date(occurred_at)
        self._record_turn_text(user_msg)
        suppressed = await self._process_updates(payload)
        self._run_decay()
        prune_cycle(
            self._hot, self._cold, self._decay_config.w_prune,
            self._current_turn, event_logger=self._event_log,
            config=self._decay_config,
        )

        auto_interval = getattr(self._decay_config, "auto_consolidate_turns", 5)
        consolidated = False
        if auto_interval > 0 and self._current_turn % auto_interval == 0:
            self._consolidator.consolidate(
                self._current_turn, event_logger=self._event_log,
            )
            consolidated = True

        return RememberResult(
            turn=self._current_turn,
            updates_ingested=len(payload.updates),
            overrides_suppressed=suppressed,
            agent_id=agent,
            conversation_id=conv,
            consolidated=consolidated,
        )

    async def forget_entity(
        self,
        entity_name: str,
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> int:
        """Close every active record for one entity within this tenant's scope.

        Forgetting is temporal invalidation, not destruction: the active Hot RAM
        vectors are stamped invalid at the current turn and their history stays
        in Cold ROM, exactly as an override does. Returns how many records were
        invalidated (0 when nothing active matched).
        """
        agent, conv = self._resolve_tenant(agent_id, conversation_id)
        active = [r for r in self._hot.get_by_entity(entity_name) if r.invalid_at is None]
        if not active:
            return 0
        self._current_turn += 1
        self._hot.set_invalid_at_many([(r.id, self._current_turn) for r in active])
        return len(active)

    async def forget_record(
        self,
        record_id: str,
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        """Close a single record within this tenant's scope, history preserved.

        The temporal invalidation contract, applied to one record instead of a
        whole entity. Returns False when the record does not exist or is already
        invalidated, so callers can report a clean miss.
        """
        self._resolve_tenant(agent_id, conversation_id)
        record = self._hot.get_record(record_id)
        if record is None or record.invalid_at is not None:
            return False
        self._current_turn += 1
        self._hot.set_invalid_at_many([(record_id, self._current_turn)])
        return True

    async def ingest_updates(
        self,
        updates: list[MemoryStateUpdate],
        occurred_at: str | None = None,
    ) -> None:
        """Manually ingest memory updates without an LLM call.

        Useful for benchmarking and testing.

        `occurred_at` is when this turn happened, in whatever wording the caller
        wants read back ("15 May 2023"). Supplying it is what lets retrieved
        facts be dated in the prompt instead of stamped with a turn number.
        """
        self._current_turn += 1
        self._record_turn_date(occurred_at)
        await self._process_updates(UnifiedMemoryPayload(updates=updates))
        self._run_decay()
        prune_cycle(
            self._hot, self._cold, self._decay_config.w_prune,
            self._current_turn, event_logger=self._event_log,
            config=self._decay_config,
        )

        auto_interval = getattr(self._decay_config, "auto_consolidate_turns", 5)
        if auto_interval > 0 and self._current_turn % auto_interval == 0:
            self._consolidator.consolidate(
                self._current_turn, event_logger=self._event_log,
            )

    async def consolidate(self) -> int:
        """Manually invoke REM sleep consolidation pass over Hot RAM."""
        return self._consolidator.consolidate(
            self._current_turn, event_logger=self._event_log,
        )

    def _record_turn_date(self, occurred_at: str | None) -> None:
        """Note when the current turn happened, if the caller said.

        Tolerant of archives that predate the table and of Cold ROM backends
        that do not implement it, since neither is a reason to fail an ingest.
        The router's cache is dropped so a turn ingested after an earlier
        retrieval is dated rather than silently falling back to a turn number.
        """
        if not occurred_at:
            return
        recorder = getattr(self._cold, "record_turn_timestamp", None)
        if callable(recorder):
            recorder(self._current_turn, occurred_at)
            self._router.reset_turn_dates()

    def _record_turn_text(self, text: str) -> None:
        """Keep the turn's own wording alongside the facts drawn from it.

        Only when hydration is switched on. The text is small next to the store
        it sits in, but writing it unconditionally would grow every archive to
        pay for a feature most callers have not enabled.
        """
        if not self._decay_config.hydrate_top_k or not text:
            return
        recorder = getattr(self._cold, "record_turn_text", None)
        if callable(recorder):
            recorder(self._current_turn, text)

    async def _process_updates(self, payload: UnifiedMemoryPayload) -> int:
        """Process extracted memory updates: log, suppress overrides, upsert.

        Returns the number of superseded records suppressed by overrides, so
        callers can report how much was displaced rather than only how much was
        added.
        """
        updates = payload.updates
        if not updates:
            return 0

        suppressed = 0

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

        # Entity names a link is allowed to point at: everything already stored,
        # plus every fact in this same payload. The batch has to be included
        # because the extractor is told it may link to another entity in the
        # same tool call, and those are not in Hot RAM until the loop below
        # writes them -- resolving against storage alone would reject exactly
        # the links the prompt asks for.
        known_entities: list[str] = []
        if self._decay_config.resolve_link_targets:
            known_entities = [r.entity_name for r in self._hot.get_all()]
            known_entities += [u.entity_name for u in updates]

        for update, embedding in zip(updates, embeddings):
            update.entity_name = sanitize_entity_name(update.entity_name)
            if update.overrides_entity:
                update.overrides_entity = sanitize_entity_name(update.overrides_entity)
            update.related_entities = [sanitize_entity_name(r) for r in update.related_entities]

            # The archive gets the same vector Hot RAM is about to store, which
            # is what lets Cold ROM answer by meaning rather than by shared
            # words. It is free: the embedding has already been paid for above.
            self._cold.append_event(
                update, self._current_turn, embedding, valid_at=self._current_turn
            )

            existing = self._hot.get_by_entity(update.entity_name)
            if update.overrides_entity:
                existing += self._hot.get_by_entity(update.overrides_entity)

            overrides = detect_override(update, existing)
            if overrides:
                invalidations = [
                    (old_record.id, self._current_turn)
                    for old_record in overrides
                ]
                self._hot.set_invalid_at_many(invalidations)
                # The archive has to hear about it too. The router queries Cold
                # ROM on every retrieval, so a fact withdrawn from Hot RAM alone
                # is pulled straight back by the next keyword or vector hit and
                # reaches the prompt exactly as before. Measured over 101
                # belief-update questions, Hot-only invalidation left the stale
                # value in the prompt 64 times out of 101 -- one more than doing
                # nothing at all.
                self._cold.invalidate_facts([
                    (old_record.entity_name, old_record.fact_content,
                     self._current_turn)
                    for old_record in overrides
                ])
                suppressed += len(overrides)
                for old_record in overrides:
                    self._event_log.log(
                        create_suppression_event(
                            old_record, old_record.weight,
                            suppressed_by=update.entity_name,
                            turn=self._current_turn,
                        )
                    )

            related = list(update.related_entities)
            if self._decay_config.resolve_link_targets:
                # Own name excluded before matching, not after: a fact called
                # "caroline_adoption_goal" is its own closest match by name
                # overlap, so leaving it in would turn most unresolvable links
                # into self-references rather than dropping them.
                related = resolve_link_targets(
                    related,
                    (e for e in known_entities
                     if e.lower() != update.entity_name.lower()),
                    min_overlap=self._decay_config.link_match_threshold,
                )

            record = MemoryRecord(
                entity_name=update.entity_name,
                fact_content=update.fact_content,
                memory_type=update.memory_type,
                weight=1.0,
                consolidation_index=0,
                created_at_turn=self._current_turn,
                last_reinforced_turn=self._current_turn,
                related_entities=related,
                valid_at=self._current_turn,
                valid_at_text=(update.valid_at or None),
            )
            self._hot.upsert(record, embedding)

        return suppressed

    def _run_decay(self) -> None:
        """Apply decay to all mutable records in Hot RAM."""
        # Deferred reinforcements have to land first. Decay reads the weight
        # and the consolidation index that the buffer is holding, so decaying
        # ahead of a flush would compute against stale values and the flush
        # would then overwrite the result. Free when deferral is off, since
        # nothing is ever buffered.
        self._router.flush_reinforcements()

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
            mutable, self._current_turn, self._decay_config, graph,
            event_logger=self._event_log,
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

    def get_event_stats(self) -> dict:
        """Get statistics about the cognitive event log."""
        return {
            "total_events": self._event_log.count(),
            "by_type": self._event_log.count_by_type(),
        }

    def get_events(self, **kwargs) -> list:
        """Query cognitive events with flexible filters.

        Keyword args: turn_from, turn_to, event_types, entity_name,
        record_id, limit, offset.
        """
        return self._event_log.query(**kwargs)

    def get_event_timeline(self, limit: int = 100) -> list[dict]:
        """Aggregated event counts per turn, grouped by event type."""
        return self._event_log.get_timeline(limit=limit)

    def get_entity_events(self, entity_name: str, limit: int = 50) -> list:
        """All cognitive events for a specific entity."""
        return self._event_log.get_entity_history(entity_name, limit=limit)

    def maintain(self) -> bool:
        """Settle pending writes and compact the Hot RAM store.

        Meant to be called between conversations, never inside one: compaction
        costs seconds in exchange for making every later read faster, which only
        pays back over the reads that follow it. Returns whether the compaction
        itself ran -- see HotStorage.compact.
        """
        self._router.flush_reinforcements()
        return self._hot.compact()

    def close(self) -> None:
        """Close storage resources and connections."""
        # Anything still buffered is lost once the process ends, so it is
        # written now rather than silently dropped.
        self._router.flush_reinforcements()
        self._event_log.close()
        self._cold.close()

    async def __aenter__(self) -> NeuromorphicMemory:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()

    def __enter__(self) -> NeuromorphicMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SyncNeuromorphicMemory:
    """Synchronous wrapper around NeuromorphicMemory for non-async code.

    Wraps every async method with asyncio.run() so callers don't need
    an event loop. Suitable for scripts, notebooks, and CLI tools.
    """

    def __init__(self, memory: NeuromorphicMemory) -> None:
        self._memory = memory

    @classmethod
    def from_config(
        cls,
        config: NMafcConfig | None = None,
        config_path: str | Path = "configs/default.toml",
    ) -> SyncNeuromorphicMemory:
        mem = NeuromorphicMemory.from_config(config=config, config_path=config_path)
        return cls(mem)

    @classmethod
    def from_providers(
        cls,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        config: NMafcConfig | None = None,
        config_path: str | Path = "configs/default.toml",
    ) -> SyncNeuromorphicMemory:
        mem = NeuromorphicMemory(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            config=config,
            config_path=config_path,
        )
        return cls(mem)

    @property
    def current_turn(self) -> int:
        return self._memory.current_turn

    def process_turn_sync(
        self,
        user_msg: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        return asyncio.run(self._memory.process_turn(user_msg, conversation_history))

    def ingest_updates_sync(self, updates: list[MemoryStateUpdate]) -> None:
        asyncio.run(self._memory.ingest_updates(updates))

    def recall(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        top_k: int | None = None,
        turn: int | None = None,
    ) -> RecallResult:
        """Sync variant of `NeuromorphicMemory.recall`."""
        return asyncio.run(self._memory.recall(
            query, agent_id=agent_id, conversation_id=conversation_id,
            top_k=top_k, turn=turn,
        ))

    def remember(
        self,
        messages: list[dict],
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        occurred_at: str | None = None,
    ) -> RememberResult:
        """Sync variant of `NeuromorphicMemory.remember`."""
        return asyncio.run(self._memory.remember(
            messages, agent_id=agent_id, conversation_id=conversation_id,
            occurred_at=occurred_at,
        ))

    def forget_entity(
        self,
        entity_name: str,
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> int:
        """Sync variant of `NeuromorphicMemory.forget_entity`."""
        return asyncio.run(self._memory.forget_entity(
            entity_name, agent_id=agent_id, conversation_id=conversation_id,
        ))

    def forget_record(
        self,
        record_id: str,
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        """Sync variant of `NeuromorphicMemory.forget_record`."""
        return asyncio.run(self._memory.forget_record(
            record_id, agent_id=agent_id, conversation_id=conversation_id,
        ))

    def consolidate_sync(self) -> int:
        return asyncio.run(self._memory.consolidate())

    def rollback_sync(self, to_turn: int) -> int:
        return asyncio.run(self._memory.rollback(to_turn))

    def get_hot_stats(self) -> dict:
        return self._memory.get_hot_stats()

    def get_cold_stats(self) -> dict:
        return self._memory.get_cold_stats()

    def get_event_stats(self) -> dict:
        return self._memory.get_event_stats()

    def get_events(self, **kwargs) -> list:
        return self._memory.get_events(**kwargs)

    def get_event_timeline(self, limit: int = 100) -> list[dict]:
        return self._memory.get_event_timeline(limit=limit)

    def get_entity_events(self, entity_name: str, limit: int = 50) -> list:
        return self._memory.get_entity_events(entity_name, limit=limit)

    def maintain(self) -> bool:
        return self._memory.maintain()

    def close(self) -> None:
        self._memory.close()

    def __enter__(self) -> SyncNeuromorphicMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


