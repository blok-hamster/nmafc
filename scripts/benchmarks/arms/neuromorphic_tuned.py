"""Arm 4: Neuromorphic memory with a decay horizon sized to the conversation.

Identical to the `neuromorphic` arm in every respect but one:
`lambda_active_context` is lowered from 0.05 to 0.005. Ephemeral decay,
override suppression and the prune threshold are untouched, so this arm still
forgets small talk and still suppresses contradicted facts -- it simply does not
forget mid-term context before the conversation is over.

Why the default value fails on retrospective QA
-----------------------------------------------
A record is evicted once its weight falls to `w_prune`, so an ActiveContext
memory that is never re-retrieved survives

    exp(-lambda * dt) <= w_prune  ->  dt >= ln(1 / w_prune) / lambda

turns. At the published defaults that is ln(10) / 0.05 ~= 46 turns. LoCoMo
conversations run 186-345 exchanges and ask every question only after the last
one, so ActiveContext records are guaranteed to be gone before the first
question is asked. The reinforcement half of the design (LTP resetting weight on
retrieval) never fires either, because nothing is retrieved mid-conversation.
Decay therefore runs unopposed for the entire ingestion.

Choosing the value
------------------
lambda is set so the retention horizon exceeds the longest conversation in the
corpus rather than by tuning against a score:

    longest conversation        345 exchanges
    required                    ln(10) / lambda > 345  ->  lambda < 0.0067
    chosen                      0.005  ->  horizon ln(10) / 0.005 ~= 460 turns

This is the general principle that the memory horizon should outlast the
interaction horizon. It is not fitted to LoCoMo's answers; only to its length,
which is a property of the deployment, not of the test set. Deployments with
longer-running conversations should scale lambda down further.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import DecayConfig
from nmafc.storage.config import NMafcConfig, StorageConfig
from nmafc.wrapper import NeuromorphicMemory

from ..evaluation.metrics import ArmResponse
from .base import BenchmarkArm, SHORT_ANSWER_RULES, build_exchanges, strip_answer

# Retention horizon ~= ln(1/w_prune) / lambda = ln(10) / 0.005 ~= 460 turns.
LAMBDA_ACTIVE_CONTEXT_TUNED = 0.005

ANSWER_SYSTEM_PROMPT = """You have a knowledge graph of facts from past conversations, shown in <FACTS> tags.
Answer the question using these facts. Combine and reason across multiple facts when needed.
Always prefer giving an answer over refusing — if the facts support a reasonable inference, state it directly.""" + SHORT_ANSWER_RULES


class NeuromorphicTunedArm(BenchmarkArm):
    """NMAFC with decay active but its horizon sized to the conversation."""

    supports_ingest_resume = True

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        storage_dir: str | None = None,
        decay_overrides: dict | None = None,
    ) -> None:
        super().__init__(name="neuromorphic_tuned")
        self._llm = llm_provider
        self._embedder = embedding_provider
        self._storage_dir = storage_dir or tempfile.mkdtemp(
            prefix="nmafc_bench_neurotuned_"
        )
        self._decay_overrides = decay_overrides or {}
        self._memory: NeuromorphicMemory = None  # type: ignore[assignment]
        self._init_memory()

    def _init_memory(self) -> None:
        """Initialize NeuromorphicMemory with a lengthened ActiveContext horizon.

        Every field other than lambda_active_context is left at its DecayConfig
        default, so this arm and the `neuromorphic` arm differ by exactly one
        number and any gap between them is attributable to that number alone.

        Run-level settings (max_hops, beta) are the exception, and only because
        the runner passes the same `decay_overrides` to both arms: they are
        properties of the run, not of this arm. lambda_active_context is applied
        last so a run-level override can never silently turn this arm back into
        the untuned one.
        """
        decay_overrides: dict = {
            **self._decay_overrides,
            "lambda_active_context": LAMBDA_ACTIVE_CONTEXT_TUNED,
        }

        config = NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(Path(self._storage_dir) / "hot_lancedb"),
                cold_uri=str(Path(self._storage_dir) / "cold.db"),
            ),
            decay=DecayConfig(**decay_overrides),
        )
        self._memory = NeuromorphicMemory(
            llm_provider=self._llm,
            embedding_provider=self._embedder,
            config=config,
        )

    def prepare_store(
        self, store_dir: str, conversation_id: str, fingerprint: str
    ) -> int:
        """Reopen a matching half-built store, or start this conversation over.

        Restoring `_current_turn` is not optional. It lives in memory and starts
        at 0, so reopening a store without it would leave records stamped
        `created_at_turn=168` in a system that believes no turns have happened:
        every decay and reinforcement calculation afterwards would be computed
        against a clock running 168 turns behind the data.
        """
        from scripts.benchmarks import ingest_checkpoint

        target = Path(store_dir)
        state = ingest_checkpoint.read(target, conversation_id, fingerprint)
        if state is None:
            self._storage_dir = str(target)
            self.reset()
            return 0

        if self._memory:
            self._memory.close()
        self._storage_dir = str(target)
        self._init_memory()
        self._memory._current_turn = state.turn
        return state.exchanges_done

    async def ingest_conversation(
        self,
        turns: list[dict],
        start_at: int = 0,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Process conversation through the full neuromorphic pipeline.

        Skipping the first `start_at` exchanges is safe because
        `build_exchanges` is a pure function of `turns`: the same history always
        yields the same list in the same order, so index N here is the same
        exchange index N was in the run that was interrupted.
        """
        for index, exchange in enumerate(build_exchanges(turns)):
            if index < start_at:
                continue
            await self._memory.process_turn(user_msg=exchange)
            if on_progress is not None:
                on_progress(index + 1, self._memory.current_turn)

    async def answer_question(self, question: str) -> ArmResponse:
        """Answer using neuromorphic retrieval with the lengthened horizon."""
        start = time.perf_counter()

        retrieved = await self._memory._router.retrieve(
            question, self._memory.current_turn + 1
        )
        memory_context = self._memory._router.format_context(retrieved)
        context_tokens = len(memory_context) // 4

        system = ANSWER_SYSTEM_PROMPT
        if memory_context:
            system += f"\n\n{memory_context}"

        response_text, _ = await self._llm.chat_with_extraction(
            messages=[{"role": "user", "content": question}],
            system_prompt=system,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        prompt_tokens = (len(system) + len(question)) // 4
        completion_tokens = len(response_text) // 4

        response = ArmResponse(
            answer=strip_answer(response_text),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_tokens=context_tokens,
        )
        self.metrics.record(response)
        return response

    def reset(self) -> None:
        """Reinitialize memory from scratch."""
        if self._memory:
            self._memory.close()
        import shutil
        shutil.rmtree(self._storage_dir, ignore_errors=True)
        Path(self._storage_dir).mkdir(parents=True, exist_ok=True)
        self._init_memory()
