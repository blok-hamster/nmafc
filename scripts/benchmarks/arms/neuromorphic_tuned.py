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
from pathlib import Path

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import DecayConfig
from nmafc.storage.config import NMafcConfig, StorageConfig
from nmafc.wrapper import NeuromorphicMemory

from ..evaluation.metrics import ArmResponse
from .base import BenchmarkArm, SHORT_ANSWER_RULES, build_exchanges

# Retention horizon ~= ln(1/w_prune) / lambda = ln(10) / 0.005 ~= 460 turns.
LAMBDA_ACTIVE_CONTEXT_TUNED = 0.005

ANSWER_SYSTEM_PROMPT = """You are a conversational AI assistant with a neuromorphic memory system.
Relevant memories from past conversations are provided below, ranked by salience.
Answer the user's question based ONLY on information from your memories.
If the answer is not in your memories, say "I don't know" or "This information is not available."
Be concise — answer in a few words or a short phrase when possible.""" + SHORT_ANSWER_RULES


class NeuromorphicTunedArm(BenchmarkArm):
    """NMAFC with decay active but its horizon sized to the conversation."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        storage_dir: str | None = None,
    ) -> None:
        super().__init__(name="neuromorphic_tuned")
        self._llm = llm_provider
        self._embedder = embedding_provider
        self._storage_dir = storage_dir or tempfile.mkdtemp(
            prefix="nmafc_bench_neurotuned_"
        )
        self._memory: NeuromorphicMemory = None  # type: ignore[assignment]
        self._init_memory()

    def _init_memory(self) -> None:
        """Initialize NeuromorphicMemory with a lengthened ActiveContext horizon.

        Every field other than lambda_active_context is left at its DecayConfig
        default, so this arm and the `neuromorphic` arm differ by exactly one
        number and any gap between them is attributable to that number alone.
        """
        config = NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(Path(self._storage_dir) / "hot_lancedb"),
                cold_uri=str(Path(self._storage_dir) / "cold.db"),
            ),
            decay=DecayConfig(
                lambda_active_context=LAMBDA_ACTIVE_CONTEXT_TUNED,
            ),
        )
        self._memory = NeuromorphicMemory(
            llm_provider=self._llm,
            embedding_provider=self._embedder,
            config=config,
        )

    async def ingest_conversation(self, turns: list[dict]) -> None:
        """Process conversation through the full neuromorphic pipeline."""
        for exchange in build_exchanges(turns):
            await self._memory.process_turn(user_msg=exchange)

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
            system += f"\n\n=== RELEVANT MEMORIES ===\n{memory_context}\n=== END MEMORIES ==="

        response_text, _ = await self._llm.chat_with_extraction(
            messages=[{"role": "user", "content": question}],
            system_prompt=system,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        prompt_tokens = (len(system) + len(question)) // 4
        completion_tokens = len(response_text) // 4

        response = ArmResponse(
            answer=response_text.strip(),
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
