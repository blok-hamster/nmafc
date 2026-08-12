"""Arm 2: NMAFC with decay disabled (SOTA stateful baseline).

Uses the real NeuromorphicMemory wrapper but with all decay/pruning disabled:
- lambda_active_context = 0.0 (no temporal decay)
- lambda_ephemeral = 0.0 (ephemeral also never decays)
- gamma = 1.0 (overrides don't suppress — old facts stay at full weight)
- w_prune = 0.0 (never evict anything)

This simulates a MemGPT/Zep-style "keep everything" approach using our own
architecture. This isolates the EXACT contribution of cognitive decay and
active pruning to the final accuracy/cost metrics.
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
from .base import BenchmarkArm

ANSWER_SYSTEM_PROMPT = """You are a conversational AI assistant with a memory system.
Relevant memories from past conversations are provided below.
Answer the user's question based ONLY on information from your memories.
If the answer is not in your memories, say "I don't know" or "This information is not available."
Be concise — answer in a few words or a short phrase when possible."""


class StatefulNoDecayArm(BenchmarkArm):
    """NMAFC with decay completely disabled — everything persists forever."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        storage_dir: str | None = None,
    ) -> None:
        super().__init__(name="stateful_nodecay")
        self._llm = llm_provider
        self._embedder = embedding_provider
        self._storage_dir = storage_dir or tempfile.mkdtemp(prefix="nmafc_bench_nodecay_")
        self._memory: NeuromorphicMemory = None  # type: ignore[assignment]
        self._init_memory()

    def _init_memory(self) -> None:
        """Initialize NeuromorphicMemory with decay disabled."""
        config = NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(Path(self._storage_dir) / "hot_lancedb"),
                cold_uri=str(Path(self._storage_dir) / "cold.db"),
            ),
            decay=DecayConfig(
                lambda_active_context=0.0,
                lambda_ephemeral=0.0,
                gamma=1.0,
                w_prune=0.0,
            ),
        )
        self._memory = NeuromorphicMemory(
            llm_provider=self._llm,
            embedding_provider=self._embedder,
            config=config,
        )

    async def ingest_conversation(self, turns: list[dict]) -> None:
        """Process conversation through the memory wrapper."""
        for turn in turns:
            if turn["role"] == "user":
                await self._memory.process_turn(
                    user_msg=turn["content"],
                    conversation_history=[turn],
                )

    async def answer_question(self, question: str) -> ArmResponse:
        """Answer using neuromorphic retrieval (without decay)."""
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

    def update_storage_metrics(self) -> None:
        if self._memory:
            stats = self._memory.get_hot_stats()
            self.metrics.hot_storage_records = stats.get("count", 0)
            cold_stats = self._memory.get_cold_stats()
            self.metrics.cold_storage_events = cold_stats.get("total_events", 0)
