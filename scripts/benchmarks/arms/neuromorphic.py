"""Arm 3: Full Neuromorphic Memory (the proposed system).

Uses the real NeuromorphicMemory wrapper with production configuration:
- Cognitive decay (Ebbinghaus-inspired exponential)
- Spaced repetition (LTP reinforcement on retrieval)
- Active synaptic pruning (override detection + gamma suppression)
- Typed decay rates (CoreAnchor=0, ActiveContext=0.05, Ephemeral=0.69)

This is the system we're proposing for TMLR publication.
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
from .base import BenchmarkArm, SHORT_ANSWER_RULES, build_exchanges

ANSWER_SYSTEM_PROMPT = """You are a conversational AI assistant with a neuromorphic memory system.
Relevant memories from past conversations are provided below, ranked by salience.
Answer the user's question based ONLY on information from your memories.
If the answer is not in your memories, say "I don't know" or "This information is not available."
Be concise — answer in a few words or a short phrase when possible.""" + SHORT_ANSWER_RULES


class NeuromorphicArm(BenchmarkArm):
    """Full NMAFC with decay, reinforcement, and pruning active."""

    supports_ingest_resume = True

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        config: NMafcConfig | None = None,
        storage_dir: str | None = None,
        decay_overrides: dict | None = None,
    ) -> None:
        super().__init__(name="neuromorphic")
        self._llm = llm_provider
        self._embedder = embedding_provider
        self._storage_dir = storage_dir or tempfile.mkdtemp(prefix="nmafc_bench_neuro_")
        self._decay_overrides = decay_overrides or {}
        self._config = config
        self._memory: NeuromorphicMemory = None  # type: ignore[assignment]
        self._init_memory()

    def _init_memory(self) -> None:
        """Initialize NeuromorphicMemory with production config.

        `decay_overrides` carries run-level DecayConfig settings the runner
        applies identically to every neuromorphic arm (max_hops, beta). They are
        properties of the run, not of this arm, which is why they arrive as a
        dict rather than as a growing list of keyword arguments. Anything absent
        keeps its DecayConfig default.
        """
        if self._config is None:
            decay = DecayConfig(**self._decay_overrides)
            self._config = NMafcConfig(
                storage=StorageConfig(
                    hot_uri=str(Path(self._storage_dir) / "hot_lancedb"),
                    cold_uri=str(Path(self._storage_dir) / "cold.db"),
                ),
                decay=decay,
            )
        else:
            self._config.storage.hot_uri = str(Path(self._storage_dir) / "hot_lancedb")
            self._config.storage.cold_uri = str(Path(self._storage_dir) / "cold.db")

        self._memory = NeuromorphicMemory(
            llm_provider=self._llm,
            embedding_provider=self._embedder,
            config=self._config,
        )

    def prepare_store(
        self, store_dir: str, conversation_id: str, fingerprint: str
    ) -> int:
        """Reopen a matching half-built store, or start this conversation over.

        See `NeuromorphicTunedArm.prepare_store` for why `_current_turn` has to
        be restored along with the data.
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
        """Process conversation through the full neuromorphic pipeline."""
        for index, exchange in enumerate(build_exchanges(turns)):
            if index < start_at:
                continue
            await self._memory.process_turn(user_msg=exchange)
            if on_progress is not None:
                on_progress(index + 1, self._memory.current_turn)

    async def answer_question(self, question: str) -> ArmResponse:
        """Answer using neuromorphic retrieval with full decay/pruning."""
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
