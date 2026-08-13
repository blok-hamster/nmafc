"""Arm 2: Conventional RAG over the conversation transcript.

This is the standard industry baseline that NMAFC is being measured against:
chunk the transcript, embed the chunks, retrieve top-k by cosine similarity,
stuff them into the prompt. Nothing else.

Deliberately absent, so the comparison isolates what NMAFC actually adds:
  - no cognitive decay (chunks never lose weight)
  - no spaced repetition / LTP (retrieval does not mutate the store)
  - no override detection or pruning (contradictions coexist forever)
  - no spreading activation (no graph traversal past the vector hit)
  - no cold-storage fallback, no similarity threshold
  - no LLM calls during ingestion (chunking is mechanical, not extractive)

That last point is the key asymmetry to keep in mind when reading results:
RAG ingests a conversation for free, while the memory arms each spend an
LLM call per turn on extraction. RAG's advantage is ingestion cost; the
question is whether it holds up on retrieval accuracy.

Storage is the same LanceDB HotStorage the other arms use, so retrieval
differences come from strategy rather than from a different vector index.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import MemoryRecord, MemoryType
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage

from ..evaluation.metrics import ArmResponse
from .base import BenchmarkArm, SHORT_ANSWER_RULES

ANSWER_SYSTEM_PROMPT = """You are a conversational AI assistant with retrieval over past conversations.
Excerpts from the conversation history are provided below, ranked by relevance.
Answer the user's question based ONLY on information from those excerpts.
If the answer is not in the excerpts, say "I don't know" or "This information is not available."
Be concise — answer in a few words or a short phrase when possible.""" + SHORT_ANSWER_RULES

# Sliding window over turns. Overlap keeps a fact and its follow-up in at
# least one common chunk, so an answer split across two turns stays retrievable.
CHUNK_TURNS = int(os.environ.get("NMAFC_RAG_CHUNK_TURNS", "4"))
CHUNK_STRIDE = int(os.environ.get("NMAFC_RAG_CHUNK_STRIDE", "2"))
TOP_K = int(os.environ.get("NMAFC_RAG_TOP_K", "10"))


class RagArm(BenchmarkArm):
    """Conventional chunk-embed-retrieve RAG baseline."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        storage_dir: str | None = None,
        chunk_turns: int = CHUNK_TURNS,
        chunk_stride: int = CHUNK_STRIDE,
        top_k: int = TOP_K,
    ) -> None:
        super().__init__(name="rag")
        self._llm = llm_provider
        self._embedder = embedding_provider
        self._storage_dir = storage_dir or tempfile.mkdtemp(prefix="nmafc_bench_rag_")
        self._chunk_turns = chunk_turns
        self._chunk_stride = chunk_stride
        self._top_k = top_k
        self._embedding_dim: int | None = None
        self._chunk_count = 0
        self._store: HotStorage = None  # type: ignore[assignment]
        self._init_store()

    def _init_store(self) -> None:
        config = StorageConfig(
            hot_uri=str(Path(self._storage_dir) / "hot_lancedb"),
            cold_uri=str(Path(self._storage_dir) / "cold.db"),
            agent_id="bench_rag",
            conversation_id="default",
        )
        if self._embedding_dim is not None:
            config.embedding_dim = self._embedding_dim
        self._store = HotStorage(config)

    @staticmethod
    def _render_turn(turn: dict) -> str:
        """Render one turn as 'Speaker: text', preferring the real speaker name."""
        speaker = turn.get("speaker") or turn.get("role", "unknown")
        return f"{speaker}: {turn['content']}"

    def _chunk(self, turns: list[dict]) -> list[str]:
        """Sliding window of `chunk_turns` turns advancing by `chunk_stride`.

        Every chunk is stamped with the session date of its first turn. A
        retrieved chunk is read in isolation, so without the stamp there is
        nothing in context to answer "when did this happen?" — and the stamp
        also gives the embedding some temporal signal to match against.
        """
        lines = [self._render_turn(t) for t in turns]
        dates = [t.get("date") or "" for t in turns]
        if not lines:
            return []

        def stamp(start: int, window: list[str]) -> str:
            date = dates[start] if start < len(dates) else ""
            header = f"[Session — {date}]\n" if date else ""
            return header + "\n".join(window)

        if len(lines) <= self._chunk_turns:
            return [stamp(0, lines)]

        chunks = []
        for start in range(0, len(lines), self._chunk_stride):
            window = lines[start : start + self._chunk_turns]
            if not window:
                break
            chunks.append(stamp(start, window))
            if start + self._chunk_turns >= len(lines):
                break  # final window already reached the end
        return chunks

    async def ingest_conversation(self, turns: list[dict]) -> None:
        """Chunk, embed, and store. No LLM call — this is the whole point."""
        chunks = self._chunk(turns)
        if not chunks:
            return

        embeddings = await self._embedder.embed(chunks)

        # LanceDB fixes vector width at table creation, and the benchmark's
        # embedder may not match StorageConfig's default. Rebuild once we know.
        if self._embedding_dim is None and embeddings:
            self._embedding_dim = len(embeddings[0])
            shutil.rmtree(self._storage_dir, ignore_errors=True)
            Path(self._storage_dir).mkdir(parents=True, exist_ok=True)
            self._init_store()

        for chunk, embedding in zip(chunks, embeddings):
            record = MemoryRecord(
                entity_name=f"chunk_{self._chunk_count:05d}",
                fact_content=chunk,
                # CoreAnchor purely so nothing in the stack can decay it.
                # RAG has no notion of memory type; this is a static store.
                memory_type=MemoryType.CORE_ANCHOR,
                weight=1.0,
                consolidation_index=0,
                created_at_turn=0,
                last_reinforced_turn=0,
            )
            self._store.upsert(record, embedding)
            self._chunk_count += 1

    async def answer_question(self, question: str) -> ArmResponse:
        """Top-k cosine retrieval, then answer. No reranking, no threshold."""
        start = time.perf_counter()

        query_vec = await self._embedder.embed_single(question)
        results = self._store.search(query_vec, top_k=self._top_k)

        context = "\n\n---\n\n".join(r.record.fact_content for r in results)
        context_tokens = len(context) // 4

        system = ANSWER_SYSTEM_PROMPT
        if context:
            system += f"\n\n=== RETRIEVED EXCERPTS ===\n{context}\n=== END EXCERPTS ==="

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
        """Drop the index and start clean."""
        shutil.rmtree(self._storage_dir, ignore_errors=True)
        Path(self._storage_dir).mkdir(parents=True, exist_ok=True)
        self._chunk_count = 0
        self._init_store()

    def update_storage_metrics(self) -> None:
        if self._store:
            self.metrics.hot_storage_records = self._store.count()
            # RAG keeps no append-only event log; there is nothing to replay.
            self.metrics.cold_storage_events = 0
