"""Ingestion must embed a turn's facts in one request, correctly aligned.

_process_updates() previously called embed_single() inside its loop, which is
embed([one_text]) -- one sequential HTTPS round-trip per extracted fact. Token
cost was already a third of the RAG arm's while wall-clock was nearly three
times worse, and this loop is where that went.

The batch is only safe if the returned vectors stay paired with the facts that
produced them. A silent misalignment stores each fact against another fact's
vector, which does not fail anywhere: it just makes every later retrieval
subtly wrong. These tests pin the call count and the pairing together, because
the optimisation is worthless without the second guarantee.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import DecayConfig, MemoryStateUpdate, MemoryType
from nmafc.storage.config import NMafcConfig, StorageConfig
from nmafc.wrapper import NeuromorphicMemory

EMBED_DIM = 8


class CountingEmbedder(EmbeddingProvider):
    """Records every call, and gives each distinct text a distinguishable vector."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def vector_for(self, text: str) -> list[float]:
        seed = (sum(ord(c) for c in text) % 97) / 100.0
        return [seed + i * 0.001 for i in range(EMBED_DIM)]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector_for(text) for text in texts]


class DroppingEmbedder(CountingEmbedder):
    """A provider that silently returns fewer vectors than it was given."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = await super().embed(texts)
        return vectors[:-1] if len(vectors) > 1 else vectors


class StubLLM(LLMProvider):
    async def chat_with_extraction(
        self, messages: list[dict], system_prompt: str
    ) -> tuple[str, list[MemoryStateUpdate]]:
        return ("", [])


def build_memory(embedder: EmbeddingProvider) -> NeuromorphicMemory:
    tmp = tempfile.mkdtemp()
    config = NMafcConfig(
        storage=StorageConfig(
            hot_uri=str(Path(tmp) / "lance"),
            cold_uri=str(Path(tmp) / "cold.db"),
            embedding_dim=EMBED_DIM,
        ),
        decay=DecayConfig(),
    )
    return NeuromorphicMemory(
        llm_provider=StubLLM(), embedding_provider=embedder, config=config
    )


def make_updates(count: int) -> list[MemoryStateUpdate]:
    return [
        MemoryStateUpdate(
            entity_name=f"entity_{i}",
            fact_content=f"Distinct fact number {i}",
            memory_type=MemoryType.ACTIVE_CONTEXT,
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_one_request_per_turn_not_one_per_fact():
    embedder = CountingEmbedder()
    memory = build_memory(embedder)
    updates = make_updates(6)

    await memory.ingest_updates(updates)

    assert len(embedder.calls) == 1
    assert embedder.calls[0] == [u.fact_content for u in updates]
    memory.close()


@pytest.mark.asyncio
async def test_each_fact_keeps_its_own_vector():
    """Searching with a fact's own vector must return that fact, not a neighbour."""
    embedder = CountingEmbedder()
    memory = build_memory(embedder)
    updates = make_updates(5)

    await memory.ingest_updates(updates)

    for update in updates:
        hits = memory._hot.search(embedder.vector_for(update.fact_content), top_k=1)
        assert hits, f"nothing retrieved for {update.entity_name}"
        assert hits[0].record.fact_content == update.fact_content
        assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    memory.close()


@pytest.mark.asyncio
async def test_short_response_raises_rather_than_misaligning():
    memory = build_memory(DroppingEmbedder())

    with pytest.raises(RuntimeError, match="vectors"):
        await memory.ingest_updates(make_updates(4))
    memory.close()


@pytest.mark.asyncio
async def test_empty_turn_calls_no_provider():
    embedder = CountingEmbedder()
    memory = build_memory(embedder)

    await memory.ingest_updates([])

    assert embedder.calls == []
    memory.close()
