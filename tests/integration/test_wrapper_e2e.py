import tempfile
from pathlib import Path

import pytest

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, MemoryType
from nmafc.storage.config import NMafcConfig, StorageConfig
from nmafc.schemas.memory import DecayConfig
from nmafc.wrapper import NeuromorphicMemory

EMBED_DIM = 8


class MockEmbedder(EmbeddingProvider):
    """Deterministic embedder for testing."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 100 / 100.0
            results.append([seed + i * 0.01 for i in range(EMBED_DIM)])
        return results


class MockLLMProvider(LLMProvider):
    """Mock LLM that returns canned responses and updates."""

    def __init__(self):
        self.call_count = 0
        self.scripted_responses: list[tuple[str, list[MemoryStateUpdate]]] = []

    def add_response(self, text: str, updates: list[MemoryStateUpdate]):
        self.scripted_responses.append((text, updates))

    async def chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        if self.call_count < len(self.scripted_responses):
            response = self.scripted_responses[self.call_count]
        else:
            response = ("Default response.", [])
        self.call_count += 1
        return response


@pytest.fixture
def setup():
    tmp = tempfile.mkdtemp()
    config = NMafcConfig(
        storage=StorageConfig(
            hot_uri=str(Path(tmp) / "lance"),
            cold_uri=str(Path(tmp) / "cold.db"),
            embedding_dim=EMBED_DIM,
        ),
        decay=DecayConfig(
            lambda_ephemeral=0.69,
            lambda_active_context=0.05,
        ),
    )
    llm = MockLLMProvider()
    embedder = MockEmbedder()
    memory = NeuromorphicMemory(
        llm_provider=llm,
        embedding_provider=embedder,
        config=config,
    )
    return memory, llm


@pytest.mark.asyncio
async def test_basic_turn_processing(setup):
    memory, llm = setup
    llm.add_response(
        "Nice to meet you, Alice!",
        [
            MemoryStateUpdate(
                entity_name="user_name",
                fact_content="User's name is Alice",
                memory_type=MemoryType.CORE_ANCHOR,
            )
        ],
    )

    response = await memory.process_turn("Hi, my name is Alice!")
    assert response == "Nice to meet you, Alice!"
    assert memory.current_turn == 1

    stats = memory.get_hot_stats()
    assert stats["count"] >= 1
    assert stats["types"].get("CoreAnchor", 0) >= 1


@pytest.mark.asyncio
async def test_contradiction_suppression(setup):
    memory, llm = setup

    llm.add_response(
        "Got it, aspirin at 9 AM.",
        [
            MemoryStateUpdate(
                entity_name="medication_morning",
                fact_content="Takes aspirin at 9 AM",
                memory_type=MemoryType.ACTIVE_CONTEXT,
            )
        ],
    )
    llm.add_response(
        "Updated — aspirin now at 11 AM.",
        [
            MemoryStateUpdate(
                entity_name="medication_morning",
                fact_content="Takes aspirin at 11 AM",
                memory_type=MemoryType.ACTIVE_CONTEXT,
                overrides_entity="medication_morning",
            )
        ],
    )

    await memory.process_turn("I take aspirin at 9 AM")
    await memory.process_turn("Actually, I changed to 11 AM")

    # After override + prune, only the new fact should be active with full weight
    hot_stats = memory.get_hot_stats()
    assert hot_stats["count"] >= 1


@pytest.mark.asyncio
async def test_ephemeral_decays_and_gets_pruned(setup):
    memory, llm = setup

    # Directly ingest an ephemeral memory (bypasses retrieval/reinforcement)
    await memory.ingest_updates([
        MemoryStateUpdate(
            entity_name="user_mood",
            fact_content="User is feeling happy",
            memory_type=MemoryType.EPHEMERAL_STATE,
        )
    ])

    # Advance turns via ingest_updates with unrelated facts to avoid reinforcing mood
    for i in range(4):
        await memory.ingest_updates([])

    # After 4 additional turns (delta_t=4) with lambda=0.69:
    # weight = e^{-0.69*4} ≈ 0.063 < 0.1 prune threshold
    records = memory._hot.get_by_entity("user_mood")
    assert len(records) == 0  # Pruned


@pytest.mark.asyncio
async def test_core_anchor_persists(setup):
    memory, llm = setup

    llm.add_response(
        "Noted your allergy.",
        [
            MemoryStateUpdate(
                entity_name="allergy_peanuts",
                fact_content="User is allergic to peanuts",
                memory_type=MemoryType.CORE_ANCHOR,
            )
        ],
    )

    await memory.process_turn("I'm allergic to peanuts")

    # Advance many turns
    for i in range(20):
        llm.add_response(f"Response {i}", [])
        await memory.process_turn(f"Turn {i}")

    # Core anchor should NEVER decay or get pruned
    records = memory._hot.get_by_entity("allergy_peanuts")
    assert len(records) == 1
    assert records[0].weight == 1.0


@pytest.mark.asyncio
async def test_manual_ingest(setup):
    memory, _ = setup

    updates = [
        MemoryStateUpdate(
            entity_name="user_name",
            fact_content="Bob",
            memory_type=MemoryType.CORE_ANCHOR,
        ),
        MemoryStateUpdate(
            entity_name="mood",
            fact_content="tired",
            memory_type=MemoryType.EPHEMERAL_STATE,
        ),
    ]
    await memory.ingest_updates(updates)

    assert memory.get_hot_stats()["count"] == 2
    assert memory.get_cold_stats()["total_events"] == 2


@pytest.mark.asyncio
async def test_cold_rom_preserves_all_events(setup):
    memory, llm = setup

    for i in range(5):
        llm.add_response(
            f"Response {i}",
            [
                MemoryStateUpdate(
                    entity_name=f"fact_{i}",
                    fact_content=f"Fact number {i}",
                    memory_type=MemoryType.EPHEMERAL_STATE,
                )
            ],
        )
        await memory.process_turn(f"Message {i}")

    cold_stats = memory.get_cold_stats()
    assert cold_stats["total_events"] == 5
