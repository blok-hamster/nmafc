"""Phase 1.1-1.5: recall() and remember() primitives.

Covers the read-only contract of recall (1.1), the extract-only ingest of
remember (1.2), the synchronous wrapper (1.3), the abort/partial-transcript
contract (1.4), and tenant scoping via args (1.5).
"""

from __future__ import annotations

import asyncio

import pytest

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, MemoryType
from nmafc.storage.config import NMafcConfig
from nmafc.wrapper import NeuromorphicMemory, SyncNeuromorphicMemory

EMBED_DIM = 8


class MockEmbedder(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * EMBED_DIM
            for i, ch in enumerate(text.encode("utf-8")[:EMBED_DIM]):
                vec[i % EMBED_DIM] += (ch % 31) / 31.0
            out.append(vec)
        return out


class MockLLMProvider(LLMProvider):
    """Queues canned responses; records the system prompt it was given."""

    def __init__(self) -> None:
        self._queue: list[tuple[str, list[MemoryStateUpdate]]] = []
        self.seen_system_prompts: list[str] = []

    def add_response(self, text: str, updates: list[MemoryStateUpdate]) -> None:
        self._queue.append((text, updates))

    async def chat_with_extraction(
        self, messages: list[dict], system_prompt: str
    ) -> tuple[str, list[MemoryStateUpdate]]:
        self.seen_system_prompts.append(system_prompt)
        if not self._queue:
            return "", []
        return self._queue.pop(0)


@pytest.fixture
def config(tmp_path) -> NMafcConfig:
    hot = tmp_path / "hot.lancedb"
    cold = tmp_path / "cold.db"
    event = tmp_path / "events.jsonl"
    cfg = NMafcConfig()
    cfg.storage.hot_uri = str(hot)
    cfg.storage.cold_uri = str(cold)
    cfg.storage.event_log_uri = str(event)
    cfg.storage.agent_id = "agent-a"
    cfg.storage.conversation_id = "conv-1"
    # Pin to the mock's width; NMAFC_EMBEDDING_DIM (if set) would otherwise
    # open the probe-skipping path and the table would be created at the real
    # model's dimension.
    cfg.storage.embedding_dim = EMBED_DIM
    return cfg


@pytest.fixture
def llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def memory(config, llm) -> NeuromorphicMemory:
    mem = NeuromorphicMemory(
        llm_provider=llm,
        embedding_provider=MockEmbedder(),
        config=config,
    )
    yield mem
    mem.close()


def _fact(text: str, entity: str = "pet", **kwargs) -> MemoryStateUpdate:
    return MemoryStateUpdate(
        entity_name=entity,
        fact_content=text,
        memory_type=MemoryType.CORE_ANCHOR,
        **kwargs,
    )


# --- 1.2 remember: extract-only ingest ---

def test_remember_ingests_to_hot_and_cold(memory, llm):
    llm.add_response("", [_fact("I adopted a corgi named Miso.")])

    result = asyncio.run(memory.remember(
        [{"role": "user", "content": "I just adopted a corgi named Miso."}],
    ))

    assert result.updates_ingested == 1
    assert result.turn >= 1

    hot = memory.get_hot_stats()
    assert hot["count"] == 1

    cold = memory.get_cold_stats()
    assert cold["total_events"] >= 1


def test_remember_produces_no_assistant_response(memory, llm):
    llm.add_response("Sure! I will remember that.", [_fact("likes tea")])

    # The provider text is discarded: remember must not answer.
    asyncio.run(memory.remember(
        [{"role": "user", "content": "I like tea."}],
    ))
    # Extract-only mode must have been requested.
    assert any("Extract-Only Mode" in p for p in llm.seen_system_prompts)


def test_remember_processes_overrides(memory, llm):
    llm.add_response("", [_fact("Takes lisinopril.", entity="bp_med")])
    asyncio.run(memory.remember(
        [{"role": "user", "content": "I take lisinopril."}],
    ))
    llm.add_response("", [_fact(
        "Takes losartan.", entity="bp_med_new",
        overrides_entity="bp_med",
    )])
    result = asyncio.run(memory.remember(
        [{"role": "user", "content": "Switched to losartan."}],
    ))

    assert result.overrides_suppressed == 1
    by_entity = {r.entity_name: r for r in memory._hot.get_all()}
    # Old record invalidated, not deleted; the new one is the sole valid state.
    assert by_entity["bp_med"].invalid_at is not None
    assert by_entity["bp_med_new"].invalid_at is None


def test_remember_processes_links(memory, llm):
    llm.add_response("", [
        _fact("Adopted Miso.", entity="corgi_adoption"),
        MemoryStateUpdate(
            entity_name="current_pet",
            fact_content="Current pet is Miso.",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            related_entities=["corgi_adoption"],
        ),
    ])
    asyncio.run(memory.remember([
        {"role": "user", "content": "I adopted a corgi named Miso."},
    ]))

    records = memory._hot.get_all()
    by_entity = {r.entity_name: r for r in records}
    # Link resolution may fuzzy-match; at minimum both facts are stored and
    # the link target is recorded against an existing entity.
    assert "corgi_adoption" in by_entity
    assert "current_pet" in by_entity
    assert by_entity["current_pet"].related_entities


# --- 1.4 abort / partial / empty transcript ---

def test_remember_empty_messages_is_noop(memory):
    before = memory.get_hot_stats()["count"]
    result = asyncio.run(memory.remember([]))
    assert result.updates_ingested == 0
    assert result.turn == memory.current_turn
    assert memory.get_hot_stats()["count"] == before


def test_remember_assistant_only_is_noop(memory, llm):
    before = memory.get_hot_stats()["count"]
    result = asyncio.run(memory.remember([
        {"role": "assistant", "content": "hello"},
    ]))
    assert result.updates_ingested == 0
    assert memory.get_hot_stats()["count"] == before
    # No provider call should be needed for a transcript with no user turn.
    assert llm.seen_system_prompts == []


def test_remember_blank_user_message_is_noop(memory):
    result = asyncio.run(memory.remember([
        {"role": "user", "content": "   "},
    ]))
    assert result.updates_ingested == 0
    assert result.turn == memory.current_turn


def test_remember_picks_last_user_message(memory, llm):
    llm.add_response("", [_fact("Prefers green tea.")])
    result = asyncio.run(memory.remember([
        {"role": "user", "content": "I drink coffee."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Actually I prefer green tea."},
    ]))
    assert result.updates_ingested == 1


# --- 1.1 recall: read-only contract ---

def test_recall_returns_bounded_string_and_hits(memory, llm):
    llm.add_response("", [_fact("Adopted a corgi named Miso.", entity="pet")])
    asyncio.run(memory.remember(
        [{"role": "user", "content": "I adopted a corgi named Miso."}],
    ))

    result = asyncio.run(memory.recall("what pet do I have?"))

    assert isinstance(result.context, str)
    assert result.hits, "expected at least one hit"
    hit = result.hits[0]
    assert hit.entity_name
    assert hit.source in {"hot_vector", "cold_semantic", "bfs_hot", "bfs_cold"}
    assert result.token_estimate == (len(result.context) // 4
                                     if result.context else 0)

    # Bounded: top_k must limit hits.
    bounded = asyncio.run(memory.recall("pet", top_k=1))
    assert len(bounded.hits) <= 1


def test_recall_writes_nothing(memory, llm):
    llm.add_response("", [_fact("Adopted a corgi named Miso.")])
    asyncio.run(memory.remember(
        [{"role": "user", "content": "I adopted a corgi named Miso."}],
    ))

    turn_before = memory.current_turn
    hot_before = memory.get_hot_stats()
    cold_before = memory.get_cold_stats()
    events_before = memory.get_event_stats()

    asyncio.run(memory.recall("what pet?"))
    asyncio.run(memory.recall("what pet?"))

    assert memory.current_turn == turn_before
    assert memory.get_hot_stats() == hot_before
    assert memory.get_cold_stats() == cold_before
    assert memory.get_event_stats() == events_before


def test_recall_emits_no_retrieval_events(memory, llm):
    llm.add_response("", [_fact("Adopted a corgi named Miso.")])
    asyncio.run(memory.remember(
        [{"role": "user", "content": "I adopted a corgi named Miso."}],
    ))
    events_before = memory.get_event_stats()["by_type"]
    asyncio.run(memory.recall("what pet?"))
    after = memory.get_event_stats()["by_type"]
    # Retrieval events (reinforcement side effects) must not appear from recall.
    assert after.get("RETRIEVAL", 0) == events_before.get("RETRIEVAL", 0)


# --- 1.5 tenant params ---

def test_recall_defaults_to_bound_tenant(memory):
    result = asyncio.run(memory.recall("anything"))
    assert result.agent_id == "agent-a"
    assert result.conversation_id == "conv-1"


def test_remember_defaults_to_bound_tenant(memory, llm):
    llm.add_response("", [_fact("x")])
    result = asyncio.run(memory.remember(
        [{"role": "user", "content": "x"}],
    ))
    assert result.agent_id == "agent-a"
    assert result.conversation_id == "conv-1"


def test_recall_rejects_mismatched_tenant(memory):
    with pytest.raises(ValueError, match="bound"):
        asyncio.run(memory.recall("x", agent_id="other-agent"))


def test_remember_rejects_mismatched_tenant(memory, llm):
    with pytest.raises(ValueError, match="bound"):
        asyncio.run(memory.remember(
            [{"role": "user", "content": "x"}],
            conversation_id="other-conv",
        ))


def test_recall_matching_tenant_args_ok(memory, llm):
    llm.add_response("", [_fact("y")])
    asyncio.run(memory.remember(
        [{"role": "user", "content": "y"}],
        agent_id="agent-a",
        conversation_id="conv-1",
    ))
    result = asyncio.run(memory.recall(
        "y", agent_id="agent-a", conversation_id="conv-1",
    ))
    assert result.agent_id == "agent-a"


# --- 1.6 from_env / from_openai ---

def test_from_env_or_toml_none_uses_env_and_defaults(monkeypatch):
    # No TOML involved: environment over library defaults only.
    monkeypatch.setenv("NMAFC_LLM_PROVIDER_MODEL", "ollama/llama3")
    monkeypatch.delenv("NMAFC_EMBEDDING_PROVIDER_MODEL", raising=False)
    monkeypatch.delenv("NMAFC_EMBEDDING_DIM", raising=False)

    cfg = NMafcConfig.from_env_or_toml(None)

    assert cfg.llm_provider_model == "ollama/llama3"
    assert cfg.embedding_provider_model.startswith("openai/")
    assert cfg.storage.agent_id == "default"


def test_from_openai_forces_openai_models(tmp_path, monkeypatch):
    seen = {}

    def fake_llm(model):
        seen["llm"] = model
        return MockLLMProvider()

    def fake_embed(model):
        seen["embed"] = model
        return MockEmbedder()

    import nmafc.integration.factory as factory
    monkeypatch.setattr(factory, "create_llm_provider", fake_llm)
    monkeypatch.setattr(factory, "create_embedding_provider", fake_embed)

    memory = NeuromorphicMemory.from_openai()
    memory.close()

    assert seen["llm"] == "openai/gpt-4o-mini"
    assert seen["embed"] == "openai/text-embedding-3-small"


# --- 1.3 sync variants ---

def test_sync_remember_and_recall(config, llm):
    llm.add_response("", [_fact("Owns a red bicycle.")])
    mem = SyncNeuromorphicMemory(NeuromorphicMemory(
        llm_provider=llm,
        embedding_provider=MockEmbedder(),
        config=config,
    ))
    try:
        stored = mem.remember(
            [{"role": "user", "content": "I own a red bicycle."}],
        )
        assert stored.updates_ingested == 1

        result = mem.recall("what bike?")
        assert isinstance(result.context, str)
        assert result.agent_id == "agent-a"
    finally:
        mem.close()


# --- 1.7 forget_entity: temporal invalidation with tombstoned archive ---

def test_forget_entity_invalidates_and_hides_from_later_recall(memory, llm):
    llm.add_response("", [_fact("Password is hunter2.", entity="creds")])
    asyncio.run(memory.remember([
        {"role": "user", "content": "My password is hunter2"},
    ]))

    before = asyncio.run(memory.recall("password"))
    assert "hunter2" in before.context

    forgotten = asyncio.run(memory.forget_entity("creds"))
    assert forgotten == 1

    remaining = asyncio.run(memory.recall("password"))
    assert "hunter2" not in remaining.context
    # The archive row still exists for history; the tombstone hides it from
    # active recall but not from the record itself.
    assert memory._hot.tombstoned_entities() == {"creds"}

    again = asyncio.run(memory.forget_entity("creds"))
    assert again == 0
