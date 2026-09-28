"""Phase 2A: MemoryOpenAI client wrapper interposition tests.

Covers the recall -> inject -> forward -> remember pipeline: injected system
preamble, post-response extraction, streaming reassembly (remember only after
the stream is exhausted), opt-out flags, and __getattr__ delegation.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.proxy import MemoryOpenAI
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


def _fact(text: str, entity: str = "pet", **kwargs) -> MemoryStateUpdate:
    return MemoryStateUpdate(
        entity_name=entity,
        fact_content=text,
        memory_type=MemoryType.CORE_ANCHOR,
        **kwargs,
    )


@pytest.fixture
def config(tmp_path) -> NMafcConfig:
    cfg = NMafcConfig()
    cfg.storage.hot_uri = str(tmp_path / "hot.lancedb")
    cfg.storage.cold_uri = str(tmp_path / "cold.db")
    cfg.storage.event_log_uri = str(tmp_path / "events.jsonl")
    cfg.storage.agent_id = "agent-a"
    cfg.storage.conversation_id = "conv-1"
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
    # Seed one fact so recall has something to inject.
    llm.add_response("", [_fact("Adopted a corgi named Miso.")])
    asyncio.run(mem.remember(
        [{"role": "user", "content": "I just adopted a corgi named Miso."}],
    ))
    yield mem
    mem.close()


class FakeAsyncCompletions:
    def __init__(self, responses) -> None:
        self.calls: list[dict] = []
        self.responses = responses

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._stream()
        return self.responses.pop(0)

    async def _stream(self):
        for text in ["Good ", "morning!"]:
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
            )


class FakeAsyncClient:
    def __init__(self, responses) -> None:
        self.chat = SimpleNamespace(completions=FakeAsyncCompletions(responses))


class FakeSyncCompletions:
    def __init__(self, responses) -> None:
        self.calls: list[dict] = []
        self.responses = responses

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeSyncClient:
    def __init__(self, responses) -> None:
        self.chat = SimpleNamespace(completions=FakeSyncCompletions(responses))


def _reply(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


# --- recall -> inject -> forward -------------------------------------

def test_injects_preamble_before_forward(memory, llm):
    fake = FakeAsyncClient(responses=[_reply("Sure thing!")])
    client = MemoryOpenAI(memory, client=fake)

    asyncio.run(client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What pet do I have?"}],
    ))

    sent = fake.chat.completions.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "Miso" in sent[0]["content"]
    assert sent[-1] == {"role": "user", "content": "What pet do I have?"}


def test_no_injection_when_memory_has_no_hits(config, llm):
    fresh = NeuromorphicMemory(
        llm_provider=llm,
        embedding_provider=MockEmbedder(),
        config=config,
    )
    try:
        fake = FakeAsyncClient(responses=[_reply("I don't know yet.")])
        client = MemoryOpenAI(fresh, client=fake)

        asyncio.run(client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Anything stored?"}],
        ))

        sent = fake.chat.completions.calls[0]["messages"]
        assert sent[0]["role"] == "user", "no preamble injected when recall is empty"
    finally:
        fresh.close()


# --- post-response remember -------------------------------------------

def test_remembers_reply_after_response(memory, llm):
    fake = FakeAsyncClient(responses=[_reply("A corgi named Miso!")])
    client = MemoryOpenAI(memory, client=fake)

    # Queue the extraction result the wrapper's remember() will trigger.
    llm.add_response("", [_fact("User asked about their pet.", entity="query_log")])

    asyncio.run(client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What pet do I have?"}],
    ))

    assert any("Extract-Only Mode" in p for p in llm.seen_system_prompts)
    assert llm.seen_system_prompts, "remember must have run post-response"


def test_remember_can_be_opted_out(memory, llm):
    fake = FakeAsyncClient(responses=[_reply("Hi")])
    client = MemoryOpenAI(memory, client=fake)

    asyncio.run(client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        memory_remember=False,
    ))

    prompts_before = 1  # fixture's seeding remember() already ran one extract
    assert len(llm.seen_system_prompts) == prompts_before, (
        "no extraction should run when opted out"
    )


def test_recall_can_be_opted_out(memory, llm):
    fake = FakeAsyncClient(responses=[_reply("Ok")])
    client = MemoryOpenAI(memory, client=fake)

    asyncio.run(client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What pet do I have?"}],
        memory_recall=False,
    ))

    sent = fake.chat.completions.calls[0]["messages"]
    assert sent[0]["role"] == "user", "no system preamble injected when recall off"


# --- streaming -------------------------------------------------------

def test_stream_forwards_chunks_and_remembers_after(memory, llm):
    fake = FakeAsyncClient(responses=[_reply("unused")])
    client = MemoryOpenAI(memory, client=fake)

    llm.add_response("", [_fact("Stream ended cleanly.", entity="stream_log")])

    async def run():
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello stream"}],
            stream=True,
        )
        got = [chunk.choices[0].delta.content async for chunk in stream]
        return "".join(got)

    assert asyncio.run(run()) == "Good morning!"
    # Extraction ran only after the full stream was consumed.
    assert any("Extract-Only Mode" in p for p in llm.seen_system_prompts)


def test_stream_injection_applied(memory, llm):
    fake = FakeAsyncClient(responses=[_reply("x")])
    client = MemoryOpenAI(memory, client=fake)

    async def run():
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What pet do I have?"}],
            stream=True,
        )
        async for _ in stream:
            pass

    asyncio.run(run())

    sent = fake.chat.completions.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "Miso" in sent[0]["content"]


# --- sync client -----------------------------------------------------

def test_sync_client_pipeline(config, llm):
    memory = NeuromorphicMemory(
        llm_provider=llm,
        embedding_provider=MockEmbedder(),
        config=config,
    )
    llm.add_response("", [_fact("Owns a red bike.")])
    asyncio.run(memory.remember(
        [{"role": "user", "content": "I own a red bike."}],
    ))

    fake = FakeSyncClient(responses=[_reply("Nice bike!")])
    client = MemoryOpenAI(SyncNeuromorphicMemory(memory), client=fake)

    llm.add_response("", [_fact("User likes bikes.", entity="pref")])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "I have a red bike?"}],
    )

    assert response.choices[0].message.content == "Nice bike!"
    sent = fake.chat.completions.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "bike" in sent[0]["content"]
    assert any("Extract-Only Mode" in p for p in llm.seen_system_prompts)
    memory.close()


# --- delegation ------------------------------------------------------

def test_attribute_delegation(memory):
    fake = SimpleNamespace(missing_thing="nope")
    client = MemoryOpenAI(memory, client=fake)
    # Delegation forwards; missing attrs still raise AttributeError.
    assert client.missing_thing == "nope"
    with pytest.raises(AttributeError):
        client.definitely_not_here
