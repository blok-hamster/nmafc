"""Phase 2A.2: HTTP proxy /v1/chat/completions.

Exercises the FastAPI endpoint with a stubbed upstream: recall->inject into
the forwarded body, tenant scoping via body params, post-response remember,
and SSE streaming. The OpenAI-shaped client contract is what the exit
criterion ("OpenAI(base_url=...) passes the same test") reduces to offline.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import openai.types.chat as oc
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, MemoryType
from nmafc.web.app import create_app
from nmafc.web.deps import get_memory

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


def make_chat_completion(content: str = "Hello!"):
    return oc.ChatCompletion(
        id="chatcmpl-test",
        choices=[
            oc.chat_completion.Choice(
                index=0,
                message=oc.ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
                logprobs=None,
            )
        ],
        created=1720000000,
        model="gpt-4o-mini",
        object="chat.completion",
    )


def make_chunk(text: str):
    return oc.chat_completion_chunk.ChatCompletionChunk(
        id="chatcmpl-test",
        choices=[
            oc.chat_completion_chunk.Choice(
                index=0,
                delta=oc.chat_completion_chunk.ChoiceDelta(
                    role="assistant", content=text
                ),
                finish_reason=None,
                logprobs=None,
            )
        ],
        created=1720000000,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )


class FakeAsyncCompletions:
    def __init__(self, replies) -> None:
        self.calls: list[dict] = []
        self.replies = replies

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._stream()
        return self.replies.pop(0)

    async def _stream(self):
        for text in ["Good ", "morning!"]:
            yield make_chunk(text)


class FakeAsyncClient:
    def __init__(self, replies) -> None:
        self.chat = SimpleNamespace(completions=FakeAsyncCompletions(replies))


@pytest_asyncio.fixture
def app_env(tmp_path, monkeypatch):
    """Configured app: temp storage, mocked providers, stubbed upstream."""
    # The toml value would be overridden by the shell's NMAFC_EMBEDDING_DIM;
    # pre-set storage URIs + dim so create_app's load_dotenv() does not leak
    # the repo .env's ./data paths or dim into these tests.
    monkeypatch.setenv("NMAFC_EMBEDDING_DIM", str(EMBED_DIM))
    monkeypatch.setenv("NMAFC_HOT_URI", str(tmp_path / "hot.lancedb"))
    monkeypatch.setenv("NMAFC_COLD_URI", str(tmp_path / "cold.db"))
    monkeypatch.setenv("NMAFC_EVENT_LOG_URI", str(tmp_path / "events.jsonl"))
    cfg = tmp_path / "custom.toml"
    cfg.write_text(
        f'[storage]\n'
        f'hot_uri = "{tmp_path}/hot.lancedb"\n'
        f'cold_uri = "{tmp_path}/cold.db"\n'
        f'event_log_uri = "{tmp_path}/events.jsonl"\n'
        f'embedding_dim = {EMBED_DIM}\n'
        f'agent_id = "agent-a"\n'
        f'conversation_id = "conv-1"\n'
        f'[embedding]\nprovider_model = "mock/text-embedding-3-small"\n'
        f'[llm]\nprovider_model = "mock/gpt-4o-mini"\n'
    )

    import nmafc.integration.factory as factory

    llm = MockLLMProvider()
    monkeypatch.setattr(factory, "create_llm_provider", lambda model: llm)
    monkeypatch.setattr(factory, "create_embedding_provider", lambda model: MockEmbedder())

    return cfg, llm


@pytest.fixture
def client(app_env, tmp_path):
    cfg, llm = app_env
    app = create_app(config_path=str(cfg))
    with TestClient(app) as c:
        yield c, llm


def _seed(cfg):
    mem = get_memory("agent-a", "conv-1")
    asyncio.run(mem.ingest_updates([_fact("Loves espresso.", entity="coffee")]))


# ── non-stream ───────────────────────────────────────────────────────

def test_non_stream_injects_and_remembers(app_env, client, monkeypatch):
    cfg, llm = app_env
    fake = FakeAsyncClient(replies=[make_chat_completion("Great choice!")])

    import nmafc.web.proxy as proxy
    monkeypatch.setattr(proxy, "upstream_client", lambda: fake)

    _seed(cfg)
    llm.add_response("", [_fact("Asked about drinks.", entity="topic")])

    c, _ = client
    resp = c.post("/v1/chat/completions", json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Do I like coffee?"}],
        "agent_id": "agent-a",
        "conversation_id": "conv-1",
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Great choice!"

    sent = fake.chat.completions.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "espresso" in sent[0]["content"]

    assert any("Extract-Only Mode" in p for p in llm.seen_system_prompts)
    assert get_memory("agent-a", "conv-1").get_hot_stats()["count"] == 2


def test_proxy_default_tenant(app_env, client, monkeypatch):
    cfg, llm = app_env
    fake = FakeAsyncClient(replies=[make_chat_completion("Hi")])
    import nmafc.web.proxy as proxy
    monkeypatch.setattr(proxy, "upstream_client", lambda: fake)

    c, _ = client
    resp = c.post("/v1/chat/completions", json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert resp.status_code == 200
    # No body tenant -> default tenant memory is used and answers nothing.
    default_mem = get_memory("default", "default")
    assert default_mem is not None
    assert default_mem._hot._agent_id == "default"


# ── stream ───────────────────────────────────────────────────────────

def test_stream_returns_sse_and_remembers_after(app_env, client, monkeypatch):
    cfg, llm = app_env
    fake = FakeAsyncClient(replies=[make_chat_completion("unused")])
    import nmafc.web.proxy as proxy
    monkeypatch.setattr(proxy, "upstream_client", lambda: fake)

    _seed(cfg)
    llm.add_response("", [_fact("Stream conversation happened.", entity="stream")])

    c, _ = client
    resp = c.post("/v1/chat/completions", json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Say good morning"}],
        "stream": True,
        "agent_id": "agent-a",
        "conversation_id": "conv-1",
    })

    assert resp.status_code == 200
    text = resp.text
    assert '"content": "Good "' in text
    assert '"content": "morning!"' in text
    assert '"content": "Good morning!"' not in text  # token-scoped deltas
    assert text.strip().endswith("data: [DONE]")
    assert fake.chat.completions.calls[0]["messages"][0]["role"] == "system"
    assert any("Extract-Only Mode" in p for p in llm.seen_system_prompts)
    assert get_memory("agent-a", "conv-1").get_hot_stats()["count"] == 2
