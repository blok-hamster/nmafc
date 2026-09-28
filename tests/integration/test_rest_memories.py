"""Phase 2C: Mem0-style `/v1/memories` + bearer auth + llms.txt.

Runs against the FastAPI app with mocked providers and pinned storage env, so
nothing touches the network or the repo's ./data. Covers the add / search /
get / patch / delete / history lifecycle, tenant scoping via body/query, the
bearer-key gate (open when unconfigured, enforced when configured), and
/llms.txt.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import nmafc.integration.factory as factory
from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, MemoryType
from nmafc.web.app import create_app

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

    def add_response(self, text: str, updates: list[MemoryStateUpdate]) -> None:
        self._queue.append((text, updates))

    async def chat_with_extraction(self, messages, system_prompt):
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
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NMAFC_EMBEDDING_DIM", str(EMBED_DIM))
    monkeypatch.setenv("NMAFC_HOT_URI", str(tmp_path / "hot.lancedb"))
    monkeypatch.setenv("NMAFC_COLD_URI", str(tmp_path / "cold.db"))
    monkeypatch.setenv("NMAFC_EVENT_LOG_URI", str(tmp_path / "events.jsonl"))
    monkeypatch.delenv("NMAFC_API_KEY", raising=False)
    monkeypatch.delenv("NMAFC_API_KEYS", raising=False)

    llm = MockLLMProvider()
    monkeypatch.setattr(factory, "create_llm_provider", lambda model: llm)
    monkeypatch.setattr(factory, "create_embedding_provider", lambda model: MockEmbedder())

    cfg = tmp_path / "custom.toml"
    cfg.write_text(
        f'[storage]\n'
        f'hot_uri = "{tmp_path}/hot.lancedb"\n'
        f'cold_uri = "{tmp_path}/cold.db"\n'
        f'event_log_uri = "{tmp_path}/events.jsonl"\n'
        f'embedding_dim = {EMBED_DIM}\n'
        f'[embedding]\nprovider_model = "mock/t3s"\n'
        f'[llm]\nprovider_model = "mock/gpt-4o-mini"\n'
    )

    app = create_app(config_path=str(cfg))
    with TestClient(app) as c:
        yield c, llm


def _add(client, messages, llm, updates, **tenant):
    llm.add_response("", updates)
    return client.post("/v1/memories", json={"messages": messages, **tenant})


def test_memories_lifecycle(client):
    c, llm = client

    resp = _add(c, [
        {"role": "user", "content": "My name is Casey"},
    ], llm, [_fact("Name is Casey.", entity="user")], user_id="u-1")
    assert resp.status_code == 200
    assert resp.json()["updates_ingested"] == 1

    search = c.post("/v1/memories/search", json={"query": "your name", "user_id": "u-1"})
    assert search.status_code == 200
    results = search.json()["results"]
    assert len(results) >= 1
    mem_id = results[0]["id"]
    assert results[0]["fact"] == "Name is Casey."

    # Tenant isolation: a different user sees an empty store.
    other = c.post("/v1/memories/search", json={"query": "your name", "agent_id": "other"})
    assert other.json()["results"] == []

    got = c.get(f"/v1/memories/{mem_id}", params={"user_id": "u-1"})
    assert got.status_code == 200
    assert got.json()["id"] == mem_id

    patched = c.patch(
        f"/v1/memories/{mem_id}",
        json={"fact": "Name is Casey Jones.", "user_id": "u-1"},
    )
    assert patched.status_code == 200
    new_id = patched.json()["id"]
    assert new_id != mem_id

    again = c.post("/v1/memories/search", json={"query": "your name", "user_id": "u-1"})
    top = again.json()["results"][0]
    assert top["fact"] == "Name is Casey Jones."

    history = c.get(f"/v1/memories/{mem_id}/history", params={"user_id": "u-1"})
    assert history.status_code == 200
    assert history.json()["current"]["id"] == mem_id
    assert isinstance(history.json()["history"], list)

    deleted = c.delete(f"/v1/memories/{new_id}", params={"user_id": "u-1"})
    assert deleted.status_code == 200

    gone = c.post("/v1/memories/search", json={"query": "Casey Jones", "user_id": "u-1"})
    assert gone.json()["results"] == []


def test_list_and_tombstoned_archive(client):
    c, llm = client
    _add(c, [{"role": "user", "content": "I like espresso"}], llm,
         [_fact("Loves espresso.", entity="coffee")], user_id="u-2")

    listed = c.get("/v1/memories", params={"user_id": "u-2"})
    assert listed.status_code == 200
    assert any(r["entity_name"] == "coffee" for r in listed.json()["results"])


def test_auth_gate(client, monkeypatch):
    c, _ = client

    # No keys configured -> open.
    assert c.post("/v1/memories/search", json={"query": "x"}).status_code == 200

    monkeypatch.setenv("NMAFC_API_KEYS", "nmafc_test_key")
    assert c.post("/v1/memories/search", json={"query": "x"}).status_code == 401
    assert c.post(
        "/v1/memories/search", json={"query": "x"},
        headers={"Authorization": "Bearer wrong"},
    ).status_code == 401
    ok = c.post(
        "/v1/memories/search", json={"query": "x"},
        headers={"Authorization": "Bearer nmafc_test_key"},
    )
    assert ok.status_code == 200


def test_llms_txt(client):
    c, _ = client
    resp = c.get("/llms.txt")
    assert resp.status_code == 200
    assert "/v1/memories" in resp.text
    assert "/v1/chat/completions" in resp.text
