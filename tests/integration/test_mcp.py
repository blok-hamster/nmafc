"""Phase 2B: the nmafc-mcp server's six tools.

The code-memory half is offline by construction -- a parse, hashes and
pointers, no API calls -- so `index_repo -> find_callers -> repo_status` is
tested against a scratch repository exactly as an agent would drive it. The
conversational half routes through the same recall/remember/forget contract
as the middleware and needs mocked providers, which are injected by patching
`_tenant_memory` so nothing touches the network.
"""

from __future__ import annotations

import pathlib

import pytest

import nmafc.mcp_server as mcp_server
from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, MemoryType
from nmafc.storage.config import NMafcConfig
from nmafc.wrapper import SyncNeuromorphicMemory

EMBED_DIM = 8


def _fact(text: str, entity: str = "pet", **kwargs) -> MemoryStateUpdate:
    return MemoryStateUpdate(
        entity_name=entity,
        fact_content=text,
        memory_type=MemoryType.CORE_ANCHOR,
        **kwargs,
    )


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


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / "a.py").write_text(
        "def foo():\n    return 1\n\n"
        "def bar():\n    return foo() + 1\n"
    )
    (repo / "b.py").write_text(
        "from a import foo\n"
        "def baz():\n    return foo() * 2\n"
    )
    return repo


# ── code memory (offline) ────────────────────────────────────────────

def test_index_then_find_callers_then_stale(tmp_path):
    repo = _repo(tmp_path)
    indices: dict = {}

    stats = mcp_server.index_repo_tool(indices, str(repo))
    assert stats["files"] == 2
    assert stats["symbols"] == 3  # foo, bar, baz

    callers = mcp_server.find_callers_tool(indices, "foo", str(repo))
    lines = callers["callers"]
    assert len(lines) == 2  # b.py:baz calls foo, a.py:bar calls foo
    assert any("b.py:" in line and "baz" in line for line in lines)
    assert any("a.py:" in line and "bar" in line for line in lines)

    status_before = mcp_server.repo_status_tool(indices, str(repo))
    assert status_before["stale_count"] == 0

    (repo / "b.py").write_text(
        "from a import foo\n"
        "def baz():\n    return foo() * 3\n"
    )
    status = mcp_server.repo_status_tool(indices, str(repo))
    assert status["stale_count"] == 1
    assert "b.py" in status["stale_files"] and "a.py" not in status["stale_files"]


# ── conversational memory (mock providers injected) ──────────────────

@pytest.fixture
def sync_mem(tmp_path, monkeypatch):
    """A real store bound to the default tenant, with mocked providers."""
    # Env pins keep the embedding probe and the repo .env out of the picture.
    monkeypatch.setenv("NMAFC_EMBEDDING_DIM", str(EMBED_DIM))
    cfg = NMafcConfig(
        storage=NMafcConfig().storage.model_copy(
            update={
                "hot_uri": str(tmp_path / "hot.lancedb"),
                "cold_uri": str(tmp_path / "cold.db"),
                "event_log_uri": str(tmp_path / "events.jsonl"),
                "embedding_dim": EMBED_DIM,
            }
        )
    )
    llm = MockLLMProvider()
    mem = SyncNeuromorphicMemory.from_providers(
        llm_provider=llm,
        embedding_provider=MockEmbedder(),
        config=cfg,
    )
    yield mem, llm
    mem.close()


def test_memory_tools_recall_remember_forget(sync_mem, monkeypatch):
    mem, llm = sync_mem
    monkeypatch.setattr(mcp_server, "_tenant_memory", lambda c, a, b: mem)
    memories: dict = {}

    empty = mcp_server.memory_recall_tool(memories, "anything")
    assert empty["context"] == "" and empty["hits"] == 0

    # Remember two facts, then recall picks the relevant one back up.
    llm.add_response("", [_fact("Loves espresso.", entity="coffee")])
    llm.add_response("", [_fact("Name is Casey.", entity="user")])
    mcp_server.memory_remember_tool(memories, [
        {"role": "user", "content": "I like coffee"},
        {"role": "assistant", "content": "Good for you"},
    ])
    mcp_server.memory_remember_tool(memories, [
        {"role": "user", "content": "My name is Casey"},
        {"role": "assistant", "content": "Nice to meet you Casey"},
    ])

    hit = mcp_server.memory_recall_tool(memories, "what's my name")
    assert hit["hits"] >= 1
    assert "Casey" in hit["context"]

    gone = mcp_server.memory_forget_tool(memories, "user")
    assert gone["records_invalidated"] >= 1

    after = mcp_server.memory_recall_tool(memories, "Casey")
    assert "Casey" not in after["context"]

    also_gone = mcp_server.memory_forget_tool(memories, "nope")
    assert also_gone["records_invalidated"] == 0
