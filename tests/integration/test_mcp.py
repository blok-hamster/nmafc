"""Phase 2B: the nmafc-mcp server's nine tools.

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


def test_symbol_context_under_budget(tmp_path):
    repo = _repo(tmp_path)
    indices: dict = {}

    out = mcp_server.symbol_context_tool(indices, "what does foo do", str(repo))
    assert out["symbols"] >= 1
    assert "def foo" in out["context"] and "return 1" in out["context"]
    assert out["token_estimate"] <= 1500

    tight = mcp_server.symbol_context_tool(indices, "foo", str(repo), max_tokens=20)
    assert tight["token_estimate"] <= 20

    # A question that names no known symbol gets nothing, not a guess.
    none = mcp_server.symbol_context_tool(indices, "how are invoices rounded", str(repo))
    assert none == {"context": "", "symbols": 0, "token_estimate": 0}

    # An edit is picked up without re-indexing by hand.
    (repo / "a.py").write_text("def foo():\n    return 41\n")
    fresh = mcp_server.symbol_context_tool(indices, "foo", str(repo))
    assert "return 41" in fresh["context"]


def test_notes_add_recall_and_exact_staleness(tmp_path, monkeypatch):
    monkeypatch.setenv("NMAFC_DATA_DIR", str(tmp_path / "data"))
    repo = _repo(tmp_path)
    stores: dict = {}

    added = mcp_server.note_add_tool(
        stores, "foo is the only place the default is set.", "a.py", "foo", str(repo))
    assert added["stored"] is True and added["anchor"] == "a.py::foo"

    refused = mcp_server.note_add_tool(stores, "x", "missing.py", "", str(repo))
    assert refused["stored"] is False and "not found" in refused["error"]

    hit = mcp_server.note_recall_tool(stores, "where is the default set", repo_path=str(repo))
    assert hit["notes"] == 1 and hit["stale"] == 0
    assert "foo is the only place the default is set." in hit["context"]

    # Unrelated question: empty, and costs nothing.
    miss = mcp_server.note_recall_tool(stores, "kubernetes ingress", repo_path=str(repo))
    assert miss == {"context": "", "notes": 0, "stale": 0, "token_estimate": 0}

    # Editing b.py leaves a note about a.py::foo fresh; editing foo marks it stale.
    (repo / "b.py").write_text("from a import foo\ndef baz():\n    return foo() * 3\n")
    assert mcp_server.note_recall_tool(stores, "default", repo_path=str(repo))["stale"] == 0
    (repo / "a.py").write_text("def foo():\n    return 2\n\ndef bar():\n    return foo() + 1\n")
    stale = mcp_server.note_recall_tool(stores, "default", repo_path=str(repo))
    assert stale["stale"] == 1 and "[STALE" in stale["context"]

    # A second server process sees the same notes: they live on disk, outside the repo.
    again = mcp_server.note_recall_tool({}, "default", repo_path=str(repo))
    assert again["notes"] == 1
    assert not list(repo.rglob("*.jsonl"))


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


def test_recall_reply_is_lean_and_can_be_capped(sync_mem, monkeypatch):
    mem, llm = sync_mem
    monkeypatch.setattr(mcp_server, "_tenant_memory", lambda c, a, b: mem)
    memories: dict = {}

    for i in range(6):
        llm.add_response("", [_fact(f"Casey fact number {i} " + "z" * 120, entity=f"e{i}")])
        mcp_server.memory_remember_tool(memories, [
            {"role": "user", "content": f"note {i}"},
        ])

    lean = mcp_server.memory_recall_tool(memories, "Casey")
    assert lean["relevant"] is True and lean["hits"] >= 1
    assert all("fact" not in h for h in lean["top_hits"])      # no duplicated fact text
    assert lean["token_estimate"] > 0

    full = mcp_server.memory_recall_tool(memories, "Casey", include_facts=True)
    assert all("fact" in h for h in full["top_hits"])

    capped = mcp_server.memory_recall_tool(memories, "Casey", max_tokens=40)
    assert capped["token_estimate"] <= 40
    assert len(capped["context"]) < len(lean["context"])


def test_recall_relevance_floor(sync_mem, monkeypatch):
    mem, llm = sync_mem
    monkeypatch.setattr(mcp_server, "_tenant_memory", lambda c, a, b: mem)
    memories: dict = {}
    llm.add_response("", [_fact("Name is Casey.", entity="user")])
    mcp_server.memory_remember_tool(memories, [{"role": "user", "content": "My name is Casey"}])

    open_floor = mcp_server.memory_recall_tool(memories, "Casey", min_score=-1.0)
    assert open_floor["relevant"] is True and "Casey" in open_floor["context"]

    # A floor nothing can reach: the store answers "nothing relevant" in a few tokens.
    closed = mcp_server.memory_recall_tool(memories, "Casey", min_score=10.0)
    assert closed["relevant"] is False
    assert closed["context"] == "" and closed["top_hits"] == [] and closed["token_estimate"] == 0


def test_remember_passes_a_date_through(sync_mem, monkeypatch):
    mem, llm = sync_mem
    monkeypatch.setattr(mcp_server, "_tenant_memory", lambda c, a, b: mem)
    seen: list = []
    real = mem.remember

    def spy(transcript, **kwargs):
        seen.append(kwargs.get("occurred_at"))
        return real(transcript, **kwargs)

    monkeypatch.setattr(mem, "remember", spy)
    llm.add_response("", [_fact("Likes tea.", entity="drink")])
    llm.add_response("", [_fact("Likes coffee.", entity="drink2")])
    mcp_server.memory_remember_tool({}, [{"role": "user", "content": "tea"}])
    mcp_server.memory_remember_tool({}, [{"role": "user", "content": "coffee"}],
                                    occurred_at="2026-03-01")

    import re
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", seen[0])      # defaulted to today's date
    assert seen[1] == "2026-03-01"


def test_fit_cuts_on_line_boundaries():
    text = "\n".join(f"line {i} " + "q" * 40 for i in range(30))
    out = mcp_server._fit(text, 50)
    assert mcp_server.tokens(out) <= 50
    assert out and all(line in text.split("\n") for line in out.split("\n"))
    assert mcp_server._fit("short", 50) == "short"
