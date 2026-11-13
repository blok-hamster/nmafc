"""MCP server: cause and effect memory for Claude/Cursor agents under `nmafc-mcp`.

This is the distribution channel that needs no code changed at all. A Claude
Desktop or Cursor install registers the server once and every agent gains two
stores that the conversational one cannot serve:

  - **conversational memory** (`memory_recall` / `memory_remember` /
    `memory_forget`) -- the same recall/remember/forget contract as the OpenAI
    middleware, exposed as tools an agent calls when an answer depends on what
    was said before;
  - **code memory** (`index_repo` / `find_callers` / `repo_status` /
    `symbol_context`) -- the exact symbol graph, which needs no embeddings and
    no API calls, only a parse. The index is cached per repository inside the
    server process, so a session of questions pays for the parse once and
    `repo_status` is a hash comparison after that. `symbol_context` returns a
    symbol's source and its neighbourhood under a hard token budget.
  - **anchored notes** (`note_add` / `note_recall`) -- findings about code,
    stored as written with no model call, each tied to a file or symbol and
    flagged stale the moment that anchor's bytes change.

Nothing here routes through `process_turn()`. Retrieval must not generate and
extraction must not attach, and the whole reason a second distribution channel
exists is that raw recall/remember is already enough.

`mcp<2` is required (the 2.x line renamed FastMCP and broke the tool API).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from nmafc.code.index import DEFAULT_IGNORE, SymbolIndex
from nmafc.code.notes import DEFAULT_MIN_COVERAGE, FRESH, NoteStore, render_notes
from nmafc.code.render import pointer, render_symbols, tokens
from nmafc.storage.config import NMafcConfig
from nmafc.wrapper import SyncNeuromorphicMemory

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - extra not installed
    FastMCP = None  # type: ignore[assignment,misc]

MISSING_MCP = (
    "The mcp extra is not installed. Run `pip install 'nmafc[mcp]'` to expose "
    "the nmafc-mcp tools."
)


def _base_config() -> NMafcConfig:
    path = Path("configs/default.toml")
    return NMafcConfig.from_env_or_toml(path if path.exists() else None)


def _tenant_memory(
    memories: dict[tuple[str, str], SyncNeuromorphicMemory],
    agent_id: str,
    conversation_id: str,
) -> SyncNeuromorphicMemory:
    key = (agent_id, conversation_id)
    mem = memories.get(key)
    if mem is None:
        base = _base_config()
        tenant = NMafcConfig(
            storage=base.storage.model_copy(
                update={"agent_id": agent_id, "conversation_id": conversation_id}
            ),
            decay=base.decay,
            llm_provider_model=base.llm_provider_model,
            embedding_provider_model=base.embedding_provider_model,
        )
        mem = SyncNeuromorphicMemory.from_config(config=tenant)
        memories[key] = mem
    return mem


def _index(
    indices: dict[str, SymbolIndex],
    repo_path: str | None,
) -> SymbolIndex:
    if not repo_path:
        repo_path = "."
    path = str(Path(repo_path).resolve())
    idx = indices.get(path)
    if idx is None:
        idx = SymbolIndex(path)
        idx.build(ignore=DEFAULT_IGNORE)
        indices[path] = idx
    return idx


def _callers_lines(index: SymbolIndex, symbol_name: str) -> list[str]:
    lines: list[str] = []
    for symbol in index.callers(symbol_name):
        head = symbol.signature or symbol.qualname
        doc = f"  # {symbol.docline}" if symbol.docline else ""
        lines.append(f"{pointer(symbol)}  {head}{doc}")
    if not lines:
        lines.append("(no callers found)")
    return lines


# Tool bodies are plain functions taking explicit state, so they are callable
# from tests without an event loop: the FastMCP closures below are one-line
# wrappers, and the sync memory wrapper's asyncio.run() works wherever the MCP
# runtime runs a sync tool. State is per-server, created in create_server().

def index_repo_tool(indices: dict[str, SymbolIndex], repo_path: str | None = None) -> dict:
    """Parse a repository into the exact symbol graph. A parse, no tokens."""
    idx = _index(indices, repo_path or ".")
    return {
        "root": str(idx.root),
        "files": len(idx.files),
        "symbols": len(idx.symbols),
    }


def find_callers_tool(
    indices: dict[str, SymbolIndex],
    symbol_name: str,
    repo_path: str | None = None,
) -> dict:
    """Everything that references a symbol, as path:line pointers."""
    idx = _index(indices, repo_path or ".")
    return {
        "symbol": symbol_name,
        "callers": _callers_lines(idx, symbol_name),
        "stale_count": len(idx.stale(ignore=DEFAULT_IGNORE)),
    }


def repo_status_tool(indices: dict[str, SymbolIndex], repo_path: str | None = None) -> dict:
    """Indexed files whose bytes changed (exact staleness) plus index size."""
    idx = _index(indices, repo_path or ".")
    changed = idx.stale(ignore=DEFAULT_IGNORE)
    return {
        "root": str(idx.root),
        "files": len(idx.files),
        "symbols": len(idx.symbols),
        "stale_files": changed,
        "stale_count": len(changed),
    }


def symbol_context_tool(
    indices: dict[str, SymbolIndex],
    query: str,
    repo_path: str | None = None,
    max_tokens: int = 1500,
    hops: int = 1,
) -> dict:
    """Source of the symbols a question names, plus their neighbourhood.

    The named symbols come back in full, what they call and what calls them as
    a signature each, and anything further out as a `path:start-end` pointer.
    `max_tokens` is a hard cap on the block. A question that names no known
    symbol returns an empty context rather than a guess.
    """
    idx = _index(indices, repo_path or ".")
    idx.refresh(ignore=DEFAULT_IGNORE)
    items = idx.retrieve(query, hops=max(hops, 0))
    context = render_symbols(idx, items, max(max_tokens, 0)) if items else ""
    return {
        "context": context,
        "symbols": len(items),
        "token_estimate": tokens(context),
    }


def _notes(stores: dict[str, NoteStore], repo_path: str | None) -> NoteStore:
    path = str(Path(repo_path or ".").resolve())
    store = stores.get(path)
    if store is None:
        store = NoteStore(path)
        stores[path] = store
    return store


def note_add_tool(
    stores: dict[str, NoteStore],
    text: str,
    path: str,
    symbol: str = "",
    repo_path: str | None = None,
) -> dict:
    """Store a finding about code as written, anchored to a file or a symbol."""
    store = _notes(stores, repo_path)
    try:
        note = store.add(text, path, symbol)
    except ValueError as exc:
        return {"stored": False, "error": str(exc)}
    return {"stored": True, "id": note.id, "anchor": note.anchor}


def note_recall_tool(
    stores: dict[str, NoteStore],
    query: str = "",
    path: str | None = None,
    symbol: str | None = None,
    repo_path: str | None = None,
    max_tokens: int = 800,
    top_k: int = 8,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict:
    """Notes matching a question and/or an anchor, with exact staleness.

    `min_coverage` is the relevance bar: a note must cover at least that share
    of the question's words (rare words count for more) to be returned.
    """
    store = _notes(stores, repo_path)
    rows = store.search(query, path=path, symbol=symbol, top_k=top_k,
                        min_coverage=min_coverage)
    context, shown = render_notes(rows, max_tokens)
    return {
        "context": context,
        "notes": shown,
        "stale": sum(1 for _, status, _ in rows[:shown] if status != FRESH),
        "token_estimate": tokens(context),
    }


def _fit(context: str, max_tokens: int) -> str:
    """Cut a context block at a line boundary so it stays within a token cap."""
    if tokens(context) <= max_tokens:
        return context
    kept: list[str] = []
    used = 0
    for line in context.split("\n"):
        cost = tokens(line) + 1
        if used + cost > max_tokens:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept)


def memory_recall_tool(
    memories: dict[tuple[str, str], SyncNeuromorphicMemory],
    query: str,
    agent_id: str = "default",
    conversation_id: str = "default",
    top_k: int | None = None,
    max_tokens: int | None = None,
    min_score: float | None = None,
    include_facts: bool = False,
) -> dict:
    """Recall memories matching a question. Read-only; nothing is written.

    `context` already holds every recalled fact, so `top_hits` lists only the
    entity and score of each unless `include_facts` is set; repeating the fact
    text there roughly doubled the size of a reply. `max_tokens` caps the
    context. `min_score` is a relevance floor: when the best hit scores below
    it the reply carries an empty context and `relevant: false`, so a question
    the store knows nothing about costs a few tokens rather than a page of
    unrelated memories. The floor has no default because a sensible value
    depends on the embedding model in use.
    """
    mem = _tenant_memory(memories, agent_id, conversation_id)
    result = mem.recall(query, agent_id=agent_id, conversation_id=conversation_id, top_k=top_k)
    scores = [h.score for h in result.hits if h.score is not None]
    relevant = True
    if min_score is not None and (not scores or max(scores) < min_score):
        relevant = False
    context = result.context if relevant else ""
    if max_tokens is not None:
        context = _fit(context, max(max_tokens, 0))
    top_hits = []
    for h in result.hits[:10]:
        row = {
            "entity": h.entity_name,
            "score": round(h.score, 3) if h.score is not None else None,
        }
        if include_facts:
            row["fact"] = h.fact_content
        top_hits.append(row)
    return {
        "context": context,
        "relevant": relevant,
        "hits": len(result.hits),
        "top_hits": top_hits if relevant else [],
        "token_estimate": tokens(context),
    }


def memory_remember_tool(
    memories: dict[tuple[str, str], SyncNeuromorphicMemory],
    transcript: list[dict],
    agent_id: str = "default",
    conversation_id: str = "default",
    occurred_at: str | None = None,
) -> dict:
    """Remember a chat transcript (list of {'role','content'} messages).

    `occurred_at` is when the conversation happened. It defaults to today's
    UTC date: with no date at all the extractor has nothing to resolve "last
    week" against and was seen stamping a fact three years in the past.
    """
    mem = _tenant_memory(memories, agent_id, conversation_id)
    when = occurred_at or datetime.now(timezone.utc).date().isoformat()
    result = mem.remember(
        transcript, agent_id=agent_id, conversation_id=conversation_id, occurred_at=when,
    )
    return {
        "turn": result.turn,
        "updates_ingested": result.updates_ingested,
        "overrides_suppressed": result.overrides_suppressed,
        "consolidated": result.consolidated,
    }


def memory_forget_tool(
    memories: dict[tuple[str, str], SyncNeuromorphicMemory],
    entity_name: str,
    agent_id: str = "default",
    conversation_id: str = "default",
) -> dict:
    """Temporally invalidate all active records for one entity."""
    mem = _tenant_memory(memories, agent_id, conversation_id)
    invalidated = mem.forget_entity(entity_name, agent_id=agent_id, conversation_id=conversation_id)
    return {"entity_name": entity_name, "records_invalidated": invalidated}


def create_server() -> FastMCP:
    """Build the MCP server with all nine tools registered."""
    if FastMCP is None:  # pragma: no cover - extra not installed
        raise RuntimeError(MISSING_MCP)

    mcp = FastMCP(
        "nmafc",
        instructions=(
            "Persistent memory for agents: conversational facts via recall/"
            "remember/forget, an exact symbol graph for code via "
            "index_repo/find_callers/repo_status/symbol_context, and findings "
            "about code via note_add/note_recall. Save what you learn about a "
            "repository with note_add: it costs no model call and is flagged "
            "stale when the code it describes changes."
        ),
    )
    memories: dict[tuple[str, str], SyncNeuromorphicMemory] = {}
    indices: dict[str, SymbolIndex] = {}
    notes: dict[str, NoteStore] = {}

    @mcp.tool()
    def memory_recall(
        query: str,
        agent_id: str = "default",
        conversation_id: str = "default",
        top_k: int | None = None,
        max_tokens: int | None = None,
        min_score: float | None = None,
        include_facts: bool = False,
    ) -> dict:
        """Recall memories matching a question. Read-only; nothing is written.

        `max_tokens` caps the returned context. `min_score` is a relevance
        floor: below it the reply is empty with `relevant: false`. Set
        `include_facts` to repeat each fact's text in `top_hits`.
        """
        return memory_recall_tool(
            memories, query, agent_id, conversation_id, top_k,
            max_tokens, min_score, include_facts,
        )

    @mcp.tool()
    def memory_remember(
        transcript: list[dict],
        agent_id: str = "default",
        conversation_id: str = "default",
        occurred_at: str | None = None,
    ) -> dict:
        """Remember a chat transcript (list of {'role','content'} messages).

        The last user message is the turn; earlier messages are history and the
        assistant reply is extracted from alongside it. No reply is generated.
        `occurred_at` is when the conversation happened; it defaults to today.
        """
        return memory_remember_tool(
            memories, transcript, agent_id, conversation_id, occurred_at,
        )

    @mcp.tool()
    def memory_forget(
        entity_name: str,
        agent_id: str = "default",
        conversation_id: str = "default",
    ) -> dict:
        """Temporally invalidate all active records for one entity."""
        return memory_forget_tool(memories, entity_name, agent_id, conversation_id)

    @mcp.tool()
    def index_repo(repo_path: str | None = None) -> dict:
        """Parse a repository into the exact symbol graph (a parse, no tokens)."""
        return index_repo_tool(indices, repo_path)

    @mcp.tool()
    def find_callers(
        symbol_name: str,
        repo_path: str | None = None,
    ) -> dict:
        """Everything that references a symbol, as path:line pointers."""
        return find_callers_tool(indices, symbol_name, repo_path)

    @mcp.tool()
    def repo_status(repo_path: str | None = None) -> dict:
        """Indexed files whose bytes changed (exact staleness) plus index size."""
        return repo_status_tool(indices, repo_path)

    @mcp.tool()
    def symbol_context(
        query: str,
        repo_path: str | None = None,
        max_tokens: int = 1500,
        hops: int = 1,
    ) -> dict:
        """Source of the symbols a question names, plus callers and callees,
        under a hard token budget. A question naming no symbol returns nothing."""
        return symbol_context_tool(indices, query, repo_path, max_tokens, hops)

    @mcp.tool()
    def note_add(
        text: str,
        path: str,
        symbol: str = "",
        repo_path: str | None = None,
    ) -> dict:
        """Save a finding about code exactly as written, anchored to a file or
        to one symbol in it. No model call. Flagged stale if the anchor changes."""
        return note_add_tool(notes, text, path, symbol, repo_path)

    @mcp.tool()
    def note_recall(
        query: str = "",
        path: str | None = None,
        symbol: str | None = None,
        repo_path: str | None = None,
        max_tokens: int = 800,
        top_k: int = 8,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ) -> dict:
        """Saved findings matching a question and/or a file or symbol, each
        marked fresh or stale by an exact hash check, under a token budget.
        Returns nothing when no note covers enough of the question."""
        return note_recall_tool(
            notes, query, path, symbol, repo_path, max_tokens, top_k, min_coverage,
        )

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="nmafc MCP server (stdio)")
    parser.add_argument(
        "--list-tools", action="store_true",
        help="Print the registered tool names and exit, without launching stdio",
    )
    args = parser.parse_args()

    if FastMCP is None:
        print(MISSING_MCP, file=sys.stderr)
        sys.exit(1)

    server = create_server()
    if args.list_tools:
        for tool in server._tool_manager.list_tools():  # noqa: SLF001
            print(tool.name)
        return
    server.run()


if __name__ == "__main__":
    main()
