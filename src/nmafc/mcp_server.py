"""MCP server: cause and effect memory for Claude/Cursor agents under `nmafc-mcp`.

This is the distribution channel that needs no code changed at all. A Claude
Desktop or Cursor install registers the server once and every agent gains two
stores that the conversational one cannot serve:

  - **conversational memory** (`memory_recall` / `memory_remember` /
    `memory_forget`) -- the same recall/remember/forget contract as the OpenAI
    middleware, exposed as tools an agent calls when an answer depends on what
    was said before;
  - **code memory** (`index_repo` / `find_callers` / `repo_status`) -- the exact
    symbol graph, which needs no embeddings and no API calls, only a parse. The
    index is cached per repository inside the server process, so a session of
    questions pays for the parse once and `repo_status` is a hash comparison
    after that.

Nothing here routes through `process_turn()`. Retrieval must not generate and
extraction must not attach, and the whole reason a second distribution channel
exists is that raw recall/remember is already enough.

`mcp<2` is required (the 2.x line renamed FastMCP and broke the tool API).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nmafc.code.index import DEFAULT_IGNORE, SymbolIndex
from nmafc.code.render import pointer
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


def memory_recall_tool(
    memories: dict[tuple[str, str], SyncNeuromorphicMemory],
    query: str,
    agent_id: str = "default",
    conversation_id: str = "default",
    top_k: int | None = None,
) -> dict:
    """Recall memories matching a question. Read-only; nothing is written."""
    mem = _tenant_memory(memories, agent_id, conversation_id)
    result = mem.recall(query, agent_id=agent_id, conversation_id=conversation_id, top_k=top_k)
    return {
        "context": result.context,
        "hits": len(result.hits),
        "top_hits": [
            {
                "entity": h.entity_name,
                "fact": h.fact_content,
                "score": round(h.score, 3) if h.score is not None else None,
            }
            for h in result.hits[:10]
        ],
    }


def memory_remember_tool(
    memories: dict[tuple[str, str], SyncNeuromorphicMemory],
    transcript: list[dict],
    agent_id: str = "default",
    conversation_id: str = "default",
) -> dict:
    """Remember a chat transcript (list of {'role','content'} messages)."""
    mem = _tenant_memory(memories, agent_id, conversation_id)
    result = mem.remember(transcript, agent_id=agent_id, conversation_id=conversation_id)
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
    """Build the MCP server with all six tools registered."""
    if FastMCP is None:  # pragma: no cover - extra not installed
        raise RuntimeError(MISSING_MCP)

    mcp = FastMCP(
        "nmafc",
        instructions=(
            "Persistent memory for agents: conversational facts via recall/"
            "remember/forget, plus an exact symbol graph for code via "
            "index_repo/find_callers/repo_status."
        ),
    )
    memories: dict[tuple[str, str], SyncNeuromorphicMemory] = {}
    indices: dict[str, SymbolIndex] = {}

    @mcp.tool()
    def memory_recall(
        query: str,
        agent_id: str = "default",
        conversation_id: str = "default",
        top_k: int | None = None,
    ) -> dict:
        """Recall memories matching a question. Read-only; nothing is written."""
        return memory_recall_tool(memories, query, agent_id, conversation_id, top_k)

    @mcp.tool()
    def memory_remember(
        transcript: list[dict],
        agent_id: str = "default",
        conversation_id: str = "default",
    ) -> dict:
        """Remember a chat transcript (list of {'role','content'} messages).

        The last user message is the turn; earlier messages are history and the
        assistant reply is extracted from alongside it. No reply is generated.
        """
        return memory_remember_tool(memories, transcript, agent_id, conversation_id)

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
