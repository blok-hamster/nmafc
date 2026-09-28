"""Mem0-style REST API: `/v1/memories` with add/search/get/patch/delete/history.

Migrating from Mem0 is changing the base URL and the key: `POST /v1/memories`,
`POST /v1/memories/search`, `GET /v1/memories`, `GET|PATCH|DELETE
/v1/memories/{id}`, and `GET /v1/memories/{id}/history`. Tenant scope is Mem0's
own shape -- `agent_id` / `conversation_id` (or `user_id`) in the body, query
string, or `X-Agent-Id` / `X-Conversation-Id` headers, with the usual bearer key.

The semantics are NMAFC's, not Mem0's, and the differences are deliberate:
"update" is override-by-invalidation (a new Hot RAM line, the old one closed
for history), and "delete" is temporal invalidation, never destruction.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, status

from nmafc.schemas.memory import MemoryStateUpdate, MemoryType
from nmafc.web.deps import get_memory
from nmafc.wrapper import NeuromorphicMemory

router = APIRouter(prefix="/v1/memories", tags=["rest"])


def _run(coro):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _record_view(record) -> dict:
    return {
        "id": record.id,
        "entity_name": record.entity_name,
        "fact": record.fact_content,
        "memory_type": record.memory_type.value,
        "weight": record.weight,
        "created_at_turn": record.created_at_turn,
        "valid_at": record.valid_at,
        "invalid_at": record.invalid_at,
    }


def _memory_for(**tenant: str | None) -> NeuromorphicMemory:
    """The per-tenant memory instance, created on first use like the rest of
    the web app's per-tenant cache."""
    agent = tenant.get("agent_id") or tenant.get("user_id") or "default"
    conv = tenant.get("conversation_id") or "default"
    return get_memory(agent, conv)


def _hits_view(result) -> list[dict]:
    return [
        {
            "id": h.id,
            "entity_name": h.entity_name,
            "fact": h.fact_content,
            "memory_type": (
                h.memory_type.value
                if hasattr(h.memory_type, "value")
                else h.memory_type
            ),
            "score": h.score,
            "source": h.source,
            "hop_distance": h.hop_distance,
        }
        for h in result.hits
    ]


@router.get("", status_code=status.HTTP_200_OK)
def list_memories(
    agent_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """List memory records for the tenant, most recent first."""
    mem = _memory_for(agent_id=agent_id, conversation_id=conversation_id, user_id=user_id)
    records = sorted(mem._hot.get_all(), key=lambda r: r.created_at_turn, reverse=True)
    return {"results": [_record_view(r) for r in records[:limit]]}


@router.post("", status_code=status.HTTP_200_OK)
async def add_memories(body: dict):
    """Add memories from a transcript. Body: {"messages": [...], "agent_id"?}."""
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        raise HTTPException(status_code=422, detail="Field 'messages' (list) is required")
    mem = _memory_for(**body)
    result = await mem.remember(messages)
    return {
        "message": "Memory added successfully.",
        "turn": result.turn,
        "updates_ingested": result.updates_ingested,
    }


@router.post("/search", status_code=status.HTTP_200_OK)
async def search_memories(body: dict):
    """Search memories by relevance. Body: {"query": "...", "limit"?: 100}."""
    query = body.get("query")
    if not query or not isinstance(query, str):
        raise HTTPException(status_code=422, detail="Field 'query' (string) is required")
    mem = _memory_for(**body)
    result = await mem.recall(query, top_k=int(body.get("limit") or 100))
    return {"results": _hits_view(result), "context": result.context}


@router.get("/{memory_id}", status_code=status.HTTP_200_OK)
def get_memory_record(
    memory_id: str,
    agent_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
):
    """Fetch one record by id within the tenant scope."""
    mem = _memory_for(agent_id=agent_id, conversation_id=conversation_id, user_id=user_id)
    record = mem._hot.get_record(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _record_view(record)


@router.patch("/{memory_id}", status_code=status.HTTP_200_OK)
async def patch_memory(memory_id: str, body: dict):
    """Replace a memory line under the same entity (override semantics).

    The old record is closed and a new line written; history is preserved in
    the archive. Accepts {"fact": "...", "entity_name"?: "...", "memory_type"?}.
    """
    tenant = {"agent_id": body.get("agent_id") or body.get("user_id"),
              "conversation_id": body.get("conversation_id")}
    mem = _memory_for(**tenant)
    record = mem._hot.get_record(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    if record.invalid_at is not None:
        raise HTTPException(status_code=409, detail="Memory already invalidated")

    fact = body.get("fact")
    if not fact or not isinstance(fact, str):
        raise HTTPException(status_code=422, detail="Field 'fact' (string) is required")

    entity = body.get("entity_name") or record.entity_name
    raw_type = body.get("memory_type", record.memory_type.value)
    if isinstance(raw_type, str):
        try:
            memory_type = MemoryType(raw_type)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"memory_type must be one of {[t.value for t in MemoryType]}",
            ) from None
    else:
        memory_type = raw_type

    if not await mem.forget_record(memory_id):
        raise HTTPException(status_code=409, detail="Memory could not be replaced")
    await mem.ingest_updates([MemoryStateUpdate(
        entity_name=entity,
        fact_content=fact,
        memory_type=memory_type,
    )])
    current = mem._hot.get_by_entity(entity)
    latest = max((r for r in current if r.invalid_at is None), key=lambda r: r.created_at_turn)
    return {"message": "Memory updated successfully.", "id": latest.id}


@router.delete("/{memory_id}", status_code=status.HTTP_200_OK)
async def delete_memory(
    memory_id: str,
    agent_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
):
    """Temporally invalidate a record. The archive history remains."""
    mem = _memory_for(agent_id=agent_id, conversation_id=conversation_id, user_id=user_id)
    removed = await mem.forget_record(memory_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Memory not found or already invalidated")
    return {"message": "Memory deleted successfully.", "id": memory_id}


@router.get("/{memory_id}/history", status_code=status.HTTP_200_OK)
def memory_history(
    memory_id: str,
    agent_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
):
    """All versions of a memory line: current record plus archive events."""
    mem = _memory_for(agent_id=agent_id, conversation_id=conversation_id, user_id=user_id)
    record = mem._hot.get_record(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    archive = mem.get_entity_events(record.entity_name, limit=100)
    return {
        "memory_id": memory_id,
        "entity_name": record.entity_name,
        "current": _record_view(record),
        "history": [
            {
                "turn": ev.get("turn"),
                "type": (
                    ev.get("type").value
                    if isinstance(ev.get("type"), str) is False and hasattr(ev.get("type"), "value")
                    else ev.get("type")
                ),
                "entity_name": ev.get("entity_name"),
                "fact": ev.get("fact_content") or ev.get("facts") or ev.get("payload"),
            }
            for ev in (e if isinstance(e, dict) else e.model_dump() for e in archive)
        ],
    }
