"""Memory Explorer endpoints — browse and search Hot RAM records."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from nmafc.web.deps import get_tenant_memory
from nmafc.wrapper import NeuromorphicMemory

router = APIRouter(prefix="/api/memory", tags=["memory"])

MemDep = Annotated[NeuromorphicMemory, Depends(get_tenant_memory())]


@router.get("/all")
def get_all_records(mem: MemDep):
    """All Hot RAM records for the current agent+conversation."""
    records = mem._hot.get_all()
    return [r.model_dump() for r in records]


@router.get("/mutable")
def get_mutable_records(mem: MemDep):
    """All non-CoreAnchor records (decaying memories)."""
    records = mem._hot.get_all_mutable()
    return [r.model_dump() for r in records]


@router.get("/search")
def search_memory(
    mem: MemDep,
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=100),
):
    """Vector similarity search across Hot RAM."""
    import asyncio
    embedder = mem._embedder
    loop = asyncio.get_event_loop()
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            query_vec = pool.submit(asyncio.run, embedder.embed_single(q)).result()
    else:
        query_vec = asyncio.run(embedder.embed_single(q))
    results = mem._hot.search(query_vec, top_k=top_k)
    return [
        {
            "record": r.record.model_dump(),
            "score": r.score,
            "hops": r.hops,
        }
        for r in results
    ]


@router.get("/entity/{entity_name}")
def get_by_entity(entity_name: str, mem: MemDep):
    """All records for a specific entity."""
    records = mem._hot.get_by_entity(entity_name)
    return [r.model_dump() for r in records]


@router.get("/{record_id}")
def get_record(record_id: str, mem: MemDep):
    """Single record by ID."""
    record = mem._hot.get_record(record_id)
    if record is None:
        return {"error": "Record not found"}
    return record.model_dump()


@router.get("")
def get_stats(mem: MemDep):
    """Hot RAM statistics: count, avg weight, type breakdown."""
    return mem.get_hot_stats()
