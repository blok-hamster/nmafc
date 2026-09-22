"""Memory Explorer endpoints — browse and search Hot RAM records."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from nmafc.web.deps import get_tenant_memory
from nmafc.wrapper import NeuromorphicMemory

router = APIRouter(prefix="/api/memory", tags=["memory"])

MemDep = Annotated[NeuromorphicMemory, Depends(get_tenant_memory())]


def _created_date(mem: NeuromorphicMemory, record) -> str | None:
    """The calendar date behind a record's creation turn, if the store has one.

    Resolved through the router so the list view and the engine agree on what
    "the date of a turn" means (nearest earlier dated turn, clock time
    stripped). None for stores that were never dated -- the explorer falls back
    to the turn number.
    """
    turn = record.created_at_turn or record.valid_at
    if not turn:
        return None
    date = mem._router._date_for(turn)
    return mem._router._day(date) if date else None


def _with_created_date(mem: NeuromorphicMemory, record) -> dict:
    data = record.model_dump()
    data["created_date"] = _created_date(mem, record)
    return data


@router.get("/all")
def get_all_records(mem: MemDep):
    """All Hot RAM records for the current agent+conversation."""
    records = mem._hot.get_all()
    return [_with_created_date(mem, r) for r in records]


@router.get("/mutable")
def get_mutable_records(mem: MemDep):
    """All non-CoreAnchor records (decaying memories)."""
    records = mem._hot.get_all_mutable()
    return [_with_created_date(mem, r) for r in records]


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
            "record": _with_created_date(mem, r.record),
            "score": r.score,
            "hops": r.hops,
        }
        for r in results
    ]


@router.get("/entity/{entity_name}")
def get_by_entity(entity_name: str, mem: MemDep):
    """All records for a specific entity."""
    records = mem._hot.get_by_entity(entity_name)
    return [_with_created_date(mem, r) for r in records]


@router.get("/{record_id}")
def get_record(record_id: str, mem: MemDep):
    """Single record by ID."""
    record = mem._hot.get_record(record_id)
    if record is None:
        return {"error": "Record not found"}
    return _with_created_date(mem, record)


@router.get("/{record_id}/source")
def get_record_source(record_id: str, mem: MemDep):
    """A record's validity dates and the verbatim turns behind it.

    The validity block mirrors what the router renders into the prompt: the
    extractor's own date wording wins, then the calendar date of the turn that
    mentioned the fact, then a turn number. `has_dates` is False for stores
    that were never dated, so the UI can point at the backfill scripts instead
    of pretending turn numbers are dates.

    Source turns carry `created_at_turn` alongside `valid_at` and `invalid_at`,
    because a fact stated long after an event was extracted from the turn it
    was *said* in. Backends without turn_text return no text and the view
    degrades to turn numbers.
    """
    record = mem._hot.get_record(record_id)
    if record is None:
        return {"error": "Record not found"}

    timestamps: dict[int, str] = {}
    ts_getter = getattr(mem._cold, "turn_timestamps", None)
    if callable(ts_getter):
        timestamps = ts_getter()

    turns = sorted({
        t for t in (record.created_at_turn, record.valid_at, record.invalid_at) if t
    })

    texts: dict[int, str] = {}
    texts_getter = getattr(mem._cold, "text_for_turns", None)
    if callable(texts_getter):
        texts = texts_getter(turns)

    router = mem._router

    def resolve(turn: int | None) -> str | None:
        if not turn:
            return None
        date = router._date_for(turn)
        return router._day(date) if date else None

    validity = {
        "valid_at": record.valid_at,
        "valid_at_text": record.valid_at_text,
        "valid_date": resolve(record.valid_at or record.created_at_turn),
        "invalid_at": record.invalid_at,
        "invalid_date": resolve(record.invalid_at),
        "has_dates": bool(timestamps),
    }

    source_turns = [
        {"turn": t, "timestamp": timestamps.get(t), "text": texts.get(t)}
        for t in turns
    ]

    return {
        "record": _with_created_date(mem, record),
        "validity": validity,
        "source_turns": source_turns,
    }


@router.get("")
def get_stats(mem: MemDep):
    """Hot RAM statistics: count, avg weight, type breakdown."""
    return mem.get_hot_stats()
