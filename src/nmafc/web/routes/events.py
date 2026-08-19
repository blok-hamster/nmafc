"""Event Log endpoints — query cognitive events, timeline, entity history."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from nmafc.schemas.events import EventType

from nmafc.web.deps import get_tenant_memory
from nmafc.wrapper import NeuromorphicMemory

router = APIRouter(prefix="/api/events", tags=["events"])

MemDep = Annotated[NeuromorphicMemory, Depends(get_tenant_memory())]


@router.get("")
def query_events(
    mem: MemDep,
    turn_from: int | None = Query(None, description="Start turn (inclusive)"),
    turn_to: int | None = Query(None, description="End turn (inclusive)"),
    event_type: str | None = Query(None, description="Filter by event type"),
    entity_name: str | None = Query(None, description="Filter by entity name"),
    record_id: str | None = Query(None, description="Filter by record ID"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Query cognitive events with flexible filters."""
    kwargs: dict = {"limit": limit, "offset": offset}
    if turn_from is not None:
        kwargs["turn_from"] = turn_from
    if turn_to is not None:
        kwargs["turn_to"] = turn_to
    if event_type:
        kwargs["event_types"] = [EventType(event_type)]
    if entity_name:
        kwargs["entity_name"] = entity_name
    if record_id:
        kwargs["record_id"] = record_id
    events = mem.get_events(**kwargs)
    return [e.model_dump(mode="json") for e in events]


@router.get("/timeline")
def get_timeline(mem: MemDep, limit: int = Query(100, ge=1, le=1000)):
    """Aggregated event counts per turn, grouped by event type."""
    return mem.get_event_timeline(limit=limit)


@router.get("/overrides")
def get_override_events(mem: MemDep, limit: int = Query(50, ge=1, le=500)):
    """All suppression/override events."""
    events = mem.get_events(event_types=[EventType.SUPPRESSION], limit=limit)
    return [e.model_dump(mode="json") for e in events]


@router.get("/consolidations")
def get_consolidation_events(mem: MemDep, limit: int = Query(50, ge=1, le=500)):
    """All consolidation events (ActiveContext -> CoreAnchor)."""
    events = mem.get_events(event_types=[EventType.CONSOLIDATION], limit=limit)
    return [e.model_dump(mode="json") for e in events]


@router.get("/prunes")
def get_prune_events(mem: MemDep, limit: int = Query(50, ge=1, le=500)):
    """All pruning/eviction events."""
    events = mem.get_events(event_types=[EventType.PRUNE], limit=limit)
    return [e.model_dump(mode="json") for e in events]


@router.get("/ltp")
def get_ltp_events(mem: MemDep, limit: int = Query(50, ge=1, le=500)):
    """All LTP (spaced repetition) events."""
    events = mem.get_events(event_types=[EventType.LTP], limit=limit)
    return [e.model_dump(mode="json") for e in events]


@router.get("/entity/{entity_name}")
def get_entity_history(entity_name: str, mem: MemDep, limit: int = Query(50, ge=1, le=500)):
    """All cognitive events for a specific entity."""
    events = mem.get_entity_events(entity_name, limit=limit)
    return [e.model_dump(mode="json") for e in events]


@router.get("/stats")
def get_event_stats(mem: MemDep):
    """Event log statistics: total count, breakdown by type."""
    return mem.get_event_stats()
