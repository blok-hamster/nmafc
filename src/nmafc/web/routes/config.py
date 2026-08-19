"""Config and stats endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from nmafc.web.deps import get_tenant_memory
from nmafc.wrapper import NeuromorphicMemory

router = APIRouter(prefix="/api", tags=["config"])

MemDep = Annotated[NeuromorphicMemory, Depends(get_tenant_memory())]


@router.get("/config")
def get_config():
    """Base NMafcConfig as JSON (tenant-independent)."""
    from nmafc.web.deps import _BASE_CONFIG
    if _BASE_CONFIG is None:
        return {"error": "Config not loaded"}
    return _BASE_CONFIG.model_dump()


@router.get("/health")
def health_check():
    """Health check."""
    return {"status": "ok"}


@router.get("/stats")
def get_all_stats(mem: MemDep):
    """Combined stats: Hot RAM, Cold ROM, and event log."""
    return {
        "hot": mem.get_hot_stats(),
        "cold": mem.get_cold_stats(),
        "events": mem.get_event_stats(),
        "current_turn": mem.current_turn,
    }
