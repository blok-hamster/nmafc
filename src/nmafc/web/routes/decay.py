"""Decay Curve projection endpoints — visualize weight trajectories."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from nmafc.engine.decay import compute_lambda, compute_weight
from nmafc.web.deps import get_tenant_memory
from nmafc.wrapper import NeuromorphicMemory

router = APIRouter(prefix="/api/decay", tags=["decay"])

MemDep = Annotated[NeuromorphicMemory, Depends(get_tenant_memory())]


@router.get("/curves")
def get_decay_curves(
    mem: MemDep,
    turns_ahead: int = Query(50, ge=1, le=500, description="How many turns to project"),
):
    """Projected decay curves for all mutable records."""
    records = mem._hot.get_all_mutable()
    config = mem._decay_config
    current_turn = mem.current_turn

    curves = []
    for rec in records:
        points = []
        for delta in range(turns_ahead + 1):
            t = current_turn + delta
            lam = compute_lambda(rec.memory_type, rec.consolidation_index, config)
            w = compute_weight(rec.weight, lam, delta)
            points.append({"turn": t, "weight": round(w, 6)})
        curves.append({
            "record_id": rec.id,
            "entity_name": rec.entity_name,
            "memory_type": rec.memory_type.value,
            "current_weight": rec.weight,
            "consolidation_index": rec.consolidation_index,
            "points": points,
        })

    return {
        "current_turn": current_turn,
        "prune_threshold": config.w_prune,
        "curves": curves,
    }


@router.get("/curves/{record_id}")
def get_single_curve(
    record_id: str,
    mem: MemDep,
    turns_ahead: int = Query(50, ge=1, le=500),
):
    """Decay curve projection for a single record."""
    record = mem._hot.get_record(record_id)
    if record is None:
        return {"error": "Record not found"}

    config = mem._decay_config
    current_turn = mem.current_turn

    points = []
    for delta in range(turns_ahead + 1):
        t = current_turn + delta
        lam = compute_lambda(record.memory_type, record.consolidation_index, config)
        w = compute_weight(record.weight, lam, delta)
        points.append({"turn": t, "weight": round(w, 6)})

    return {
        "record_id": record.id,
        "entity_name": record.entity_name,
        "memory_type": record.memory_type.value,
        "current_weight": record.weight,
        "consolidation_index": record.consolidation_index,
        "prune_threshold": config.w_prune,
        "points": points,
    }


@router.get("/params")
def get_decay_params(mem: MemDep):
    """Current decay hyperparameters."""
    return mem._decay_config.model_dump()


@router.get("/compare")
def compare_curves(
    mem: MemDep,
    record_ids: str = Query(..., description="Comma-separated record IDs"),
    turns_ahead: int = Query(50, ge=1, le=500),
):
    """Compare decay curves for specific records side-by-side."""
    config = mem._decay_config
    current_turn = mem.current_turn
    ids = [rid.strip() for rid in record_ids.split(",") if rid.strip()]

    curves = []
    for rid in ids:
        rec = mem._hot.get_record(rid)
        if rec is None:
            continue
        points = []
        for delta in range(turns_ahead + 1):
            t = current_turn + delta
            lam = compute_lambda(rec.memory_type, rec.consolidation_index, config)
            w = compute_weight(rec.weight, lam, delta)
            points.append({"turn": t, "weight": round(w, 6)})
        curves.append({
            "record_id": rec.id,
            "entity_name": rec.entity_name,
            "memory_type": rec.memory_type.value,
            "current_weight": rec.weight,
            "consolidation_index": rec.consolidation_index,
            "points": points,
        })

    return {
        "current_turn": current_turn,
        "prune_threshold": config.w_prune,
        "curves": curves,
    }
