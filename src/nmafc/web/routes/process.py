"""Turn processing and memory manipulation endpoints."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends

from nmafc.schemas.memory import MemoryStateUpdate
from nmafc.web.deps import get_tenant_memory
from nmafc.web.ws import manager
from nmafc.wrapper import NeuromorphicMemory

router = APIRouter(prefix="/api", tags=["process"])

MemDep = Annotated[NeuromorphicMemory, Depends(get_tenant_memory())]


class ProcessRequest(BaseModel):
    user_msg: str = Field(..., min_length=1, description="User message to process")
    conversation_history: list[dict] | None = Field(
        default=None, description="Optional conversation history"
    )


class IngestRequest(BaseModel):
    updates: list[MemoryStateUpdate] = Field(..., min_length=1)


class RollbackRequest(BaseModel):
    to_turn: int = Field(..., ge=0, description="Turn number to roll back to")


@router.post("/process")
async def process_turn(req: ProcessRequest, mem: MemDep):
    """Process a user message through the full neuromorphic pipeline."""
    response = await mem.process_turn(req.user_msg, req.conversation_history)

    await manager.broadcast({
        "type": "turn_processed",
        "turn": mem.current_turn,
        "response": response[:200],
        "stats": mem.get_hot_stats(),
    }, agent_id=mem._hot._agent_id, conversation_id=mem._hot._conversation_id)

    return {
        "response": response,
        "turn": mem.current_turn,
    }


@router.post("/ingest")
async def ingest_updates(req: IngestRequest, mem: MemDep):
    """Manually ingest memory updates without an LLM call."""
    await mem.ingest_updates(req.updates)

    await manager.broadcast({
        "type": "memory_update",
        "turn": mem.current_turn,
        "count": len(req.updates),
    }, agent_id=mem._hot._agent_id, conversation_id=mem._hot._conversation_id)

    return {"status": "ok", "turn": mem.current_turn}


@router.post("/consolidate")
async def consolidate(mem: MemDep):
    """Manually trigger REM sleep consolidation."""
    count = await mem.consolidate()

    if count > 0:
        await manager.broadcast({
            "type": "consolidation",
            "turn": mem.current_turn,
            "count": count,
        }, agent_id=mem._hot._agent_id, conversation_id=mem._hot._conversation_id)

    return {"consolidated": count}


@router.post("/rollback")
async def rollback(req: RollbackRequest, mem: MemDep):
    """Roll back memory state to a specific turn."""
    restored = await mem.rollback(req.to_turn)

    await manager.broadcast({
        "type": "rollback",
        "turn": req.to_turn,
        "restored": restored,
    }, agent_id=mem._hot._agent_id, conversation_id=mem._hot._conversation_id)

    return {"restored": restored, "current_turn": mem.current_turn}
