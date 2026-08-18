"""Dependency injection for the NMAFC Web UI.

Provides per-tenant (agent_id, conversation_id) NeuromorphicMemory instances
with lazy creation and caching. Each unique pair gets its own memory, backed
by the same base config but isolated in Hot RAM, Cold ROM, and Event Log.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import Header, HTTPException

from nmafc.wrapper import NeuromorphicMemory

_BASE_CONFIG = None
_memory_cache: dict[tuple[str, str], NeuromorphicMemory] = {}


def set_base_config(config) -> None:
    """Store the base NMafcConfig used to spawn per-tenant instances."""
    global _BASE_CONFIG
    _BASE_CONFIG = config


def get_memory(agent_id: str = "default", conversation_id: str = "default") -> NeuromorphicMemory:
    """Get or create a NeuromorphicMemory for the given tenant pair."""
    key = (agent_id, conversation_id)
    mem = _memory_cache.get(key)
    if mem is not None:
        return mem

    if _BASE_CONFIG is None:
        raise RuntimeError("NeuromorphicMemory not initialized. Call set_base_config() first.")

    from nmafc.storage.config import NMafcConfig
    tenant_cfg = NMafcConfig(
        storage=_BASE_CONFIG.storage.model_copy(
            update={"agent_id": agent_id, "conversation_id": conversation_id}
        ),
        decay=_BASE_CONFIG.decay,
        retrieval=_BASE_CONFIG.retrieval,
        llm_provider_model=_BASE_CONFIG.llm_provider_model,
        embedding_provider_model=_BASE_CONFIG.embedding_provider_model,
    )
    mem = NeuromorphicMemory.from_config(config=tenant_cfg)
    _memory_cache[key] = mem
    return mem


async def _get_memory_dependency(
    x_agent_id: Annotated[str | None, Header()] = None,
    x_conversation_id: Annotated[str | None, Header()] = None,
) -> NeuromorphicMemory:
    """FastAPI dependency: extract tenant from headers, return scoped memory."""
    return get_memory(
        agent_id=x_agent_id or "default",
        conversation_id=x_conversation_id or "default",
    )


def get_tenant_memory():
    """Return the FastAPI dependency callable for tenant-scoped memory."""
    return _get_memory_dependency


def shutdown_all() -> None:
    """Close every cached tenant memory and clear the cache."""
    global _BASE_CONFIG
    for mem in _memory_cache.values():
        mem.close()
    _memory_cache.clear()
    _BASE_CONFIG = None
