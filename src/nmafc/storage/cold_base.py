from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nmafc.schemas.memory import MemoryStateUpdate


class ColdStorageBase(ABC):
    """Abstract interface for Cold ROM event log backends.

    Implementations: ColdStorage (SQLite), PostgresColdStorage (PostgreSQL).

    All implementations accept agent_id and conversation_id for multi-tenant isolation.
    Queries are scoped to the (agent_id, conversation_id) pair — no cross-tenant leakage.
    """

    _agent_id: str
    _conversation_id: str

    @abstractmethod
    def append_event(self, update: MemoryStateUpdate, turn: int) -> int: ...

    @abstractmethod
    def mark_inactive(self, event_id: int) -> None: ...

    @abstractmethod
    def get_active_events(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_events_for_entity(self, entity_name: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def keyword_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]: ...

    @abstractmethod
    def count_active(self) -> int: ...

    @abstractmethod
    def count_total(self) -> int: ...

    @abstractmethod
    def close(self) -> None: ...
