"""WebSocket connection manager for real-time event broadcasting."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections and broadcasts cognitive events.

    Connections are tracked per (agent_id, conversation_id) pair so that
    broadcast events are only sent to clients subscribed to that tenant.
    """

    def __init__(self) -> None:
        self._connections: dict[WebSocket, tuple[str, str]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        agent_id: str = "default",
        conversation_id: str = "default",
    ) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[websocket] = (agent_id, conversation_id)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.pop(websocket, None)

    async def resubscribe(
        self,
        websocket: WebSocket,
        agent_id: str,
        conversation_id: str,
    ) -> None:
        """Switch a connected client to a different tenant."""
        async with self._lock:
            if websocket in self._connections:
                self._connections[websocket] = (agent_id, conversation_id)

    async def broadcast(
        self,
        event: dict[str, Any],
        agent_id: str = "default",
        conversation_id: str = "default",
    ) -> None:
        """Push an event to clients subscribed to the given tenant."""
        message = json.dumps(event, default=str)
        dead: list[WebSocket] = []
        async with self._lock:
            for ws, (a, c) in self._connections.items():
                if a == agent_id and c == conversation_id:
                    try:
                        await ws.send_text(message)
                    except Exception:
                        dead.append(ws)
            for ws in dead:
                self._connections.pop(ws, None)

    @property
    def active_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()
