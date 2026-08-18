"""NMAFC Web UI — FastAPI application factory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from nmafc.web.deps import get_tenant_memory, set_base_config, shutdown_all
from nmafc.web.routes import config, decay, events, graph, memory, process
from nmafc.web.ws import manager


def create_app(config_path: str | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    The base config is stored at startup; per-tenant NeuromorphicMemory
    instances are created lazily on first request for each
    (agent_id, conversation_id) pair.
    """
    if config_path is None:
        config_path = os.environ.get("NMAFC_CONFIG_PATH", "configs/default.toml")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    app = FastAPI(
        title="NMAFC Web UI",
        description="Visual memory explorer for the Neuromorphic Memory Architecture",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(memory.router)
    app.include_router(graph.router)
    app.include_router(events.router)
    app.include_router(decay.router)
    app.include_router(config.router)
    app.include_router(process.router)

    @app.on_event("startup")
    async def startup():
        from nmafc.storage.config import NMafcConfig

        path = Path(config_path)
        if path.exists():
            cfg = NMafcConfig.from_env_or_toml(path)
        else:
            cfg = NMafcConfig.from_env_or_toml(config_path)

        set_base_config(cfg)

    @app.on_event("shutdown")
    async def shutdown():
        shutdown_all()

    @app.websocket("/ws/live")
    async def websocket_endpoint(websocket: WebSocket):
        # Tenant scoping via optional query params on the WS URL:
        #   ws://localhost:8000/ws/live?agent_id=acme&conversation_id=conv-1
        agent_id = websocket.query_params.get("agent_id", "default")
        conversation_id = websocket.query_params.get("conversation_id", "default")

        await manager.connect(websocket, agent_id=agent_id, conversation_id=conversation_id)
        try:
            while True:
                raw = await websocket.receive_text()
                # Client can send a subscribe message to switch tenant mid-session
                try:
                    msg = json.loads(raw)
                    if msg.get("type") == "subscribe":
                        new_agent = msg.get("agent_id", agent_id)
                        new_conv = msg.get("conversation_id", conversation_id)
                        await manager.resubscribe(websocket, new_agent, new_conv)
                        agent_id = new_agent
                        conversation_id = new_conv
                except (json.JSONDecodeError, TypeError):
                    pass
        except WebSocketDisconnect:
            await manager.disconnect(websocket)

    return app


def main():
    parser = argparse.ArgumentParser(description="NMAFC Web UI Server")
    parser.add_argument(
        "--config", default="configs/default.toml",
        help="Path to NMAFC TOML config file",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Install web dependencies: pip install nmafc[web]", file=sys.stderr)
        sys.exit(1)

    app = create_app(config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
