"""NMAFC Web UI — FastAPI application factory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from nmafc.web.deps import require_api_key, set_base_config, shutdown_all
from nmafc.web.proxy import router as proxy_router
from nmafc.web.routes import config, decay, events, graph, memory, process, rest_memories
from nmafc.web.ws import manager

STATIC_UI_SUFFIXES = {
    ".html", ".js", ".css", ".json", ".map", ".svg", ".png", ".jpg",
    ".jpeg", ".gif", ".webp", ".ico", ".txt", ".woff", ".woff2", ".ttf",
}


def static_ui_dir() -> Path | None:
    """Resolve the exported web dashboard directory, if one is present.

    Precedence: NMAFC_STATIC_UI_DIR, then the repo checkout's web-ui/out.
    The Docker image sets NMAFC_STATIC_UI_DIR; the one-port story is that the
    same FastAPI process serves both the /api and the exported dashboard.
    """
    env = os.environ.get("NMAFC_STATIC_UI_DIR")
    if env:
        candidate = Path(env)
        if candidate.is_dir():
            return candidate
    checkout_out = Path(__file__).resolve().parents[2] / "web-ui" / "out"
    return checkout_out if checkout_out.is_dir() else None


def _dispatch_static(full_path: str):
    """Serve one file from the exported dashboard, with SPA fallback to index.

    The exported Next.js site also answers client-side routes. Starlette's
    StaticFiles would 404 on those, so any path that is not a real file falls
    back to the route's generated .html (App Router export) and then to
    index.html. Only the known static suffixes are served, and the resolved
    path must stay inside the export directory, so this can never escape to
    config or store files.
    """
    root = static_ui_dir()
    if root is None:
        raise HTTPException(status_code=404, detail="Static dashboard not built")
    if full_path:
        raw = Path(full_path)
        if raw.suffix and raw.suffix.lower() not in STATIC_UI_SUFFIXES:
            raise HTTPException(status_code=404, detail="Not found")
        resolved = root.joinpath(*raw.parts)
    else:
        resolved = root / "index.html"
    candidates = [resolved]
    if not resolved.suffix:
        candidates.append(resolved.with_suffix(".html"))
        if raw.parts:
            candidates.append(root.joinpath(*raw.parts) / "index.html")
    candidates.append(root / "index.html")
    root_resolved = root.resolve()
    for candidate in candidates:
        candidate_resolved = candidate.resolve()
        if not candidate_resolved.is_relative_to(root_resolved):
            raise HTTPException(status_code=404, detail="Not found")
        if candidate_resolved.is_file():
            return FileResponse(candidate_resolved)
    raise HTTPException(status_code=404, detail="Not found")


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
    app.include_router(proxy_router)
    app.include_router(
        rest_memories.router,
        dependencies=[Depends(require_api_key)],
    )

    llms_txt_block = """# NMAFC

Neuromorphic Memory Architecture for Conversational AI: a biologically
inspired, bounded memory store with recall, extraction on ingestion, override
suppression, decay, and pruning.

## Machine-facing API

- POST /v1/chat/completions  OpenAI-compatible proxy with memory interposed
                             (recall -> inject -> remember). Tenant:
                             agent_id / conversation_id in the body.
- POST /v1/memories          Add memories from a chat transcript.
- POST /v1/memories/search   Search memories by relevance.
- GET  /v1/memories          List records for a tenant.
- GET  /v1/memories/{id}     Fetch one record.
- PATCH /v1/memories/{id}    Replace a memory line (override, history kept).
- DELETE /v1/memories/{id}   Forget a record (temporal invalidation).
- GET  /v1/memories/{id}/history  All versions of a memory line.

Tenant scope: agent_id / conversation_id (or Mem0's user_id) via body, query
string, or X-Agent-Id / X-Conversation-Id headers.

Auth on /v1/memories: Authorization: Bearer <key> from NMAFC_API_KEYS
(comma-separated) or NMAFC_API_KEY. Open when no keys are configured.

Interactive docs: /docs (OpenAPI).
"""

    @app.get("/llms.txt", include_in_schema=False)
    async def llms_txt():
        return PlainTextResponse(llms_txt_block)

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

    if static_ui_dir() is not None:
        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_static(full_path: str):
            return _dispatch_static(full_path)

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
