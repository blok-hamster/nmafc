"""OpenAI-compatible HTTP proxy: `/v1/chat/completions` with memory.

Reuses the `MemoryOpenAI` interposition seam (recall -> inject -> forward ->
remember) behind a FastAPI endpoint, so an existing OpenAI app can add memory
by changing only its base URL:

    client = openai.OpenAI(base_url="http://localhost:8000")

The standard OpenAI request body is forwarded to an upstream (default
`https://api.openai.com/v1`); the proxy also accepts `agent_id` /
`conversation_id` in the body for tenant scoping. Streaming is forwarded as
SSE and extraction runs on the reassembled transcript once the stream ends.
"""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nmafc.proxy import MemoryOpenAI
from nmafc.web.deps import get_memory

router = APIRouter(tags=["proxy"])

_PASSTHROUGH = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "n",
    "response_format",
}


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[dict] = Field(..., min_length=1)
    stream: bool = False
    agent_id: str | None = None
    conversation_id: str | None = None

    def passthrough(self) -> dict:
        out = {}
        for name in _PASSTHROUGH:
            value = getattr(self, name, None)
            if value is not None:
                out[name] = value
        return out


def upstream_client() -> MemoryOpenAI:
    """Build the memory-aware upstream client once per request scope.

    Upstream base URL and key come from `NMAFC_PROXY_UPSTREAM_BASE_URL` /
    `NMAFC_PROXY_UPSTREAM_API_KEY`, falling back to `OPENAI_API_KEY` (default
    upstream: `https://api.openai.com/v1`). Tests monkeypatch this to point at
    a mock transport.
    """
    import openai

    base_url = os.environ.get("NMAFC_PROXY_UPSTREAM_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("NMAFC_PROXY_UPSTREAM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="No upstream API key: set OPENAI_API_KEY or NMAFC_PROXY_UPSTREAM_API_KEY",
        )
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    return client


async def _run(memory, req: ChatCompletionRequest):
    """Execute the interposed completion against the upstream."""
    memory_openai = MemoryOpenAI(memory, client=upstream_client())
    kwargs: dict = {
        "model": req.model,
        "messages": req.messages,
        "stream": req.stream,
        "memory_agent_id": req.agent_id,
        "memory_conversation_id": req.conversation_id,
        **req.passthrough(),
    }
    completion = await memory_openai.chat.completions.create(**kwargs)
    if req.stream:
        # Re-wrap the recorded call: MemoryOpenAI's async stream returns an
        # async generator that runs the remember() tail on exhaustion.
        async def forward():
            async for chunk in completion:
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(forward(), media_type="text/event-stream")
    return completion.model_dump()


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    memory = get_memory(
        agent_id=req.agent_id or "default",
        conversation_id=req.conversation_id or "default",
    )
    return await _run(memory, req)
