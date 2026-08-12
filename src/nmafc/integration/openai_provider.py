from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, UnifiedMemoryPayload

MEMORY_TOOL_NAME = "update_memory"

MEMORY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": MEMORY_TOOL_NAME,
        "description": (
            "Extract and record any new facts, state changes, or updates from "
            "the conversation. Call this whenever you identify information worth remembering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "description": "List of memory state updates extracted from this turn.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Unique key for the entity (e.g. 'user_allergy', 'medication_morning')",
                            },
                            "fact_content": {
                                "type": "string",
                                "description": "The factual content to remember.",
                            },
                            "memory_type": {
                                "type": "string",
                                "enum": ["CoreAnchor", "ActiveContext", "EphemeralState"],
                                "description": "Classification: CoreAnchor for permanent facts, ActiveContext for current state, EphemeralState for transient info.",
                            },
                            "overrides_entity": {
                                "type": "string",
                                "description": "Entity name of an existing memory this contradicts/replaces. Omit if not applicable.",
                            },
                        },
                        "required": ["entity_name", "fact_content", "memory_type"],
                    },
                },
            },
            "required": ["updates"],
        },
    },
}


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider with structured tool calling."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Install openai: pip install nmafc[llm]") from e

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=90.0)
        self._model = model

    async def chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        full_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}, *messages
        ]

        response: Any = None
        for attempt in range(10):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=cast(Any, full_messages),
                    tools=cast(Any, [MEMORY_TOOL_SCHEMA]),
                    tool_choice="auto",
                )
                break
            except Exception as exc:
                if attempt == 9:
                    raise exc
                await asyncio.sleep(min(30.0, 1.5 * (2.0 ** attempt)))

        if response is None:
            return "", []

        message = response.choices[0].message
        response_text: str = message.content or ""
        updates: list[MemoryStateUpdate] = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                fn = getattr(tool_call, "function", None)
                if fn is None:
                    continue
                if fn.name == MEMORY_TOOL_NAME:
                    try:
                        args = json.loads(fn.arguments)
                        payload = UnifiedMemoryPayload(**args)
                        updates.extend(payload.updates)
                    except (json.JSONDecodeError, ValueError):
                        continue

        return response_text, updates


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI-compatible text embedding provider."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Install openai: pip install nmafc[llm]") from e

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = 2048
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings
