from __future__ import annotations

import json
from typing import Any, cast

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, UnifiedMemoryPayload

MEMORY_TOOL_NAME = "update_memory"

MEMORY_TOOL_SCHEMA: dict[str, Any] = {
    "name": MEMORY_TOOL_NAME,
    "description": (
        "Extract and record any new facts, state changes, or updates from "
        "the conversation. Call this whenever you identify information worth remembering."
    ),
    "input_schema": {
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
}


class AnthropicProvider(LLMProvider):
    """Anthropic-based LLM provider with tool use."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError("Install anthropic: pip install nmafc[llm]") from e

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=cast(Any, messages),
            tools=cast(Any, [MEMORY_TOOL_SCHEMA]),
        )

        response_text = ""
        updates: list[MemoryStateUpdate] = []

        for block in response.content:
            if block.type == "text":
                response_text += block.text
            elif block.type == "tool_use" and block.name == MEMORY_TOOL_NAME:
                try:
                    tool_input = cast(dict[str, Any], block.input)
                    payload = UnifiedMemoryPayload(**tool_input)
                    updates.extend(payload.updates)
                except (ValueError, TypeError):
                    continue

        return response_text, updates


class AnthropicEmbedding(EmbeddingProvider):
    """Embedding provider using OpenAI (Anthropic has no native embedding API)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Install openai for embeddings: pip install nmafc[llm]") from e

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]
