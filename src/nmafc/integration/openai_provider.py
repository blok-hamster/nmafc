from __future__ import annotations

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
    """OpenAI-compatible LLM provider with structured tool calling.

    Works with any provider exposing an OpenAI-compatible API:
    OpenAI, Groq, OpenRouter, Together, Ollama, LM Studio, vLLM, etc.

    Args:
        model: Model identifier (e.g. "gpt-4o-mini", "llama3", "mixtral-8x7b")
        api_key: API key. Falls back to OPENAI_API_KEY env var.
        base_url: API base URL. Set this for non-OpenAI providers:
            - Groq: "https://api.groq.com/openai/v1"
            - OpenRouter: "https://openrouter.ai/api/v1"
            - Together: "https://api.together.xyz/v1"
            - Ollama: "http://localhost:11434/v1"
            - LM Studio: "http://localhost:1234/v1"
            - vLLM: "http://localhost:8000/v1"
    """

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

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        full_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}, *messages
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(Any, full_messages),
            tools=cast(Any, [MEMORY_TOOL_SCHEMA]),
            tool_choice="auto",
        )

        message = response.choices[0].message
        response_text = message.content or ""
        updates: list[MemoryStateUpdate] = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if not hasattr(tool_call, "function") or tool_call.function is None:
                    continue
                if tool_call.function.name == MEMORY_TOOL_NAME:
                    try:
                        args = json.loads(tool_call.function.arguments)
                        payload = UnifiedMemoryPayload(**args)
                        updates.extend(payload.updates)
                    except (json.JSONDecodeError, ValueError):
                        continue

        return response_text, updates


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI-compatible text embedding provider.

    Works with any provider exposing an OpenAI-compatible embeddings API:
    OpenAI, Together, Ollama, LM Studio, vLLM, etc.

    Args:
        model: Embedding model identifier.
        api_key: API key.
        base_url: API base URL for non-OpenAI providers.
    """

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
