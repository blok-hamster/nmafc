from __future__ import annotations

import json
from typing import Any

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.integration.openai_provider import MEMORY_TOOL_NAME, MEMORY_TOOL_SCHEMA
from nmafc.schemas.memory import MemoryStateUpdate, UnifiedMemoryPayload


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI LLM provider.

    Uses the openai SDK's AzureOpenAI client with deployment-based routing.

    Env vars:
        AZURE_OPENAI_ENDPOINT: e.g. "https://my-resource.openai.azure.com"
        AZURE_OPENAI_API_KEY: your Azure API key
        AZURE_OPENAI_API_VERSION: defaults to "2024-10-21"
    """

    def __init__(
        self,
        deployment: str,
        api_key: str | None = None,
        endpoint: str = "",
        api_version: str = "2024-10-21",
    ) -> None:
        try:
            from openai import AsyncAzureOpenAI
        except ImportError as e:
            raise ImportError("Install openai: pip install nmafc[llm]") from e

        if not endpoint:
            raise ValueError(
                "Azure endpoint required. Set AZURE_OPENAI_ENDPOINT or pass endpoint=."
            )

        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._deployment = deployment

    async def chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        from typing import Any, cast

        full_messages = [{"role": "system", "content": system_prompt}, *messages]

        response = await self._client.chat.completions.create(
            model=self._deployment,
            messages=cast(Any, full_messages),
            tools=cast(Any, [MEMORY_TOOL_SCHEMA]),
            tool_choice="auto",
        )

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


class AzureOpenAIEmbedding(EmbeddingProvider):
    """Azure OpenAI embedding provider."""

    def __init__(
        self,
        deployment: str,
        api_key: str | None = None,
        endpoint: str = "",
        api_version: str = "2024-10-21",
    ) -> None:
        try:
            from openai import AsyncAzureOpenAI
        except ImportError as e:
            raise ImportError("Install openai: pip install nmafc[llm]") from e

        if not endpoint:
            raise ValueError(
                "Azure endpoint required. Set AZURE_OPENAI_ENDPOINT or pass endpoint=."
            )

        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._deployment = deployment

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = 2048
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._client.embeddings.create(
                model=self._deployment,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings
