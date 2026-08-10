from __future__ import annotations

import json
from typing import Any, cast

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, MemoryType, UnifiedMemoryPayload

BEDROCK_TOOL_SCHEMA: dict[str, Any] = {
    "toolSpec": {
        "name": "update_memory",
        "description": (
            "Extract and record any new facts, state changes, or updates from "
            "the conversation. Call this whenever you identify information worth remembering."
        ),
        "inputSchema": {
            "json": {
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
                                    "description": "Unique key for the entity.",
                                },
                                "fact_content": {
                                    "type": "string",
                                    "description": "The factual content to remember.",
                                },
                                "memory_type": {
                                    "type": "string",
                                    "enum": ["CoreAnchor", "ActiveContext", "EphemeralState"],
                                    "description": "Classification of the memory.",
                                },
                                "overrides_entity": {
                                    "type": "string",
                                    "description": "Entity name this contradicts/replaces.",
                                },
                                "related_entities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Entity names linked to this fact for graph spreading activation (e.g. ['spouse_james', 'brother_david']).",
                                },
                            },
                            "required": ["entity_name", "fact_content", "memory_type"],
                        },
                    },
                },
                "required": ["updates"],
            }
        },
    }
}


class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider using the Converse API.

    Supports Claude, Llama, Mistral, Titan, and other models available on Bedrock.

    Env vars:
        AWS_REGION: defaults to "us-east-1"
        AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY: standard AWS credentials
        (or use IAM roles / AWS profiles)
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
    ) -> None:
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "Install boto3 for Bedrock support: pip install boto3"
            ) from e

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
        )
        self._model_id = model_id

    async def chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_chat_with_extraction, messages, system_prompt
        )

    def _sync_chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        bedrock_messages = []
        for msg in messages:
            bedrock_messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}],
            })

        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": system_prompt}],
            messages=bedrock_messages,
            toolConfig={"tools": [BEDROCK_TOOL_SCHEMA]},
        )

        output = response["output"]["message"]
        response_text = ""
        updates: list[MemoryStateUpdate] = []

        for block in output.get("content", []):
            if "text" in block:
                response_text += block["text"]
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                if tool_use.get("name") == "update_memory":
                    try:
                        payload = UnifiedMemoryPayload(**tool_use["input"])
                        updates.extend(payload.updates)
                    except (ValueError, KeyError):
                        continue

        return response_text, updates


class BedrockAnthropicProvider(LLMProvider):
    """AWS Bedrock provider using the Anthropic SDK's native Bedrock client.

    This is the preferred provider for Claude models on Bedrock. It uses the
    Anthropic SDK's AsyncAnthropicBedrock client which supports Bedrock API keys
    directly and the standard Anthropic tool_use format.

    Env vars:
        AWS_REGION: defaults to "us-east-1"
        ANTHROPIC_API_KEY_BEDROCK: Bedrock API key (ABSK... format)
        Or standard AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        api_key: str | None = None,
    ) -> None:
        try:
            from anthropic import AsyncAnthropicBedrock
        except ImportError as e:
            raise ImportError(
                "Install anthropic: pip install nmafc[llm]"
            ) from e

        self._client = AsyncAnthropicBedrock(
            aws_region=region,
            api_key=api_key,
        )
        self._model_id = model_id

    async def chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        from nmafc.integration.anthropic_provider import MEMORY_TOOL_NAME, MEMORY_TOOL_SCHEMA

        response = await self._client.messages.create(
            model=self._model_id,
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


class BedrockEmbedding(EmbeddingProvider):
    """AWS Bedrock embedding provider.

    Supports Amazon Titan Embeddings and Cohere Embed models on Bedrock.
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
    ) -> None:
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "Install boto3 for Bedrock support: pip install boto3"
            ) from e

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
        )
        self._model_id = model_id

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_embed, texts)

    def _sync_embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for text in texts:
            if "titan" in self._model_id.lower():
                body = json.dumps({"inputText": text})
            elif "cohere" in self._model_id.lower():
                body = json.dumps({
                    "texts": [text],
                    "input_type": "search_document",
                })
            else:
                body = json.dumps({"inputText": text})

            response = self._client.invoke_model(
                modelId=self._model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())

            if "embedding" in result:
                all_embeddings.append(result["embedding"])
            elif "embeddings" in result:
                all_embeddings.append(result["embeddings"][0])

        return all_embeddings
