"""Custom provider example — implement your own LLM and embedding providers.

This example shows how to plug in any backend by implementing
the LLMProvider and EmbeddingProvider interfaces.

Run:
    python examples/custom_provider.py
"""

from __future__ import annotations

import asyncio
from nmafc.integration.base import LLMProvider, EmbeddingProvider
from nmafc.wrapper import NeuromorphicMemory
from nmafc.schemas.memory import MemoryStateUpdate


class MyLLMProvider(LLMProvider):
    """Replace this with your actual LLM backend."""

    async def chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        # Call your LLM here (vLLM, LiteLLM, custom API, etc.)
        # For demo: return a mock response with no extractions
        last_msg = messages[-1]["content"] if messages else ""
        response = f"[MyLLM] Echo: {last_msg}"
        return response, []


class MyEmbeddingProvider(EmbeddingProvider):
    """Replace this with your actual embedding backend."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Return deterministic mock vectors (dimension=8 for demo)
        return [[0.1 + i * 0.01 for i in range(8)] for _ in texts]


async def main():
    llm = MyLLMProvider()
    embedder = MyEmbeddingProvider()

    async with await NeuromorphicMemory.from_providers(
        llm_provider=llm,
        embedding_provider=embedder,
    ) as mem:
        # Inject facts manually (bypasses LLM extraction)
        await mem.ingest_updates([
            MemoryStateUpdate(
                entity_name="user_name",
                fact_content="User is named Bob",
                memory_type="CoreAnchor",
            ),
            MemoryStateUpdate(
                entity_name="current_project",
                fact_content="Building a custom memory system",
                memory_type="ActiveContext",
                related_entities=["user_name"],
            ),
        ])

        print(f"Injected 2 facts. Turn: {mem.current_turn}")
        print(f"Stats: {mem.get_hot_stats()}")

        # Query (MyLLM won't extract anything, but retrieval still works)
        response = await mem.process_turn("Tell me about Bob")
        print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
