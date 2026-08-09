from __future__ import annotations

from abc import ABC, abstractmethod

from nmafc.schemas.memory import MemoryStateUpdate


class LLMProvider(ABC):
    """Abstract interface for LLM providers that support tool calling."""

    @abstractmethod
    async def chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        """Generate a response and extract memory state updates via tool calling.

        Returns:
            Tuple of (assistant_response_text, list_of_memory_updates)
        """
        ...


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        ...

    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]
