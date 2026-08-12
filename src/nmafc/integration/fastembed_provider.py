"""Fast ONNX CPU Embedding Provider using Qdrant FastEmbed.
0% GPU load, 0% Ollama overhead, sub-5ms local embedding latency.
Uses asyncio.to_thread for non-blocking CPU ONNX matrix calculations.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from nmafc.integration.base import EmbeddingProvider

if TYPE_CHECKING:
    from fastembed import TextEmbedding  # type: ignore[import-not-found]


class FastEmbedProvider(EmbeddingProvider):
    """Fast ONNX CPU Embeddings (BAAI/bge-small-en-v1.5, dim=384)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding as _TextEmbedding  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Install fastembed for ONNX embeddings: pip install fastembed"
            ) from e

        self._model_name = model_name
        self._model: TextEmbedding = _TextEmbedding(model_name=model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def _sync_embed() -> list[list[float]]:
            return [vec.tolist() for vec in self._model.embed(texts)]

        return await asyncio.to_thread(_sync_embed)
