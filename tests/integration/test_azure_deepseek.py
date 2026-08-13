import os
import tempfile
from pathlib import Path

import pytest

from nmafc.integration.factory import create_embedding_provider
from nmafc.integration.openai_provider import OpenAIProvider
from nmafc.schemas.memory import DecayConfig
from nmafc.storage.config import NMafcConfig, StorageConfig
from nmafc.wrapper import NeuromorphicMemory

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@pytest.mark.asyncio
async def test_azure_deepseek_v4_pro_live():
    """Live smoke test against an Azure-hosted DeepSeek deployment.

    Credentials come from the environment only. Skips when unset so the
    suite stays green without credentials.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    model = os.environ.get("AZURE_DEPLOYMENT_NAME", "DeepSeek-V4-Pro")
    embedding_model = os.environ.get(
        "NMAFC_BENCH_EMBEDDING", "ollama/nomic-embed-text"
    )

    if not endpoint or not key:
        pytest.skip(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY to run this test"
        )

    llm = OpenAIProvider(model=model, api_key=key, base_url=endpoint)
    embedder = create_embedding_provider(embedding_model)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        config = NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(Path(tmpdir) / "hot"),
                cold_uri=str(Path(tmpdir) / "cold.db"),
            ),
            decay=DecayConfig(),
        )

        mem = NeuromorphicMemory(
            llm_provider=llm, embedding_provider=embedder, config=config
        )

        resp1 = await mem.process_turn("Hi, I'm Alex and I'm a Senior AI Researcher.")
        assert resp1 is not None and len(resp1) > 0

        resp2 = await mem.process_turn("My brother Alex_Brother works at OpenAI.")
        assert resp2 is not None and len(resp2) > 0

        stats = mem.get_hot_stats()
        assert stats["count"] >= 1

        mem.close()
