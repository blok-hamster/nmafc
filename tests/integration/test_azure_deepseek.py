import pytest
import tempfile
from pathlib import Path
from nmafc.integration.openai_provider import OpenAIProvider
from nmafc.integration.factory import create_embedding_provider
from nmafc.schemas.memory import DecayConfig
from nmafc.storage.config import NMafcConfig, StorageConfig
from nmafc.wrapper import NeuromorphicMemory


@pytest.mark.asyncio
async def test_azure_deepseek_v4_pro_live():
    endpoint = "https://support-6339-resource.openai.azure.com/openai/v1"
    key = "CpWxYfzO4DnX3GaCBPz3QhsobsPrGpTtfPSXa90x4sl3pU8OTpWCJQQJ99CHACfhMk5XJ3w3AAAAACOGXXQW"
    model = "DeepSeek-V4-Pro"

    llm = OpenAIProvider(model=model, api_key=key, base_url=endpoint)
    embedder = create_embedding_provider("ollama/nomic-embed-text")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        config = NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(Path(tmpdir) / "hot"),
                cold_uri=str(Path(tmpdir) / "cold.db"),
                embedding_dim=768,
            ),
            decay=DecayConfig(),
        )

        mem = NeuromorphicMemory(llm_provider=llm, embedding_provider=embedder, config=config)

        # Test turn 1
        resp1 = await mem.process_turn("Hi, I'm Alex and I'm a Senior AI Researcher.")
        assert resp1 is not None and len(resp1) > 0

        # Test turn 2 with related entity
        resp2 = await mem.process_turn("My brother Alex_Brother works at OpenAI.")
        assert resp2 is not None and len(resp2) > 0

        # Verify Hot RAM records captured state
        stats = mem.get_hot_stats()
        assert stats["count"] >= 1

        mem.close()
