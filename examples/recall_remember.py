"""Recall and remember, and nothing else.

Zero config beyond keys — environment variables decide everything.
Works with any provider; set the embedding route to `fastembed/` for
key-free local embeddings.

Prerequisites:
    pip install "nmafc[llm,fastembed]"
    export OPENAI_API_KEY=sk-...   # only needed for the LLM route

Run with:  python examples/recall_remember.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from nmafc import NeuromorphicMemory
from nmafc.storage.config import NMafcConfig, StorageConfig


async def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="nmafc_example_"))
    config = NMafcConfig(
        llm_provider_model="openai/gpt-4o-mini",
        embedding_provider_model="fastembed/BAAI/bge-small-en-v1.5",
        agent_id="example-agent",
        conversation_id="example-conv",
        storage=StorageConfig(
            hot_uri=str(tmp / "hot.lancedb"),
            cold_uri=str(tmp / "cold.db"),
            event_log_uri=str(tmp / "events.db"),
            embedding_dim=384,
        ),
    )

    # `from_config` is synchronous — no `await`.
    memory = NeuromorphicMemory.from_config(config=config)

    # remember: store a transcript's facts without generating a reply.
    stored = await memory.remember([
        {"role": "user", "content": "I just adopted a corgi named Miso."},
    ])
    print(f"remember -> turn={stored.turn} ingested={stored.updates_ingested}")

    # recall: retrieve a bounded, read-only context for a question.
    result = await memory.recall("what pet do I have?")
    print(f"recall -> {len(result.hits)} hits, "
          f"~{result.token_estimate} tokens")
    print(result.context)

    memory.close()

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
