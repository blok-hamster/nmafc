"""Phase 1 exit example: recall and remember, and nothing else.

Requires only an OPENAI_API_KEY (and, if you prefer files, a config produced
by `nmafc init` -- but none is needed here).

Run with:  python examples/recall_remember.py
"""

from __future__ import annotations

import asyncio

from nmafc.wrapper import NeuromorphicMemory


async def main() -> None:
    # Zero config beyond keys: environment only, library defaults elsewhere.
    memory = NeuromorphicMemory.from_env()

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


if __name__ == "__main__":
    asyncio.run(main())
