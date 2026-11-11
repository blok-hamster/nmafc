"""from_openai() — zero config, one key needed.

Zero configuration besides an OpenAI key: gpt-4o-mini for extraction and
chat, text-embedding-3-small for embeddings.

Prerequisites:
    pip install nmafc[llm]
    export OPENAI_API_KEY=sk-...

Run:
    python examples/from_openai.py
"""

import asyncio

from nmafc import NeuromorphicMemory, NmafcError


async def main():
    try:
        async with await NeuromorphicMemory.from_openai() as mem:
            stored = await mem.remember([
                {"role": "user", "content": "I'm allergic to shellfish."},
            ])
            print(f"remember -> turn={stored.turn} ingested={stored.updates_ingested}")

            hits = await mem.recall("what can't I eat?")
            print(hits.context)
    except NmafcError as e:
        print(f"setup problem: {e}")
        print("fix: pip install nmafc[llm] && export OPENAI_API_KEY=sk-...")


if __name__ == "__main__":
    asyncio.run(main())
