"""OpenAI middleware: memory without changing your app.

Wrap an OpenAI client so every completion is enriched with recalled memories
and the finished conversation is remembered — streaming included.

Prerequisites:
    pip install nmafc[llm] openai
    export OPENAI_API_KEY=sk-...

Run:
    python examples/openai_middleware.py
"""

import asyncio

from openai import AsyncOpenAI

from nmafc import NeuromorphicMemory
from nmafc.proxy import MemoryOpenAI


async def main():
    mem = await NeuromorphicMemory.from_openai()
    client = MemoryOpenAI(mem, client=AsyncOpenAI())

    # Recall is injected automatically as a system preamble, then the
    # finished turn is remembered — same call, both directions.
    reply = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's my dog's name?"}],
    )
    print(reply.choices[0].message.content)

    await mem.close()


if __name__ == "__main__":
    asyncio.run(main())
