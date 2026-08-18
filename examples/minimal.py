"""Minimal NMAFC example — 10 lines to get started.

Prerequisites:
    pip install nmafc[llm]
    export OPENAI_API_KEY=sk-...

Run:
    python examples/minimal.py
"""

import asyncio
from nmafc.wrapper import NeuromorphicMemory


async def main():
    async with await NeuromorphicMemory.from_config() as mem:
        # Turn 1: Store a fact
        response = await mem.process_turn(
            "My name is Alice and I'm a software engineer."
        )
        print(f"Turn 1: {response}\n")

        # Turn 2: Memory remembers
        response = await mem.process_turn(
            "What's my name and job?"
        )
        print(f"Turn 2: {response}\n")

        # Check memory state
        stats = mem.get_hot_stats()
        print(f"Records: {stats['count']}, Types: {stats['types']}")


if __name__ == "__main__":
    asyncio.run(main())
