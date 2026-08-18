"""Preflight check for a benchmark run.

Verifies, in order:
  1. Required env vars are present (never hardcoded).
  2. The LLM endpoint answers a plain chat request.
  3. The LLM emits TOOL CALLS -- without this, memory extraction silently
     returns nothing and the memory arms would score ~0 across the run.
  4. The embedding endpoint answers and reports its dimension.
  5. End-to-end: a real NeuromorphicMemory turn actually persists records.
  6. Measured per-call latency, so runtime estimates are grounded.

Usage:
    python -m scripts.preflight
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from nmafc.integration.factory import create_embedding_provider, create_llm_provider
from nmafc.schemas.memory import DecayConfig
from nmafc.storage.config import NMafcConfig, StorageConfig
from nmafc.wrapper import NeuromorphicMemory

REQUIRED_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "NMAFC_BENCH_PROVIDER",
    "NMAFC_BENCH_EMBEDDING",
]


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def check_env() -> tuple[str, str]:
    print("[1/6] Environment")
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        print(f"  FAIL missing env vars: {missing}")
        sys.exit(1)

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    provider = os.environ["NMAFC_BENCH_PROVIDER"]
    embedding = os.environ["NMAFC_BENCH_EMBEDDING"]

    print(f"  endpoint : {endpoint}")
    print(f"  api key  : {_mask(os.environ['AZURE_OPENAI_API_KEY'])}")
    print(f"  provider : {provider}")
    print(f"  embedding: {embedding}")
    return provider, embedding


async def check_chat(provider: str) -> float:
    print("\n[2/6] Chat completion")
    llm = create_llm_provider(provider)
    start = time.perf_counter()
    text, updates = await llm.chat_with_extraction(
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        system_prompt="You are a terse assistant.",
    )
    elapsed = time.perf_counter() - start
    print(f"  response: {text.strip()[:80]!r}")
    print(f"  latency : {elapsed:.2f}s")
    if not text.strip():
        print("  FAIL empty response")
        sys.exit(1)
    print("  PASS")
    return elapsed


async def check_tool_calls(provider: str) -> float:
    print("\n[3/6] Tool calling (CRITICAL)")
    llm = create_llm_provider(provider)
    from nmafc.integration.extractor import EXTRACTION_SYSTEM_PROMPT

    start = time.perf_counter()
    text, updates = await llm.chat_with_extraction(
        messages=[
            {
                "role": "user",
                "content": (
                    "My name is Marcus and I am severely allergic to shellfish. "
                    "I also have a team meeting at 3PM today."
                ),
            }
        ],
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
    )
    elapsed = time.perf_counter() - start
    print(f"  latency        : {elapsed:.2f}s")
    print(f"  updates emitted: {len(updates)}")
    for u in updates:
        print(f"    - [{u.memory_type}] {u.entity_name}: {u.fact_content[:60]}")

    if not updates:
        print("\n  FAIL: the model returned NO tool calls.")
        print("  Memory extraction depends entirely on tool use. Without it both")
        print("  memory arms would store nothing and score ~0. Do not start the run.")
        sys.exit(1)
    print("  PASS")
    return elapsed


async def check_embeddings(embedding: str) -> tuple[int, float]:
    print("\n[4/6] Embeddings")
    embedder = create_embedding_provider(embedding)
    start = time.perf_counter()
    vecs = await embedder.embed(["hello world", "a second string"])
    elapsed = time.perf_counter() - start
    dim = len(vecs[0]) if vecs else 0
    print(f"  vectors : {len(vecs)}")
    print(f"  dimension: {dim}")
    print(f"  latency : {elapsed:.2f}s for 2 texts")
    if dim == 0:
        print("  FAIL no embedding returned")
        sys.exit(1)
    print("  PASS")
    return dim, elapsed


async def check_end_to_end(provider: str, embedding: str) -> float:
    print("\n[5/6] End-to-end memory turn")
    llm = create_llm_provider(provider)
    embedder = create_embedding_provider(embedding)

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
        start = time.perf_counter()
        await mem.process_turn("My name is Marcus and I'm allergic to shellfish.")
        elapsed = time.perf_counter() - start

        stats = mem.get_hot_stats()
        print(f"  turn latency : {elapsed:.2f}s")
        print(f"  hot records  : {stats['count']}")
        print(f"  types        : {stats.get('types')}")
        records = mem._hot.get_all()
        for r in records:
            print(f"    - [{r.memory_type.value}] {r.entity_name}: {r.fact_content[:50]}")
        mem.close()

        if stats["count"] < 1:
            print("  FAIL nothing persisted to Hot RAM")
            sys.exit(1)
    print("  PASS")
    return elapsed


def report(chat_s: float, tool_s: float, turn_s: float, embed_s: float) -> None:
    print("\n[6/6] Runtime projection")
    print(f"  measured chat latency      : {chat_s:.2f}s")
    print(f"  measured extraction latency: {tool_s:.2f}s")
    print(f"  measured full turn latency : {turn_s:.2f}s")
    print(f"  measured embed latency     : {embed_s:.2f}s / 2 texts")

    extraction_calls = (2951 + 5479) * 2
    answer_calls = (1986 + 500) * 4
    judge_calls = (1986 + 500) * 4
    total = extraction_calls + answer_calls + judge_calls
    print(f"\n  planned calls: {total:,}")
    print(f"    extraction : {extraction_calls:,}")
    print(f"    answering  : {answer_calls:,}")
    print(f"    judging    : {judge_calls:,}")

    avg = (tool_s + chat_s) / 2
    seq_hours = total * avg / 3600
    print(f"\n  sequential estimate : {seq_hours:.1f} hrs")
    for c in (8, 12, 16, 24):
        print(f"  at concurrency {c:>2}    : {seq_hours / c:.1f} hrs")


async def main() -> None:
    print("=" * 70)
    print("NMAFC BENCHMARK PREFLIGHT")
    print("=" * 70)
    provider, embedding = check_env()
    chat_s = await check_chat(provider)
    tool_s = await check_tool_calls(provider)
    _dim, embed_s = await check_embeddings(embedding)
    turn_s = await check_end_to_end(provider, embedding)
    report(chat_s, tool_s, turn_s, embed_s)
    print("\n" + "=" * 70)
    print("PREFLIGHT PASSED — safe to start the benchmark")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
