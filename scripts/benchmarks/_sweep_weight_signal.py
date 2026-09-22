"""Does a graded memory weight help decide what to retrieve?

`weight_signal` adds an RRF boost proportional to a record's weight, and it has
never been above its default of 0.0 in any run. So every decay result this
project has produced reached the score through one channel only -- pruning,
which decides what stays in hot -- and none of it through ranking, which decides
what reaches the prompt. That makes this the question the whole decay line of
work rests on, and it is answerable without generating a single answer.

Run against stores prepared by `_inject_weights.py`, never the originals: on the
real stores 97.8% of weights are exactly 1.0, so every value of weight_signal
would return the same ranking and the sweep would be guaranteed to find nothing.

No generation and no judging. What is measured is whether the gold answer is
*reachable* in the retrieved facts, which is the ceiling any prompt could work
against, plus how much the ranking actually moved:

  reach strict   every content word of the gold appears, whole-word
  reach loose    same, ignoring tokens of two characters or fewer, so a gold of
                 "7 May 2023" is carried by a fact reading "6 May 2023"
  churn          share of retrieved slots that differ from the baseline ranking

Churn is the guard against a false null. A flat reach across every setting means
one of two very different things -- weight does not help, or weight never moved
anything -- and only churn separates them.

Reinforcement is deferred and never flushed, so even the injected copies are
left as they were.

Usage:
    python scripts/benchmarks/_sweep_weight_signal.py --stores /c/nmafc_ab/w_stores
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)
from nmafc.schemas.memory import DecayConfig  # noqa: E402
from nmafc.storage.config import NMafcConfig, StorageConfig  # noqa: E402
from nmafc.wrapper import NeuromorphicMemory  # noqa: E402

from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402

SCORED = ("single-hop", "temporal", "multi-hop", "open-domain")
STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "his", "her", "their",
        "is", "was", "and", "for", "with", "he", "she", "they", "it", "that"}


def words(text: str, min_len: int = 1) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", str(text).lower())
            if w not in STOP and len(w) >= min_len]


def reaches(gold: str, haystack: str, min_len: int) -> bool:
    want = words(gold, min_len)
    if not want:
        return False
    return all(
        re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", haystack)
        for w in want
    )


def open_memory(store: Path, llm, embedder, args, weight_signal: float):
    return NeuromorphicMemory(
        llm_provider=llm,
        embedding_provider=embedder,
        config=NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(store / "hot_lancedb"),
                cold_uri=str(store / "cold.db"),
            ),
            decay=DecayConfig(
                max_hops=args.max_hops,
                top_k=args.hot,
                fallback_keyword_limit=args.cold,
                rerank_top_k=args.budget,
                weight_signal=weight_signal,
                defer_reinforcement_writes=True,
            ),
        ),
    )


async def run(args: argparse.Namespace) -> None:
    values = [float(v) for v in args.values.split(",")]
    if 0.0 not in values:
        values.insert(0, 0.0)

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    gate = asyncio.Semaphore(args.concurrency)
    stores = Path(args.stores)

    # question key -> {weight_signal: (reach_strict, reach_loose, [fact ids])}
    seen: dict[tuple[str, str], dict[float, tuple[bool, bool, list[str]]]] = {}

    for conv in load_locomo():
        store = stores / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            print(f"  [skip] no store for {conv.sample_id}")
            continue
        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED]
        if args.limit:
            questions = questions[: args.limit]

        for value in values:
            memory = open_memory(store, llm, embedder, args, value)

            async def one(qa, value=value, memory=memory):
                async with gate:
                    records = await memory._router.retrieve(
                        qa.question, memory.current_turn + 1
                    )
                text = "\n".join(r.fact_content for r in records).lower()
                key = (conv.sample_id, qa.question)
                seen.setdefault(key, {})[value] = (
                    reaches(qa.answer, text, 1),
                    reaches(qa.answer, text, 3),
                    [r.id for r in records],
                )

            await asyncio.gather(*(one(qa) for qa in questions))
            print(f"    {conv.sample_id} weight_signal={value:<5} done", flush=True)

    rows = [v for v in seen.values() if len(v) == len(values)]
    n = len(rows) or 1
    print(f"\n  {len(rows)} questions retrieved under {len(values)} settings\n")
    print(f"  {'weight_signal':>14s} {'reach strict':>13s} {'reach loose':>12s} "
          f"{'churn vs 0':>11s}")
    for value in values:
        strict = sum(r[value][0] for r in rows) / n
        loose = sum(r[value][1] for r in rows) / n
        moved = 0.0
        for r in rows:
            base, now = r[0.0][2], r[value][2]
            if base:
                moved += len(set(base) - set(now)) / len(base)
        print(f"  {value:>14} {strict:>12.1%} {loose:>11.1%} {moved / n:>10.1%}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {f"{c}||{q}": {str(k): [a, b, ids] for k, (a, b, ids) in v.items()}
             for (c, q), v in seen.items()}, indent=2), encoding="utf-8")
        print(f"\n  -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stores", required=True,
                    help="output of _inject_weights.py, NOT the run's own stores")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--values", default="0.0,0.05,0.1,0.2,0.4")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
