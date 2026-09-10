"""How much accuracy is the prompt budget throwing away?

The run answers 63.3% and the gold answer only reaches the model 72% of the
time, so conversion is already 88% and the ceiling is what caps the arm. The
arm sends 655 tokens per question against RAG's 1461: there is room to more
than double the context and still read less than the baseline does.

Retrieval currently pulls 10 records from Hot RAM, up to 20 from the archive
and whatever the graph walk adds, then `rerank_top_k` truncates to 20 before
anything is rendered. A third of the pool is discarded unseen.

Measuring what that costs does not need a rerun, or even a single generated
answer. `reciprocal_rank_fusion` scores every candidate and returns
`ranked[:top_k]`, so the prompt at budget 20 is a strict prefix of the prompt
at budget 60. One retrieval per question therefore yields the entire budget
curve at once: retrieve wide, then score each prefix. The whole sweep costs one
embedding pass.

Two things are varied, and they are not the same thing:

  * the prompt budget, which is free to sweep as above
  * the candidate pool feeding it, which is not, because widening the pool
    changes what retrieval found and needs its own pass

Running with the stock pool answers "is the reranker discarding answers it
already holds". Running with a wider pool answers "is retrieval failing to find
them at all". The first is a one-line config change if it pays; the second is
not.

"In context" is gold keyword overlap, the same crude test the other
diagnostics use. It over-reports, so a gain here is a necessary condition for
better accuracy rather than a promise of it. A flat curve is the conclusive
direction: if more context does not carry more answers, the budget is not the
cap and this rules the lever out for the price of one pass.

Usage:
    python scripts/benchmarks/_sweep_context_budget.py --run scripts/benchmarks/results/full_v3
    python scripts/benchmarks/_sweep_context_budget.py --run ... --hot-top-k 25 --cold-budget 40
"""

from __future__ import annotations

import argparse
import asyncio
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

from scripts.benchmarks._ab_budget import close_readonly  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402

SCORED = ("single-hop", "temporal", "multi-hop", "open-domain")

STOP = set(
    "the a an of in on at to for is was were and or with what which who when "
    "where how did does do had has have her his its their this that".split()
)

# Roughly four characters to the token, the same conversion used to report the
# run's 655-token context against RAG's 1461.
CHARS_PER_TOKEN = 4


def keywords(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9']+", str(text).lower())
        if len(w) > 2 and w not in STOP
    }


def overlap(gold: str, haystack: str) -> float:
    kw = keywords(gold)
    if not kw:
        return 0.0
    low = haystack.lower()
    return sum(1 for w in kw if w in low) / len(kw)


async def retrieve_with_retry(router, question: str, turn: int, attempts: int = 5):
    for attempt in range(attempts):
        try:
            return await router.retrieve(question, turn)
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"      [retry {attempt + 1} in {wait}s] {type(exc).__name__}")
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


def open_memory(store: Path, llm, embedder, args) -> NeuromorphicMemory:
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
                top_k=args.hot_top_k,
                fallback_keyword_limit=args.cold_budget,
                # Retrieved wide once; every budget is read off as a prefix of
                # this ranking, so the truncation the run actually used never
                # happens here.
                rerank_top_k=max(args.budgets),
                # This reads the run's own stores. Reinforcement rewrites every
                # record it returns, so the writeback is buffered and never
                # flushed, leaving both the facts and their weights untouched.
                defer_reinforcement_writes=True,
            ),
        ),
    )


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    budgets = sorted(args.budgets)

    hits = {b: 0 for b in budgets}
    chars = {b: 0 for b in budgets}
    per_cat = {c: {"n": 0, **{b: 0 for b in budgets}} for c in SCORED}
    pool_sizes: list[int] = []
    total = 0
    gained: list[tuple[int, str, str]] = []

    print(f"pool : hot {args.hot_top_k}, archive {args.cold_budget}, "
          f"hops {args.max_hops}")
    print(f"budgets swept: {budgets}\n")

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            print(f"  [skip] no store for {conv.sample_id}")
            continue
        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED]
        if args.limit:
            questions = questions[: args.limit]
        if not questions:
            continue

        memory = open_memory(store, llm, embedder, args)
        try:
            turn = memory.current_turn + 1
            local = {b: 0 for b in budgets}
            for qa in questions:
                records = await retrieve_with_retry(
                    memory._router, qa.question, turn
                )
                pool_sizes.append(len(records))
                total += 1
                cat = per_cat[qa.category_name]
                cat["n"] += 1

                previous = False
                for b in budgets:
                    context = memory._router.format_context(records[:b])
                    hit = overlap(qa.answer, context) >= args.threshold
                    hits[b] += hit
                    chars[b] += len(context)
                    cat[b] += hit
                    local[b] += hit
                    if hit and not previous and b != budgets[0] and len(gained) < 10:
                        gained.append((b, qa.question, str(qa.answer)))
                    previous = previous or hit
            shown = " ".join(f"{b}:{local[b]}" for b in budgets)
            print(f"  {conv.sample_id}: n={len(questions)}  {shown}")
        finally:
            close_readonly(memory)

    if not total:
        print("\nnothing measured")
        return

    print(f"\n{'=' * 66}")
    print(f"questions            : {total}")
    print(f"mean candidates found: {sum(pool_sizes) / len(pool_sizes):.1f}")
    print(f"  (a budget above this cannot add anything)\n")
    print(f"  {'budget':>7s} {'gold in context':>16s} {'vs top-20':>10s} "
          f"{'context tokens':>15s}")
    base = hits[20] if 20 in hits else hits[budgets[0]]
    for b in budgets:
        print(f"  {b:7d} {hits[b] / total:15.1%} {hits[b] - base:+10d} "
              f"{chars[b] / total / CHARS_PER_TOKEN:15.0f}")

    print(f"\n  {'category':14s} {'n':>5s} " +
          " ".join(f"{b:>7d}" for b in budgets))
    for name in SCORED:
        c = per_cat[name]
        if not c["n"]:
            continue
        cells = " ".join(f"{c[b] / c['n']:6.1%} " for b in budgets)
        print(f"  {name:14s} {c['n']:5d} {cells}")

    if gained:
        print("\n  answers only a wider budget reaches:")
        for b, q, g in gained:
            print(f"    [needs {b}] {q[:74]}")
            print(f"               gold: {g[:66]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--budgets", type=int, nargs="*",
                    default=[10, 20, 30, 40, 60])
    ap.add_argument("--hot-top-k", type=int, default=10)
    ap.add_argument("--cold-budget", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
