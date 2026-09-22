"""Can the pieces single-hop is missing be reached at all, and at what width?

The real LoCoMo run put single-hop at 50.9% against RAG's 45.6%, the one
category where the lead is not distinguishable from noise. Splitting it by how
long the gold answer is shows it is not one category at all:

    gold 1-3 words        138 questions   we are right 68.1%
    gold 4-8 words        103 questions   we are right 35.9%
    gold 9+ words          40 questions   we are right 30.0%

So the short-answer half is healthy and the list half is not. On the questions
both arms fail, coverage of the gold's content words is 28%, and the reach
diagnosis buckets 71% of them as partial arrival: some of the list came, the
rest did not. That is a retrieval width problem on a specific question shape,
and width is free to screen.

Two knobs, screened here rather than argued about:

  facts and hydration    more of both, which costs tokens
  fact separation        `fact_overlap_max` drops a printed fact wholly
                         restated by a better-ranked one. On a list question
                         the near-duplicates ARE the list -- "recommended
                         grilled vegetables" and "recommended grilled chicken"
                         overlap heavily and are two different answers. This is
                         the one setting that could be actively hurting the
                         category it was tuned on a different one for.

Scores no answers and buys no generation. Query embeddings only.

Usage:
    python -u scripts/benchmarks/_screen_single_hop.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import defaultdict
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

from scripts.benchmarks._ab_budget import (  # noqa: E402
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks._screen_hydrate_pool import content_words  # noqa: E402

# rerank, facts, pool, grounding, scan, hydrate, overlap
CONFIGS = [
    ("shipping            ", 40, 20, 40, 0.03, 8, 6, 0.8),
    ("no fact separation  ", 40, 20, 40, 0.03, 8, 6, None),
    ("30 facts            ", 40, 30, 40, 0.03, 8, 6, 0.8),
    ("30 facts, no sep    ", 40, 30, 40, 0.03, 8, 6, None),
    ("30 facts, 10 turns  ", 40, 30, 40, 0.03, 8, 10, 0.8),
    ("rerank 60, 30 facts ", 60, 30, 60, 0.03, 8, 6, 0.8),
]


def coverage(gold: str, context: str) -> float:
    wanted = content_words(gold)
    if not wanted:
        return 1.0
    return len(wanted & content_words(context)) / len(wanted)


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = [r for r in rows if r["category"] == args.category]
    losses = [r for r in rows if not r["ours_ok"]]
    wins = [r for r in rows if r["ours_ok"]]
    # Long gold is the half that is failing, but the wins are carried through
    # every configuration too. A width that lifts the losses and drops the wins
    # is not an improvement, and reach screens that looked only at losses have
    # recommended exactly that before.
    long_losses = [r for r in losses if len(content_words(r["gold"])) >= 4]
    print(f"{args.category}: {len(losses)} losses ({len(long_losses)} with "
          f"4+ gold words), {len(wins)} wins, {len(CONFIGS)} configurations, "
          f"no generation\n")
    print(f"  {'configuration':22s} {'loss reach':>10s} {'long-gold':>10s} "
          f"{'win reach':>10s} {'tokens':>8s}")

    for label, rerank, facts, pool, grounding, scan, hydrate, overlap in CONFIGS:
        seen: dict[int, float] = {}
        widths: list[int] = []
        groups = defaultdict(list)
        for row in losses + wins:
            groups[row["conv"]].append(row)
        for name, group in groups.items():
            store = Path(args.run) / "stores" / f"{args.arm}__{name}"
            memory = open_memory(store, llm, embedder, args.hot, args.cold,
                                 rerank, args.max_hops, compact=True,
                                 hydrate=hydrate, facts=facts, dedupe=True,
                                 overlap=overlap)
            cfg = memory._router._config
            cfg.hydrate_pool = pool
            cfg.source_grounding = grounding
            cfg.hydrate_scan = scan
            cfg.scan_semantic = args.semantic
            cfg.scan_semantic_floor = args.semantic_floor
            try:
                turn = memory.current_turn + 1
                for row in group:
                    recs = await with_retries(
                        lambda q=row["question"]: memory._router.retrieve(q, turn))
                    ctx = memory._router.format_context(recs or [],
                                                        row["question"])
                    seen[id(row)] = coverage(row["gold"], ctx)
                    widths.append(len(ctx) // 4)
            finally:
                close_readonly(memory)

        def mean(subset):
            got = [seen[id(r)] for r in subset if id(r) in seen]
            return 100 * statistics.mean(got) if got else 0.0

        print(f"  {label} {mean(losses):9.1f}% {mean(long_losses):9.1f}% "
              f"{mean(wins):9.1f}% {statistics.mean(widths):8.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/locomo_real")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--results", default="C:/nmafc_ab/locomo_real.json")
    ap.add_argument("--category", default="single-hop")
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--semantic", type=float, default=12.0)
    ap.add_argument("--semantic-floor", type=float, default=0.30)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
