"""What width does a list question need, and can it be paid for out of hydration?

`nmafc.integration.list_shape.wants_list` splits single-hop cleanly: it fires on
129 of 281 questions, 94.6% of them have list-shaped gold, and that half scores
35.7% against the quiet half's 63.8%. So the weak half is identified for free.

What is not free is the fix. The category-wide screen found width is the only
lever that moves reach, and 30 facts with 10 hydrated turns costs 1,446 tokens
against shipping's 979. Applied to every question that fires anywhere in the
benchmark -- 16% of the scored set once temporal, multi-hop and open-domain are
counted -- that is about 75 tokens on the overall mean, which takes 964 over
1,000 and forfeits the constraint the whole stack is built to hold.

Hence the question this screens. A list question wants *many short things*, and
hydration buys *few long things*: a hydrated turn runs about 55 tokens against a
printed fact's 30. If the shape that helps is more facts and fewer turns, the
widening pays for itself and the gate is close to token-neutral. If it needs
both, the gate is not affordable and the honest answer is to say so.

Measured on the questions the gate fires on, because those are the only ones
whose width would change. The overall mean is then arithmetic rather than
another paid run:

    policy mean = shipped mean + fire rate * (wide width - shipped width)

Wins are carried through every configuration alongside losses. A width that
lifts the losses and drops the wins is not an improvement, and loss-only screens
have recommended exactly that here before.

Scores no answers and buys no generation. Query embeddings only.

Usage:
    python -u scripts/benchmarks/_screen_list_gate.py
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
from nmafc.integration.list_shape import wants_list  # noqa: E402

from scripts.benchmarks._ab_budget import (  # noqa: E402
    SCORED,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks._screen_hydrate_pool import content_words  # noqa: E402

# rerank, facts, pool, grounding, scan, hydrate, overlap
CONFIGS = [
    ("shipping             ", 40, 20, 40, 0.03, 8, 6, 0.8),
    ("30 facts, 6 turns    ", 40, 30, 40, 0.03, 8, 6, 0.8),
    ("30 facts, 3 turns    ", 40, 30, 40, 0.03, 8, 3, 0.8),
    ("30 facts, 0 turns    ", 40, 30, 40, 0.03, 8, 0, 0.8),
    ("35 facts, 2 turns    ", 40, 35, 40, 0.03, 8, 2, 0.8),
    ("40 facts, 0 turns    ", 40, 40, 40, 0.03, 8, 0, 0.8),
    ("30 facts, 3, no sep  ", 40, 30, 40, 0.03, 8, 3, None),
]

# The first set traded hydrated turns away for printed facts and every trade
# lost: 30 facts with 3 turns reaches 57.5% where 30 facts with 6 reaches 60.8%,
# and at 0 turns the widest fact list on offer is below shipping. So on this
# question shape the turns are carrying the answer and the facts are not, which
# is the opposite of what the token arithmetic wanted and worth taking
# seriously rather than screening around. This set trades the other way.
#
# `hydrate_scan` rises with the turn count. Scanning 8 turns to hydrate 10 is
# incoherent -- the pool it draws from is smaller than the number it wants --
# and leaving it at 8 would have measured the cap rather than the width.
TURN_HEAVY = [
    ("shipping             ", 40, 20, 40, 0.03, 8, 6, 0.8),
    ("20 facts, 8 turns    ", 40, 20, 40, 0.03, 12, 8, 0.8),
    ("20 facts, 10 turns   ", 40, 20, 40, 0.03, 16, 10, 0.8),
    ("16 facts, 10 turns   ", 40, 16, 40, 0.03, 16, 10, 0.8),
    ("12 facts, 10 turns   ", 40, 12, 40, 0.03, 16, 10, 0.8),
    ("16 facts, 8 turns    ", 40, 16, 40, 0.03, 12, 8, 0.8),
    ("12 facts, 12 turns   ", 40, 12, 40, 0.03, 16, 12, 0.8),
]

SETS = {"width": CONFIGS, "turns": TURN_HEAVY}


def coverage(gold: str, context: str) -> float:
    wanted = content_words(gold)
    if not wanted:
        return 1.0
    return len(wanted & content_words(context)) / len(wanted)


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    everything = json.loads(Path(args.results).read_text(encoding="utf-8"))
    scored = [r for r in everything if r["category"] in SCORED]
    fires = [r for r in scored if wants_list(r["question"])]
    rate = len(fires) / len(scored)
    # The gate is screened on the questions it fires on across every scored
    # category, not just the one it was designed for. A rule that fires on 12%
    # of open-domain spends open-domain's tokens whether or not it was aimed
    # there, and open-domain is the category with no lead to spend.
    if args.category:
        fires = [r for r in fires if r["category"] == args.category]
    losses = [r for r in fires if not r["ours_ok"]]
    wins = [r for r in fires if r["ours_ok"]]

    print(f"gate fires on {rate * 100:.1f}% of the {len(scored)} scored "
          f"questions; screening {len(fires)} of them "
          f"({len(losses)} losses, {len(wins)} wins), no generation")
    print(f"shipped overall mean is {args.shipped_mean} tokens\n")
    print(f"  {'configuration':23s} {'loss reach':>10s} {'win reach':>10s} "
          f"{'tokens':>7s} {'delta':>7s} {'overall':>8s}")

    baseline: float | None = None
    for (label, rerank, facts, pool, grounding, scan, hydrate,
         overlap) in SETS[args.set]:
        seen: dict[int, float] = {}
        widths: list[int] = []
        groups = defaultdict(list)
        for row in fires:
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

        width = statistics.mean(widths)
        if baseline is None:
            baseline = width
        delta = width - baseline
        print(f"  {label} {mean(losses):9.1f}% {mean(wins):9.1f}% "
              f"{width:7.0f} {delta:+7.0f} "
              f"{args.shipped_mean + rate * delta:8.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/locomo_real")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--results", default="C:/nmafc_ab/locomo_real.json")
    ap.add_argument("--category", default="",
                    help="restrict the screen to one category; the fire rate "
                         "used for the overall column is always all of them")
    ap.add_argument("--set", choices=sorted(SETS), default="width")
    ap.add_argument("--shipped-mean", type=float, default=964.0)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--semantic", type=float, default=12.0)
    ap.add_argument("--semantic-floor", type=float, default=0.30)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
