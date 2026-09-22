"""Does ranking facts by their source turns move the answering record up?

`_screen_ranking.py` established the instrument and paid the only bill: for each
open-domain loss it asked the judge once which stored record states the gold
answer, and wrote the text down. 69 of the 116 losses have such a record. From
there, any retrieval change can be scored by looking for that text in the pool,
with no generation at all.

This runs that instrument over `source_grounding`: same store, same budget,
differing only in the weight given to how well a fact's source turn matches the
question, by BM25 scored within the candidate pool.

The weight is the whole question, and the scale is not arbitrary. Two facts
adjacent within one fused list differ by about 0.0003; belonging to a whole
extra fused list is worth about 0.016. The first shape of this mechanism was a
sixth list, which is the second number, and the screen below is what showed that
to be far too much.

Two thresholds, because they buy different things:

    top 6     the record is printed as a fact in its own right
    top 20    the record's source turn is hydrated, which is where the
              qualifier the extractor dropped actually lives

The second is the one this mechanism is aimed at. A question that turns on the
word *gripping* cannot be answered from a fact that does not contain it, and no
reordering puts the word into the fact -- what reordering can do is get the turn
that still holds it into the hydration window.

And the half that a gain-only screen would not show: **churn on the questions we
already win.** Reordering is not free even when it costs no tokens. If the top
six facts of a question we currently answer correctly are a different six
afterwards, the gain on the losses is being paid for somewhere, and this reports
it before a generation run does.

Reaching a record is still not answering with it. Anything that gains here needs
a paired generation A/B, both arms in one session.

Usage:
    python -u scripts/benchmarks/_screen_grounding.py --limit 5
    python -u scripts/benchmarks/_screen_grounding.py --weights 0.001,0.003

`--weights 0` is the control worth running before trusting any churn number: it
compares the off arm against itself, and it reports 0% churn, which is what says
retrieval is reproducible across handles and the churn measured elsewhere is the
mechanism rather than the instrument.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
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

from scripts.benchmarks._ab_budget import (  # noqa: E402
    close_readonly,
    open_memory,
    with_retries,
)


def memory_for(store, llm, embedder, args, weight: float):
    """A read-only handle, optionally with source grounding switched on.

    The weight is set on the config after construction rather than threaded
    through `open_memory`, which every other screen shares. Adding a keyword
    there for a setting that is zero everywhere else would change the signature
    of the one function all the paid measurements were taken through.
    """
    memory = open_memory(store, llm, embedder, args.hot, args.cold,
                         args.budget, args.max_hops, compact=True, hydrate=0)
    memory._router._config.source_grounding = weight
    return memory


async def ranks_for(memory, questions: list[str], targets: list[str]
                    ) -> list[int | None]:
    """Where each question's answering record lands, or None if unreached."""
    turn = memory.current_turn + 1
    out: list[int | None] = []
    for question, target in zip(questions, targets):
        pool = await with_retries(
            lambda q=question: memory._router.retrieve(q, turn))
        surfaced = [r.fact_content for r in (pool or [])]
        out.append(surfaced.index(target) if target in surfaced else None)
    return out


async def top_facts(memory, questions: list[str], depth: int
                    ) -> list[list[str]]:
    turn = memory.current_turn + 1
    out: list[list[str]] = []
    for question in questions:
        pool = await with_retries(
            lambda q=question: memory._router.retrieve(q, turn))
        out.append([r.fact_content for r in (pool or [])][:depth])
    return out


def report_ranks(label: str, ranks: list[int | None], base: list[int | None],
                 args) -> None:
    n = len(ranks)
    found = [r for r in ranks if r is not None]
    top_wide = sum(1 for r in found if r < args.hydrate_depth)
    top_narrow = sum(1 for r in found if r < args.fact_depth)
    # Movement is reported separately from level because a mechanism can leave
    # the totals untouched while shuffling which questions make it, and that is
    # not the same result at all.
    up = sum(1 for a, b in zip(ranks, base)
             if a is not None and (b is None or a < b))
    down = sum(1 for a, b in zip(ranks, base)
               if b is not None and (a is None or a > b))
    print(f"  {label:<12}{len(found):>6}{100 * len(found) / n:>5.0f}%"
          f"{top_wide:>7}{100 * top_wide / n:>5.0f}%"
          f"{top_narrow:>7}{100 * top_narrow / n:>5.0f}%"
          f"{up:>7}{down:>7}")


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no store at {store}")

    target_file = Path(args.targets)
    if not target_file.is_file():
        raise SystemExit(
            f"no {args.targets}. Run `_screen_ranking.py --stage targets` "
            f"first; that is the pass that costs money.")
    rows = [r for r in json.loads(target_file.read_text(encoding="utf-8"))
            if r.get("target")]
    if args.limit:
        rows = rows[: args.limit]
    questions = [r["question"] for r in rows]
    targets = [r["target"] for r in rows]

    settings = [float(x) for x in args.weights.split(",")]
    print(f"{len(rows)} answerable open-domain losses, "
          f"{len(settings) + 1} configurations, no generation\n")
    print(f"  {'':<12}{'in pool':>12}{'top ' + str(args.hydrate_depth):>12}"
          f"{'top ' + str(args.fact_depth):>12}{'up':>7}{'down':>7}")

    memory = memory_for(store, llm, embedder, args, 0.0)
    try:
        base = await ranks_for(memory, questions, targets)
        won_before = (await top_facts(memory, args.won, args.fact_depth)
                      if args.won else [])
    finally:
        close_readonly(memory)
    report_ranks("off", base, base, args)

    for weight in settings:
        memory = memory_for(store, llm, embedder, args, weight)
        try:
            ranks = await ranks_for(memory, questions, targets)
            won_after = (await top_facts(memory, args.won, args.fact_depth)
                         if args.won else [])
        finally:
            close_readonly(memory)
        report_ranks(f"w={weight:g}", ranks, base, args)
        if args.won:
            churn(won_before, won_after)

    print("\n  Reaching a record is not answering with it. A gain here still")
    print("  needs a paired generation A/B, both arms in one session.")


def churn(before: list[list[str]], after: list[list[str]]) -> None:
    changed = sum(1 for a, b in zip(before, after) if set(a) != set(b))
    moved = sum(1 for a, b in zip(before, after) if a != b and set(a) == set(b))
    print(f"    of {len(before)} questions we already win: {changed} see a "
          f"different set of facts, {moved} the same facts reordered")


def sample_won(results: str, category: str, k: int, seed: int) -> list[str]:
    """Questions we currently answer correctly, to price the reordering."""
    answered = json.loads(Path(results).read_text(encoding="utf-8"))
    won = [r["question"] for r in answered
           if r.get("category") == category and r.get("ours_ok")]
    rng = random.Random(seed)
    rng.shuffle(won)
    return won[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--targets", default="C:/nmafc_ab/ranking_targets.json")
    ap.add_argument("--results", default="C:/nmafc_ab/haystack.json")
    ap.add_argument("--category", default="open-domain")
    ap.add_argument("--weights", default="0.0003,0.001,0.003,0.01",
                    help="source_grounding weights to screen. Adjacent ranks "
                         "in one fused list differ by about 0.0003; joining a "
                         "whole extra list is worth about 0.016.")
    ap.add_argument("--hydrate-depth", type=int, default=20)
    ap.add_argument("--fact-depth", type=int, default=6)
    ap.add_argument("--won-sample", type=int, default=60,
                    help="Won questions used to price the reordering. 0 skips.")
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.won = (sample_won(args.results, args.category, args.won_sample,
                           args.seed) if args.won_sample else [])
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
