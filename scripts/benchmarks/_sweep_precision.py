"""Can the same twenty slots be filled better than they are now?

The failure classification found two things the run wastes its budget on. The
same fact reaches the prompt twice under different entity names -- one context
carried "Caroline participated in the pride parade she attended a few weeks ago"
verbatim in two of its twenty slots -- and a single entity can occupy several
slots with near-misses while the fact that answers the question sits below the
cut. Both are selection faults, not retrieval faults: the candidates are already
in the pool, the reranker just spends the budget badly.

So this changes nothing about what is retrieved. It retrieves once per question
at the run's own settings, then fills the twenty slots under different policies
and measures what each one reaches:

  stock        the ranking as it is, first twenty
  dedupe       drop repeats of a fact already selected, then take twenty
  cap-N        at most N facts per entity, then take twenty
  hydrate-5    stock twenty, plus the source turns behind the top five facts

Reachability is reported two ways because one number would be misleading. The
loose test drops tokens of two characters or fewer, so a gold of "7 May 2023"
becomes {may, 2023} and a fact reading "6 May 2023" scores as carrying it; the
strict test demands every token as a whole word and fails on any paraphrase.
The truth is between them, and a policy worth adopting should move both.

Retrieval only. No generation, no judging, reinforcement deferred and never
flushed, so the run's stores are left exactly as they were.

Usage:
    python -u scripts/benchmarks/_sweep_precision.py --limit 20
    python -u scripts/benchmarks/_sweep_precision.py --out /c/nmafc_ab/precision.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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
from nmafc.schemas.memory import DecayConfig  # noqa: E402
from nmafc.storage.config import NMafcConfig, StorageConfig  # noqa: E402
from nmafc.wrapper import NeuromorphicMemory  # noqa: E402

from scripts.benchmarks.arms.base import build_dated_exchanges  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks._probe_answered_wrong import strict_present  # noqa: E402
from scripts.benchmarks._sweep_context_budget import (  # noqa: E402
    CHARS_PER_TOKEN,
    overlap,
    retrieve_with_retry,
)

SCORED = ("single-hop", "temporal", "multi-hop", "open-domain")


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower()).strip()


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
                # The run's own hot and cold budgets. Widening either would
                # change what is retrieved, and the point here is to spend the
                # existing pool better rather than to fetch more of it.
                top_k=args.hot_top_k,
                fallback_keyword_limit=args.cold_budget,
                # Wide enough that the whole pool survives reranking, so each
                # policy can reach past the cut for replacements.
                rerank_top_k=args.pool,
                defer_reinforcement_writes=True,
            ),
        ),
    )


def select(records, budget: int, dedupe: bool, cap: int | None):
    """Fill `budget` slots from the ranking under one policy."""
    chosen = []
    seen: set[str] = set()
    per_entity: dict[str, int] = defaultdict(int)
    for r in records:
        if dedupe:
            key = normalise(r.fact_content)
            if key in seen:
                continue
            seen.add(key)
        if cap is not None:
            name = (r.entity_name or "").lower()
            if per_entity[name] >= cap:
                continue
            per_entity[name] += 1
        chosen.append(r)
        if len(chosen) == budget:
            break
    return chosen


class Tally:
    def __init__(self) -> None:
        self.n = 0
        self.loose = 0
        self.strict = 0
        self.tokens = 0
        self.slots = 0

    def add(self, loose: bool, strict: bool, tokens: int, slots: int) -> None:
        self.n += 1
        self.loose += loose
        self.strict += strict
        self.tokens += tokens
        self.slots += slots

    def row(self, label: str, base: "Tally | None") -> str:
        loose = self.loose / self.n
        strict = self.strict / self.n
        delta = ""
        if base is not None and base is not self:
            delta = (f"  {loose - base.loose / base.n:+6.1%} "
                     f"{strict - base.strict / base.n:+6.1%}")
        return (f"  {label:16s} {self.n:6d} {loose:8.1%} {strict:8.1%} "
                f"{self.tokens / self.n:8.0f} {self.slots / self.n:7.1f}{delta}")


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    policies: list[tuple[str, dict]] = [
        ("stock", {"dedupe": False, "cap": None, "hydrate": 0}),
        ("dedupe", {"dedupe": True, "cap": None, "hydrate": 0}),
        ("cap-2", {"dedupe": False, "cap": 2, "hydrate": 0}),
        ("cap-3", {"dedupe": False, "cap": 3, "hydrate": 0}),
        ("dedupe+cap-2", {"dedupe": True, "cap": 2, "hydrate": 0}),
        ("dedupe+cap-3", {"dedupe": True, "cap": 3, "hydrate": 0}),
        ("hydrate-5", {"dedupe": False, "cap": None, "hydrate": 5}),
        ("dedupe+hyd-5", {"dedupe": True, "cap": None, "hydrate": 5}),
        ("ded+cap3+hyd5", {"dedupe": True, "cap": 3, "hydrate": 5}),
    ]

    overall: dict[str, Tally] = {name: Tally() for name, _ in policies}
    by_category: dict[str, dict[str, Tally]] = {
        cat: {name: Tally() for name, _ in policies} for cat in SCORED
    }
    pool_sizes: list[int] = []
    duplicates: list[int] = []

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            continue

        exchanges = [text for text, _ in build_dated_exchanges(conv.get_flat_history())]
        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED]
        if args.limit:
            questions = questions[: args.limit]

        memory = open_memory(store, llm, embedder, args)
        try:
            turn = memory.current_turn + 1
            for qa in questions:
                pool = await retrieve_with_retry(memory._router, qa.question, turn)
                if not pool:
                    continue
                pool_sizes.append(len(pool))

                head = pool[: args.budget]
                duplicates.append(
                    len(head) - len({normalise(r.fact_content) for r in head})
                )

                for name, policy in policies:
                    chosen = select(
                        pool, args.budget, policy["dedupe"], policy["cap"]
                    )
                    context = memory._router.format_context(chosen)
                    if policy["hydrate"]:
                        # One exchange per distinct source turn, not one per
                        # fact: several facts routinely come off the same turn.
                        turns = sorted({
                            r.created_at_turn for r in chosen[: policy["hydrate"]]
                            if 1 <= r.created_at_turn <= len(exchanges)
                        })
                        context += "\n" + "\n".join(exchanges[t - 1] for t in turns)

                    loose = overlap(qa.answer, context) >= args.threshold
                    strict = strict_present(qa.answer, context)
                    tokens = len(context) // CHARS_PER_TOKEN
                    overall[name].add(loose, strict, tokens, len(chosen))
                    by_category[qa.category_name][name].add(
                        loose, strict, tokens, len(chosen)
                    )
        finally:
            memory.close()

    if not pool_sizes:
        print("nothing measured")
        return

    print(f"\n{'=' * 88}")
    print(f"questions        : {overall['stock'].n}")
    print(f"candidate pool   : mean {statistics.mean(pool_sizes):.1f}, "
          f"median {statistics.median(pool_sizes):.0f}")
    print(f"repeated facts   : mean {statistics.mean(duplicates):.2f} of the "
          f"{args.budget} slots, "
          f"{sum(1 for d in duplicates if d) / len(duplicates):.1%} of prompts "
          f"carry at least one")

    print(f"\n  {'policy':16s} {'n':>6s} {'loose':>8s} {'strict':>8s} "
          f"{'tokens':>8s} {'slots':>7s}   {'vs stock':>14s}")
    base = overall["stock"]
    for name, _ in policies:
        print(overall[name].row(name, base))

    for cat in SCORED:
        tallies = by_category[cat]
        if not tallies["stock"].n:
            continue
        print(f"\n  {cat}")
        cat_base = tallies["stock"]
        for name, _ in policies:
            print(tallies[name].row(name, cat_base))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "overall": {k: vars(v) for k, v in overall.items()},
            "by_category": {
                c: {k: vars(v) for k, v in t.items()}
                for c, t in by_category.items()
            },
            "mean_pool": statistics.mean(pool_sizes),
            "mean_duplicates": statistics.mean(duplicates),
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--pool", type=int, default=80)
    ap.add_argument("--hot-top-k", type=int, default=10)
    ap.add_argument("--cold-budget", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
