"""Two levers that need no re-ingestion, screened against the finished stores.

The open-domain deficit against RAG is 5.7 points over 830 questions, and 60%
of the questions we lose there are total misses -- gold F1 at or below 0.05,
meaning the answering fact never reached the prompt at all. It is not a wording
problem: our median open-domain answer is four words, the same as RAG's and the
same as gold. Something is keeping facts out of the context.

Two candidates can be tested without paying the 5.3 hours of LLM extraction
again, because neither one is baked into the stores:

  exclude_invalidated   198 of 3,807 hot records (5.2%) carry an invalid_at,
                        and at the default they are dropped from search
                        outright. Cold carries none, so this is the whole of
                        the effect. Set False they are kept and pushed down by
                        the reranker instead, which turns a hard delete into a
                        soft demotion. Some of those 198 demonstrably hold gold
                        text: the question "What type of car did Dave work on"
                        has gold "classic muscle car" and a killed fact reading
                        "Dave is currently working on a classic muscle car".

  hydrate_top_k         the SOURCE block puts raw turns back behind the top
                        facts, which is the only route by which detail the
                        extractor flattened can return. The extractor turns
                        "designed and built a sustainable water purifier" into
                        "achieved an engineering milestone"; RAG never loses it
                        because RAG never rewrites anything. Shipped at 5 and
                        never swept upward.

Retrieval is affected by the first lever and not the second, so this pays for
two retrievals per question and re-formats each one at every hydration level
for free. Roughly 3,000 embedding calls, no generation and no judging.

Reachability is reported loose and strict for the reason _sweep_precision.py
gives: loose drops short tokens so "6 May 2023" scores as carrying a gold of
"7 May 2023", strict demands every token as a whole word and fails on any
paraphrase. A lever worth adopting should move both, and should move them
without inflating the token count so far that the saving over RAG is gone.

Closed through `close_readonly`, so the run's stores are left exactly as they
were. Deferring the reinforcement writes does not achieve that on its own --
the ordinary `close()` commits the buffer.

Usage:
    python -u scripts/benchmarks/_sweep_recovery.py --limit 20
    python -u scripts/benchmarks/_sweep_recovery.py --out /c/nmafc_ab/recovery.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
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
from scripts.benchmarks._probe_answered_wrong import strict_present  # noqa: E402
from scripts.benchmarks._sweep_context_budget import (  # noqa: E402
    CHARS_PER_TOKEN,
    overlap,
    retrieve_with_retry,
)

SCORED = ("single-hop", "temporal", "multi-hop", "open-domain")


def open_memory(store: Path, llm, embedder, args, *, keep_invalidated: bool):
    """The arm's own retrieval settings, with one lever moved.

    Everything here except `exclude_invalidated` is the shipped configuration,
    so a difference in the output is attributable to the lever rather than to
    the harness having quietly retuned something else.
    """
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
                rerank_top_k=args.budget,
                exclude_invalidated=not keep_invalidated,
                defer_reinforcement_writes=True,
            ),
        ),
    )


class Tally:
    def __init__(self) -> None:
        self.n = 0
        self.loose = 0
        self.strict = 0
        self.tokens = 0

    def add(self, loose: bool, strict: bool, tokens: int) -> None:
        self.n += 1
        self.loose += loose
        self.strict += strict
        self.tokens += tokens

    def row(self, label: str, base: "Tally | None") -> str:
        loose = self.loose / self.n
        strict = self.strict / self.n
        delta = ""
        if base is not None and base is not self:
            delta = (f"  {loose - base.loose / base.n:+6.1%} "
                     f"{strict - base.strict / base.n:+6.1%} "
                     f"{self.tokens / self.n - base.tokens / base.n:+7.0f}")
        return (f"  {label:18s} {self.n:6d} {loose:8.1%} {strict:8.1%} "
                f"{self.tokens / self.n:8.0f}{delta}")


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    hydrations = [int(h) for h in args.hydrate.split(",")]
    # The shipped configuration, named so it is unmistakable in the output.
    baseline = f"drop-inv/hyd-{args.baseline_hydrate}"
    names: list[str] = []
    for keep in (False, True):
        for h in hydrations:
            names.append(f"{'keep-inv' if keep else 'drop-inv'}/hyd-{h}")

    overall: dict[str, Tally] = {n: Tally() for n in names}
    by_category: dict[str, dict[str, Tally]] = {
        cat: {n: Tally() for n in names} for cat in SCORED
    }
    recovered: list[tuple[str, str, str]] = []

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            continue

        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED]
        if args.limit:
            questions = questions[: args.limit]

        for keep in (False, True):
            prefix = "keep-inv" if keep else "drop-inv"
            memory = open_memory(store, llm, embedder, args, keep_invalidated=keep)
            try:
                turn = memory.current_turn + 1
                router = memory._router
                for qa in questions:
                    pool = await retrieve_with_retry(router, qa.question, turn)
                    if not pool:
                        continue
                    chosen = pool[: args.budget]
                    for h in hydrations:
                        # Hydration is a formatting decision, not a retrieval
                        # one, so every level is free once the pool is in hand.
                        router._config.hydrate_top_k = h
                        context = router.format_context(chosen)
                        loose = overlap(qa.answer, context) >= args.threshold
                        strict = strict_present(qa.answer, context)
                        tokens = len(context) // CHARS_PER_TOKEN
                        name = f"{prefix}/hyd-{h}"
                        overall[name].add(loose, strict, tokens)
                        by_category[qa.category_name][name].add(loose, strict, tokens)

                        # Worth naming individually: a question the shipped
                        # config cannot reach and the kept-invalidated config
                        # can is the whole case for the lever.
                        if (keep and h == args.baseline_hydrate and strict
                                and args.record_recovered):
                            recovered.append((conv.sample_id, qa.question, qa.answer))
            finally:
                close_readonly(memory)

    if not overall[baseline].n:
        print("nothing measured")
        return

    print(f"\n{'=' * 92}")
    print(f"questions : {overall[baseline].n}")
    print(f"baseline  : {baseline} (the shipped configuration)")
    print(f"\n  {'config':18s} {'n':>6s} {'loose':>8s} {'strict':>8s} "
          f"{'tokens':>8s}   {'vs shipped':>22s}")
    base = overall[baseline]
    for n in names:
        print(overall[n].row(n, base))

    for cat in SCORED:
        tallies = by_category[cat]
        if not tallies[baseline].n:
            continue
        print(f"\n  {cat}")
        for n in names:
            print(tallies[n].row(n, tallies[baseline]))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "overall": {k: vars(v) for k, v in overall.items()},
            "by_category": {
                c: {k: vars(v) for k, v in t.items()}
                for c, t in by_category.items()
            },
            "baseline": baseline,
            "recovered": recovered,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot-top-k", type=int, default=10)
    ap.add_argument("--cold-budget", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", default="0,5,8,12",
                    help="Comma-separated hydrate_top_k levels to compare.")
    ap.add_argument("--baseline-hydrate", type=int, default=5,
                    help="The shipped hydrate_top_k, used as the comparison row.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--record-recovered", action="store_true")
    ap.add_argument("--out")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
