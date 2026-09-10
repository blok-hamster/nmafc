"""Is the answer sitting in the source text of the facts we already retrieve?

The comparison against RAG ends in a tie that hides a split: both systems answer
797 questions correctly, but 223 are ours alone and 221 are RAG's alone, and
RAG's share concentrates in open-domain. An oracle picking the better system per
question would score 80.6% against our 64% and RAG's 66%, so fourteen points sit
in questions one of us can already answer.

There is a structural reason RAG owns its share. Nothing in this system stores
the conversation. `memory_event_log` keeps `fact_content` and the hot table
keeps a vector of it; the utterance that produced the fact is discarded at
extraction and cannot be recovered from any store. RAG's index *is* that text.
So whenever an answer turns on how something was said rather than what it
amounted to, RAG has the evidence and we structurally do not.

The proposed fix is a bridge: keep the summarised fact as the thing that decays
and gets ranked, and hang the verbatim source turn off it as an evidence layer
that never decays and is never ranked, hydrated into the prompt only for the few
top-ranked facts. Before building it, this measures whether it could work.

For each question it retrieves exactly what the run retrieved, then asks three
things of the gold answer:

  in facts     -- is it in the fact text we sent? (what we have today)
  in sources   -- is it in the exchanges those facts were extracted from?
  in top-5     -- is it in the exchanges behind just the five best facts?

The gap between the first and the second is the ceiling on hydration. The third
is what a cheap version actually buys, and its measured token cost is reported
alongside, because an unaffordable ceiling is not a plan.

Retrieval only. No generation, no judging, and closed through `close_readonly`,
so nothing on disk changes. Deferring the reinforcement writes does not achieve
that on its own -- `close()` commits the buffer.

Usage:
    python -u scripts/benchmarks/_probe_verbatim.py --limit 20
    python -u scripts/benchmarks/_probe_verbatim.py --out /c/nmafc_ab/verbatim.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
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

from scripts.benchmarks._ab_budget import close_readonly  # noqa: E402
from scripts.benchmarks.arms.base import build_dated_exchanges  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks._sweep_context_budget import (  # noqa: E402
    CHARS_PER_TOKEN,
    overlap,
    retrieve_with_retry,
)

SCORED = ("single-hop", "temporal", "multi-hop", "open-domain")


def judged(results_path: Path, arm: str | None = None) -> dict[str, bool]:
    """question text -> whether the judge marked it correct.

    Keyed on the question rather than a position, because the two runs answered
    different numbers of questions and neither records a conversation id
    alongside the answer.
    """
    payload = json.loads(results_path.read_text(encoding="utf-8"))["results"]
    key = arm or next(iter(payload))
    return {
        row["question"]: bool(row["judge_correct"])
        for row in payload[key]["question_results"]
    }


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
                rerank_top_k=args.budget,
                # Reads the run's own stores. Reinforcement rewrites every
                # record it returns, so the writeback is buffered and never
                # flushed, leaving the facts and their weights untouched.
                defer_reinforcement_writes=True,
            ),
        ),
    )


class Tally:
    """Counts one group of questions across the three reachability tests."""

    def __init__(self) -> None:
        self.n = 0
        self.facts = 0
        self.sources = 0
        self.top = 0
        self.anywhere = 0
        # Hydration adds the source text to the facts, it does not swap one for
        # the other, so the number that decides whether to build it is the
        # union. Reading the two columns separately understates it: a gold
        # answer whose keywords are split between a fact and its source turn is
        # reachable from the pair while being in neither alone.
        self.with_sources = 0
        self.with_top = 0

    def add(self, facts: bool, sources: bool, top: bool, anywhere: bool,
            with_sources: bool, with_top: bool) -> None:
        self.n += 1
        self.facts += facts
        self.sources += sources
        self.top += top
        self.anywhere += anywhere
        self.with_sources += with_sources
        self.with_top += with_top

    def row(self, label: str) -> str:
        if not self.n:
            return f"  {label:24s}       0"
        pct = lambda v: f"{v / self.n:6.1%}"  # noqa: E731
        return (f"  {label:24s} {self.n:7d}   {pct(self.facts)}  "
                f"{pct(self.with_top)}  {pct(self.with_sources)}  "
                f"{pct(self.anywhere)}"
                f"   {(self.with_top - self.facts) / self.n:+6.1%}"
                f"  {(self.with_sources - self.facts) / self.n:+6.1%}")


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    ours = judged(Path(args.run) / "results.json", args.arm)
    theirs = judged(Path(args.rag_run) / "results.json")
    print(f"ours: {len(ours)} judged questions, RAG: {len(theirs)}")

    groups: dict[str, Tally] = defaultdict(Tally)
    by_category: dict[str, Tally] = defaultdict(Tally)
    hydration_tokens: list[int] = []
    context_tokens: list[int] = []
    latencies: list[float] = []
    examples: list[dict] = []
    skipped = 0

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            continue

        # Turn numbering: the wrapper increments before processing, so exchange
        # index 0 was ingested as turn 1. Same arithmetic as the date backfill.
        exchanges = [text for text, _ in build_dated_exchanges(conv.get_flat_history())]
        transcript = "\n".join(exchanges)

        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED]
        if args.limit:
            questions = questions[: args.limit]

        memory = open_memory(store, llm, embedder, args)
        try:
            turn = memory.current_turn + 1
            for qa in questions:
                if qa.question not in ours or qa.question not in theirs:
                    skipped += 1
                    continue

                mark = time.perf_counter()
                records = await retrieve_with_retry(memory._router, qa.question, turn)
                latencies.append((time.perf_counter() - mark) * 1000)
                records = records[: args.budget]
                if not records:
                    skipped += 1
                    continue

                context = memory._router.format_context(records)
                context_tokens.append(len(context) // CHARS_PER_TOKEN)

                def source_text(subset) -> tuple[str, int]:
                    """Exchanges behind a set of facts, deduplicated.

                    Facts extracted from one exchange share a turn, so five
                    facts routinely point at fewer than five exchanges. The
                    hydration bill is per distinct turn, not per fact.
                    """
                    turns = sorted({
                        r.created_at_turn for r in subset
                        if 1 <= r.created_at_turn <= len(exchanges)
                    })
                    text = "\n".join(exchanges[t - 1] for t in turns)
                    return text, len(turns)

                all_sources, _ = source_text(records)
                top_sources, top_turns = source_text(records[: args.hydrate])
                hydration_tokens.append(len(top_sources) // CHARS_PER_TOKEN)

                in_facts = overlap(qa.answer, context) >= args.threshold
                in_sources = overlap(qa.answer, all_sources) >= args.threshold
                in_top = overlap(qa.answer, top_sources) >= args.threshold
                in_conv = overlap(qa.answer, transcript) >= args.threshold
                with_sources = (
                    overlap(qa.answer, context + "\n" + all_sources) >= args.threshold
                )
                with_top = (
                    overlap(qa.answer, context + "\n" + top_sources) >= args.threshold
                )

                we_win = ours[qa.question]
                they_win = theirs[qa.question]
                group = ("both right" if we_win and they_win else
                         "ours only" if we_win else
                         "RAG only" if they_win else "neither")

                args_ = (in_facts, in_sources, in_top, in_conv,
                         with_sources, with_top)
                groups[group].add(*args_)
                groups["all"].add(*args_)
                if group == "RAG only":
                    by_category[qa.category_name].add(*args_)

                # The case the bridge exists for: RAG answered it, we did not,
                # the fact text does not carry the answer, and the source of a
                # fact we already ranked does.
                if (group == "RAG only" and not in_facts and in_top
                        and len(examples) < args.examples):
                    examples.append({
                        "conversation": conv.sample_id,
                        "category": qa.category_name,
                        "question": qa.question,
                        "gold": qa.answer,
                        "turns": top_turns,
                        "source": top_sources[:600],
                    })
        finally:
            close_readonly(memory)

    if not groups:
        print("nothing measured")
        return

    print(f"\n{'=' * 84}")
    print("share of questions whose gold answer is reachable, by where we look")
    print(f"\n  {'group':24s} {'n':>7s}   {'facts':>7s} "
          f"{'+top' + str(args.hydrate):>7s} {'+all':>7s} {'in conv':>7s}"
          f"   {'gain5':>6s} {'gainAll':>7s}")
    for label in ("all", "both right", "ours only", "RAG only", "neither"):
        if label in groups:
            print(groups[label].row(label))

    print(f"\n  RAG-only questions by category")
    for name in SCORED:
        if name in by_category:
            print(by_category[name].row(name))

    print(f"\n{'=' * 84}")
    print(f"  context sent today          : {statistics.mean(context_tokens):7.0f} tokens")
    print(f"  hydrating the top {args.hydrate:<2d} adds   : "
          f"{statistics.mean(hydration_tokens):7.0f} tokens "
          f"(median {statistics.median(hydration_tokens):.0f})")
    print(f"  retrieval latency           : "
          f"p50 {statistics.median(latencies):.0f} ms, "
          f"mean {statistics.mean(latencies):.0f} ms")
    if skipped:
        print(f"  skipped (unjudged or empty) : {skipped}")

    for ex in examples:
        print(f"\n  [{ex['category']}] {ex['question']}")
        print(f"    gold  : {ex['gold']}")
        print(f"    source ({ex['turns']} turns behind the top {args.hydrate} facts):")
        for line in ex["source"].splitlines()[:6]:
            print(f"      {line[:96]}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "groups": {k: vars(v) for k, v in groups.items()},
            "rag_only_by_category": {k: vars(v) for k, v in by_category.items()},
            "context_tokens": statistics.mean(context_tokens),
            "hydration_tokens": statistics.mean(hydration_tokens),
            "examples": examples,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--rag-run", default="scripts/benchmarks/results/rag_rerun")
    # The run's own retrieval settings, so this measures what was actually sent.
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot-top-k", type=int, default=10)
    ap.add_argument("--cold-budget", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5,
                    help="How many top facts a cheap bridge would hydrate")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--out")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
