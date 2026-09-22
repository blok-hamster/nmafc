"""Does decay's score help, when it is measured on a test where forgetting can pay?

`weight_signal` was switched on once, measured against LoCoMo, and cost 5.3
points, which is why it has been 0.0 ever since. That measurement was taken on
the wrong benchmark. LoCoMo asks every question after the last turn about things
said once and never revised, so no question in it is answered correctly by
forgetting anything. A forgetting signal can only lose points there. The brakes
were tested by measuring top speed.

mylocomoeval is the test where forgetting can pay, and on it the recoverable
loss is precise: 14.9% of questions reach the model with only the superseded
fact in the prompt, and every single one is answered wrong. That is exactly the
case decay claims to fix. The stale fact was stated earlier and not repeated, so
it should have faded; the current fact was stated later, so it should rank
above. `weight_signal` is the one setting that lets that ranking happen, and it
has never been measured here.

Retrieval only. No answering, no judging. The three numbers that decide it:

  gold in prompt        must not fall -- a signal that buries the right fact is
                        the 5.3-point LoCoMo failure repeating itself
  stale in prompt       should fall, that is the mechanism working
  stale ONLY in prompt  the one that matters, because those are wrong 100% of
                        the time and no amount of prompting recovers them

`recency_boost` is swept alongside because it is the other disconnected route
from "when was this said" to "how highly is it ranked", and the two are cheap
to vary together.

Closed through `close_readonly`, so the stores are left as they are. Deferring
the reinforcement writes does not achieve that on its own -- `close()` commits
the buffer, which is what flattened the first version of this sweep.

Usage:
    python -u scripts/benchmarks/_sweep_decay_signal.py --limit 2
    python -u scripts/benchmarks/_sweep_decay_signal.py --out C:/nmafc_ab/decay_signal.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

from scripts.benchmarks._ab_budget import close_readonly  # noqa: E402
from scripts.benchmarks._sweep_context_budget import retrieve_with_retry  # noqa: E402
from scripts.benchmarks._test_updates import present  # noqa: E402


def open_memory(store: Path, llm, embedder, args, weight: float, recency: float):
    """The shipped configuration with the two ranking signals moved.

    Everything else is left at the values the head-to-head ran on, so a
    difference in the output belongs to the signal rather than to the harness
    having quietly retuned something alongside it.
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
                top_k=args.hot,
                fallback_keyword_limit=args.cold,
                rerank_top_k=args.budget,
                hydrate_top_k=args.hydrate,
                weight_signal=weight,
                recency_boost=recency,
                defer_reinforcement_writes=True,
            ),
        ),
    )


class Tally:
    def __init__(self) -> None:
        self.n = self.gold = self.stale = self.only_stale = self.neither = 0

    def add(self, gold: bool, stale: bool) -> None:
        self.n += 1
        self.gold += gold
        self.stale += stale
        self.only_stale += stale and not gold
        self.neither += not stale and not gold

    def row(self, label: str, base: "Tally | None") -> str:
        pct = lambda x: 100 * x / self.n  # noqa: E731
        delta = ""
        if base is not None and base is not self:
            b = lambda x: 100 * x / base.n  # noqa: E731
            delta = (f"   {pct(self.gold) - b(base.gold):+6.1f}"
                     f" {pct(self.stale) - b(base.stale):+6.1f}"
                     f" {pct(self.only_stale) - b(base.only_stale):+6.1f}")
        return (f"  {label:22s} {self.n:5d} {pct(self.gold):7.1f}% "
                f"{pct(self.stale):7.1f}% {pct(self.only_stale):7.1f}% "
                f"{pct(self.neither):7.1f}%{delta}")


async def run(args: argparse.Namespace) -> None:
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for row in questions:
        by_conv[row["conv"]].append(row)
    if args.limit:
        by_conv = {k: v[: args.limit] for k, v in by_conv.items()}

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    weights = [float(w) for w in args.weights.split(",")]
    recencies = [float(r) for r in args.recency.split(",")]
    configs = [(w, r) for w in weights for r in recencies]
    # The shipped pair, named so it is unmistakable as the comparison row.
    baseline = "w=0.0 r=0.0"
    names = [f"w={w} r={r}" for w, r in configs]
    tallies = {n: Tally() for n in names}

    total = sum(len(v) for v in by_conv.values())
    print(f"{total} questions x {len(configs)} configs, retrieval only\n")

    for conv, items in by_conv.items():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv}"
        if not store.is_dir():
            continue
        print(f"[{conv}] {len(items)} questions")
        for (weight, recency), name in zip(configs, names):
            memory = open_memory(store, llm, embedder, args, weight, recency)
            try:
                turn = memory.current_turn + 1
                router = memory._router
                for item in items:
                    pool = await retrieve_with_retry(router, item["question"], turn)
                    context = router.format_context(pool[: args.budget]) if pool else ""
                    tallies[name].add(present(item["gold"], context),
                                      present(item["stale"], context))
            finally:
                # Not `close()`. That flushes the reinforcements these queries
                # buffered, resetting weight to 1.0 across everything they
                # touched, so the second configuration in this loop would be
                # reading a store the first one had already flattened.
                close_readonly(memory)

    if not tallies[baseline].n:
        print("nothing measured")
        return

    print(f"\n{'=' * 100}")
    print(f"  {'config':22s} {'n':>5s} {'gold':>8s} {'stale':>8s} "
          f"{'only stale':>8s} {'neither':>8s}   {'vs shipped':>21s}")
    base = tallies[baseline]
    for n in names:
        print(tallies[n].row(n, base))

    print("\n  gold must not fall. stale should fall. only-stale is the one that")
    print("  decides it, because those questions are wrong 100% of the time.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {k: vars(v) for k, v in tallies.items()}, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--questions", default="C:/nmafc_ab/updates_final.json")
    ap.add_argument("--weights", default="0.0,0.05,0.1,0.2,0.4")
    ap.add_argument("--recency", default="0.0")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="C:/nmafc_ab/decay_signal.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
