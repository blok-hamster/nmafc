"""What is left inside the 2.2 seconds a question now costs?

Reinforcement writeback used to dominate: full_v2 answered at a mean of 21,306
ms per question, full_v3 at 2,159 ms, and the difference is almost entirely
`defer_reinforcement_writes` taking the delete-and-re-add out of the query path.
That fix is spent. This asks what remains, so that any further latency work is
aimed at something that is actually there.

Four regions, timed per question on the finished stores:

    embed        one network call to the embedding deployment
    retrieve     hot vector search, cold search, graph traversal, reranking
    format       rendering the context, including the hydration lookup
    generate     the answer call

Only `generate` and `embed` cross the network, and only they can be large.
Splitting them out is the point: a local cost is something this codebase can
fix, a network cost is a deployment or a model choice, and conflating the two
has already produced one round of tuning aimed at the wrong half.

Absolute network figures are only meaningful when nothing else is drawing on
the same deployment quota. Run this alone.

Usage:
    python scripts/benchmarks/_latency_split.py --limit 15
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics as st
import sys
import time
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

from scripts.benchmarks._ab_budget import SCORED, open_memory  # noqa: E402
from scripts.benchmarks.arms.neuromorphic_tuned import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
)
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402


class Timing(dict):
    """Millisecond samples, keyed by region."""

    def add(self, region: str, started: float) -> None:
        self.setdefault(region, []).append((time.perf_counter() - started) * 1000)


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    timing = Timing()
    asked = 0

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            continue
        memory = open_memory(store, llm, embedder, args.hot, args.cold,
                             args.budget, args.max_hops, args.compact, args.hydrate)
        try:
            turn = memory.current_turn + 1
            for qa in [q for q in conv.qa_pairs
                       if q.category_name in SCORED][: args.limit]:
                mark = time.perf_counter()
                await embedder.embed_single(qa.question)
                timing.add("embed", mark)

                mark = time.perf_counter()
                records = await memory._router.retrieve(qa.question, turn)
                timing.add("retrieve", mark)

                mark = time.perf_counter()
                context = memory._router.format_context(records)
                timing.add("format", mark)

                system = ANSWER_SYSTEM_PROMPT + (f"\n\n{context}" if context else "")
                mark = time.perf_counter()
                await llm.chat(
                    messages=[{"role": "user", "content": qa.question}],
                    system_prompt=system,
                )
                timing.add("generate", mark)
                asked += 1
        finally:
            memory.close()

    if not asked:
        print("nothing measured")
        return

    # `retrieve` embeds the query itself, so the separate embed call above is
    # what that costs, not an extra one the pipeline makes.
    total = st.mean([sum(v) for v in zip(*(timing[k] for k in timing))])
    print(f"\n{'=' * 58}")
    print(f"  questions: {asked}\n")
    print(f"  {'region':<12}{'mean':>9}{'p50':>9}{'p90':>9}{'share':>9}")
    for region in ("embed", "retrieve", "format", "generate"):
        got = sorted(timing[region])
        print(f"  {region:<12}{st.mean(got):>8.0f}{got[len(got) // 2]:>9.0f}"
              f"{got[int(len(got) * 0.9)]:>9.0f}{st.mean(got) / total:>8.1%}")
    print(f"  {'total':<12}{total:>8.0f}")
    local = st.mean(timing["retrieve"]) + st.mean(timing["format"])
    print(f"\n  local work (retrieve + format): {local:.0f} ms, "
          f"{local / total:.1%} of the question")
    print(f"  network (embed + generate):     {total - local:.0f} ms, "
          f"{(total - local) / total:.1%}")
    print("\n  The embed call is inside retrieve as well; it is timed here on "
          "its own\n  to show how much of retrieve is waiting on the network "
          "rather than working.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--limit", type=int, default=15)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
