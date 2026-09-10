"""Can we tell, at query time, which questions need the turns?

Every configuration screened so far spends the same number of turns on every
question. That is the wrong shape for the constraint we are actually under. The
ceiling is a **mean** of 1,000 tokens, and a turn costs about 55, so a question
answered by its facts alone could fund a question that needs twelve turns
without moving the mean at all.

That only works if the two can be told apart *before* the answer is generated,
using something already in hand. This screen tests exactly one candidate signal,
which costs nothing extra because the FACTS block is already rendered:

    covered = share of the question's content words that appear in <FACTS>

The reasoning: extraction is a summary, and the failure mode this whole session
has been chasing is a summary that dropped the word the question turns on. If a
question's rare words are all present in the facts, the facts are probably about
the thing being asked and the dialogue adds little. If they are absent, the
facts are about something adjacent and the wording is only in the turn.

**What would make the mechanism worth building.** The signal has to separate two
groups that need different budgets:

    gold in FACTS       turns are close to wasted here. Spend the floor.
    gold in SOURCE only  turns are the entire reason the answer is reachable.
                         Spend the maximum.

If `covered` is the same in both groups, there is no signal, adaptive hydration
is a coin toss dressed as a mechanism, and it should not be built. That is the
result this is looking for as hard as the other one -- three prompt levers have
already been killed this session by screening them before spending.

Reported as the mean `covered` per group plus the split at a threshold, because
a difference in means that does not survive thresholding cannot be acted on: the
mechanism has to make a decision per question, not observe a correlation.

Read-only. One embedding per question, no generation.

Usage:
    python -u scripts/benchmarks/_screen_adaptive_signal.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
from scripts.benchmarks._screen_hydrate_pool import (  # noqa: E402
    content_words,
    present,
)
from scripts.benchmarks._screen_source_block import halves  # noqa: E402


def covered(question: str, facts: str) -> float:
    """Share of the question's content words the FACTS block already carries.

    Computed against the facts only. Including SOURCE would leak the thing being
    predicted: a question whose words are in the turn is exactly the case the
    signal is supposed to spot *before* deciding how many turns to read.
    """
    asked = content_words(question)
    if not asked:
        return 1.0
    return len(asked & content_words(facts)) / len(asked)


async def render(memory, questions: list[str]) -> list[str]:
    turn = memory.current_turn + 1
    out: list[str] = []
    for question in questions:
        records = await with_retries(
            lambda q=question: memory._router.retrieve(q, turn))
        out.append(memory._router.format_context(records or [], question))
    return out


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no store at {store}")

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    memory = open_memory(store, llm, embedder, args.hot, args.cold, args.rerank,
                         args.max_hops, compact=True, hydrate=args.hydrate,
                         facts=args.facts, dedupe=True)
    memory._router._config.hydrate_pool = args.pool
    memory._router._config.source_grounding = args.grounding
    memory._router._config.hydrate_scan = args.scan
    try:
        rendered = await render(memory, [r["question"] for r in rows])
    finally:
        close_readonly(memory)

    groups: dict[str, list[float]] = {}
    for row, ctx in zip(rows, rendered):
        facts, source = halves(ctx)
        in_f, in_s = present(row["gold"], facts), present(row["gold"], source)
        if in_f:
            name = "gold in FACTS"
        elif in_s:
            name = "gold in SOURCE only"
        else:
            name = "gold in neither"
        groups.setdefault(name, []).append(covered(row["question"], facts))

    print(f"\n{len(rows)} open-domain questions, {args.hydrate} turns each\n")
    print(f"  {'':<22}{'n':>6}{'mean covered':>15}")
    order = ["gold in FACTS", "gold in SOURCE only", "gold in neither"]
    for name in order:
        vals = groups.get(name, [])
        if vals:
            print(f"  {name:<22}{len(vals):>6}{sum(vals) / len(vals):>15.3f}")

    # A difference in means is not a mechanism. This is: at each threshold, how
    # cleanly does the rule "covered < t, spend the maximum" catch the questions
    # that need turns without also catching the ones that do not.
    print(f"\n  Thresholding. 'needs turns' = gold in SOURCE only.\n")
    print(f"  {'covered <':<12}{'flagged':>9}{'of which need turns':>22}"
          f"{'need turns caught':>20}")
    need = groups.get("gold in SOURCE only", [])
    dont = groups.get("gold in FACTS", [])
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.75):
        hit = sum(1 for v in need if v < t)
        false = sum(1 for v in dont if v < t)
        flagged = hit + false
        prec = 100 * hit / flagged if flagged else 0.0
        rec = 100 * hit / len(need) if need else 0.0
        print(f"  {t:<12.2f}{flagged:>9}{prec:>21.1f}%{rec:>19.1f}%")

    base = 100 * len(need) / max(len(need) + len(dont), 1)
    print(f"\n  Base rate is {base:.1f}%. A threshold whose precision is not "
          f"clearly above\n  that is not separating anything, and the "
          f"mechanism should not be built.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/od_cheap.json")
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--rerank", type=int, default=40)
    ap.add_argument("--facts", type=int, default=16)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--grounding", type=float, default=0.003)
    ap.add_argument("--scan", type=int, default=8)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=7)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
