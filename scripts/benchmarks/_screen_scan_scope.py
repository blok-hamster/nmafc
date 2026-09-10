"""How much of the SOURCE block is spent on the wrong conversation?

One rendered context started this. Asked which novel Evan finds gripping, four
of the five hydrated turns came from other people's conversations entirely --
John and Maria about houses, Joanna about a screenplay, Tim about a different
novel -- pulled in because they contain the words "gripping" or "novel". Only
the fifth held the answer.

The cause is structural rather than a bad weight. `_build_haystack.py` lays ten
conversations end to end under one agent and one conversation id, so
`all_turn_text()` hands `_scanned_turns` all 2,957 turns of all ten and BM25
ranks across the lot. A rare word in a stranger's dialogue outranks a common
word in the right one.

The conversation boundary is not lost, though. The merge offsets turns per
source, so each conversation is a **contiguous range**, and the retrieved facts
say which range the question is about. That is inference from what retrieval
already found, not oracle knowledge of which conversation the question came
from, so a real system on a merged store could do the same thing.

This measures the prize before anything is built:

    in-block        hydrated turns within the span of the retrieved facts' turns
    out-of-block    hydrated turns outside it, which are the suspect ones
    gold in-block   whether the answer was in the range we would have kept

If out-of-block turns are common and the gold is nearly always in-block, then
scoping the scan is free precision and free tokens. If the gold is often
out-of-block, the scan is doing real work across conversations and scoping it
would break more than it fixes. Both are worth knowing and only one is expected.

Read-only, no generation.

Usage:
    python -u scripts/benchmarks/_screen_scan_scope.py --limit 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
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
from scripts.benchmarks._screen_hydrate_pool import content_words  # noqa: E402


def turn_text(store: Path) -> dict[int, str]:
    conn = sqlite3.connect(f"file:{store / 'cold.db'}?mode=ro", uri=True)
    rows = conn.execute("SELECT turn, text FROM turn_text").fetchall()
    conn.close()
    return {int(t): x for t, x in rows}


def gold_turns(gold: str, texts: dict[int, str]) -> list[int]:
    """Every turn carrying all of the gold's content words."""
    wanted = content_words(gold)
    if not wanted:
        return []
    return [t for t, x in texts.items() if wanted <= content_words(x)]


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    texts = turn_text(store)

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = [r for r in rows if r["category"] == args.category and not r["ok"]]
    if args.limit:
        rows = rows[: args.limit]

    memory = open_memory(store, llm, embedder, args.hot, args.cold, args.rerank,
                         args.max_hops, compact=True, hydrate=args.hydrate,
                         facts=args.facts, dedupe=True, overlap=args.overlap)
    memory._router._config.hydrate_pool = args.pool
    memory._router._config.source_grounding = args.grounding
    memory._router._config.hydrate_scan = args.scan
    router = memory._router

    # Retrieval is the slow part and none of the span settings change it, so
    # each question is retrieved once and every setting is scored off that.
    cases = []
    try:
        turn = memory.current_turn + 1
        for row in rows:
            recs = await with_retries(
                lambda q=row["question"]: router.retrieve(q, turn))
            recs = recs or []
            fact_turns = [r.created_at_turn for r in recs if r.created_at_turn]
            if not fact_turns:
                continue
            chosen = router._turns_by_question(row["question"], recs,
                                               args.hydrate)
            if not chosen:
                chosen = list(dict.fromkeys(fact_turns))[: args.hydrate]
            cases.append((fact_turns, chosen, gold_turns(row["gold"], texts)))
    finally:
        close_readonly(memory)

    print(f"\n{len(cases)} questions this configuration loses, "
          f"hydrate {args.hydrate}, scan {args.scan}")
    print("\n  span from    margin    span    out-of    gold in    gold turn "
          "hydrated")
    print("  top facts     turns   turns     block       span      now  "
          "if scoped")
    for k in args.span_facts:
        for margin in args.margins:
            tot = out = known = inside = now = scoped = 0
            spans = []
            for fact_turns, chosen, gt in cases:
                seed = fact_turns[:k] if k else fact_turns
                lo, hi = min(seed) - margin, max(seed) + margin
                spans.append(hi - lo)
                tot += len(chosen)
                out += sum(1 for t in chosen if not lo <= t <= hi)
                if not gt:
                    continue
                known += 1
                inside += any(lo <= t <= hi for t in gt)
                now += any(t in chosen for t in gt)
                # What the same budget reaches if every out-of-span turn is
                # replaced by the next in-span candidate. An upper bound: it
                # assumes the freed slot goes to the right turn.
                kept = [t for t in chosen if lo <= t <= hi]
                room = len(chosen) - len(kept)
                extra = [t for t in gt if lo <= t <= hi and t not in kept]
                scoped += bool(any(t in kept for t in gt) or (room and extra))
            print(f"  {k or 'all':>9}  {margin:>8}  {sum(spans) / len(spans):>6.0f}"
                  f"  {100 * out / max(tot, 1):>7.1f}%"
                  f"  {100 * inside / max(known, 1):>8.1f}%"
                  f"  {100 * now / max(known, 1):>7.1f}%"
                  f"  {100 * scoped / max(known, 1):>8.1f}%")
    print(f"\n  gold located in some turn: {known} of {len(cases)}. Store is "
          "2,957 turns.\n  'gold in span' is the safety check -- anything "
          "below 100% is reach\n  the scoping would destroy. 'if scoped' is a "
          "ceiling, not a gain.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/od_full2.json")
    ap.add_argument("--category", default="open-domain")
    ap.add_argument("--limit", type=int, default=60)
    # A conversation is roughly 300 turns and they are laid end to end, so a
    # margin much above 100 starts admitting the neighbours this is meant to
    # exclude. Swept rather than assumed.
    ap.add_argument("--margins", type=int, nargs="+",
                    default=[0, 20, 40, 80, 150])
    # The span is defined by the turns of the top-k retrieved facts. Using all
    # of them makes it loose, because the tail of the retrieved list is exactly
    # where off-topic facts sit. 0 means all.
    ap.add_argument("--span-facts", type=int, nargs="+",
                    default=[3, 5, 8, 12, 0])
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--rerank", type=int, default=40)
    ap.add_argument("--facts", type=int, default=18)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--grounding", type=float, default=0.03)
    ap.add_argument("--scan", type=int, default=8)
    ap.add_argument("--hydrate", type=int, default=7)
    ap.add_argument("--overlap", type=float, default=0.8)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
