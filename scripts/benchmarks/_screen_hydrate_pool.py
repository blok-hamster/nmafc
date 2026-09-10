"""Does letting the question choose which turns get hydrated put the answer in?

The bucket this is aimed at is the largest one in the benchmark: 47 of 116
open-domain losses already had the answer in the prompt and lost anyway, because
extraction dropped the qualifier the question turns on. `gripping`, `two weeks
before 11 August`, `next month` -- with the qualifier gone, several stored facts
match the question equally well and the model picks one.

The dropped word is still in the turn. Hydration is the only mechanism that puts
turn text into the prompt, and as shipped it chooses turns by fact rank alone:
`records[:hydrate_top_k]`, hydrate whatever turns those facts came from. The
question is used to pick lines *within* a turn and never to pick *which* turns,
so the one signal that knows what the question is asking is spent on the smaller
decision.

`hydrate_pool` widens the set of turns considered without widening how many are
taken. Same number of turns, chosen by BM25 of the question against each.

**What is measured, and it is deliberately both halves.**

    gold in context, on the questions we lose      what it buys
    gold in context, on the questions we win       what it costs

A screen that watched only the losses could not see the turn it displaced, and
displacing a turn is exactly what this does -- there is no free slot to put a
better turn in. Both slices use the same detector, so its strictness cancels out
of the difference even where it is wrong about the level.

**About that detector.** It counts a hit when every content word of the gold
answer appears in the rendered context. That is strict on long golds and
generous on one-word ones, and it is not the judge. It is trustworthy for the
*difference* between two configurations of the same questions, which is all that
is asked of it here. `--show` prints examples, because a rate nobody has read
examples of is a rate nobody should act on.

Token width is reported per configuration rather than assumed. Holding the turn
count fixed does not hold the character count fixed: turns are not all the same
length, and a longer turn that answers the question is still a cost against a
1,000-token ceiling.

**Configurations are `rerank:facts:pool` triples, and they have to move
together.** The first attempt swept `pool` alone and measured nothing at all,
for a reason worth writing down: at the shipped `rerank_top_k=20`, and about six
facts per exchange, the retrieved pool is three to six distinct turns while fact
rank already hydrates three or four of them. There was nothing to choose from.

Widening `rerank_top_k` is free of tokens -- it is a retrieval depth, not a
render -- but only if `context_facts_top_k` then caps what is printed, and
`_source_turns` is passed the whole retrieved list rather than the printed one,
so the wider pool reaches hydration while the FACTS block stays the size it was
measured at. `40:20:40` is therefore the same number of printed facts and the
same number of hydrated turns as `20:none:0`, differing only in which turns.

Usage:
    python -u scripts/benchmarks/_screen_hydrate_pool.py --limit 20
    python -u scripts/benchmarks/_screen_hydrate_pool.py --configs 40:20:40,60:20:60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
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
from nmafc.integration.grounding import _STOPWORDS  # noqa: E402

from scripts.benchmarks._ab_budget import (  # noqa: E402
    close_readonly,
    open_memory,
    with_retries,
)

_WORD = re.compile(r"[a-z0-9']+")


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower())
            if w not in _STOPWORDS and (len(w) > 2 or w.isdigit())}


def present(gold: str, context: str) -> bool:
    """Every content word of the gold appears somewhere in the context."""
    wanted = content_words(gold)
    return bool(wanted) and wanted <= content_words(context)


async def contexts(memory, questions: list[str]) -> list[str]:
    turn = memory.current_turn + 1
    out: list[str] = []
    for question in questions:
        records = await with_retries(
            lambda q=question: memory._router.retrieve(q, turn))
        out.append(memory._router.format_context(records or [], question))
    return out


def measure(rows: list[dict], rendered: list[str]) -> tuple[set[int], float]:
    """Which questions have the gold in context, and the mean width.

    The set rather than the count, because a net of zero is two different
    results. 98 before and 98 after is either nothing moved or three golds were
    displaced and three others took their place, and only one of those is safe
    to ship. Every screen in this repo that reported a net has been wrong at
    least once for exactly this reason.
    """
    hits = {i for i, (r, c) in enumerate(zip(rows, rendered))
            if present(r["gold"], c)}
    width = sum(len(c) for c in rendered) / max(len(rendered), 1) / 4
    return hits, width


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no store at {store}")

    main = json.loads(Path(args.results).read_text(encoding="utf-8"))
    pool = [r for r in main if r["category"] == args.category]
    if args.gate != "all":
        # The two answer shapes want opposite budgets -- a single value wants
        # facts and is hurt by extra dialogue offering extra candidates, a
        # compound answer wants the dialogue. Screened together those cancel
        # and every configuration looks flat, which is what happened all
        # session. Screen them apart and each path can be priced separately;
        # facts and turns trade against each other at roughly 30t to 55t, so a
        # per-path budget is free.
        from scripts.benchmarks._run_open_domain_full import gate
        want = args.gate == "gated"
        pool = [r for r in pool if bool(gate(r["question"])[0]) == want]
    lost = [r for r in pool if not r["ours_ok"]]
    won = [r for r in pool if r["ours_ok"]]
    rng = random.Random(args.seed)
    rng.shuffle(lost)
    rng.shuffle(won)
    if args.limit:
        lost, won = lost[: args.limit], won[: args.limit]
    else:
        lost, won = lost[: args.lost_sample], won[: args.won_sample]

    ql = [r["question"] for r in lost]
    qw = [r["question"] for r in won]
    settings = ["shipped"] + args.configs.split(",")
    print(f"{args.category}: {len(lost)} we lose, {len(won)} we win, "
          f"{len(settings)} configurations, no generation\n")
    print(f"  {'rerank:facts:pool':<20}{'gold in ctx, losses':>22}"
          f"{'gold in ctx, wins':>21}{'ctx tokens':>13}")

    base_l = base_w = None
    for setting in settings:
        span_facts = span_margin = 0
        if setting == "shipped":
            rerank, facts, pool, ground, scan = args.budget, None, 0, 0.0, 0
        else:
            # Trailing fields are optional so that every configuration string
            # written before the field it names existed still means the same
            # thing. Fifth is `hydrate_scan`, sixth and seventh the scan span.
            parts = setting.split(":")
            a, b, c, d = parts[:4]
            scan = int(parts[4]) if len(parts) > 4 else 0
            span_facts = int(parts[5]) if len(parts) > 5 else 0
            span_margin = int(parts[6]) if len(parts) > 6 else 0
            rerank, facts = int(a), (None if b == "none" else int(b))
            pool, ground = int(c), float(d)
        memory = open_memory(store, llm, embedder, args.hot, args.cold,
                             rerank, args.max_hops, compact=True,
                             hydrate=args.hydrate, facts=facts,
                             lines=args.lines, whole=args.whole,
                             dedupe=args.dedupe, overlap=args.overlap)
        memory._router._config.hydrate_pool = pool
        memory._router._config.source_grounding = ground
        memory._router._config.hydrate_scan = scan
        memory._router._config.scan_span_facts = span_facts
        memory._router._config.scan_span_margin = span_margin
        memory._router._config.scan_semantic = args.semantic
        memory._router._config.scan_semantic_floor = args.semantic_floor
        try:
            rl = await contexts(memory, ql)
            rw = await contexts(memory, qw)
        finally:
            close_readonly(memory)

        hl, wl = measure(lost, rl)
        hw, ww = measure(won, rw)
        if base_l is None:
            base_l, base_w, first_l = hl, hw, rl
        gain = (f"{len(hl):>4} {100 * len(hl) / len(lost):>5.1f}%  "
                f"+{len(hl - base_l)}/-{len(base_l - hl)}")
        cost = (f"{len(hw):>4} {100 * len(hw) / len(won):>5.1f}%  "
                f"+{len(hw - base_w)}/-{len(base_w - hw)}")
        print(f"  {setting:<20}{gain:>22}{cost:>21}"
              f"{(wl * len(lost) + ww * len(won)) / (len(lost) + len(won)):>13.0f}")

        if args.show and setting != "shipped":
            newly = [(r, a, b) for r, a, b in zip(lost, first_l, rl)
                     if present(r["gold"], b) and not present(r["gold"], a)]
            print(f"\n    {len(newly)} losses where the gold entered the "
                  f"context at pool={setting}:")
            for r, _, _ in newly[: args.show]:
                print(f"      Q    {r['question'][:70]}")
                print(f"      gold {r['gold'][:70]}")
                print(f"      ours {r['ours_pred'][:70]}")
            # The displaced ones matter more than the gained ones, because a
            # question we already answer correctly is a question this can only
            # break. There is no free hydration slot; every turn chosen is a
            # turn dropped.
            gone = [won[i] for i in sorted(base_w - hw)]
            print(f"\n    {len(gone)} questions we WIN where the gold left "
                  f"the context:")
            for r in gone[: args.show]:
                print(f"      Q    {r['question'][:70]}")
                print(f"      gold {r['gold'][:70]}")
            print()

    print("\n  The gold reaching the context is not the gold being answered.")
    print("  A gain here still needs a paired generation A/B.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/haystack.json")
    ap.add_argument("--category", default="open-domain")
    ap.add_argument(
        "--configs",
        default="20:none:0:0.003,40:20:40:0,40:20:40:0.003",
        help="rerank_top_k:context_facts_top_k:hydrate_pool:source_grounding. "
             "The pool counts facts, and one exchange yields about six, so 40 "
             "facts is roughly 7-12 turns to choose from. `facts` caps what is "
             "printed, which is what keeps a wider retrieval free. The default "
             "is the decomposition: each mechanism alone, then both, because "
             "two changes that fix the same questions are worth one of them.")
    ap.add_argument("--gate", default="all",
                    choices=("all", "gated", "ungated"),
                    help="screen only questions where a length rule does or "
                         "does not fire; the two want opposite budgets")
    ap.add_argument("--lost-sample", type=int, default=120)
    ap.add_argument("--won-sample", type=int, default=120)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--hydrate", type=int, default=5)
    # A turn costs 80 to 150 tokens whole and a great deal less trimmed to its
    # best lines. Once the scan is choosing turns well, trading depth of turn
    # for number of turns is the obvious next question, and these two make it
    # askable. `None` and `0` are the shipped behaviour: whole turns.
    ap.add_argument("--lines", type=int, default=None,
                    help="Keep only this many best-matching lines per turn")
    ap.add_argument("--whole", type=int, default=0,
                    help="Exempt the top N turns from --lines trimming")
    # Measured at 38.8 characters per turn, 13% of the SOURCE block, and the
    # same string on every turn of a session. Deduping is lossless by
    # construction -- turns print in order, so a header still applies to
    # everything under it -- which makes this the only token saving on the table
    # that cannot cost a single answer. It has never been switched on.
    ap.add_argument("--dedupe", action="store_true",
                    help="Print a repeated session header once per run of turns")
    # Screened in an earlier session and parked because it saved 17 tokens out
    # of a context 457 under the ceiling. The ceiling moved to 1,000 and the
    # best-reaching config lands at 1,020, so 17 to 34 tokens is now the
    # difference between shipping it and not. It runs before the print limit,
    # so it buys distinct facts rather than fewer facts.
    ap.add_argument("--overlap", type=float, default=None,
                    help="Drop a printed fact this contained in a better-ranked "
                         "one. 0.8 held presence exactly; 0.7 saved more")
    # Ranking scanned turns by meaning as well as by words. Needs the store to
    # carry a turn index; without one this is silently inert, which is the right
    # behaviour and also a good way to measure nothing by accident, so the
    # config line printed below says which store was used.
    ap.add_argument("--semantic", type=float, default=0.0,
                    help="Weight of meaning against words for scanned turns")
    ap.add_argument("--semantic-floor", type=float, default=0.0,
                    help="Minimum cosine for a turn with no matching word")
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
