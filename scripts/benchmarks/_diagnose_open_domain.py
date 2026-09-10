"""Of the open-domain questions arm B still loses, which ones had the answer?

The paid A/B moved open-domain from 75.9% to 77.9%. RAG is at 80.6%. That gap is
about seven questions out of 249, and this script says which seven are worth
attacking by splitting the 55 remaining losses along the only line that matters
for what to build next:

    reach       the gold never arrived in the rendered context. Nothing a prompt
                can do. The fix is retrieval, hydration, or extraction.

    conversion  the gold was in the rendered context and the model answered
                something else. Retrieval is finished with this question. The
                fix is in the prompt or in how the context is laid out.

The two demand opposite work, and the whole history of this benchmark is people
(me included) building a retrieval mechanism for a conversion failure. 47 of the
original 116 losses were conversion, so the prior is that this is where the
remaining gap lives.

The detector is the same one the free screens use: every content word of the gold
appears somewhere in the rendered context. It is strict on long golds -- a gold
of eight words needs all eight present -- so `reach` is over-counted and
`conversion` under-counted. That bias is in the safe direction: it will not
invent conversion failures, only miss some. `--show` prints them, because the
point is to read the questions, not the rate.

**A second cut, because the arithmetic of the categories suggests it.** Temporal
is our strongest category at +19.6 over RAG and open-domain is our only losing
one, yet a large share of open-domain questions carry a date qualifier ("on 9th
December 2023", "in September 2022"). If the date-qualified losses convert worse
than the rest, the fix is to route them the way temporal questions are routed,
and that is a free thing to try before spending on generation again.

Read-only. No generation, so no judge and no answer calls: the only spend is one
embedding per question per configuration.

Usage:
    python -u scripts/benchmarks/_diagnose_open_domain.py --show 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

from nmafc.integration.answer_type import gate  # noqa: E402
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

# A date qualifier in the question. Any of "9th December 2023", "December 2023",
# "13 March, 2023" or a bare year, which is how LoCoMo writes them.
_DATED = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"[\s,]+\d{4}\b|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def overlap(gold: str, context: str) -> float:
    """How much of the gold reached the context, when not all of it did.

    `present` is all-or-nothing, which cannot tell a gold that missed by one
    word from one that never showed up. This does, and the difference decides
    whether the answer is a rendering change or a retrieval change.
    """
    wanted = content_words(gold)
    if not wanted:
        return 0.0
    return len(wanted & content_words(context)) / len(wanted)


async def render(memory, questions: list[str]) -> list[str]:
    turn = memory.current_turn + 1
    out: list[str] = []
    for question in questions:
        records = await with_retries(
            lambda q=question: memory._router.retrieve(q, turn))
        out.append(memory._router.format_context(records or [], question))
    return out


def split(rows: list[dict], rendered: list[str]) -> tuple[list, list]:
    reach, conv = [], []
    for row, ctx in zip(rows, rendered):
        (conv if present(row["gold"], ctx) else reach).append((row, ctx))
    return reach, conv


def table(label: str, rows: list[dict], reach: list, conv: list) -> None:
    n = max(len(rows), 1)
    print(f"  {label:<22}{len(rows):>6}{len(conv):>10} "
          f"{100 * len(conv) / n:>5.1f}%{len(reach):>10} "
          f"{100 * len(reach) / n:>5.1f}%")


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no store at {store}")

    paid = json.loads(Path(args.results).read_text(encoding="utf-8"))
    lost = [r for r in paid if not r["b_ok"]]
    print(f"{len(paid)} open-domain questions answered by both arms, "
          f"{len(lost)} still lost by B\n")

    memory = open_memory(store, llm, embedder, args.hot, args.cold, args.rerank,
                         args.max_hops, compact=True, hydrate=args.hydrate,
                         facts=args.facts)
    memory._router._config.hydrate_pool = args.pool
    memory._router._config.source_grounding = args.grounding
    memory._router._config.hydrate_scan = args.scan
    try:
        rendered = await render(memory, [r["question"] for r in lost])
    finally:
        close_readonly(memory)

    reach, conv = split(lost, rendered)
    print(f"  {'':<22}{'n':>6}{'conversion':>16}{'reach':>16}")
    print("  " + "-" * 60)
    table("all B losses", lost, reach, conv)

    dated = [r for r in lost if _DATED.search(r["question"])]
    plain = [r for r in lost if not _DATED.search(r["question"])]
    d_ctx = [c for r, c in zip(lost, rendered) if _DATED.search(r["question"])]
    p_ctx = [c for r, c in zip(lost, rendered)
             if not _DATED.search(r["question"])]
    dr, dc = split(dated, d_ctx)
    pr, pc = split(plain, p_ctx)
    table("date-qualified", dated, dr, dc)
    table("no date", plain, pr, pc)

    typed = [r for r in lost if gate(r["question"])[1]]
    t_ctx = [c for r, c in zip(lost, rendered) if gate(r["question"])[1]]
    tr, tc = split(typed, t_ctx)
    table("type-demanding", typed, tr, tc)

    # How close the reach failures came. A gold missing one content word of five
    # is a rendering problem; a gold missing all five was never retrieved, and
    # only the second one is a case for more retrieval.
    near = [(overlap(r["gold"], c), r, c) for r, c in reach]
    near.sort(reverse=True, key=lambda t: t[0])
    almost = [t for t in near if t[0] >= args.near]
    print(f"\n  {len(almost)} of the {len(reach)} reach failures have "
          f">={args.near:.0%} of the gold's words in context already.")

    print(f"\n{'=' * 70}\nCONVERSION: gold in the prompt, B answered otherwise")
    for row, ctx in conv[: args.show]:
        print(f"\n  Q     {row['question']}")
        print(f"  gold  {row['gold']}")
        print(f"  A     {row['a'][:110]}")
        print(f"  B     {row['b'][:110]}")
        if args.context:
            print("  ---- rendered context ----")
            print("  " + ctx[: args.context].replace("\n", "\n  "))

    print(f"\n{'=' * 70}\nREACH: closest misses first")
    for score, row, _ in near[: args.show]:
        missing = content_words(row["gold"]) - content_words(_)
        print(f"\n  {score:.0%} Q     {row['question']}")
        print(f"      gold  {row['gold']}")
        print(f"      B     {row['b'][:110]}")
        print(f"      missing from context: {sorted(missing)}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "conversion": [r for r, _ in conv],
            "reach": [r for r, _ in reach],
        }, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/open_domain.json")
    # Arm B's settings, so what is rendered here is what was answered there.
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--rerank", type=int, default=40)
    ap.add_argument("--facts", type=int, default=20)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--grounding", type=float, default=0.003)
    ap.add_argument("--scan", type=int, default=0)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--near", type=float, default=0.5)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--context", type=int, default=0,
                    help="Print this many characters of the rendered context "
                         "for each conversion failure. 0 prints none.")
    ap.add_argument("--out", default="C:/nmafc_ab/od_diagnosis.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
