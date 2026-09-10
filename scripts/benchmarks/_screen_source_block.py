"""How many answers are in the SOURCE block that no prompt has ever mentioned?

The rendered context has two halves. `<FACTS>` is what the extractor wrote at
ingestion; `<SOURCE>` is verbatim dialogue, hydrated at query time, and it is
between a fifth and a third of the tokens we spend.

Every prompt in this repo describes the first half only:

    "You have a knowledge graph of facts from past conversations, shown in
     <FACTS> tags. Answer the question using these facts."

    "COPY THE WORDING FROM THE FACTS. Reuse the exact nouns and adjectives that
     appear in the retrieved facts rather than substituting your own."

So the model is told what the facts are, told to answer from them, and told to
take its wording from them, while the SOURCE block sits underneath unannounced
and, on the wording rule, actively disfavoured. That is a plausible explanation
for the shape of the open-domain losses, where a third of what we get wrong has
the gold in the prompt and our answer is a fact-shaped paraphrase of it: gold
"embracing the creative process without restraint", ours "spontaneous strokes
and bold colors".

Plausible is not measured, and a prompt change cannot be screened without
generating. What *can* be screened for free is the size of the prize, by
rendering the two blocks separately and asking where each gold actually is:

    gold in SOURCE only, and we lost      what a source-aware prompt could win
    gold in FACTS only, and we won        what it could cost, if the model
                                          swung the other way
    gold in both                          untouched either way
    gold in neither                       a reach failure, out of scope here

The first two numbers bracket the paid run. If the first is small there is
nothing to buy and the credit is better spent elsewhere.

Same all-content-words detector as the other screens, with the same bias: strict
on long golds, so `neither` is over-counted and the two interesting cells are
under-counted. Under-counting the prize is the safe direction for a decision
about whether to spend.

Read-only, no generation. One embedding per question.

Usage:
    python -u scripts/benchmarks/_screen_source_block.py --show 10
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
from scripts.benchmarks._screen_hydrate_pool import present  # noqa: E402


def halves(context: str) -> tuple[str, str]:
    """The FACTS block and the SOURCE block, separately.

    Split on the tags rather than re-rendering each half, so what is measured is
    the string the model was actually given, including the case where one half
    is missing entirely.
    """
    facts = source = ""
    if "<FACTS>" in context:
        facts = context.split("<FACTS>", 1)[1].split("</FACTS>", 1)[0]
    if "<SOURCE>" in context:
        source = context.split("<SOURCE>", 1)[1].split("</SOURCE>", 1)[0]
    return facts, source


async def render(memory, questions: list[str]) -> list[str]:
    turn = memory.current_turn + 1
    out: list[str] = []
    for question in questions:
        records = await with_retries(
            lambda q=question: memory._router.retrieve(q, turn))
        out.append(memory._router.format_context(records or [], question))
    return out


def where(row: dict, context: str) -> str:
    facts, source = halves(context)
    in_f, in_s = present(row["gold"], facts), present(row["gold"], source)
    if in_f and in_s:
        return "both"
    if in_s:
        return "source only"
    if in_f:
        return "facts only"
    return "neither"


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no store at {store}")

    paid = json.loads(Path(args.results).read_text(encoding="utf-8"))
    memory = open_memory(store, llm, embedder, args.hot, args.cold, args.rerank,
                         args.max_hops, compact=True, hydrate=args.hydrate,
                         facts=args.facts)
    memory._router._config.hydrate_pool = args.pool
    memory._router._config.source_grounding = args.grounding
    # This screen priced the "prefer SOURCE" rule once already, at hydrate 5
    # with no scan, and found it a bad bet. The scan changed what SOURCE
    # contains -- turns chosen by the question rather than by fact rank -- so
    # that price is stale for any configuration that turns it on, and a stale
    # price is worse than none. Re-run this before quoting the old numbers.
    memory._router._config.hydrate_scan = args.scan
    try:
        rendered = await render(memory, [r["question"] for r in paid])
    finally:
        close_readonly(memory)

    tagged = [(r, c, where(r, c)) for r, c in zip(paid, rendered)]
    cells = ["source only", "both", "facts only", "neither"]

    print(f"\n{len(paid)} open-domain questions, arm B's retrieval, "
          f"where the gold actually is\n")
    print(f"  {'':<14}{'we won':>10}{'we lost':>10}{'total':>10}")
    print("  " + "-" * 44)
    for cell in cells:
        won = sum(1 for r, _, w in tagged if w == cell and r["b_ok"])
        lost = sum(1 for r, _, w in tagged if w == cell and not r["b_ok"])
        print(f"  {cell:<14}{won:>10}{lost:>10}{won + lost:>10}")

    prize = [(r, c) for r, c, w in tagged if w == "source only" and not r["b_ok"]]
    risk = [(r, c) for r, c, w in tagged if w == "facts only" and r["b_ok"]]
    n = len(paid)
    print(f"\n  Upside if a source-aware prompt converted every one: "
          f"+{100 * len(prize) / n:.1f} points ({len(prize)} questions)")
    print(f"  Downside if it lost every facts-only win: "
          f"-{100 * len(risk) / n:.1f} points ({len(risk)} questions)")
    print("  Neither bound will happen. They say whether the run is worth "
          "buying,\n  not what it will return.")

    # The token share, because the argument for telling the model about SOURCE
    # is also an argument for it being worth what it costs.
    share = [len(halves(c)[1]) / max(len(c), 1) for c in rendered]
    print(f"\n  SOURCE is {100 * sum(share) / len(share):.0f}% of the rendered "
          f"context by characters.")

    print(f"\n{'=' * 70}\nLOST, gold in SOURCE and not in FACTS:")
    for row, _ in prize[: args.show]:
        print(f"\n  Q     {row['question']}")
        print(f"  gold  {row['gold']}")
        print(f"  ours  {row['b'][:100]}")

    print(f"\n{'=' * 70}\nWON, gold in FACTS and not in SOURCE (the risk):")
    for row, _ in risk[: args.show]:
        print(f"\n  Q     {row['question']}")
        print(f"  gold  {row['gold']}")
        print(f"  ours  {row['b'][:100]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/open_domain.json")
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--rerank", type=int, default=40)
    ap.add_argument("--facts", type=int, default=20)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--grounding", type=float, default=0.003)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--scan", type=int, default=0)
    ap.add_argument("--show", type=int, default=8)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
