"""What is the cheapest context that holds the open-domain answers RAG gets and we miss?

Open-domain decides the benchmark. It is 839 of 1,535 scored questions, we are
116 behind RAG on it and 63 ahead, and it is the only category costing us the
overall result. At parity the overall goes 65.1% to 68.6% against RAG's 64.4%,
turning a +0.7 that no test can distinguish from noise into a +4.2.

The hypothesis, and it is a hypothesis until this script runs. Our facts are
summaries. `_probe_stored_vs_reached.py` shows extraction losing exactly the
details a question asks for: a store holding "Melanie has a dog and a cat"
against a gold of two cats, Chicago present only as a place somebody suggested
and never as a place visited. Summaries are the right shape for temporal, where
we win by 18.7 because extraction adds dated structure, and the wrong shape for
a question about a specific detail. RAG has no such problem: it never summarises
anything, it hands over the transcript.

If that is right, the answering text is sitting in `turn_text` unread. The
router already knows how to fetch it -- `_source_turns` hydrates the verbatim
turns behind the best-ranked facts -- and `hydrate_top_k` is set to 5. We are
also spending 994 context tokens against RAG's 1,451, so there is room to read
more of the source and still be the cheaper arm.

An earlier version of this script swept hydration depth alone and answered that
much: depth 20 puts the gold in front of the model for 51.7% of the losses
against 40.5% shipped, and the generation A/B turned that into 78.2% against
74.3%. It also costs 1,847 tokens against RAG's 1,459, which is a configuration
we cannot ship. More accurate and more expensive is not a win.

So this version sweeps both halves of the context separately, `facts:lines`.
The router will not do that on its own -- it hydrates `records[:hydrate_top_k]`
out of the same list it renders, so cutting facts cuts source with it and the
two knobs behave as one. `compose` below separates them, and the reason to
separate them is that the context pays twice for the same information. Every
fact is a summary of a turn, and at depth 20 the facts cost about 619 tokens
and the turns behind them about 1,250. The turns are the half that wins: they
still carry the qualifier -- gripping, next month, two weeks before 11 August --
that extraction stripped out and that decides the answer. If six facts over
twenty turns of source holds the answer as often as twenty over twenty, the
deep-hydration result is available at roughly RAG's price.

Presence is measured only on the questions we lose, and that is deliberate.
Hydration appends, so presence is monotonic in depth and measuring it on
questions already won can only produce a flat line. Presence is also not
accuracy: it says the answer was in the prompt, not that the model picked it,
so anything that wins here still has to survive a generation A/B.

Stores are opened read-only. One embedding per question and one judge call per
setting per question.

Usage:
    python -u scripts/benchmarks/_sweep_hydration.py --limit 5
    python -u scripts/benchmarks/_sweep_hydration.py --out C:/nmafc_ab/hydration.json
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
from scripts.benchmarks._sweep_context_budget import retrieve_with_retry  # noqa: E402

JUDGE_SYSTEM = """\
You decide whether a piece of information is present in some text. You are not
answering the question and not judging whether anything is true.

You get a QUESTION, the ANSWER that is correct, and a CONTEXT. Decide whether
the CONTEXT states the ANSWER to the QUESTION. Count it as present when the
wording differs but the information is the same. Do not count it as present when
the context is merely about the same topic, gives a different value, or would
require guessing.

Reply with JSON and nothing else: {"present": true|false}"""


def parse(text: str) -> tuple[int, int, int, int, int, int]:
    """`facts:turns[:lines[:dedupe[:whole[:overlap]]]]`, padded with the shipped
    behaviour.

    `lines` 0 means the whole turn, which is what hydration did before the turn
    could be trimmed, `dedupe` 0 means every repeated session header is printed,
    `whole` 0 means no turn is exempt from trimming, and `overlap` 0 means no
    pattern separation. So a two-field setting means exactly what it meant before
    this argument grew, and every earlier sweep in the log is still readable.

    `overlap` is a percentage rather than a fraction so the notation stays
    integers throughout: 80 is a containment threshold of 0.80.
    """
    parts = [int(x) for x in text.split(":")]
    parts += [0] * (6 - len(parts))
    return tuple(parts[:6])  # type: ignore[return-value]


def compose(router, pool, question: str, setting: tuple[int, ...]) -> str:
    """A context holding `facts` rendered facts and `lines` hydrated turns.

    `format_context` cannot express this on its own. It renders every record it
    is handed and then hydrates `records[:hydrate_top_k]` out of that same list,
    so the source depth is capped by the fact count and cutting facts silently
    cuts source with it. Shipped, that makes the two knobs one knob.

    They should not be one knob. A fact is a summary of a turn, so a context
    carrying both pays twice for the same information: at depth 20 the facts
    cost about 619 tokens and the turns behind them about 1,250. The turns are
    the half that wins -- they keep the qualifier extraction stripped out -- and
    the facts are the half we can afford to lose. Rendering six facts over
    twenty turns of source buys the deep-hydration result at roughly RAG's
    price, which is the only shape that satisfies "more accurate and not more
    expensive".

    `context_facts_top_k` is what separates them, so this goes through
    `format_context` rather than around it: the string measured here is the
    string the shipped answer path would build at the same settings.
    """
    facts, turns, lines, dedupe, whole, overlap = setting
    config = router._config
    config.context_facts_top_k = facts
    config.hydrate_top_k = turns
    config.hydrate_lines = lines or None
    config.dedupe_source_headers = bool(dedupe)
    config.hydrate_full_turns = whole
    # Pattern separation, and note what it cannot do here: it only ever removes
    # rendered fact lines, and `_source_turns` reads the unseparated list, so a
    # setting with it on is a strict subset of the same setting with it off. Its
    # presence score therefore cannot rise, only hold or fall. A flat presence
    # line at a lower token count is the result worth having; a fallen one means
    # it is deleting facts that carried the answer.
    config.fact_overlap_max = (overlap / 100) if overlap else None
    return router.format_context(pool, question) if pool else ""


async def judge(llm, question: str, gold: str, context: str) -> bool | None:
    prompt = (f"QUESTION: {question}\nANSWER: {gold}\n\n"
              f"CONTEXT:\n{context or '(nothing)'}")
    text = await llm.chat(messages=[{"role": "user", "content": prompt}],
                          system_prompt=JUDGE_SYSTEM)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        # Unparseable is unknown, not absent. Recording it as absent would make
        # every parse failure look like evidence against hydration.
        return bool(json.loads(text[start:end + 1]).get("present"))
    except json.JSONDecodeError:
        return None


async def run(args: argparse.Namespace) -> None:
    answered = json.loads(Path(args.results).read_text(encoding="utf-8"))
    items = [r for r in answered if r["category"] == args.category]
    if args.select == "losses":
        items = [r for r in items if r["rag_ok"] and not r["ours_ok"]]
    if args.limit:
        items = items[: args.limit]
    # "facts:lines". Both matter and only sweeping one of them misses the
    # cheapest configurations entirely. A fact is a summary of a line, so
    # sending twenty facts and the lines behind them pays twice for the same
    # information: at depth 20 the facts cost about 619 tokens and the source
    # about 1,250. Spending fact slots on source lines is the only way to get
    # more of what wins without spending more in total.
    labels = args.settings.split(",")
    settings = {label: parse(label) for label in labels}

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    gate = asyncio.Semaphore(args.concurrency)

    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no store at {store}")

    widest = max(max(s[0], s[1]) for s in settings.values())
    print(f"{len(items)} {args.category} questions"
          + (" RAG answered and we did not" if args.select == "losses"
             else ", every one of them"))
    print(f"settings facts:turns:lines:dedupe {args.settings}, retrieval only\n")

    memory = open_memory(store, llm, embedder, args.hot, args.cold, widest,
                         args.max_hops, compact=True, hydrate=widest)
    rows: list[dict] = []
    try:
        router = memory._router
        turn = memory.current_turn + 1

        async def one(item: dict) -> dict:
            pool = await retrieve_with_retry(router, item["question"], turn)
            pool = pool or []
            row = {"conv": item["conv"], "question": item["question"],
                   "gold": item["gold"]}
            # Every depth is formatted first, with no await in between.
            # Hydration depth is read off the config inside `format_context`,
            # and the retrieved pool does not depend on it, so the same pool is
            # formatted at each depth and the comparison is paired to the
            # record. Building the contexts before judging any of them keeps
            # the mutation of that shared field off the event loop, which is
            # what makes it safe to judge them concurrently.
            contexts = {}
            for key, setting in settings.items():
                # Slicing the pool stands in for a smaller `rerank_top_k`. The
                # ranking is the same order either way, so the top twelve of a
                # top-twenty are the twelve a top-twelve would have returned.
                contexts[key] = compose(router, pool, item["question"], setting)
                row[f"chars@{key}"] = len(contexts[key])

            if args.cost_only:
                return row

            async def ask(key: str):
                async with gate:
                    return key, await with_retries(
                        lambda: judge(llm, item["question"], item["gold"],
                                      contexts[key]))

            for key, verdict in await asyncio.gather(
                    *(ask(k) for k in contexts)):
                row[f"present@{key}"] = verdict
            return row

        # Sequential over questions, concurrent over nothing else: the depths
        # share one mutable config field, so two questions in flight at once
        # would read each other's depth.
        for index, item in enumerate(items, 1):
            rows.append(await one(item))
            if index % args.checkpoint_every == 0:
                print(f"  {index}/{len(items)}")
                if args.out:
                    Path(args.out).write_text(json.dumps(rows, indent=2),
                                              encoding="utf-8")
    finally:
        close_readonly(memory)

    if not rows:
        print("nothing measured")
        return

    n = len(rows)
    if args.cost_only:
        print(f"\n{'=' * 62}")
        print(f"  context cost over {n} questions, nothing judged")
        print(f"  RAG spends {args.rag_tokens} on these\n")
        print(f"  {'setting':>18s} {'context':>9s} {'headroom':>9s}")
        for key in sorted(settings, key=lambda k: -sum(
                r[f"chars@{k}"] for r in rows)):
            tokens = sum(r[f"chars@{key}"] for r in rows) // n // 4
            room = args.rag_tokens - tokens
            print(f"  {key:>18s} {tokens:8d}t "
                  f"{room:+8d}t{'' if room >= 0 else '   OVER'}")
        print("\n  Judge only the ones with headroom to spare.")
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=2),
                                      encoding="utf-8")
        return

    shipped = f"{args.shipped}"
    base = sum(1 for r in rows if r.get(f"present@{shipped}")) \
        if f"chars@{shipped}" in rows[0] else None
    print(f"\n{'=' * 82}")
    print(f"  gold answer present in the prompt, {n} questions we currently lose")
    print(f"  RAG spends {args.rag_tokens} tokens on these and gets them right; "
          f"anything above that line is\n  a configuration we cannot ship.\n")
    print(f"  {'setting':>18s} {'present':>8s} {'share':>8s} "
          f"{'vs shipped':>11s} {'context':>9s}")
    hits_for = {k: sum(1 for r in rows if r.get(f"present@{k}")) for k in settings}
    cost_for = {k: sum(r[f"chars@{k}"] for r in rows) // n // 4 for k in settings}
    for key in sorted(settings, key=lambda k: -hits_for[k]):
        hits = hits_for[key]
        delta = f"{100 * (hits - base) / n:+10.1f}" if base is not None else "-"
        flag = "" if cost_for[key] <= args.rag_tokens else "   OVER BUDGET"
        print(f"  {key:>18s} {hits:8d} {100 * hits / n:7.1f}% "
              f"{delta} {cost_for[key]:8d}t{flag}")

    affordable = [k for k in settings if cost_for[k] <= args.rag_tokens]
    if affordable and base is not None:
        best = max(affordable, key=lambda k: hits_for[k])
        print(f"\n  best inside {args.rag_tokens} tokens: {best}, "
              f"{100 * (hits_for[best] - base) / n:+.1f} points of presence "
              f"over shipped, at {cost_for[best]}t")

    unparsed = sum(1 for r in rows for k in settings
                   if r.get(f"present@{k}") is None)
    if unparsed:
        print(f"\n  {unparsed} unparsed verdicts, counted as absent nowhere")
    print("\n  These are questions RAG got right and we got wrong. Gold that is")
    print("  absent at every depth was never written down in any form, and no")
    print("  amount of hydration reaches it -- that share is the extraction")
    print("  bill. Gold that appears as depth rises is recoverable for tokens.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="C:/nmafc_ab/haystack.json",
                    help="A finished paired run, used to pick the losses.")
    ap.add_argument("--category", default="open-domain")
    ap.add_argument("--select", choices=("losses", "all"), default="losses",
                    help="Which questions to sweep. Losses is what presence "
                         "is for. Pricing needs `all`: the losses are the "
                         "questions with the widest source blocks, so their "
                         "token cost is the worst case and not the bill.")
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--settings",
                    default="20:5,20:20,16:16,12:12,12:20,10:20,8:20,6:20,"
                            "4:20,2:20,0:20,0:16,6:16",
                    help="facts:lines pairs. Fewer facts pays for more lines.")
    ap.add_argument("--shipped", default="20:5",
                    help="The setting everything else is compared against.")
    ap.add_argument("--cost-only", action="store_true",
                    help="Price the settings without judging them. Retrieval "
                         "and formatting are local, so this is one embedding "
                         "per question and no generation at all -- run it "
                         "first and only judge the settings that fit.")
    ap.add_argument("--rag-tokens", type=int, default=1459,
                    help="RAG's measured context in this category. Anything "
                         "above it cannot ship.")
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="C:/nmafc_ab/hydration.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
