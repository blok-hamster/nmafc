"""NMAFC against RAG in one loop, so neither the clock nor the quota can lie.

Every comparison against RAG so far has been cross-run: our number came from
one day and RAG's from another. That is fatal for latency and merely unsafe for
accuracy. Two runs of the *identical* RAG arm came out at 4,694 ms and 2,460 ms,
91% apart, purely because the shared Azure deployment was busier on one of the
days. Any latency claim built on that comparison is measuring the provider's
mood. Accuracy drifts less but still drifts, by 1.5 to 2 points, which is wider
than the 1.0-point gap the cross-run tables show.

The fix is to stop comparing across time. Each question is answered by both
arms inside the same worker, back to back, seconds apart. The provider cannot
be congested for one of those calls and idle for the next, so ambient load
lands on both arms equally and cancels when the per-question difference is
taken. The reported latency figure is therefore the median of the paired
differences, not the difference of the two medians -- those are not the same
number, and only the first one is robust to a queue that drifts over an hour.

Which arm goes first alternates by question index. Whoever runs first pays for
any connection warm-up and whoever runs second may benefit from a warm route,
so a fixed order would hand one arm a systematic advantage; alternating spends
that bias equally in both directions.

Concurrency parallelises across *pairs*, never within one. Both halves of a
pair stay adjacent on the same worker, which is the entire basis of the
pairing. Raising `--concurrency` shortens the run and inflates both arms'
absolute latency together, leaving the paired difference intact.

Accuracy is scored at the same time because the answers are being generated
anyway, which makes this the same-session head-to-head that every previous
result has been missing. Two statistics are reported:

  accuracy   McNemar exact on the discordant pairs
  latency    sign test on which arm was faster per question, plus the median
             paired difference in milliseconds

Both arms read the finished stores from `--run`. Ours is reused as-is. RAG's
index is rebuilt from the dataset because the earlier RAG runs kept only their
results, not their stores -- that costs embedding calls and no LLM extraction,
so it is minutes rather than the 5.3 hours ingestion takes for us.

Our arm's context is configurable here, and that is what makes this the harness
for the decisive run rather than only the baseline one. `_ab_hydration.py` scores
a new context setting against *saved* answers, which is half the price and
carries the risk that killed its last attempt: the saved baseline had drifted 3.0
points by the time the candidate ran, and the effect being looked for is about
that size. Here both arms are generated today, so drift cannot enter, and no
gate is needed. It costs roughly twice as much and it is the only design whose
answer can be trusted.

The context flags default to the configuration this harness was written
against, so running it with no new flags reproduces that baseline exactly. The
stack has moved a long way since then -- semantic turn ranking, a hydration
pool, source grounding, fact separation, and a length-rule prompt that only
fires on short-answer questions -- and none of those were expressible here.
`--shipping` sets all of them at once, which is deliberate: the settled
configuration is nine numbers and a prompt, and typing eight of them correctly
produces a run that looks valid and measures something nobody chose. Any flag
given explicitly still wins, so `--shipping --hydrate 10` is a single named
departure from a known stack rather than a fresh guess.

Semantic turn ranking needs `turn_vectors` in the store. Every per-conversation
store here predates it, so run `_build_turn_index.py --store <path>` on each one
first; the startup line counts the index and says so when it is empty, because
otherwise the run is the old configuration wearing the new numbers.

Usage:
    python -u scripts/benchmarks/_ab_vs_rag.py --limit 10 --out C:/nmafc_ab/vs_rag_smoke.json
    python -u scripts/benchmarks/_ab_vs_rag.py --out C:/nmafc_ab/vs_rag.json

    # the settled stack, on the real per-conversation LoCoMo task
    python -u scripts/benchmarks/_ab_vs_rag.py --shipping \
        --out C:/nmafc_ab/vs_rag_shipping.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import tempfile
from collections import defaultdict
from math import comb
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

from scripts.benchmarks._ab_answer_type import PROMPT_B, answer_with  # noqa: E402
from scripts.benchmarks._ab_conversion import VARIANTS, parse_clauses  # noqa: E402
from scripts.benchmarks.arms.base import timer_split, timer_start  # noqa: E402
from scripts.benchmarks.arms.neuromorphic_tuned import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
)
from scripts.benchmarks.arms.rag import RagArm  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402
from scripts.benchmarks._ab_budget import (  # noqa: E402
    CHARS_PER_TOKEN,
    PERMANENT,
    SCORED,
    close_readonly,
    open_memory,
    with_retries,
)

# The settled configuration, as measured on the merged haystack: 79.4% against
# RAG's 80.6% on open-domain at 957 tokens to RAG's 1,459, and ahead of RAG on
# every other category. Kept as one object rather than as a row of argparse
# defaults so that "the shipping stack" is a thing that can be asked for by
# name, and so that changing the harness's historical defaults is not the price
# of being able to run it.
SHIPPING = {
    "budget": 40,
    "facts": 20,
    "pool": 40,
    "grounding": 0.03,
    "scan": 8,
    "hydrate": 6,
    "semantic": 12.0,
    "semantic_floor": 0.30,
    "overlap": 0.8,
    "dedupe_headers": True,
    "answer_type": True,
    "variant": "qualifier",
    "variant_where": "gated",
}


async def timed_ours(memory, llm, question: str, prompt: str, tag: str) -> dict:
    """Our arm, timed with quota-waiting split out of the work figure.

    The prompt and the answer-type tag are passed in rather than fixed here,
    because half the measured gain on short-answer questions lives in them and
    a harness that hard-codes the plain prompt cannot express the stack.
    """
    mark = timer_start()
    pred, chars = await answer_with(memory, llm, question, prompt, tag)
    work_ms, throttle_ms = timer_split(mark)
    return {"pred": pred, "chars": chars, "ms": work_ms, "throttle": throttle_ms}


async def timed_rag(arm: RagArm, question: str) -> dict:
    """RAG's own answer path, which already splits the quota wait itself."""
    response = await arm.answer_question(question)
    return {
        "pred": response.answer,
        "chars": response.context_tokens * CHARS_PER_TOKEN,
        "ms": response.latency_ms,
        "throttle": response.throttle_ms,
    }


def mcnemar(fixed: int, broke: int) -> float:
    """Exact two-sided binomial on the discordant pairs."""
    n = fixed + broke
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(fixed, broke) + 1))
    return min(1.0, 2 * tail / 2 ** n)


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    # Named `limit`, not `gate`: `gate` is the answer-type length rule imported
    # above, and shadowing it here would silently disable the prompt routing.
    limit = asyncio.Semaphore(args.concurrency)

    # Resume is keyed on the question, not on the configuration, so a results
    # file left by an earlier run resumes silently under whatever settings are
    # current -- and a complete one answers nothing and reports the old numbers
    # as the new configuration's. Fingerprint everything that changes an answer
    # and refuse to mix two of them in one file.
    fingerprint = {k: getattr(args, k) for k in (
        "run", "arm", "budget", "hot", "cold", "max_hops", "hydrate", "facts",
        "hydrate_lines", "hydrate_full_turns", "dedupe_headers", "pool",
        "grounding", "scan", "semantic", "semantic_floor", "overlap",
        "answer_type", "variant", "variant_where", "categories", "sample",
        "seed", "compact", "limit")}
    out = Path(args.out)
    stamp = out.with_suffix(".config.json")
    rows: list[dict] = []
    if out.is_file() and not args.restart:
        previous = (json.loads(stamp.read_text(encoding="utf-8"))
                    if stamp.is_file() else None)
        if previous != fingerprint:
            if previous is None:
                what = ("was written before configurations were recorded, so "
                        "there is no way to tell whether it matches")
            else:
                differs = sorted(k for k in fingerprint
                                 if previous.get(k) != fingerprint[k])
                what = ("was written under a different configuration: "
                        + ", ".join(f"{k} {previous.get(k)!r} -> "
                                    f"{fingerprint[k]!r}" for k in differs))
            raise SystemExit(
                f"{out} {what}.\nResuming would mix two configurations in one "
                f"results file. Use --restart to overwrite, or --out to write "
                f"somewhere new.")
        rows = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(rows)} questions already answered\n")
    stamp.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    done = {(r["conv"], r["question"]) for r in rows}
    blocked: list[tuple[str, str]] = []

    # Printed rather than stored, because the output file is a flat list of
    # rows and every reader of it assumes that. It belongs in the log next to
    # the numbers it produced.
    print(f"ours: top_k {args.hot}, cold {args.cold}, rerank {args.budget}, "
          f"hops {args.max_hops}")
    print(f"      facts {'all' if args.facts is None else args.facts}, "
          f"source turns {args.hydrate} from pool {args.pool} scan {args.scan}, "
          f"grounding {args.grounding}, overlap {args.overlap}, "
          f"lines {'whole' if args.hydrate_lines is None else args.hydrate_lines}"
          f", whole {args.hydrate_full_turns}, "
          f"dedupe headers {args.dedupe_headers}")
    print(f"      semantic {args.semantic} floor {args.semantic_floor}, "
          f"{'gated answer-type' if args.answer_type else 'plain prompt'}, "
          # Printed from the parsed pairs rather than the raw spec, so a log
          # says where each clause actually landed. Echoing the spec and then
          # the default placement reads as "commit_short on gated" for a run
          # that put it everywhere.
          f"{', '.join(f'{c} on {w}' for c, w in args.clauses)
             or 'no conversion rule'}")
    print(f"      categories {args.categories or 'all'}"
          f"{f', {args.sample} sampled per conversation' if args.sample else ''}\n")

    def checkpoint() -> None:
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            print(f"  [skip] no store for {conv.sample_id}")
            continue

        # `SCORED` is the default set, not a ceiling. It was both until now,
        # which meant `--categories adversarial` selected nothing and every
        # whole-benchmark table in this project quietly covered four of
        # LoCoMo's five categories -- the four we do best on. An explicit
        # request is honoured as given.
        wanted = (set(args.categories.split(",")) if args.categories
                  else set(SCORED))
        questions = [qa for qa in conv.qa_pairs if qa.category_name in wanted]
        if args.sample and len(questions) > args.sample:
            # Random rather than the first N, so a cheap run is a fair read of
            # the category and not a read of however LoCoMo happened to order
            # it. Seeded, so a resumed run samples the same questions.
            questions = sorted(
                random.Random(args.seed).sample(questions, args.sample),
                key=lambda qa: qa.question)
        if args.limit:
            questions = questions[: args.limit]
        questions = [qa for qa in questions
                     if (conv.sample_id, qa.question) not in done]
        if not questions:
            continue

        memory = open_memory(store, llm, embedder, args.hot, args.cold,
                             args.budget, args.max_hops,
                             compact=args.compact, hydrate=args.hydrate,
                             facts=args.facts, lines=args.hydrate_lines,
                             whole=args.hydrate_full_turns,
                             dedupe=args.dedupe_headers, overlap=args.overlap)
        # Settings newer than `open_memory`'s signature. Left alone when unset
        # so an old invocation still reproduces what it used to.
        for attribute, value in (("hydrate_pool", args.pool),
                                 ("source_grounding", args.grounding),
                                 ("hydrate_scan", args.scan),
                                 ("scan_semantic", args.semantic),
                                 ("scan_semantic_floor", args.semantic_floor)):
            if value is not None:
                setattr(memory._router._config, attribute, value)
        # Semantic ranking is the largest single retrieval gain in the stack and
        # it fails silently on a store with no turn index. Counted per store,
        # because these ten were ingested before the index existed.
        if args.semantic:
            indexed = len(getattr(memory._router._cold, "all_turn_vectors",
                                  dict)())
            if not indexed:
                print(f"  [{conv.sample_id}] NO TURN INDEX -- semantic ranking "
                      f"is inert here; run _build_turn_index.py --store "
                      f"{store}")
        rag_dir = tempfile.mkdtemp(prefix=f"vs_rag_{conv.sample_id}_")
        rag = RagArm(llm, embedder, storage_dir=rag_dir)
        print(f"[{conv.sample_id}] building RAG index "
              f"({len(questions)} questions to answer)")
        await with_retries(lambda: rag.ingest_conversation(conv.get_flat_history()))

        try:
            async def attempt(index: int, qa):
                # A length rule fires on the short-answer questions and on no
                # others, and the conversion clause goes only where it was
                # measured to help. Both are resolved before the timer starts
                # so the routing costs neither arm a millisecond.
                rule, tag = gate(qa.question) if args.answer_type else (None, "")
                prompt = PROMPT_B if rule else ANSWER_SYSTEM_PROMPT
                for clause, where in args.clauses:
                    if where == "everywhere" or (where == "gated") == bool(rule):
                        prompt = prompt + VARIANTS[clause]
                async with limit:
                    # Alternate who goes first so warm-up cost is shared.
                    if index % 2 == 0:
                        a = await with_retries(lambda: timed_ours(memory, llm, qa.question, prompt, tag))
                        b = await with_retries(lambda: timed_rag(rag, qa.question))
                    else:
                        b = await with_retries(lambda: timed_rag(rag, qa.question))
                        a = await with_retries(lambda: timed_ours(memory, llm, qa.question, prompt, tag))
                    ja = await with_retries(
                        lambda: judge_answer(qa.question, a["pred"], str(qa.answer), judge))
                    jb = await with_retries(
                        lambda: judge_answer(qa.question, b["pred"], str(qa.answer), judge))
                    return {
                        "conv": conv.sample_id,
                        "question": qa.question,
                        "category": qa.category_name,
                        "gold": str(qa.answer),
                        "first": "ours" if index % 2 == 0 else "rag",
                        "ours_pred": a["pred"], "rag_pred": b["pred"],
                        # `.correct`, not the object: judge_answer returns a
                        # JudgeResult, and truth-testing one is always True.
                        "ours_ok": ja.correct, "rag_ok": jb.correct,
                        "ours_ms": a["ms"], "rag_ms": b["ms"],
                        "ours_throttle": a["throttle"], "rag_throttle": b["throttle"],
                        "ours_chars": a["chars"], "rag_chars": b["chars"],
                    }

            async def one(index: int, qa):
                try:
                    return await attempt(index, qa)
                except Exception as exc:  # noqa: BLE001
                    # The content filter refuses a handful of LoCoMo questions
                    # outright. Dropping them from both arms keeps the pairing
                    # honest; killing the run over them costs everything.
                    if type(exc).__name__ not in PERMANENT:
                        raise
                    blocked.append((conv.sample_id, qa.question))
                    return None

            pending = [one(i, qa) for i, qa in enumerate(questions)]
            for start in range(0, len(pending), args.checkpoint_every):
                batch = await asyncio.gather(*pending[start:start + args.checkpoint_every])
                rows.extend(r for r in batch if r)
                checkpoint()
                print(f"  {len(rows)} answered")
        finally:
            close_readonly(memory)
            rag.reset()

    if not rows:
        print("nothing measured")
        return
    checkpoint()
    report(rows, blocked)


def report(rows: list[dict], blocked: list[tuple[str, str]]) -> None:
    n = len(rows)
    print(f"\n{'=' * 92}")
    print(f"paired questions : {n}")
    if blocked:
        print(f"blocked by filter: {len(blocked)} (dropped from both arms)")

    def block(label: str, subset: list[dict]) -> None:
        m = len(subset)
        if not m:
            return
        ours = sum(r["ours_ok"] for r in subset)
        rag = sum(r["rag_ok"] for r in subset)
        fixed = sum(1 for r in subset if r["ours_ok"] and not r["rag_ok"])
        broke = sum(1 for r in subset if r["rag_ok"] and not r["ours_ok"])
        p = mcnemar(fixed, broke)
        verdict = ("ours" if fixed > broke else "RAG") if p < 0.05 else "tie"
        print(f"  {label:13s} {m:5d}  ours {100 * ours / m:5.1f}%  "
              f"RAG {100 * rag / m:5.1f}%  {100 * (ours - rag) / m:+5.1f}  "
              f"p={p:<9.3g} {verdict}")

    print("\nACCURACY (same session, same judge, same questions)")
    block("OVERALL", rows)
    print()
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    # Whatever was actually asked, not whatever the default set names, so a
    # category outside `SCORED` still gets a line of its own.
    for cat in list(SCORED) + [c for c in by_cat if c not in SCORED]:
        block(cat, by_cat[cat])

    # Latency: the median of the per-question differences, plus a sign test on
    # which arm won each question. Comparing the two arms' separate medians
    # would reintroduce exactly the drift this harness exists to remove.
    diffs = [r["ours_ms"] - r["rag_ms"] for r in rows]
    ours_faster = sum(1 for d in diffs if d < 0)
    rag_faster = sum(1 for d in diffs if d > 0)
    p_lat = mcnemar(ours_faster, rag_faster)
    print(f"\nLATENCY (work only, provider quota waiting excluded)")
    print(f"  ours    median {statistics.median(r['ours_ms'] for r in rows):8.0f} ms   "
          f"mean {statistics.mean(r['ours_ms'] for r in rows):8.0f} ms   "
          f"throttle {statistics.mean(r['ours_throttle'] for r in rows):7.0f} ms")
    print(f"  RAG     median {statistics.median(r['rag_ms'] for r in rows):8.0f} ms   "
          f"mean {statistics.mean(r['rag_ms'] for r in rows):8.0f} ms   "
          f"throttle {statistics.mean(r['rag_throttle'] for r in rows):7.0f} ms")
    print(f"  paired  median difference {statistics.median(diffs):+8.0f} ms "
          f"(negative means we are faster)")
    print(f"  ours faster on {ours_faster} questions, RAG faster on {rag_faster}, "
          f"sign test p={p_lat:.3g}")
    if p_lat >= 0.05:
        print("  -> no reliable latency difference")
    else:
        print(f"  -> {'ours' if ours_faster > rag_faster else 'RAG'} is reliably faster")

    print(f"\nCONTEXT")
    print(f"  ours    {statistics.mean(r['ours_chars'] for r in rows) / CHARS_PER_TOKEN:8.0f} tokens")
    print(f"  RAG     {statistics.mean(r['rag_chars'] for r in rows) / CHARS_PER_TOKEN:8.0f} tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--shipping", action="store_true",
                    help="Use the settled configuration for every context and "
                         "prompt setting not given explicitly on the command "
                         "line. Without it the historical defaults apply.")
    # `None` rather than the old number, so `--shipping` can tell "unset" from
    # "asked for the old value" and an explicit flag always wins.
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=None)
    ap.add_argument("--facts", type=int, default=None,
                    help="Facts written into the FACTS block. Unset writes "
                         "every retrieved fact, which is what shipped does and "
                         "what every earlier measurement here used.")
    ap.add_argument("--hydrate-lines", type=int, default=None,
                    help="Keep only the N best-matching speaker lines of each "
                         "hydrated turn. Unset keeps the whole turn.")
    ap.add_argument("--hydrate-full-turns", type=int, default=0,
                    help="Top-ranked turns kept whole before --hydrate-lines "
                         "applies to the rest.")
    ap.add_argument("--dedupe-headers", action="store_true",
                    help="Print a repeated session header once per run of "
                         "turns instead of on every turn.")
    ap.add_argument("--pool", type=int, default=None,
                    help="Source turns considered before hydration picks. "
                         "Unset leaves the library default.")
    ap.add_argument("--grounding", type=float, default=None,
                    help="Weight on a fact's own source turns when ranking "
                         "the hydration pool.")
    ap.add_argument("--scan", type=int, default=None,
                    help="Turns scanned per fact when filling the pool.")
    ap.add_argument("--semantic", type=float, default=None,
                    help="Weight on meaning, not words, when ranking turns "
                         "for a hydration slot. Needs a turn index in the "
                         "store or it does nothing.")
    ap.add_argument("--semantic-floor", type=float, default=None,
                    help="Similarity below which the semantic term is "
                         "ignored. Measured never to bind at 0.30.")
    ap.add_argument("--overlap", type=float, default=None,
                    help="Drop a printed fact wholly restated by a "
                         "better-ranked one at this overlap.")
    ap.add_argument("--answer-type", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Route short-answer questions through the length "
                         "rules. Off reproduces the plain prompt.")
    ap.add_argument("--variant", default=None,
                    help="Conversion clauses from _ab_conversion.py, "
                         "comma-separated, each optionally as name@where "
                         "where `where` is ungated, gated or everywhere. A "
                         "bare name takes --variant-where. So "
                         "`qualifier@gated,commit_short@everywhere` is two "
                         "clauses on two different paths, and `qualifier` "
                         "alone is what it always was.")
    ap.add_argument("--variant-where", default=None,
                    choices=("ungated", "gated", "everywhere"),
                    help="Placement for any clause given without an @. "
                         "Measured to help on gated and to churn on ungated.")
    ap.add_argument("--categories", default="",
                    help="Comma-separated categories to score. Empty scores "
                         "all of them.")
    ap.add_argument("--sample", type=int, default=0,
                    help="Random questions per conversation, after the "
                         "category filter. 0 takes all of them.")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--compact", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Parallel question PAIRS. Never splits a pair.")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="C:/nmafc_ab/vs_rag.json")
    args = ap.parse_args()

    # Historical defaults for the settings that predate `--shipping`, so a bare
    # run of this file keeps meaning what it meant. `--shipping` overlays the
    # settled configuration on top of those, and an explicit flag beats both.
    defaults = {"budget": 20, "hydrate": 5, "answer_type": False,
                "variant_where": "gated"}
    if args.shipping:
        defaults = {**defaults, **SHIPPING}
    for key, value in defaults.items():
        # `--dedupe-headers` is a plain store_true, so "not given" reads as
        # False rather than as None. Nothing here ever wants it off once asked
        # for, so treating False as unset is safe for this one key.
        unset = (not args.dedupe_headers if key == "dedupe_headers"
                 else getattr(args, key) is None)
        if unset:
            setattr(args, key, value)

    # Parsed once here rather than per question, so a bad clause name fails
    # before the run opens a store instead of two hours into it.
    args.clauses = parse_clauses(args.variant, args.variant_where)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
