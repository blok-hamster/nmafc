"""Two context configurations, one store, one session, same prompt.

The screen this exists for: 26 open-domain questions that were right at twenty
facts broke when the fact list was cut to six, and 26 questions is the whole
remaining 3.2-point gap against RAG. Putting facts back is the obvious answer
and it is genuinely two-sided, because 17 of the questions the cut *fixed* were
fixed by deleting a competing summary the model had been answering from. Adding
facts should re-break some of those. Which effect is larger is not predictable
from the presence probes, because presence cannot see either one: the text was
already in the prompt in both directions.

So it has to be generated, and both arms have to be generated together --
cross-run drift on this benchmark is 1.5 to 2 points and the effect being looked
for is about that size. That is what this harness is for and what makes it
different from `_ab_hydration.py`, which compares one live arm against saved
answers and needs a drift check to license it.

Cost is scored, not just accuracy. RAG's tightest per-category spend on the
merged store is 1,395 tokens on multi-hop, and every candidate has to sit under
it in *every* category, not on average. `12:24:1:1:4` prices at 1,395 exactly,
with no headroom, so a B arm that measures over is disqualified whatever it
scores.

Read it as a screen. Both arms are ours, so there is no RAG generation to pay
for and a 400-question sample is about half the price of the full paired run. If
B does not clear the noise floor, the candidate stands and nothing further is
spent.

Usage:
    python -u scripts/benchmarks/_ab_context.py --categories open-domain \
        --sample 400 --restart --out C:/nmafc_ab/context_facts.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
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

from scipy.stats import binomtest  # noqa: E402

from scripts.benchmarks._ab_budget import (  # noqa: E402
    PERMANENT,
    SCORED,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks._ab_vs_rag import timed_ours  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402


def describe(facts, hydrate, lines, whole, dedupe) -> str:
    """The `facts:turns:lines:dedupe:whole` shorthand the sweeps report in."""
    return (f"{'all' if facts is None else facts}:{hydrate}:"
            f"{'whole' if lines is None else lines}:{int(dedupe)}:{whole}")


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    gate = asyncio.Semaphore(args.concurrency)

    out = Path(args.out)
    rows: list[dict] = []
    if out.is_file() and not args.restart:
        rows = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(rows)} questions already answered\n")
    done = {(r["conv"], r["question"]) for r in rows}
    blocked: list[tuple[str, str]] = []

    wanted = set(args.categories.split(",")) if args.categories else set(SCORED)
    chosen: list[tuple[str, object]] = []
    for conv in load_locomo():
        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED
                     and qa.category_name in wanted]
        if args.limit:
            questions = questions[: args.limit]
        chosen += [(conv.sample_id, qa) for qa in questions]
    # Shuffled and sampled before `done` is applied, so a resumed run keeps
    # working through the same subset rather than drawing a fresh one. Same
    # seed as the prompt screen, so the two screens cover the same questions
    # and their discordant sets can be compared directly.
    random.Random(args.seed).shuffle(chosen)
    if args.sample:
        chosen = chosen[: args.sample]
    questions = [(c, qa) for c, qa in chosen if (c, qa.question) not in done]

    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no store at {store}")

    # Two handles rather than one reopened between arms: the settings that
    # differ are render-time, so one store can serve both, but a shared memory
    # object would also share its reinforcement buffer and arm A's reads would
    # show up in the weights arm B ranks against.
    def handle(facts, hydrate, lines, whole, dedupe):
        return open_memory(store, llm, embedder, args.hot, args.cold,
                           args.budget, args.max_hops, compact=args.compact,
                           hydrate=hydrate, facts=facts, lines=lines,
                           whole=whole, dedupe=dedupe)

    a_cfg = (args.a_facts, args.a_hydrate, args.a_hydrate_lines,
             args.a_hydrate_full_turns, args.a_dedupe_headers)
    b_cfg = (args.b_facts, args.b_hydrate, args.b_hydrate_lines,
             args.b_hydrate_full_turns, args.b_dedupe_headers)
    if a_cfg == b_cfg:
        raise SystemExit(
            f"A and B are both {describe(*a_cfg)}. Two identical arms measure "
            f"only the judge's noise floor."
        )

    mem_a, mem_b = handle(*a_cfg), handle(*b_cfg)
    print(f"store: {store}")
    print(f"A  {describe(*a_cfg)}")
    print(f"B  {describe(*b_cfg)}")
    print(f"retrieval: hot {args.hot}, cold {args.cold}, budget {args.budget}, "
          f"hops {args.max_hops}, compact {args.compact}")
    print(f"{len(questions)} questions, same answer prompt on both sides\n")

    async def one(conv_id: str, qa):
        async with gate:
            try:
                # Alternating is pointless here and would confound the latency
                # column, so A always runs first and latency is reported only
                # as a sanity check, not as a claim.
                a = await with_retries(lambda: timed_ours(mem_a, llm, qa.question))
                b = await with_retries(lambda: timed_ours(mem_b, llm, qa.question))
                a_v = await with_retries(
                    lambda: judge_answer(qa.question, a["pred"], str(qa.answer),
                                         judge))
                b_v = await with_retries(
                    lambda: judge_answer(qa.question, b["pred"], str(qa.answer),
                                         judge))
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ not in PERMANENT:
                    raise
                blocked.append((conv_id, qa.question))
                return None
        return {
            "conv": conv_id,
            "question": qa.question,
            "category": qa.category_name,
            "gold": str(qa.answer),
            "a": a["pred"], "b": b["pred"],
            "a_chars": a["chars"], "b_chars": b["chars"],
            "a_ms": a["ms"], "b_ms": b["ms"],
            "a_ok": bool(a_v.correct), "b_ok": bool(b_v.correct),
        }

    try:
        pending = [one(c, qa) for c, qa in questions]
        for start in range(0, len(pending), args.checkpoint_every):
            batch = await asyncio.gather(
                *pending[start:start + args.checkpoint_every])
            rows.extend(r for r in batch if r)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"  {len(rows)}/{len(questions) + len(done)} answered")
    finally:
        close_readonly(mem_a)
        close_readonly(mem_b)

    if not rows:
        print("nothing measured")
        return

    n = len(rows)
    a_ok = sum(r["a_ok"] for r in rows)
    b_ok = sum(r["b_ok"] for r in rows)
    fixed = sum(1 for r in rows if r["b_ok"] and not r["a_ok"])
    broke = sum(1 for r in rows if r["a_ok"] and not r["b_ok"])
    p = binomtest(fixed, fixed + broke, 0.5).pvalue if fixed + broke else 1.0
    a_t = sum(r["a_chars"] for r in rows) // 4 // n
    b_t = sum(r["b_chars"] for r in rows) // 4 // n

    print(f"\n{'=' * 76}")
    print(f"  n={n}   A {a_ok} ({a_ok/n:.2%})   B {b_ok} ({b_ok/n:.2%})   "
          f"{b_ok - a_ok:+d} ({100 * (b_ok - a_ok)/n:+.1f} points)")
    print(f"  fixed {fixed}  broke {broke}  p={p:.3g}   "
          f"blocked by content filter {len(blocked)}")
    print(f"  context  A {a_t}t   B {b_t}t   ceiling {args.ceiling}t")
    if b_t > args.ceiling:
        print(f"\n  B is {b_t - args.ceiling} tokens OVER RAG's tightest "
              f"per-category spend. Disqualified on cost whatever it scored: "
              f"the standing requirement is to be under RAG in every category, "
              f"not on average.")

    # Per-category, because the ceiling is per-category and a sampled run can
    # hide a category going over inside an average that does not.
    cats = sorted({r["category"] for r in rows})
    if len(cats) > 1:
        print()
        for cat in cats:
            sub = [r for r in rows if r["category"] == cat]
            m = len(sub)
            print(f"  {cat:12s} n={m:4d}  "
                  f"A {100 * sum(r['a_ok'] for r in sub) / m:5.1f}%  "
                  f"B {100 * sum(r['b_ok'] for r in sub) / m:5.1f}%  "
                  f"A {sum(r['a_chars'] for r in sub) // 4 // m:5d}t  "
                  f"B {sum(r['b_chars'] for r in sub) // 4 // m:5d}t")

    print(f"\n-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack",
                    help="Store name under <run>/stores/<arm>__<store>.")

    # A defaults to the shipped candidate, so `--b-*` alone is a complete
    # experiment and the control cannot be misconfigured by omission.
    ap.add_argument("--a-facts", type=int, default=6)
    ap.add_argument("--a-hydrate", type=int, default=28)
    ap.add_argument("--a-hydrate-lines", type=int, default=1)
    ap.add_argument("--a-hydrate-full-turns", type=int, default=4)
    ap.add_argument("--a-dedupe-headers", action=argparse.BooleanOptionalAction,
                    default=True)

    ap.add_argument("--b-facts", type=int, default=12)
    ap.add_argument("--b-hydrate", type=int, default=24)
    ap.add_argument("--b-hydrate-lines", type=int, default=1)
    ap.add_argument("--b-hydrate-full-turns", type=int, default=4)
    ap.add_argument("--b-dedupe-headers", action=argparse.BooleanOptionalAction,
                    default=True)

    ap.add_argument("--ceiling", type=int, default=1395,
                    help="RAG's tightest per-category context spend on this "
                         "store, which is multi-hop. Over it, an arm cannot "
                         "ship however well it scores.")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--hot", type=int, default=20)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--compact", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--categories", default="")
    ap.add_argument("--sample", type=int, default=0,
                    help="Total questions, drawn at random after the category "
                         "filter. 0 answers all of them.")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0,
                    help="Questions per conversation, before sampling.")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="C:/nmafc_ab/ab_context.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
