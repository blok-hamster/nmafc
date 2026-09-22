"""Is the validity suffix worth what it costs?

Every fact is rendered with a trailing span, and measured over 200 prompts on
the finished stores that span is 227 tokens of a 1,124-token context -- 20.2%.
All 4,000 of the suffixes in that sample ended "- present", because nothing in
these stores has been invalidated, so one fifth of the prompt is spent on a
word that is true of every line. The clock time is the same kind of cost: a
session timestamp reads "7:18 pm on 27 May, 2023" and the hour is 11.2 of its
25.8 characters, on a benchmark where no question asks the time of day.

`compact_validity` drops both. It renders "(27 May, 2023)" where the default
renders "(Valid: 7:18 pm on 27 May, 2023 - present)", and takes the context to
982 tokens: 142 fewer, 12.6%.

No information leaves the prompt. The date survives in full, and an end date is
still rendered when a fact actually has one. That is the whole claim being
tested, and it needs testing because it is a claim about a model's reading and
not about arithmetic: shorter is not automatically better if "Valid:" was doing
work as a label, or if the hour was disambiguating same-day facts.

Both arms read the same stores, retrieve identically with the same settings and
the same hydration budget, and are judged by the same judge. The only thing
that differs is how the span is written. Paired over the same questions.

Read-only: retrieval buffers its reinforcement writes and never flushes.

Usage:
    python scripts/benchmarks/_ab_compact.py --out /c/nmafc_ab/ab_compact.json
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

from scripts.benchmarks._ab_budget import (  # noqa: E402
    CHARS_PER_TOKEN,
    PERMANENT,
    SCORED,
    answer,
    open_memory,
    with_retries,
)
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402


def mcnemar(fixed: int, broke: int) -> float:
    """Exact two-sided binomial test on the discordant pairs."""
    from math import comb

    n = fixed + broke
    if n == 0:
        return 1.0
    k = min(fixed, broke)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


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

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            print(f"  [skip] no store for {conv.sample_id}")
            continue
        questions = [
            qa for qa in conv.qa_pairs
            if qa.category_name in SCORED
            and (conv.sample_id, qa.question) not in done
        ]
        if args.limit:
            questions = questions[: args.limit]
        if not questions:
            continue

        # One handle per arm. They retrieve identically, but a shared handle
        # would also share the reinforcement buffer, and arm A's reads would
        # then be visible in the weights arm B ranks against.
        mem_a = open_memory(store, llm, embedder, args.hot, args.cold,
                            args.budget, args.max_hops, False, args.hydrate)
        mem_b = open_memory(store, llm, embedder, args.hot, args.cold,
                            args.budget, args.max_hops, True, args.hydrate)

        async def one(qa, conv=conv):
            async with gate:
                try:
                    a_text, a_chars = await with_retries(
                        lambda: answer(mem_a, llm, qa.question))
                    b_text, b_chars = await with_retries(
                        lambda: answer(mem_b, llm, qa.question))
                    a_v = await with_retries(
                        lambda: judge_answer(qa.question, a_text, qa.answer, judge))
                    b_v = await with_retries(
                        lambda: judge_answer(qa.question, b_text, qa.answer, judge))
                except Exception as exc:  # noqa: BLE001
                    if type(exc).__name__ not in PERMANENT:
                        raise
                    blocked.append((conv.sample_id, qa.question))
                    return None
            return {
                "conv": conv.sample_id,
                "question": qa.question,
                "category": qa.category_name,
                "gold": qa.answer,
                "a": a_text, "b": b_text,
                "a_chars": a_chars, "b_chars": b_chars,
                "a_ok": bool(a_v.correct), "b_ok": bool(b_v.correct),
            }

        pending = [one(qa) for qa in questions]
        for start in range(0, len(pending), args.checkpoint_every):
            batch = await asyncio.gather(*pending[start:start + args.checkpoint_every])
            rows.extend(r for r in batch if r)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"    {conv.sample_id}: "
                  f"{min(start + args.checkpoint_every, len(pending))}/{len(pending)}")

        got = [r for r in rows if r["conv"] == conv.sample_id]
        print(f"  {conv.sample_id}: {sum(r['a_ok'] for r in got)}"
              f" -> {sum(r['b_ok'] for r in got)} of {len(got)}")

    if not rows:
        print("nothing measured")
        return

    n = len(rows)
    a = sum(r["a_ok"] for r in rows)
    b = sum(r["b_ok"] for r in rows)
    fixed = sum(1 for r in rows if r["b_ok"] and not r["a_ok"])
    broke = sum(1 for r in rows if r["a_ok"] and not r["b_ok"])
    a_tok = sum(r["a_chars"] for r in rows) / n / CHARS_PER_TOKEN
    b_tok = sum(r["b_chars"] for r in rows) / n / CHARS_PER_TOKEN

    print(f"\n{'=' * 66}")
    print(f"  n = {n}")
    print(f"  A  verbose span   {a:5d}  {a / n:7.2%}   {a_tok:6.0f} tokens")
    print(f"  B  compact span   {b:5d}  {b / n:7.2%}   {b_tok:6.0f} tokens")
    print(f"     delta          {b - a:+5d}  {(b - a) / n:+7.2%}   "
          f"{b_tok - a_tok:+6.0f} tokens ({(b_tok - a_tok) / a_tok:+.1%})")
    print(f"\n  fixed {fixed}   broke {broke}   "
          f"McNemar two-sided p = {mcnemar(fixed, broke):.4g}")
    if blocked:
        print(f"  blocked by content filter: {len(blocked)}")

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    print(f"\n  {'category':<14}{'n':>6}{'A':>9}{'B':>9}{'delta':>9}")
    for cat in SCORED:
        got = by_cat.get(cat)
        if not got:
            continue
        ca = sum(r["a_ok"] for r in got) / len(got)
        cb = sum(r["b_ok"] for r in got) / len(got)
        print(f"  {cat:<14}{len(got):>6}{ca:>8.1%}{cb:>9.1%}{cb - ca:>+9.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="ab_compact.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
