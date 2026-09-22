"""Does naming the kind of answer wanted convert retrieved evidence into score?

The diagnosis is in `_probe_answer_type.py` and the mechanism in
`nmafc.integration.answer_type`. In short: over the 1,535 paired questions of
`haystack.json`, the 238 that name the kind of answer they want scored 56.7% for
us against RAG's 61.8%, while the other 1,297 scored 66.7% against 64.9%. The
whole of the deficit sits in questions that name a type, and the answers show
why -- asked for a state we returned the town, asked for a country we returned
the city, asked for a console we returned the maker. RAG made the same three
mistakes, which is what a shared prompt rule looks like from the outside.

This arm changes the prompt and nothing else. Same store, same retrieval
settings, same hydration budget, same judge.

Two design choices that decide whether the answer is worth anything:

* **The rule is gated on the tag, so the untyped controls are free.** The first
  run of this put `TYPE_RULE` in the system prompt on every question, which made
  the untyped controls compulsory: a rule sitting in the prompt can move a
  question that names no type just by being there, and it did, by -1.8 points
  over 218 of them. `answer_type.gate` removes that instead of measuring it --
  on an untyped question B's prompt is now byte-identical to A's, the same call.
  The controls are still reported, because a slice nobody prints is a slice
  nobody checks, but they are copied from arm A rather than bought, since a
  second sample of an identical prompt measures only the model's own noise.

* **RAG is not re-run.** Its prompt is untouched, so a second RAG run would
  measure nothing but drift and cost as much as the arm under test. The
  comparison that matters here is B against A, both generated now, in one
  session, against one store. Once B is known to beat A, `haystack.json` holds
  RAG's paired answers for the same questions.

Read-only: retrieval buffers its reinforcement writes and never flushes.

Usage:
    python -u scripts/benchmarks/_ab_answer_type.py --limit 2 --out C:/nmafc_ab/at_smoke.json
    python -u scripts/benchmarks/_ab_answer_type.py --out C:/nmafc_ab/answer_type.json
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

from scipy.stats import binomtest  # noqa: E402

from nmafc.integration.answer_type import TYPE_RULE, gate  # noqa: E402
from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks._ab_budget import (  # noqa: E402
    PERMANENT,
    SCORED,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks.arms.base import strip_answer  # noqa: E402
from scripts.benchmarks.arms.neuromorphic_tuned import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
)
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402

# Inserted ahead of SHORT_ANSWER_RULES, for the same reason `_ab_prompt.py`
# gives: the rule qualifies "COPY THE WORDING FROM THE FACTS", and a
# qualification read after the thing it qualifies arrives too late.
_ANCHOR = "\nCRITICAL: You are completing a fill-in-the-blank quiz."
PROMPT_B = ANSWER_SYSTEM_PROMPT.replace(_ANCHOR, TYPE_RULE + _ANCHOR)


async def answer_with(memory, llm, question: str, system_prompt: str,
                      tag: str = "") -> tuple[str, int]:
    """One answer, and every token charged to it beyond the question itself.

    The tag is counted into the returned width even though it is derived from
    the question rather than fetched from the store. It is not context by any
    reasonable definition, and counting it anyway is the only way the sub-1,000
    claim stays checkable by someone who did not write this.
    """
    retrieved = await memory._router.retrieve(question, memory.current_turn + 1)
    context = memory._router.format_context(retrieved, question)
    blocks = [b for b in (tag, context) if b]
    system = system_prompt + ("\n\n" + "\n\n".join(blocks) if blocks else "")
    text = await llm.chat(
        messages=[{"role": "user", "content": question}], system_prompt=system
    )
    return strip_answer(text), len(context) + len(tag)


def choose(args: argparse.Namespace) -> list[tuple[str, object, str]]:
    """Every typed question, plus untyped controls drawn at random.

    Stratified rather than sampled flat: a flat sample of 400 would hold about
    62 typed questions, which is too few to see a gain worth having, and about
    338 controls, which is many more than are needed to see a regression.
    """
    wanted = set(args.categories.split(",")) if args.categories else set(SCORED)
    typed: list[tuple[str, object, str]] = []
    untyped: list[tuple[str, object, str]] = []
    for conv in load_locomo():
        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED
                     and qa.category_name in wanted]
        if args.limit:
            questions = questions[: args.limit]
        for qa in questions:
            _, tag = gate(qa.question)
            (typed if tag else untyped).append((conv.sample_id, qa, tag))

    rng = random.Random(args.seed)
    rng.shuffle(typed)
    rng.shuffle(untyped)
    if args.typed:
        typed = typed[: args.typed]
    return typed + untyped[: args.controls]


async def run(args: argparse.Namespace) -> None:
    if PROMPT_B == ANSWER_SYSTEM_PROMPT:
        raise SystemExit(
            "TYPE_RULE was not inserted: the anchor text has moved in "
            "SHORT_ANSWER_RULES. Fix the anchor rather than running two "
            "identical arms."
        )

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    limit = asyncio.Semaphore(args.concurrency)

    out = Path(args.out)
    rows: list[dict] = []
    if out.is_file() and not args.restart:
        rows = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(rows)} already answered\n")
    done = {(r["conv"], r["question"]) for r in rows}
    blocked: list[tuple[str, str]] = []

    chosen = choose(args)
    pending = [(c, qa, tag) for c, qa, tag in chosen
               if (c, qa.question) not in done]

    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no merged store at {store}; run _build_haystack.py first")

    # Two handles on one store rather than one shared. The arms differ only in
    # the prompt, but a shared memory object shares its reinforcement buffer,
    # and arm A's reads would then be visible in the weights arm B ranks against.
    def handle():
        return open_memory(store, llm, embedder, args.hot, args.cold,
                           args.budget, args.max_hops, compact=args.compact,
                           hydrate=args.hydrate, facts=args.facts,
                           lines=args.hydrate_lines,
                           whole=args.hydrate_full_turns,
                           dedupe=args.dedupe_headers)

    mem_a, mem_b = handle(), handle()
    n_typed = sum(1 for _, _, t in pending if t)
    print(f"store: {store}")
    print(f"A is the shipped prompt, B adds TYPE_RULE ({len(TYPE_RULE) // 4}t) "
          f"plus a per-question tag, on typed questions only")
    print(f"{len(pending)} questions: {n_typed} typed, "
          f"{len(pending) - n_typed} untyped controls copied from A\n")

    async def one(conv_id: str, qa, tag: str):
        async with limit:
            try:
                a_text, a_chars = await with_retries(
                    lambda: answer_with(mem_a, llm, qa.question,
                                        ANSWER_SYSTEM_PROMPT))
                a_v = await with_retries(
                    lambda: judge_answer(qa.question, a_text, str(qa.answer), judge))
                if not tag:
                    # Gated: no tag means no rule, which means B's call is A's
                    # call. Buying a second sample of it would measure the
                    # provider's sampling and charge for the privilege.
                    b_text, b_chars, b_v = a_text, a_chars, a_v
                else:
                    b_text, b_chars = await with_retries(
                        lambda: answer_with(mem_b, llm, qa.question, PROMPT_B,
                                            tag))
                    b_v = await with_retries(
                        lambda: judge_answer(qa.question, b_text, str(qa.answer),
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
            "tag": tag,
            "a": a_text, "b": b_text,
            "a_chars": a_chars, "b_chars": b_chars,
            "a_ok": bool(a_v.correct), "b_ok": bool(b_v.correct),
        }

    try:
        tasks = [one(c, qa, t) for c, qa, t in pending]
        for start in range(0, len(tasks), args.checkpoint_every):
            batch = await asyncio.gather(
                *tasks[start:start + args.checkpoint_every])
            rows.extend(r for r in batch if r)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"  {len(rows)}/{len(pending) + len(done)} answered")
    finally:
        close_readonly(mem_a)
        close_readonly(mem_b)

    if not rows:
        print("nothing measured")
        return
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report(rows, blocked)
    print(f"\n-> {args.out}")


def report(rows: list[dict], blocked: list[tuple[str, str]]) -> None:
    def block(rs: list[dict], label: str) -> None:
        if not rs:
            return
        n = len(rs)
        a = sum(r["a_ok"] for r in rs)
        b = sum(r["b_ok"] for r in rs)
        fixed = sum(1 for r in rs if r["b_ok"] and not r["a_ok"])
        broke = sum(1 for r in rs if r["a_ok"] and not r["b_ok"])
        p = binomtest(fixed, fixed + broke).pvalue if fixed + broke else 1.0
        ac = sum(r["a_chars"] for r in rs) / n / 4
        bc = sum(r["b_chars"] for r in rs) / n / 4
        print(f"{label:<14}{n:>5}{a / n * 100:>8.1f}%{b / n * 100:>8.1f}%"
              f"{(b - a) / n * 100:>+8.1f}{fixed:>6}{broke:>6}{p:>9.3g}"
              f"{ac:>8.0f}{bc:>8.0f}")

    print(f"\n{'':<14}{'n':>5}{'A':>9}{'B':>9}{'diff':>8}{'fix':>6}{'brk':>6}"
          f"{'p':>9}{'A ctx':>8}{'B ctx':>8}")
    print("-" * 82)
    block([r for r in rows if r["tag"]], "typed")
    block([r for r in rows if not r["tag"]], "untyped ctrl")
    print("-" * 82)
    block(rows, "ALL")
    print()
    for cat in sorted({r["category"] for r in rows}):
        block([r for r in rows if r["category"] == cat and r["tag"]],
              f"typed {cat[:8]}")

    flipped = [r for r in rows if r["b_ok"] != r["a_ok"]]
    print(f"\n{len(flipped)} answers changed verdict. Regressions first:")
    for r in sorted(flipped, key=lambda r: r["b_ok"])[:20]:
        mark = "FIX " if r["b_ok"] else "BRK "
        print(f"\n  {mark}{r['category']}  tag={r['tag'] or '(none)'}")
        print(f"    Q    {r['question']}")
        print(f"    gold {r['gold']}")
        print(f"    A    {r['a']}")
        print(f"    B    {r['b']}")
    if blocked:
        print(f"\n{len(blocked)} questions blocked by the provider")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--typed", type=int, default=0,
                    help="Cap on typed questions. 0 runs all of them.")
    ap.add_argument("--controls", type=int, default=200,
                    help="Untyped questions, to catch a regression caused by "
                         "the rule sitting in the prompt.")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--facts", type=int, default=None)
    ap.add_argument("--hydrate-lines", type=int, default=None)
    ap.add_argument("--hydrate-full-turns", type=int, default=0)
    ap.add_argument("--dedupe-headers", action="store_true")
    ap.add_argument("--categories", default="")
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--compact", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0,
                    help="Questions per conversation, before stratifying.")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="C:/nmafc_ab/answer_type.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
