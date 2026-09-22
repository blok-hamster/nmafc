"""Does a wider prompt budget convert into judged accuracy?

`_sweep_context_budget.py` measures how often the gold answer reaches the
model. That is a ceiling, not a score: keyword overlap over-reports, and extra
facts can distract as easily as they can inform. Only generating and judging
settles it.

Both arms read the same stores through the same prompt, the same model and the
same judge. They differ in `rerank_top_k` and in the size of the candidate pool
feeding it, and in nothing else. Retrieval ranking is unchanged, so a wider
budget strictly adds lower-ranked facts to the prompt: any question that gets
worse got worse from distraction, and the run reports those separately because
that is the risk this change carries.

All 1,540 scored questions, not a targeted subset. The budget alters almost
every prompt, so there is no subset that isolates the effect, and the headline
comparison against RAG is over the whole set or it is not a comparison.

Read-only: retrieval buffers its reinforcement writes and never flushes.

Usage:
    python scripts/benchmarks/_ab_budget.py --run scripts/benchmarks/results/full_v3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
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
from nmafc.schemas.memory import DecayConfig  # noqa: E402
from nmafc.storage.config import NMafcConfig, StorageConfig  # noqa: E402
from nmafc.wrapper import NeuromorphicMemory  # noqa: E402

from scripts.benchmarks.arms.neuromorphic_tuned import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
)
from scripts.benchmarks.arms.base import strip_answer  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402

SCORED = ("single-hop", "temporal", "multi-hop", "open-domain")
CHARS_PER_TOKEN = 4

# Errors that will not come good on a retry. Matched by class name so this does
# not have to import the provider SDK to name its exception types.
PERMANENT = {"BadRequestError", "UnprocessableEntityError", "PermissionDeniedError"}


def open_memory(store: Path, llm, embedder, top_k, cold, budget, hops,
                compact=False, hydrate=0, facts=None, lines=None, whole=0,
                dedupe=False, overlap=None):
    return NeuromorphicMemory(
        llm_provider=llm,
        embedding_provider=embedder,
        config=NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(store / "hot_lancedb"),
                cold_uri=str(store / "cold.db"),
            ),
            decay=DecayConfig(
                max_hops=hops,
                top_k=top_k,
                fallback_keyword_limit=cold,
                rerank_top_k=budget,
                compact_validity=compact,
                hydrate_top_k=hydrate,
                # None renders every retrieved fact, which is what every
                # measurement before this argument existed did.
                context_facts_top_k=facts,
                # Likewise: `lines=None` hydrates whole turns, `whole=0` and
                # `dedupe=False` leave the source block exactly as it was.
                hydrate_lines=lines,
                hydrate_full_turns=whole,
                dedupe_source_headers=dedupe,
                # Screened at overlap 80 as "17 tokens saved, presence flat"
                # and parked, correctly, while context sat 457 under the
                # ceiling. The ceiling is now 1,000 and the render is 968, so
                # the headroom that made it not worth having is gone. It drops
                # a printed fact wholly restated by a better-ranked one, and
                # runs before the print limit, so it buys distinct facts rather
                # than fewer facts. Hydration reads the unseparated list, so
                # turns are byte-identical either way.
                fact_overlap_max=overlap,
                defer_reinforcement_writes=True,
            ),
        ),
    )


def close_readonly(memory) -> None:
    """Close a store without writing back what reading it produced.

    `defer_reinforcement_writes=True` is not enough on its own, and every
    harness here assumed it was. It only buffers the writebacks; `close()` then
    calls `flush_reinforcements()` and commits them, which resets `weight` to
    1.0 and advances `consolidation_index` and `last_reinforced_turn` on every
    record the queries touched. A screen that opens a store, asks a few hundred
    questions and closes it therefore leaves the store measurably different
    from how it found it, and a sweep that reopens the same store per
    configuration has its later configurations reading the damage its earlier
    ones did.

    Dropping the buffer first makes the read genuinely read-only. Use this
    anywhere the store is being measured rather than used.
    """
    memory._router._pending_reinforcements = {}
    memory.close()


async def answer(memory, llm, question: str) -> tuple[str, int]:
    retrieved = await memory._router.retrieve(question, memory.current_turn + 1)
    # The question is passed so that `hydrate_lines` can pick the lines it is
    # about. Omitting it silently degrades sparse hydration to a choice made
    # from the retrieved facts alone.
    context = memory._router.format_context(retrieved, question)
    system = ANSWER_SYSTEM_PROMPT + (f"\n\n{context}" if context else "")
    text = await llm.chat(
        messages=[{"role": "user", "content": question}], system_prompt=system
    )
    return strip_answer(text), len(context)


async def with_retries(factory, budget_seconds: float = 3600.0):
    """Keep asking until it works, the provider refuses, or the budget runs out.

    Five attempts with exponential backoff spans about thirty seconds, which is
    the right size for a blip and useless for the failure that actually ends
    these runs: the machine sleeps, every open connection dies at once, and all
    twenty-four workers exhaust their attempts within half a minute of each
    other while the network is still down.

    So the budget is wall-clock and generous, and the delay is capped so a long
    outage is retried at a steady minute rather than backing off into hours.
    Only a provider refusal short-circuits it, because that is the one failure
    repeating cannot fix.
    """
    started = time.monotonic()
    delay = 2.0
    attempt = 0
    while True:
        try:
            return await factory()
        except Exception as exc:  # noqa: BLE001
            # A rejected request is rejected the same way every time -- LoCoMo
            # covers gender identity and bereavement, and the content filter
            # refuses a handful of them outright. Retrying those wastes the
            # budget on a question the provider will never serve.
            if type(exc).__name__ in PERMANENT:
                raise
            attempt += 1
            if time.monotonic() - started > budget_seconds:
                raise
            print(f"      [retry {attempt} in {delay:.0f}s] {type(exc).__name__}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    gate = asyncio.Semaphore(args.concurrency)

    print(f"A: budget {args.budget_a}, pool hot {args.hot_a} archive {args.cold_a}")
    print(f"B: budget {args.budget_b}, pool hot {args.hot_b} archive {args.cold_b}\n")

    # Resume from whatever a previous attempt banked. Both arms are read-only
    # and the stores never change, so an answer written on an earlier attempt is
    # the same answer this one would produce; re-paying for it buys nothing. A
    # power cut mid-run cost an entire attempt once because the only checkpoint
    # was at conversation boundaries, and the first conversation is 152
    # questions long.
    out = Path(args.out)
    rows: list[dict] = []
    if out.is_file() and not args.restart:
        rows = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(rows)} questions already answered\n")
    done_keys = {(r["conv"], r["question"]) for r in rows}
    blocked: list[tuple[str, str]] = []

    def checkpoint() -> None:
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            print(f"  [skip] no store for {conv.sample_id}")
            continue
        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED]
        if args.limit:
            questions = questions[: args.limit]
        questions = [
            qa for qa in questions
            if (conv.sample_id, qa.question) not in done_keys
        ]
        if not questions:
            continue

        mem_a = open_memory(store, llm, embedder,
                            args.hot_a, args.cold_a, args.budget_a, args.max_hops,
                            args.compact_a, args.hydrate_a)
        mem_b = open_memory(store, llm, embedder,
                            args.hot_b, args.cold_b, args.budget_b, args.max_hops,
                            args.compact_b, args.hydrate_b)
        try:
            async def one(qa):
                try:
                    return await attempt(qa)
                except Exception as exc:  # noqa: BLE001
                    # The provider's content filter refuses a few LoCoMo
                    # questions outright, on the judge call as often as the
                    # answer. Dropping those questions from both arms keeps the
                    # comparison paired and costs the run nothing; killing the
                    # run over them costs everything. They are counted and
                    # listed so the denominator stays honest.
                    if type(exc).__name__ not in PERMANENT:
                        raise
                    blocked.append((conv.sample_id, qa.question))
                    return None

            async def attempt(qa):
                async with gate:
                    pred_a, chars_a = await with_retries(
                        lambda: answer(mem_a, llm, qa.question))
                    pred_b, chars_b = await with_retries(
                        lambda: answer(mem_b, llm, qa.question))
                    ja = await with_retries(
                        lambda: judge_answer(qa.question, pred_a, str(qa.answer), judge))
                    jb = await with_retries(
                        lambda: judge_answer(qa.question, pred_b, str(qa.answer), judge))
                    return {
                        "conv": conv.sample_id,
                        "question": qa.question,
                        "category": qa.category_name,
                        "gold": str(qa.answer),
                        "a": pred_a, "b": pred_b,
                        "a_chars": chars_a, "b_chars": chars_b,
                        "a_ok": ja.correct, "b_ok": jb.correct,
                    }

            fresh: list[dict] = []
            pending = [asyncio.ensure_future(one(qa)) for qa in questions]
            try:
                for finished in asyncio.as_completed(pending):
                    row = await finished
                    if row is None:
                        continue
                    fresh.append(row)
                    rows.append(row)
                    # Checkpoint on a question count rather than at the end of
                    # the conversation, so an interruption loses seconds of work
                    # instead of the whole conversation.
                    if len(fresh) % args.checkpoint_every == 0:
                        checkpoint()
                        print(f"    {conv.sample_id}: {len(fresh)}/{len(questions)}")
            except BaseException:
                for task in pending:
                    task.cancel()
                checkpoint()
                raise
            checkpoint()
            ok_a = sum(r["a_ok"] for r in fresh)
            ok_b = sum(r["b_ok"] for r in fresh)
            print(f"  {conv.sample_id}: {ok_a}/{len(fresh)} -> {ok_b}/{len(fresh)}")
        finally:
            mem_a.close()
            mem_b.close()

    if not rows:
        print("\nnothing answered")
        return

    n = len(rows)
    a = sum(r["a_ok"] for r in rows)
    b = sum(r["b_ok"] for r in rows)
    print(f"\n{'=' * 66}")
    print(f"questions answered twice : {n}")
    if blocked:
        print(f"  refused by content filter, dropped from both arms: {len(blocked)}")
        for sample, question in blocked[:6]:
            print(f"    [{sample}] {question[:70]}")
    print(f"  A  budget {args.budget_a:<3d}          : {a:5d}  {a / n:6.1%}")
    print(f"  B  budget {args.budget_b:<3d}          : {b:5d}  {b / n:6.1%}")
    print(f"  change                 : {b - a:+5d}  {(b - a) / n:+6.1%}")
    print(f"\n  gained by the wider budget : "
          f"{sum(1 for r in rows if r['b_ok'] and not r['a_ok'])}")
    print(f"  lost to distraction        : "
          f"{sum(1 for r in rows if r['a_ok'] and not r['b_ok'])}")
    print(f"\n  context tokens : A {sum(r['a_chars'] for r in rows) / n / CHARS_PER_TOKEN:.0f}"
          f"   B {sum(r['b_chars'] for r in rows) / n / CHARS_PER_TOKEN:.0f}"
          f"   (RAG baseline 1461)")

    print(f"\n  {'category':14s} {'n':>5s} {'A':>8s} {'B':>8s} {'change':>8s}")
    for cat in SCORED:
        s = [r for r in rows if r["category"] == cat]
        if not s:
            continue
        sa = sum(r["a_ok"] for r in s)
        sb = sum(r["b_ok"] for r in s)
        print(f"  {cat:14s} {len(s):5d} {sa / len(s):7.1%} {sb / len(s):7.1%} "
              f"{sb - sa:+8d}")

    gained = [r for r in rows if r["b_ok"] and not r["a_ok"]][:8]
    if gained:
        print("\n  answers the wider budget reached:")
        for r in gained:
            print(f"    Q   : {r['question'][:80]}")
            print(f"    gold: {r['gold'][:66]}")
            print(f"    was : {r['a'][:90]}")
            print(f"    now : {r['b'][:90]}\n")
    print(f"  full detail written to {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--budget-a", type=int, default=20)
    ap.add_argument("--hot-a", type=int, default=10)
    ap.add_argument("--cold-a", type=int, default=20)
    ap.add_argument("--budget-b", type=int, default=30)
    ap.add_argument("--hot-b", type=int, default=10)
    ap.add_argument("--cold-b", type=int, default=20)
    ap.add_argument("--compact-a", action="store_true")
    ap.add_argument("--compact-b", action="store_true")
    # Attaching the source turns behind the top-ranked facts. Needs the
    # turn_text table, which _backfill_turn_text.py fills without any LLM call.
    ap.add_argument("--hydrate-a", type=int, default=0)
    ap.add_argument("--hydrate-b", type=int, default=0)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="questions between writes to --out")
    ap.add_argument("--restart", action="store_true",
                    help="ignore any banked answers in --out and start over")
    ap.add_argument("--out", default="scripts/benchmarks/results/ab_budget.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
