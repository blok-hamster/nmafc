"""Does reading the source turns behind the facts win open-domain back?

Open-domain is the only category costing us the benchmark: 839 questions, 116
lost to RAG against 63 won, and at parity the overall goes 65.1% to 68.6%
against RAG's 64.4%. Nothing else on the test is worth this much.

`_sweep_hydration.py` says what is wrong with those 116, and it is not what the
retrieval sweeps assumed:

    hydrate    gold present in the prompt
          0      37.1%
          5      40.5%   <- shipped
         12      45.7%
         20      51.7%

At shipped settings the answer was already in front of the model for 47 of the
116 and we answered wrong anyway. The predictions show why. Only 10 say we do
not know; the rest are confident and wrong in the same way, naming a real fact
about the right subject that answers a different question. Asked which novel
Evan finds gripping we answer with a different novel he read; asked which city
Tim suggested for the team trip we answer with the earlier brainstorm list. The
question carries a qualifier -- gripping, next month, two weeks before 11 August
-- and extraction stripped it out. RAG strips nothing, so its chunk still holds
the word that decides the answer.

Hydration is the mechanism that puts it back: `_source_turns` fetches the
verbatim turns behind the best-ranked facts. Depth 20 recovers presence for 13
more of the 116 outright, and should also help the 47, because the source turn
carries the qualifier the fact lost. Presence cannot measure that second effect
at all -- the text was already there -- which is why this has to be a generation
A/B rather than another probe.

That A/B has now run once, at twenty facts and twenty source turns, and it went
78.2% against 74.3% shipped, p = 6.74e-05. It also cost 1,847 tokens against
RAG's 1,459 and still sat under RAG's 80.6%, so it was worse on both counts that
matter and could not ship.

`context_facts_top_k` is what makes the second attempt worth running. Sweeping
the two halves of the context apart shows the fact list was almost entirely
duplication: six facts over twenty source turns holds the gold for 60 of the 116
losses, exactly as often as twenty facts over twenty, at 1,434 tokens instead of
1,865. That alone turns the more expensive arm into the cheaper one.

It may also be the more accurate one, and for a reason presence cannot see. The
47 questions whose answer was already in the prompt were answered with a real
fact about the right subject that answers a different question -- which is what
losing to a competing summary looks like. Deleting fourteen of those summaries
while keeping every source turn removes the thing that was being picked instead.
So the expectation here is not "as good for less"; it is better for less. If it
is merely as good, we are cheaper than RAG and still behind it, and the gap has
to be closed by ranking instead.

Cost is why this runs one arm and not three. RAG's answers and our shipped
answers for all 839 already exist in `haystack.json`, so only the deep arm is
generated. That saves roughly half the tokens and buys one problem: those
answers come from a different session, and cross-run drift on this benchmark is
1.5-2 points, which is the size of the effect being looked for. So a sample of
the shipped configuration is re-run first, in this session, and compared against
its own saved answers. If the drift is larger than the effect, the saved answers
are not a usable baseline and the run stops before spending anything on the deep
arm. That check is the whole justification for the cheap design and it runs
first for that reason.

Usage:
    python -u scripts/benchmarks/_ab_hydration.py --limit 20 --drift 10
    python -u scripts/benchmarks/_ab_hydration.py --deep-facts 6 \
        --out C:/nmafc_ab/hydration_ab_6x20.json
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

from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402
from scripts.benchmarks._ab_budget import (  # noqa: E402
    PERMANENT,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks._ab_vs_rag import timed_ours  # noqa: E402


def mcnemar(rows: list[dict], a: str, b: str) -> tuple[int, int, float]:
    """Discordant pairs and the exact binomial p. Ties carry no information."""
    won = sum(1 for r in rows if r[a] and not r[b])
    lost = sum(1 for r in rows if r[b] and not r[a])
    if won + lost == 0:
        return 0, 0, 1.0
    return won, lost, binomtest(won, won + lost, 0.5).pvalue


async def answer_all(items, memory, llm, judge, gate, label, checkpoint,
                     out: Path | None, rows: list[dict], done: set):
    """Generate and judge, checkpointing so a stopped run resumes for free."""
    pending = [i for i in items if i["question"] not in done]
    blocked: list[str] = []
    print(f"{label}: {len(pending)} to answer "
          f"({len(items) - len(pending)} already done)")

    async def one(item: dict):
        async with gate:
            try:
                got = await with_retries(
                    lambda: timed_ours(memory, llm, item["question"]))
                verdict = await with_retries(
                    lambda: judge_answer(item["question"], got["pred"],
                                         item["gold"], judge))
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ not in PERMANENT:
                    raise
                blocked.append(item["question"])
                return None
        return {
            "conv": item["conv"], "question": item["question"],
            "gold": item["gold"], "category": item["category"],
            "deep_pred": got["pred"], "deep_ok": verdict.correct,
            "deep_ms": got["ms"], "deep_chars": got["chars"],
            # Carried through from the saved run so the comparison needs no
            # second lookup and the output file stands alone.
            "shipped_ok": item["ours_ok"], "rag_ok": item["rag_ok"],
            "shipped_chars": item["ours_chars"], "rag_chars": item["rag_chars"],
        }

    tasks = [one(i) for i in pending]
    for start in range(0, len(tasks), checkpoint):
        batch = await asyncio.gather(*tasks[start:start + checkpoint])
        rows.extend(r for r in batch if r)
        if out:
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"  {len(rows)}/{len(items)} answered")
    if blocked:
        print(f"  {len(blocked)} blocked by the content filter, dropped")
    return rows


async def run(args: argparse.Namespace) -> None:
    saved = json.loads(Path(args.results).read_text(encoding="utf-8"))
    items = [r for r in saved if r["category"] == args.category]
    if args.limit:
        items = items[: args.limit]

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"]))
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    gate = asyncio.Semaphore(args.concurrency)

    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no store at {store}")

    deep_facts = "all" if args.deep_facts is None else args.deep_facts
    print(f"{len(items)} {args.category} questions, store {args.store}")
    print(f"shipped all facts + {args.shipped_hydrate} source turns  ->  "
          f"deep {deep_facts} facts + {args.deep_hydrate} source turns, "
          f"retrieving {args.budget}\n")

    # ---- drift check, first, because it can cancel the expensive part -------
    sample = random.Random(args.seed).sample(items, min(args.drift, len(items)))
    memory = open_memory(store, llm, embedder, args.hot, args.cold, args.budget,
                         args.max_hops, compact=True,
                         hydrate=args.shipped_hydrate)
    checked: list[dict] = []
    try:
        await answer_all(sample, memory, llm, judge, gate, "drift check",
                         args.checkpoint_every, None, checked, set())
    finally:
        close_readonly(memory)

    if not checked:
        print("drift check produced nothing")
        return
    now = 100 * sum(r["deep_ok"] for r in checked) / len(checked)
    then = 100 * sum(r["shipped_ok"] for r in checked) / len(checked)
    agree = sum(1 for r in checked if r["deep_ok"] == r["shipped_ok"])
    print(f"\n  drift: shipped config scored {then:.1f}% when saved, "
          f"{now:.1f}% today on the same {len(checked)} questions "
          f"({now - then:+.1f}), same verdict on {agree}")
    if abs(now - then) > args.max_drift and not args.force:
        print(f"\n  STOP. Drift of {abs(now - then):.1f} points exceeds the "
              f"{args.max_drift} allowed, and the effect being looked for is "
              f"about that size.\n  The saved answers are not a usable "
              f"baseline today. Nothing has been spent on the deep arm.\n"
              f"  Re-run with --force to override, or run both arms in one "
              f"session at full cost.")
        return
    print("  within tolerance, the saved answers are a usable baseline\n")

    # ---- the deep arm -------------------------------------------------------
    out = Path(args.out)
    rows: list[dict] = []
    if out.is_file() and not args.restart:
        rows = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(rows)} already answered")
    done = {r["question"] for r in rows}

    memory = open_memory(store, llm, embedder, args.hot, args.cold, args.budget,
                         args.max_hops, compact=True, hydrate=args.deep_hydrate,
                         facts=args.deep_facts)
    try:
        await answer_all(items, memory, llm, judge, gate,
                         f"{deep_facts} facts + {args.deep_hydrate} turns",
                         args.checkpoint_every,
                         out, rows, done)
    finally:
        close_readonly(memory)

    if not rows:
        print("nothing measured")
        return

    n = len(rows)
    deep = 100 * sum(r["deep_ok"] for r in rows) / n
    shipped = 100 * sum(r["shipped_ok"] for r in rows) / n
    rag = 100 * sum(r["rag_ok"] for r in rows) / n
    print(f"\n{'=' * 88}")
    print(f"  {args.category}, {n} questions\n")
    rag_tokens = sum(r["rag_chars"] for r in rows) // 4 // n
    deep_tokens = sum(r["deep_chars"] for r in rows) // 4 // n
    print(f"  shipped  all:{args.shipped_hydrate:<3d} {shipped:6.1f}%   "
          f"{sum(r['shipped_chars'] for r in rows) // 4 // n:5d} tokens")
    print(f"  deep     {str(deep_facts) + ':' + str(args.deep_hydrate):<7s} "
          f"{deep:6.1f}%   {deep_tokens:5d} tokens"
          f"{'' if deep_tokens <= rag_tokens else '   OVER RAG'}")
    print(f"  RAG              {rag:6.1f}%   {rag_tokens:5d} tokens")

    for label, a, b in (("deep vs shipped", "deep_ok", "shipped_ok"),
                        ("deep vs RAG", "deep_ok", "rag_ok"),
                        ("shipped vs RAG", "shipped_ok", "rag_ok")):
        won, lost, p = mcnemar(rows, a, b)
        verdict = "no difference" if p >= 0.05 else (
            "better" if won > lost else "worse")
        print(f"\n  {label:16s} won {won:4d}  lost {lost:4d}  "
              f"net {won - lost:+4d}  p={p:.3g}  -> {verdict}")

    print("\n  The shipped and RAG columns come from a different session; the")
    print("  drift check above is what licenses comparing them. Deep against")
    print("  shipped is the measurement, deep against RAG is the one that")
    print("  decides whether the category stops costing us the benchmark.")

    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="C:/nmafc_ab/haystack.json")
    ap.add_argument("--category", default="open-domain")
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--shipped-hydrate", type=int, default=5)
    ap.add_argument("--deep-hydrate", type=int, default=20)
    ap.add_argument("--deep-facts", type=int, default=None,
                    help="Facts rendered in the deep arm. Unset renders all "
                         "of them, which costs about 619 tokens at a budget of "
                         "20 and restates what the source turns already say. "
                         "Trimming it is how deep hydration fits RAG's budget.")
    ap.add_argument("--drift", type=int, default=100,
                    help="Questions re-run at the shipped setting first.")
    ap.add_argument("--max-drift", type=float, default=2.0,
                    help="Points of drift above which the saved baseline is "
                         "refused and nothing is spent on the deep arm.")
    ap.add_argument("--force", action="store_true",
                    help="Run the deep arm even if the drift check fails.")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="C:/nmafc_ab/hydration_ab.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
