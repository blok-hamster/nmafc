"""mylocomoeval: does the memory answer with what is true, or what used to be true?

LoCoMo asks every question after the last turn, about things usually said once,
so a memory that never forgets is never punished for it. That is why decay
tuning has never bought more than about 1.65 points there: the benchmark cannot
see forgetting in either direction. LongMemEval can, through its
`knowledge-update` type, but its haystacks would cost roughly 250 hours of
ingestion to build.

mylocomoeval is the middle path. `_mine_updates.py` reads the LoCoMo transcripts
we have already ingested and pulls out the moments where something stated
earlier stops being true, each carrying both the answer that is now correct and
the answer it replaced. Nothing is invented and nothing is re-ingested: the
questions are real changes from real conversations, and the stores are the ones
already on disk.

Both arms answer every question in the same loop, seconds apart, alternating who
goes first, for the reason `_ab_vs_rag.py` gives at length: a comparison across
two runs measures the provider's queue as much as the arms.

RAG is here as the control, and the prediction should be stated before the run
rather than after. RAG has no supersession at all. It hands over whatever chunks
match and lets the model choose. The earlier single-arm result showed that when
both the old and the new fact reach the prompt the model picks correctly 87.5%
of the time, so RAG scoring level with us is a live possibility -- and would
mean our forgetting machinery buys nothing over simply showing both versions.
That is a real finding, not a failed run, and it is the reason to run the
control at all: 75.2% alone says nothing without a yardstick.

Four numbers per arm, and the third is the one that matters:

  correct           answered with the current state
  answered stale    answered with the superseded state
  both in prompt    retrieval showed the model old AND new, leaving it to choose
  only stale        retrieval showed the old one alone, which is the failure
                    that cannot be reasoned around

Read-only, but only because it closes through `close_readonly`. Buffering the
reinforcement writes is not enough on its own; see that function. The stores are
left
exactly as they were.

Usage:
    python -u scripts/benchmarks/_mylocomoeval.py --limit 8
    python -u scripts/benchmarks/_mylocomoeval.py --out /c/nmafc_ab/mylocomoeval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks.arms.rag import ANSWER_SYSTEM_PROMPT, RagArm  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402
from scripts.benchmarks._ab_budget import (  # noqa: E402
    PERMANENT,
    answer,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks._test_updates import present  # noqa: E402


async def rag_answer(arm: RagArm, question: str) -> tuple[str, str]:
    """RAG's answer and the context it was given.

    Rebuilt here rather than calling `answer_question`, because the presence
    checks have to read exactly what the model was shown and that method
    returns only a token count. The retrieval and the prompt assembly are the
    arm's own, so what is measured is still RAG's behaviour.
    """
    query_vec = await arm._embedder.embed_single(question)
    results = arm._store.search(query_vec, top_k=arm._top_k)
    context = "\n\n---\n\n".join(r.record.fact_content for r in results)
    system = ANSWER_SYSTEM_PROMPT
    if context:
        system += f"\n\n=== RETRIEVED EXCERPTS ===\n{context}\n=== END EXCERPTS ==="
    text = await arm._llm.chat(
        messages=[{"role": "user", "content": question}], system_prompt=system
    )
    # `.strip()`, matching `answer_question` exactly. The other arms run their
    # replies through `strip_answer`; RAG does not, and imposing it here would
    # be this harness tidying the control's output rather than measuring it.
    return text.strip(), context


async def ours_answer(memory, llm, question: str) -> tuple[str, str]:
    text, _ = await answer(memory, llm, question)
    retrieved = await memory._router.retrieve(question, memory.current_turn + 1)
    return text, memory._router.format_context(retrieved, question)


def mcnemar(fixed: int, broke: int) -> float:
    n = fixed + broke
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(fixed, broke) + 1)) / 2 ** n)


async def run(args: argparse.Namespace) -> None:
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for row in questions:
        by_conv[row["conv"]].append(row)
    if args.limit:
        by_conv = {k: v[: args.limit] for k, v in by_conv.items()}
    total = sum(len(v) for v in by_conv.values())
    print(f"mylocomoeval: {total} questions across {len(by_conv)} conversations\n")

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    gate = asyncio.Semaphore(args.concurrency)

    conversations = {c.sample_id: c for c in load_locomo()}
    rows: list[dict] = []

    for conv, items in by_conv.items():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv}"
        if not store.is_dir() or conv not in conversations:
            print(f"  [skip] no store or transcript for {conv}")
            continue

        memory = open_memory(store, llm, embedder, args.hot, args.cold,
                             args.budget, args.max_hops, args.compact, args.hydrate)
        rag = RagArm(llm, embedder,
                     storage_dir=tempfile.mkdtemp(prefix=f"mylce_{conv}_"))
        print(f"[{conv}] building RAG index, {len(items)} questions")
        await with_retries(
            lambda: rag.ingest_conversation(conversations[conv].get_flat_history()))

        try:
            async def one(index: int, item: dict) -> dict | None:
                async with gate:
                    try:
                        if index % 2 == 0:
                            ours = await with_retries(
                                lambda: ours_answer(memory, llm, item["question"]))
                            theirs = await with_retries(
                                lambda: rag_answer(rag, item["question"]))
                        else:
                            theirs = await with_retries(
                                lambda: rag_answer(rag, item["question"]))
                            ours = await with_retries(
                                lambda: ours_answer(memory, llm, item["question"]))
                        va = await with_retries(lambda: judge_answer(
                            item["question"], ours[0], item["gold"], judge))
                        vb = await with_retries(lambda: judge_answer(
                            item["question"], theirs[0], item["gold"], judge))
                    except Exception as exc:  # noqa: BLE001
                        if type(exc).__name__ not in PERMANENT:
                            raise
                        return None

                out = {**item}
                for label, (text, context), verdict in (
                    ("ours", ours, va), ("rag", theirs, vb)
                ):
                    out[f"{label}_answer"] = text
                    out[f"{label}_correct"] = bool(verdict.correct)
                    # Scored by whole-word match rather than by the judge: the
                    # judge is asked whether the answer matches the gold, and a
                    # second judge call per arm to ask about the stale value
                    # would double the cost for a check a string settles.
                    out[f"{label}_answered_stale"] = present(item["stale"], text)
                    out[f"{label}_gold_in_prompt"] = present(item["gold"], context)
                    out[f"{label}_stale_in_prompt"] = present(item["stale"], context)
                return out

            done = await asyncio.gather(
                *(one(i, item) for i, item in enumerate(items)))
            rows.extend(r for r in done if r)
            Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"  {len(rows)} answered")
        finally:
            close_readonly(memory)
            rag.reset()

    if not rows:
        print("nothing measured")
        return
    report(rows)
    print(f"\n-> {args.out}")


def report(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n{'=' * 78}")
    print(f"mylocomoeval: {n} paired questions\n")

    for label, title in (("ours", "NMAFC"), ("rag", "RAG (control)")):
        correct = sum(r[f"{label}_correct"] for r in rows)
        stale = sum(r[f"{label}_answered_stale"] for r in rows)
        gold_in = sum(r[f"{label}_gold_in_prompt"] for r in rows)
        stale_in = sum(r[f"{label}_stale_in_prompt"] for r in rows)
        both = [r for r in rows
                if r[f"{label}_gold_in_prompt"] and r[f"{label}_stale_in_prompt"]]
        only_stale = [r for r in rows
                      if r[f"{label}_stale_in_prompt"]
                      and not r[f"{label}_gold_in_prompt"]]
        bad_both = sum(1 for r in both if not r[f"{label}_correct"])
        bad_only = sum(1 for r in only_stale if not r[f"{label}_correct"])
        print(f"  {title}")
        print(f"    correct              {correct:4d}  {100 * correct / n:5.1f}%")
        print(f"    answered stale       {stale:4d}  {100 * stale / n:5.1f}%")
        print(f"    gold in prompt       {gold_in:4d}  {100 * gold_in / n:5.1f}%")
        print(f"    stale in prompt      {stale_in:4d}  {100 * stale_in / n:5.1f}%")
        if both:
            print(f"    both in prompt       {len(both):4d}  {100 * len(both) / n:5.1f}%"
                  f"   wrong {bad_both}/{len(both)} = {100 * bad_both / len(both):.1f}%")
        if only_stale:
            print(f"    only stale in prompt {len(only_stale):4d}  "
                  f"{100 * len(only_stale) / n:5.1f}%"
                  f"   wrong {bad_only}/{len(only_stale)} = "
                  f"{100 * bad_only / len(only_stale):.1f}%")
        print()

    fixed = sum(1 for r in rows if r["ours_correct"] and not r["rag_correct"])
    broke = sum(1 for r in rows if r["rag_correct"] and not r["ours_correct"])
    p = mcnemar(fixed, broke)
    ours = sum(r["ours_correct"] for r in rows)
    rag = sum(r["rag_correct"] for r in rows)
    verdict = ("NMAFC" if fixed > broke else "RAG") if p < 0.05 else "tie"
    print(f"  HEAD TO HEAD  ours {100 * ours / n:.1f}%  RAG {100 * rag / n:.1f}%  "
          f"{100 * (ours - rag) / n:+.1f}   ours-only {fixed}, RAG-only {broke}, "
          f"p={p:.3g}  ->  {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    # Windows-shaped defaults on purpose. Git Bash rewrites a `/c/...` argument
    # into `C:\...` on the way into argv, so the other harnesses' POSIX-looking
    # defaults only ever worked because every run passed the path explicitly.
    ap.add_argument("--questions", default="C:/nmafc_ab/updates_final.json")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--compact", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="C:/nmafc_ab/mylocomoeval.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
