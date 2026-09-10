"""Does the memory answer with what is true, or with what used to be true?

The questions come from `_mine_updates.py`: each names something that changed
during the conversation, and carries both the answer that is true at the end and
the answer that has been superseded. Asking only "was it right" cannot separate
a system that forgot the answer from one that remembered the old one too well,
and only the second says anything about decay. So every question is scored twice
and every prompt is inspected.

Four numbers come out, and the third is the one that matters:

  correct       answered with the current state
  stale         answered with the superseded state
  both in prompt   the retrieval put the old fact AND the new fact in front of
                the model, leaving it to choose
  stale | both  how often it chose wrongly when it had both

That last one is the cost of never invalidating anything, stated as a rate. It
is measurable today, on the stores as they are, because nothing here needs
weights to vary -- which is what makes it worth running before committing to the
re-ingestion that a decay fix would need.

Read-only, via `close_readonly`. Buffering the reinforcement writes does not
achieve that by itself; the ordinary `close()` commits the buffer.

Usage:
    python scripts/benchmarks/_test_updates.py --questions /c/nmafc_ab/updates_clean.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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
    PERMANENT,
    answer,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402

STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "his", "her", "their",
        "is", "was", "and", "for", "with", "he", "she", "they", "it"}


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", str(text).lower()) if w not in STOP}


def present(needle: str, haystack: str) -> bool:
    """Whether every content word of `needle` appears in `haystack`.

    Whole-word, so "two" does not match "twofold" and a year is not found inside
    a longer number. Deliberately strict: the question this answers is whether
    the model was *shown* the fact, and a partial match does not show it.
    """
    want = tokens(needle)
    if not want:
        return False
    low = haystack.lower()
    return all(
        re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", low) for w in want
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default="update_results.json")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for row in questions:
        by_conv[row["conv"]].append(row)
    print(f"{len(questions)} questions across {len(by_conv)} conversations\n")

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    gate = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []

    for conv, items in by_conv.items():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv}"
        if not store.is_dir():
            print(f"  [skip] no store for {conv}")
            continue
        memory = open_memory(store, llm, embedder, args.hot, args.cold,
                             args.budget, args.max_hops, False, args.hydrate)
        try:
            async def one(item: dict) -> dict | None:
                async with gate:
                    try:
                        text, size = await with_retries(
                            lambda: answer(memory, llm, item["question"])
                        )
                        # The prompt itself, rebuilt rather than returned, so
                        # the two presence checks read exactly what the model
                        # was shown rather than an approximation of it.
                        retrieved = await memory._router.retrieve(
                            item["question"], memory.current_turn + 1
                        )
                        context = memory._router.format_context(retrieved)
                        verdict = await with_retries(
                            lambda: judge_answer(
                                item["question"], text, item["gold"], judge
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        if type(exc).__name__ not in PERMANENT:
                            raise
                        return None
                return {
                    **item,
                    "answer": text,
                    "chars": size,
                    "correct": bool(verdict.correct),
                    # Scored by string rather than by the judge: the judge is
                    # asked whether the answer matches the gold, and asking it a
                    # second question about the stale value doubles the cost of
                    # the run for a check that a whole-word match settles.
                    "answered_stale": present(item["stale"], text),
                    "gold_in_prompt": present(item["gold"], context),
                    "stale_in_prompt": present(item["stale"], context),
                }

            done = await asyncio.gather(*(one(i) for i in items))
            got = [d for d in done if d]
            rows.extend(got)
            ok = sum(d["correct"] for d in got)
            print(f"  {conv:10s} {ok:3d}/{len(got):3d}")
        finally:
            close_readonly(memory)

    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    n = len(rows) or 1
    correct = sum(r["correct"] for r in rows)
    stale = sum(r["answered_stale"] and not r["correct"] for r in rows)
    both = [r for r in rows if r["gold_in_prompt"] and r["stale_in_prompt"]]
    only_stale = [r for r in rows if r["stale_in_prompt"] and not r["gold_in_prompt"]]
    print(f"\n  questions              {len(rows)}")
    print(f"  correct                {correct:4d}  {correct/n:6.1%}")
    print(f"  answered stale value   {stale:4d}  {stale/n:6.1%}")
    print(f"  gold reached prompt    {sum(r['gold_in_prompt'] for r in rows):4d}"
          f"  {sum(r['gold_in_prompt'] for r in rows)/n:6.1%}")
    print(f"  stale reached prompt   {sum(r['stale_in_prompt'] for r in rows):4d}"
          f"  {sum(r['stale_in_prompt'] for r in rows)/n:6.1%}")
    if both:
        wrong = sum(1 for r in both if not r["correct"])
        print(f"  both in prompt         {len(both):4d}  {len(both)/n:6.1%}"
              f"   of which wrong {wrong}/{len(both)} = {wrong/len(both):.1%}")
    if only_stale:
        wrong = sum(1 for r in only_stale if not r["correct"])
        print(f"  only stale in prompt   {len(only_stale):4d}  {len(only_stale)/n:6.1%}"
              f"   of which wrong {wrong}/{len(only_stale)} = {wrong/len(only_stale):.1%}")
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
