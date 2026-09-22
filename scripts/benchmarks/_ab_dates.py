"""Does dating the facts change the answers, or only the context?

The store now knows when each turn happened and the router renders that as a
date instead of a turn number. Whether that is worth anything can only be
settled by generating answers and judging them, because the failure it targets
is a reading failure: the facts were retrieved, presented as "(Valid: turn 220 -
present)", and the model correctly reported that it had been given no date.

A full rerun would answer 1,540 questions twice to measure a change that cannot
touch most of them. This asks only the questions where the fix has something to
act on, drawn from the run's own answers:

  * answers that pleaded no date was given
  * answers that quoted a turn number back as if it were one
  * answers that hedged or narrated the context instead of committing
  * every "when" question, whether or not it went wrong

Both arms answer through the same prompt, the same model and the same judge as
the original run. The only difference between them is which store they read, and
the two stores differ only in the turn_timestamps table. Retrieval is identical,
so any change is attributable to what the model was shown, not to what was
found.

Arm A is the untouched run directory; arm B is a copy with the dates backfilled.
Neither is written to: retrieval defers its reinforcement writes and the buffer
is never flushed.

Usage:
    python scripts/benchmarks/_ab_dates.py \
        --run-a scripts/benchmarks/results/full_v3 \
        --run-b /c/nmafc_ab
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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
from nmafc.schemas.memory import DecayConfig  # noqa: E402
from nmafc.storage.config import NMafcConfig, StorageConfig  # noqa: E402
from nmafc.wrapper import NeuromorphicMemory  # noqa: E402

from scripts.benchmarks.arms.neuromorphic_tuned import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
)
from scripts.benchmarks.arms.base import strip_answer  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402

SCORED = {"single-hop", "temporal", "multi-hop", "open-domain"}

HEDGE = re.compile(
    r"no specific|not recorded|aren't recorded|do not (mention|specify)|"
    r"don't (mention|specify)|no information|not mention|isn't (mentioned|recorded)|"
    r"not specified|unclear|cannot determine|can't determine|let me |"
    r"the facts (only )?(mention|say|state|show)|based on the facts",
    re.I,
)
TURN_QUOTED = re.compile(r"turn [0-9]+", re.I)
WHEN = re.compile(r"^\s*when\b|^\s*(in what|what) (year|month|date)|how long ago", re.I)


def affected(run: Path) -> set[str]:
    """Questions the date fix could plausibly move, from the run's own answers."""
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    arm = data["results"][next(iter(data["results"]))]
    picked = set()
    for q in arm["question_results"]:
        if q["category"] not in SCORED:
            continue
        predicted = str(q["predicted"])
        if (
            HEDGE.search(predicted)
            or TURN_QUOTED.search(predicted)
            or WHEN.search(q["question"])
        ):
            picked.add(q["question"])
    return picked


def open_memory(store: Path, llm, embedder) -> NeuromorphicMemory:
    return NeuromorphicMemory(
        llm_provider=llm,
        embedding_provider=embedder,
        config=NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(store / "hot_lancedb"),
                cold_uri=str(store / "cold.db"),
            ),
            decay=DecayConfig(max_hops=2, defer_reinforcement_writes=True),
        ),
    )


async def answer(memory: NeuromorphicMemory, llm, question: str) -> str:
    retrieved = await memory._router.retrieve(question, memory.current_turn + 1)
    context = memory._router.format_context(retrieved)
    system = ANSWER_SYSTEM_PROMPT + (f"\n\n{context}" if context else "")
    text = await llm.chat(
        messages=[{"role": "user", "content": question}],
        system_prompt=system,
    )
    return strip_answer(text)


async def with_retries(coro_factory, attempts: int = 5):
    for attempt in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(2 ** (attempt + 1))
            print(f"      [retry {attempt + 1}] {type(exc).__name__}")
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> None:
    run_a, run_b = Path(args.run_a), Path(args.run_b)
    wanted = affected(run_a)
    print(f"questions the date fix could touch: {len(wanted)}\n")

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    gate = asyncio.Semaphore(args.concurrency)

    rows: list[dict] = []

    for conv in load_locomo():
        store_a = run_a / "stores" / f"{args.arm}__{conv.sample_id}"
        store_b = run_b / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store_a.is_dir() or not store_b.is_dir():
            continue
        questions = [qa for qa in conv.qa_pairs if qa.question in wanted]
        if args.limit:
            questions = questions[: args.limit]
        if not questions:
            continue

        mem_a = open_memory(store_a, llm, embedder)
        mem_b = open_memory(store_b, llm, embedder)
        try:
            async def one(qa):
                async with gate:
                    pred_a = await with_retries(lambda: answer(mem_a, llm, qa.question))
                    pred_b = await with_retries(lambda: answer(mem_b, llm, qa.question))
                    ja = await with_retries(
                        lambda: judge_answer(qa.question, pred_a, str(qa.answer), judge)
                    )
                    jb = await with_retries(
                        lambda: judge_answer(qa.question, pred_b, str(qa.answer), judge)
                    )
                    return {
                        "conv": conv.sample_id,
                        "question": qa.question,
                        "category": qa.category_name,
                        "gold": str(qa.answer),
                        "a": pred_a, "b": pred_b,
                        "a_ok": ja.correct, "b_ok": jb.correct,
                    }

            done = await asyncio.gather(*(one(qa) for qa in questions))
            rows.extend(done)
            ok_a = sum(r["a_ok"] for r in done)
            ok_b = sum(r["b_ok"] for r in done)
            print(f"  {conv.sample_id}: {ok_a}/{len(done)} -> {ok_b}/{len(done)}")
        finally:
            mem_a.close()
            mem_b.close()

    if not rows:
        print("nothing answered")
        return

    out = Path(args.out)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    n = len(rows)
    a = sum(r["a_ok"] for r in rows)
    b = sum(r["b_ok"] for r in rows)
    print(f"\n{'=' * 62}")
    print(f"questions answered twice : {n}")
    print(f"  A  turn numbers        : {a:4d}  {a / n:6.1%}")
    print(f"  B  real dates          : {b:4d}  {b / n:6.1%}")
    print(f"  change                 : {b - a:+4d}  {(b - a) / n:+6.1%}")
    print(f"\n  fixed by dates         : {sum(1 for r in rows if r['b_ok'] and not r['a_ok'])}")
    print(f"  broken by dates        : {sum(1 for r in rows if r['a_ok'] and not r['b_ok'])}")

    print(f"\n  {'category':14s} {'n':>5s} {'A':>8s} {'B':>8s}")
    for cat in sorted({r["category"] for r in rows}):
        s = [r for r in rows if r["category"] == cat]
        print(f"  {cat:14s} {len(s):5d} "
              f"{sum(r['a_ok'] for r in s) / len(s):7.1%} "
              f"{sum(r['b_ok'] for r in s) / len(s):7.1%}")

    gained = [r for r in rows if r["b_ok"] and not r["a_ok"]][:10]
    if gained:
        print("\n  answers the dates fixed:")
        for r in gained:
            print(f"    Q   : {r['question'][:86]}")
            print(f"    gold: {r['gold'][:70]}")
            print(f"    was : {r['a'][:100]}")
            print(f"    now : {r['b'][:100]}\n")
    print(f"  full detail written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-a", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="scripts/benchmarks/results/ab_dates.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
