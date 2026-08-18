"""Paired A/B: does searching Cold ROM by meaning help, or hurt?

Dense archive fallback shipped in the same run as two extractor changes, so
every score since has been unattributable -- three subsequent fixes targeted the
extractor and the gap never closed, which is exactly the pattern you get when
the suspect is the change nobody isolated. This isolates it.

Design, following _ab_multihop:

* **Paired against one store.** Both conditions answer the same questions from
  the same ingested memory, so the only difference is `cold_semantic_fallback`.

* **Store copies, not one store.** Retrieval reinforces every record it returns,
  resetting weights. Sharing a store would let whichever condition ran first
  change what the second one sees.

* **Ingestion is skipped entirely.** The stores are the ones the last full run
  left behind, so this costs answering and judging only -- roughly ten minutes
  against thirty. That is the whole point: ingestion is the expensive half and
  the retrieval question does not need it re-done.

* **Conditions run concurrently, questions within a condition sequentially.**
  Each condition owns its own store copy, so they cannot interfere; within a
  condition the writes from reinforcement must stay ordered.

Run:
    python -m scripts.benchmarks._ab_cold_semantic --store <dir> --conversation conv-26
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks.arms.neuromorphic_tuned import (  # noqa: E402
    NeuromorphicTunedArm,
)
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.f1_score import compute_f1  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402
from scripts.benchmarks.resilience import RateLimiter, RetryingLLMProvider  # noqa: E402

CONDITIONS = [("cabinet OFF", False), ("cabinet ON ", True)]


async def run_condition(
    label: str,
    semantic: bool,
    store_dir: str,
    questions: list,
    max_hops: int,
    theta: float,
    llm,
    embedder,
) -> list[dict]:
    # The tuned arm takes decay_overrides rather than a whole config, which is
    # what keeps it one number away from the untuned arm. It applies
    # lambda_active_context itself, and last, so passing it here would be
    # redundant at best and would look like this experiment was tuning decay.
    arm = NeuromorphicTunedArm(
        llm_provider=llm,
        embedding_provider=embedder,
        storage_dir=store_dir,
        decay_overrides={
            "max_hops": max_hops,
            "cold_semantic_fallback": semantic,
            "theta": theta,
        },
    )
    print(f"  [{label}] {len(questions)} questions, "
          f"{arm._memory._hot.count()} records in memory", flush=True)

    out = []
    for i, qa in enumerate(questions, 1):
        resp = await arm.answer_question(qa.question)
        out.append({
            "question": qa.question,
            "gold": qa.answer,
            "predicted": resp.answer,
            "category": qa.category,
            "f1": compute_f1(resp.answer, qa.answer),
            "context_tokens": resp.context_tokens,
            "latency_ms": resp.latency_ms,
        })
        if i % 25 == 0:
            print(f"    [{label}] {i}/{len(questions)}", flush=True)
    arm._memory.close()
    return out


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, help="Existing ingested store dir")
    parser.add_argument("--conversation", required=True)
    parser.add_argument("--max-hops", type=int, default=2)
    # The default of 0.45 gates the archive shut: measured over 304 LoCoMo
    # questions on real stores, Hot RAM's best hit fell below it 5 times. Both
    # conditions get the same theta, so raising it does not bias the comparison
    # -- it only makes the branch under test one the run actually reaches.
    parser.add_argument("--theta", type=float, default=0.65)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    args = parser.parse_args()

    if not Path(args.store, "cold.db").exists():
        print(f"No cold.db under {args.store}")
        return 1

    conv = next(
        (c for c in load_locomo() if c.sample_id == args.conversation), None
    )
    if conv is None:
        print(f"No such conversation: {args.conversation}")
        return 1
    questions = conv.qa_pairs
    print(f"{conv.sample_id}: {len(questions)} questions, store={args.store}\n")

    limiter = RateLimiter()
    llm = RetryingLLMProvider(
        create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"]), limiter
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    async def condition(label: str, semantic: bool) -> list[dict]:
        copy_dir = tempfile.mkdtemp(prefix="nmafc_abcold_")
        shutil.rmtree(copy_dir)
        shutil.copytree(args.store, copy_dir)
        return await run_condition(
            label, semantic, copy_dir, questions, args.max_hops, args.theta,
            llm, embedder
        )

    results = dict(
        zip(
            [c[0] for c in CONDITIONS],
            await asyncio.gather(*(condition(*c) for c in CONDITIONS)),
        )
    )

    judge = RetryingLLMProvider(
        create_llm_provider(
            os.environ.get("NMAFC_BENCH_JUDGE") or os.environ["NMAFC_BENCH_PROVIDER"]
        ),
        RateLimiter(),
    )
    print("\njudging ...", flush=True)
    sem = asyncio.Semaphore(args.judge_concurrency)

    async def mark(row: dict) -> None:
        async with sem:
            # .correct, not bool(result): JudgeResult is a dataclass and is
            # always truthy, which silently scores every answer correct.
            verdict = await judge_answer(
                row["question"], row["predicted"], row["gold"], judge
            )
            row["correct"] = verdict.correct

    await asyncio.gather(*(mark(r) for rows in results.values() for r in rows))

    print("\n" + "=" * 62)
    for label, rows in results.items():
        print("%-12s acc=%.4f f1=%.4f ctx=%.0f lat=%.0f" % (
            label,
            statistics.mean(1.0 if r["correct"] else 0.0 for r in rows),
            statistics.mean(r["f1"] for r in rows),
            statistics.mean(r["context_tokens"] for r in rows),
            statistics.mean(r["latency_ms"] for r in rows),
        ))

    off = {r["question"]: r["correct"] for r in results["cabinet OFF"]}
    on = {r["question"]: r["correct"] for r in results["cabinet ON "]}
    both = set(off) & set(on)
    gained = sum(1 for q in both if on[q] and not off[q])
    lost = sum(1 for q in both if off[q] and not on[q])
    print(f"\ncabinet ON vs OFF: gained {gained}, lost {lost}, n={len(both)}")
    try:
        from scipy.stats import binomtest
        if gained + lost:
            print("McNemar exact p = %.4f" % binomtest(gained, gained + lost, 0.5).pvalue)
        else:
            print("no discordant pairs -- the two conditions answered identically")
    except ImportError:
        pass

    by_cat: dict = defaultdict(dict)
    for label, rows in results.items():
        d = defaultdict(list)
        for r in rows:
            d[r["category"]].append(1.0 if r["correct"] else 0.0)
        for k, v in d.items():
            by_cat[k][label] = statistics.mean(v)
    print()
    for cat in sorted(by_cat):
        cells = "  ".join(f"{lab}={by_cat[cat].get(lab, float('nan')):.3f}"
                          for lab, _ in CONDITIONS)
        print(f"  category {cat}: {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
