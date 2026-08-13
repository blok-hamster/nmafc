"""Run the LoCoMo benchmark suite.

Evaluates 3 arms on the LoCoMo dataset using both F1 scoring
and LLM-as-judge. Follows the evaluation protocol from
Maharana et al. (2024).

Usage:
    python -m scripts.benchmarks.run_locomo \
        --provider bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
    python -m scripts.benchmarks.run_locomo \
        --arms neuromorphic --conversations 2
    python -m scripts.benchmarks.run_locomo \
        --categories 1,2 --output results/locomo_quick/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Windows terminals default to cp1252, which cannot encode the box-drawing
# characters in the progress output (nor much of the dataset text). A run that
# dies hours in on a print() is not an acceptable failure mode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from nmafc.integration.factory import (
    create_embedding_provider,
    create_llm_provider,
)

from .arms.base import BenchmarkArm
from .arms.neuromorphic import NeuromorphicArm
from .arms.neuromorphic_tuned import NeuromorphicTunedArm
from .arms.rag import RagArm
from .arms.raw_llm import RawLLMArm
from .arms.stateful_nodecay import StatefulNoDecayArm
from .datasets.locomo_loader import (
    CATEGORY_NAMES,
    LoCoMoConversation,
    get_dataset_stats,
    load_locomo,
)
from .evaluation.f1_score import compute_f1
from .evaluation.llm_judge import judge_batch
from .evaluation.metrics import BenchmarkResult
from .resilience import (
    RateLimiter,
    RetryingEmbeddingProvider,
    RetryingLLMProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark suite")
    parser.add_argument(
        "--arms",
        default="raw,rag,neuromorphic,neuromorphic_tuned",
        help="Comma-separated arm names to evaluate",
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=None,
        help="Number of conversations to evaluate (default: all)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="QA categories to include: 1,2,3,4,5 (default: all)",
    )
    parser.add_argument(
        "--output",
        default="scripts/benchmarks/results/locomo/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("NMAFC_BENCH_PROVIDER", "ollama/llama3.2"),
        help="LLM provider string (e.g. bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0)",
    )
    parser.add_argument(
        "--judge",
        default=os.environ.get("NMAFC_BENCH_JUDGE"),
        help=(
            "Judge provider string (env: NMAFC_BENCH_JUDGE). Defaults to "
            "--provider, which makes the answering model grade its own output; "
            "prefer a different model family so the judge is independent of "
            "every arm."
        ),
    )
    parser.add_argument(
        "--embedding",
        default=os.environ.get("NMAFC_BENCH_EMBEDDING", "ollama/nomic-embed-text"),
        help="Embedding provider string",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation (F1 only)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("NMAFC_BENCH_CONCURRENCY", "8")),
        help=(
            "Conversations evaluated in parallel. Measured quota is 500k TPM / "
            "500 RPM, which at observed latency saturates near 9; higher values "
            "mostly produce 429s."
        ),
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=int(os.environ.get("NMAFC_BENCH_JUDGE_CONCURRENCY", "12")),
        help="Parallel judge calls (judging runs as a separate phase)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap questions per conversation (pilot runs only; omit for the full set)",
    )
    return parser.parse_args()


def _write_checkpoint(
    output_dir: Path, arm_name: str, by_conversation: dict[str, list[dict]]
) -> None:
    """Persist per-conversation results so a killed run can resume."""
    path = output_dir / f"checkpoint_{arm_name}.json"
    payload = {
        "arm": arm_name,
        "completed_conversations": len(by_conversation),
        "by_conversation": by_conversation,
    }
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)  # atomic, so a crash mid-write cannot corrupt the file


def _load_checkpoints(output_dir: Path, arm_names: list[str]) -> dict[str, dict]:
    """Read prior checkpoints so completed conversations are not re-run."""
    completed: dict[str, dict] = {}
    for name in arm_names:
        arm_label = {"raw": "raw_llm", "stateful": "stateful_nodecay"}.get(name, name)
        path = output_dir / f"checkpoint_{arm_label}.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            completed[arm_label] = data.get("by_conversation", {})
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: ignoring unreadable checkpoint {path.name}: {exc}")
    return completed


def make_arm(name: str, llm_provider, embedding_provider) -> BenchmarkArm | None:
    """Build one arm instance. Each parallel worker needs its own.

    Arms hold live memory state and call reset() between conversations, so a
    single shared instance cannot be evaluated on two conversations at once.
    """
    if name == "raw":
        return RawLLMArm(llm_provider=llm_provider)
    if name == "rag":
        return RagArm(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
        )
    if name == "stateful":
        return StatefulNoDecayArm(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
        )
    if name == "neuromorphic":
        return NeuromorphicArm(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
        )
    if name == "neuromorphic_tuned":
        return NeuromorphicTunedArm(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
        )
    print(f"WARNING: Unknown arm '{name}', skipping")
    return None


def create_arms(
    arm_names: list[str],
    llm_provider,
    embedding_provider,
) -> list[BenchmarkArm]:
    """Instantiate requested benchmark arms."""
    arms = [make_arm(n, llm_provider, embedding_provider) for n in arm_names]
    return [a for a in arms if a is not None]


def merge_metrics(target: BenchmarkArm, workers: list[BenchmarkArm]) -> None:
    """Fold per-worker metrics back into one arm-level view."""
    for w in workers:
        target.metrics._latencies.extend(w.metrics._latencies)
        target.metrics._prompt_tokens.extend(w.metrics._prompt_tokens)
        target.metrics._completion_tokens.extend(w.metrics._completion_tokens)
        target.metrics._context_tokens.extend(w.metrics._context_tokens)
        target.metrics.hot_storage_records += w.metrics.hot_storage_records
        target.metrics.cold_storage_events += w.metrics.cold_storage_events
    # Storage counts are per-conversation; report the average footprint rather
    # than a sum across every conversation the worker happened to handle.
    n = max(1, len(workers))
    target.metrics.hot_storage_records //= n
    target.metrics.cold_storage_events //= n


async def evaluate_arm_on_conversation(
    arm: BenchmarkArm,
    conv: LoCoMoConversation,
    categories: list[int] | None,
    max_questions: int | None = None,
) -> list[dict]:
    """Run one arm on one conversation, return per-question results.

    Questions stay sequential within a conversation on purpose: retrieval
    reinforces memory (LTP) for the stateful arms, so answering concurrently
    against one arm instance would let questions interleave and perturb each
    other's state. Parallelism comes from running conversations side by side,
    each on its own arm instance.

    Judging is deliberately not done here — it runs as a separate batched
    phase so judge calls can be parallelized independently of answering.
    """
    arm.reset()

    # Ingest all sessions
    history = conv.get_flat_history()
    await arm.ingest_conversation(history)

    results = []
    qa_pairs = conv.qa_pairs
    if categories:
        qa_pairs = [qa for qa in qa_pairs if qa.category in categories]
    if max_questions is not None:
        qa_pairs = qa_pairs[:max_questions]

    for i, qa in enumerate(qa_pairs):
        try:
            response = await arm.answer_question(qa.question)
        except Exception as e:
            print(f"    ERROR on Q{i}: {e}")
            results.append({
                "question": qa.question,
                "gold_answer": qa.answer,
                "predicted": "",
                "category": qa.category_name,
                "f1": 0.0,
                "judge_correct": None,
                "error": str(e),
            })
            continue

        results.append({
            "question": qa.question,
            "gold_answer": qa.answer,
            "predicted": response.answer,
            "category": qa.category_name,
            "f1": compute_f1(response.answer, qa.answer),
            "judge_correct": None,
            "latency_ms": response.latency_ms,
            "context_tokens": response.context_tokens,
        })

    arm.update_storage_metrics()
    return results


async def run_judge_phase(
    results: list[dict],
    judge_provider,
    concurrency: int,
) -> None:
    """Judge every answered question in parallel, updating rows in place."""
    scorable = [r for r in results if not r.get("error")]
    if not scorable:
        return

    print(f"    judging {len(scorable)} answers (concurrency {concurrency})...")
    verdicts = await judge_batch(
        items=[
            {
                "question": r["question"],
                "predicted": r["predicted"],
                "gold_answer": r["gold_answer"],
            }
            for r in scorable
        ],
        judge_provider=judge_provider,
        concurrency=concurrency,
    )
    for row, verdict in zip(scorable, verdicts):
        row["judge_correct"] = verdict.correct if verdict is not None else None


async def run_benchmark(args: argparse.Namespace) -> None:
    """Main benchmark loop."""
    print("=" * 70)
    print("LoCoMo Benchmark Suite — NMAFC")
    print("=" * 70)

    # Load dataset
    print("\n[1/4] Loading LoCoMo dataset from HuggingFace...")
    conversations = load_locomo()
    stats = get_dataset_stats(conversations)
    print(f"  Loaded {stats['conversations']} conversations, "
          f"{stats['total_qa_pairs']} QA pairs, "
          f"{stats['total_turns']} turns")

    if args.conversations:
        conversations = conversations[:args.conversations]
        print(f"  (limited to first {args.conversations} conversations)")

    categories = None
    if args.categories:
        categories = [int(c) for c in args.categories.split(",")]
        print(f"  Categories: {[CATEGORY_NAMES[c] for c in categories]}")

    # Initialize providers
    print("\n[2/4] Initializing providers...")
    print(f"  LLM: {args.provider}")
    print(f"  Embedding: {args.embedding}")
    # One limiter shared by every arm and by the judge: the quota is
    # per-deployment, so concurrent arms draw from the same bucket.
    limiter = RateLimiter()
    llm_provider = RetryingLLMProvider(create_llm_provider(args.provider), limiter)
    embedding_provider = RetryingEmbeddingProvider(
        create_embedding_provider(args.embedding),
        limiter=None if args.embedding.startswith(("ollama/", "fastembed/")) else limiter,
    )
    print(f"  Concurrency: {args.concurrency} conversations, "
          f"{args.judge_concurrency} judges")

    judge_provider = None
    if not args.skip_judge:
        judge_str = args.judge or args.provider
        print(f"  Judge: {judge_str}")
        if judge_str == args.provider:
            print("  WARNING: judge and answering model are identical — the model "
                  "is grading its own output. Set NMAFC_BENCH_JUDGE to a "
                  "different model family for an independent judge.")
        # A judge on a different deployment has its own quota, so it must not
        # queue behind the answering model's limiter; sharing one bucket across
        # two providers throttles calls that were never rate-limited.
        judge_limiter = limiter if judge_str == args.provider else RateLimiter()
        judge_provider = RetryingLLMProvider(
            create_llm_provider(judge_str), judge_limiter
        )

    # Create arms
    arm_names = [n.strip() for n in args.arms.split(",")]
    print(f"\n[3/4] Arms: {arm_names}")

    # Run evaluation
    print("\n[4/4] Running evaluation...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = _load_checkpoints(output_dir, arm_names) if args.checkpoint else {}

    all_results: dict[str, BenchmarkResult] = {}

    for arm_name in arm_names:
        template = make_arm(arm_name, llm_provider, embedding_provider)
        if template is None:
            continue

        print(f"\n{'─' * 50}")
        print(f"  ARM: {template.name}")
        print(f"{'─' * 50}")

        done_conversations = completed.get(template.name, {})
        pending = [c for c in conversations if c.sample_id not in done_conversations]
        all_question_results = [
            row for c in conversations if c.sample_id in done_conversations
            for row in done_conversations[c.sample_id]
        ]
        if all_question_results:
            print(f"  resuming: {len(done_conversations)} conversations already "
                  f"complete, {len(pending)} to go")

        # One arm instance per worker slot, reused across conversations.
        n_workers = max(1, min(args.concurrency, len(pending)))
        workers = [
            make_arm(arm_name, llm_provider, embedding_provider)
            for _ in range(n_workers)
        ]
        queue: asyncio.Queue = asyncio.Queue()
        for ci, conv in enumerate(pending):
            queue.put_nowait((ci, conv))

        results_lock = asyncio.Lock()
        progress = {"done": 0}

        async def worker(slot: int) -> None:
            arm = workers[slot]
            while True:
                try:
                    ci, conv = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    rows = await evaluate_arm_on_conversation(
                        arm=arm, conv=conv, categories=categories,
                        max_questions=args.max_questions,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad conversation
                    print(f"    ERROR conversation {conv.sample_id}: {exc}")
                    rows = []

                async with results_lock:
                    all_question_results.extend(rows)
                    done_conversations[conv.sample_id] = rows
                    progress["done"] += 1
                    print(f"  [{template.name}] conv {progress['done']}/{len(pending)} "
                          f"({conv.sample_id}) — {len(rows)} answered")
                    _write_checkpoint(output_dir, template.name, done_conversations)

        start_arm = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(n_workers)))
        print(f"  answering finished in {(time.perf_counter() - start_arm) / 60:.1f} min")

        if not args.skip_judge and judge_provider:
            await run_judge_phase(
                all_question_results, judge_provider, args.judge_concurrency
            )
            # Judge verdicts were written into the same row objects the
            # checkpoint holds, so re-dump to persist them.
            _write_checkpoint(output_dir, template.name, done_conversations)

        merge_metrics(template, workers)

        # Aggregate results
        benchmark_result = _aggregate_results(template, all_question_results)
        all_results[template.name] = benchmark_result

        if isinstance(llm_provider, RetryingLLMProvider):
            print(f"  api: {llm_provider.stats()} | limiter: {limiter.stats()}")

        # Print summary for this arm
        _print_arm_summary(benchmark_result)

    # Save final results
    final_output = {
        "metadata": {
            "dataset": "locomo",
            "provider": args.provider,
            "embedding": args.embedding,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "conversations_evaluated": len(conversations),
        },
        "results": {name: result.to_dict() for name, result in all_results.items()},
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 70}")

    # Print comparison table
    _print_comparison_table(all_results)


def _aggregate_results(arm: BenchmarkArm, question_results: list[dict]) -> BenchmarkResult:
    """Aggregate per-question results into a BenchmarkResult."""
    result = BenchmarkResult(
        arm_name=arm.name,
        dataset="locomo",
        metrics=arm.metrics,
        question_results=question_results,
    )

    if not question_results:
        return result

    # Overall F1
    f1_scores = [r["f1"] for r in question_results]
    result.overall_f1 = sum(f1_scores) / len(f1_scores)

    # Overall judge accuracy
    judged = [r for r in question_results if r.get("judge_correct") is not None]
    if judged:
        result.overall_accuracy = sum(1 for r in judged if r["judge_correct"]) / len(judged)

    # Per-category breakdown
    categories_seen: dict[str, list[dict]] = {}
    for r in question_results:
        cat = r["category"]
        categories_seen.setdefault(cat, []).append(r)

    for cat, cat_results in categories_seen.items():
        result.f1_by_category[cat] = sum(r["f1"] for r in cat_results) / len(cat_results)
        cat_judged = [r for r in cat_results if r.get("judge_correct") is not None]
        if cat_judged:
            result.accuracy_by_category[cat] = (
                sum(1 for r in cat_judged if r["judge_correct"]) / len(cat_judged)
            )

    return result


def _print_arm_summary(result: BenchmarkResult) -> None:
    """Print a summary for one arm."""
    print(f"\n  Summary for {result.arm_name}:")
    print(f"    Overall F1:       {result.overall_f1:.3f}")
    if result.overall_accuracy > 0:
        print(f"    Judge Accuracy:   {result.overall_accuracy:.3f}")
    print("    Per-category F1:")
    for cat, f1 in sorted(result.f1_by_category.items()):
        acc = result.accuracy_by_category.get(cat, 0)
        print(f"      {cat:20s}: F1={f1:.3f}  Acc={acc:.3f}")
    if result.metrics:
        print(f"    Avg latency:      {result.metrics.avg_latency_ms:.0f} ms")
        print(f"    Avg context:      {result.metrics.avg_context_tokens:.0f} tokens")
        print(f"    Total tokens:     {result.metrics.total_tokens:,}")


def _print_comparison_table(results: dict[str, BenchmarkResult]) -> None:
    """Print a comparison table across all arms."""
    print(f"\n{'=' * 70}")
    print(f"{'COMPARISON TABLE':^70}")
    print(f"{'=' * 70}")
    print(f"{'Arm':<20} {'F1':>8} {'Accuracy':>10} {'Avg Ctx':>10} {'Latency':>10} {'Tokens':>12}")
    print(f"{'─' * 70}")
    for name, r in results.items():
        ctx = f"{r.metrics.avg_context_tokens:.0f}" if r.metrics else "—"
        lat = f"{r.metrics.avg_latency_ms:.0f}ms" if r.metrics else "—"
        tok = f"{r.metrics.total_tokens:,}" if r.metrics else "—"
        print(f"{name:<20} {r.overall_f1:>8.3f} {r.overall_accuracy:>10.3f} "
              f"{ctx:>10} {lat:>10} {tok:>12}")
    print(f"{'=' * 70}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
