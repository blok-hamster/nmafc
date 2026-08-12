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
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nmafc.integration.factory import (
    create_embedding_provider,
    create_llm_provider,
)

from .arms.base import BenchmarkArm
from .arms.neuromorphic import NeuromorphicArm
from .arms.raw_llm import RawLLMArm
from .arms.stateful_nodecay import StatefulNoDecayArm
from .datasets.locomo_loader import (
    CATEGORY_NAMES,
    LoCoMoConversation,
    get_dataset_stats,
    load_locomo,
)
from .evaluation.f1_score import compute_f1
from .evaluation.llm_judge import judge_answer
from .evaluation.metrics import BenchmarkResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark suite")
    parser.add_argument(
        "--arms",
        default="raw,stateful,neuromorphic",
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
        default=None,
        help="Judge provider string (default: same as --provider)",
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
    return parser.parse_args()


def create_arms(
    arm_names: list[str],
    llm_provider,
    embedding_provider,
) -> list[BenchmarkArm]:
    """Instantiate requested benchmark arms."""
    arms: list[BenchmarkArm] = []
    for name in arm_names:
        if name == "raw":
            arms.append(RawLLMArm(llm_provider=llm_provider))
        elif name == "stateful":
            arms.append(StatefulNoDecayArm(
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
            ))
        elif name == "neuromorphic":
            arms.append(NeuromorphicArm(
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
            ))
        else:
            print(f"WARNING: Unknown arm '{name}', skipping")
    return arms


async def evaluate_arm_on_conversation(
    arm: BenchmarkArm,
    conv: LoCoMoConversation,
    categories: list[int] | None,
    judge_provider,
    skip_judge: bool,
) -> list[dict]:
    """Run one arm on one conversation, return per-question results."""
    arm.reset()

    # Ingest all sessions
    history = conv.get_flat_history()
    await arm.ingest_conversation(history)

    results = []
    qa_pairs = conv.qa_pairs
    if categories:
        qa_pairs = [qa for qa in qa_pairs if qa.category in categories]

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

        f1 = compute_f1(response.answer, qa.answer)

        judge_correct = None
        if not skip_judge and judge_provider:
            try:
                judge_result = await judge_answer(
                    question=qa.question,
                    predicted=response.answer,
                    gold_answer=qa.answer,
                    judge_provider=judge_provider,
                )
                judge_correct = judge_result.correct
            except Exception:
                judge_correct = None

        results.append({
            "question": qa.question,
            "gold_answer": qa.answer,
            "predicted": response.answer,
            "category": qa.category_name,
            "f1": f1,
            "judge_correct": judge_correct,
            "latency_ms": response.latency_ms,
            "context_tokens": response.context_tokens,
        })

    arm.update_storage_metrics()
    return results


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
    llm_provider = create_llm_provider(args.provider)
    embedding_provider = create_embedding_provider(args.embedding)

    judge_provider = None
    if not args.skip_judge:
        judge_str = args.judge or args.provider
        print(f"  Judge: {judge_str}")
        judge_provider = create_llm_provider(judge_str)

    # Create arms
    arm_names = [n.strip() for n in args.arms.split(",")]
    print(f"\n[3/4] Arms: {arm_names}")
    arms = create_arms(arm_names, llm_provider, embedding_provider)

    # Run evaluation
    print("\n[4/4] Running evaluation...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, BenchmarkResult] = {}

    for arm in arms:
        print(f"\n{'─' * 50}")
        print(f"  ARM: {arm.name}")
        print(f"{'─' * 50}")

        all_question_results: list[dict] = []

        for ci, conv in enumerate(conversations):
            qa_count = len(conv.qa_pairs)
            if categories:
                qa_count = len([q for q in conv.qa_pairs if q.category in categories])
            print(f"  Conv {ci+1}/{len(conversations)} ({conv.sample_id}): "
                  f"{conv.num_sessions} sessions, {qa_count} QA pairs")

            results = await evaluate_arm_on_conversation(
                arm=arm,
                conv=conv,
                categories=categories,
                judge_provider=judge_provider,
                skip_judge=args.skip_judge,
            )
            all_question_results.extend(results)

            # Checkpoint after each conversation
            checkpoint = {
                "arm": arm.name,
                "completed_conversations": ci + 1,
                "results": all_question_results,
            }
            checkpoint_path = output_dir / f"checkpoint_{arm.name}.json"
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)

        # Aggregate results
        benchmark_result = _aggregate_results(arm, all_question_results)
        all_results[arm.name] = benchmark_result

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
