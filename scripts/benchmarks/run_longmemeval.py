"""Run the LongMemEval benchmark suite.

Evaluates 3 arms on the LongMemEval dataset using
LLM-as-judge scoring. Follows the evaluation protocol
from the Zep paper (Rasmussen et al., 2025).

Usage:
    python -m scripts.benchmarks.run_longmemeval \
        --provider bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
    python -m scripts.benchmarks.run_longmemeval \
        --variant oracle --limit 50
    python -m scripts.benchmarks.run_longmemeval \
        --arms neuromorphic \
        --types temporal-reasoning,multi-session
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

from nmafc.integration.factory import create_embedding_provider, create_llm_provider

from .arms.base import BenchmarkArm
from .arms.neuromorphic import NeuromorphicArm
from .arms.raw_llm import RawLLMArm
from .arms.stateful_nodecay import StatefulNoDecayArm
from .datasets.longmemeval_loader import (
    LongMemEvalQuestion,
    get_dataset_stats,
    load_longmemeval,
)
from .evaluation.f1_score import compute_f1
from .evaluation.llm_judge import judge_answer
from .evaluation.metrics import BenchmarkResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark suite")
    parser.add_argument(
        "--arms",
        default="raw,stateful,neuromorphic",
        help="Comma-separated arm names to evaluate",
    )
    parser.add_argument(
        "--variant",
        default="oracle",
        choices=["oracle", "s", "m"],
        help="LongMemEval variant: oracle (shortest), s, m (longest context)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions to evaluate (default: all 500)",
    )
    parser.add_argument(
        "--types",
        default=None,
        help="Question types to include (comma-separated, default: all)",
    )
    parser.add_argument(
        "--output",
        default="scripts/benchmarks/results/longmemeval/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("NMAFC_BENCH_PROVIDER", "ollama/llama3.2"),
        help="LLM provider string",
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
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent questions per arm",
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


async def evaluate_arm_on_question(
    arm: BenchmarkArm,
    question: LongMemEvalQuestion,
    judge_provider,
) -> dict:
    """Run one arm on one LongMemEval question."""
    arm.reset()

    # Ingest the haystack sessions
    history = question.get_flat_history()
    await arm.ingest_conversation(history)

    # Answer the question
    try:
        response = await arm.answer_question(question.question)
    except Exception as e:
        return {
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "gold_answer": question.answer,
            "predicted": "",
            "f1": 0.0,
            "judge_correct": False,
            "error": str(e),
            "latency_ms": 0.0,
            "context_tokens": 0,
            "haystack_turns": question.total_turns,
        }

    # F1 score
    f1 = compute_f1(response.answer, question.answer)

    # LLM-as-judge
    judge_correct = False
    try:
        judge_result = await judge_answer(
            question=question.question,
            predicted=response.answer,
            gold_answer=question.answer,
            judge_provider=judge_provider,
        )
        judge_correct = judge_result.correct
    except Exception:
        judge_correct = False

    arm.update_storage_metrics()

    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "gold_answer": question.answer,
        "predicted": response.answer,
        "f1": f1,
        "judge_correct": judge_correct,
        "latency_ms": response.latency_ms,
        "context_tokens": response.context_tokens,
        "haystack_turns": question.total_turns,
    }


async def run_benchmark(args: argparse.Namespace) -> None:
    """Main benchmark loop."""
    print("=" * 70)
    print("LongMemEval Benchmark Suite — NMAFC")
    print("=" * 70)

    # Load dataset
    print(f"\n[1/4] Loading LongMemEval ({args.variant}) from HuggingFace...")
    questions = load_longmemeval(variant=args.variant)
    stats = get_dataset_stats(questions)
    print(
        f"  Loaded {stats['total_questions']} questions across "
        f"{len(stats['questions_by_type'])} types"
    )
    for qtype, count in stats["questions_by_type"].items():
        print(f"    {qtype}: {count}")

    # Filter by type
    if args.types:
        type_filter = [t.strip() for t in args.types.split(",")]
        questions = [q for q in questions if q.question_type in type_filter]
        print(f"  Filtered to types: {type_filter} ({len(questions)} questions)")

    # Limit
    if args.limit:
        questions = questions[:args.limit]
        print(f"  Limited to first {args.limit} questions")

    # Initialize providers
    print("\n[2/4] Initializing providers...")
    print(f"  LLM: {args.provider}")
    print(f"  Embedding: {args.embedding}")
    llm_provider = create_llm_provider(args.provider)
    embedding_provider = create_embedding_provider(args.embedding)

    judge_str = args.judge or args.provider
    print(f"  Judge: {judge_str}")
    judge_provider = create_llm_provider(judge_str)

    # Create arms
    arm_names = [n.strip() for n in args.arms.split(",")]
    print(f"\n[3/4] Arms: {arm_names}")
    arms = create_arms(arm_names, llm_provider, embedding_provider)

    # Run evaluation
    print(f"\n[4/4] Running evaluation ({len(questions)} questions × {len(arms)} arms)...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, BenchmarkResult] = {}

    for arm in arms:
        print(f"\n{'─' * 50}")
        print(f"  ARM: {arm.name}")
        print(f"{'─' * 50}")

        question_results: list[dict] = []

        for qi, question in enumerate(questions):
            if qi % 10 == 0:
                print(f"  Progress: {qi}/{len(questions)} "
                      f"({qi/len(questions)*100:.0f}%)")

            result = await evaluate_arm_on_question(
                arm=arm,
                question=question,
                judge_provider=judge_provider,
            )
            question_results.append(result)

            # Checkpoint every 25 questions
            if (qi + 1) % 25 == 0:
                checkpoint = {
                    "arm": arm.name,
                    "completed": qi + 1,
                    "results": question_results,
                }
                checkpoint_path = output_dir / f"checkpoint_{arm.name}.json"
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint, f, indent=2)

        # Aggregate
        benchmark_result = _aggregate_results(arm, question_results, args.variant)
        all_results[arm.name] = benchmark_result
        _print_arm_summary(benchmark_result)

    # Save final results
    final_output = {
        "metadata": {
            "dataset": "longmemeval",
            "variant": args.variant,
            "provider": args.provider,
            "embedding": args.embedding,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "questions_evaluated": len(questions),
        },
        "results": {name: result.to_dict() for name, result in all_results.items()},
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {results_path}")
    _print_comparison_table(all_results)


def _aggregate_results(
    arm: BenchmarkArm,
    question_results: list[dict],
    variant: str,
) -> BenchmarkResult:
    """Aggregate per-question results."""
    result = BenchmarkResult(
        arm_name=arm.name,
        dataset="longmemeval",
        variant=variant,
        metrics=arm.metrics,
        question_results=question_results,
    )

    if not question_results:
        return result

    # Overall accuracy (judge)
    judged = [r for r in question_results if "judge_correct" in r]
    if judged:
        result.overall_accuracy = sum(1 for r in judged if r["judge_correct"]) / len(judged)

    # Overall F1
    result.overall_f1 = sum(r["f1"] for r in question_results) / len(question_results)

    # Per-type breakdown
    types_seen: dict[str, list[dict]] = {}
    for r in question_results:
        qtype = r["question_type"]
        types_seen.setdefault(qtype, []).append(r)

    for qtype, type_results in types_seen.items():
        result.f1_by_category[qtype] = (
            sum(r["f1"] for r in type_results) / len(type_results)
        )
        type_judged = [r for r in type_results if "judge_correct" in r]
        if type_judged:
            result.accuracy_by_category[qtype] = (
                sum(1 for r in type_judged if r["judge_correct"]) / len(type_judged)
            )

    return result


def _print_arm_summary(result: BenchmarkResult) -> None:
    """Print summary for one arm."""
    print(f"\n  Summary for {result.arm_name}:")
    print(f"    Overall Accuracy: {result.overall_accuracy:.3f}")
    print(f"    Overall F1:       {result.overall_f1:.3f}")
    print("    Per-type accuracy:")
    for qtype, acc in sorted(result.accuracy_by_category.items()):
        f1 = result.f1_by_category.get(qtype, 0)
        print(f"      {qtype:30s}: Acc={acc:.3f}  F1={f1:.3f}")
    if result.metrics:
        print(f"    Avg latency:      {result.metrics.avg_latency_ms:.0f} ms")
        print(f"    Avg context:      {result.metrics.avg_context_tokens:.0f} tokens")


def _print_comparison_table(results: dict[str, BenchmarkResult]) -> None:
    """Print comparison table matching Zep paper format."""
    print(f"\n{'=' * 70}")
    print(f"{'COMPARISON TABLE (Zep paper format)':^70}")
    print(f"{'=' * 70}")
    print(f"{'Arm':<20} {'Accuracy':>10} {'F1':>8} {'Latency':>10} {'Ctx Tokens':>12}")
    print(f"{'─' * 70}")
    for name, r in results.items():
        lat = f"{r.metrics.avg_latency_ms:.0f}ms" if r.metrics else "—"
        ctx = f"{r.metrics.avg_context_tokens:.0f}" if r.metrics else "—"
        print(f"{name:<20} {r.overall_accuracy:>10.3f} {r.overall_f1:>8.3f} "
              f"{lat:>10} {ctx:>12}")

    # Per-type breakdown
    print(f"\n{'Per-Type Accuracy Breakdown':^70}")
    print(f"{'─' * 70}")
    all_types = set()
    for r in results.values():
        all_types.update(r.accuracy_by_category.keys())

    header = f"{'Type':<30}"
    for name in results.keys():
        header += f" {name:>12}"
    print(header)
    print(f"{'─' * 70}")

    for qtype in sorted(all_types):
        row = f"{qtype:<30}"
        for name, r in results.items():
            acc = r.accuracy_by_category.get(qtype, 0)
            row += f" {acc:>12.3f}"
        print(row)

    print(f"{'=' * 70}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
