"""Main entry point to execute the 100-case 5-Arm memory benchmark suite."""

from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if str(Path(root_dir) / "src") not in sys.path:
    sys.path.insert(0, str(Path(root_dir) / "src"))

from scripts.benchmarks.datasets import load_benchmark_dataset
from scripts.benchmarks.runners import FrameworkRunners
from scripts.benchmarks.evaluator import evaluate_result
from scripts.benchmarks.visualize import generate_visualizations


async def main() -> None:
    endpoint = "https://support-6339-resource.openai.azure.com/openai/v1"
    key = "CpWxYfzO4DnX3GaCBPz3QhsobsPrGpTtfPSXa90x4sl3pU8OTpWCJQQJ99CHACfhMk5XJ3w3AAAAACOGXXQW"
    model = "DeepSeek-V4-Pro"

    print("=" * 80, flush=True)
    print("🚀 EXECUTING 100-CASE LoCoMo & LongMemEval BENCHMARK SUITE", flush=True)
    print(f"   Model: {model} via Azure OpenAI Endpoint", flush=True)
    print("=" * 80, flush=True)

    dataset = load_benchmark_dataset()
    print(f"Loaded {len(dataset)} benchmark test cases.", flush=True)

    runners = FrameworkRunners(endpoint=endpoint, api_key=key, model=model)
    scored_results = []

    for idx, case in enumerate(dataset, start=1):
        print(f"\n[{idx}/100] Running {case.id} ({case.category}): '{case.query}'", flush=True)

        res1, res2, res3, res4, res5 = await asyncio.gather(
            runners.run_vanilla_llm(case),
            runners.run_naive_rag(case),
            runners.run_memgpt_baseline(case),
            runners.run_zep_baseline(case),
            runners.run_neuromorphic_memory(case),
        )

        scored1, scored2, scored3, scored4, scored5 = (
            evaluate_result(res1),
            evaluate_result(res2),
            evaluate_result(res3),
            evaluate_result(res4),
            evaluate_result(res5),
        )

        scored_results.extend([scored1, scored2, scored3, scored4, scored5])

        print(f"  • Vanilla LLM:      Acc={scored1.accuracy_score:.2f} | Latency={scored1.latency_sec:.2f}s", flush=True)
        print(f"  • Naive RAG:        Acc={scored2.accuracy_score:.2f} | Latency={scored2.latency_sec:.2f}s", flush=True)
        print(f"  • MemGPT/Letta:     Acc={scored3.accuracy_score:.2f} | Latency={scored3.latency_sec:.2f}s", flush=True)
        print(f"  • Zep (Graph):      Acc={scored4.accuracy_score:.2f} | Latency={scored4.latency_sec:.2f}s", flush=True)
        print(f"  • Neuromorphic V2:  Acc={scored5.accuracy_score:.2f} | Latency={scored5.latency_sec:.2f}s", flush=True)

    dashboard_path = generate_visualizations(scored_results)

    print("\n" + "=" * 80, flush=True)
    print("✅ BENCHMARK SUITE COMPLETE!", flush=True)
    print(f"   Dashboard generated: {dashboard_path}", flush=True)
    print("   Markdown report generated: charts/summary_report.md", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
