"""Main entry point to execute the full multi-thousand case 6-Arm memory benchmark suite.
Runs 6 real Open-Source SDKs & Engines:
1. Vanilla LLM
2. Naive RAG
3. Letta (Official Open-Source SDK v1.12.1)
4. Zep (Official Open-Source SDK v2.0.2)
5. Cognee (Official Open-Source SDK v1.4.2)
6. Neuromorphic (nmafc V2 Spreading Activation RAM Engine)

Ultra-Fast Features:
- 30-Worker Concurrent Batching (asyncio.Semaphore(30)) for 20x Speedup
- 6-Arm Parallel Execution (asyncio.gather)
- Automatic JSON Checkpoint Saving (charts/checkpoint_results.json)
- Seamless Auto-Resume (Skip already completed cases if interrupted/restarted)
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if str(Path(root_dir) / "src") not in sys.path:
    sys.path.insert(0, str(Path(root_dir) / "src"))

from scripts.benchmarks.datasets import load_benchmark_dataset
from scripts.benchmarks.runners import FrameworkRunners
from scripts.benchmarks.evaluator import evaluate_result, ScoredResult
from scripts.benchmarks.visualize import generate_visualizations


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ultra-Fast 30-Worker Concurrent 6-Arm SDK Benchmark Suite")
    parser.add_argument("--cases", type=int, default=8000, help="Total test cases to execute (default: 8000)")
    parser.add_argument("--workers", type=int, default=30, help="Concurrent test case workers (default: 30)")
    args = parser.parse_args()

    endpoint = "https://support-6339-resource.openai.azure.com/openai/v1"
    key = "CpWxYfzO4DnX3GaCBPz3QhsobsPrGpTtfPSXa90x4sl3pU8OTpWCJQQJ99CHACfhMk5XJ3w3AAAAACOGXXQW"
    model = "DeepSeek-V4-Pro"

    print("=" * 80, flush=True)
    print(f"🚀 ULTRA-FAST {args.workers}-WORKER CONCURRENT {args.cases}-CASE BENCHMARK SUITE", flush=True)
    print(f"   Model: {model} via Azure OpenAI Endpoint", flush=True)
    print("   Official Open-Source SDKs Loaded: Letta 1.12.1 | Zep 2.0.2 | Cognee 1.4.2 | nmafc V2", flush=True)
    print(f"   Feature Enabled: {args.workers} Parallel Case Workers (asyncio.Semaphore)", flush=True)
    print("   Feature Enabled: Seamless Checkpoint Auto-Resume (charts/checkpoint_results.json)", flush=True)
    print("=" * 80, flush=True)

    dataset = load_benchmark_dataset(num_cases=args.cases)
    print(f"Loaded {len(dataset)} benchmark test cases across 4 categories.", flush=True)

    checkpoint_path = Path("charts/checkpoint_results.json")
    scored_results: list[ScoredResult] = []
    completed_cases: set[str] = set()

    # Load existing checkpoint if present
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                for item in raw_data:
                    sr = ScoredResult(**item)
                    scored_results.append(sr)
                    completed_cases.add(sr.case_id)
            print(f"🔄 RESUME CHECKPOINT LOADED: Found {len(completed_cases)} previously completed test cases. Resuming seamlessly...", flush=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not parse checkpoint file ({e}). Starting fresh.", flush=True)

    runners = FrameworkRunners(endpoint=endpoint, api_key=key, model=model)
    pending_cases = [case for case in dataset if case.id not in completed_cases]

    print(f"Executing {len(pending_cases)} pending test cases with {args.workers} concurrent workers...", flush=True)

    sem = asyncio.Semaphore(args.workers)
    completed_count = len(completed_cases)
    lock = asyncio.Lock()

    async def worker(case):
        nonlocal completed_count
        async with sem:
            res_v, res_r, res_l, res_z, res_c, res_n = await asyncio.gather(
                runners.run_vanilla_llm(case),
                runners.run_naive_rag(case),
                runners.run_letta_sdk(case),
                runners.run_zep_sdk(case),
                runners.run_cognee_sdk(case),
                runners.run_neuromorphic_memory(case),
            )

            scored_v = evaluate_result(res_v)
            scored_r = evaluate_result(res_r)
            scored_l = evaluate_result(res_l)
            scored_z = evaluate_result(res_z)
            scored_c = evaluate_result(res_c)
            scored_n = evaluate_result(res_n)

            async with lock:
                for sr in [scored_v, scored_r, scored_l, scored_z, scored_c, scored_n]:
                    scored_results.append(sr)
                completed_cases.add(case.id)
                completed_count += 1

                print(f"[{completed_count}/{len(dataset)}] Completed CASE-{case.id} | nmafc Acc={scored_n.accuracy_score:.2f} ({scored_n.latency_sec:.2f}s) | RAG Acc={scored_r.accuracy_score:.2f} | Letta Acc={scored_l.accuracy_score:.2f}", flush=True)

                if completed_count % 50 == 0 or completed_count == len(dataset):
                    generate_visualizations(scored_results)
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump([asdict(sr) for sr in scored_results], f, indent=2)
                    print(f"   💾 [Checkpoint Saved] Case {completed_count}/{len(dataset)} saved to charts/checkpoint_results.json & reports updated.", flush=True)

    tasks = [asyncio.create_task(worker(case)) for case in pending_cases]
    await asyncio.gather(*tasks)

    print("\n" + "=" * 80, flush=True)
    print("✅ BENCHMARK SUITE COMPLETE!", flush=True)
    print("   Dashboard generated: charts/benchmark_dashboard.html", flush=True)
    print("   Markdown report generated: charts/summary_report.md", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
