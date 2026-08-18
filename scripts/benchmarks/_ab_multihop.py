"""Paired A/B: does Spreading Activation actually help multi-hop questions?

Multi-hop is the category the graph exists to serve — questions whose answer is
spread across facts that no single vector hit returns. The graph was inert until
`related_entities` was added to the OpenAI tool schema, so this is the first
measurement of what it contributes.

Design notes, because the obvious version of this experiment is misleading:

* **Paired, not two independent runs.** Each question is answered twice against
  the *same* store, once with `max_hops=0` and once with `max_hops=2`. Only the
  graph differs, so the per-question delta is attributable. Comparing against the
  old pilot instead would be worthless: that "multi-hop" cell was two questions.

* **Two store copies.** Retrieval reinforces every record it returns, resetting
  weights to 1.0. Running both conditions against one store would let the first
  condition alter what the second one sees. The store is ingested once, then
  copied, so each condition gets an identical untouched snapshot.

* **Order is fixed but irrelevant** for the same reason — neither condition can
  contaminate the other once they are on separate copies.

* n=13, one conversation (conv-30 has no multi-hop questions at all). This is a
  directional signal to decide whether the full run is worth launching, not a
  publishable result. It reports per-question deltas rather than only means, so
  a mean moved by one outlier is visible.

Run:
    python -m scripts.benchmarks._ab_multihop
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

from dotenv import load_dotenv

from nmafc.integration.factory import create_embedding_provider, create_llm_provider
from nmafc.schemas.memory import DecayConfig
from nmafc.storage.config import NMafcConfig, StorageConfig

from .arms.neuromorphic import NeuromorphicArm
from .datasets.locomo_loader import load_locomo
from .evaluation.f1_score import compute_f1
from .evaluation.llm_judge import judge_answer
from .resilience import RateLimiter, RetryingLLMProvider

load_dotenv()

CONDITIONS = [("graph OFF", 0), ("graph ON ", 2)]


async def run_condition(
    label: str,
    max_hops: int,
    store_dir: str,
    questions: list,
    llm,
    embedder,
) -> list[dict]:
    config = NMafcConfig(
        storage=StorageConfig(),
        decay=DecayConfig(max_hops=max_hops),
    )
    arm = NeuromorphicArm(
        llm_provider=llm,
        embedding_provider=embedder,
        config=config,
        storage_dir=store_dir,
    )
    print(f"\n  [{label}] answering {len(questions)} questions "
          f"(max_hops={max_hops}, store holds {arm._memory._hot.count()} records)")

    out = []
    for i, qa in enumerate(questions, 1):
        resp = await arm.answer_question(qa.question)
        out.append({
            "question": qa.question,
            "gold": qa.answer,
            "predicted": resp.answer,
            "f1": compute_f1(resp.answer, qa.answer),
            "context_tokens": resp.context_tokens,
            "latency_ms": resp.latency_ms,
        })
        print(f"    {i}/{len(questions)}  f1={out[-1]['f1']:.3f}  "
              f"ctx={resp.context_tokens}")
    arm._memory.close()
    return out


async def main() -> None:
    conv = next(c for c in load_locomo() if c.sample_id == "conv-26")
    questions = [qa for qa in conv.qa_pairs if qa.category == 3]
    turns = conv.get_flat_history()
    print(f"{conv.sample_id}: {len(turns)} turns, "
          f"{len(questions)} multi-hop questions")

    limiter = RateLimiter()
    llm = RetryingLLMProvider(
        create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"]), limiter
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    # ---- Ingest once. This is the expensive half of the experiment. ----
    base_dir = tempfile.mkdtemp(prefix="nmafc_ab_base_")
    arm = NeuromorphicArm(
        llm_provider=llm,
        embedding_provider=embedder,
        config=NMafcConfig(storage=StorageConfig(), decay=DecayConfig()),
        storage_dir=base_dir,
    )
    print(f"\ningesting {conv.sample_id} ...")
    await arm.ingest_conversation(turns)
    print(f"ingested. records in Hot RAM: {arm._memory._hot.count()}")
    arm._memory.close()

    results = {}
    for label, hops in CONDITIONS:
        copy_dir = tempfile.mkdtemp(prefix=f"nmafc_ab_h{hops}_")
        shutil.rmtree(copy_dir)
        shutil.copytree(base_dir, copy_dir)
        results[label] = await run_condition(
            label, hops, copy_dir, questions, llm, embedder
        )

    # ---- Judge both conditions with the independent judge. ----
    judge = RetryingLLMProvider(
        create_llm_provider(os.environ.get("NMAFC_BENCH_JUDGE")
                            or os.environ["NMAFC_BENCH_PROVIDER"]),
        RateLimiter(),
    )
    print("\njudging ...")
    for label in results:
        for row in results[label]:
            verdict = await judge_answer(
                row["question"], row["predicted"], row["gold"], judge
            )
            row["judge"] = verdict.correct

    # ---- Report per-question, not just means. ----
    off, on = results["graph OFF"], results["graph ON "]
    print("\n" + "=" * 78)
    print(f"{'#':>2}  {'F1 off':>7} {'F1 on':>7} {'delta':>7}   "
          f"{'ctx off':>7} {'ctx on':>7}   judge off/on")
    print("-" * 78)
    for i, (a, b) in enumerate(zip(off, on), 1):
        mark = "  <-- " if abs(b["f1"] - a["f1"]) > 0.05 else "      "
        print(f"{i:>2}  {a['f1']:>7.3f} {b['f1']:>7.3f} {b['f1'] - a['f1']:>+7.3f}   "
              f"{a['context_tokens']:>7} {b['context_tokens']:>7}   "
              f"{str(a['judge'])[0]}/{str(b['judge'])[0]}{mark}")

    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    print("-" * 78)
    print(f"{'mean':>2}  {mean(off, 'f1'):>7.3f} {mean(on, 'f1'):>7.3f} "
          f"{mean(on, 'f1') - mean(off, 'f1'):>+7.3f}   "
          f"{mean(off, 'context_tokens'):>7.0f} {mean(on, 'context_tokens'):>7.0f}")
    print(f"\njudge accuracy   OFF {sum(r['judge'] for r in off)}/{len(off)}"
          f"   ON {sum(r['judge'] for r in on)}/{len(on)}")
    print(f"latency (ms)     OFF {mean(off, 'latency_ms'):.0f}"
          f"   ON {mean(on, 'latency_ms'):.0f}")

    better = sum(1 for a, b in zip(off, on) if b["f1"] > a["f1"] + 0.05)
    worse = sum(1 for a, b in zip(off, on) if b["f1"] < a["f1"] - 0.05)
    print(f"\nper-question: graph helped {better}, hurt {worse}, "
          f"unchanged {len(off) - better - worse}  (n={len(off)})")
    print("n=13 on one conversation -- directional only, not a result.")


if __name__ == "__main__":
    asyncio.run(main())
