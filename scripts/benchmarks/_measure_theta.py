"""Measure the real top-1 cosine similarity distribution for LoCoMo questions.

DecayConfig.theta decides when Hot RAM is judged to hold nothing on-topic and
the Cold ROM keyword fallback fires. Until the search metric was fixed the score
was clamped to 0.0 for every hit, so the threshold was unreachable and its value
never mattered. Now that it does, this reports what real questions actually
score against a real store, so theta can be set from the score distribution
rather than guessed.

Deliberately reports similarity only -- it does not look at whether the answer
was correct. Tuning a threshold against benchmark answers is fitting the test
set; choosing it from the separation between on-topic and off-topic queries is
not.

Throwaway diagnostic, not part of the benchmark. Run:
    python -m scripts.benchmarks._measure_theta <store_dir> [n_questions]
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys

from dotenv import load_dotenv

from nmafc.integration.factory import create_embedding_provider
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage

from .datasets.locomo_loader import load_locomo

load_dotenv()


async def main() -> None:
    store_dir = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    # Take the scope from the stored rows, not from the environment: HotStorage
    # filters every search on agent_id + conversation_id, and these stores were
    # written with StorageConfig's defaults rather than the NMAFC_AGENT_ID in
    # .env. Reading the env here silently filters out all 410 records and every
    # question then scores 0.000.
    import lancedb

    probe = lancedb.connect(os.path.join(store_dir, "hot_lancedb"))
    sample = probe.open_table("memory_vectors").search().limit(200).to_list()

    config = StorageConfig(
        hot_uri=os.path.join(store_dir, "hot_lancedb"),
        cold_uri=os.path.join(store_dir, "cold.db"),
        embedding_dim=len(sample[0]["vector"]),
        agent_id=sample[0]["agent_id"],
        conversation_id=sample[0]["conversation_id"],
    )
    hot = HotStorage(config)
    print(f"store: {store_dir}")
    print(f"records: {hot.count()}  scope: "
          f"{config.agent_id}/{config.conversation_id}  dim: {config.embedding_dim}")

    # Identify which conversation this store holds by matching entity names
    # against each conversation's speakers -- the temp dirs are named by a random
    # suffix, so there is nothing in the path to go on.
    entity_blob = " ".join(r["entity_name"].lower() for r in sample)

    convs = load_locomo()
    best, best_score = None, -1
    for conv in convs:
        hits = sum(
            entity_blob.count(name.lower())
            for name in (conv.speaker_a, conv.speaker_b)
        )
        if hits > best_score:
            best, best_score = conv, hits
    assert best is not None
    print(f"matched: {best.sample_id} ({best.speaker_a} / {best.speaker_b}), "
          f"{best_score} entity-name hits")

    # Questions this store CAN answer, and questions it cannot. The second set
    # comes from a different conversation -- different people, different events
    # -- so it is the "Hot RAM holds nothing on-topic" case that theta exists to
    # detect. Setting theta from the gap between these two distributions uses no
    # answer keys and so is not fitting the test set.
    other = next(c for c in convs if c.sample_id != best.sample_id)
    on_topic = [qa.question for qa in best.qa_pairs][:limit]
    off_topic = [qa.question for qa in other.qa_pairs][:limit]

    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    async def top1_scores(questions: list[str]) -> list[float]:
        scores = []
        for q in questions:
            hits = hot.search(await embedder.embed_single(q), top_k=1)
            scores.append(hits[0].score if hits else 0.0)
        return sorted(scores)

    on = await top1_scores(on_topic)
    off = await top1_scores(off_topic)

    def summarise(label: str, xs: list[float]) -> None:
        n = len(xs)
        print(f"\n{label} (n={n})")
        print(f"  min {xs[0]:.3f}   p10 {xs[int(0.10 * n)]:.3f}   "
              f"median {statistics.median(xs):.3f}   "
              f"p90 {xs[min(n - 1, int(0.90 * n))]:.3f}   max {xs[-1]:.3f}")

    print("\n=== Top-1 cosine similarity ===")
    summarise(f"ON-topic  ({best.sample_id}, answerable here)", on)
    summarise(f"OFF-topic ({other.sample_id}, not in this store)", off)

    print("\ntheta   on-topic falling back   off-topic correctly falling back")
    for theta in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        fp = sum(1 for s in on if s < theta)
        tp = sum(1 for s in off if s < theta)
        print(f"  {theta:.2f}  {fp:>3}/{len(on)} ({fp / len(on):>4.0%})"
              f"            {tp:>3}/{len(off)} ({tp / len(off):>4.0%})")


if __name__ == "__main__":
    asyncio.run(main())
