"""Check that extraction actually populates `related_entities`.

The field was declared on MemoryStateUpdate, written by HotStorage and traversed
by QueryRouter, but was missing from the OpenAI tool schema and unmentioned in
the extraction prompt -- so the model was never able to emit it and Spreading
Activation had no edges to walk. Measured baseline before the fix: 0 of 675
records across two populated benchmark stores had a single link.

This ingests real LoCoMo turns through the real pipeline and reports the link
density, which is the only way to tell the difference between "the model can now
emit links" and "the schema accepts a field nothing fills in".

Throwaway diagnostic, not part of the benchmark. Run:
    python -m scripts.benchmarks._verify_graph_links [n_turns]
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import tempfile

from dotenv import load_dotenv

from nmafc.integration.factory import create_embedding_provider, create_llm_provider
from nmafc.storage.config import NMafcConfig
from nmafc.wrapper import NeuromorphicMemory

from .arms.base import build_exchanges
from .datasets.locomo_loader import load_locomo

load_dotenv()


async def main() -> None:
    n_turns = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    conv = next(c for c in load_locomo() if c.sample_id == "conv-26")
    # Same exchange construction the real arm uses, so link density here is
    # measured on the input the benchmark actually feeds the extractor.
    exchanges = build_exchanges(conv.get_flat_history())[:n_turns]
    print(f"{conv.sample_id}: replaying {len(exchanges)} exchanges "
          f"({conv.speaker_a} / {conv.speaker_b})")

    tmpdir = tempfile.mkdtemp(prefix="nmafc_graphcheck_")
    config = NMafcConfig.from_env_or_toml("configs/default.toml")
    config.storage.hot_uri = os.path.join(tmpdir, "hot_lancedb")
    config.storage.cold_uri = os.path.join(tmpdir, "cold.db")

    memory = NeuromorphicMemory(
        llm_provider=create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"]),
        embedding_provider=create_embedding_provider(
            os.environ["NMAFC_BENCH_EMBEDDING"]
        ),
        config=config,
    )

    for i, exchange in enumerate(exchanges, 1):
        await memory.process_turn(user_msg=exchange)
        if i % 5 == 0:
            print(f"  exchange {i}/{len(exchanges)}")

    records = memory._hot.get_all()
    linked = [r for r in records if r.related_entities]
    total_links = sum(len(r.related_entities) for r in records)

    print(f"\nrecords:           {len(records)}")
    print(f"with >=1 link:     {len(linked)} "
          f"({len(linked) / max(1, len(records)):.0%})   [baseline: 0 of 675]")
    print(f"total links:       {total_links}")

    # A link only spreads activation if it names an entity that exists. Dangling
    # pointers look like a working graph in the record dump but traverse to
    # nothing, so count them separately.
    known = {r.entity_name.lower() for r in records}
    dangling = [
        rel for r in records for rel in r.related_entities if rel.lower() not in known
    ]
    print(f"dangling pointers: {len(dangling)} "
          f"({len(dangling) / max(1, total_links):.0%} of links)")

    print("\nsample:")
    for r in linked[:8]:
        print(f"  {r.entity_name} -> {r.related_entities}")

    # Link density is only good news if traversal stays selective. Spreading
    # Activation walks 2 hops, so a densely linked store can reach most of
    # itself from any starting point -- which would undo the context-efficiency
    # result (415 tokens vs RAG's 1497) and put back the per-record reinforcement
    # cost that batching just removed. Compare hop-0 against the full traversal.
    from nmafc.schemas.memory import DecayConfig
    from nmafc.integration.query_router import QueryRouter

    questions = [qa.question for qa in conv.qa_pairs][:15]
    base = memory._decay_config.model_dump()

    print(f"\nRetrieval fan-out over {len(questions)} questions "
          f"(store holds {len(records)} records, top_k={base['top_k']})")
    # theta=0.0 disables the Cold ROM fallback, isolating what the graph alone
    # contributes; the configured theta shows what a query actually receives.
    for theta, label in ((0.0, "graph only "), (base["theta"], "graph+cold")):
        for hops in (0, 1, 2):
            router = QueryRouter(
                memory._hot,
                memory._cold,
                memory._embedder,
                DecayConfig(**{**base, "max_hops": hops, "theta": theta}),
            )
            # Time it too. Every retrieved record gets reinforced, so a working
            # graph means bigger reinforcement batches -- the batching fix that
            # took retrieval from 51,068 ms to 1,322 ms was measured when the
            # graph was inert, and that number needs re-checking now it is not.
            counts, timings = [], []
            for q in questions:
                t0 = time.perf_counter()
                got = await router.retrieve(q, memory.current_turn + 1)
                timings.append((time.perf_counter() - t0) * 1000)
                counts.append(len(got))
            avg = sum(counts) / len(counts)
            print(f"  {label} theta={theta:<4} hops={hops}: avg {avg:5.1f} records "
                  f"({avg / len(records):>3.0%} of store), max {max(counts)}"
                  f"  |  {sum(timings) / len(timings):7.1f} ms/query")


if __name__ == "__main__":
    asyncio.run(main())
