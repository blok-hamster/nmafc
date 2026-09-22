"""When an answer hedged, was the fact actually missing from its context?

9.6% of answers in the full_v3 run either refused ("the facts do not mention
...") or thought out loud ("Let me look more carefully..."). Those answers score
14.4% against 68.5% for the rest, and they account for 22.5% of every question
RAG got right and we did not.

That statistic on its own does not say what to fix, and the two possibilities
call for opposite work:

  * The fact was not retrieved. Then hedging is the correct response to an empty
    context, the loss belongs to extraction or search, and tightening the answer
    prompt would only convert honest refusals into confident errors -- which
    scores the same and is worse.
  * The fact was retrieved and the model declined to commit. Then the prompt is
    the bug, and it is the cheapest fix available.

So this replays retrieval for the hedged questions only and reports how often
the gold answer was present in the context that produced the hedge. Retrieval
only: no answers are generated, so it costs one embedding call per question.

Reads the run's own results file to decide which questions hedged, so the
regex lives in one place and the set stays tied to the answers it describes.

Usage:
    python scripts/benchmarks/_check_hedged_context.py \
        --run scripts/benchmarks/results/full_v3
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

from datasets.locomo_loader import load_locomo  # noqa: E402

# Refusals and leaked reasoning. Deliberately broad: an answer that opens with
# "the facts only mention" has already spent its budget describing the context
# instead of answering, whether or not it eventually commits.
HEDGE = re.compile(
    r"no specific|not recorded|aren't recorded|do not (mention|specify)|"
    r"don't (mention|specify)|no information|not mention|isn't (mentioned|recorded)|"
    r"not specified|unclear|cannot determine|can't determine|let me |"
    r"the facts (only )?(mention|say|state|show)|based on the facts",
    re.I,
)

SCORED = {"single-hop", "temporal", "multi-hop", "open-domain"}

STOP = set(
    "the a an of in on at to for is was were and or with what which who when "
    "where how did does do had has have her his its their this that".split()
)


def keywords(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9']+", str(text).lower())
        if len(w) > 2 and w not in STOP
    }


def overlap(gold: str, haystack: str) -> float:
    kw = keywords(gold)
    if not kw:
        return 0.0
    low = haystack.lower()
    return sum(1 for w in kw if w in low) / len(kw)


def hedged_questions(run: Path) -> dict[str, dict]:
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    arm = data["results"][next(iter(data["results"]))]
    return {
        q["question"]: q
        for q in arm["question_results"]
        if q["category"] in SCORED and HEDGE.search(str(q["predicted"]))
    }


async def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    hedged = hedged_questions(run_dir)
    print(f"hedged answers in {run_dir.name}: {len(hedged)}\n")

    llm = create_llm_provider(os.environ.get("NMAFC_BENCH_PROVIDER", "ollama/llama3.2"))
    embedder = create_embedding_provider(
        os.environ.get("NMAFC_BENCH_EMBEDDING", "ollama/nomic-embed-text")
    )

    present = missing = 0
    examples: list[tuple[str, str, str]] = []

    for conv in load_locomo():
        store = run_dir / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            continue
        mine = [qa for qa in conv.qa_pairs if qa.question in hedged]
        if not mine:
            continue

        memory = NeuromorphicMemory(
            llm_provider=llm,
            embedding_provider=embedder,
            config=NMafcConfig(
                storage=StorageConfig(
                    hot_uri=str(store / "hot_lancedb"),
                    cold_uri=str(store / "cold.db"),
                ),
                # Read-only: never let a diagnostic reinforce the run it measures.
                decay=DecayConfig(max_hops=2, defer_reinforcement_writes=True),
            ),
        )
        try:
            turn = memory.current_turn + 1
            for qa in mine:
                context = memory._router.format_context(
                    await memory._router.retrieve(qa.question, turn)
                )
                if overlap(qa.answer, context) >= args.threshold:
                    present += 1
                    if len(examples) < 10:
                        examples.append(
                            (qa.question, str(qa.answer),
                             str(hedged[qa.question]["predicted"]))
                        )
                else:
                    missing += 1
        finally:
            memory.close()

    total = present + missing
    if not total:
        print("nothing checked")
        return

    print("=" * 62)
    print(f"hedged answers replayed        : {total}")
    print(f"  gold WAS in their context    : {present:4d}  {present / total:6.1%}"
          "   -> prompt refused a fact it had")
    print(f"  gold was NOT in their context: {missing:4d}  {missing / total:6.1%}"
          "   -> honest refusal, retrieval's fault")

    if examples:
        print("\n  refused despite having the fact:")
        for q, gold, pred in examples:
            print(f"    Q   : {q[:88]}")
            print(f"    gold: {gold[:80]}")
            print(f"    said: {pred[:110]}")
            print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--threshold", type=float, default=0.6)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
