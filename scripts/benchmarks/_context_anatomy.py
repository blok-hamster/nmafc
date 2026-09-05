"""What is actually occupying the prompt budget?

The wider budget buys 23 questions for 674 extra tokens per question. Before
trading any of that accuracy back for size, it is worth knowing what those
tokens are: distinct facts, repeated facts, or formatting.

Every retrieved fact is rendered as

    <fact text> (Valid: <from> - <to>)

so each one carries a fixed overhead whether or not its date is doing any work,
and a "- present" suffix that is true of most facts in the store. Duplicated
information is the other candidate: retrieval dedupes by entity name, which
does nothing about two differently-named entities carrying the same sentence.

Counts characters and converts at four to the token, the same conversion used
everywhere else in this comparison. Retrieval only -- no generation, no
judging, and the stores are opened read-only.

Usage:
    python scripts/benchmarks/_context_anatomy.py --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections import Counter
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

from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402

SCORED = {"single-hop", "temporal", "multi-hop", "open-domain"}
CHARS_PER_TOKEN = 4
VALID = re.compile(r"\s*\((?:Valid: )?[^()]*\)\s*$")


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    total = {"context": 0, "facts": 0, "suffix": 0, "header": 0}
    repeats = {"chars": 0, "facts": 0}
    present_suffix = 0
    lines_seen = 0
    questions = 0
    dupe_examples: list[tuple[str, str]] = []

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            continue
        memory = NeuromorphicMemory(
            llm_provider=llm,
            embedding_provider=embedder,
            config=NMafcConfig(
                storage=StorageConfig(
                    hot_uri=str(store / "hot_lancedb"),
                    cold_uri=str(store / "cold.db"),
                ),
                decay=DecayConfig(
                    max_hops=2,
                    top_k=args.hot_top_k,
                    fallback_keyword_limit=args.cold_budget,
                    rerank_top_k=args.budget,
                    defer_reinforcement_writes=True,
                    compact_validity=args.compact,
                ),
            ),
        )
        try:
            turn = memory.current_turn + 1
            picked = [qa for qa in conv.qa_pairs
                      if qa.category_name in SCORED][: args.limit]
            for qa in picked:
                records = await memory._router.retrieve(qa.question, turn)
                context = memory._router.format_context(records)
                if not context:
                    continue
                questions += 1
                total["context"] += len(context)

                body = 0
                seen: Counter[str] = Counter()
                for line in context.splitlines():
                    if not line.strip():
                        continue
                    stripped = VALID.sub("", line)
                    if stripped == line:
                        # No validity suffix: a header or a bare line.
                        total["header"] += len(line)
                        continue
                    lines_seen += 1
                    total["facts"] += len(stripped)
                    total["suffix"] += len(line) - len(stripped)
                    if "- present)" in line:
                        present_suffix += 1
                    key = normalise(stripped)
                    seen[key] += 1
                    if seen[key] > 1:
                        repeats["facts"] += 1
                        repeats["chars"] += len(line)
                        if len(dupe_examples) < 6:
                            dupe_examples.append((qa.question, stripped))
                    body += len(line)
        finally:
            memory.close()

    if not questions:
        print("nothing measured")
        return

    per = lambda k: total[k] / questions / CHARS_PER_TOKEN  # noqa: E731
    print(f"\n{'=' * 62}")
    print(f"questions sampled      : {questions}")
    print(f"facts per prompt       : {lines_seen / questions:.1f}")
    print(f"\n  tokens per prompt")
    print(f"    total              : {per('context'):7.0f}")
    print(f"    fact text          : {per('facts'):7.0f}   "
          f"{total['facts'] / total['context']:5.1%}")
    print(f"    '(Valid: ...)'     : {per('suffix'):7.0f}   "
          f"{total['suffix'] / total['context']:5.1%}")
    print(f"    header/other       : {per('header'):7.0f}   "
          f"{total['header'] / total['context']:5.1%}")
    print(f"\n  validity suffixes ending '- present': "
          f"{present_suffix}/{lines_seen} = {present_suffix / lines_seen:.1%}")
    print(f"  repeated fact text  : {repeats['facts']} lines, "
          f"{repeats['chars'] / questions / CHARS_PER_TOKEN:.0f} tokens per prompt")

    if dupe_examples:
        print("\n  examples of the same sentence retrieved twice:")
        for question, fact in dupe_examples:
            print(f"    [{question[:44]}]")
            print(f"      {fact[:88]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--hot-top-k", type=int, default=25)
    ap.add_argument("--cold-budget", type=int, default=40)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--compact", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
