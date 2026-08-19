"""Is the answer missing from memory, missing from the context, or ignored?

On the full LoCoMo run the tuned arm scored 0.036 on the 446 'adversarial'
questions -- a label that turns out to be misleading, since 444 of them have
ordinary factual gold answers rather than 'not mentioned'. The system abstained
on 93% of them. A keyword sweep of the persisted stores showed the answer was
already in Hot RAM for 92.7% of the ones it got wrong, so the failure is not
storage and not the Cold ROM gate: the fact is in memory and does not reach the
answer.

That leaves exactly two possibilities, and this script separates them by
replaying retrieval against the stores the run left behind:

  ranking  -- retrieval scores the right fact below the cut, so it never enters
              the prompt. The lever is top_k, the scoring function, or theta.
  refusal  -- the fact is in the prompt and the model still says it does not
              know. The lever is the answer prompt.

Only the retrieval half is replayed. No answers are generated and no judging is
done, so the cost is one embedding call per question rather than a full run --
the point of persisting stores in the first place.

Matching a gold answer inside the context is done by keyword overlap, which is
crude: it will call a hit where the words appear in an unrelated fact. It is
deliberately the same crude test used on the store contents, so the two are
comparable, and it biases towards 'the context was fine' -- the conclusion it
would take to blame the prompt. If it still reports misses, those are real.
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

STOP = set(
    "the a an of in on at to for is was were and or with what which who when "
    "where how did does do had has have her his its their this that".split()
)


def keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", str(text).lower()) if len(w) > 2 and w not in STOP}


def overlap(gold: str, haystack: str) -> float:
    kw = keywords(gold)
    if not kw:
        return 0.0
    low = haystack.lower()
    return sum(1 for w in kw if w in low) / len(kw)


async def retrieve_with_retry(router, question: str, turn: int, attempts: int = 5):
    """Retrieve, retrying transient network failures.

    The embedding call goes over the network, and a single ConnectTimeout was
    enough to abandon a run three conversations from the end -- the overnight
    benchmark saw four such blips and rode all of them out because its own
    client retries. This does not, so it retries here. Backoff is 2/4/8/16s.
    A failure that survives all attempts is re-raised: silently returning an
    empty context would score as 'the answer was absent', turning a network
    problem into false evidence of a ranking problem.
    """
    for attempt in range(attempts):
        try:
            return await router.retrieve(question, turn)
        except Exception as exc:  # noqa: BLE001 - provider errors are not a shared base class
            if attempt == attempts - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    [retry {attempt + 1}/{attempts - 1} in {wait}s] {type(exc).__name__}")
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> None:
    checkpoint = json.load(open(args.checkpoint, encoding="utf-8"))
    stores = Path(args.stores)
    arm = checkpoint["arm"]

    llm = create_llm_provider(os.environ.get("NMAFC_BENCH_PROVIDER", "ollama/llama3.2"))
    embedder = create_embedding_provider(
        os.environ.get("NMAFC_BENCH_EMBEDDING", "ollama/nomic-embed-text")
    )

    totals = {"n": 0, "in_context": 0, "abstained": 0, "abstain_with_context": 0}
    per_conv = []
    misses: list[tuple[str, str, str]] = []
    hits: list[tuple[str, str, str]] = []

    for conv, rows in checkpoint["by_conversation"].items():
        wanted = [
            r
            for r in rows
            if r["category"] in args.categories and not r["judge_correct"]
        ]
        if not wanted:
            continue
        store = stores / f"{arm}__{conv}"
        if not store.is_dir():
            print(f"  [skip] no store for {conv}")
            continue

        config = NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(store / "hot_lancedb"), cold_uri=str(store / "cold.db")
            ),
            decay=DecayConfig(max_hops=args.max_hops, beta=args.beta),
        )
        memory = NeuromorphicMemory(
            llm_provider=llm, embedding_provider=embedder, config=config
        )
        try:
            turn = memory.current_turn + 1
            local = {"n": 0, "in_context": 0}
            for r in wanted:
                retrieved = await retrieve_with_retry(
                    memory._router, r["question"], turn
                )
                context = memory._router.format_context(retrieved)
                present = overlap(r["gold_answer"], context) >= args.threshold
                abstained = bool(
                    re.search(
                        r"don'?t know|not available|no information|cannot determine|"
                        r"not mentioned|unable to",
                        str(r["predicted"]),
                        re.I,
                    )
                )
                totals["n"] += 1
                local["n"] += 1
                totals["in_context"] += present
                local["in_context"] += present
                totals["abstained"] += abstained
                totals["abstain_with_context"] += present and abstained
                bucket = hits if present else misses
                if len(bucket) < 10:
                    bucket.append((conv, r["question"], str(r["gold_answer"])))
            per_conv.append((conv, local["n"], local["in_context"]))
            print(
                f"  {conv}: {local['in_context']}/{local['n']} "
                f"({local['in_context'] / local['n']:.0%}) had the answer in context"
            )
        finally:
            memory.close()

    n = totals["n"] or 1
    print(f"\n{'=' * 70}\nRetrieval diagnosis: {arm}, categories={args.categories}")
    print(f"{'=' * 70}")
    print(f"  wrong answers replayed              : {totals['n']}")
    print(
        f"  gold answer PRESENT in context      : {totals['in_context']} "
        f"({totals['in_context'] / n:.1%})   <- refusal (prompt problem)"
    )
    print(
        f"  gold answer ABSENT from context     : {n - totals['in_context']} "
        f"({(n - totals['in_context']) / n:.1%})   <- ranking (retrieval problem)"
    )
    print(f"  model abstained                     : {totals['abstained']} ({totals['abstained'] / n:.1%})")
    print(
        f"  abstained WITH the answer in context : {totals['abstain_with_context']} "
        f"({totals['abstain_with_context'] / n:.1%})"
    )

    print("\n  examples where the answer WAS in context but the model refused:")
    for c, q, g in hits[:6]:
        print(f"    [{c}] {q[:64]}\n          gold: {g[:60]}")
    print("\n  examples where retrieval never surfaced it:")
    for c, q, g in misses[:6]:
        print(f"    [{c}] {q[:64]}\n          gold: {g[:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stores", required=True)
    parser.add_argument("--categories", nargs="+", default=["adversarial"])
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="fraction of gold keywords that must appear in context to count as present",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
