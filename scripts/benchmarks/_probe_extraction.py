"""Fast check on what the extractor actually records, before spending a full run.

pilot6 cost an hour to discover that an extractor prompt change had halved the
number of facts recorded -- 350 down to 185 on the same conversation -- which no
amount of retrieval work can compensate for, because a fact never extracted
cannot be retrieved at any tier. That is a two-minute question answered at
one-hour cost.

This probe answers it directly: run the current prompt over a sample of real
exchanges and report facts per exchange and the tier split. Extraction here is
run concurrently, which the benchmark cannot do -- override detection reads Hot
RAM state written by earlier turns, so ingestion must stay ordered. This probe
writes nothing, so it has no such constraint.

Reference points, same conversation, first 25 exchanges:
  pilot5 prompt (before the tier rules)  ~1.9 facts/exchange, 82% CoreAnchor
  pilot6 prompt (over-aggressive)        ~1.0 facts/exchange
A fix is working if density is back near pilot5's while CoreAnchor stays down.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections import Counter
from pathlib import Path

SPEECH_ACT = re.compile(
    r"(told|tells?|said|says?|shares?|shared|asks?|asked|mentions?|mentioned"
    r"|agrees?|agreed|reacts?|reacted|discusses?|discussed|expresses?|expressed"
    r"|acknowledges?|acknowledged|suggests?|suggested|responds?|responded"
    r"|greet\w*|compliments?|replies|replied)",
    re.IGNORECASE,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from nmafc.integration.extractor import StateExtractor  # noqa: E402
from nmafc.integration.factory import create_llm_provider  # noqa: E402

from scripts.benchmarks.arms.base import build_exchanges  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", default="conv-30")
    parser.add_argument("--exchanges", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--provider",
        default=os.environ.get("NMAFC_BENCH_PROVIDER", "ollama/llama3.2"),
    )
    args = parser.parse_args()

    conversations = load_locomo()
    match = [c for c in conversations if c.sample_id == args.conversation]
    if not match:
        available = ", ".join(c.sample_id for c in conversations[:10])
        print(f"No such conversation: {args.conversation}. Available: {available}")
        return 1

    exchanges = build_exchanges(match[0].get_flat_history())[: args.exchanges]
    extractor = StateExtractor(create_llm_provider(args.provider))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def extract(text: str):
        async with semaphore:
            try:
                _, payload = await extractor.extract(user_msg=text)
                return payload.updates
            except Exception as exc:  # a failed call is not zero facts
                print(f"  extraction failed: {exc}")
                return None

    results = await asyncio.gather(*(extract(x) for x in exchanges))

    ok = [r for r in results if r is not None]
    if not ok:
        print("Every extraction failed; nothing to report.")
        return 1

    facts = [u for updates in ok for u in updates]
    tiers = Counter(u.memory_type.value for u in facts)
    empty = sum(1 for updates in ok if not updates)
    speech = [u for u in facts if SPEECH_ACT.search(u.entity_name)]

    print(f"\n{args.conversation}: {len(ok)}/{len(exchanges)} exchanges extracted")
    print(f"facts               {len(facts)}")
    print(f"facts per exchange  {len(facts) / len(ok):.2f}")
    print(f"silent exchanges    {empty} ({empty / len(ok):.0%})")
    for tier, count in tiers.most_common():
        print(f"  {tier:<16} {count:>4}  {count / len(facts):.1%}")

    # Entities named for the telling rather than the fact. pilot7 lost 6 points
    # to this: 44% of everything it marked EphemeralState was named
    # `x_shares_y_with_z`, so the real fact inside inherited the throwaway tier
    # and was deleted within a few turns. The count belongs next to the tier
    # split because the two together are what predict retrievability -- a
    # healthy tier split built out of speech-act names is still broken.
    print(f"speech-act names    {len(speech)} ({len(speech) / len(facts):.0%})")
    for u in speech[:8]:
        print(f"  ! {u.entity_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
