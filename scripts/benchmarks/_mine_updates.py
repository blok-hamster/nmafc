"""Find the moments in LoCoMo where something stated earlier stops being true.

LoCoMo asks every question after the last turn, about things usually said once,
so a memory that never forgets is never punished for it. That is why decay
tuning has never bought more than about 1.65 points: the benchmark cannot see
forgetting, in either direction. To claim a forgetting mechanism helps, there
has to be something whose removal helps, and these are those questions.

The stores cannot supply them. Mining `memory_event_log` for one entity holding
two facts returns 213 entities out of 8,026, and reading them shows nearly all
are restatements rather than changes. The reason is the naming: the extractor
writes `jon_job_loss_jan_2023` and `gina_encouragement_9_july_2023`, one name
per event rather than one per subject, so two facts about the same thing land on
different names and never collide. That is also why `detect_override`, which
matches on identical entity_name, has fired zero times across 8,276 facts.

So the source is the conversation itself, which `turn_text` now holds. One call
per conversation with the whole transcript in front of it, because an update and
the thing it supersedes are routinely two hundred turns apart and no chunking
that fits a context window keeps them together.

What comes out is deliberately paired: each question carries the answer that is
now true and the answer that used to be. That second field is the measurement.
Accuracy alone cannot tell "wrong because it forgot" from "wrong because it
remembered too well", and only the second is evidence about decay.

Writes JSON only. Reads the stores read-only and touches no fact.

Usage:
    python scripts/benchmarks/_mine_updates.py --out /c/nmafc_ab/updates.json
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

from nmafc.integration.factory import create_llm_provider  # noqa: E402
from nmafc.storage.cold import ColdStorage  # noqa: E402

from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402

SYSTEM = """\
You are building an evaluation set that tests whether a memory system keeps
outdated information.

You will be given a full conversation between two people, turn by turn, with
dates. Find every place where something one of them says is later SUPERSEDED --
where a fact stated earlier stops being true because of something said later.

Genuine supersessions look like:
  - a plan that changes ("I'm going to Spain" ... later ... "we went to Portugal instead")
  - a state that ends ("I'm unemployed" ... later ... "I started my own shop")
  - a possession, relationship, address, job, pet, or health state that changes
  - an intention that is carried out, abandoned, or reversed

These are NOT supersessions, and you must exclude them:
  - the same fact restated, elaborated or summarised later
  - two different things that are both still true at the end
  - a sequence of events where each step remains true (a holiday, then a second holiday)
  - anything where you cannot name a specific earlier value that is now wrong

For each genuine supersession, write ONE question whose correct answer is the
state at the END of the conversation. The question must be answerable without
seeing the conversation, so name the person and the topic explicitly. It must
NOT hint at which answer is current, and must not use words like "now",
"currently", "still", "latest" or "most recent" -- a system that has forgotten
nothing should be genuinely able to answer it with the outdated value.

Return STRICT JSON, an object with one key "updates", holding a list of objects:
  "question"    the question, one sentence
  "gold"        the answer that is true at the end, as short as possible
  "stale"       the earlier answer that is now wrong, as short as possible
  "gold_turn"   the turn number where the current state is established
  "stale_turn"  the turn number where the outdated state was stated
  "topic"       two or three words naming what changed

Return at most 25. Quality matters far more than quantity: if a conversation
contains only three real supersessions, return three. Return an empty list
rather than padding with restatements.
"""


def transcript(store: Path) -> str:
    """The conversation as ingested, straight out of turn_text."""
    cold = ColdStorage(str(store / "cold.db"))
    try:
        texts = cold.text_for_turns(list(range(1, 5000)))
    finally:
        cold.close()
    return "\n".join(f"[turn {t}] {texts[t]}" for t in sorted(texts))


def parse(raw: str) -> list[dict]:
    """The updates list, from a reply that may be wrapped in a fenced block."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        # Some replies carry a sentence before the object. Take the outermost
        # braces rather than failing the whole conversation over a preamble.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    updates = data.get("updates", []) if isinstance(data, dict) else data
    return updates if isinstance(updates, list) else []


REQUIRED = ("question", "gold", "stale")


def usable(row: dict) -> bool:
    """Rows missing a field, or whose two answers agree, measure nothing."""
    if not all(str(row.get(f, "")).strip() for f in REQUIRED):
        return False
    return str(row["gold"]).strip().lower() != str(row["stale"]).strip().lower()


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--out", default="updates.json")
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    gate = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []

    async def one(sample_id: str) -> None:
        store = Path(args.run) / "stores" / f"{args.arm}__{sample_id}"
        if not (store / "cold.db").is_file():
            print(f"  [skip] no store for {sample_id}")
            return
        text = transcript(store)
        if not text:
            print(f"  [skip] no turn text for {sample_id}")
            return
        async with gate:
            reply = await llm.chat(
                messages=[{"role": "user", "content": text}], system_prompt=SYSTEM
            )
        found = [r for r in parse(reply) if usable(r)]
        for row in found:
            row["conv"] = sample_id
        rows.extend(found)
        print(f"  {sample_id:10s} {len(text):8d} chars -> {len(found):3d} updates")

    await asyncio.gather(*(one(c.sample_id) for c in load_locomo()))

    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  {len(rows)} update questions -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
