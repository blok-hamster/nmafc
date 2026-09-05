"""Find the pairs of stored facts where the later one makes the earlier untrue.

`detect_override` has fired zero times across 8,276 facts, because it matches on
identical `entity_name` and the extractor names entities per event --
`jon_job_loss_jan_2023` and `jon_new_shop_july_2023` are the same subject under
two names, so they never collide. Nothing is ever invalidated, and a superseded
fact competes for a prompt slot on equal terms with the fact that replaced it
forever after.

The fix is to match on what a fact is about rather than on what it is called,
and the hot store already carries the means: every record has its embedding. So
candidates are pairs that are close in vector space and separated in time, and
the model only ever adjudicates that shortlist. At a cosine of 0.7 that is 7,073
pairs across the ten stores rather than the 800,000 a full pairwise scan would
produce.

The prompt is deliberately hard to say yes to. Mining these conversations for
supersessions found that most same-subject pairs are restatements or
elaborations, and a detector that treats those as contradictions would delete
true facts -- which is worse than the current behaviour of deleting nothing.

Writes JSON only. Reads the stores read-only and changes no record; applying the
verdicts is `_apply_invalidation.py`.

Usage:
    python scripts/benchmarks/_detect_supersessions.py --out /c/nmafc_ab/superseded.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from glob import glob
from pathlib import Path

import lancedb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from nmafc.integration.factory import create_llm_provider  # noqa: E402

SYSTEM = """\
You are auditing a memory system. Each item below is a PAIR of facts about the
same conversation, the EARLIER one stated first, with the turn each was said on.

For each pair, decide whether the LATER fact makes the EARLIER fact NO LONGER
TRUE. You are looking for the earlier fact having been overtaken by events.

Answer "yes" ONLY when the two cannot both be true at the end of the
conversation. Examples of yes:
  earlier "Jon is unemployed"          later "Jon opened his own bakery"
  earlier "Mel plans to visit Spain"   later "Mel went to Portugal instead"
  earlier "Caroline lives in Leeds"    later "Caroline moved to Bristol"

Answer "no" for everything else, and most pairs are "no". In particular:
  - the same fact restated, elaborated, or summarised
  - two things that are both still true (two hobbies, two trips, two friends)
  - a plan followed by that plan being carried out (both remain true)
  - a step in a sequence where every step still happened
  - anything about different people, or where you cannot be sure

Read dates with particular care. These facts were written by an extractor that
resolves phrases like "two weekends ago" into calendar dates, and it resolves
the same event inconsistently on different turns. So two records describing ONE
event with different dates are a duplicate, NOT a supersession, and the answer
is "no". Only answer "yes" on a date when the thing itself was genuinely
rescheduled -- a plan for August that the speaker later says has been moved to
November. If the two could be the same occasion written up twice, answer "no".

If in doubt, answer "no". Wrongly marking a true fact as superseded destroys
information; missing one merely leaves things as they are.

Return STRICT JSON: an object with one key "verdicts", a list of objects:
  "i"          the pair's index number, exactly as given
  "superseded" true or false
  "why"        at most eight words, only when true
"""


def candidates(store: Path, threshold: float, cap: int) -> list[dict]:
    """Pairs that are close in meaning and separated in time, closest first."""
    db = lancedb.connect(str(store / "hot_lancedb"))
    frame = db.open_table(db.table_names()[0]).to_pandas()
    if len(frame) < 2:
        return []

    vectors = np.vstack(frame["vector"].values).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = vectors @ vectors.T
    turns = frame["created_at_turn"].values

    # Strictly later, so each unordered pair is considered once and the
    # direction of supersession is fixed by the clock rather than guessed.
    rows, cols = np.where((similarity > threshold) & (turns[None, :] > turns[:, None]))
    order = np.argsort(-similarity[rows, cols])[:cap]

    out = []
    for index in order:
        i, j = int(rows[index]), int(cols[index])
        out.append({
            "earlier_id": frame["id"].iloc[i],
            "later_id": frame["id"].iloc[j],
            "earlier": str(frame["fact_content"].iloc[i]),
            "later": str(frame["fact_content"].iloc[j]),
            "earlier_turn": int(turns[i]),
            "later_turn": int(turns[j]),
            "similarity": round(float(similarity[i, j]), 3),
        })
    return out


def parse(raw: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    verdicts = data.get("verdicts", []) if isinstance(data, dict) else data
    return verdicts if isinstance(verdicts, list) else []


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--cap", type=int, default=2000,
                    help="most candidate pairs to adjudicate per store")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default="superseded.json")
    args = ap.parse_args()

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    gate = asyncio.Semaphore(args.concurrency)
    found: list[dict] = []

    for store in sorted(glob(str(Path(args.run) / "stores" / f"{args.arm}__*"))):
        conv = Path(store).name.split("__")[-1]
        pairs = candidates(Path(store), args.threshold, args.cap)
        if not pairs:
            print(f"  {conv:8s} no candidates")
            continue

        batches = [pairs[i:i + args.batch] for i in range(0, len(pairs), args.batch)]
        hits: list[dict] = []

        async def one(batch: list[dict], offset: int) -> None:
            listing = "\n\n".join(
                f'[{offset + n}] EARLIER (turn {p["earlier_turn"]}): {p["earlier"]}\n'
                f'     LATER (turn {p["later_turn"]}): {p["later"]}'
                for n, p in enumerate(batch)
            )
            async with gate:
                reply = await llm.chat(
                    messages=[{"role": "user", "content": listing}],
                    system_prompt=SYSTEM,
                )
            for verdict in parse(reply):
                if not verdict.get("superseded"):
                    continue
                index = verdict.get("i")
                if not isinstance(index, int) or not (
                    offset <= index < offset + len(batch)
                ):
                    continue
                hits.append({
                    **batch[index - offset],
                    "conv": conv,
                    "why": str(verdict.get("why", ""))[:80],
                })

        await asyncio.gather(*(
            one(b, i * args.batch) for i, b in enumerate(batches)
        ))
        found.extend(hits)
        print(f"  {conv:8s} {len(pairs):5d} pairs -> {len(hits):4d} superseded"
              f"  ({len(hits)/len(pairs):.1%})", flush=True)

    Path(args.out).write_text(json.dumps(found, indent=2), encoding="utf-8")
    losers = {h["earlier_id"] for h in found}
    print(f"\n  {len(found)} supersessions over {len(losers)} distinct facts"
          f"\n  -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
