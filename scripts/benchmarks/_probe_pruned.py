"""When an answer is wrong, was the evidence deleted, or just not retrieved?

The hot store holds 3,807 records; the append-only cold log holds the 8,276 that
were ever written. The 4,469 in the gap were pruned during ingestion, and 97.8%
of them are EphemeralState -- the one class whose half-life is short enough to
cross the prune threshold. So the framework does forget, heavily, but only by
deletion: a fact is either wiped within a turn or two or it is immortal.

That makes over-forgetting a live hypothesis again, and a different one from the
staleness hypothesis `_test_updates.py` tested and largely refuted. The question
is not whether outdated facts linger, it is whether needed ones were destroyed.

Every wrong answer is sorted by where its gold answer can still be found:

  in a surviving fact      retrieval or ranking failed; deletion is blameless
  only in a deleted fact   forgetting cost this question, and a fade instead of
                           a delete would have kept it reachable
  in neither               extraction never captured it; decay is blameless

Correct answers are scored the same way as a control. If deleted facts turn up
just as often behind right answers as behind wrong ones, the deletions were not
load-bearing and the number means nothing.

Presence is strict whole-word containment of every content word, the same test
`_test_updates.py` uses. Loose keyword overlap reports a gold of "7 May 2023" as
present in a fact reading "6 May 2023", which is exactly the error that would
manufacture a finding here.

No API calls, no writes. Reads the stores read-only.

Usage:
    python scripts/benchmarks/_probe_pruned.py --answers /c/nmafc_ab/ab_hydrate.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lancedb  # noqa: E402

STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "his", "her", "their",
        "is", "was", "and", "for", "with", "he", "she", "they", "it", "that"}


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", str(text).lower()) if w not in STOP]


def present_in(gold: str, haystack: str) -> bool:
    words = content_words(gold)
    if not words:
        return False
    return all(
        re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", haystack)
        for w in words
    )


def split_store(store: str) -> tuple[str, str, Counter]:
    """One blob of surviving fact text, one of deleted, and the deleted types.

    Matched on (entity_name, fact_content) rather than on id: the cold log
    autoincrements its own primary key and does not carry the record UUID, so
    the text is the only join available. Duplicates collapse, which is the safe
    direction -- a fact written twice and pruned once counts as surviving.
    """
    conn = sqlite3.connect(f"file:{os.path.join(store, 'cold.db')}?mode=ro", uri=True)
    cold = conn.execute(
        "SELECT entity_name, fact_content, memory_type FROM memory_event_log"
    ).fetchall()
    conn.close()

    db = lancedb.connect(os.path.join(store, "hot_lancedb"))
    frame = db.open_table(db.table_names()[0]).to_pandas()
    alive = {
        (str(e), str(f))
        for e, f in zip(frame["entity_name"], frame["fact_content"])
    }

    surviving: list[str] = []
    deleted: list[str] = []
    types: Counter = Counter()
    for entity, fact, memory_type in cold:
        if (str(entity), str(fact)) in alive:
            surviving.append(str(fact))
        else:
            deleted.append(str(fact))
            types[memory_type] += 1
    return "\n".join(surviving).lower(), "\n".join(deleted).lower(), types


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--field", default="b_ok",
                    help="which arm's verdict to read (b_ok is the shipping one)")
    args = ap.parse_args()

    rows = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_conv[row["conv"]].append(row)

    tally: dict[bool, Counter] = {True: Counter(), False: Counter()}
    lost_types: Counter = Counter()
    examples: list[tuple[str, str]] = []

    for store in sorted(glob(str(Path(args.run) / "stores" / f"{args.arm}__*"))):
        conv = os.path.basename(store).split("__")[-1]
        if conv not in by_conv:
            continue
        surviving, deleted, types = split_store(store)
        print(f"  {conv:10s} kept {len(surviving):8d} chars, "
              f"deleted {len(deleted):8d} chars, {len(by_conv[conv]):4d} questions")

        for row in by_conv[conv]:
            gold = row.get("gold", "")
            in_alive = present_in(gold, surviving)
            in_dead = present_in(gold, deleted)
            if in_alive:
                bucket = "in a surviving fact"
            elif in_dead:
                bucket = "ONLY in a deleted fact"
            else:
                bucket = "in neither"
            correct = bool(row.get(args.field))
            tally[correct][bucket] += 1
            if bucket == "ONLY in a deleted fact":
                lost_types.update(types)
                if not correct and len(examples) < 8:
                    examples.append((row["question"], str(gold)))

    order = ["in a surviving fact", "ONLY in a deleted fact", "in neither"]
    wrong_n = sum(tally[False].values()) or 1
    right_n = sum(tally[True].values()) or 1
    print(f"\n{'where the gold answer still exists':26s} "
          f"{'wrong':>16s} {'correct (control)':>19s}")
    for bucket in order:
        w, r = tally[False][bucket], tally[True][bucket]
        print(f"{bucket:26s} {w:6d} {w/wrong_n:8.1%} {r:9d} {r/right_n:9.1%}")
    print(f"{'total':26s} {wrong_n:6d} {'':8s} {right_n:9d}")

    lost = tally[False]["ONLY in a deleted fact"]
    print(f"\n  {lost} wrong answers ({lost/wrong_n:.1%} of errors) need a fact "
          f"that was deleted.")
    print("  Compare against the control column: deletion only explains the gap "
          "between them.")
    if examples:
        print("\n  examples:")
        for question, gold in examples:
            print(f"    {question[:82]}\n        gold: {gold[:70]}")


if __name__ == "__main__":
    main()
