"""Mark the superseded facts invalid, on a copy of the stores.

`_detect_supersessions.py` finds pairs where a later fact makes an earlier one
untrue. This writes those verdicts in, setting `invalid_at` on the loser to the
turn the winner was stated -- the turn the old fact stopped being true.

Both tiers are marked, and both have to be. `exclude_invalidated` defaults to
True, so the record drops out of hot vector search and entity lookup -- but the
router queries the archive on every retrieval, and marking hot alone left the
stale value reaching the prompt 64 times out of 101, one MORE than doing
nothing. The row itself is never deleted: it keeps its place in the append-only
log with `invalid_at` set, so the audit trail and any as-of-turn query still
see it. It is out of the running, not destroyed.

The copy is the point. This is the first thing in the project that removes a
fact from normal retrieval, and the detector's precision is imperfect --
inspection of its output found it occasionally reads the extractor's own
inconsistent date resolutions as a change of plan. Running it against the real
stores would bake that into every later experiment.

Output is laid out as <dest>/stores/<arm>__<conv> so the existing scripts can
read it with --run <dest> and no other change.

Usage:
    python scripts/benchmarks/_apply_invalidation.py \
        --verdicts /c/nmafc_ab/superseded.json --dest /c/nmafc_ab/inv_run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

import lancedb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nmafc.storage.cold import ColdStorage  # noqa: E402
from nmafc.storage.config import StorageConfig  # noqa: E402
from nmafc.storage.hot import HotStorage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--recopy", action="store_true")
    args = ap.parse_args()

    verdicts = json.loads(Path(args.verdicts).read_text(encoding="utf-8"))

    # A fact can lose to several later facts. The earliest winner is the turn it
    # actually stopped being true, so that is the one recorded.
    by_conv: dict[str, dict[str, int]] = defaultdict(dict)
    for v in verdicts:
        held = by_conv[v["conv"]].get(v["earlier_id"])
        turn = int(v["later_turn"])
        by_conv[v["conv"]][v["earlier_id"]] = min(held, turn) if held else turn

    dest = Path(args.dest) / "stores"
    dest.mkdir(parents=True, exist_ok=True)

    total = 0
    for source in sorted(glob(str(Path(args.run) / "stores" / f"{args.arm}__*"))):
        name = Path(source).name
        conv = name.split("__")[-1]
        target = dest / name
        if target.exists() and args.recopy:
            shutil.rmtree(target)
        if not target.exists():
            print(f"  copying {name} ...", flush=True)
            shutil.copytree(source, target)

        marks = by_conv.get(conv, {})
        if not marks:
            print(f"  {conv:8s} nothing to invalidate")
            continue

        hot = HotStorage(StorageConfig(
            hot_uri=str(target / "hot_lancedb"),
            cold_uri=str(target / "cold.db"),
        ))
        try:
            hot.set_invalid_at_many(list(marks.items()))
        finally:
            close = getattr(hot, "close", None)
            if close:
                close()

        # Read back rather than trust the write: set_invalid_at_many rebuilds
        # rows through a delete + add, and a silent no-op here would leave the
        # experiment comparing two identical stores.
        db = lancedb.connect(str(target / "hot_lancedb"))
        frame = db.open_table(db.table_names()[0]).to_pandas()
        marked = int(frame["invalid_at"].notna().sum())
        total += marked

        # And the archive, which is the half that decides what reaches the
        # prompt. The router queries Cold ROM on every retrieval, so a fact
        # withdrawn from Hot RAM alone comes straight back through keyword or
        # vector search. Keyed on (entity_name, fact_content) because the
        # archive autoincrements its own primary key and holds no UUID, so the
        # pairing has to be recovered from the Hot frame.
        naming = frame.set_index("id")
        cold_marks = [
            (str(naming.at[record_id, "entity_name"]),
             str(naming.at[record_id, "fact_content"]), turn)
            for record_id, turn in marks.items()
            if record_id in naming.index
        ]
        # The tenant keys are read off the rows rather than assumed. Every
        # store the benchmark wrote uses ('default', 'default'), and passing
        # the conversation name instead would match nothing and report a
        # confident zero.
        cold_path = target / "cold.db"
        with sqlite3.connect(cold_path) as probe:
            tenants = probe.execute(
                "SELECT DISTINCT agent_id, conversation_id FROM memory_event_log"
            ).fetchall()
        if len(tenants) != 1:
            raise SystemExit(f"{conv}: expected one tenant, found {tenants}")
        cold = ColdStorage(str(cold_path),
                           agent_id=tenants[0][0], conversation_id=tenants[0][1])
        try:
            cold_marked = cold.invalidate_facts(cold_marks)
        finally:
            cold.close()

        print(f"  {conv:8s} invalidated {marked:4d} of {len(marks)} requested"
              f", {len(frame)} facts total"
              f"  |  cold rows {cold_marked:4d}")

    print(f"\n  {total} facts invalidated\n  -> {Path(args.dest)}")


if __name__ == "__main__":
    main()
