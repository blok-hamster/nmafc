"""Give the stores the weight spread a working decay would have produced.

`weight_signal` cannot be tested on the stores as they are. 97.8% of records sit
at weight exactly 1.0, so a boost proportional to weight is the same boost for
almost every record and changes no ranking. That flatness is not what decay
would have produced either: it is an artefact of the harness, which reinforces
on every read and reopens stores with the turn clock at zero, leaving 3,477 of
3,807 records claiming a reinforcement turn earlier than their creation turn.

So the stored weight is discarded and recomputed from the one field the harness
never touched, `created_at_turn`:

    weight = exp(-lambda_base(type) * (last_turn - created_at_turn))

This is the no-reinforcement case: every fact decays from the moment it was
written and nothing ever refreshes it. That is deliberately the widest spread
available -- real recall would flatten it -- and it is the right first test,
because a graded weight that cannot help ranking at maximum spread will not help
at less. At the tuned rate it puts 30.3% of records between 0.1 and 1.0 with
ActiveContext median 0.517, against 68.7% pinned at 1.0 by CoreAnchor's
lambda of zero.

Two things it does NOT simulate, both of which matter when reading the result:
pruning (a real decay run would have removed the low-weight records during
ingestion, so some of what is graded here would not exist), and consolidation
(promotion to CoreAnchor happened under the old weights).

Writes only to `--dest`. The source stores are copied, never opened for writing.

Usage:
    python scripts/benchmarks/_inject_weights.py --dest /c/nmafc_ab/w_stores
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from glob import glob
from pathlib import Path

import lancedb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# The shipping arm's rates. lambda_active_context is the tuned 0.005 rather than
# the 0.05 default: at 0.05 a fact a hundred turns old lands at 0.0067 and 24.8%
# of the store falls below the prune floor, which grades nothing and merely
# empties hot.
LAMBDA = {"CoreAnchor": 0.0, "ActiveContext": 0.005, "EphemeralState": 0.69}


def inject(store: Path) -> tuple[int, float]:
    """Rewrite every weight in one store. Returns the count and the median."""
    db = lancedb.connect(str(store / "hot_lancedb"))
    table = db.open_table(db.table_names()[0])
    frame = table.to_pandas()
    if frame.empty:
        return 0, 1.0

    # The clock the harness left intact. last_reinforced_turn is unusable here
    # and current_turn is not persisted, so the newest record's creation turn
    # stands in for "now" -- correct to within one turn, since the last thing
    # ingested was written at the final turn.
    last_turn = int(frame["created_at_turn"].max())

    rows = table.search().limit(len(frame) + 1000).to_list()
    for row in rows:
        row.pop("_distance", None)
        age = max(0, last_turn - int(row["created_at_turn"]))
        rate = LAMBDA.get(row["memory_type"], 0.0)
        row["weight"] = math.exp(-rate * age)

    table.delete("id IS NOT NULL")
    table.add(rows)

    graded = sorted(r["weight"] for r in rows)
    return len(rows), graded[len(graded) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--recopy", action="store_true",
                    help="delete and re-copy stores that are already present")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    sources = sorted(glob(str(Path(args.run) / "stores" / f"{args.arm}__*")))
    if not sources:
        raise SystemExit(f"no stores matching {args.arm}__* under {args.run}")

    for source in sources:
        name = Path(source).name
        target = dest / name
        if target.exists() and args.recopy:
            shutil.rmtree(target)
        if not target.exists():
            print(f"  copying {name} ...", flush=True)
            shutil.copytree(source, target)
        n, median = inject(target)
        print(f"  {name:34s} {n:5d} records, median weight {median:.3f}")

    print(f"\n  -> {dest}")


if __name__ == "__main__":
    main()
