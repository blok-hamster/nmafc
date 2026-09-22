"""Give an already-built store the dates of the turns it ingested.

Facts were reaching the model as "(Valid: turn 220 - present)". The wrapper now
records when each turn happened and the router renders that as a date, but
stores built before it existed have an empty turn_timestamps table, so they
still fall back to turn numbers.

Nothing has to be re-extracted to fix that. The date of a turn is a property of
the transcript, not of anything the model produced: `build_dated_exchanges` is a
pure function of the conversation, so exchange N always carries the same session
date, and the wrapper assigns turn N+1 to exchange N. The mapping is therefore
recoverable exactly, for free, with no LLM calls.

The one thing this cannot recover is the per-fact date the extractor read out of
the conversation ("I started last January"). That string was discarded at write
time and is in neither store, so facts dated earlier than the turn that
mentioned them stay dated by their turn until the store is rebuilt. Turn dates
are the floor this restores, not the ceiling.

Writes only the turn_timestamps table. No fact, link, weight or vector is
touched, so a backfilled store is the same store with dates added.

Usage:
    python scripts/benchmarks/_backfill_turn_dates.py --run scripts/benchmarks/results/full_v3
    python scripts/benchmarks/_backfill_turn_dates.py --run ... --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nmafc.storage.cold import ColdStorage  # noqa: E402

from scripts.benchmarks.arms.base import build_dated_exchanges  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory holding stores/")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--write", action="store_true",
                    help="apply the backfill (default: report only)")
    args = ap.parse_args()

    stores = Path(args.run) / "stores"
    print(f"run  : {args.run}")
    print(f"mode : {'WRITE' if args.write else 'dry run, nothing written'}\n")
    print(f"  {'store':16s} {'turns':>7s} {'dated':>7s} {'already':>8s}  span")

    total_turns = total_dated = 0
    for conv in load_locomo():
        store = stores / f"{args.arm}__{conv.sample_id}"
        cold_db = store / "cold.db"
        if not cold_db.is_file():
            print(f"  {conv.sample_id:16s} no store, skipped")
            continue

        # Turn numbering: the wrapper increments before processing, so exchange
        # index 0 was ingested as turn 1.
        dated = [
            (index + 1, date)
            for index, (_, date) in enumerate(
                build_dated_exchanges(conv.get_flat_history())
            )
            if date
        ]
        turns = len(build_dated_exchanges(conv.get_flat_history()))

        cold = ColdStorage(str(cold_db))
        try:
            already = len(cold.turn_timestamps())
            span = ""
            if dated:
                span = f"{dated[0][1]}  ..  {dated[-1][1]}"
            print(f"  {conv.sample_id:16s} {turns:7d} {len(dated):7d} "
                  f"{already:8d}  {span}")
            if args.write:
                for turn, date in dated:
                    cold.record_turn_timestamp(turn, date)
        finally:
            cold.close()

        total_turns += turns
        total_dated += len(dated)

    print(f"\n  {'ALL':16s} {total_turns:7d} {total_dated:7d}")
    if total_turns:
        print(f"  turns carrying a date: {total_dated / total_turns:.1%}")
    if not args.write:
        print("\n  re-run with --write to apply")


if __name__ == "__main__":
    main()
