"""Put the conversation text back into finished stores.

Hydration needs the wording of the turn a fact came from, and the stores built
in August were ingested before `turn_text` existed, so they hold facts whose
sources are unrecoverable from the store itself. They are recoverable from the
dataset, though: LoCoMo is on disk, exchanges are deterministic, and the turn
numbering is fixed, so the table can be filled without a single LLM call.

That is the whole reason to prefer an evidence layer keyed by turn over adding
a verbatim quote to each fact. A quote would have to come from the extractor
and would mean re-ingesting all ten conversations, which is 5.3 hours and puts
8,276 existing facts at risk. This is a few seconds and touches nothing else.

Turn numbering: the wrapper increments before processing, so exchange index 0
was ingested as turn 1. Same arithmetic as the date backfill, and wrong by one
would hydrate a neighbouring turn rather than fail loudly, so it is asserted
against the dates already in the store where those exist.

Dry run by default; pass --write to make changes.

Usage:
    python scripts/benchmarks/_backfill_turn_text.py
    python scripts/benchmarks/_backfill_turn_text.py --write
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
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    print(f"{'conversation':16s} {'turns':>7s} {'already':>8s} {'chars':>10s}  dates")
    total_turns = total_chars = 0

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        cold_db = store / "cold.db"
        if not cold_db.is_file():
            print(f"  {conv.sample_id:16s} no store")
            continue

        exchanges = build_dated_exchanges(conv.get_flat_history())
        chars = sum(len(text) for text, _ in exchanges)

        cold = ColdStorage(str(cold_db))
        try:
            already = len(cold.text_for_turns(list(range(1, len(exchanges) + 1))))

            # The dates went in under the same numbering. Where a turn is dated,
            # the exchange at that index must carry the same date in its header,
            # or the two backfills disagree and one of them is off by one.
            dates = cold.turn_timestamps()
            mismatched = sum(
                1 for turn, date in dates.items()
                if 1 <= turn <= len(exchanges) and exchanges[turn - 1][1] != date
            )
            flag = "OK" if not mismatched else f"{mismatched} MISMATCHED"

            print(f"  {conv.sample_id:16s} {len(exchanges):7d} {already:8d} "
                  f"{chars:10d}  {flag}")

            if args.write and not mismatched:
                for index, (text, _) in enumerate(exchanges):
                    cold.record_turn_text(index + 1, text)
        finally:
            cold.close()

        total_turns += len(exchanges)
        total_chars += chars

    print(f"\n  {'total':16s} {total_turns:7d} {'':8s} {total_chars:10d}")
    if not args.write:
        print("\n  dry run, nothing written. Pass --write to fill the table.")


if __name__ == "__main__":
    main()
