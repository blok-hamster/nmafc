"""Give the per-conversation LoCoMo stores the turn text they were ingested from.

The ten stores under `full_v3` were built before hydration existed, so they hold
facts and turn dates and nothing else. The shipping stack prints six source
turns next to twenty facts; on a store with an empty `turn_text` there is
nothing to print, so a run there would silently measure the facts-only
configuration and report it as the current one. That is the whole reason this
file exists.

Re-ingesting would fix it and costs 5.3 hours of LLM extraction. Nothing needs
to be re-extracted: `wrapper.process_turn` writes turn N's text as the Nth
`user_msg` it was handed, and the benchmark hands it
`build_dated_exchanges(turns)[N-1]`, which is a pure function of the transcript.
So the text is recoverable from the dataset alone, for free.

"Recoverable" is a claim, not a fact, and it is checked rather than trusted.
Every store already holds `turn_timestamps`, written by the same loop from the
same exchange list at the same turn number. If the dates rebuilt from the
dataset match the dates in the store, turn for turn, then the numbering the
facts' `source_turns` refer to is the numbering being written here. Any
mismatch aborts that store and leaves it exactly as it was, because a turn_text
table that is off by one is worse than an empty one: it would hydrate confident
and wrong.

Turn vectors are a separate pass. Run `_build_turn_index.py --store <path>`
afterwards to buy those, which is embeddings only.

Usage:
    python -u scripts/benchmarks/_backfill_turn_text.py --check
    python -u scripts/benchmarks/_backfill_turn_text.py
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


def verify(cold: ColdStorage,
           exchanges: list[tuple[str, str | None]]) -> tuple[bool, str]:
    """Does the rebuilt exchange list line up with the dates already stored?"""
    stored = cold.turn_timestamps()
    if not stored:
        return False, "no turn_timestamps to check the numbering against"
    rebuilt = {index + 1: date for index, (_, date) in enumerate(exchanges)
               if date is not None}
    if len(rebuilt) != len(stored):
        return False, (f"{len(rebuilt)} dated turns rebuilt, {len(stored)} "
                       f"stored")
    wrong = [turn for turn, date in stored.items() if rebuilt.get(turn) != date]
    if wrong:
        return False, (f"{len(wrong)} dates disagree, first at turn "
                       f"{min(wrong)}: stored {stored[min(wrong)]!r}, rebuilt "
                       f"{rebuilt.get(min(wrong))!r}")
    return True, f"{len(stored)} dated turns match"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--check", action="store_true",
                    help="Report what would happen and write nothing")
    args = ap.parse_args()

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        db = store / "cold.db"
        if not db.is_file():
            print(f"[{conv.sample_id}] no store")
            continue

        exchanges = build_dated_exchanges(conv.get_flat_history())
        # Through the store's own class, not hand-rolled SQL: `turn_text` is
        # keyed on (agent, conversation, turn) and `record_turn_text` upserts on
        # exactly that. A second table created here with a different key would
        # work today and break the next real ingestion.
        cold = ColdStorage(str(db))
        try:
            already = len(cold.all_turn_text())
            ok, why = verify(cold, exchanges)
            state = f"{len(exchanges)} exchanges, {already} already stored"
            if not ok:
                print(f"[{conv.sample_id}] SKIP -- {why} ({state})")
                continue
            if already:
                print(f"[{conv.sample_id}] already has turn text ({state})")
                continue
            if args.check:
                print(f"[{conv.sample_id}] would write {len(exchanges)} turns "
                      f"-- {why}")
                continue
            for index, (text, _) in enumerate(exchanges):
                cold.record_turn_text(index + 1, text)
            print(f"[{conv.sample_id}] wrote {len(exchanges)} turns -- {why}")
        finally:
            cold.close()


if __name__ == "__main__":
    main()
