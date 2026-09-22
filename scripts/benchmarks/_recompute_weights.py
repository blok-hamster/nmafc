"""Recompute decay weights from when a fact was learned, not from when the
benchmark last happened to read it.

`_sweep_decay_signal.py` came back perfectly flat: every value of
`weight_signal` from 0.0 to 0.4 produced identical retrieval. The reason is not
that the setting is unwired -- `reranking.py` reads it and adds
`weight_signal * record.weight` to the entity score. The reason is that
290 of 294 stored weights are exactly 1.0. Adding a constant to every score
reorders nothing, so the sweep was measuring a no-op.

Two things put every weight at 1.0, and they stack:

  `decay_record` returns early for CoreAnchor, unconditionally. That is 237 of
  294 records, 80% of Hot RAM, which can never fade whatever the settings say.

  For the rest, `delta_t = current_turn - last_reinforced_turn`, and it returns
  early when that is at or below zero. The median `last_reinforced_turn` in
  conv-26 is 212 against a final turn of 211, so the median fact reads as
  reinforced one turn *after* the conversation ended. Retrieval reinforces what
  it touches, and ingestion retrieves on every turn, so by the end the
  benchmark's own reading had refreshed nearly everything. The stored figure
  measures the benchmark, not the framework.

So this recomputes each weight from `created_at_turn`, which no read ever
touches, using the tier's own rate. That answers the counterfactual the flat
sweep could not: *if decay had not been erased by the benchmark reading the
store, would its score improve retrieval on a test where forgetting pays?*

`--core-lambda` optionally gives CoreAnchor a non-zero rate. Nothing in the
framework does this, and it is not proposed as a change to it. Without it 80%
of records stay pinned at 1.0 and the counterfactual cannot be asked at all.

Copies before writing. The source run cost 5.3 hours and is not reproducible
from anything cheaper.

Usage:
    python -u scripts/benchmarks/_recompute_weights.py --dry-run
    python -u scripts/benchmarks/_recompute_weights.py --dst C:/nmafc_ab/decayed --core-lambda 0.002
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lancedb  # noqa: E402

from nmafc.engine.decay import compute_alpha, compute_weight  # noqa: E402
from nmafc.schemas.memory import DecayConfig, MemoryType  # noqa: E402

TIERS = {
    "CoreAnchor": MemoryType.CORE_ANCHOR,
    "ActiveContext": MemoryType.ACTIVE_CONTEXT,
    "EphemeralState": MemoryType.EPHEMERAL_STATE,
}


def rate(tier: str, k: int, config: DecayConfig, core_lambda: float) -> float:
    """The fade rate for one record, with CoreAnchor optionally unpinned.

    `compute_lambda` is reimplemented rather than called only so the CoreAnchor
    base can be substituted; the alpha term is the framework's own, so a
    well-consolidated fact still fades more slowly than a fresh one.
    """
    member = TIERS.get(tier)
    if member is None:
        return 0.0
    base = core_lambda if member is MemoryType.CORE_ANCHOR \
        else config.get_lambda_base(member)
    # clustering is 0 here: the graph protection term needs neighbours the
    # offline pass does not load, and leaving it out only makes facts fade
    # slightly faster than the live engine would fade them.
    return base * compute_alpha(int(k or 0), config.eta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--dst", default="C:/nmafc_ab/decayed")
    ap.add_argument("--core-lambda", type=float, default=0.0,
                    help="Fade rate for CoreAnchor. 0.0 leaves 80%% pinned at 1.0.")
    ap.add_argument("--reset-k", action="store_true",
                    help="Treat every fact as never reinforced. The stored "
                         "consolidation_index has a median of 62 and a maximum "
                         "of 1401, which drives alpha to about 0.0001 and "
                         "multiplies every fade rate down to nothing. Those "
                         "counts came from the benchmark reading the store, "
                         "not from anyone repeating themselves, so k=0 is the "
                         "other bound on the same counterfactual.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = DecayConfig()
    src = Path(args.src) / "stores"
    dst = Path(args.dst)
    stores = sorted(p for p in src.iterdir() if p.is_dir())

    print(f"lambda: core={args.core_lambda} "
          f"active={config.get_lambda_base(MemoryType.ACTIVE_CONTEXT)} "
          f"ephemeral={config.get_lambda_base(MemoryType.EPHEMERAL_STATE)}\n")

    for store in stores:
        table_dir = store / "hot_lancedb"
        if not table_dir.is_dir():
            continue

        # `dst/stores/<name>`, matching the layout every other harness expects
        # from a run directory, so the output is a drop-in `--run`.
        target = dst / "stores" / store.name
        if not args.dry_run:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(store, target)
            table_dir = target / "hot_lancedb"

        db = lancedb.connect(str(table_dir))
        table = db.open_table("memory_vectors")
        frame = table.to_pandas()

        # Evaluated at the last turn of the conversation, which is when every
        # question in the benchmark is asked.
        now = int(frame["created_at_turn"].max())
        weights = [
            compute_weight(1.0, rate(row.memory_type,
                                     0 if args.reset_k else row.consolidation_index,
                                     config, args.core_lambda),
                           max(0, now - int(row.created_at_turn)))
            for row in frame.itertuples()
        ]
        before = frame["weight"]
        frame["weight"] = weights

        pinned = sum(1 for w in weights if w > 0.999)
        print(f"[{store.name}] {len(frame):5d} facts, final turn {now:4d}   "
              f"mean {before.mean():.3f} -> {frame['weight'].mean():.3f}   "
              f"still at 1.0: {pinned} ({100 * pinned / len(frame):.0f}%)   "
              f"distinct {frame['weight'].round(3).nunique()}")

        if not args.dry_run:
            db.create_table("memory_vectors", data=frame, mode="overwrite")

    if args.dry_run:
        print("\ndry run, nothing written")
    else:
        print(f"\nwritten to {dst}, source untouched")
        print("cold.db copied unchanged: it carries no weight column, and the "
              "reranker reads weight off the hot record")


if __name__ == "__main__":
    main()
