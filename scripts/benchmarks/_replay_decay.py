"""What the stored weights would be under a different consolidation policy.

The activation-gated hydration idea needs weight to vary. It does not: 97.8% of
the 3,807 records in the ten LoCoMo stores sit at exactly 1.0, ActiveContext
included. The cause is not the CoreAnchor mislabelling but the consolidation
coefficient. `compute_lambda` scales the base rate by alpha = e^{-eta*k}, and k
runs to a median of 63 and a maximum of 1071 on conversations of ~200 turns,
because reinforcement fires for every hot-sourced record in every ranking and
ingestion retrieves too. At eta 0.15 a k of 63 divides the decay rate by 12,700,
which puts ActiveContext's half-life at 176,000 turns.

This asks whether any cheap change to that policy produces a usable spread,
before anyone pays 5.3 hours of re-ingestion to find out. Nothing is written.

The arithmetic is deliberately the one-shot form, w = w0 * e^{-lambda(k) * age},
not a turn-by-turn replay: reconstructing the true path needs the per-turn k
history out of the event log, which is only worth doing for a policy that
survives this screen.

Two things follow from that, and the first row is the check on both. It prints
81.7% at 1.0 where the store itself holds 97.8%, and the gap is a knife edge
rather than a disagreement: the median ActiveContext record decays to 0.9996
over its lifetime under either arithmetic, which is a hair under the 0.9999 the
column tests for. Nothing here can separate a weight of 0.9996 from one of 1.0,
and neither could a gate, which is the finding rather than an artefact of it.

The surviving records are also a biased sample -- anything that fell past
w_prune was evicted during ingestion and is not on disk to be replayed -- so the
pruned column reads as "would be lost in a fresh run under this policy", not as
a count of anything currently missing.

Read-only. Opens each hot store, computes, prints, closes.

Usage:
    python scripts/benchmarks/_replay_decay.py
    python scripts/benchmarks/_replay_decay.py --run scripts/benchmarks/results/full_v3
"""

from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lancedb  # noqa: E402

# The shipped defaults, restated rather than imported: this script is asking what
# happens when they change, so reading them from DecayConfig would make the
# baseline column move whenever someone retunes the framework and silently
# invalidate the comparison.
ETA = 0.15
LAMBDA_BASE = {"CoreAnchor": 0.0, "ActiveContext": 0.05, "EphemeralState": 0.69}
W_PRUNE = 0.1


class Policy:
    """One consolidation rule, as a name and a lambda-from-(type, k)."""

    def __init__(self, name: str, eta: float | None = ETA,
                 k_cap: int | None = None, demote_anchors: bool = False) -> None:
        self.name = name
        self.eta = eta
        self.k_cap = k_cap
        # Treating CoreAnchor as ActiveContext, to see the two known defects
        # together. 82% of anchors are misclassified (measured 2026-08-17), and
        # fixing the extractor moves them into exactly this class, so the
        # combined row is the one that says whether that fix would be worth it.
        self.demote_anchors = demote_anchors

    def rate(self, memory_type: str, k: int) -> float:
        if self.demote_anchors and memory_type == "CoreAnchor":
            memory_type = "ActiveContext"
        base = LAMBDA_BASE.get(memory_type, 0.0)
        if base == 0.0:
            return 0.0
        if self.k_cap is not None:
            k = min(k, self.k_cap)
        return base * math.exp(-self.eta * k)


POLICIES = [
    Policy("current (eta 0.15)"),
    Policy("k capped at 20", k_cap=20),
    Policy("k capped at 10", k_cap=10),
    Policy("k capped at 5", k_cap=5),
    Policy("eta 0.02", eta=0.02),
    Policy("no consolidation", eta=0.0),
    Policy("k cap 10 + anchors demoted", k_cap=10, demote_anchors=True),
    Policy("k cap 20 + anchors demoted", k_cap=20, demote_anchors=True),
]


def load(run: Path, arm: str) -> list[tuple[str, int, int, int]]:
    """Every record as (memory_type, k, created_at_turn, store_last_turn)."""
    rows: list[tuple[str, int, int, int]] = []
    for store in sorted(glob(str(run / "stores" / f"{arm}__*"))):
        uri = os.path.join(store, "hot_lancedb")
        if not os.path.isdir(uri):
            continue
        db = lancedb.connect(uri)
        for table in db.table_names():
            df = db.open_table(table).to_pandas()
            if df.empty:
                continue
            # Questions are asked after the last turn, so that turn is "now"
            # and the age of a record is how long it sat before being queried.
            last = int(df["created_at_turn"].max())
            rows.extend(
                (str(mt), int(k), int(turn), last)
                for mt, k, turn in zip(
                    df["memory_type"], df["consolidation_index"],
                    df["created_at_turn"],
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    args = ap.parse_args()

    rows = load(Path(args.run), args.arm)
    if not rows:
        print("no records found")
        return
    print(f"{len(rows)} records across {args.arm}\n")

    header = (f"{'policy':28s} {'==1.0':>7s} {'>0.9':>7s} {'0.1-0.9':>8s} "
              f"{'<prune':>7s} {'median':>8s} {'spread':>7s}")
    print(header)
    print("-" * len(header))

    for policy in POLICIES:
        weights = [
            math.exp(-policy.rate(mt, k) * max(0, last - turn))
            for mt, k, turn, last in rows
        ]
        n = len(weights)
        at_one = sum(1 for w in weights if w >= 0.9999)
        high = sum(1 for w in weights if w > 0.9)
        # The band that a threshold could actually cut through. A gate needs
        # records on both sides of it, and only these are on the near side
        # without having been pruned out of the store entirely.
        middle = sum(1 for w in weights if W_PRUNE <= w <= 0.9)
        pruned = sum(1 for w in weights if w < W_PRUNE)
        print(f"{policy.name:28s} {at_one/n:6.1%} {high/n:6.1%} {middle/n:7.1%} "
              f"{pruned/n:6.1%} {st.median(weights):8.4f} {middle/n:6.1%}")

    print("\n  ==1.0   nothing to gate on")
    print("  0.1-0.9 the usable band: a threshold can separate these")
    print(f"  <prune  would have been evicted at w_prune {W_PRUNE}, so lost")


if __name__ == "__main__":
    main()
