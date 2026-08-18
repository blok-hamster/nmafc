"""Micro-benchmark + equivalence check for the batched decay write path.

Compares the old per-record update_weight() loop against the new
apply_weight_updates() batch on identical data, asserting the resulting
(id -> weight) state matches exactly before reporting any speedup.

    python -u scripts/benchmarks/_bench_decay.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

os.environ.setdefault("NMAFC_EMBEDDING_DIM", "8")

from nmafc.schemas.memory import MemoryRecord, MemoryType
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage

DIM = 8
TURNS = 12
NEW_PER_TURN = 3


def fresh_store(tag: str) -> HotStorage:
    tmp = tempfile.mkdtemp(prefix=f"nmafc_decaybench_{tag}_")
    return HotStorage(StorageConfig(
        hot_uri=str(Path(tmp) / "hot"),
        cold_uri=str(Path(tmp) / "cold.db"),
        embedding_dim=DIM,
    ))


def record(i: int, turn: int) -> MemoryRecord:
    return MemoryRecord(
        entity_name=f"entity_{i}",
        fact_content=f"fact number {i}",
        memory_type=MemoryType.ACTIVE_CONTEXT,
        weight=1.0,
        consolidation_index=0,
        created_at_turn=turn,
        last_reinforced_turn=turn,
        related_entities=[],
    )


def simulate(store: HotStorage, batched: bool) -> tuple[float, dict[str, float]]:
    """Ingest NEW_PER_TURN records per turn, decaying everything each turn."""
    start = time.time()
    n = 0
    for turn in range(1, TURNS + 1):
        for _ in range(NEW_PER_TURN):
            store.upsert(record(n, turn), [float(n % 7)] * DIM)
            n += 1

        mutable = store.get_all_mutable()
        # deterministic stand-in for decay_all: weight shrinks with turn distance
        updates = [
            (r.id, round(1.0 / (1 + turn - r.created_at_turn), 6)) for r in mutable
        ]
        if batched:
            store.apply_weight_updates(updates)
        else:
            for record_id, weight in updates:
                store.update_weight(record_id, weight)

    elapsed = time.time() - start
    # keyed by entity_name, not id: each run mints fresh UUIDs, so ids are not
    # comparable across stores. entity_name is stable and unique per record here.
    state = {r.entity_name: round(r.weight, 6) for r in store.get_all()}
    return elapsed, state


def main() -> None:
    print(f"simulating {TURNS} turns x {NEW_PER_TURN} new memories/turn "
          f"({TURNS * NEW_PER_TURN} records)\n")

    slow_time, slow_state = simulate(fresh_store("loop"), batched=False)
    print(f"  per-record loop : {slow_time:6.2f}s")

    fast_time, fast_state = simulate(fresh_store("batch"), batched=True)
    print(f"  batched         : {fast_time:6.2f}s")

    if slow_state != fast_state:
        only_slow = {k: v for k, v in slow_state.items() if fast_state.get(k) != v}
        print(f"\n  MISMATCH — states differ on {len(only_slow)} record(s)")
        for k, v in list(only_slow.items())[:5]:
            print(f"    {k}: loop={v} batch={fast_state.get(k)}")
        sys.exit(1)

    print(f"\n  states identical across {len(slow_state)} records")
    if fast_time > 0:
        print(f"  speedup: {slow_time / fast_time:.1f}x")


if __name__ == "__main__":
    main()
