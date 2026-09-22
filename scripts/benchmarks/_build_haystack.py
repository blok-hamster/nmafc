"""Merge the ten finished stores into one, to build LongMemEval's haystack for free.

What makes LongMemEval hard is not the questions. It is that the answering fact
sits among a very large amount of unrelated material. Our stores hold one
conversation each, about 800 facts, so a question only ever competes against
its own conversation. That is an easy setting, and the +7.9 we measured on
mylocomoeval was measured in it.

The merge costs nothing. Every store already tags its records
`conversation_id = 'default'`, and `HotStorage` filters searches on exactly
`agent_id = ... AND conversation_id = ...`, so records from ten stores dropped
into one table are all visible to every query with no code change. The vectors
are already computed and the facts already extracted, so there is no embedding
call and no LLM call. Merged, a question faces 8,276 facts instead of ~830.

Turn numbers are the one thing that cannot be copied as they are. Every store
counts from 1 and runs to somewhere between 186 and 345, so ten stores merged
untouched would have ten different facts claiming turn 50, and the hydration
step that pulls raw conversation text back into the prompt would fetch whichever
one it happened to find. Each source therefore gets a turn offset, laid end to
end.

The offsets do a second job. Decay measures elapsed turns, and no fact in a
single store is more than 345 turns old. End to end the earliest facts are over
2,700 turns old, so anything that depends on age has roughly ten times as much
to work with. See RECENT_OPS.md for why that turned out not to help.

`memory_event_log.id` is an autoincrement rowid and `memory_fts` is an
external-content FTS5 index keyed on it, so ids are renumbered on the way in and
the index is rebuilt at the end rather than copied.

Usage:
    python -u scripts/benchmarks/_build_haystack.py --dry-run
    python -u scripts/benchmarks/_build_haystack.py --dst C:/nmafc_ab/haystack
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lancedb  # noqa: E402
import pandas as pd  # noqa: E402

# Columns holding a turn index, which must be shifted with the conversation.
TURN_COLUMNS = ("created_at_turn", "last_reinforced_turn")


def source_stores(run: Path, arm: str) -> list[Path]:
    stores = sorted(p for p in (run / "stores").iterdir()
                    if p.is_dir() and p.name.startswith(f"{arm}__"))
    if not stores:
        raise SystemExit(f"no stores matching {arm}__* under {run}")
    return stores


def turn_span(store: Path) -> int:
    conn = sqlite3.connect(store / "cold.db")
    try:
        return int(conn.execute("SELECT COALESCE(MAX(turn), 0) FROM turn_text")
                   .fetchone()[0])
    finally:
        conn.close()


def merge_cold(stores: list[Path], offsets: dict[Path, int], out: Path) -> dict:
    """One archive holding every conversation, turns laid end to end.

    The first store's file is copied to carry the schema, indexes and the FTS
    trigger across exactly, then emptied. Rebuilding the schema by hand here
    would let it drift from whatever `ColdStorage` actually creates.
    """
    shutil.copy2(stores[0] / "cold.db", out)
    conn = sqlite3.connect(out)
    conn.executescript(
        "DELETE FROM memory_event_log;"
        "DELETE FROM turn_text;"
        "DELETE FROM turn_timestamps;"
        "DELETE FROM sqlite_sequence;"
    )
    conn.commit()

    columns = [r[1] for r in conn.execute("PRAGMA table_info(memory_event_log)")]
    # `id` is reassigned by the autoincrement, so it is not carried over.
    carried = [c for c in columns if c != "id"]
    placeholders = ", ".join("?" for _ in carried)
    turn_index = carried.index("turn")

    counts = {"facts": 0, "turns": 0, "stamps": 0}
    for store in stores:
        shift = offsets[store]
        src = sqlite3.connect(store / "cold.db")
        try:
            rows = src.execute(
                f"SELECT {', '.join(carried)} FROM memory_event_log").fetchall()
            shifted = []
            for row in rows:
                row = list(row)
                row[turn_index] = (row[turn_index] or 0) + shift
                shifted.append(row)
            conn.executemany(
                f"INSERT INTO memory_event_log ({', '.join(carried)}) "
                f"VALUES ({placeholders})", shifted)
            counts["facts"] += len(shifted)

            for table, key in (("turn_text", "text"),
                               ("turn_timestamps", "occurred_at")):
                got = src.execute(
                    f"SELECT agent_id, conversation_id, turn, {key} FROM {table}"
                ).fetchall()
                conn.executemany(
                    f"INSERT OR REPLACE INTO {table} "
                    f"(agent_id, conversation_id, turn, {key}) VALUES (?, ?, ?, ?)",
                    [(a, c, t + shift, v) for a, c, t, v in got])
                counts["turns" if table == "turn_text" else "stamps"] += len(got)
        finally:
            src.close()

    conn.commit()
    # External-content FTS5: the trigger fires on insert but the index is
    # keyed on rowids that have just been reassigned, so it is rebuilt whole.
    conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")
    conn.commit()
    counts["fts"] = conn.execute("SELECT count(*) FROM memory_fts").fetchone()[0]
    conn.close()
    return counts


def merge_hot(stores: list[Path], offsets: dict[Path, int], out: Path) -> dict:
    frames = []
    for store in stores:
        db = lancedb.connect(str(store / "hot_lancedb"))
        frame = db.open_table("memory_vectors").to_pandas()
        for column in TURN_COLUMNS:
            if column in frame.columns:
                frame[column] = frame[column] + offsets[store]
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    duplicates = int(merged["id"].duplicated().sum())
    if duplicates:
        # Hot ids are content-derived, so a collision would silently drop a
        # fact from search rather than raise anywhere downstream.
        raise SystemExit(f"{duplicates} duplicate hot ids across stores; "
                         "merging would lose facts")

    out.mkdir(parents=True, exist_ok=True)
    lancedb.connect(str(out)).create_table(
        "memory_vectors", data=merged, mode="overwrite")
    return {
        "facts": len(merged),
        "conversation_ids": sorted(merged["conversation_id"].unique().tolist()),
        "agent_ids": sorted(merged["agent_id"].unique().tolist()),
        "turn_span": (int(merged["created_at_turn"].min()),
                      int(merged["created_at_turn"].max())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--dst", default="C:/nmafc_ab/haystack")
    ap.add_argument("--name", default="haystack",
                    help="Store name under <dst>/stores/<arm>__<name>.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stores = source_stores(Path(args.run), args.arm)
    offsets: dict[Path, int] = {}
    running = 0
    for store in stores:
        offsets[store] = running
        span = turn_span(store)
        print(f"  {store.name:34s} turns 1-{span:<4d} -> "
              f"{running + 1}-{running + span}")
        running += span
    print(f"\nmerged span: {running} turns")

    if args.dry_run:
        print("dry run, nothing written")
        return

    target = Path(args.dst) / "stores" / f"{args.arm}__{args.name}"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)

    cold = merge_cold(stores, offsets, target / "cold.db")
    hot = merge_hot(stores, offsets, target / "hot_lancedb")

    print(f"\ncold: {cold['facts']} facts, {cold['turns']} turns, "
          f"{cold['stamps']} timestamps, fts rows {cold['fts']}")
    print(f"hot : {hot['facts']} facts, turns "
          f"{hot['turn_span'][0]}-{hot['turn_span'][1]}")
    print(f"      agent_id={hot['agent_ids']} "
          f"conversation_id={hot['conversation_ids']}")
    print(f"\n-> {target}")
    print(f"use with:  --run {args.dst} --arm {args.arm}  "
          f"(store name '{args.name}')")


if __name__ == "__main__":
    main()
