"""Where do the 9.8 seconds per question actually go?

full_v3 answered at a median of 9838 ms per question while RAG answered at 2357
ms with more than twice the context, so the generation call is not the
difference. Cold ROM was timed directly and costs 0.7 ms per query, which rules
out the archive. That leaves the Hot RAM path, and this measures it.

Three operations run per question, and the third is the one under suspicion:

    search()              vector top-k -- one indexed lookup
    get_by_entities()     graph traversal -- a WHERE scan per hop, no index
    apply_reinforcements() LTP writeback -- delete + re-add of every surviving
                          record, which appends a new table version each time

LanceDB is append-only: a delete + add does not overwrite in place, it writes a
new fragment and tombstones the old rows. Doing that once per question leaves
the table increasingly fragmented, and every later scan has to read across the
fragments. If that is the cost, it compounds over a run -- question 1500 is
slower than question 1 for reasons that have nothing to do with the question.

So the reinforcement loop here is run repeatedly against a COPY of a real store,
reporting the per-call cost as fragments accumulate, and then again after
compaction. A flat line means fragmentation is not the problem. A rising line
that compaction flattens means it is, and that the fix is a maintenance call
rather than anything about the memory model.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def timed(fn, repeat: int) -> float:
    """Median milliseconds over `repeat` calls, to blunt one-off stalls."""
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default=None, help="store dir name; default = largest")
    ap.add_argument("--cycles", type=int, default=60,
                    help="simulated questions, each doing one reinforcement")
    ap.add_argument("--repeat", type=int, default=7)
    args = ap.parse_args()

    import lancedb
    import numpy as np

    base = HERE / "results" / args.run / "stores"
    cands = [p for p in base.glob("*/hot_lancedb") if args.arm in p.parent.name]
    if not cands:
        raise SystemExit("no hot stores under %s" % base)
    src = (base / args.store / "hot_lancedb") if args.store else max(
        cands, key=lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()))

    tmp = Path(tempfile.mkdtemp(prefix="nmafc_lat_"))
    dst = tmp / "hot_lancedb"
    shutil.copytree(src, dst)
    print("store   : %s" % src.parent.name)
    print("copy    : %s\n" % dst)

    db = lancedb.connect(str(dst))
    table = db.open_table("memory_vectors")
    rows = table.search().limit(100000).to_list()
    n = len(rows)
    dim = len(rows[0]["vector"])
    entities = sorted({r["entity_name"] for r in rows})
    ids = [r["id"] for r in rows]
    rng = np.random.default_rng(0)
    q = rng.normal(size=dim).astype("float32")
    q /= np.linalg.norm(q)

    print("=" * 70)
    print("PER-OPERATION COST   (rows=%d, dim=%d, distinct entities=%d)"
          % (n, dim, len(entities)))
    print("=" * 70)

    def do_search():
        table.search(q.tolist()).distance_type("cosine").where(
            "invalid_at IS NULL").limit(10).to_list()

    # Two hops of traversal routinely touch dozens of entities per question.
    hop = entities[:40]
    quoted = ", ".join("'%s'" % e.replace("'", "''") for e in hop)

    def do_entities():
        table.search().where(
            "entity_name IN (%s) AND invalid_at IS NULL" % quoted
        ).limit(500).to_list()

    print("  search()            vector top-10   : %8.2f ms"
          % timed(do_search, args.repeat))
    print("  get_by_entities()   40 entities     : %8.2f ms"
          % timed(do_entities, args.repeat))

    # --- reinforcement, repeated, watching for drift as versions pile up ---
    print("\n" + "=" * 70)
    print("REINFORCEMENT WRITEBACK, REPEATED   (%d simulated questions)" % args.cycles)
    print("=" * 70)
    print("  each cycle reinforces 20 records: delete + re-add, as retrieval does")
    print("  %-8s %12s %14s %14s" % ("cycle", "reinforce", "search after", "traverse after"))

    marks = {1, 5, 10, 20, 30, 40, 50, args.cycles}
    for cycle in range(1, args.cycles + 1):
        batch = ids[(cycle * 20) % max(1, n - 20):][:20] or ids[:20]
        quoted_b = ", ".join("'%s'" % i for i in batch)
        t0 = time.perf_counter()
        got = table.search().where("id IN (%s)" % quoted_b).limit(10000).to_list()
        for row in got:
            row.pop("_distance", None)
            row["weight"] = 1.0
            row["last_reinforced_turn"] = cycle
        table.delete("id IN (%s)" % quoted_b)
        table.add(got)
        reinforce_ms = (time.perf_counter() - t0) * 1000

        if cycle in marks:
            print("  %-8d %11.2f %13.2f %14.2f" % (
                cycle, reinforce_ms,
                timed(do_search, 3), timed(do_entities, 3)))

    try:
        versions = len(table.list_versions())
    except Exception:
        versions = -1

    print("\n  table versions accumulated: %s" % (versions if versions >= 0 else "n/a"))

    print("\n" + "=" * 70)
    print("AFTER COMPACTION")
    print("=" * 70)
    t0 = time.perf_counter()
    try:
        table.optimize()
        opt_ms = (time.perf_counter() - t0) * 1000
        print("  optimize()                        : %8.2f ms (one call)" % opt_ms)
    except Exception as exc:
        print("  optimize() unavailable: %s" % exc)
    print("  search()            vector top-10   : %8.2f ms" % timed(do_search, args.repeat))
    print("  get_by_entities()   40 entities     : %8.2f ms" % timed(do_entities, args.repeat))

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n  temp copy removed; the real store was never written to")
    return 0


if __name__ == "__main__":
    sys.exit(main())
