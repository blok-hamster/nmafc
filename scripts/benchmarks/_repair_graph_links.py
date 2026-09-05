"""Repair an already-built store's graph links, without re-ingesting it.

`resolve_link_targets` maps each extracted link onto an entity that actually
exists, so a store built with it on never accumulates dead pointers. The stores
on disk were built before it existed, and rebuilding them costs one extraction
call per exchange across ten conversations: roughly five hours, to test a change
that only affects what retrieval can traverse.

The repair can be done offline because nothing was lost. Cold ROM is
append-only, so it still holds every record ever written together with the links
the extractor gave it -- including the links to nowhere that were later swept
out of Hot RAM as dead pointers. Re-resolving those against the entity names
that actually exist recovers the ones that were near misses.

Replayed in turn order over the ten LoCoMo stores, counting a link only when
its target is still alive:

    today                 0.92 links per fact, 31% of facts stranded
    with resolution       1.50 links per fact, 10% of facts stranded

Two caveats worth stating in any write-up that leans on this:

  * Resolution here sees the entity names that exist *now*. The live path
    resolves against the names that existed at that turn, so a link written
    before its target was created resolves here and would not have then. This
    script replays turn by turn to stay faithful, but a store repaired after
    the fact still cannot reproduce interactions between resolution and the
    pruning that happened along the way.
  * It repairs the graph, not the facts. Records pruned during the original run
    stay pruned; only the links between the survivors change.

Writes to the stores in place. Pass --backup unless the stores are disposable.

Usage:
    python scripts/benchmarks/_repair_graph_links.py --run scripts/benchmarks/results/full_v3
    python scripts/benchmarks/_repair_graph_links.py --run ... --backup --write
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nmafc.engine.linking import resolve_link_targets
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage


def archived_links(cold_db: Path) -> dict[int, list[tuple[str, list[str]]]]:
    """Every record ever written, with its original links, grouped by turn."""
    con = sqlite3.connect(str(cold_db))
    try:
        rows = con.execute(
            "SELECT turn, entity_name, related_entities FROM memory_event_log "
            "ORDER BY turn, id"
        ).fetchall()
    finally:
        con.close()

    by_turn: dict[int, list[tuple[str, list[str]]]] = {}
    for turn, name, related in rows:
        if isinstance(related, str):
            try:
                related = json.loads(related)
            except ValueError:
                related = []
        by_turn.setdefault(turn, []).append((name, list(related or [])))
    return by_turn


def plan_store(store_dir: Path, threshold: float):
    """Return (hot, updates, before, after) for one store without writing.

    `before` and `after` are (links_per_fact, percent_stranded), counting a link
    only when its target survived -- the same definition on both sides, since
    counting unresolvable links as links is what made the graph look healthier
    than it was in the first place.
    """
    hot = HotStorage(StorageConfig(
        hot_uri=str(store_dir / "hot_lancedb"),
        cold_uri=str(store_dir / "cold.db"),
        agent_id="default",
        conversation_id="default",
    ))
    live = hot.get_all()
    if not live:
        return hot, [], (0.0, 0.0), (0.0, 0.0)

    survivors = {r.entity_name.lower() for r in live}
    by_turn = archived_links(store_dir / "cold.db")

    # Replay in turn order, so a link can only resolve against entity names
    # that existed by then plus the ones written in the same turn -- which is
    # the scope the live path has, and the prompt's "another entity in this
    # same tool call".
    known: list[str] = []
    resolved: dict[str, list[str]] = {}
    for turn in sorted(by_turn):
        batch = by_turn[turn]
        scope = known + [name for name, _ in batch]
        for name, links in batch:
            resolved[name.lower()] = resolve_link_targets(
                links,
                [e for e in scope if e.lower() != name.lower()],
                min_overlap=threshold,
            )
        known.extend(name for name, _ in batch)

    updates: list[tuple[str, list[str]]] = []
    before_links = before_stranded = 0
    after_links = after_stranded = 0
    for record in live:
        current = [x for x in record.related_entities if x.lower() in survivors]
        before_links += len(current)
        before_stranded += not current

        new_links = [
            x for x in resolved.get(record.entity_name.lower(), [])
            if x.lower() in survivors
        ]
        after_links += len(new_links)
        after_stranded += not new_links

        if new_links != list(record.related_entities):
            updates.append((record.id, new_links))

    n = len(live)
    return (hot, updates,
            (before_links / n, 100 * before_stranded / n),
            (after_links / n, 100 * after_stranded / n))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory holding stores/")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="minimum token overlap for a link name to resolve")
    ap.add_argument("--write", action="store_true",
                    help="apply the repair (default: report only)")
    ap.add_argument("--backup", action="store_true",
                    help="copy each store to <name>.pre_repair before writing")
    args = ap.parse_args()

    stores = [s for s in sorted((Path(args.run) / "stores").glob(f"{args.arm}__*"))
              if not s.name.endswith(".pre_repair")]
    if not stores:
        raise SystemExit(f"no {args.arm} stores under {args.run}")

    print(f"run  : {args.run}")
    print(f"mode : {'WRITE' if args.write else 'dry run, nothing written'}\n")
    print(f"  {'store':16s} {'facts':>6s} {'links/fact':>18s} {'stranded':>16s}")

    facts = wb = sb = wa = sa = 0
    for store in stores:
        hot, updates, before, after = plan_store(store, args.threshold)
        n = hot.count()
        if not n:
            print(f"  {store.name[-12:]:16s} {'empty':>6s}")
            continue
        print(f"  {store.name[-12:]:16s} {n:6d} "
              f"{before[0]:8.2f} -> {after[0]:6.2f} "
              f"{before[1]:7.0f}% -> {after[1]:4.0f}%")
        facts += n
        wb += before[0] * n
        sb += before[1] * n
        wa += after[0] * n
        sa += after[1] * n

        if args.write:
            if args.backup:
                backup = store.with_name(store.name + ".pre_repair")
                if backup.exists():
                    print(f"    backup exists, refusing to overwrite: {backup.name}")
                    continue
                shutil.copytree(store, backup)
            hot.apply_relation_updates(updates)
            hot.compact()
            print(f"    {len(updates)} record(s) rewritten")

    if facts:
        print(f"\n  {'ALL':16s} {facts:6d} "
              f"{wb / facts:8.2f} -> {wa / facts:6.2f} "
              f"{sb / facts:7.0f}% -> {sa / facts:4.0f}%")
    if not args.write:
        print("\n  re-run with --write --backup to apply")


if __name__ == "__main__":
    main()
