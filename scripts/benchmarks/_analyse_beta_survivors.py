"""Why did clustering protection (beta) cost accuracy?

beta scales each fact's decay rate by (1 - beta * C), where C is the local
clustering coefficient of its entity. At beta = 0.5 the tuned arm lost 4.8
accuracy points (p = 0.049), and the loss was not spread evenly: single-hop
questions fell 25.6 points while multi-hop rose 7.7. Context shrank as well,
1571 -> 1351 tokens, which rules out the simplest story -- if protection merely
let more facts survive, retrieval would have returned more, not less.

The hypothesis: protection is earned by C, C rewards facts sitting in
interlinked neighbourhoods, and those are the general facts about a person. The
specific ones -- a price, a game title, a date -- hang off the graph with a link
or two and no triangle between their neighbours, so they score C = 0, receive no
protection, and are pruned first. Single-hop questions ask for exactly those.

Method. Matching entities by name across the two runs does not work: measured on
conv-26, only 81 of ~460 entity names appear in both runs, because the extractor
names the same fact differently each time. So this compares *curves*, not pairs.
Within each run independently, survival is measured against that run's own C.
At beta = 0, C multiplies out of the decay rate entirely, so its survival-vs-C
curve is the baseline: whatever relationship it shows is a confound (older facts
are both better connected and more reinforced), not protection. The treatment's
curve is read against it, difference-in-differences.

Cold ROM is append-only, so it holds every fact ever written including the
pruned ones; an entity in the cold log and absent from hot did not survive.

Only ActiveContext and EphemeralState are counted. CoreAnchor has lambda = 0, so
(1 - beta * C) multiplies zero and beta cannot reach it. That is also why the
sample is small: ~18% of extracted facts are decay-eligible at all.

Read-only. No API calls; every store is opened read-only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nmafc.engine.decay import clustering_coefficient  # noqa: E402

DECAYING = ("ActiveContext", "EphemeralState")
BUCKETS = ((0.0, "C = 0"), (0.34, "C 0.01-0.33"), (0.67, "C 0.34-0.66"), (1.01, "C 0.67-1"))


def _links(raw: str | None) -> list[str]:
    """Parse the stored related_entities.

    The column holds a JSON array. An earlier version of this script split it on
    commas, which turned ["a", "b"] into two malformed strings and silently gave
    every entity the wrong neighbours -- hence the explicit parse and the
    fallback rather than a bare json.loads.
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not isinstance(parsed, list):
        return []
    return [str(p).strip().lower() for p in parsed if str(p).strip()]


def load_cold(store: Path) -> dict[str, dict]:
    """Latest event per entity from the cold log, keyed by lowercased name."""
    con = sqlite3.connect(f"file:{store / 'cold.db'}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT entity_name, memory_type, related_entities, is_active, turn, "
            "fact_content FROM memory_event_log ORDER BY id"
        ).fetchall()
    finally:
        con.close()

    latest: dict[str, dict] = {}
    for name, mtype, related, is_active, turn, content in rows:
        latest[name.lower()] = {
            "memory_type": mtype,
            "links": _links(related),
            "is_active": bool(is_active),
            "turn": turn,
            "content": content,
        }
    return latest


def load_hot(store: Path) -> set[str]:
    """Surviving entity names in Hot RAM."""
    import lancedb

    db = lancedb.connect(str(store / "hot_lancedb"))
    frame = db.open_table("memory_vectors").to_pandas()[["entity_name"]]
    return {str(n).lower() for n in frame["entity_name"]}


def build_graph(cold: dict[str, dict]) -> dict[str, set[str]]:
    """Undirected adjacency, following decay.build_entity_graph's rules."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for entity, row in cold.items():
        adjacency[entity]
        for neighbour in row["links"]:
            if neighbour != entity:
                adjacency[entity].add(neighbour)
                adjacency[neighbour].add(entity)
    return dict(adjacency)


def bucket_of(c: float) -> str:
    for edge, label in BUCKETS:
        if c <= edge or edge == 0.0 and c == 0.0:
            if c == 0.0:
                return BUCKETS[0][1]
            if c < edge:
                return label
    return BUCKETS[-1][1]


def survival_curve(store: Path) -> tuple[dict[str, dict], list[tuple]]:
    """Survival rate by C bucket for one run, plus the per-entity detail."""
    cold = load_cold(store)
    hot = load_hot(store)
    graph = build_graph(cold)

    rows: dict[str, dict] = defaultdict(lambda: {"n": 0, "survived": 0, "deg": 0})
    detail = []
    for entity, row in cold.items():
        if row["memory_type"] not in DECAYING or not row["is_active"]:
            continue
        c = clustering_coefficient(graph, entity)
        b = bucket_of(c)
        survived = entity in hot
        rows[b]["n"] += 1
        rows[b]["survived"] += survived
        rows[b]["deg"] += len(graph.get(entity, ()))
        detail.append((entity, c, len(graph.get(entity, ())), survived, row["content"]))
    return dict(rows), detail


def merge(a: dict[str, dict], b: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"n": 0, "survived": 0, "deg": 0})
    for src in (a, b):
        for k, v in src.items():
            for field in ("n", "survived", "deg"):
                out[k][field] += v[field]
    return dict(out)


def show(title: str, rows: dict[str, dict]) -> None:
    print(f"\n  {title}")
    print(f"    {'bucket':<14}{'n':>5}{'avg deg':>9}{'survived':>11}")
    order = [label for _e, label in BUCKETS]
    for b in sorted(rows, key=lambda x: order.index(x) if x in order else 9):
        r = rows[b]
        print(
            f"    {b:<14}{r['n']:>5}{r['deg'] / r['n']:>9.1f}"
            f"{r['survived'] / r['n']:>10.1%}"
        )
    tot_n = sum(r["n"] for r in rows.values())
    tot_s = sum(r["survived"] for r in rows.values())
    print(f"    {'ALL':<14}{tot_n:>5}{'':>9}{tot_s / tot_n:>10.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", nargs="+", required=True, help="beta=0 stores")
    parser.add_argument("--treatment", nargs="+", required=True, help="beta>0 stores")
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    ctrl_rows, ctrl_detail = {}, []
    for s in args.control:
        r, d = survival_curve(Path(s))
        ctrl_rows, ctrl_detail = merge(ctrl_rows, r), ctrl_detail + d
    treat_rows, treat_detail = {}, []
    for s in args.treatment:
        r, d = survival_curve(Path(s))
        treat_rows, treat_detail = merge(treat_rows, r), treat_detail + d

    print("=" * 70)
    print(f"Survival of decay-eligible facts, by clustering coefficient")
    print("=" * 70)
    show(f"control  beta = 0   ({len(args.control)} stores)", ctrl_rows)
    show(f"treatment beta = {args.beta}  ({len(args.treatment)} stores)", treat_rows)

    print("\n  difference-in-differences (treatment - control, by bucket)")
    for b in [label for _e, label in BUCKETS]:
        if b in ctrl_rows and b in treat_rows:
            pc = ctrl_rows[b]["survived"] / ctrl_rows[b]["n"]
            pt = treat_rows[b]["survived"] / treat_rows[b]["n"]
            print(f"    {b:<14}{pt - pc:>+8.1%}   (n={ctrl_rows[b]['n']} vs {treat_rows[b]['n']})")

    print("\n  what high-C facts actually are (treatment, C > 0, survived):")
    hi = [d for d in treat_detail if d[1] > 0 and d[3]]
    for e, c, deg, _s, content in sorted(hi, key=lambda x: -x[1])[:10]:
        print(f"    C={c:.2f} deg={deg:<3} {e}\n         {str(content)[:96]}")
    print("\n  what C = 0 facts that were pruned actually are (treatment):")
    lo = [d for d in treat_detail if d[1] == 0 and not d[3]]
    for e, c, deg, _s, content in lo[:10]:
        print(f"    C=0.00 deg={deg:<3} {e}\n         {str(content)[:96]}")


if __name__ == "__main__":
    main()
