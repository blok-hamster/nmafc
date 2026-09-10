"""Embed a store's turns so hydration can rank them by meaning.

A separate pass on purpose. Ingestion decides what is worth remembering and
costs hours; this reads what is already there and costs one embedding call per
few hundred turns, so a store can gain the index or decline it without either
decision touching the other.

It writes `turn_vectors`, which nothing searches to answer a query. The vectors
only order turns competing for a hydration slot that a fact has already won, so
this does not put raw turns into the retrieval budget -- the objection the cold
store's own schema comment raises against indexing turn text, and a fair one.

Vectors already computed by `_screen_turn_embeddings.py` are picked up from the
cache beside the store rather than bought again.

Usage:
    python -u scripts/benchmarks/_build_turn_index.py
    python -u scripts/benchmarks/_build_turn_index.py --check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from nmafc.integration.factory import create_embedding_provider  # noqa: E402


async def run(args: argparse.Namespace) -> None:
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    db = store / "cold.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS turn_vectors (
                        agent_id TEXT NOT NULL DEFAULT 'default',
                        conversation_id TEXT NOT NULL DEFAULT 'default',
                        turn INTEGER NOT NULL,
                        vector TEXT NOT NULL,
                        PRIMARY KEY (agent_id, conversation_id, turn))""")
    rows = conn.execute(
        "SELECT agent_id, conversation_id, turn, text FROM turn_text").fetchall()
    have = {r[0] for r in conn.execute("SELECT turn FROM turn_vectors")}
    print(f"{len(rows)} turns, {len(have)} already indexed")

    if args.check:
        conn.close()
        return

    todo = [r for r in rows if r["turn"] not in have]
    if not todo:
        print("nothing to do")
        conn.close()
        return

    # The screen embedded these already. Reusing its cache is not an
    # optimisation so much as the only version that cannot end up with two sets
    # of vectors for the same turns from two runs of the same model.
    cache = store / "turn_vectors.json"
    cached: dict[int, list[float]] = {}
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        cached = {int(t): v for t, v in blob.get("vectors", {}).items()}
        print(f"  reusing {len(cached)} vectors from the screen's cache")

    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    written = 0
    for i in range(0, len(todo), 256):
        chunk = todo[i:i + 256]
        missing = [r for r in chunk if r["turn"] not in cached]
        if missing:
            vecs = await embedder.embed([r["text"] for r in missing])
            for r, v in zip(missing, vecs):
                cached[r["turn"]] = v
        conn.executemany(
            """INSERT INTO turn_vectors
                   (agent_id, conversation_id, turn, vector)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(agent_id, conversation_id, turn)
               DO UPDATE SET vector = excluded.vector""",
            [(r["agent_id"], r["conversation_id"], r["turn"],
              json.dumps(cached[r["turn"]])) for r in chunk])
        conn.commit()
        written += len(chunk)
        print(f"  indexed {written}/{len(todo)}", flush=True)
    conn.close()
    print(f"done, {written} turns indexed into {db}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--check", action="store_true",
                    help="Report what is indexed and write nothing")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
