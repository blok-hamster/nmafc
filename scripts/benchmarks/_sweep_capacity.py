"""Forgetting triggered by crowding rather than by age.

Every forgetting rule tried so far fires on a clock. Decay charges a fact for
every turn it survives; demote charges it for being old next to a newer rival.
Both lost, and they lost the same way: age does not predict being out of date. A
fact about who somebody is stays true for the whole conversation while getting
steadily older, and the clock bills it anyway.

This is the other trigger, and it is the one the brain actually seems to use.
Nothing is forgotten because time passed. Things are forgotten because the space
they occupy is contested -- new material arrives that overlaps them, and the
overlap, not the age, is what does the damage. A memory with no competitor is
not under pressure however old it is.

So:

  TRIGGER   a fact's own neighbourhood, not the clock. Count how many other
            facts in the store sit within `tau` cosine of it. Below `cap`
            neighbours nothing happens at all, at any age.

  ACTION    inside a crowded neighbourhood, keep the newest and suppress the
            rest. A fact is only ever dropped in favour of something newer
            saying almost the same thing, which is the one circumstance where
            age is evidence rather than noise.

The consequence worth stating in advance is that this rule is self-limiting. On
a small store it does nothing, because nothing is crowded. Every earlier sweep
ran on single-conversation stores of roughly 830 facts and would have shown a
flat line whatever the parameters, which is why this was never worth running
before the merged store existed. `--store haystack` holds 12,000 facts over ten
transcripts and is the first setting where the trigger can fire at all. Both are
run here, and the difference between them is the measurement.

Two controls, because a pruning rule that helps is worthless if the selection is
doing none of the work:

  OLDEST    drop the same number of facts, chosen by age alone. This is decay
            at matched aggression, and it isolates the trigger from the volume.
  RANDOM    drop the same number of facts, chosen by coin. Seeded.

If capacity pruning cannot beat a coin at matched volume, the density rule is
decoration.

No LLM calls. One embedding per question for retrieval; the density scan reads
vectors already on disk. Stores are opened read-only and nothing is written.

Usage:
    python -u scripts/benchmarks/_sweep_capacity.py --limit 2
    python -u scripts/benchmarks/_sweep_capacity.py --run C:/nmafc_ab/haystack \
        --store haystack --out C:/nmafc_ab/capacity_haystack.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import lancedb  # noqa: E402
import numpy as np  # noqa: E402

from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks._ab_budget import (  # noqa: E402
    close_readonly,
    open_memory,
)
from scripts.benchmarks._sweep_context_budget import retrieve_with_retry  # noqa: E402
from scripts.benchmarks._sweep_supersession import Tally  # noqa: E402
from scripts.benchmarks._test_updates import present  # noqa: E402


class DensityIndex:
    """Every visible fact of one store, with its vector and its turn.

    Keyed on fact text throughout, for the reason `FactIndex` documents at
    length: a record the router pulled from the archive has no id, because
    `_cold_row_to_record` deliberately invents none, so an id-keyed structure
    misses every archived fact silently rather than loudly.
    """

    def __init__(self, store: Path) -> None:
        vectors: list[np.ndarray] = []
        self.facts: list[str] = []
        turns: list[int] = []
        frame = lancedb.connect(str(store / "hot_lancedb")) \
            .open_table("memory_vectors").to_pandas()
        for row in frame.itertuples():
            vectors.append(np.asarray(row.vector, dtype=np.float32))
            self.facts.append(row.fact_content)
            turns.append(int(row.created_at_turn))
        conn = sqlite3.connect(store / "cold.db")
        try:
            rows = conn.execute(
                "SELECT fact_content, turn, embedding FROM memory_event_log "
                " WHERE is_active = 1 AND invalid_at IS NULL "
                "   AND embedding IS NOT NULL").fetchall()
        finally:
            conn.close()
        width = len(vectors[0])
        for fact, turn, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.size != width:
                continue
            vectors.append(vector)
            self.facts.append(fact)
            turns.append(int(turn))
        matrix = np.vstack(vectors).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.where(norms == 0, 1, norms)
        self.turns = np.asarray(turns, dtype=np.int64)
        self._scans: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def crowding(self, tau: float, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Cached on tau. The scan is the expensive part and `cap` is applied
        to its output, so a grid of caps costs one scan, not one each."""
        if tau not in self._scans:
            self._scans[tau] = self._scan(tau, block)
        return self._scans[tau]

    def _scan(self, tau: float, block: int) -> tuple[np.ndarray, np.ndarray]:
        """For every fact: how many neighbours within `tau`, and their newest turn.

        Blocked rather than one big product. The full similarity matrix for the
        merged store is 12,000 square, which is 144 million entries and roughly
        half a gigabyte before anything useful has been computed. Only two
        summaries per row are ever needed, so each block is reduced and thrown
        away.
        """
        n = len(self.facts)
        counts = np.zeros(n, dtype=np.int32)
        newest = np.full(n, -1, dtype=np.int64)
        for start in range(0, n, block):
            stop = min(start + block, n)
            sims = self.matrix[start:stop] @ self.matrix.T
            near = sims >= tau
            # A fact is its own nearest neighbour at cosine 1. Counting it
            # would make `cap` mean one thing at the boundary and another
            # everywhere else.
            for offset in range(stop - start):
                near[offset, start + offset] = False
            counts[start:stop] = near.sum(axis=1)
            masked = np.where(near, self.turns[None, :], -1)
            newest[start:stop] = masked.max(axis=1)
        return counts, newest

    def suppressed(self, tau: float, cap: int) -> set[str]:
        """Facts in a crowded neighbourhood that something newer overlaps."""
        counts, newest = self.crowding(tau)
        drop = (counts >= cap) & (newest > self.turns)
        return {self.facts[i] for i in np.flatnonzero(drop)}

    def oldest(self, count: int) -> set[str]:
        """The `count` oldest facts. Decay at matched volume."""
        if count <= 0:
            return set()
        return {self.facts[i] for i in np.argsort(self.turns)[:count]}

    def random(self, count: int, seed: int) -> set[str]:
        if count <= 0:
            return set()
        rng = random.Random(seed)
        return set(rng.sample(self.facts, min(count, len(self.facts))))


def screen(pool: list, blocked: set[str]) -> list:
    return [r for r in pool if r.fact_content not in blocked]


async def run(args: argparse.Namespace) -> None:
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for row in questions:
        by_conv[row["conv"]].append(row)
    if args.limit:
        by_conv = {k: v[: args.limit] for k, v in by_conv.items()}

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    taus = [float(t) for t in args.taus.split(",")]
    caps = [int(c) for c in args.caps.split(",")]
    policies = [(t, c) for t in taus for c in caps]
    names = ["shipped"]
    names += [f"capacity tau={t} cap={c}" for t, c in policies]
    names += [f"  control oldest tau={t} cap={c}" for t, c in policies]
    names += [f"  control random tau={t} cap={c}" for t, c in policies]
    tallies = {n: Tally() for n in names}
    # How much of the store each policy silences, and how much of that ever
    # reaches a prompt. A rule that suppresses thousands of facts none of which
    # were being retrieved scores identically to shipped, and reading that as
    # "harmless" rather than "inert" is the error this counter prevents.
    silenced: dict[str, int] = defaultdict(int)
    hit: dict[str, int] = defaultdict(int)

    total = sum(len(v) for v in by_conv.values())
    where = args.store or "one store per conversation"
    print(f"{total} questions against {where}, pool {args.pool} -> prompt "
          f"{args.budget}, retrieval only\n")

    if args.store:
        groups = [(args.store, [i for v in by_conv.values() for i in v])]
    else:
        groups = list(by_conv.items())

    for name, items in groups:
        store = Path(args.run) / "stores" / f"{args.arm}__{name}"
        if not store.is_dir():
            print(f"[{name}] no store at {store}, skipped")
            continue
        index = DensityIndex(store)
        print(f"[{name}] {len(items)} questions, {len(index.facts)} facts")

        blocked: dict[str, set[str]] = {"shipped": set()}
        for tau, cap in policies:
            crowded = index.suppressed(tau, cap)
            blocked[f"capacity tau={tau} cap={cap}"] = crowded
            # Matched volume, so the trigger is compared against the clock and
            # against a coin at the same aggression rather than at its own.
            blocked[f"  control oldest tau={tau} cap={cap}"] = \
                index.oldest(len(crowded))
            blocked[f"  control random tau={tau} cap={cap}"] = \
                index.random(len(crowded), args.seed)
            share = 100 * len(crowded) / max(len(index.facts), 1)
            print(f"          tau={tau} cap={cap}: trigger fires on "
                  f"{len(crowded)} facts ({share:.1f}% of the store)")
        for key, facts in blocked.items():
            silenced[key] += len(facts)

        memory = open_memory(store, llm, embedder, args.hot, args.cold,
                             args.pool, args.max_hops, compact=True,
                             hydrate=args.hydrate)
        try:
            router = memory._router
            turn = memory.current_turn + 1
            for item in items:
                pool = await retrieve_with_retry(router, item["question"], turn)
                if not pool:
                    continue
                for key in names:
                    kept = screen(pool, blocked[key]) if blocked[key] else pool
                    hit[key] += len(pool) - len(kept)
                    context = router.format_context(kept[: args.budget])
                    tallies[key].add(present(item["gold"], context),
                                     present(item["stale"], context))
        finally:
            close_readonly(memory)

    if not tallies["shipped"].n:
        print("nothing measured")
        return

    print(f"\n{'=' * 112}")
    print(f"  {'policy':30s} {'n':>5s} {'gold':>8s} {'stale':>8s} "
          f"{'only stale':>8s} {'neither':>8s}   {'vs shipped':>21s}  "
          f"{'silenced':>9s} {'in pools':>9s}")
    base = tallies["shipped"]
    for n in names:
        print(f"{tallies[n].row(n, base)}  {silenced[n]:9d} {hit[n]:9d}")

    print("\n  silenced is facts the policy removed from the store; in pools is")
    print("  how many of those it actually took out of a retrieved pool. A")
    print("  policy with a large silenced and a near-zero in pools has not been")
    print("  shown to be safe, only shown to be inert.")
    print("  A rule worth keeping holds gold flat or up while stale and")
    print("  only-stale fall, AND beats its own matched-volume controls.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {k: {**vars(v), "silenced": silenced[k], "in_pools": hit[k]}
             for k, v in tallies.items()}, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="",
                    help="One merged store for every question, e.g. haystack. "
                         "Empty means the per-conversation store.")
    ap.add_argument("--questions", default="C:/nmafc_ab/updates_final.json")
    ap.add_argument("--taus", default="0.80,0.85,0.90",
                    help="Cosine at which two facts count as neighbours.")
    ap.add_argument("--caps", default="2,5,10",
                    help="Neighbours a fact may have before its region counts "
                         "as crowded.")
    ap.add_argument("--pool", type=int, default=40,
                    help="Candidates the policy chooses from. Must exceed --budget.")
    ap.add_argument("--budget", type=int, default=20,
                    help="Records reaching the prompt.")
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="C:/nmafc_ab/capacity.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
