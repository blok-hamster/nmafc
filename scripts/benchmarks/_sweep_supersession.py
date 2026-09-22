"""A replacement for decay, screened before anything in `src/` is touched.

Decay measured properly is harmful: connected up on mylocomoeval it removes
stale facts (62.4% to 51.5%) but removes the answering fact almost twice as
fast (79.2% to 60.4%), and the case that decides correctness gets worse
(stale-only 14.9% to 19.8%). See RECENT_OPS.md for the full table.

The reason is that decay's only input is age, and age does not predict being
out of date. A fact about who somebody is stays true for the whole conversation
while getting steadily older, and decay charges it for every turn. The
superseded facts are older than what replaced them, but nowhere near enough to
pay for the damage done to everything else that is old and still true.

So the fix is to stop treating age as evidence. Two policies here, both
retrieval-time, both needing no re-ingestion and no extra API call:

  DEMOTE     Age only counts when something newer says the same thing. A fact
             is pushed down only if another retrieved fact is about the same
             subject and is newer. A fact with no newer rival is the best
             answer available however old it is, so it is left alone. That is
             the whole difference from decay, which charges everything for age
             whether or not a replacement exists.

  EXPAND     Do not delete what looks superseded, follow it. A fact that has
             been overtaken is an excellent pointer to whatever overtook it,
             because the two are about the same thing and therefore sit close
             together in the vector space. So each retrieved fact is used as a
             query in its own right, and newer neighbours are pulled in. This
             is aimed at the 14.9% where only the stale fact was retrieved and
             the answer never reached the prompt at all -- pruning cannot fix
             those, because the thing that is missing is not the thing being
             pruned.

EXPAND costs nothing because the vectors are already stored: a fact's own
embedding is read out of the table and used as the query, so no text is
embedded and no model is called.

Both policies select from a pool deliberately wider than the prompt budget.
Reordering a pool that is already exactly the budget changes nothing that a
presence check can see, which is the trap the first decay sweep fell into.

Subject matching is token overlap on the fact text rather than `entity_name`.
The extractor names one entity per event (`jon_job_loss_jan_2023`), so two
facts about the same job land on different names and never match. That is the
same naming that has kept `detect_override` at zero firings across 8,276 facts.

Usage:
    python -u scripts/benchmarks/_sweep_supersession.py --limit 2
    python -u scripts/benchmarks/_sweep_supersession.py --out C:/nmafc_ab/supersession_policy.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

import numpy as np  # noqa: E402

from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)
from nmafc.integration.query_router import QueryRouter  # noqa: E402
from nmafc.schemas.memory import DecayConfig, MemoryRecord  # noqa: E402
from nmafc.storage.config import NMafcConfig, StorageConfig  # noqa: E402
from nmafc.wrapper import NeuromorphicMemory  # noqa: E402

from scripts.benchmarks._ab_budget import close_readonly  # noqa: E402
from scripts.benchmarks._sweep_context_budget import retrieve_with_retry  # noqa: E402
from scripts.benchmarks._test_updates import present, tokens  # noqa: E402


def open_memory(store: Path, llm, embedder, args):
    """The shipped settings, except the pool is wider than the prompt budget.

    `rerank_top_k` is raised to `--pool` so a selection policy has something to
    choose between. At the shipped value the reranker returns exactly as many
    records as the prompt takes, and any policy that only reorders is invisible.
    """
    return NeuromorphicMemory(
        llm_provider=llm,
        embedding_provider=embedder,
        config=NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(store / "hot_lancedb"),
                cold_uri=str(store / "cold.db"),
            ),
            decay=DecayConfig(
                max_hops=args.max_hops,
                top_k=args.hot,
                fallback_keyword_limit=args.cold,
                rerank_top_k=args.pool,
                hydrate_top_k=args.hydrate,
                defer_reinforcement_writes=True,
            ),
        ),
    )


def subject_overlap(a: str, b: str) -> float:
    """How much two facts are about the same thing, 0 to 1.

    Jaccard over content words. Deliberately blunt: the aim is to spot that two
    facts concern one subject, not to decide which is true.
    """
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def demote(records: list, threshold: float) -> list:
    """Push a record down only when a newer record says the same thing.

    Order is otherwise preserved, so a record with no newer rival keeps exactly
    the rank retrieval gave it. This is the whole departure from decay: age is
    charged for only in the presence of a replacement.
    """
    superseded = set()
    for i, rec in enumerate(records):
        for j, other in enumerate(records):
            if i == j or other.created_at_turn <= rec.created_at_turn:
                continue
            if subject_overlap(rec.fact_content, other.fact_content) >= threshold:
                superseded.add(i)
                break
    keep = [r for i, r in enumerate(records) if i not in superseded]
    pushed = [r for i, r in enumerate(records) if i in superseded]
    return keep + pushed


class FactIndex:
    """Every stored vector, Hot RAM and Cold ROM, so any fact can be a query.

    Reading a stored embedding costs nothing. Re-embedding the fact's text
    would cost one call per candidate per question and produce nearly the same
    vector, since that text is what was embedded in the first place.

    Keyed on fact text, not on id, and that is not a stylistic choice. A record
    the router pulled out of the archive has no id: `_cold_row_to_record`
    deliberately invents none, so `MemoryRecord` falls back to a fresh uuid4.
    An id-keyed index therefore misses every archived candidate silently -- it
    does not fail, it just returns nothing. A trace of the first version showed
    23 of 38 pool records present in the index, and the missing 15 were exactly
    the archived ones. Since Cold holds 8,276 facts against Hot's 3,807, an
    index built from Hot alone was also refusing to look at most of the store.
    """

    def __init__(self, store: Path) -> None:
        import lancedb

        vectors: list[np.ndarray] = []
        self.records: list = []

        frame = lancedb.connect(str(store / "hot_lancedb")) \
            .open_table("memory_vectors").to_pandas()
        for row in frame.itertuples():
            vectors.append(np.asarray(row.vector, dtype=np.float32))
            # LanceDB hands back a numpy array for the list column, and
            # `or []` on one raises rather than falling back.
            related = row.related_entities
            self.records.append(MemoryRecord(
                id=row.id,
                entity_name=row.entity_name,
                fact_content=row.fact_content,
                memory_type=row.memory_type,
                created_at_turn=int(row.created_at_turn),
                last_reinforced_turn=int(row.last_reinforced_turn),
                related_entities=[] if related is None else [str(e) for e in related],
            ))

        conn = sqlite3.connect(store / "cold.db")
        conn.row_factory = sqlite3.Row
        try:
            # Same visibility rule the shipped archive search applies, so the
            # index cannot offer a neighbour retrieval would never return.
            rows = conn.execute(
                "SELECT turn, entity_name, fact_content, memory_type, "
                "       related_entities, embedding "
                "  FROM memory_event_log "
                " WHERE is_active = 1 AND invalid_at IS NULL "
                "   AND embedding IS NOT NULL").fetchall()
        finally:
            conn.close()
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size != len(vectors[0]):
                # A width change means the archive spans two embedding models.
                continue
            vectors.append(vector)
            self.records.append(QueryRouter._cold_row_to_record(dict(row)))

        matrix = np.vstack(vectors).astype(float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.where(norms == 0, 1, norms)
        self.turns = np.array([r.created_at_turn for r in self.records])
        # First occurrence wins. Hot is loaded first, so a fact held in both
        # tiers is followed from its Hot copy, which carries the real id.
        self.by_text: dict[str, int] = {}
        for position, record in enumerate(self.records):
            self.by_text.setdefault(record.fact_content.strip(), position)

    def newer_neighbours(self, record, top_k: int, floor: float = 0.0) -> list:
        position = self.by_text.get(record.fact_content.strip())
        if position is None:
            return []
        vector = self.matrix[position]
        if not vector.any():
            return []
        scores = self.matrix @ vector
        # Only what was said later than the fact being followed. An older
        # neighbour is a restatement, not a replacement. This also drops the
        # fact's own row and its copy in the other tier, both of which carry
        # the same turn.
        scores = np.where(self.turns > record.created_at_turn, scores, -1.0)
        order = np.argsort(-scores)[:top_k]
        # A floor matters more than the count. Without one, every fact returns
        # its `top_k` nearest later facts whether or not any of them is about
        # the same thing, and the prompt fills with confident noise.
        return [self.records[i] for i in order if scores[i] >= max(floor, 1e-9)]


def expand(records: list, index: FactIndex, follow: int, per: int,
           floor: float) -> list:
    """Follow each of the leading facts to whatever was said about it later.

    A neighbour is placed immediately behind the fact that led to it, not at
    the end of the list. Position is the entire point: the case being attacked
    is the one where a superseded fact ranked highly and its replacement was
    never retrieved, so the replacement has to land next to it to reach the
    prompt. Appending to the tail of a pool that is wider than the prompt
    budget is a guaranteed no-op, which is what the first version of this
    measured.

    Duplicates are held off by text as well as by id, since the same fact
    reaches the pool from Hot with a real id and from Cold with a fresh uuid.
    """
    seen_ids = {r.id for r in records}
    seen_text = {r.fact_content.strip() for r in records}
    out: list = []
    for position, rec in enumerate(records):
        out.append(rec)
        if position >= follow:
            continue
        for found in index.newer_neighbours(rec, per, floor):
            text = found.fact_content.strip()
            if found.id in seen_ids or text in seen_text:
                continue
            seen_ids.add(found.id)
            seen_text.add(text)
            out.append(found)
    return out


class Tally:
    def __init__(self) -> None:
        self.n = self.gold = self.stale = self.only_stale = self.neither = 0

    def add(self, gold: bool, stale: bool) -> None:
        self.n += 1
        self.gold += gold
        self.stale += stale
        self.only_stale += stale and not gold
        self.neither += not stale and not gold

    def row(self, label: str, base: "Tally | None") -> str:
        p = lambda x: 100 * x / self.n  # noqa: E731
        delta = ""
        if base is not None and base is not self:
            b = lambda x: 100 * x / base.n  # noqa: E731
            delta = (f"   {p(self.gold) - b(base.gold):+6.1f}"
                     f" {p(self.stale) - b(base.stale):+6.1f}"
                     f" {p(self.only_stale) - b(base.only_stale):+6.1f}")
        return (f"  {label:24s} {self.n:5d} {p(self.gold):7.1f}% "
                f"{p(self.stale):7.1f}% {p(self.only_stale):7.1f}% "
                f"{p(self.neither):7.1f}%{delta}")


async def run(args: argparse.Namespace) -> None:
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for row in questions:
        by_conv[row["conv"]].append(row)
    if args.limit:
        by_conv = {k: v[: args.limit] for k, v in by_conv.items()}

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    thresholds = [float(t) for t in args.thresholds.split(",")]
    names = ["shipped"]
    names += [f"demote t={t}" for t in thresholds]
    names += ["expand"]
    names += [f"expand+demote t={t}" for t in thresholds]
    tallies = {n: Tally() for n in names}

    total = sum(len(v) for v in by_conv.values())
    print(f"{total} questions, pool {args.pool} -> prompt {args.budget}, "
          f"retrieval only\n")
    # How often expansion fires at all. A policy that never fires scores
    # identically to shipped, and reading that as "no effect" rather than "no
    # attempt" is the mistake this counter exists to prevent.
    reached = followed = 0

    for conv, items in by_conv.items():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv}"
        if not store.is_dir():
            continue
        print(f"[{conv}] {len(items)} questions")
        index = FactIndex(store)
        print(f"          index: {len(index.records)} vectors "
              f"({len(index.by_text)} distinct facts)")
        memory = open_memory(store, llm, embedder, args)
        try:
            router = memory._router
            for item in items:
                pool = await retrieve_with_retry(router, item["question"],
                                                 memory.current_turn + 1)
                if not pool:
                    continue
                widened = expand(pool, index, args.follow, args.per, args.sim)
                reached += len(widened) - len(pool)
                followed += min(args.follow, len(pool))

                def score(records, name: str) -> None:
                    context = router.format_context(records[: args.budget])
                    tallies[name].add(present(item["gold"], context),
                                      present(item["stale"], context))

                score(pool, "shipped")
                score(widened, "expand")
                for t in thresholds:
                    score(demote(pool, t), f"demote t={t}")
                    score(demote(widened, t), f"expand+demote t={t}")
        finally:
            close_readonly(memory)

    if not tallies["shipped"].n:
        print("nothing measured")
        return

    print(f"\n{'=' * 104}")
    print(f"  {'policy':24s} {'n':>5s} {'gold':>8s} {'stale':>8s} "
          f"{'only stale':>8s} {'neither':>8s}   {'vs shipped':>21s}")
    base = tallies["shipped"]
    for n in names:
        print(tallies[n].row(n, base))

    print(f"\n  expansion pulled in {reached} facts across {followed} follows "
          f"({100 * reached / max(followed, 1):.1f}%) at sim>={args.sim}")
    print("  decay's failure was -18.8 gold for -10.9 stale. A policy worth")
    print("  keeping holds gold flat or up while stale and only-stale fall.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {k: vars(v) for k, v in tallies.items()}, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--questions", default="C:/nmafc_ab/updates_final.json")
    ap.add_argument("--pool", type=int, default=40,
                    help="Candidates the policy chooses from. Must exceed --budget.")
    ap.add_argument("--budget", type=int, default=20, help="Records reaching the prompt.")
    ap.add_argument("--thresholds", default="0.2,0.3,0.4")
    ap.add_argument("--follow", type=int, default=5,
                    help="How many leading facts to follow forward in time.")
    ap.add_argument("--per", type=int, default=1,
                    help="Newer neighbours pulled in per followed fact.")
    ap.add_argument("--sim", type=float, default=0.8,
                    help="Cosine floor a newer neighbour must clear to be "
                         "pulled in. Without one the prompt fills with noise.")
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="C:/nmafc_ab/supersession_policy.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
