"""When the gold IS in a turn and we still miss it, where did BM25 rank that turn?

The oracle screen settled the ceiling: of the 55 open-domain questions arm B
loses, 36 have every content word of the gold sitting in some turn of the raw
transcript. The RAG cross-tabulation settled that this is where the gap lives --
RAG answers 58% of those 36 correctly against 32% of the 19 nobody can reach. So
the words are there, RAG finds them, and we do not.

`hydrate_scan` now ranks the whole conversation by BM25 and takes the best few.
That is the mechanism that should be finding these turns. This asks the only
question left about it: **for each gold-bearing turn, what rank does BM25 give
it out of the whole conversation?**

The two answers point at completely different builds, and both are cheap enough
that guessing between them would be indefensible:

    rank is shallow      the scorer already ranks the right turn near the top
                         and the budget throws it away. The fix is arithmetic --
                         hydrate more turns, or spend fewer tokens per turn --
                         and it is a config change, not a feature.

    rank is deep         BM25 does not see the turn at all. The question and the
                         turn are about the same thing in different words, which
                         is what a lexical scorer is definitionally blind to.
                         The fix is a semantic turn index: embed each turn once,
                         rank by cosine. That is a real build, but it needs no
                         re-ingestion and no extractor change.

The rank is computed against **every turn of the store**, which is the same pool
`_scanned_turns` ranks over, so the number printed is the rank that mechanism
actually sees rather than a friendlier one measured on a subset.

**Ties are broken pessimistically.** A turn scoring zero shares no distinguishing
word with the question, and `_scanned_turns` drops those outright, so they are
reported as unranked rather than given the rank their position in a sorted list
would flatter them with.

Read-only, reads cold ROM directly. No model, no embeddings, no spend.

Usage:
    python -u scripts/benchmarks/_screen_scan_rank.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nmafc.integration.grounding import source_scores  # noqa: E402

from scripts.benchmarks._screen_hydrate_pool import present  # noqa: E402


def turn_text(store: Path) -> dict[int, str]:
    """Every turn in the store as turn -> text.

    The oracle screen keyed by conversation and found the haystack holds one, so
    there is nothing to scope to here either. Keying by turn is what matters
    now: a rank is meaningless without knowing which row it belongs to.
    """
    db = store / "cold.db"
    if not db.is_file():
        raise SystemExit(f"no cold store at {db}")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute("SELECT turn, text FROM turn_text").fetchall()
    conn.close()
    return {int(t): x for t, x in rows}


def band(rank: int | None, budget: int) -> str:
    if rank is None:
        return "unranked (score 0)"
    if rank <= budget:
        return f"top {budget}"
    if rank <= 3 * budget:
        return f"top {3 * budget}"
    if rank <= 100:
        return "top 100"
    return "deeper than 100"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/open_domain.json")
    ap.add_argument("--budget", type=int, default=8,
                    help="The hydrate_scan setting being judged")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    texts = turn_text(store)
    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    lost = [r for r in rows if not r["b_ok"]]
    print(f"{len(texts)} turns, {len(lost)} open-domain losses\n")

    tally: dict[str, list[tuple]] = {}
    unreachable = 0
    for r in lost:
        holders = [t for t, x in texts.items() if present(r["gold"], x)]
        if not holders:
            unreachable += 1
            continue
        scores = source_scores(r["question"], texts)
        order = sorted(texts, key=lambda t: (-scores.get(t, 0.0), t))
        # The best rank any gold-bearing turn achieves. If several turns carry
        # the gold, the scan only has to find one of them, so taking the best is
        # the honest reading of what the mechanism needs to do.
        best = None
        for i, t in enumerate(order, start=1):
            if t in holders and scores.get(t, 0.0) > 0.0:
                best = i
                break
        tally.setdefault(band(best, args.budget), []).append((best, r, holders))

    reachable = sum(len(v) for v in tally.values())
    print(f"  {unreachable} of {len(lost)} losses have the gold in no turn "
          f"at all.\n  Of the {reachable} that do, where BM25 ranks the "
          f"gold-bearing turn:\n")
    order = [f"top {args.budget}", f"top {3 * args.budget}", "top 100",
             "deeper than 100", "unranked (score 0)"]
    for name in order:
        group = tally.get(name, [])
        if group:
            print(f"    {name:<22}{len(group):>5}"
                  f"{100 * len(group) / max(reachable, 1):>7.1f}%")

    print(f"\n  A turn in the top {args.budget} is one the scan already offers, "
          f"so a loss\n  there is a conversion or rendering failure, not a "
          f"search one.")

    for name in order[1:]:
        group = tally.get(name, [])
        if not group:
            continue
        print(f"\n{'=' * 70}\n{name}: {len(group)} questions BM25 cannot find")
        for rank, r, holders in group[: args.show]:
            hit = texts[holders[0]].replace("\n", " ")
            print(f"\n  rank {rank}  Q     {r['question'][:66]}")
            print(f"           gold  {r['gold'][:66]}")
            print(f"           turn  {hit[:100]}")


if __name__ == "__main__":
    main()
