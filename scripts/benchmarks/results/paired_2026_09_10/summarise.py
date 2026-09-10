"""Regenerate every table in the README's results section from these files.

No API calls, no store access, no dependencies outside the standard library.
Run it from the repository root:

    python scripts/benchmarks/results/paired_2026_09_10/summarise.py

The point of committing the per-question JSON is that a reader does not have
to take a number on trust. Every figure in the README comes out of here.

Why the rows look the way they do: both arms answered every question in one
window against the same persisted stores, and their predictions were written to
the *same row*. That is what makes McNemar exact the right test -- it needs the
pairing, and a design that runs two arms separately and subtracts the totals
cannot supply it.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections import defaultdict

HERE = pathlib.Path(__file__).parent

# The four categories every published LoCoMo comparison reports on. Adversarial
# is measured and printed below rather than dropped, because a category is not
# excludable on the grounds that we lose it -- see the README.
SCORED = ("single-hop", "temporal", "multi-hop", "open-domain")

# chars // 4. Crude, applied identically to both arms, and the comparison is
# what is being claimed rather than the absolute.
CHARS_PER_TOKEN = 4


def mcnemar_exact(rows, a: str, b: str) -> tuple[int, int, float]:
    """Two-sided exact test on the discordant pairs only.

    Concordant pairs carry no information about which arm is better, which is
    the whole reason to pair: on 1,985 questions where the arms agree 1,473
    times, the 512 disagreements are the entire evidence and a test on the
    totals throws most of the resolution away.
    """
    only_a = sum(1 for r in rows if r[a] and not r[b])
    only_b = sum(1 for r in rows if r[b] and not r[a])
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1))
    return only_a, only_b, min(1.0, 2 * tail / 2 ** n)


def line(name: str, rows: list[dict], a: str = "ours", b: str = "rag") -> None:
    if not rows:
        return
    acc_a = sum(r[f"{a}_ok"] for r in rows) / len(rows) * 100
    acc_b = sum(r[f"{b}_ok"] for r in rows) / len(rows) * 100
    tok_a = sum(r[f"{a}_chars"] for r in rows) / len(rows) / CHARS_PER_TOKEN
    tok_b = sum(r[f"{b}_chars"] for r in rows) / len(rows) / CHARS_PER_TOKEN
    won, lost, p = mcnemar_exact(rows, f"{a}_ok", f"{b}_ok")
    print(f"  {name:<14} n={len(rows):<5} {acc_a:5.1f}%  {acc_b:5.1f}%  "
          f"{acc_a - acc_b:+5.1f}   +{won}/-{lost:<4} p={p:<10.3g} "
          f"{tok_a:5.0f}t vs {tok_b:5.0f}t")


def main() -> None:
    rows = json.loads((HERE / "locomo_all.json").read_text(encoding="utf-8"))
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)

    print(f"\nPaired LoCoMo run, {len(rows)} questions, ours vs RAG")
    print(f"  {'category':<14} {'n':<7} {'ours':<7} {'RAG':<7} {'lead':<7} "
          f"{'discordant':<12} {'McNemar':<12} context")
    for category in SCORED:
        line(category, by_cat[category])
    line("adversarial", by_cat["adversarial"])
    print()
    line("SCORED (4)", [r for r in rows if r["category"] in SCORED])
    line("ALL (5)", rows)
    print()


if __name__ == "__main__":
    main()
