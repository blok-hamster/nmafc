"""What does a symbol-indexed context actually cost, against a fair baseline?

The unfair comparison, and the one most retrieval papers make, is against
stuffing the repository into the prompt. Nobody does that. What a competent
coding agent does is grep for a name and read the file it is in, and that is a
strong baseline: it is exact, it needs no index, and it is usually right. So
that is what this measures against.

Both numbers are the same approximation used everywhere else in this
repository, `chars // 4`.

Three things are reported and none of them is accuracy:

  1. **Build cost.** Zero API calls and zero tokens, because indexing is a
     parse. Wall-clock is printed so the claim "cheap enough to re-run on every
     edit" can be checked rather than asserted.
  2. **Context cost per query**, symbol-indexed against read-the-file.
  3. **Whether the answering definition survives the budget**, which is the only
     thing that makes a saving meaningful. A cheaper context that dropped the
     symbol asked about is not cheaper, it is broken.

There is no accuracy column because this repository has no code benchmark, and
inventing one here would be scoring the mechanism against questions written by
the person who built it.

Usage:
    python -u scripts/benchmarks/_measure_code_context.py --root src
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nmafc.code.index import SymbolIndex  # noqa: E402
from nmafc.code.render import render_symbols, suggest_budget, tokens  # noqa: E402

# Real questions about this codebase, each naming a symbol that exists. The
# lexical resolver answers questions that name something; these are the shape it
# is for, and a question naming nothing is measured separately below.
QUERIES = [
    "what does format_context do",
    "how does retrieve use rerank",
    "what calls decay_record",
    "explain compute_lambda",
    "where is separate_facts used",
    "what does flush_reinforcements touch",
    "how does prepare_store resume",
    "what does apply_reinforcements do",
]


def file_baseline(index: SymbolIndex, symbols) -> int:
    """Tokens to read every file the answering symbols live in, once each."""
    seen: set[str] = set()
    total = 0
    for symbol in symbols:
        if symbol.path in seen:
            continue
        seen.add(symbol.path)
        try:
            total += tokens((index.root / symbol.path).read_text(
                encoding="utf-8"))
        except OSError:
            continue
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="src")
    ap.add_argument("--ceiling", type=int, default=1200,
                    help="Largest context the renderer may spend on one query.")
    ap.add_argument("--hops", type=int, default=1)
    args = ap.parse_args()

    start = time.perf_counter()
    index = SymbolIndex(args.root)
    count = index.build()
    build_ms = (time.perf_counter() - start) * 1000

    print(f"indexed {count} symbols in {len(index.files)} files, "
          f"{build_ms:.0f} ms, 0 API calls, 0 tokens\n")

    print(f"  {'query':38s} {'ours':>7s} {'file':>7s} {'saved':>7s} {'seed kept':>10s}")
    ours_total = baseline_total = 0
    for query in QUERIES:
        items = index.retrieve(query, hops=args.hops)
        if not items:
            print(f"  {query[:38]:38s} {'-':>7s} {'-':>7s} {'-':>7s} "
                  f"{'no symbol':>10s}")
            continue
        budget = suggest_budget(items, args.ceiling)
        block = render_symbols(index, items, budget_tokens=budget)
        seeds = [s for s, hop in items if hop == 0]
        base = file_baseline(index, seeds)
        ours = tokens(block)
        ours_total += ours
        baseline_total += base
        # The saving only counts if the definition asked about is still in the
        # block. A budget that dropped the seed has not saved anything.
        kept = all(f"{s.path}:{s.lineno}-" in block for s in seeds)
        saved = f"{100 * (base - ours) / base:.0f}%" if base else "-"
        print(f"  {query[:38]:38s} {ours:6d}t {base:6d}t {saved:>7s} "
              f"{('yes' if kept else 'NO'):>10s}")

    if baseline_total:
        print(f"\n  total {ours_total}t against {baseline_total}t, "
              f"{100 * (baseline_total - ours_total) / baseline_total:.0f}% less "
              f"than reading the files")

    empty = index.retrieve("how does the whole system fit together", hops=1)
    print(f"\n  a question naming no symbol returns {len(empty)} symbols, "
          f"by design -- this resolves names and does not read sentences.")

    stale_start = time.perf_counter()
    stale = index.stale()
    print(f"  staleness check over {len(index.files)} files: "
          f"{(time.perf_counter() - stale_start) * 1000:.0f} ms, "
          f"{len(stale)} changed")


if __name__ == "__main__":
    main()
