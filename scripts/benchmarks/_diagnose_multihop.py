"""What actually goes wrong on multi-hop, question by question, for nothing.

Multi-hop is 96 questions and the shipped configuration is 1.0 behind RAG on it.
That headline invites the wrong move -- tuning retrieval until the number goes
green -- because 1.0 of 96 is one question, and a configuration picked because it
flipped one question is a configuration picked by noise.

The useful split is not ours-versus-RAG. It is:

  * **contested** -- exactly one arm is right. This is the only part the headline
    measures, and on multi-hop it is about 19 questions in 96. Anything that
    moves it moves it by a point at a time.
  * **shared failure** -- both arms wrong. This is the large bucket, and it is
    where a mechanism rather than a knob has room to work. A question both a
    vector index and a fact graph miss is usually failing for a structural
    reason: the answer needs two facts joined, and neither arm was ever holding
    both.
  * **shared success** -- both right. Nothing to win here, but everything to
    lose: any change is scored against how much of this it breaks.

So this reports all four cells, then dumps the contested and shared-failure
questions in full. Reading them is the point; the counts are just the index.

Free. Reads the saved paired run, generates nothing, calls no API.

Usage:
    python -u scripts/benchmarks/_diagnose_multihop.py
    python -u scripts/benchmarks/_diagnose_multihop.py --category temporal
    python -u scripts/benchmarks/_diagnose_multihop.py --show shared --limit 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CELLS = ("both right", "we win", "we lose", "both wrong")


def cell(row: dict) -> str:
    ours, rag = bool(row["ours_ok"]), bool(row["rag_ok"])
    if ours and rag:
        return "both right"
    if ours:
        return "we win"
    if rag:
        return "we lose"
    return "both wrong"


def wrap(text: str, width: int, indent: str) -> str:
    """Fold to width without importing textwrap's paragraph handling, which
    collapses the newlines that separate a gold answer's alternatives."""
    out: list[str] = []
    for line in str(text).split("\n"):
        while len(line) > width:
            cut = line.rfind(" ", 0, width)
            cut = cut if cut > 0 else width
            out.append(line[:cut])
            line = line[cut:].lstrip()
        out.append(line)
    return ("\n" + indent).join(out)


def dump(rows: list[dict], title: str, limit: int) -> None:
    print(f"\n{'=' * 78}\n{title}  ({len(rows)})\n{'=' * 78}")
    for i, r in enumerate(rows[:limit], 1):
        print(f"\n[{i}] {r['conv']}   ours {r['ours_chars'] // 4}t   "
              f"rag {r['rag_chars'] // 4}t")
        print(f"  Q     {wrap(r['question'], 68, '        ')}")
        print(f"  gold  {wrap(r['gold'], 68, '        ')}")
        print(f"  ours  {wrap(r['ours_pred'], 68, '        ')}")
        print(f"  rag   {wrap(r['rag_pred'], 68, '        ')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack.json")
    ap.add_argument("--category", default="multi-hop")
    ap.add_argument("--show", default="contested",
                    choices=["contested", "shared", "both", "none"],
                    help="Which failure bucket to print in full.")
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    rows = json.loads(Path(args.run).read_text(encoding="utf-8"))
    rows = [r for r in rows if r["category"] == args.category]
    if not rows:
        raise SystemExit(f"no {args.category} rows in {args.run}")

    counts = {c: 0 for c in CELLS}
    for r in rows:
        counts[cell(r)] += 1
    n = len(rows)
    ours = sum(bool(r["ours_ok"]) for r in rows)
    rag = sum(bool(r["rag_ok"]) for r in rows)

    print(f"{args.category}: {n} questions")
    print(f"  ours {ours / n * 100:5.1f}%   RAG {rag / n * 100:5.1f}%   "
          f"diff {(ours - rag) / n * 100:+.1f}")
    print()
    for c in CELLS:
        print(f"  {c:<12} {counts[c]:>4}  {counts[c] / n * 100:5.1f}%")
    print()
    print(f"  contested        {counts['we win'] + counts['we lose']:>4}  "
          f"<- everything the headline can move")
    print(f"  reachable ceiling if every shared failure were fixed: "
          f"{(ours + counts['both wrong']) / n * 100:.1f}%")

    ours_t = sum(r["ours_chars"] for r in rows) // len(rows) // 4
    rag_t = sum(r["rag_chars"] for r in rows) // len(rows) // 4
    print(f"\n  context   ours {ours_t}t   RAG {rag_t}t")

    if args.show in ("contested", "both"):
        dump([r for r in rows if cell(r) == "we lose"],
             "WE LOSE -- RAG right, we are wrong", args.limit)
    if args.show in ("shared", "both"):
        dump([r for r in rows if cell(r) == "both wrong"],
             "BOTH WRONG -- the bucket a mechanism could open", args.limit)


if __name__ == "__main__":
    main()
