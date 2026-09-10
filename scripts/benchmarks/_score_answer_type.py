"""What the answer-type arm is worth once the rule is gated, and against RAG.

Two questions the A/B report does not answer, both free: every number here comes
from answers already generated and paid for.

**1. Gate the rule and the control regression goes away by construction.**

The arm as run put TYPE_RULE in the system prompt on every question, typed or
not, and the untyped controls paid for it. That is avoidable. The system prompt
is built per call, so the rule can be included only when the detector fires --
and on a question where it does not fire, the resulting prompt is byte-identical
to arm A's. Not similar: identical. Same system prompt, same question, same
retrieval, same call.

So arm A's answer to an untyped question *is* the gated arm's answer to it, and
the gated arm's score can be read off the file that already exists. There is no
cost to measure on the untyped slice because there is no difference to measure.
This is a design claim rather than an empirical one, and it is stronger for it:
an A/B on that slice would only be measuring the model's own sampling noise.

**2. Against RAG, on the questions this actually touches.**

`haystack.json` holds RAG's paired answer to every one of these questions from
the main run. RAG's prompt is untouched by any of this, so its answers stand.
The main run's `ours_ok` is printed alongside arm A as a drift check: A is the
same prompt re-run months later, so the gap between them is what a rerun costs
and it bounds how much of any gain is real.

Usage:
    python -u scripts/benchmarks/_score_answer_type.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest


def gated(row: dict) -> tuple[bool, int]:
    """The gated arm's verdict and context width for one question.

    Typed: arm B, which carries the rule and the tag. Untyped: arm A, because
    a gated arm asking an untyped question makes the same call arm A made.
    """
    if row["tag"]:
        return row["b_ok"], row["b_chars"]
    return row["a_ok"], row["a_chars"]


def line(label: str, n: int, *cells: str) -> None:
    print(f"  {label:<16}{n:>5}" + "".join(f"{c:>10}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ab", default="C:/nmafc_ab/answer_type_fixed.json")
    ap.add_argument("--main", default="C:/nmafc_ab/haystack.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.ab).read_text(encoding="utf-8"))
    main_run = {(r["conv"], r["question"]): r
                for r in json.loads(Path(args.main).read_text(encoding="utf-8"))}

    missing = [r for r in rows if (r["conv"], r["question"]) not in main_run]
    paired = [r for r in rows if (r["conv"], r["question"]) in main_run]

    print(f"{len(rows)} questions from the paired A/B, {len(paired)} of them "
          f"also in the main run")
    if missing:
        print(f"  {len(missing)} absent from the main run, excluded from the "
              f"RAG columns only")

    print("\nGATED ARM vs the shipped prompt. Untyped questions are arm A by "
          "construction,")
    print("so every flip below is a typed question.\n")
    line("", 0, "shipped", "gated", "diff", "fix", "brk", "p", "ctx")
    print("  " + "-" * 91)

    def gate_block(rs: list[dict], label: str) -> None:
        if not rs:
            return
        n = len(rs)
        verdicts = [gated(r) for r in rs]
        a = sum(r["a_ok"] for r in rs)
        g = sum(ok for ok, _ in verdicts)
        fix = sum(1 for r, (ok, _) in zip(rs, verdicts) if ok and not r["a_ok"])
        brk = sum(1 for r, (ok, _) in zip(rs, verdicts) if r["a_ok"] and not ok)
        p = binomtest(fix, fix + brk).pvalue if fix + brk else 1.0
        ctx = sum(c for _, c in verdicts) / n / 4
        line(label, n, f"{a / n * 100:.1f}%", f"{g / n * 100:.1f}%",
             f"{(g - a) / n * 100:+.1f}", str(fix), str(brk), f"{p:.3g}",
             f"{ctx:.0f}")

    gate_block(rows, "ALL")
    print()
    for cat in sorted({r["category"] for r in rows}):
        gate_block([r for r in rows if r["category"] == cat], cat)

    print("\n\nAGAINST RAG on the same questions. `ours (main)` is the same "
          "prompt as `shipped`,")
    print("run months earlier -- the gap between those two columns is rerun "
          "drift, not a change.\n")
    line("", 0, "RAG", "ours(main)", "shipped", "gated", "vs RAG")
    print("  " + "-" * 91)

    def rag_block(rs: list[dict], label: str) -> None:
        if not rs:
            return
        n = len(rs)
        ref = [main_run[(r["conv"], r["question"])] for r in rs]
        rag = sum(m["rag_ok"] for m in ref)
        old = sum(m["ours_ok"] for m in ref)
        a = sum(r["a_ok"] for r in rs)
        g = sum(gated(r)[0] for r in rs)
        line(label, n, f"{rag / n * 100:.1f}%", f"{old / n * 100:.1f}%",
             f"{a / n * 100:.1f}%", f"{g / n * 100:.1f}%",
             f"{(g - rag) / n * 100:+.1f}")

    rag_block(paired, "ALL")
    print()
    for cat in sorted({r["category"] for r in paired}):
        rag_block([r for r in paired if r["category"] == cat], cat)
    print()
    typed = [r for r in paired if r["tag"]]
    rag_block(typed, "typed only")

    print("\n  Context, mean tokens per question:")
    ref = [main_run[(r["conv"], r["question"])] for r in paired]
    print(f"    RAG {sum(m['rag_chars'] for m in ref) / len(ref) / 4:.0f}"
          f"   gated "
          f"{sum(gated(r)[1] for r in paired) / len(paired) / 4:.0f}")


if __name__ == "__main__":
    main()
