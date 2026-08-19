"""Produce the LoCoMo numbers the README quotes, from the committed results JSON.

Every figure in the README's results section comes out of this script, so a
reader can regenerate the table rather than trust it. Point it at one or two
run directories; with two, it also runs the paired McNemar test between the
matching arms.

The one judgement call it encodes is which questions count. LoCoMo ships 1,986
QA pairs in five categories, and the fifth -- 'adversarial' -- is excluded by
every published comparison (Mem0, Zep, MemMachine and others all report on
1,540). The category was meant to hold unanswerable questions, but the released
gold answers are ordinary facts: 444 of the 446 have a real answer rather than
'not mentioned', so a system that correctly declines is marked wrong. Scoring it
measures the grader's defect, not the system's memory.

Both denominators are printed. The 4-category figure is the one comparable to
published work; the 5-category figure is kept so the exclusion is visible rather
than quietly applied.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SCORED = ("single-hop", "temporal", "multi-hop", "open-domain")
EXCLUDED = "adversarial"


def load(run: Path) -> dict[str, list[dict]]:
    """Question-level rows per arm."""
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    return {arm: entry["question_results"] for arm, entry in data["results"].items()}


def key(row: dict) -> tuple:
    """Identity of a question, for pairing rows across two runs.

    Pairs on the question text plus its category rather than on position: the
    arms do not necessarily emit rows in the same order, and a positional pair
    would silently compare different questions.
    """
    return (row.get("conversation_id"), row["category"], row["question"])


def summarise(rows: list[dict]) -> dict:
    out = {"all": [0, 0], "scored": [0, 0], "by_cat": defaultdict(lambda: [0, 0])}
    f1_scored, ctx, lat = [], [], []
    for r in rows:
        correct = bool(r["judge_correct"])
        cat = r["category"]
        out["all"][0] += 1
        out["all"][1] += correct
        out["by_cat"][cat][0] += 1
        out["by_cat"][cat][1] += correct
        if cat in SCORED:
            out["scored"][0] += 1
            out["scored"][1] += correct
            if r.get("f1") is not None:
                f1_scored.append(r["f1"])
            if r.get("context_tokens"):
                ctx.append(r["context_tokens"])
            if r.get("latency_ms"):
                lat.append(r["latency_ms"])
    out["f1"] = sum(f1_scored) / len(f1_scored) if f1_scored else None
    out["ctx"] = sum(ctx) / len(ctx) if ctx else None
    out["lat"] = sum(lat) / len(lat) if lat else None
    return out


def mcnemar(before: list[dict], after: list[dict]) -> tuple[int, int, float] | None:
    """Paired exact test over questions present in both runs, scored categories only."""
    try:
        from scipy.stats import binomtest
    except ImportError:
        return None
    b = {key(r): bool(r["judge_correct"]) for r in before if r["category"] in SCORED}
    a = {key(r): bool(r["judge_correct"]) for r in after if r["category"] in SCORED}
    shared = b.keys() & a.keys()
    gained = sum(1 for k in shared if a[k] and not b[k])
    lost = sum(1 for k in shared if b[k] and not a[k])
    if gained + lost == 0:
        return gained, lost, 1.0
    return gained, lost, binomtest(gained, gained + lost, 0.5).pvalue


def table(title: str, per_arm: dict[str, dict]) -> None:
    print(f"\n{title}")
    print(f"  {'arm':<22}{'5-cat':>9}{'4-cat':>9}{'F1':>8}{'ctx tok':>10}{'ms':>9}")
    for arm, s in per_arm.items():
        acc5 = s["all"][1] / s["all"][0]
        acc4 = s["scored"][1] / s["scored"][0]
        f1 = f"{s['f1']:.4f}" if s["f1"] is not None else "-"
        ctx = f"{s['ctx']:,.0f}" if s["ctx"] else "-"
        lat = f"{s['lat']:,.0f}" if s["lat"] else "-"
        print(f"  {arm:<22}{acc5:>9.4f}{acc4:>9.4f}{f1:>8}{ctx:>10}{lat:>9}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory holding results.json")
    ap.add_argument("--baseline", help="earlier run to compare against (paired)")
    args = ap.parse_args()

    run = load(Path(args.run))
    per_arm = {arm: summarise(rows) for arm, rows in run.items()}

    print("=" * 72)
    print(f"LoCoMo summary: {args.run}")
    print("=" * 72)
    table("accuracy (F1/context/latency over the 4 scored categories)", per_arm)

    any_arm = next(iter(per_arm.values()))
    n5, n4 = any_arm["all"][0], any_arm["scored"][0]
    print(f"\n  denominators: {n5} all-category, {n4} scored "
          f"({EXCLUDED} excluded, n={n5 - n4})")

    print("\n  per-category accuracy")
    cats = list(SCORED) + [EXCLUDED]
    print(f"  {'arm':<22}" + "".join(f"{c:>14}" for c in cats))
    for arm, s in per_arm.items():
        cells = ""
        for c in cats:
            n, k = s["by_cat"].get(c, [0, 0])
            cells += f"{(k / n if n else 0):>10.3f}({n:>3})".rjust(14)
        print(f"  {arm:<22}{cells}")

    if args.baseline:
        base = load(Path(args.baseline))
        print(f"\n  paired McNemar vs {args.baseline} (scored categories only)")
        for arm in per_arm:
            if arm not in base:
                continue
            res = mcnemar(base[arm], run[arm])
            if res is None:
                print(f"    {arm:<22} scipy not installed")
                continue
            gained, lost, p = res
            b4 = summarise(base[arm])
            acc_b = b4["scored"][1] / b4["scored"][0]
            acc_a = per_arm[arm]["scored"][1] / per_arm[arm]["scored"][0]
            print(f"    {arm:<22}{acc_b:.4f} -> {acc_a:.4f}  "
                  f"(+{gained}/-{lost}, p={p:.4g})")


if __name__ == "__main__":
    main()
