"""Would cutting the self-doubt off the end of an answer raise the score?

The answers in the tuned run are not uniformly verbose: the median is 4 words
against a gold median of 3, which is fine. The damage sits in a 27% tail with a
recognisable shape -- the model states the correct answer, then appends a caveat
disputing the question's own premise, almost always about a date:

    "James and Samantha decided to move in together. The facts show this
     decision was discussed and made, though the exact date of 31 October 2022
     isn't explicitly mentioned in the facts I have."

Token-overlap F1 charges for every one of those trailing words while the useful
answer sits in the first six. The existing preamble stripper does not catch it,
because the hedge arrives *after* the answer rather than before it.

This script measures the idea before anyone pays to test it. It replays a
finished results file, applies a trailing-hedge trim, and re-scores with the
same F1 the harness uses. Nothing is written and no model is called, so the
number it prints is the gain available from post-processing alone -- entirely
separate from any gain a reworded prompt might add on a rerun.

Judge verdicts are left exactly as they were. Trimming cannot change what a
judge already decided about text it already read, and pretending otherwise
would be inventing accuracy. F1 is the only figure this can honestly move.

Usage:
    python scripts/benchmarks/_trim_hedges.py --run scripts/benchmarks/results/full_v3
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

# Phrases that open a retraction. Each is matched only when it follows at least
# one sentence of actual answer, so a reply that legitimately begins "However"
# is untouched -- the goal is to drop the second thought, never the first.
#
# Deliberately narrow. "but" alone would decapitate "No, but she wants to",
# where the clause after it carries the answer, so only the hedging collocations
# ("but there is no", "but the exact") are listed.
HEDGE_ONSET = re.compile(
    r"(?i)[\s,.;—-]*\b(?:"
    r"but (?:the |there |this |that |it |i |no |neither |none )"
    r"|though (?:the |there |this |that |it |no )"
    r"|although (?:the |there |this |that |it )"
    r"|however\b"
    r"|the facts? (?:show|state|mention|indicate|suggest|say|do not|don't|only)"
    r"|there (?:is|are|was|were) no (?:specific |explicit |exact )?"
    r"(?:mention|record|information|detail|reference)"
    r"|no (?:specific |explicit |exact )?(?:mention|record|information) (?:of|is)"
    r"|(?:the )?exact date (?:is|isn'?t|was)"
    r"|i (?:do not|don'?t) have"
    r"|based on (?:the|my) (?:facts|memories)"
    r"|\(valid"
    r")"
)


def trim(answer: str) -> str:
    """Cut an answer at the point it starts disputing the question.

    Returns the original when the hedge starts at position zero: an answer that
    is nothing but a caveat has no confident half to keep, and truncating it to
    empty would score zero where the untrimmed text might still overlap.
    """
    text = answer.strip()
    match = HEDGE_ONSET.search(text)
    if not match or match.start() == 0:
        return text
    kept = text[: match.start()].strip(" ,;.—-")
    return kept or text


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluation.f1_score import compute_f1

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory holding results.json")
    ap.add_argument("--arm", default="", help="limit to one arm")
    ap.add_argument("--show", type=int, default=8, help="example trims to print")
    args = ap.parse_args()

    payload = json.loads(
        (Path(args.run) / "results.json").read_text(encoding="utf-8")
    )

    for arm_name, arm in sorted(payload.get("results", {}).items()):
        if args.arm and arm_name != args.arm:
            continue
        rows = arm.get("question_results") or arm.get("results") or []
        # The adversarial category is excluded from scored accuracy elsewhere in
        # this harness, so including it here would measure a gain on questions
        # the reported figure never counts.
        rows = [r for r in rows if r.get("category") != "adversarial"]
        if not rows:
            continue

        before, after, changed = [], [], []
        for row in rows:
            original = str(row.get("predicted", ""))
            trimmed = trim(original)
            old = float(row.get("f1") or 0.0)
            new = compute_f1(trimmed, str(row.get("gold_answer", "")))
            before.append(old)
            after.append(new)
            if trimmed != original:
                changed.append((new - old, row, original, trimmed))

        words_before = st.mean(len(str(r["predicted"]).split()) for r in rows)
        words_after = st.mean(len(trim(str(r["predicted"])).split()) for r in rows)

        print(f"\n{arm_name}  (n={len(rows)} scored)")
        print(f"  answers trimmed : {len(changed)} "
              f"({100 * len(changed) / len(rows):.0f}%)")
        print(f"  mean words      : {words_before:.1f} -> {words_after:.1f}")
        print(f"  mean F1         : {st.mean(before):.4f} -> {st.mean(after):.4f} "
              f"({st.mean(after) - st.mean(before):+.4f})")
        helped = sum(1 for d, *_ in changed if d > 0.001)
        hurt = sum(1 for d, *_ in changed if d < -0.001)
        print(f"  of those trimmed: {helped} improved, {hurt} worsened, "
              f"{len(changed) - helped - hurt} unchanged")

        if args.show and changed:
            print("\n  largest gains:")
            for delta, row, original, trimmed in sorted(changed, reverse=True,
                                                        key=lambda c: c[0])[:args.show]:
                print(f"    +{delta:.3f}  gold={row['gold_answer']!r}")
                print(f"       was: {original[:120]!r}")
                print(f"       now: {trimmed[:120]!r}")
            worst = sorted(changed, key=lambda c: c[0])[:3]
            if worst and worst[0][0] < -0.001:
                print("\n  worst regressions:")
                for delta, row, original, trimmed in worst:
                    if delta >= -0.001:
                        continue
                    print(f"    {delta:.3f}  gold={row['gold_answer']!r}")
                    print(f"       was: {original[:120]!r}")
                    print(f"       now: {trimmed[:120]!r}")


if __name__ == "__main__":
    main()
