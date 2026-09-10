"""How often is a losing answer a *piece* of the right answer rather than a wrong one?

Reading the 55 open-domain questions arm B still loses, one shape appears in both
halves of the diagnosis and in both halves of the rendered context:

    gold  They make him feel calm and don't require much looking after
    ours  they make him feel calm

    gold  Savana, Sleep
    ours  Savana

    gold  Max is 8 years old and Luna is 5 years old
    ours  5 years old

    gold  Sprucing up his business plan, tweaking his pitch to investors, and
          working on an online platform
    ours  sprucing up his business plan and tweaking his investor pitch

None of these is a retrieval failure in the usual sense. The answer was found and
then truncated, and the four earlier regressions that the keep-top-rank fix was
written for were the same shape. That matters because the two candidate fixes are
priced completely differently: another retrieval change costs tokens against a
1,000-token ceiling, and a completeness rule in the prompt costs about thirty
tokens once.

The shipped rules do say "Name EVERY item the facts support" -- but only inside
the bullet that begins "For lists: comma-separated nouns only". A gold like
"calm and low-maintenance" is not a list of nouns, so there is a real chance the
model never applies the rule to the answers that need it.

**What is counted.** A loss is `partial` when our answer's content words are a
proper subset of the gold's: everything we said was right, and we stopped early.
It is `disjoint` when we share no content word with the gold -- a different
answer, which no completeness rule reaches. `overlapping` is the rest.

**And the risk, on the same footing.** The wins are counted the same way, for
the cell that a completeness rule endangers: wins where our answer is already a
superset of the gold. Those are wins carrying extra material the judge tolerated,
and telling the model to say more puts them nearest the edge.

Pure text analysis of the saved paired run. No store, no model, no spend.

Usage:
    python -u scripts/benchmarks/_screen_partial_answers.py --show 14
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.benchmarks._screen_hydrate_pool import content_words  # noqa: E402

# Answers the model gives when it has nothing, which are not partial answers and
# would otherwise count as disjoint and inflate the bucket that says "unfixable".
_EMPTY = {"no information available", "no information", "unknown", "none"}


def shape(ours: str, gold: str) -> str:
    if ours.strip().lower().rstrip(".") in _EMPTY:
        return "refused"
    a, b = content_words(ours), content_words(gold)
    if not a or not b:
        return "disjoint"
    if a < b:
        return "partial"
    if b < a:
        return "over-complete"
    if a == b:
        return "same words"
    if a & b:
        return "overlapping"
    return "disjoint"


_SPECIFIC = re.compile(r"\b(?:[A-Z][a-z]{2,}|\d+)")


def names_something(text: str) -> set[str]:
    """Proper nouns and numbers: the words that make an answer specific.

    A crude stand-in for "the gold is a particular thing and ours is the
    category it belongs to", which is the other shape in the losses:

        gold  Hoodies          ours  clothing
        gold  Witcher 3        ours  childhood sketches
        gold  tree pose        ours  Dancer Pose

    Only the first and second are what this catches -- a capital or a digit is
    evidence of a particular thing, and lower-case common nouns like "hoodies"
    slip through unless the gold happens to capitalise them. So this
    under-counts, which is the right direction for a number that would be used
    to justify spending. It is a floor, not a rate.
    """
    return set(_SPECIFIC.findall(text)) - {"The", "They", "His", "Her", "She",
                                           "This", "That", "Yes", "None"}


def tally(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(shape(r[key], r["gold"]), []).append(r)
    return out


def show(title: str, groups: dict[str, list[dict]], total: int) -> None:
    print(f"\n  {title}  (n={total})")
    order = ["partial", "over-complete", "same words", "overlapping",
             "disjoint", "refused"]
    for name in order:
        rows = groups.get(name, [])
        if rows:
            print(f"    {name:<16}{len(rows):>5}{100 * len(rows) / total:>7.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="C:/nmafc_ab/open_domain.json")
    ap.add_argument("--arm", default="b", choices=["a", "b"])
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    key, ok = args.arm, f"{args.arm}_ok"
    lost = [r for r in rows if not r[ok]]
    won = [r for r in rows if r[ok]]

    print(f"\n{len(rows)} open-domain questions, arm {args.arm.upper()}: "
          f"{len(won)} right, {len(lost)} wrong")
    show("LOST -- what shape is the wrong answer?", tally(lost, key), len(lost))
    show("WON  -- what shape is the right one?", tally(won, key), len(won))

    partial = tally(lost, key).get("partial", [])
    over = tally(won, key).get("over-complete", [])
    n = len(rows)
    print(f"\n  Reachable by a completeness rule: {len(partial)} "
          f"(+{100 * len(partial) / n:.1f} points if every one converted)")
    print(f"  Wins already carrying extra words: {len(over)} "
          f"(-{100 * len(over) / n:.1f} if every one broke)")
    print("  Both are bounds nobody will hit. They price the paid run.")

    # The other shape: the gold names a particular thing and we named a class.
    # Counted on losses and on wins together, because a rule telling the model
    # to prefer the specific noun can only be worth buying if the wins are not
    # already living off the general one.
    vague = [r for r in lost if names_something(r["gold"]) - names_something(r[key])]
    safe = [r for r in won if names_something(r["gold"]) - names_something(r[key])]
    print(f"\n  Gold names something ours does not: {len(vague)} of "
          f"{len(lost)} losses, and {len(safe)} of {len(won)} wins.")
    print("  A floor, not a rate: it only sees capitals and digits, so a gold "
          "like\n  'hoodies' against ours of 'clothing' is invisible to it.")
    for r in vague[: args.show]:
        print(f"    gold {r['gold'][:52]:<54}ours {r[key][:40]}")

    print(f"\n{'=' * 70}\nLOST and partial -- we said part of the gold and stopped:")
    for r in partial[: args.show]:
        print(f"\n  Q     {r['question']}")
        print(f"  gold  {r['gold']}")
        print(f"  ours  {r[key][:100]}")

    dis = tally(lost, key).get("disjoint", []) + tally(lost, key).get(
        "overlapping", [])
    print(f"\n{'=' * 70}\n{len(dis)} LOST and not partial -- a completeness rule "
          f"cannot touch these:")
    for r in dis[: args.show]:
        print(f"\n  Q     {r['question']}")
        print(f"  gold  {r['gold']}")
        print(f"  ours  {r[key][:100]}")


if __name__ == "__main__":
    main()
