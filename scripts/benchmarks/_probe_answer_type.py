"""How often does the answer name the wrong *kind* of thing?

Reading the multi-hop failures turned up a repeated shape that no accuracy
number exposes. Asked "in which state is the shelter", we answered "Stamford";
asked "in what country did she buy the pendant", we answered "Paris"; asked
"what console does he own", we answered "Nintendo". Each time the evidence was
in hand and the answer stopped one step short of the category the question
named. RAG made the same three mistakes, which is why they sit in the bucket
both arms fail rather than in the contested one.

The cause is visible in the prompt. `SHORT_ANSWER_RULES` says to reuse the
wording of the facts because a synonym scores as a miss, and that is correct for
almost every question -- it is worth about half the open-domain F1. On a
question that names the category it wants, though, it is precisely wrong:
"Connecticut" is not a synonym for "Stamford", it is the answer, and "Stamford"
is the evidence for it.

So this measures the class rather than arguing about it. A question is
*type-demanding* when its wh-phrase names the kind of answer wanted -- a state,
a country, a console, an age. For those, accuracy is reported separately from
the rest, for both arms.

Two ways of reading the output, and only one of them justifies building
anything:

  * If type-demanding questions score near the category average, there is no
    class here, only three anecdotes, and the right move is to leave the prompt
    alone.
  * If they score materially below it *for both arms*, the shortfall is caused
    by something both prompts share, which is the copy-the-wording rule, and a
    per-question hint naming the type wanted is a mechanism rather than a knob.

Deliberately not measured here: whether our answer *is* of the wrong type.
Deciding that "Stamford" is a city and not a state needs a gazetteer, and a
detector that needs world knowledge to find failures caused by missing world
knowledge would be measuring its own vocabulary. Accuracy against gold needs
none.

Free. Reads the saved paired run, generates nothing, calls no API.

Usage:
    python -u scripts/benchmarks/_probe_answer_type.py
    python -u scripts/benchmarks/_probe_answer_type.py --show
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Categories a question can name outright. Kept to nouns whose *level* is
# unambiguous, because the whole point is that the question fixes a level the
# evidence does not. "place" and "thing" are excluded for that reason: they
# demand nothing, so a question containing them is not type-demanding.
TYPE_NOUNS = (
    r"state|country|city|town|province|county|region|continent|nation|"
    r"month|year|day|date|weekday|season|decade|"
    r"console|device|brand|model|make|company|team|band|"
    r"language|nationality|religion|"
    r"breed|species|genre|colour|color|"
    r"job|profession|occupation|career|degree|major|field|subject|"
    r"sport|instrument|dish|cuisine|drink|meal|"
    r"book|movie|film|show|song|album|game|app"
)

# "which US state", "in what country", "what kind of job". The optional words
# between the wh and the noun carry the qualifier ("US", "outdoor") which is
# part of what makes the demand specific, so they are captured with the noun.
NAMED_TYPE = re.compile(
    rf"\b(?:which|what)\s+(?:kind\s+of\s+|sort\s+of\s+|type\s+of\s+)?"
    rf"(?:\w+\s+){{0,2}}({TYPE_NOUNS})s?\b",
    re.I,
)

# Wh-phrases that name a type without naming a noun. "How old" wants an age and
# nothing else; "how many" wants a number. These are as type-demanding as the
# nouns above and would be missed by a noun list.
IMPLIED_TYPE = (
    (re.compile(r"\bhow\s+old\b", re.I), "an age"),
    (re.compile(r"\bhow\s+(?:many|much)\b", re.I), "a number"),
    (re.compile(r"\bhow\s+(?:often|frequently)\b", re.I), "a frequency"),
    (re.compile(r"\bhow\s+long\b", re.I), "a duration"),
)


def demanded_type(question: str) -> str | None:
    """The kind of answer this question names, or None if it names none."""
    match = NAMED_TYPE.search(question)
    if match:
        return match.group(0).split(None, 1)[1].strip().lower()
    for pattern, label in IMPLIED_TYPE:
        if pattern.search(question):
            return label
    return None


def rate(rows: list[dict], key: str) -> float:
    return sum(bool(r[key]) for r in rows) / len(rows) * 100 if rows else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack.json")
    ap.add_argument("--show", action="store_true",
                    help="Print the type-demanding questions both arms failed.")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    rows = json.loads(Path(args.run).read_text(encoding="utf-8"))
    for r in rows:
        r["demand"] = demanded_type(r["question"])

    print(f"{'category':<12}{'n':>5}  {'--- type-demanding ---':^28}  "
          f"{'--- the rest ---':^22}")
    print(f"{'':<12}{'':>5}  {'n':>4}{'ours':>8}{'RAG':>8}{'diff':>8}  "
          f"{'n':>4}{'ours':>8}{'RAG':>8}")
    print("-" * 78)

    for cat in sorted({r["category"] for r in rows}) + ["ALL"]:
        rs = rows if cat == "ALL" else [r for r in rows if r["category"] == cat]
        typed = [r for r in rs if r["demand"]]
        rest = [r for r in rs if not r["demand"]]
        if not typed:
            continue
        line = (f"{cat:<12}{len(rs):>5}  {len(typed):>4}"
                f"{rate(typed, 'ours_ok'):>7.1f}%{rate(typed, 'rag_ok'):>7.1f}%"
                f"{rate(typed, 'ours_ok') - rate(typed, 'rag_ok'):>+8.1f}  "
                f"{len(rest):>4}{rate(rest, 'ours_ok'):>7.1f}%"
                f"{rate(rest, 'rag_ok'):>7.1f}%")
        print(("-" * 78 + "\n" + line) if cat == "ALL" else line)

    typed = [r for r in rows if r["demand"]]
    both = [r for r in typed if not r["ours_ok"] and not r["rag_ok"]]
    print(f"\ntype-demanding questions both arms fail: {len(both)} "
          f"of {len(typed)} ({len(both) / len(typed) * 100:.1f}%)")
    rest_both = [r for r in rows if not r["demand"]
                 and not r["ours_ok"] and not r["rag_ok"]]
    print(f"the rest, for comparison:               {len(rest_both)} "
          f"of {len(rows) - len(typed)} "
          f"({len(rest_both) / (len(rows) - len(typed)) * 100:.1f}%)")

    counts: dict[str, int] = {}
    for r in both:
        counts[r["demand"]] = counts.get(r["demand"], 0) + 1
    print("\nmost-failed demanded types:")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {n:>3}  {name}")

    if args.show:
        print("\n" + "=" * 78)
        for i, r in enumerate(both[: args.limit], 1):
            print(f"\n[{i}] {r['category']}  wants: {r['demand']}")
            print(f"  Q     {r['question']}")
            print(f"  gold  {r['gold']}")
            print(f"  ours  {r['ours_pred']}")
            print(f"  rag   {r['rag_pred']}")


if __name__ == "__main__":
    main()
