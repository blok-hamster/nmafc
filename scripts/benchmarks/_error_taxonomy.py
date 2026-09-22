"""What do the wrong answers actually look like?

_retrieval_funnel.py showed that at least 23.7% of retrieved facts still produce
a wrong answer, which makes the answering step the cheapest remaining lever --
it costs no extra context, unlike raising top-k. But "fix the answer prompt" is
not an instruction until you know what the failures are.

This classifies every judged-wrong prediction against its gold answer, using
only results.json. No model calls, no store access, no embeddings.

The classes are ordered most- to least-specific and each prediction takes the
first that matches, so the counts partition cleanly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

REFUSAL = re.compile(
    r"\b(i (don'?t|do not) know|no information|not available|cannot determine"
    r"|can'?t determine|not (?:enough|sufficient) (?:information|context)"
    r"|no relevant|unable to (?:determine|answer)|not (?:mentioned|specified|stated)"
    r"|nothing in the (?:facts|context))\b",
    re.IGNORECASE,
)
PREAMBLE = re.compile(
    r"^(based on|according to|the facts (?:show|indicate)|looking at|from the"
    r"|it (?:appears|seems)|the (?:answer|context) )",
    re.IGNORECASE,
)
YESNO = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
YEAR = re.compile(r"\b(19|20)\d{2}\b")
MONTH = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.IGNORECASE)
RELATIVE = re.compile(
    r"\b(before|after|ago|last|next|following|previous|prior|earlier|later"
    r"|week|weekend|sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b",
    re.IGNORECASE,
)
STOP = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or",
        "is", "was", "were", "are", "be", "with", "by", "from", "that", "her",
        "his", "she", "he", "they", "it", "its", "their", "them", "as"}


def words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split()
            if w and w not in STOP]


def classify(gold: str, pred: str) -> str:
    g, p = words(gold), words(pred)
    gs, ps = set(g), set(p)

    if not p:
        return "empty answer"
    if REFUSAL.search(pred or ""):
        return "refused / said it did not know"

    overlap = len(gs & ps) / len(gs) if gs else 0.0

    # Date questions: is the content right but the form wrong?
    gold_dated = bool(YEAR.search(gold) or MONTH.search(gold))
    pred_dated = bool(YEAR.search(pred) or MONTH.search(pred))
    if gold_dated and pred_dated:
        gy = set(YEAR.findall(gold)) == set(YEAR.findall(pred))
        gm = {m.lower()[:3] for m in MONTH.findall(gold)} == \
             {m.lower()[:3] for m in MONTH.findall(pred)}
        if RELATIVE.search(gold) and not RELATIVE.search(pred):
            return "date: gold is relative, answer gave an absolute date"
        if gy and gm:
            return "date: same month and year, judged wrong anyway"
        if gy and not gm:
            return "date: right year, wrong or missing month"
        return "date: wrong date"
    if gold_dated and not pred_dated:
        return "date question answered without a date"

    if YESNO.match(gold) and YESNO.match(pred):
        if YESNO.match(gold).group(1).lower() != YESNO.match(pred).group(1).lower():
            return "yes/no: opposite answer"
        return "yes/no: agreed, but the reason was judged wrong"
    if YESNO.match(gold) and not YESNO.match(pred):
        return "yes/no question not answered yes or no"

    # List answers: gold enumerates several items.
    gold_list = len(re.split(r",| and ", gold)) >= 3
    if gold_list:
        if overlap >= 0.99:
            return "list: all items present, judged wrong anyway"
        if overlap >= 0.5:
            return "list: partially complete (half or more of the items)"
        return "list: mostly missing items"

    if PREAMBLE.match((pred or "").strip()):
        return "preamble instead of a bare answer"
    if overlap >= 0.99:
        return "all gold words present, judged wrong anyway"
    if overlap >= 0.5:
        return "partly right (half or more of gold words)"
    if len(p) > 25:
        return "long answer, little overlap"
    return "different answer"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    path = HERE / "results" / args.run / "results.json"
    rows = json.load(open(path, encoding="utf-8"))["results"][args.arm]["question_results"]

    kinds: Counter[str] = Counter()
    percat: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    wrong = 0
    lengths: list[int] = []

    for r in rows:
        if r.get("category") == "adversarial":
            continue
        gold, pred = str(r.get("gold_answer", "")), str(r.get("predicted", ""))
        lengths.append(len(words(pred)))
        if r.get("judge_correct"):
            continue
        wrong += 1
        kind = classify(gold, pred)
        kinds[kind] += 1
        percat[r.get("category", "?")][kind] += 1
        if len(samples[kind]) < args.examples:
            samples[kind].append((str(r.get("question", "")), gold, pred))

    print("run=%s arm=%s  wrong answers=%d  (adversarial excluded)\n"
          % (args.run, args.arm, wrong))
    print("=" * 76)
    print("WHAT THE WRONG ANSWERS LOOK LIKE")
    print("=" * 76)
    for kind, n in kinds.most_common():
        print("  %-52s %5d %5.1f%%" % (kind, n, 100.0 * n / wrong))

    if lengths:
        over = sum(1 for n in lengths if n > 5)
        print("\n  answer length: median %d words, %.1f%% exceed the 5-word cap"
              % (sorted(lengths)[len(lengths) // 2], 100.0 * over / len(lengths)))

    print("\n  dominant failure per category")
    for cat in sorted(percat):
        top = percat[cat].most_common(2)
        total = sum(percat[cat].values())
        print("    %-13s %s" % (cat, "; ".join(
            "%s (%.0f%%)" % (k, 100.0 * v / total) for k, v in top)))

    print("\n" + "=" * 76)
    print("EXAMPLES")
    print("=" * 76)
    for kind, _ in kinds.most_common(6):
        print("\n  -- %s" % kind)
        for q, gold, pred in samples[kind]:
            print("     Q    : %s" % q[:68])
            print("     gold : %s" % gold[:68])
            print("     said : %s" % pred[:68])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
