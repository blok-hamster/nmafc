"""Can the relative-date failures be repaired without re-ingesting?

_error_taxonomy.py found 30 wrong answers where the gold answer dates something
relative to another date ("the Sunday before 25 May 2023") and the arm answered
with the resolved absolute date. Extraction resolved the reference and threw the
speaker's phrasing away, so the store no longer holds the form the question is
scored against.

Re-ingesting fixes this -- the extractor now keeps both -- but costs one LLM call
per exchange across every conversation. This asks whether a cheaper repair is
available: the archive records a `turn` for every fact, and `build_exchanges`
derives exchanges from the transcript deterministically, so a fact can be traced
back to the text it came from with no model calls at all.

The question this answers is narrow and factual: for each relative-date failure,
does the source transcript still contain the phrasing the gold answer wants?

    IN TRANSCRIPT    a mechanical repair can recover it -- copy the phrase back
    NOT IN TRANSCRIPT the phrasing is the annotator's, not the speaker's, and
                     no amount of re-reading the source will produce it

Only the first group is repairable offline. The second is a scoring artefact and
re-ingestion will not fix it either, which is worth knowing before paying for a
re-ingestion on its account.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ia", HERE / "_ingest_audit.py")
ia = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ia)

spec2 = importlib.util.spec_from_file_location("et", HERE / "_error_taxonomy.py")
et = importlib.util.module_from_spec(spec2)
assert spec2.loader is not None
spec2.loader.exec_module(et)

RELATIVE_CLASS = "date: gold is relative, answer gave an absolute date"

# The anchor phrase inside a relative gold answer: the part that has to survive
# extraction for the answer to be reproducible ("the sunday before", "the week
# after"). Matched against the transcript rather than against the store, because
# the store is what we are asking whether to rebuild.
ANCHOR = re.compile(
    r"\b((?:the\s+)?(?:day|week|weekend|month|night|evening|morning)?\s*"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)?\s*"
    r"(?:before|after|prior to|following|preceding|ago|earlier|later))\b",
    re.IGNORECASE,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    base = HERE / "results" / args.run
    convs = {c["sample_id"]: c
             for c in json.load(open(ia.locomo_path(), encoding="utf-8"))}

    # question -> transcript it came from, so a failure can be checked against
    # its own source rather than against the whole corpus.
    source: dict[str, str] = {}
    for conv in convs.values():
        text = ia.norm(ia.session_text(conv))
        for qa in conv.get("qa", []):
            source[ia.norm(str(qa.get("question", "")))] = text

    rows = json.load(open(base / "results.json", encoding="utf-8")
                     )["results"][args.arm]["question_results"]

    verdict: Counter[str] = Counter()
    shown: list[tuple[str, str, str, str, bool]] = []

    for r in rows:
        if r.get("category") == "adversarial" or r.get("judge_correct"):
            continue
        gold, pred = str(r.get("gold_answer", "")), str(r.get("predicted", ""))
        if et.classify(gold, pred) != RELATIVE_CLASS:
            continue

        text = source.get(ia.norm(str(r.get("question", ""))), "")
        anchors = [a.strip() for a in ANCHOR.findall(gold) if a.strip()]
        # A gold answer can carry more than one anchor; recoverable means every
        # one of them is still findable in the speakers' own words.
        recoverable = bool(anchors) and all(
            ia.norm(a) in text for a in anchors)

        verdict["recoverable from transcript" if recoverable
                else "phrasing is the annotator's, not in transcript"] += 1
        if len(shown) < args.examples:
            shown.append((str(r.get("question", "")), gold, pred,
                          "; ".join(anchors) or "(no anchor phrase)", recoverable))

    total = sum(verdict.values())
    print("run=%s arm=%s  relative-date failures=%d\n" % (args.run, args.arm, total))
    if not total:
        print("  none found")
        return 0

    print("=" * 76)
    print("CAN THE PHRASING BE RECOVERED WITHOUT RE-INGESTING?")
    print("=" * 76)
    for key, n in verdict.most_common():
        print("  %-52s %4d %5.1f%%" % (key, n, 100.0 * n / total))

    print("\n  examples")
    for q, gold, pred, anchor, ok in shown:
        print("\n   [%s] %s" % ("RECOVERABLE" if ok else "not in source", q[:56]))
        print("      gold   : %s" % gold[:70])
        print("      said   : %s" % pred[:70])
        print("      anchor : %s" % anchor[:70])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
