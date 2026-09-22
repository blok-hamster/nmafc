"""How many LoCoMo questions ask about a picture no text pipeline ever sees?

LoCoMo sessions carry image captions ("shared a photo of a golden retriever"),
and some questions are asked directly about what is *in* those images. Every arm
here reads text alone, so a question whose answer lives in the picture is
unanswerable by construction -- not a retrieval failure, not an extraction
failure, and not something more tuning can recover.

That matters before deciding what to do about the largest wrong-answer bucket
("gave a different answer", 230 of 565). If a meaningful slice of it is image
questions, the reachable ceiling is lower than the raw count suggests and effort
spent chasing those cases is spent for nothing.

Counts two things separately, because they answer different questions:

  - questions whose wording refers to an image ("in the photo", "the picture
    shows"), which is the population a text arm cannot answer; and
  - of those, how many the arms actually got wrong, which is the part of the
    wrong-answer bucket that is explained rather than merely correlated.

Reads the dataset and a results file. Makes no model calls and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Wording that points at a shared image rather than at something said. Kept
# narrow deliberately: "see" and "look" are everyday conversational verbs and
# matching them would sweep in ordinary questions, inflating the count in the
# direction that flatters the arms.
IMAGE_PHRASES = re.compile(
    r"(?i)\b("
    r"in (?:the|this|that|her|his|their) (?:photo|picture|image|pic|selfie|snapshot)"
    r"|(?:photo|picture|image|pic|selfie|snapshot)s? (?:that |which )?(?:she|he|they|"
    r"\w+) (?:shared|posted|sent|took|uploaded)"
    r"|(?:the|a) (?:photo|picture|image|pic|selfie|snapshot) (?:of|shows|showed|"
    r"depicts|depicted|features)"
    r"|shown in the (?:photo|picture|image)"
    r"|what (?:is|are|was|were) .{0,40}(?:wearing|holding|doing) in"
    r"|(?:dancers|people|person|man|woman|figures?) in the"
    r")\b"
)


def load_questions(dataset: Path) -> dict[str, str]:
    """Map question text to its category name, across every conversation."""
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for conv in raw:
        for qa in conv.get("qa", []):
            question = qa.get("question")
            if question:
                out[question] = str(qa.get("category", "?"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Path to a results.json")
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional LoCoMo json, only used to report the dataset-wide rate",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    arms = payload.get("results", {})

    print(f"results: {args.results}\n")

    if args.dataset:
        questions = load_questions(Path(args.dataset))
        hits = [q for q in questions if IMAGE_PHRASES.search(q)]
        pct = 100 * len(hits) / max(1, len(questions))
        print(f"dataset: {len(hits)} of {len(questions)} questions "
              f"({pct:.1f}%) refer to an image")
        by_cat = Counter(questions[q] for q in hits)
        for cat, n in by_cat.most_common():
            print(f"    category {cat}: {n}")
        print()

    for arm_name, arm in sorted(arms.items()):
        rows = arm.get("question_results") or arm.get("results") or []
        if not rows:
            continue

        image_rows = [r for r in rows if IMAGE_PHRASES.search(r.get("question", ""))]
        # `is False` rather than falsy: judge_correct=None means unjudged, which
        # is not the same claim as "the judge marked this wrong".
        wrong = [r for r in rows if r.get("judge_correct") is False]
        image_wrong = [r for r in image_rows if r.get("judge_correct") is False]
        image_right = [r for r in image_rows if r.get("judge_correct") is True]

        print(f"{arm_name}:")
        print(f"    questions               : {len(rows)}")
        print(f"    image-referring         : {len(image_rows)} "
              f"({100 * len(image_rows) / max(1, len(rows)):.1f}%)")
        print(f"    of those, judged wrong  : {len(image_wrong)}")
        print(f"    of those, judged right  : {len(image_right)}")
        if wrong:
            print(f"    share of all wrong answers explained by images: "
                  f"{100 * len(image_wrong) / len(wrong):.1f}% "
                  f"({len(image_wrong)} of {len(wrong)})")
        print()

    if image_rows := [r for arm in arms.values()
                      for r in (arm.get("question_results") or [])
                      if IMAGE_PHRASES.search(r.get("question", ""))]:
        print("sample of matched questions:")
        seen = set()
        for row in image_rows:
            q = row["question"]
            if q in seen:
                continue
            seen.add(q)
            print(f"  - {q}")
            print(f"      gold: {row.get('gold_answer')!r}")
            if len(seen) >= 12:
                break


if __name__ == "__main__":
    main()
