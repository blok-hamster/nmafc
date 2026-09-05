"""When the answer was already in the prompt and we still got it wrong, why?

The verbatim probe found that the gold answer is present in the context we send
on 81.4% of questions while the run scores 64%, and that on the 233 questions
RAG wins and we lose the answer was in our own prompt 70.8% of the time. Across
every question we fail, roughly two thirds had the evidence in front of the
model. Nothing about storage, hydration or reranking touches those.

That leaves three possible explanations, and they call for opposite fixes:

  refused    -- the model declined to answer despite holding the evidence. A
                prompt problem. The answer prompt already forbids "No
                information available", so any of these are it being overridden.
  near miss  -- the answer carries the gold content and was judged wrong
                anyway: wrong format, too much hedging around it, or a strict
                judge. A formatting problem, and cheap to fix.
  confident  -- a specific, wrong answer. The model was misled by something
                else in the context. A precision problem, fixed by ranking
                fewer and better facts, not by retrieving more.

This classifies them, because the three have nothing in common except the
symptom, and guessing which one dominates is how the last several days were
spent.

Retrieval only, reading the run's own stores with reinforcement deferred and
never flushed. Predictions are read from the run's results file rather than
regenerated, so no answers are produced and no judge is called.

Usage:
    python -u scripts/benchmarks/_probe_answered_wrong.py --limit 20
    python -u scripts/benchmarks/_probe_answered_wrong.py --out /c/nmafc_ab/wrong.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks._probe_verbatim import open_memory  # noqa: E402
from scripts.benchmarks._sweep_context_budget import (  # noqa: E402
    overlap,
    retrieve_with_retry,
)

SCORED = ("single-hop", "temporal", "multi-hop", "open-domain")

# Phrasings the answer prompt explicitly forbids, plus the ways a model says the
# same thing when it ignores that instruction.
REFUSAL = re.compile(
    r"no (information|record|mention|details|data|specific|date|exact|further)"
    r"|not (mentioned|specified|available|provided|stated|indicated|clear|given)"
    r"|does not (say|mention|specify|state|provide|indicate|give)"
    r"|doesn't (say|mention|specify|state|give)"
    r"|(is|are|was|were) not given|no .{0,20} (is|are|was|were) given"
    r"|cannot (be )?(determine|answer|tell)|can't (determine|tell)"
    r"|unable to (determine|answer)|unknown|unclear|insufficient",
    re.IGNORECASE,
)

# The loose presence test drops tokens of two characters or fewer, which is
# harmless for prose and wrong for dates: the gold "7 May 2023" becomes
# {may, 2023}, so a fact reading "6 May 2023" is scored as carrying it. Since
# 321 of the 1,540 questions are temporal and many golds are three words long,
# that inflates every presence figure.
#
# Strict keeps every token, numerals included, and demands all of them as whole
# words. It is the opposite bias, failing on any paraphrase, so the two together
# bracket the truth rather than either standing alone.
STRICT_STOP = set(
    "the a an of in on at to for is was were and or with what which who when "
    "where how did does do had has have her his its their this that".split()
)


def strict_present(gold: str, haystack: str) -> bool:
    tokens = [w for w in re.findall(r"[a-z0-9']+", str(gold).lower())
              if w not in STRICT_STOP]
    if not tokens:
        return False
    low = haystack.lower()
    return all(
        re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", low) for w in tokens
    )


def load_run(results_path: Path, arm: str | None = None) -> dict[str, dict]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))["results"]
    key = arm or next(iter(payload))
    return {row["question"]: row for row in payload[key]["question_results"]}


def classify(gold: str, predicted: str, near: float) -> str:
    if not predicted.strip():
        return "empty"
    if REFUSAL.search(predicted):
        return "refused"
    if overlap(gold, predicted) >= near:
        return "near miss"
    return "confident"


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    ours = load_run(Path(args.run) / "results.json", args.arm)

    buckets: dict[str, int] = defaultdict(int)
    strict_buckets: dict[str, int] = defaultdict(int)
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lengths: dict[str, list[int]] = defaultdict(list)
    samples: dict[str, list[dict]] = defaultdict(list)
    wrong_total = 0
    evidence_absent = 0
    strict_absent = 0

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            continue

        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED]
        if args.limit:
            questions = questions[: args.limit]

        memory = open_memory(store, llm, embedder, args)
        try:
            turn = memory.current_turn + 1
            for qa in questions:
                row = ours.get(qa.question)
                if row is None or row["judge_correct"]:
                    continue
                wrong_total += 1

                records = await retrieve_with_retry(memory._router, qa.question, turn)
                context = memory._router.format_context(records[: args.budget])
                strict = strict_present(qa.answer, context)
                if not strict:
                    strict_absent += 1
                if overlap(qa.answer, context) < args.threshold:
                    # Genuinely not retrieved. A different problem, counted so
                    # the two failure families can be sized against each other.
                    evidence_absent += 1
                    continue

                predicted = str(row.get("predicted", ""))
                bucket = classify(qa.answer, predicted, args.near)
                buckets[bucket] += 1
                if strict:
                    strict_buckets[bucket] += 1
                by_category[qa.category_name][bucket] += 1
                lengths[bucket].append(len(predicted.split()))

                if len(samples[bucket]) < args.examples:
                    carrying = [
                        line for line in context.splitlines()
                        if overlap(qa.answer, line) >= args.threshold
                    ]
                    samples[bucket].append({
                        "conversation": conv.sample_id,
                        "category": qa.category_name,
                        "question": qa.question,
                        "gold": qa.answer,
                        "predicted": predicted,
                        "f1": row.get("f1"),
                        "carrying": carrying[:3],
                    })
        finally:
            memory.close()

    scored = sum(buckets.values())
    if not scored:
        print("nothing measured")
        return

    print(f"\n{'=' * 76}")
    print(f"questions we got wrong          : {wrong_total}")
    print(f"  evidence was never retrieved  : {evidence_absent} "
          f"({evidence_absent / wrong_total:.1%})")
    strict_scored = sum(strict_buckets.values())
    print(f"  evidence in prompt, loose test: {scored} "
          f"({scored / wrong_total:.1%})")
    print(f"  evidence in prompt, strict    : {wrong_total - strict_absent} "
          f"({(wrong_total - strict_absent) / wrong_total:.1%})")
    print(f"\n  of those {scored}, what the answer looked like "
          f"(strict subset in brackets):\n")
    print(f"  {'bucket':12s} {'n':>6s} {'share':>7s} {'strict':>8s} "
          f"{'median words':>13s}")
    for name in ("refused", "near miss", "confident", "empty"):
        if buckets.get(name):
            words = statistics.median(lengths[name]) if lengths[name] else 0
            share = (f"{strict_buckets.get(name, 0) / strict_scored:6.1%}"
                     if strict_scored else "     -")
            print(f"  {name:12s} {buckets[name]:6d} "
                  f"{buckets[name] / scored:6.1%} {share:>8s} {words:13.0f}")

    print(f"\n  {'category':12s} " + " ".join(f"{b:>10s}" for b in
                                              ("refused", "near miss", "confident")))
    for name in SCORED:
        if name in by_category:
            row = by_category[name]
            print(f"  {name:12s} " + " ".join(
                f"{row.get(b, 0):10d}" for b in ("refused", "near miss", "confident")))

    for name in ("refused", "near miss", "confident"):
        for ex in samples.get(name, []):
            print(f"\n  [{name} / {ex['category']}] {ex['question']}")
            print(f"    gold      : {ex['gold'][:140]}")
            print(f"    predicted : {ex['predicted'][:220]}")
            print(f"    f1        : {ex['f1']}")
            for line in ex["carrying"]:
                print(f"    context   > {line[:140]}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "wrong_total": wrong_total,
            "evidence_absent": evidence_absent,
            "buckets": dict(buckets),
            "by_category": {k: dict(v) for k, v in by_category.items()},
            "samples": {k: v for k, v in samples.items()},
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot-top-k", type=int, default=10)
    ap.add_argument("--cold-budget", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Gold coverage needed to call the evidence present")
    ap.add_argument("--near", type=float, default=0.5,
                    help="Gold coverage in the answer that counts as a near miss")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--out")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
