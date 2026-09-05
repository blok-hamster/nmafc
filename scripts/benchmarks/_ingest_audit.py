"""Is the ingested memory accurate and complete, and are answers dying in Ephemeral?

Two questions, one body of evidence, no model calls:

1. COVERAGE -- for every LoCoMo question, is the gold answer recoverable from
   the facts ingestion actually wrote? A gold answer present in the source
   transcript but absent from the store is information lost at ingestion, which
   no amount of retrieval or prompt work can recover.

2. TIER RISK -- of the answers that ARE stored, how many are held only in
   EphemeralState? Ephemeral decays at lambda 0.69, halving in about one turn,
   so a gold answer stored only there is on a short fuse. This is the specific
   risk created by the tiered prompt moving 55% of facts into that tier.

Cross-tabbed against judge_correct from results.json, this separates three very
different failure modes that all look identical in the accuracy column:
    not ingested        -> extractor problem
    ingested, wrong tier -> decay problem
    ingested, right tier, still wrong -> retrieval or answer-prompt problem

Matching is lexical and deliberately generous: a gold answer counts as covered
when enough of its content words appear in a single stored fact. That biases
toward reporting coverage as GOOD, so the losses it does report are real.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))
CATEGORY = {1: "single-hop", 2: "temporal", 3: "multi-hop",
            4: "open-domain", 5: "adversarial"}
TIER_RANK = {"CoreAnchor": 3, "ActiveContext": 2, "EphemeralState": 1}

STOP = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "was", "were", "are", "be", "been", "with", "by", "from", "that", "this",
    "it", "its", "as", "he", "she", "they", "his", "her", "their", "them",
    "has", "have", "had", "did", "does", "do", "will", "would", "about",
}

MONTHS = ("january february march april may june july august september "
          "october november december").split()


def locomo_path() -> Path:
    hits = sorted(CACHE.glob("datasets--KimmoZZZ--locomo/snapshots/*/locomo10.json"))
    if not hits:
        raise SystemExit("locomo10.json not in the HuggingFace cache")
    return hits[-1]


def norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def content_words(text: str) -> list[str]:
    return [w for w in norm(text).split() if w not in STOP and len(w) > 1]


def covered_by(fact: str, want: list[str], threshold: float) -> bool:
    """Does one stored fact carry enough of the gold answer's content words?"""
    if not want:
        return False
    have = set(norm(fact).split())
    hits = sum(1 for w in want if w in have)
    return hits / len(want) >= threshold


def session_text(conv: dict) -> str:
    out = []
    for key, val in conv.get("conversation", {}).items():
        if not re.fullmatch(r"session_\d+", key) or not isinstance(val, list):
            continue
        for turn in val:
            if isinstance(turn, dict):
                out.append(str(turn.get("text", "")))
    return " ".join(out)


def load_store(db: Path) -> list[tuple[str, str, str]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT entity_name, fact_content, memory_type FROM memory_event_log"
    ).fetchall()
    con.close()
    return [(a or "", b or "", c or "") for a, b, c in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="fraction of gold content words needed in one fact")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    base = Path(__file__).resolve().parent / "results" / args.run
    convs = {c["sample_id"]: c for c in json.load(open(locomo_path(), encoding="utf-8"))}

    verdicts: dict[str, dict] = {}
    res_path = base / "results.json"
    if res_path.exists():
        data = json.load(open(res_path, encoding="utf-8"))["results"]
        if args.arm in data:
            for q in data[args.arm]["question_results"]:
                verdicts[norm(q["question"])] = q

    stats: Counter[str] = Counter()
    by_cat: dict[str, Counter[str]] = defaultdict(Counter)
    lost: list[tuple[str, str, str]] = []
    fuse: list[tuple[str, str, str]] = []
    ephem_wrong = 0

    stores = sorted(base.glob("stores/*/cold.db"))
    print("run=%s arm=%s  stores=%d  threshold=%.2f\n"
          % (args.run, args.arm, len(stores), args.threshold))

    for db in stores:
        sample = db.parent.name.split("__")[-1]
        conv = convs.get(sample)
        if conv is None:
            print("  no dataset entry for", sample)
            continue
        facts = load_store(db)
        transcript = norm(session_text(conv))

        for qa in conv.get("qa", []):
            cat = CATEGORY.get(qa.get("category", 0), "unknown")
            if cat == "adversarial":
                continue  # gold answer is a refusal; nothing to locate
            gold = str(qa.get("answer", "")).strip()
            want = content_words(gold)
            if not want:
                continue

            in_source = covered_by(transcript, want, args.threshold)
            best = 0
            for _, fact, tier in facts:
                if covered_by(fact, want, args.threshold):
                    best = max(best, TIER_RANK.get(tier, 0))
                    if best == 3:
                        break

            judged = verdicts.get(norm(qa.get("question", "")))
            correct = bool(judged.get("judge_correct")) if judged else None

            if best == 0:
                key = "absent from store (was in transcript)" if in_source \
                    else "absent from store (also not literal in transcript)"
                stats[key] += 1
                by_cat[cat][key] += 1
                if in_source and len(lost) < 400:
                    lost.append((sample, qa.get("question", ""), gold))
            elif best == 1:
                stats["stored ONLY in EphemeralState"] += 1
                by_cat[cat]["stored ONLY in EphemeralState"] += 1
                if correct is False:
                    ephem_wrong += 1
                if len(fuse) < 400:
                    fuse.append((sample, qa.get("question", ""), gold))
            else:
                stats["stored in a durable tier"] += 1
                by_cat[cat]["stored in a durable tier"] += 1

    total = sum(stats.values())
    print("=" * 78)
    print("GOLD-ANSWER COVERAGE IN THE INGESTED STORE   (n=%d, adversarial excluded)"
          % total)
    print("=" * 78)
    for key, n in stats.most_common():
        print("  %-46s %5d  %5.1f%%" % (key, n, 100.0 * n / total))

    print("\n  by category")
    print("  %-14s %8s %8s %8s" % ("", "durable", "ephem", "absent"))
    for cat in sorted(by_cat):
        c = by_cat[cat]
        absent = sum(v for k, v in c.items() if k.startswith("absent"))
        tot = sum(c.values())
        print("  %-14s %7.1f%% %7.1f%% %7.1f%%   (n=%d)" % (
            cat,
            100.0 * c["stored in a durable tier"] / tot,
            100.0 * c["stored ONLY in EphemeralState"] / tot,
            100.0 * absent / tot, tot))

    only = stats["stored ONLY in EphemeralState"]
    if only:
        print("\n  of the %d answers held only in EphemeralState, %d were judged WRONG (%.1f%%)"
              % (only, ephem_wrong, 100.0 * ephem_wrong / only))

    print("\n  examples: gold answer in the transcript but NOT in the store")
    for sample, q, gold in lost[:args.examples]:
        print("     [%s] %s" % (sample, q[:62]))
        print("            gold: %s" % gold[:70])

    print("\n  examples: gold answer stored only in the fast-decay tier")
    for sample, q, gold in fuse[:args.examples]:
        print("     [%s] %s" % (sample, q[:62]))
        print("            gold: %s" % gold[:70])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
