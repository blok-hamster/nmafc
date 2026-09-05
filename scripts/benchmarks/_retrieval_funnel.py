"""Where does a question stop being answerable: storage, retrieval, or answering?

_ingest_audit.py established that ~90% of gold answers are somewhere in the
store. That leaves two candidate bottlenecks, and the accuracy column cannot
tell them apart. This splits them into a three-stage funnel:

    stage 1  IN STORE    the gold answer exists in the ingested facts
    stage 2  RETRIEVED   it survives into a top-k retrieval budget
    stage 3  ANSWERED    the judge marked the model's reply correct

A large stage 1 -> 2 drop means ranking is dropping facts it holds. A large
stage 2 -> 3 drop means the facts reach the model and the model fails to use
them, which is an answer-prompt problem, not a memory problem.

Retrieval here is Cold ROM's own BM25 index (the memory_fts FTS5 table written
during ingestion), not the Hot RAM vector search: pyarrow is blocked on this
machine and embeddings would cost quota. That makes stage 2 a LOWER BOUND --
lexical search is weaker than the real hybrid retriever, so anything BM25 finds
the real pipeline would almost certainly also find. A stage 2 number that is
already high is therefore trustworthy; a low one is not conclusive on its own.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ia", HERE / "_ingest_audit.py")
ia = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ia)


def fts_query(question: str) -> str:
    """Build a safe FTS5 OR-query from a natural-language question."""
    words = [w for w in ia.content_words(question) if len(w) > 2]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


def retrieve(con: sqlite3.Connection, question: str, limit: int) -> list[str]:
    query = fts_query(question)
    if not query:
        return []
    try:
        rows = con.execute(
            "SELECT fact_content FROM memory_fts WHERE memory_fts MATCH ? "
            "ORDER BY bm25(memory_fts) LIMIT ?",
            (query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] or "" for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--budgets", type=int, nargs="*", default=[10, 20, 40])
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    base = HERE / "results" / args.run
    convs = {c["sample_id"]: c
             for c in json.load(open(ia.locomo_path(), encoding="utf-8"))}

    verdict: dict[str, bool] = {}
    res = base / "results.json"
    if res.exists():
        data = json.load(open(res, encoding="utf-8"))["results"]
        if args.arm in data:
            for q in data[args.arm]["question_results"]:
                verdict[ia.norm(q["question"])] = bool(q.get("judge_correct"))

    main_budget = args.budgets[len(args.budgets) // 2]
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    percat: dict[str, Counter[str]] = defaultdict(Counter)
    held_not_used: list[tuple[str, str, str]] = []
    not_ranked: list[tuple[str, str, str]] = []

    for db in sorted(base.glob("stores/*/cold.db")):
        sample = db.parent.name.split("__")[-1]
        conv = convs.get(sample)
        if conv is None:
            continue
        facts = ia.load_store(db)
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

        for qa in conv.get("qa", []):
            cat = ia.CATEGORY.get(qa.get("category", 0), "?")
            if cat == "adversarial":
                continue
            question = str(qa.get("question", ""))
            want = ia.content_words(str(qa.get("answer", "")))
            if not want:
                continue

            all_text = " ".join(f for _, f, _ in facts)
            in_store = ia.covered_by(all_text, want, args.threshold)
            correct = verdict.get(ia.norm(question))

            for budget in args.budgets:
                got = ia.covered_by(" ".join(retrieve(con, question, budget)),
                                    want, args.threshold)
                c = counts[budget]
                c["total"] += 1
                if in_store:
                    c["in store"] += 1
                    if got:
                        c["retrieved"] += 1
                        if correct:
                            c["answered"] += 1
                        elif correct is False and len(held_not_used) < 300:
                            held_not_used.append(
                                (sample, question, str(qa.get("answer", ""))))
                    elif len(not_ranked) < 300 and budget == main_budget:
                        not_ranked.append(
                            (sample, question, str(qa.get("answer", ""))))
                if budget == main_budget:
                    pc = percat[cat]
                    pc["total"] += 1
                    pc["in store"] += int(in_store)
                    pc["retrieved"] += int(in_store and got)
                    pc["answered"] += int(in_store and got and bool(correct))
        con.close()

    print("run=%s arm=%s  (adversarial excluded, BM25 retrieval = lower bound)\n"
          % (args.run, args.arm))
    print("=" * 74)
    print("FUNNEL, as a share of all scored questions")
    print("=" * 74)
    print("  %-9s %10s %11s %10s" % ("budget", "in store", "retrieved", "answered"))
    for budget in args.budgets:
        c = counts[budget]
        t = c["total"] or 1
        print("  top-%-6d %9.1f%% %10.1f%% %9.1f%%" % (
            budget, 100 * c["in store"] / t,
            100 * c["retrieved"] / t, 100 * c["answered"] / t))

    c = counts[main_budget]
    print("\n  drop-offs at top-%d:" % main_budget)
    if c["in store"]:
        print("    held but not ranked into the budget : %d  (%.1f%% of stored)"
              % (c["in store"] - c["retrieved"],
                 100 * (c["in store"] - c["retrieved"]) / c["in store"]))
    if c["retrieved"]:
        print("    reached the model and still wrong   : %d  (%.1f%% of retrieved)"
              % (c["retrieved"] - c["answered"],
                 100 * (c["retrieved"] - c["answered"]) / c["retrieved"]))

    print("\n  by category, at top-%d" % main_budget)
    print("  %-13s %9s %10s %9s" % ("", "in store", "retrieved", "answered"))
    for cat in sorted(percat):
        p = percat[cat]
        t = p["total"] or 1
        print("  %-13s %8.1f%% %9.1f%% %8.1f%%   (n=%d)" % (
            cat, 100 * p["in store"] / t, 100 * p["retrieved"] / t,
            100 * p["answered"] / t, t))

    print("\n  examples: fact retrieved, answer still wrong")
    for sample, q, gold in held_not_used[:args.examples]:
        print("     [%s] %s" % (sample, q[:60]))
        print("            gold: %s" % gold[:66])
    print("\n  examples: fact in store, never ranked into the budget")
    for sample, q, gold in not_ranked[:args.examples]:
        print("     [%s] %s" % (sample, q[:60]))
        print("            gold: %s" % gold[:66])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
