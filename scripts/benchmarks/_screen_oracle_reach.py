"""Is the answer anywhere in the raw conversation at all?

Every retrieval change measured here moves the same number a few points: how
often the gold reaches the rendered context. Reading those numbers, nobody has
asked what the ceiling is. This asks.

For each question we get wrong, it reads every turn of that conversation out of
cold ROM and checks whether **any single turn** contains every content word of
the gold. No ranking, no budget, no scoring -- just presence. That is the best
any turn-retrieval method could ever do, because a method that reads all of them
is the limit of a method that reads eight.

The answer decides what to build, and the two outcomes point opposite ways:

    ceiling is high      the words are there and our scorer is not finding
                         them. BM25 is lexical, so a turn that says "Witcher 3
                         got me started" does not match a question asking what
                         "inspired" someone. Building a semantic turn index
                         would be worth it, and it needs no re-ingestion --
                         embedding a few hundred turns per conversation is
                         cheap and touches no extractor.

    ceiling is low       the gold's wording is not in the transcript in the
                         form the detector wants, and no retrieval change of
                         any kind reaches it. Stop spending on retrieval.

**The detector is the same strict all-content-words rule as every other screen
here, and its weakness matters more in this script than anywhere else.** A gold
of "He found her appearance and eyes amazing" is a paraphrase of the dialogue,
not a quotation of it, so it will read as absent even though a human would find
it. So the number this prints is a *floor* on the ceiling. If it comes back low
that is partly the detector, and the split by gold length below is there to say
how much: short golds are quotations and long ones are paraphrases.

Read-only, and it reads cold ROM directly. No model, no embeddings, no spend.

Usage:
    python -u scripts/benchmarks/_screen_oracle_reach.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.benchmarks._screen_hydrate_pool import (  # noqa: E402
    content_words,
    present,
)


def turns_for(store: Path) -> dict[str, list[str]]:
    """Every turn of every conversation in one merged store.

    Keyed by conversation so a gold is only ever looked for in the transcript it
    came from. Looking across conversations would find "Sweden" somewhere and
    call it reachable, which is how an oracle screen tells you a comfortable
    lie.
    """
    db = store / "cold.db"
    if not db.is_file():
        raise SystemExit(f"no cold store at {db}")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: dict[str, list[str]] = {}
    for conv, text in conn.execute(
            "SELECT conversation_id, text FROM turn_text"):
        out.setdefault(conv, []).append(text)
    conn.close()
    return out


def best_turn(gold: str, texts: list[str]) -> bool:
    return any(present(gold, t) for t in texts)


def scope(by_conv: dict[str, list[str]], conv: str) -> list[str]:
    """The turns a gold is allowed to be found in.

    Scoping to the question's own conversation is the honest thing to do, and
    the first version of this did only that -- and reported 0.0% on the
    questions we get *right*, which is impossible and is what a lookup that
    always misses looks like.

    The haystack is one merged store: every conversation was ingested under a
    single `conversation_id`, so the results file's `conv` is a sample id that
    matches no key here. When the store holds one conversation there is nothing
    to scope to, and searching all of it is the only option. That makes the
    ceiling generous -- a gold could be matched by a turn from a different
    transcript -- so a HIGH number from a merged store proves less than a low
    one does. A low number is still decisive: if the words are absent from
    2,957 turns they are absent from the 300 that mattered.
    """
    if conv in by_conv:
        return by_conv[conv]
    if len(by_conv) == 1:
        return next(iter(by_conv.values()))
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/open_domain.json")
    ap.add_argument("--rag", default="C:/nmafc_ab/haystack.json")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    by_conv = turns_for(store)
    print(f"{sum(len(v) for v in by_conv.values())} turns across "
          f"{len(by_conv)} conversations\n")

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    lost = [r for r in rows if not r["b_ok"]]
    won = [r for r in rows if r["b_ok"]]

    def rate(rs: list[dict], label: str) -> list[dict]:
        misses = [r for r in rs
                  if not best_turn(r["gold"], scope(by_conv, r["conv"]))]
        hits = len(rs) - len(misses)
        print(f"  {label:<28}{hits:>5} of {len(rs):<5}"
              f"{100 * hits / max(len(rs), 1):>7.1f}%")
        return misses

    print("  Gold present in SOME turn of the right conversation:")
    misses = rate(lost, "questions we get wrong")
    rate(won, "questions we get right")

    # Short golds are quotations of the dialogue and long ones are the
    # dataset's paraphrase of it, so the detector is honest about the first
    # group and unfair to the second. Splitting says how much of a low ceiling
    # is the transcript and how much is the measuring stick.
    print("\n  By how long the gold is, on the questions we get wrong:")
    for label, lo, hi in (("1-2 content words", 1, 2),
                          ("3-5 content words", 3, 5),
                          ("6+ content words", 6, 99)):
        band = [r for r in lost if lo <= len(content_words(r["gold"])) <= hi]
        if band:
            rate(band, f"    {label}")

    print(f"\n  {len(misses)} golds are in NO turn of their own conversation. "
          f"No retrieval\n  change of any kind can reach these:")
    for r in misses[: args.show]:
        print(f"    gold {r['gold'][:56]:<58}ours {r['b'][:36]}")

    # The question the whole session turns on. RAG reads the same transcript we
    # do, so a gold whose words are in no turn is out of its reach too. If RAG
    # still wins those, its advantage is not retrieval and no retrieval work of
    # ours will close the gap. If it wins the reachable ones, better search is
    # exactly the answer and the ceiling above says how much is left in it.
    rag = json.loads(Path(args.rag).read_text(encoding="utf-8"))
    verdict = {(r["conv"], r["question"]): r for r in rag}
    print(f"\n  Of the questions WE get wrong, how does RAG do?")
    print(f"    {'':<26}{'n':>5}{'RAG right':>12}")
    for label, group in (("gold is in some turn",
                          [r for r in lost if r not in misses]),
                         ("gold is in no turn", misses)):
        paired = [verdict.get((r["conv"], r["question"])) for r in group]
        seen = [p for p in paired if p and "rag_ok" in p]
        if not seen:
            print(f"    {label:<26}{len(group):>5}{'not paired':>12}")
            continue
        right = sum(1 for p in seen if p["rag_ok"])
        print(f"    {label:<26}{len(seen):>5}{right:>7}"
              f"{100 * right / len(seen):>5.0f}%")


if __name__ == "__main__":
    main()
