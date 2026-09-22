"""Of the answers that never reached the prompt, how many were in the store?

This is the question the engram literature forces. Ryan et al. induced amnesia
in mice, found the animals could not retrieve a memory by any natural cue, and
then recovered it by stimulating the tagged neurons directly -- so the trace was
intact and the deficit was in access, not in storage. Storage and retrieval come
apart, and which one has failed is an empirical question rather than a
preference.

Our equivalent is the 14.9% of mylocomoeval questions where only the superseded
fact reached the prompt and the current one did not. Every one is answered
wrong, and it is the largest recoverable block on the test. Two completely
different failures produce that symptom:

  the fact is in the store and retrieval did not surface it
      -- an access failure, fixable at query time, and the direction every
         reranking experiment so far has been aimed at

  the fact is not in the store at all
      -- a write-time failure, and no amount of reranking can recover it

Decay, demote and expand all assumed the first. None helped. This checks the
assumption rather than continuing to act on it.

THE INSTRUMENT IS THE WHOLE PROBLEM, and the first version of this script got
it wrong. It reused `present`, which asks for every content word of the gold
answer as a whole word in a single record. That reported 19 answers as never
extracted. Reading the nearest stored fact for each showed most of them sitting
in the store in different words: "tried to get it fixed" against a stored
"John is trying to get his car fixed", "roasted veg" against "roasted
vegetables", "fixed and going strong" against "fixed and running strong". The
test was measuring inflection, not storage, and it failed the same way on both
sides of the comparison -- so it under-counted what reached the prompt too, and
the access-versus-write split it produced meant nothing.

So both sides are judged by the model, in one call per question that sees the
retrieved context and the store's best candidates together and answers about
each. The strict word test is kept alongside as a floor, because a split that
moves when the instrument changes is a fact about the instrument.

Reads the stores read-only. One LLM call and one embedding per question.

Usage:
    python scripts/benchmarks/_probe_stored_vs_reached.py --limit 2
    python scripts/benchmarks/_probe_stored_vs_reached.py --out C:/nmafc_ab/stored_vs_reached.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
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

import lancedb  # noqa: E402
import numpy as np  # noqa: E402

from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks._ab_budget import (  # noqa: E402
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks._sweep_context_budget import retrieve_with_retry  # noqa: E402
from scripts.benchmarks._test_updates import present, tokens  # noqa: E402

JUDGE_SYSTEM = """\
You decide whether a piece of information is present in some text. You are not
answering the question and not judging whether anything is true.

You get a QUESTION, the ANSWER that is correct, and two bodies of text:
RETRIEVED (what a memory system put in front of a model) and STORED (candidate
records held in that system's database).

For each body of text, decide whether it states the ANSWER to the QUESTION.
Count it as present when the wording differs but the information is the same:
"is trying to get his car fixed" states the answer "tried to get it fixed", and
"roasted vegetables" states the answer "roasted veg". Do not count it as present
when the text is merely about the same topic, mentions an earlier or different
value, or would require guessing.

Reply with JSON and nothing else:
{"in_retrieved": true|false, "in_stored": true|false}"""


class StoreIndex:
    """Every visible fact, both tiers, with its vector, for shortlisting.

    Cold is filtered the way the archive search filters it, so a fact retrieval
    could never return does not count as stored.

    The shortlist has to be built by meaning, and the first two versions of this
    script were not. They ranked the store by content-word overlap with the gold
    answer, which collapses whenever the answer is short. For "How many cats
    does Melanie have?" the gold is "two", so every fact in the store containing
    the word "two" scores a perfect 1.0 and the top eight are drawn arbitrarily
    from hundreds of ties. The judge then sees eight unrelated facts, says the
    answer is not among them, and the question is recorded as never extracted.
    That inverted the whole result: it made the judge look harsher than
    whole-word matching, when in fact whole-word matching was passing those same
    short answers automatically -- "two" appears somewhere, therefore stored.

    So the shortlist is nearest-neighbour on the question and answer together,
    against the same vectors retrieval uses. One embedding per question.
    """

    def __init__(self, store: Path) -> None:
        vectors: list[np.ndarray] = []
        self.facts: list[str] = []
        frame = lancedb.connect(str(store / "hot_lancedb")) \
            .open_table("memory_vectors").to_pandas()
        for row in frame.itertuples():
            vectors.append(np.asarray(row.vector, dtype=np.float32))
            self.facts.append(row.fact_content)
        conn = sqlite3.connect(store / "cold.db")
        try:
            rows = conn.execute(
                "SELECT fact_content, embedding FROM memory_event_log "
                " WHERE is_active = 1 AND invalid_at IS NULL "
                "   AND embedding IS NOT NULL").fetchall()
        finally:
            conn.close()
        width = len(vectors[0])
        for fact, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.size != width:
                continue
            vectors.append(vector)
            self.facts.append(fact)
        matrix = np.vstack(vectors).astype(float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.where(norms == 0, 1, norms)

    def nearest(self, query: np.ndarray, k: int) -> list[str]:
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        scores = self.matrix @ (query / norm)
        return [self.facts[i] for i in np.argsort(-scores)[:k]]


def best_candidates(nearest: list[str], retrieved: list[str]) -> list[str]:
    """Nearest store records, with whatever retrieval surfaced put in front.

    Seeding with the retrieved records is a correctness requirement rather than
    a nicety. The first run shortlisted without them and reported the store
    holding the answer for 73.7% of questions while the prompt held it for
    78.9% -- impossible, since every retrieved record came out of the store.
    """
    seen = set(retrieved)
    return list(retrieved) + [f for f in nearest if f not in seen]


async def judge(llm, question: str, gold: str, retrieved: str,
                stored: list[str]) -> dict:
    prompt = (f"QUESTION: {question}\nANSWER: {gold}\n\n"
              f"RETRIEVED:\n{retrieved or '(nothing)'}\n\n"
              f"STORED:\n" + ("\n".join(f"- {s}" for s in stored) or "(nothing)"))
    text = await llm.chat(messages=[{"role": "user", "content": prompt}],
                          system_prompt=JUDGE_SYSTEM)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        # Unparseable means unknown, not absent. Recording it as absent would
        # quietly push every parse failure into the write-failure column.
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


async def run(args: argparse.Namespace) -> None:
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for row in questions:
        by_conv[row["conv"]].append(row)
    if args.limit:
        by_conv = {k: v[: args.limit] for k, v in by_conv.items()}

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    gate = asyncio.Semaphore(args.concurrency)

    # One merged store answers every question, or one store per conversation.
    # The merged store is what a haystack run retrieved from, so a probe of its
    # results has to read the same pile it read.
    groups = ([(args.store, [i for v in by_conv.values() for i in v])]
              if args.store else list(by_conv.items()))

    rows: list[dict] = []
    for conv, items in groups:
        store = Path(args.run) / "stores" / f"{args.arm}__{conv}"
        if not store.is_dir():
            continue
        index = StoreIndex(store)
        facts = index.facts
        memory = open_memory(store, llm, embedder, args.hot, args.cold,
                             args.budget, args.max_hops, compact=True,
                             hydrate=args.hydrate)
        print(f"[{conv}] {len(items)} questions, {len(facts)} facts")
        try:
            router = memory._router
            turn = memory.current_turn + 1

            async def one(item: dict) -> dict:
                pool = await retrieve_with_retry(router, item["question"], turn)
                kept = pool[: args.budget] if pool else []
                context = router.format_context(kept) if kept else ""
                # Question and answer together. The question alone would rank
                # by topic and could miss the record holding the value; the
                # answer alone is useless when it is one word.
                probe = f"{item['question']} {item['gold']}"
                vector = np.asarray(
                    (await with_retries(lambda: embedder.embed([probe])))[0],
                    dtype=float)
                shortlist = best_candidates(
                    index.nearest(vector, args.candidates),
                    [r.fact_content for r in kept])
                async with gate:
                    verdict = await with_retries(
                        lambda: judge(llm, item["question"], item["gold"],
                                      context, shortlist))
                reached = bool(verdict.get("in_retrieved"))
                return {
                    "conv": conv,
                    "question": item["question"],
                    "gold": item["gold"],
                    # Judged: wording-tolerant, the instrument that decides.
                    # A fact in the prompt is in the store by construction, so
                    # the disjunction is not charity, it is the definition.
                    "stored": bool(verdict.get("in_stored")) or reached,
                    "reached_prompt": reached,
                    "judged": bool(verdict),
                    # Strict whole-word, kept as a floor for comparison. Same
                    # disjunction, for a second reason: `present` is asked of
                    # one record at a time in the store and of the whole
                    # context at once, so a gold answer spread over two
                    # retrieved facts reads present in the prompt and absent
                    # from the store.
                    "stored_strict": present(item["gold"], context) or any(
                        present(item["gold"], f) for f in facts),
                    "reached_strict": present(item["gold"], context),
                    # Optional. mylocomoeval questions carry the superseded
                    # value; a question list drawn from a plain benchmark run
                    # has no such thing, and an empty string would read as
                    # present in every context.
                    "stale_reached": bool(item.get("stale")) and present(
                        item["stale"], context),
                }

            rows.extend(await asyncio.gather(*(one(i) for i in items)))
        finally:
            close_readonly(memory)

    if not rows:
        print("nothing measured")
        return

    n = len(rows)
    unparsed = sum(1 for r in rows if not r["judged"])
    print(f"\n{'=' * 78}")
    print(f"{n} questions" + (f", {unparsed} unparsed verdicts" if unparsed else ""))

    for label, sk, rk in (("JUDGED  ", "stored", "reached_prompt"),
                          ("STRICT  ", "stored_strict", "reached_strict")):
        stored = sum(r[sk] for r in rows)
        reached = sum(r[rk] for r in rows)
        missed = [r for r in rows if not r[rk]]
        access = sum(1 for r in missed if r[sk])
        absent = len(missed) - access
        share = lambda x: 100 * x / max(len(missed), 1)  # noqa: E731
        print(f"\n{label} gold stored     {stored:4d}  {100 * stored / n:5.1f}%"
              f"   <- the ceiling")
        print(f"         gold in prompt  {reached:4d}  {100 * reached / n:5.1f}%"
              f"   <- what retrieval achieves")
        print(f"         headroom        {stored - reached:4d} questions")
        print(f"         of {len(missed):3d} misses: {access:3d} stored but not "
              f"surfaced ({share(access):.0f}% ACCESS), "
              f"{absent:3d} never stored ({share(absent):.0f}% WRITE)")

    print("\n  STRICT is shown for contrast, not as a second opinion. It passes"
          "\n  any short answer automatically: gold \"two\" is whole-word present"
          "\n  in every store holding the word \"two\" anywhere. JUDGED decides.")

    only_stale = [r for r in rows if r["stale_reached"] and not r["reached_prompt"]]
    recoverable = sum(1 for r in only_stale if r["stored"])
    print(f"\n  decisive block, only the superseded fact reached the prompt: "
          f"{len(only_stale)}")
    print(f"    gold was in the store for {recoverable} of them "
          f"({100 * recoverable / max(len(only_stale), 1):.0f}%)")
    print("\n  Headroom is the whole point. A large one means the answering fact")
    print("  is sitting in the store unreached and reranking is worth doing. A")
    print("  small one means retrieval is near its ceiling and the points are")
    print("  at write time, where no reranking can reach them.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="",
                    help="One merged store for every question, e.g. haystack. "
                         "Empty means the per-conversation store.")
    ap.add_argument("--questions", default="C:/nmafc_ab/updates_final.json")
    ap.add_argument("--candidates", type=int, default=12,
                    help="Store records shown to the judge, nearest first.")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="C:/nmafc_ab/stored_vs_reached.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
