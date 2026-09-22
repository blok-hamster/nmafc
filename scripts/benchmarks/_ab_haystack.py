"""Both arms against one merged pile, which is the test LongMemEval is actually for.

Everything measured so far was measured on easy ground. Each store held one
conversation, so a question only ever competed against the ~830 facts from its
own transcript. The +0.5 overall against RAG, the +17.1 on temporal and the
+7.9 on mylocomoeval were all taken in that setting, and none of them says
anything about what happens when the pile grows. LongMemEval's whole difficulty
is that the answering fact sits in a haystack, and its haystacks would cost
roughly 250 hours of ingestion to build.

`_build_haystack.py` builds one for nothing by merging the ten finished stores:
8,276 archive facts and 3,807 fast-store facts over 2,957 turns, ten times the
material behind every question. This runs both arms against it, and RAG gets
the same treatment -- one index over all ten transcripts, so neither arm is
answering from a smaller world than the other.

The questions are LoCoMo's own 1,540 scored ones, unchanged, already answered
once by both arms on the single-conversation stores. So the comparison is
paired twice over: the two arms against each other within this run, and this
run against `vs_rag.json` question by question. What moves between the two is
only the size of the haystack.

The prediction, stated in advance because stating it afterwards is worthless:
both arms should score worse, since ten times the noise is the point. The
question is which one degrades faster. If our margin widens as the pile grows,
that is the strongest claim this framework can make and it is exactly what
LongMemEval exists to test. If it narrows, the earlier numbers only hold on
easy ground, which is worth knowing before committing 250 hours to the real
thing.

Pairing, alternation and the throttle split all work as `_ab_vs_rag.py`
describes them; the reasoning there applies unchanged and is not repeated.

Our arm's context is configurable here, and that is what makes this the harness
for the decisive run and not only the baseline one. `_ab_hydration.py` scores a
new context setting against the *saved* answers in `haystack.json`, at half the
price and with the risk that killed its last attempt: the saved baseline had
drifted 3.0 points by the time the candidate ran, and the effect being looked
for is about that size. Here both arms are generated in the same session, so
drift cannot enter and no gate is needed. It costs roughly twice as much and it
is the only design whose answer can be trusted.

The context flags default to what produced `haystack.json`, so running this with
no new flags reproduces the baseline head-to-head.

Usage:
    python -u scripts/benchmarks/_ab_haystack.py --limit 2 --out C:/nmafc_ab/haystack_smoke.json
    python -u scripts/benchmarks/_ab_haystack.py --out C:/nmafc_ab/haystack.json

    # the candidate context: six facts over forty single-line source turns,
    # eight of them kept whole, repeated session headers printed once
    python -u scripts/benchmarks/_ab_haystack.py --hot 20 --budget 40 \
        --facts 6 --hydrate 40 --hydrate-lines 1 --hydrate-full-turns 8 \
        --dedupe-headers --categories open-domain \
        --out C:/nmafc_ab/haystack_candidate_open.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
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

from scripts.benchmarks.arms.rag import RagArm  # noqa: E402
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402
from scripts.benchmarks._ab_budget import (  # noqa: E402
    PERMANENT,
    SCORED,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks._ab_vs_rag import report, timed_ours, timed_rag  # noqa: E402


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    gate = asyncio.Semaphore(args.concurrency)

    out = Path(args.out)
    rows: list[dict] = []
    if out.is_file() and not args.restart:
        rows = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(rows)} already answered\n")
    done = {(r["conv"], r["question"]) for r in rows}
    blocked: list[tuple[str, str]] = []

    conversations = list(load_locomo())
    wanted = set(args.categories.split(",")) if args.categories else set(SCORED)
    chosen: list[tuple[str, object]] = []
    for conv in conversations:
        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED
                     and qa.category_name in wanted]
        if args.limit:
            questions = questions[: args.limit]
        chosen += [(conv.sample_id, qa) for qa in questions]

    # Shuffled, seeded, so that any prefix is a fair sample of the whole set.
    # Grouped by conversation, a run stopped half way through -- because the
    # credit ran out, or the machine slept -- would have answered five
    # transcripts fully and five not at all, and the categories are not evenly
    # spread across transcripts. Seeded so a resume rebuilds the same order.
    random.Random(args.seed).shuffle(chosen)
    # Sampled after the shuffle and before `done` is applied, so a resumed run
    # keeps working on the same subset instead of drawing a fresh one from
    # whatever is left.
    if args.sample:
        chosen = chosen[: args.sample]
    pending = [(c, qa) for c, qa in chosen
               if (c, qa.question) not in done]

    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no merged store at {store}; run _build_haystack.py first")

    memory = open_memory(store, llm, embedder, args.hot, args.cold, args.budget,
                         args.max_hops, compact=args.compact,
                         hydrate=args.hydrate, facts=args.facts,
                         lines=args.hydrate_lines,
                         whole=args.hydrate_full_turns,
                         dedupe=args.dedupe_headers)

    # Kept on disk rather than in a temp directory, and reused if it is already
    # populated. `HotStorage.upsert` searches the whole table for the id before
    # every insert, so building one index of roughly 3,000 chunks is quadratic
    # and takes about an hour. In `_ab_vs_rag.py` that cost never showed,
    # because each conversation got its own index of about 300 chunks; ten
    # times the rows is a hundred times the work. Rebuilding it on every resume
    # would cost more than the questions do.
    rag_dir = Path(args.rag_dir)
    rag_dir.mkdir(parents=True, exist_ok=True)
    rag = RagArm(llm, embedder, storage_dir=str(rag_dir))

    print(f"store: {store}")
    # Printed rather than stored: the output file is a flat list of rows and
    # every reader of it assumes that, so the configuration belongs in the log
    # beside the numbers it produced.
    print(f"ours:  top_k {args.hot}, cold {args.cold}, rerank {args.budget}, "
          f"hops {args.max_hops}")
    print(f"       facts {'all' if args.facts is None else args.facts}, "
          f"source turns {args.hydrate}, lines "
          f"{'whole' if args.hydrate_lines is None else args.hydrate_lines}, "
          f"kept whole {args.hydrate_full_turns}, "
          f"dedupe headers {args.dedupe_headers}")
    print(f"       categories {args.categories or 'all'}"
          f"{f', {args.sample} sampled' if args.sample else ''}")
    if rag._store.count() and not args.rebuild_rag:
        print(f"reusing RAG index at {rag_dir}: {rag._store.count()} chunks")
    else:
        if rag._store.count():
            # Ingesting on top of a populated index would double every chunk.
            rag.reset()
        print(f"building RAG's haystack over {len(conversations)} transcripts "
              "(one hour, once)")
        for conv in conversations:
            # One index over everything, so RAG faces the same pile we do.
            # Embedding only, no extraction.
            await with_retries(lambda: rag.ingest_conversation(conv.get_flat_history()))
        print(f"RAG index holds {rag._store.count()} chunks")
    print(f"{len(pending)} questions to answer\n")

    try:
        async def attempt(index: int, conv_id: str, qa):
            async with gate:
                if index % 2 == 0:
                    a = await with_retries(lambda: timed_ours(memory, llm, qa.question))
                    b = await with_retries(lambda: timed_rag(rag, qa.question))
                else:
                    b = await with_retries(lambda: timed_rag(rag, qa.question))
                    a = await with_retries(lambda: timed_ours(memory, llm, qa.question))
                ja = await with_retries(
                    lambda: judge_answer(qa.question, a["pred"], str(qa.answer), judge))
                jb = await with_retries(
                    lambda: judge_answer(qa.question, b["pred"], str(qa.answer), judge))
                return {
                    "conv": conv_id,
                    "question": qa.question,
                    "category": qa.category_name,
                    "gold": str(qa.answer),
                    "first": "ours" if index % 2 == 0 else "rag",
                    "ours_pred": a["pred"], "rag_pred": b["pred"],
                    "ours_ok": ja.correct, "rag_ok": jb.correct,
                    "ours_ms": a["ms"], "rag_ms": b["ms"],
                    "ours_throttle": a["throttle"], "rag_throttle": b["throttle"],
                    "ours_chars": a["chars"], "rag_chars": b["chars"],
                }

        async def one(index: int, conv_id: str, qa):
            try:
                return await attempt(index, conv_id, qa)
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ not in PERMANENT:
                    raise
                blocked.append((conv_id, qa.question))
                return None

        tasks = [one(i, c, qa) for i, (c, qa) in enumerate(pending)]
        for start in range(0, len(tasks), args.checkpoint_every):
            batch = await asyncio.gather(*tasks[start:start + args.checkpoint_every])
            rows.extend(r for r in batch if r)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"  {len(rows)}/{len(pending) + len(done)} answered")
    finally:
        close_readonly(memory)
        # Not `rag.reset()`. That deletes the index directory, and this one is
        # kept deliberately so a resume does not spend another hour building it.

    if not rows:
        print("nothing measured")
        return
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report(rows, blocked)
    print(f"\n-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack",
                    help="Merged store name under <run>/stores/<arm>__<store>.")
    ap.add_argument("--rag-dir", default="C:/nmafc_ab/haystack_rag",
                    help="Where RAG's index lives. Reused if already populated.")
    ap.add_argument("--rebuild-rag", action="store_true",
                    help="Discard the cached RAG index and build it again.")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--facts", type=int, default=None,
                    help="Facts written into the FACTS block. Unset writes "
                         "every retrieved fact, which is what shipped does and "
                         "what produced haystack.json.")
    ap.add_argument("--hydrate-lines", type=int, default=None,
                    help="Keep only the N best-matching speaker lines of each "
                         "hydrated turn. Unset keeps the whole turn.")
    ap.add_argument("--hydrate-full-turns", type=int, default=0,
                    help="Top-ranked turns kept whole before --hydrate-lines "
                         "applies to the rest.")
    ap.add_argument("--dedupe-headers", action="store_true",
                    help="Print a repeated session header once per run of "
                         "turns instead of on every turn.")
    ap.add_argument("--categories", default="",
                    help="Comma-separated categories to score. Empty scores "
                         "all of them.")
    ap.add_argument("--sample", type=int, default=0,
                    help="Total questions to answer, drawn at random after the "
                         "category filter. 0 answers all of them.")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--compact", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0,
                    help="Questions per conversation, before sampling.")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="C:/nmafc_ab/haystack.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
