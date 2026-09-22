"""Does telling the model what <SOURCE> is convert hydration into more score?

Hydration ships the verbatim turns behind the top-ranked facts in a <SOURCE>
block, and it is worth +1.75 points at p=0.021. But ANSWER_SYSTEM_PROMPT opens
"You have a knowledge graph of facts, shown in <FACTS> tags" and never mentions
<SOURCE> at all, so the block arrives unannounced. Worse, SHORT_ANSWER_RULES
says "COPY THE WORDING FROM THE FACTS", and the facts are the extractor's
paraphrase while LoCoMo's reference answers are drawn from what was actually
said. The model is being pointed at the reworded copy and told to quote it.

That predicts a specific error, and the failure analysis found it: 138 near
misses, median four words, right answer in the wrong words. This arm changes
nothing but the instructions.

Both arms read the same stores with the same retrieval settings and the same
hydration budget, so any gap is the prompt. Paired over the same questions, and
scored with the same judge as every other run.

Read-only: retrieval buffers its reinforcement writes and never flushes.

Usage:
    python scripts/benchmarks/_ab_prompt.py --out /c/nmafc_ab/ab_prompt.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

from scripts.benchmarks._ab_budget import (  # noqa: E402
    PERMANENT,
    SCORED,
    open_memory,
    with_retries,
)
from scripts.benchmarks.arms.base import strip_answer  # noqa: E402
from scripts.benchmarks.arms.neuromorphic_tuned import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
)
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402

# Inserted ahead of SHORT_ANSWER_RULES so its "copy the wording from the facts"
# line is read in the light of this, rather than contradicting it after the
# fact. Deliberately short: the existing prompt is long and heavily tuned, and
# a rewrite would make the comparison unattributable.
SOURCE_RULES = """
A <SOURCE> block may follow the facts. It holds the conversation's own words, \
from the turns the best-ranked facts were drawn from. The facts are a summary \
written afterwards, in different words; <SOURCE> is what was actually said.

- The reference answer is taken from what was said, so when <SOURCE> covers the \
question, answer in its wording. Where <FACTS> and <SOURCE> express the same \
thing differently, use the <SOURCE> wording.
- <SOURCE> is evidence, not a topic list. A turn is there because a fact ranked \
well, not because everything in it is relevant, so do not answer from it unless \
it addresses the question.
- If the two disagree on substance rather than wording, prefer whichever is \
dated later. If neither is dated, prefer <FACTS>.
"""

PROMPT_B = ANSWER_SYSTEM_PROMPT.replace(
    "\nCRITICAL: You are completing a fill-in-the-blank quiz.",
    SOURCE_RULES + "\nCRITICAL: You are completing a fill-in-the-blank quiz.",
)


async def answer_with(memory, llm, question: str, system_prompt: str):
    """One answer, and the context it was given, under a chosen prompt."""
    retrieved = await memory._router.retrieve(question, memory.current_turn + 1)
    context = memory._router.format_context(retrieved)
    system = system_prompt + (f"\n\n{context}" if context else "")
    text = await llm.chat(
        messages=[{"role": "user", "content": question}], system_prompt=system
    )
    return strip_answer(text), len(context)


async def run(args: argparse.Namespace) -> None:
    if PROMPT_B == ANSWER_SYSTEM_PROMPT:
        raise SystemExit(
            "SOURCE_RULES was not inserted: the anchor text has moved in "
            "SHORT_ANSWER_RULES. Fix the anchor rather than running two "
            "identical arms."
        )

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
        print(f"resuming: {len(rows)} questions already answered\n")
    done = {(r["conv"], r["question"]) for r in rows}
    blocked: list[tuple[str, str]] = []

    for conv in load_locomo():
        store = Path(args.run) / "stores" / f"{args.arm}__{conv.sample_id}"
        if not store.is_dir():
            print(f"  [skip] no store for {conv.sample_id}")
            continue
        questions = [
            qa for qa in conv.qa_pairs
            if qa.category_name in SCORED
            and (conv.sample_id, qa.question) not in done
        ]
        if args.limit:
            questions = questions[: args.limit]
        if not questions:
            continue

        # Two handles on one store rather than one shared: the arms differ only
        # in the prompt, but sharing a memory object would also share its
        # reinforcement buffer, and arm A's reads would then be visible in the
        # weights arm B ranks against.
        mem_a = open_memory(store, llm, embedder, args.hot, args.cold,
                            args.budget, args.max_hops, False, args.hydrate)
        mem_b = open_memory(store, llm, embedder, args.hot, args.cold,
                            args.budget, args.max_hops, False, args.hydrate)
        async def one(qa, conv=conv):
            async with gate:
                try:
                    a_text, a_chars = await with_retries(
                        lambda: answer_with(
                            mem_a, llm, qa.question, ANSWER_SYSTEM_PROMPT
                        )
                    )
                    b_text, b_chars = await with_retries(
                        lambda: answer_with(mem_b, llm, qa.question, PROMPT_B)
                    )
                    a_v = await with_retries(
                        lambda: judge_answer(qa.question, a_text, qa.answer, judge)
                    )
                    b_v = await with_retries(
                        lambda: judge_answer(qa.question, b_text, qa.answer, judge)
                    )
                except Exception as exc:  # noqa: BLE001
                    if type(exc).__name__ not in PERMANENT:
                        raise
                    blocked.append((conv.sample_id, qa.question))
                    return None
            return {
                "conv": conv.sample_id,
                "question": qa.question,
                "category": qa.category_name,
                "gold": qa.answer,
                "a": a_text, "b": b_text,
                "a_chars": a_chars, "b_chars": b_chars,
                "a_ok": bool(a_v.correct), "b_ok": bool(b_v.correct),
            }

        pending = [one(qa) for qa in questions]
        for start in range(0, len(pending), args.checkpoint_every):
            batch = await asyncio.gather(*pending[start:start + args.checkpoint_every])
            rows.extend(r for r in batch if r)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"    {conv.sample_id}: "
                  f"{min(start + args.checkpoint_every, len(pending))}/{len(pending)}")

        got = [r for r in rows if r["conv"] == conv.sample_id]
        print(f"  {conv.sample_id}: {sum(r['a_ok'] for r in got)}"
              f" -> {sum(r['b_ok'] for r in got)} of {len(got)}")

    n = len(rows) or 1
    a = sum(r["a_ok"] for r in rows)
    b = sum(r["b_ok"] for r in rows)
    fixed = sum(1 for r in rows if r["b_ok"] and not r["a_ok"])
    broke = sum(1 for r in rows if r["a_ok"] and not r["b_ok"])
    print(f"\n  n={len(rows)}   A {a} ({a/n:.2%})   B {b} ({b/n:.2%})   "
          f"{b - a:+d} ({(b - a)/n:+.2%})")
    print(f"  fixed {fixed}  broke {broke}   blocked by content filter {len(blocked)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="ab_prompt.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
