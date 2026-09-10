"""Bring the answer-type A/B up to date with the fixed detector, cheaply.

`_ab_answer_type.py` ran with a detector that had two faults: `make` was in the
type-noun list, and a matched span that contained a disallowed word had that
word deleted rather than the match rejected. Between them they tagged
"What did Maria make for her home" as asking for a "Maria make" and
"What happened to John's job" as asking for a "happened John's job". Both are
verb clauses that happen to end on a noun the list knows.

Both faults are fixed in `nmafc.integration.answer_type`. The question is what
that costs to confirm, and the answer is: much less than the run did.

**Arm A never sees the tag.** Its prompt is the shipped one, unchanged by the
fix, so every A answer and every A verdict in the existing file is still exactly
what arm A produces today. Regenerating them would buy nothing but drift.

**Arm B only changes where the tag changes.** B's system prompt carries
TYPE_RULE on every question either way; the per-question line is the only thing
the detector touches. A question whose fixed tag equals its stored tag was
generated from a byte-identical prompt, and its answer stands.

So the honest repair is: recompute the tag for every question already run,
regenerate B for the ones that differ, and report the whole set relabelled. That
is a paired A/B over a fixed question set, both arms valid, at a cost set by how
wrong the detector was rather than by how many questions there are.

One thing this deliberately does not do is re-select the sample. The original
run took every typed question plus 200 untyped controls, and the fix moves
questions across that line, so a fresh selection would draw different controls.
Reporting the questions actually asked, relabelled, keeps the pairing intact and
changes only which slice each question is counted in. The slice sizes shift and
the report prints them.

`--dry-run` is the default and costs nothing: it says how many answers the
repair would regenerate before any of them are.

Usage:
    python -u scripts/benchmarks/_ab_answer_type_repair.py
    python -u scripts/benchmarks/_ab_answer_type_repair.py --run-it
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

from nmafc.integration.answer_type import type_tag  # noqa: E402
from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks._ab_answer_type import (  # noqa: E402
    PROMPT_B,
    answer_with,
    report,
)
from scripts.benchmarks._ab_budget import (  # noqa: E402
    PERMANENT,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402


def restale(rows: list[dict]) -> list[dict]:
    """The rows whose B answer was generated from a prompt we would not use now.

    Each gains a `fixed_tag`; the rest keep theirs and are left alone.
    """
    stale = []
    for row in rows:
        fixed = type_tag(row["question"])
        if fixed != row.get("tag", ""):
            row["fixed_tag"] = fixed
            stale.append(row)
    return stale


def summarise(rows: list[dict], stale: list[dict]) -> None:
    gained = [r for r in stale if r["fixed_tag"] and not r["tag"]]
    lost = [r for r in stale if r["tag"] and not r["fixed_tag"]]
    altered = [r for r in stale if r["tag"] and r["fixed_tag"]]
    print(f"{len(rows)} questions already answered, {len(stale)} need arm B "
          f"regenerating\n")
    for label, group in (("lost a tag", lost), ("gained a tag", gained),
                         ("tag reworded", altered)):
        if not group:
            continue
        print(f"  {len(group):>3}  {label}")
        for row in group[:8]:
            was = row["tag"] or "(none)"
            now = row["fixed_tag"] or "(none)"
            print(f"         {row['question'][:64]}")
            print(f"           {was}  ->  {now}")
        if len(group) > 8:
            print(f"         ... and {len(group) - 8} more")
    typed_now = sum(1 for r in rows if type_tag(r["question"]))
    print(f"\n  slices after the fix: {typed_now} typed, "
          f"{len(rows) - typed_now} untyped controls")


async def run(args: argparse.Namespace) -> None:
    src = Path(args.source)
    if not src.is_file():
        raise SystemExit(f"no {src}; run _ab_answer_type.py first")
    rows = json.loads(src.read_text(encoding="utf-8"))
    stale = restale(rows)
    summarise(rows, stale)

    if not stale:
        print("\nnothing to regenerate; the file already reflects the fix")
        return
    if args.dry_run:
        print(f"\ndry run. {2 * len(stale)} paid calls to repair "
              f"({len(stale)} answers, {len(stale)} verdicts). --run-it to do it.")
        return

    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no merged store at {store}")

    memory = open_memory(store, llm, embedder, args.hot, args.cold, args.budget,
                         args.max_hops, compact=args.compact,
                         hydrate=args.hydrate)
    gate = asyncio.Semaphore(args.concurrency)
    blocked: list[tuple[str, str]] = []

    async def redo(row: dict) -> None:
        async with gate:
            try:
                text, chars = await with_retries(
                    lambda: answer_with(memory, llm, row["question"], PROMPT_B,
                                        row["fixed_tag"]))
                verdict = await with_retries(
                    lambda: judge_answer(row["question"], text, row["gold"],
                                         judge))
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ not in PERMANENT:
                    raise
                blocked.append((row["conv"], row["question"]))
                return
        row["b"], row["b_chars"], row["b_ok"] = text, chars, bool(verdict.correct)
        row["tag"] = row["fixed_tag"]

    out = Path(args.out)
    try:
        for start in range(0, len(stale), args.checkpoint_every):
            batch = stale[start:start + args.checkpoint_every]
            await asyncio.gather(*(redo(r) for r in batch))
            # Checkpoints keep `fixed_tag` on the rows still to do, so an
            # interrupted repair says which ones they were. It is stripped
            # from the finished file below.
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"  {min(start + len(batch), len(stale))}/{len(stale)} "
                  f"regenerated")
    finally:
        close_readonly(memory)

    for row in rows:
        row.pop("fixed_tag", None)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report(rows, blocked)
    print(f"\n-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="C:/nmafc_ab/answer_type.json")
    ap.add_argument("--out", default="C:/nmafc_ab/answer_type_fixed.json")
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    ap.add_argument("--compact", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--dry-run", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--run-it", dest="dry_run", action="store_false")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
