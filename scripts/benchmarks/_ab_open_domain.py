"""The full open-domain stack against the shipped arm, both generated together.

Open-domain is the only category still losing to RAG and it is 839 of 1,535
questions, so it sets the overall result. Three mechanisms are now built and
screened against it, and this is the run that says whether the screens convert.

They divide cleanly, which is why they are run together rather than one at a
time. The failure being attacked has two halves and each mechanism takes one:

    reach       is the answer in the prompt at all?
                `source_grounding` ranks facts by how well the turn behind them
                matches the question. `hydrate_pool` chooses which turns get
                hydrated by the same measure instead of by fact rank.

    conversion  given that it is in the prompt, is it the value chosen?
                answer-type conditioning, gated so that a question naming no
                type gets the shipped prompt byte-for-byte.

47 of 116 open-domain losses already had the answer in the prompt and lost
anyway, so reach alone was never going to be enough -- that bucket is a
conversion failure by definition. Equally, conversion cannot help a question
whose answer never arrives. Screened separately they overlap (the free screen
puts grounding at +6/-1, hydration at +9/-3 and both at +10/-3 on 120 losses);
screened against *different* halves of the problem they should not.

Arm B, in full:

    rerank_top_k          20 -> 40     retrieval depth, renders nothing
    context_facts_top_k      -> 20     caps printed facts, keeps the above free
    hydrate_pool             -> 40     question chooses turns, same turn count
    source_grounding         -> 0.003  tie-break, sized like adjacent ranks
    prompt                   -> gated answer-type conditioning

Screened width is 981 tokens against the shipped arm's 995, so B is expected to
come in cheaper as well as better. It is measured here rather than assumed, and
a B arm that measures over the ceiling is disqualified whatever it scores.

**Both arms are generated in this session.** Cross-run drift on this benchmark is
1.5 to 2 points and the effect being looked for is about that size, so comparing
B against saved answers would be comparing it against noise. RAG is not re-run:
its prompt and its retrieval are untouched by all of this, and `haystack.json`
holds its paired answer to every one of these questions already.

Read-only: both handles drop their reinforcement buffers rather than flushing.

Usage:
    python -u scripts/benchmarks/_ab_open_domain.py --sample 6 --out C:/nmafc_ab/od_smoke.json
    python -u scripts/benchmarks/_ab_open_domain.py --sample 250
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

from scipy.stats import binomtest  # noqa: E402

from nmafc.integration.answer_type import gate  # noqa: E402
from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks._ab_answer_type import PROMPT_B, answer_with  # noqa: E402
from scripts.benchmarks._ab_budget import (  # noqa: E402
    PERMANENT,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks.arms.base import SOURCE_RULE  # noqa: E402
from scripts.benchmarks.arms.neuromorphic_tuned import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
)
from scripts.benchmarks.datasets.locomo_loader import load_locomo  # noqa: E402
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402


def choose(args: argparse.Namespace) -> list[tuple[str, object]]:
    picked: list[tuple[str, object]] = []
    for conv in load_locomo():
        for qa in conv.qa_pairs:
            if qa.category_name == args.category:
                picked.append((conv.sample_id, qa))
    rng = random.Random(args.seed)
    rng.shuffle(picked)
    return picked[: args.sample] if args.sample else picked


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no merged store at {store}")

    # Two handles on one store. The arms differ in retrieval, so a shared
    # handle would let arm A's reads reach the weights arm B ranks against.
    # Hydration depth is per-arm now. It stopped being a constant the moment
    # the thing under test became cost: turns are the dearest line in the
    # render at about 55 tokens each, so an arm that cannot set its own turn
    # count cannot be made cheaper than the arm it is measured against.
    a_hydrate = args.hydrate if args.a_hydrate is None else args.a_hydrate
    mem_a = open_memory(store, llm, embedder, args.hot, args.cold,
                        args.a_rerank, args.max_hops, compact=True,
                        hydrate=a_hydrate, facts=args.a_facts,
                        dedupe=args.a_dedupe)
    mem_a._router._config.hydrate_pool = args.a_pool
    mem_a._router._config.source_grounding = args.a_grounding
    mem_a._router._config.hydrate_scan = args.a_scan
    mem_b = open_memory(store, llm, embedder, args.hot, args.cold, args.rerank,
                        args.max_hops, compact=True, hydrate=args.hydrate,
                        facts=args.facts, dedupe=args.dedupe)
    mem_b._router._config.hydrate_pool = args.pool
    mem_b._router._config.source_grounding = args.grounding
    mem_b._router._config.hydrate_scan = args.scan

    out = Path(args.out)
    rows: list[dict] = []
    if out.is_file() and not args.restart:
        rows = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(rows)} already answered\n")
    done = {(r["conv"], r["question"]) for r in rows}
    pending = [(c, qa) for c, qa in choose(args)
               if (c, qa.question) not in done]
    blocked: list[tuple[str, str]] = []

    def describe(label: str, rerank, facts, pool, ground, scan, typed,
                 hyd, dedupe) -> str:
        return (f"{label} rerank {rerank}, "
                f"{'all' if facts is None else facts} facts printed, "
                f"hydrate {hyd} from pool {pool} scan {scan}, "
                f"grounding {ground}, "
                f"{'deduped headers' if dedupe else 'headers as stored'}, "
                f"{'gated answer-type' if typed else 'shipped prompt'}")

    print(f"store: {store}")
    print(describe("A ", args.a_rerank, args.a_facts, args.a_pool,
                   args.a_grounding, args.a_scan, args.a_typed,
                   a_hydrate, args.a_dedupe))
    print(describe("B ", args.rerank, args.facts, args.pool,
                   args.grounding, args.scan, True,
                   args.hydrate, args.dedupe))
    print(f"{len(pending)} {args.category} questions\n")

    limit = asyncio.Semaphore(args.concurrency)

    async def one(conv_id: str, qa):
        rule, tag = gate(qa.question)
        prompt_b = PROMPT_B if rule else ANSWER_SYSTEM_PROMPT
        # Arm A carries the gated prompt only when it is being run as the
        # previous best rather than as the shipped arm. Passing the tag with
        # the shipped prompt would print an "Asked for:" line no rule explains.
        prompt_a = prompt_b if args.a_typed else ANSWER_SYSTEM_PROMPT
        tag_a = tag if args.a_typed else ""
        # Appended to B only, so A vs B isolates this one change when A is run
        # as the current best. Appended rather than spliced because the rules it
        # qualifies have to have been read first for "no fact names it" to mean
        # anything.
        if args.source_rule:
            prompt_b = prompt_b + SOURCE_RULE
        async with limit:
            try:
                a_text, a_chars = await with_retries(
                    lambda: answer_with(mem_a, llm, qa.question, prompt_a,
                                        tag_a))
                b_text, b_chars = await with_retries(
                    lambda: answer_with(mem_b, llm, qa.question, prompt_b, tag))
                a_v = await with_retries(
                    lambda: judge_answer(qa.question, a_text, str(qa.answer),
                                         judge))
                b_v = await with_retries(
                    lambda: judge_answer(qa.question, b_text, str(qa.answer),
                                         judge))
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ not in PERMANENT:
                    raise
                blocked.append((conv_id, qa.question))
                return None
        return {
            "conv": conv_id, "question": qa.question,
            "category": qa.category_name, "gold": str(qa.answer), "tag": tag,
            "a": a_text, "b": b_text,
            "a_chars": a_chars, "b_chars": b_chars,
            "a_ok": bool(a_v.correct), "b_ok": bool(b_v.correct),
        }

    try:
        tasks = [one(c, qa) for c, qa in pending]
        for start in range(0, len(tasks), args.checkpoint_every):
            batch = await asyncio.gather(
                *tasks[start:start + args.checkpoint_every])
            rows.extend(r for r in batch if r)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"  {len(rows)}/{len(pending) + len(done)} answered")
    finally:
        close_readonly(mem_a)
        close_readonly(mem_b)

    if not rows:
        print("nothing measured")
        return
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report(rows, blocked, args)
    print(f"\n-> {args.out}")


def report(rows: list[dict], blocked, args) -> None:
    def block(rs: list[dict], label: str) -> None:
        if not rs:
            return
        n = len(rs)
        a = sum(r["a_ok"] for r in rs)
        b = sum(r["b_ok"] for r in rs)
        fix = sum(1 for r in rs if r["b_ok"] and not r["a_ok"])
        brk = sum(1 for r in rs if r["a_ok"] and not r["b_ok"])
        p = binomtest(fix, fix + brk).pvalue if fix + brk else 1.0
        ac = sum(r["a_chars"] for r in rs) / n / 4
        bc = sum(r["b_chars"] for r in rs) / n / 4
        print(f"{label:<16}{n:>5}{a / n * 100:>8.1f}%{b / n * 100:>8.1f}%"
              f"{(b - a) / n * 100:>+8.1f}{fix:>6}{brk:>6}{p:>9.3g}"
              f"{ac:>8.0f}{bc:>8.0f}")

    print(f"\n{'':<16}{'n':>5}{'A':>9}{'B':>9}{'diff':>8}{'fix':>6}{'brk':>6}"
          f"{'p':>9}{'A ctx':>8}{'B ctx':>8}")
    print("-" * 84)
    block(rows, "ALL")
    # Split by whether the prompt half fired. Not an attribution -- retrieval
    # changed on every question in B -- but if the whole gain sits in the typed
    # slice then the retrieval half bought nothing and should be dropped rather
    # than shipped on the strength of a screen.
    block([r for r in rows if r["tag"]], "  typed")
    block([r for r in rows if not r["tag"]], "  untyped")

    widest = max(r["b_chars"] for r in rows) / 4
    mean = sum(r["b_chars"] for r in rows) / len(rows) / 4
    verdict = "under" if mean <= args.ceiling else "OVER"
    print(f"\nB context: mean {mean:.0f}t, widest single question {widest:.0f}t"
          f"  ({verdict} the {args.ceiling}t ceiling)")

    flipped = [r for r in rows if r["b_ok"] != r["a_ok"]]
    print(f"\n{len(flipped)} answers changed verdict. Regressions first:")
    for r in sorted(flipped, key=lambda r: r["b_ok"])[: args.show]:
        mark = "FIX " if r["b_ok"] else "BRK "
        print(f"\n  {mark}tag={r['tag'] or '(none)'}")
        print(f"    Q    {r['question']}")
        print(f"    gold {r['gold']}")
        print(f"    A    {r['a']}")
        print(f"    B    {r['b']}")
    if blocked:
        print(f"\n{len(blocked)} questions blocked by the provider")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--category", default="open-domain")
    ap.add_argument("--sample", type=int, default=250)
    # Arm A: the shipped settings, unchanged.
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hydrate", type=int, default=5)
    # Arm A's retrieval, spelled out. The defaults below are the shipped
    # settings, so a run that passes none of these compares "shipped" against
    # arm B exactly as every earlier run did.
    #
    # They exist because the comparison worth paying for has changed. Once a
    # retrieval stack is measured and kept, the next question is never "stack
    # versus shipped" again -- it is "does this one prompt change help, holding
    # retrieval fixed", and that needs A to be the current best rather than the
    # old default. Without these the only way to ask it was to edit the file.
    ap.add_argument("--a-hydrate", type=int, default=None,
                    help="Turns hydrated for arm A. Defaults to --hydrate, so "
                         "a run that does not set it holds turn count equal "
                         "and is comparing something else.")
    ap.add_argument("--a-dedupe", action="store_true")
    ap.add_argument("--a-rerank", type=int, default=20)
    ap.add_argument("--a-facts", type=int, default=None,
                    help="Cap on printed facts for arm A. Omit for uncapped, "
                         "which is what shipped does.")
    ap.add_argument("--a-pool", type=int, default=0)
    ap.add_argument("--a-grounding", type=float, default=0.0)
    ap.add_argument("--a-scan", type=int, default=0)
    ap.add_argument("--a-typed", action="store_true",
                    help="Give arm A the answer-type prompt and tag too. Set "
                         "this when the thing under test is NOT the typed "
                         "prompt, or A loses on a difference nobody asked "
                         "about and the result says nothing.")
    # Arm B: the stack.
    ap.add_argument("--rerank", type=int, default=40)
    ap.add_argument("--facts", type=int, default=20)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--grounding", type=float, default=0.003)
    # Off by default. The screen puts the knee at or below 8 and shows no gain
    # past it, so 8 is the setting to run, not the ceiling of a sweep.
    ap.add_argument("--scan", type=int, default=0,
                    help="Rank the whole conversation and offer the best N "
                         "turns to hydration, past what retrieval found")
    ap.add_argument("--ceiling", type=int, default=1000,
                    help="Our own budget, not RAG's. RAG spends 1,459 on "
                         "open-domain; the standing constraint is 1,000.")
    # Lossless: a session header repeats verbatim on every turn of that
    # session, and turns print in order, so one header per run of turns says
    # exactly what many did. Measured at 13% of the SOURCE block but only 13
    # tokens saved in practice, because the question scan picks turns scattered
    # across sessions and scattered turns share no header to drop.
    ap.add_argument("--dedupe", action="store_true",
                    help="Print a repeated session header once per run of turns")
    ap.add_argument("--source-rule", action="store_true",
                    help="Tell arm B that SOURCE exists and when to use it. "
                         "Screened at +4.8 upside against -10.0 exposure at "
                         "scan 8; both are bounds nobody hits.")
    ap.add_argument("--show", type=int, default=14)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="C:/nmafc_ab/open_domain.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
