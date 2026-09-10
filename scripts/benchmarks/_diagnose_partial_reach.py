"""Did we give the model the whole answer, or only part of it?

The deficit has been split into "the gold never arrived" and "the gold arrived
and the model failed to assemble it", and the second half has been treated as a
prompt problem. That split was never checked. `present()` asks whether *every*
content word of the gold is in the context and returns one bit, so a question
where we supplied "showed him his work" and the gold was "showed him his work
and gave him advice" is counted the same as one where we supplied nothing.

Those are opposite problems. If the context holds the whole answer and the model
gives half, that is the model, and the fix is instruction, which costs money to
test. If the context holds half the answer, that is us, and the fix is retrieval,
which is free to screen. The two have been pooled, and the paid experiment was
about to be chosen without knowing which one is bigger.

So this measures coverage rather than presence: what fraction of the gold's
content words reached the context. Then it buckets the questions we lose:

    nothing        no content word arrived -- the gold is simply not there
    fragment       under half arrived
    most of it     half or more but not all -- we dropped a piece
    all of it      everything arrived and we still got it wrong

Only the last bucket is a prompt problem. The one before it is ours.

Reads a saved results file, re-renders the context for the config given, and
scores nothing. No generation.

Usage:
    python -u scripts/benchmarks/_diagnose_partial_reach.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

from scripts.benchmarks._ab_budget import (  # noqa: E402
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks._screen_hydrate_pool import content_words  # noqa: E402


def coverage(gold: str, context: str) -> float:
    """Fraction of the gold's content words that reached the context."""
    wanted = content_words(gold)
    if not wanted:
        return 1.0
    return len(wanted & content_words(context)) / len(wanted)


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = [r for r in rows if r["category"] == args.category]
    if args.gate in ("gated", "ungated"):
        # The two paths fail differently -- the short-answer path picks the
        # wrong one of several similar candidates, the other under-answers --
        # so pooling them hides both. Split on the same gate the run uses.
        from scripts.benchmarks._run_open_domain_full import gate
        want = args.gate == "gated"
        rows = [r for r in rows if bool(gate(r["question"])[0]) == want]
    ours = [r for r in rows if not r.get("ok", r.get("ours_ok"))]
    # The questions that decide the comparison: RAG answers them, we do not.
    # `--both-wrong` asks a different question, and on single-hop it is the one
    # that matters: 36% of that category is failed by both arms, which is three
    # times the open-domain rate. A cell that large is not a disadvantage
    # against RAG, it is something the task is doing to both of us, and looking
    # only at the questions RAG wins would never show it.
    if args.both_wrong:
        target = [r for r in ours if not r.get("rag_ok")]
    elif args.beaten_only:
        target = [r for r in ours if r.get("rag_ok")]
    else:
        target = ours
    if args.limit:
        target = target[: args.limit]

    # One store or ten. The merged haystack is a single store and every earlier
    # use of this file assumed that; real LoCoMo is per-conversation, and a row
    # carries the conversation it came from, so grouping by it lets the same
    # diagnosis run on either without two copies of the code.
    if args.store == "per-conv":
        groups = defaultdict(list)
        for row in target:
            groups[row["conv"]].append(row)
    else:
        groups = {args.store: target}

    covs: list[tuple[float, dict]] = []
    for name, group in groups.items():
        store = Path(args.run) / "stores" / f"{args.arm}__{name}"
        memory = open_memory(store, llm, embedder, args.hot, args.cold,
                             args.rerank, args.max_hops, compact=True,
                             hydrate=args.hydrate, facts=args.facts,
                             dedupe=True, overlap=args.overlap)
        cfg = memory._router._config
        cfg.hydrate_pool = args.pool
        cfg.source_grounding = args.grounding
        cfg.hydrate_scan = args.scan
        cfg.scan_semantic = args.semantic
        cfg.scan_semantic_floor = args.semantic_floor
        try:
            turn = memory.current_turn + 1
            for row in group:
                recs = await with_retries(
                    lambda q=row["question"]: memory._router.retrieve(q, turn))
                ctx = memory._router.format_context(recs or [], row["question"])
                covs.append((coverage(row["gold"], ctx), row))
            print(f"  {name}: {len(group)} done", flush=True)
        finally:
            close_readonly(memory)

    n = len(covs)
    buckets = {
        "nothing arrived": [c for c in covs if c[0] == 0.0],
        "a fragment": [c for c in covs if 0.0 < c[0] < 0.5],
        "most of it, a piece missing": [c for c in covs if 0.5 <= c[0] < 1.0],
        "all of it": [c for c in covs if c[0] >= 1.0],
    }
    label = ("questions neither arm answers" if args.both_wrong else
             "questions RAG answers and we do not" if args.beaten_only
             else "questions we get wrong")
    print(f"\n{n} {label}, config: facts {args.facts}, hydrate {args.hydrate}, "
          f"semantic {args.semantic}\n")
    for name, got in buckets.items():
        print(f"  {name:<30}{len(got):>5}  {100 * len(got) / max(n, 1):>5.1f}%")
    ours_fault = len(buckets["a fragment"]) + len(
        buckets["most of it, a piece missing"])
    print(f"\n  a retrieval problem (something arrived, not all)"
          f"{ours_fault:>6}  {100 * ours_fault / max(n, 1):>5.1f}%")
    print(f"  a prompt problem (all of it arrived)         "
          f"{len(buckets['all of it']):>6}"
          f"  {100 * len(buckets['all of it']) / max(n, 1):>5.1f}%")

    print("\n  Where a piece is missing, the piece:")
    for cov, row in sorted(buckets["most of it, a piece missing"],
                           key=lambda c: -c[0])[: args.show]:
        print(f"\n    Q  {row['question']}")
        print(f"    A  {row['gold']}   ({100 * cov:.0f}% of it reached)")
        print(f"    we said  {row.get('pred', row.get('ours_pred', ''))[:90]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/od_full2.json")
    ap.add_argument("--category", default="open-domain")
    ap.add_argument("--beaten-only", action="store_true", default=True)
    ap.add_argument("--all-losses", dest="beaten_only", action="store_false")
    ap.add_argument("--both-wrong", action="store_true",
                    help="Only the questions neither arm answers. Overrides "
                         "--beaten-only.")
    ap.add_argument("--gate", default="all",
                    choices=("all", "gated", "ungated"),
                    help="restrict to questions where a length rule does or "
                         "does not fire")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--rerank", type=int, default=40)
    ap.add_argument("--facts", type=int, default=20)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--grounding", type=float, default=0.03)
    ap.add_argument("--scan", type=int, default=8)
    ap.add_argument("--hydrate", type=int, default=6)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--semantic", type=float, default=12.0)
    ap.add_argument("--semantic-floor", type=float, default=0.30)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
