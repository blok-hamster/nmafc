"""Which retrieval settings pull the answering record into the pool, for free?

The probe split the open-domain losses three ways: the answer was already in the
prompt for 47 of them, it was in the store but never surfaced for 21, and it was
never written down for 33. Hydration attacks the first bucket and extraction the
third. This is the second bucket, and it is the only lever on the list that
costs nothing at inference -- a record that was always there, ranked into the
pool instead of out of it, is accuracy at zero tokens.

Screening it cheaply needs one idea. A sweep that judges every configuration
pays a generation-sized bill per configuration, which is what makes retrieval
tuning expensive and what has limited every previous sweep to a handful of
settings. But the judge is only being asked the same question each time -- is
the answering record in this pool -- and the answering record does not change
when the configuration does. So identify it once, by text, and every
configuration after that is scored by looking for that text in the pool. One
judge pass, then arithmetic.

Two passes, and the first is the only one that spends anything:

    stage 1   for each loss, shortlist the store by meaning and ask the judge
              which shortlisted record states the gold answer. Records the
              record's text. About 116 judge calls, once, reusable forever.

    stage 2   for each configuration, retrieve and report where that text
              lands. One embedding per question per configuration and no
              generation at all.

Rank matters more than membership, and by two different thresholds. A record
ranked inside `context_facts_top_k` is written into the prompt as a fact. A
record ranked inside `hydrate_top_k` is not written as a fact but its source
turn is, which at the settings that survive the token budget is the wider and
more useful of the two. So the report gives both.

What it cannot tell you is whether the model then answers correctly, and it
cannot tell you what a configuration costs the questions we currently win.
Anything that looks good here still needs a paired generation A/B, both arms in
one session, because the saved baselines have drifted.

Usage:
    python -u scripts/benchmarks/_screen_ranking.py --stage targets --limit 5
    python -u scripts/benchmarks/_screen_ranking.py --stage sweep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

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
from scripts.benchmarks._probe_stored_vs_reached import (  # noqa: E402
    StoreIndex,
    best_candidates,
)
from scripts.benchmarks._sweep_context_budget import retrieve_with_retry  # noqa: E402

PICK_SYSTEM = """\
You are given a QUESTION, the ANSWER that is correct, and a numbered list of
CANDIDATE records from a memory system's database.

Pick the single candidate that states the ANSWER to the QUESTION. Wording may
differ as long as the information is the same: "is trying to get his car fixed"
states "tried to get it fixed". Do not pick a candidate that is merely about the
same topic, gives a different or earlier value, or would require guessing. If no
candidate states the answer, say so rather than picking the closest one -- a
wrong pick here becomes a retrieval target that can never be worth reaching.

Reply with JSON and nothing else: {"pick": <number>} or {"pick": null}"""


async def pick(llm, question: str, gold: str, candidates: list[str]) -> int | None:
    listing = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))
    text = await llm.chat(
        messages=[{"role": "user", "content":
                   f"QUESTION: {question}\nANSWER: {gold}\n\n"
                   f"CANDIDATES:\n{listing or '(nothing)'}"}],
        system_prompt=PICK_SYSTEM)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        value = json.loads(text[start:end + 1]).get("pick")
    except json.JSONDecodeError:
        return None
    if not isinstance(value, int) or not 0 <= value < len(candidates):
        return None
    return value


async def find_targets(args, llm, embedder, items, store) -> list[dict]:
    """Stage 1. The answering record for each loss, by text, judged once."""
    index = StoreIndex(store)
    memory = open_memory(store, llm, embedder, args.hot, args.cold,
                         args.budget, args.max_hops, compact=True, hydrate=0)
    gate = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []
    print(f"stage 1: {len(items)} questions against {len(index.facts)} facts")
    try:
        router = memory._router
        turn = memory.current_turn + 1

        async def one(item: dict) -> dict:
            pool = await retrieve_with_retry(router, item["question"], turn)
            surfaced = [r.fact_content for r in (pool or [])]
            # Question and answer together, as in the probe. The question alone
            # ranks by topic and can miss the record holding the value; the
            # answer alone is useless when it is one word.
            probe = f"{item['question']} {item['gold']}"
            vector = np.asarray(
                (await with_retries(lambda: embedder.embed([probe])))[0],
                dtype=float)
            shortlist = best_candidates(
                index.nearest(vector, args.candidates), surfaced)
            async with gate:
                chosen = await with_retries(
                    lambda: pick(llm, item["question"], item["gold"],
                                 shortlist))
            return {"question": item["question"], "gold": item["gold"],
                    "target": None if chosen is None else shortlist[chosen],
                    # Where the shipped configuration put it, so stage 2 has a
                    # baseline that needed no second retrieval.
                    "shipped_rank": (surfaced.index(shortlist[chosen])
                                     if chosen is not None
                                     and shortlist[chosen] in surfaced
                                     else None)}

        for start in range(0, len(items), args.checkpoint_every):
            batch = await asyncio.gather(
                *(one(i) for i in items[start:start + args.checkpoint_every]))
            rows.extend(batch)
            print(f"  {len(rows)}/{len(items)}")
    finally:
        close_readonly(memory)
    return rows


async def rank_under(args, llm, embedder, targets, store, setting) -> list[int | None]:
    """Stage 2. Where each target lands under one configuration. No generation."""
    hot, cold, budget, hops = setting
    memory = open_memory(store, llm, embedder, hot, cold, budget, hops,
                         compact=True, hydrate=0)
    ranks: list[int | None] = []
    try:
        router = memory._router
        turn = memory.current_turn + 1
        for row in targets:
            pool = await retrieve_with_retry(router, row["question"], turn)
            surfaced = [r.fact_content for r in (pool or [])]
            ranks.append(surfaced.index(row["target"])
                         if row["target"] in surfaced else None)
    finally:
        close_readonly(memory)
    return ranks


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no store at {store}")
    out = Path(args.targets)

    if args.stage == "targets":
        answered = json.loads(Path(args.results).read_text(encoding="utf-8"))
        items = [r for r in answered if r["category"] == args.category
                 and r["rag_ok"] and not r["ours_ok"]]
        if args.limit:
            items = items[: args.limit]
        rows = await find_targets(args, llm, embedder, items, store)
        found = [r for r in rows if r["target"]]
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n  {len(found)}/{len(rows)} losses have an answering record "
              f"in the store")
        inside = sum(1 for r in found if r["shipped_rank"] is not None)
        print(f"  {inside} of those are already in the shipped pool of "
              f"{args.budget}, {len(found) - inside} are not reached at all")
        print(f"\n-> {args.targets}")
        return

    if not out.is_file():
        raise SystemExit(f"run --stage targets first, no {args.targets}")
    targets = [r for r in json.loads(out.read_text(encoding="utf-8"))
               if r["target"]]
    if args.limit:
        targets = targets[: args.limit]
    n = len(targets)
    settings = [tuple(int(x) for x in s.split(":"))
                for s in args.settings.split(",")]

    print(f"stage 2: {n} answerable losses, {len(settings)} configurations, "
          f"no generation\n")
    print(f"  {'hot':>4s} {'cold':>5s} {'pool':>5s} {'hops':>5s} "
          f"{'in pool':>9s} {'top 20':>8s} {'top 6':>7s}")
    for setting in settings:
        ranks = await rank_under(args, llm, embedder, targets, store, setting)
        found = [r for r in ranks if r is not None]
        # Two thresholds because they buy different things. Inside the hydrate
        # depth the record's source turn reaches the prompt; inside the fact
        # count the record itself is written out as well.
        top20 = sum(1 for r in found if r < args.hydrate_depth)
        top6 = sum(1 for r in found if r < args.fact_depth)
        print(f"  {setting[0]:4d} {setting[1]:5d} {setting[2]:5d} "
              f"{setting[3]:5d} {len(found):5d} {100 * len(found) / n:4.0f}% "
              f"{top20:4d} {100 * top20 / n:3.0f}% {top6:3d} "
              f"{100 * top6 / n:3.0f}%")

    print(f"\n  Reaching a record is not answering with it. Anything that gains")
    print(f"  here still needs a paired generation A/B, both arms in one")
    print(f"  session, and a check that it costs nothing on the questions we")
    print(f"  already win.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("targets", "sweep"), default="targets")
    ap.add_argument("--results", default="C:/nmafc_ab/haystack.json")
    ap.add_argument("--category", default="open-domain")
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--targets", default="C:/nmafc_ab/ranking_targets.json")
    ap.add_argument("--settings", default="10:20:20:2,20:20:20:2,10:40:20:2,"
                                          "10:20:20:3,20:40:20:3,10:20:40:2",
                    help="hot:cold:pool:hops per configuration.")
    ap.add_argument("--hydrate-depth", type=int, default=20,
                    help="Rank below which a record's source turn is hydrated.")
    ap.add_argument("--fact-depth", type=int, default=6,
                    help="Rank below which the record is written as a fact.")
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
