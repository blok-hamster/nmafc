"""Re-classify already-extracted facts, without re-reading a single conversation.

Extraction and classification are two different jobs that the pipeline happens
to do in one call. Turning conversations into facts is the expensive one: 5,882
LoCoMo turns took 5.3 hours of LLM work. Deciding whether an existing fact is
permanent or temporary reads the fact, not the conversation, and there are only
3,807 of them. Batched forty to a call that is about ninety-five calls.

So the 68.4% CoreAnchor rate does not need re-ingestion to attack. It needs the
facts we already have, shown to a model whose only job is the tier.

The rules are sliced out of the live extraction prompt rather than restated
here. Restating them would let the relabel drift away from what extraction was
actually told, and then a difference in the output would be a difference in the
instructions rather than a difference in doing one job instead of two.

Nothing is written to the source run. The stores are copied first and every
edit lands on the copy, because these stores cost 5.3 hours and there is no
second copy of them.

What this can and cannot buy is worth stating plainly before it is run. The
tier sets a decay rate, decay sets a weight, and `weight_signal` is 0, so the
weight does not enter ranking at all. The tier also decides what pruning
demotes out of Hot RAM, and that does change what retrieval sees. So a null
result here is the expected result, and would be evidence about the plumbing
rather than about the classifier.

Usage:
    python -u scripts/benchmarks/_relabel_tiers.py --limit-stores 1 --dry-run
    python -u scripts/benchmarks/_relabel_tiers.py --dst /c/nmafc_ab/relabelled
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sqlite3
import sys
from collections import Counter
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

from nmafc.integration.extractor import EXTRACTION_SYSTEM_PROMPT  # noqa: E402
from nmafc.integration.factory import create_llm_provider  # noqa: E402

from scripts.benchmarks._ab_budget import with_retries  # noqa: E402

TIERS = ("CoreAnchor", "ActiveContext", "EphemeralState")


def classification_rules() -> str:
    """The tier rules exactly as extraction receives them.

    Sliced from the live prompt so the two cannot diverge. If the heading ever
    moves this raises rather than silently relabelling against nothing, because
    a classifier running on an empty rulebook would look like a working one.
    """
    start = EXTRACTION_SYSTEM_PROMPT.find("## Memory Classification Rules:")
    if start < 0:
        raise RuntimeError("classification rules not found in extraction prompt")
    end = EXTRACTION_SYSTEM_PROMPT.find("## ", start + 5)
    rules = EXTRACTION_SYSTEM_PROMPT[start:end if end > 0 else None].strip()
    if len(rules) < 200:
        raise RuntimeError(f"classification rules look truncated ({len(rules)} chars)")
    return rules


SYSTEM = """You classify stored memory facts into one of three tiers.

{rules}

You will be given a numbered list of facts. Reply with ONLY a JSON array of
objects, one per fact, in the same order:

[{{"n": 1, "tier": "CoreAnchor"}}, {{"n": 2, "tier": "EphemeralState"}}]

No prose, no code fence. Every fact must get exactly one tier."""


def parse_tiers(text: str, expected: int) -> list[str] | None:
    """The array out of a reply, or None if it cannot be trusted.

    A short or malformed batch is rejected whole rather than partially applied:
    a misaligned array would relabel facts against other facts' verdicts, which
    is worse than not relabelling them at all.
    """
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) != expected:
        return None
    out = []
    for item in parsed:
        tier = (item or {}).get("tier") if isinstance(item, dict) else None
        if tier not in TIERS:
            return None
        out.append(tier)
    return out


async def classify(llm, facts: list[str], batch: int, rules: str) -> list[str] | None:
    """A tier for every fact, or None if any batch could not be parsed."""
    labels: list[str] = []
    for start in range(0, len(facts), batch):
        window = facts[start:start + batch]
        listing = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(window))
        text = await with_retries(lambda: llm.chat(
            messages=[{"role": "user", "content": listing}],
            system_prompt=SYSTEM.format(rules=rules),
        ))
        got = parse_tiers(text, len(window))
        if got is None:
            print(f"      [unparseable batch of {len(window)}, store skipped]")
            return None
        labels.extend(got)
    return labels


def relabel_cold(cold: Path, mapping: dict[str, str]) -> int:
    """Apply the new tiers to the archive's event log.

    Cold is searched on every query at the current settings, so leaving it on
    the old labels would relabel only half the store and make the screen that
    follows uninterpretable.
    """
    if not cold.is_file():
        return 0
    conn = sqlite3.connect(cold)
    changed = 0
    try:
        for fact, tier in mapping.items():
            cur = conn.execute(
                "UPDATE memory_event_log SET memory_type = ? "
                "WHERE fact_content = ? AND memory_type != ?",
                (tier, fact, tier),
            )
            changed += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return changed


async def run(args: argparse.Namespace) -> None:
    rules = classification_rules()
    print(f"tier rules: {len(rules)} chars, sliced from the live extraction prompt\n")
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])

    src = Path(args.src) / "stores"
    dst = Path(args.dst)
    stores = sorted(p for p in src.iterdir() if p.is_dir())
    if args.limit_stores:
        stores = stores[: args.limit_stores]

    before: Counter = Counter()
    after: Counter = Counter()
    moves: Counter = Counter()
    calls = 0

    for store in stores:
        table_dir = store / "hot_lancedb"
        if not table_dir.is_dir():
            continue

        target = dst / store.name
        if not args.dry_run:
            # Copy first. These stores cost 5.3 hours and are not reproducible
            # from anything cheaper, so nothing writes to the originals.
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(store, target)
            table_dir = target / "hot_lancedb"

        db = lancedb.connect(str(table_dir))
        table = db.open_table("memory_vectors")
        frame = table.to_pandas()
        facts = frame["fact_content"].astype(str).tolist()
        before.update(frame["memory_type"].tolist())

        print(f"[{store.name}] {len(facts)} facts")
        calls += (len(facts) + args.batch - 1) // args.batch
        labels = await classify(llm, facts, args.batch, rules)
        if labels is None:
            continue

        for old, new in zip(frame["memory_type"].tolist(), labels):
            if old != new:
                moves[f"{old} -> {new}"] += 1
        after.update(labels)

        if args.dry_run:
            continue

        frame["memory_type"] = labels
        db.create_table("memory_vectors", data=frame, mode="overwrite")
        changed = relabel_cold(target / "cold.db",
                               dict(zip(facts, labels)))
        print(f"    hot rewritten, cold rows updated: {changed}")

    def show(title: str, counts: Counter) -> None:
        total = sum(counts.values()) or 1
        print(f"\n  {title}")
        for tier in TIERS:
            print(f"    {tier:16s} {counts[tier]:5d}  {100 * counts[tier] / total:5.1f}%")

    print(f"\n{'=' * 70}")
    print(f"llm calls: {calls}")
    show("before", before)
    show("after", after)
    total = sum(before.values()) or 1
    agree = total - sum(moves.values())
    print(f"\n  unchanged: {agree}/{total} ({100 * agree / total:.1f}%)")
    if moves:
        print("\n  changes")
        for move, n in moves.most_common():
            print(f"    {move:34s} {n:5d}")
    if not args.dry_run:
        print(f"\nrelabelled stores written to {dst}")
        print("source run untouched")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="scripts/benchmarks/results/full_v3")
    ap.add_argument("--dst", default="/c/nmafc_ab/relabelled")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--limit-stores", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Classify and report, but write nothing.")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
