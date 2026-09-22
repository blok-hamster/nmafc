"""Does a denser graph put the right fact in front of the model more often?

`resolve_link_targets` takes usable links from 0.92 to 1.50 per fact and drops
stranded facts from 31% to 9%. That is a property of the store, not a result:
the question that matters is whether retrieval, which traverses those links,
now reaches facts it previously could not.

This answers that without generating a single answer. It replays retrieval
against two sets of stores over identical questions and reports how often the
gold answer appears in the context that would have been sent to the model.
Generation and judging are the expensive halves of a run; retrieval is one
embedding call per question, and that call is shared between the two arms here,
so the whole comparison costs about what one arm's embeddings cost.

Sharing the query vector matters for more than cost. Embedding is deterministic
in principle but the comparison is between two retrievals that differ by a
handful of links, and re-embedding would introduce a second difference between
the arms for no reason. One vector, two stores, one variable.

What a positive result means, and what it does not:

  * "In context" is keyword overlap against the gold answer, the same crude
    test the other diagnostics use. It over-reports -- the words can appear in
    an unrelated fact -- so a gain here is necessary evidence that the fix
    reaches more answers, not sufficient evidence that accuracy improves. The
    model still has to use what it is given.
  * A null result is much more informative. If the same facts reach the prompt
    either way, the graph work cannot be improving accuracy, and there is no
    point paying for a full rerun to find that out.

Usage:
    python scripts/benchmarks/_ab_link_repair.py \
        --stores-a scripts/benchmarks/results/full_v3/stores \
        --stores-b /tmp/repaired/stores
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
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
from nmafc.schemas.memory import DecayConfig  # noqa: E402
from nmafc.storage.config import NMafcConfig, StorageConfig  # noqa: E402
from nmafc.wrapper import NeuromorphicMemory  # noqa: E402

from datasets.locomo_loader import load_locomo  # noqa: E402

STOP = set(
    "the a an of in on at to for is was were and or with what which who when "
    "where how did does do had has have her his its their this that".split()
)

SCORED = {"single-hop", "temporal", "multi-hop", "open-domain"}


def keywords(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9']+", str(text).lower())
        if len(w) > 2 and w not in STOP
    }


def overlap(gold: str, haystack: str) -> float:
    kw = keywords(gold)
    if not kw:
        return 0.0
    low = haystack.lower()
    return sum(1 for w in kw if w in low) / len(kw)


class CachingEmbedder:
    """Embed each distinct text once, then serve both arms from memory.

    Wraps rather than subclasses the provider: the arms only ever go through
    `embed` and `embed_single`, and anything else the real provider exposes is
    forwarded untouched.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def embed_single(self, text: str) -> list[float]:
        if text not in self._cache:
            self.calls += 1
            self._cache[text] = await self._inner.embed_single(text)
        return self._cache[text]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self.calls += 1
            for text, vector in zip(missing, await self._inner.embed(missing)):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]


async def retrieve_with_retry(router, question: str, turn: int, attempts: int = 5):
    """Retrieve, retrying transient network failures rather than scoring a miss."""
    for attempt in range(attempts):
        try:
            return await router.retrieve(question, turn)
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"      [retry {attempt + 1} in {wait}s] {type(exc).__name__}")
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


def open_memory(store: Path, llm, embedder, args) -> NeuromorphicMemory:
    config = NMafcConfig(
        storage=StorageConfig(
            hot_uri=str(store / "hot_lancedb"),
            cold_uri=str(store / "cold.db"),
        ),
        decay=DecayConfig(
            max_hops=args.max_hops,
            beta=args.beta,
            # Retrieval reinforces what it returns, rewriting each surviving
            # record. Left on, a diagnostic would edit the stores it is meant to
            # be measuring, and arm A here is the real run's output. Deferring
            # the writeback and never flushing makes both arms read-only. It
            # costs nothing in fidelity at the default weight_signal of 0, where
            # weight does not enter ranking.
            defer_reinforcement_writes=True,
        ),
    )
    return NeuromorphicMemory(
        llm_provider=llm, embedding_provider=embedder, config=config
    )


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ.get("NMAFC_BENCH_PROVIDER", "ollama/llama3.2"))
    embedder = CachingEmbedder(
        create_embedding_provider(
            os.environ.get("NMAFC_BENCH_EMBEDDING", "ollama/nomic-embed-text")
        )
    )

    totals = {"n": 0, "a": 0, "b": 0, "b_only": 0, "a_only": 0}
    # Multi-hop is the category the graph is supposed to serve, so an aggregate
    # that mixes it with single-hop can hide a real effect inside a null.
    by_cat: dict[str, dict[str, int]] = {}
    chars = {"a": 0, "b": 0}
    gained: list[tuple[str, str]] = []
    lost: list[tuple[str, str]] = []

    for conv in load_locomo():
        questions = [qa for qa in conv.qa_pairs if qa.category_name in SCORED]
        if args.limit:
            questions = questions[: args.limit]
        if not questions:
            continue

        store_a = Path(args.stores_a) / f"{args.arm}__{conv.sample_id}"
        store_b = Path(args.stores_b) / f"{args.arm}__{conv.sample_id}"
        if not store_a.is_dir() or not store_b.is_dir():
            print(f"  [skip] missing store for {conv.sample_id}")
            continue

        mem_a = open_memory(store_a, llm, embedder, args)
        mem_b = open_memory(store_b, llm, embedder, args)
        try:
            turn = max(mem_a.current_turn, mem_b.current_turn) + 1
            local = {"n": 0, "a": 0, "b": 0}
            for qa in questions:
                ctx_a = mem_a._router.format_context(
                    await retrieve_with_retry(mem_a._router, qa.question, turn)
                )
                ctx_b = mem_b._router.format_context(
                    await retrieve_with_retry(mem_b._router, qa.question, turn)
                )
                hit_a = overlap(qa.answer, ctx_a) >= args.threshold
                hit_b = overlap(qa.answer, ctx_b) >= args.threshold

                cat = by_cat.setdefault(
                    qa.category_name, {"n": 0, "a": 0, "b": 0}
                )
                cat["n"] += 1
                cat["a"] += hit_a
                cat["b"] += hit_b
                chars["a"] += len(ctx_a)
                chars["b"] += len(ctx_b)

                local["n"] += 1
                local["a"] += hit_a
                local["b"] += hit_b
                totals["n"] += 1
                totals["a"] += hit_a
                totals["b"] += hit_b
                if hit_b and not hit_a:
                    totals["b_only"] += 1
                    if len(gained) < 8:
                        gained.append((qa.question, str(qa.answer)))
                elif hit_a and not hit_b:
                    totals["a_only"] += 1
                    if len(lost) < 8:
                        lost.append((qa.question, str(qa.answer)))
            print(f"  {conv.sample_id}: {local['a']}/{local['n']} -> "
                  f"{local['b']}/{local['n']} in context")
        finally:
            mem_a.close()
            mem_b.close()

    n = totals["n"]
    if not n:
        print("\nnothing compared")
        return

    print(f"\n{'=' * 62}")
    print(f"questions compared          : {n}")
    print(f"embedding calls made        : {embedder.calls} (shared between arms)")
    print(f"  A  gold answer in context : {totals['a']:5d}  {totals['a'] / n:6.1%}")
    print(f"  B  gold answer in context : {totals['b']:5d}  {totals['b'] / n:6.1%}")
    print(f"  change                    : {totals['b'] - totals['a']:+5d}  "
          f"{(totals['b'] - totals['a']) / n:+6.1%}")
    print(f"\n  reached only by B (gained): {totals['b_only']}")
    print(f"  reached only by A (lost)  : {totals['a_only']}")

    print(f"\n  {'category':14s} {'n':>5s} {'A':>8s} {'B':>8s} {'change':>8s}")
    for name in ("single-hop", "temporal", "multi-hop", "open-domain"):
        c = by_cat.get(name)
        if not c:
            continue
        print(f"  {name:14s} {c['n']:5d} {c['a'] / c['n']:7.1%} "
              f"{c['b'] / c['n']:7.1%} {(c['b'] - c['a']) / c['n']:+7.1%}")

    print(f"\n  mean context chars        : A {chars['a'] / n:7.0f}  "
          f"B {chars['b'] / n:7.0f}")

    if gained:
        print("\n  examples the repaired graph reached and the old one did not:")
        for q, g in gained:
            print(f"    Q: {q[:82]}")
            print(f"       gold: {g[:70]}")
    if lost:
        print("\n  examples the repair lost:")
        for q, g in lost:
            print(f"    Q: {q[:82]}")
            print(f"       gold: {g[:70]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stores-a", required=True, help="baseline stores directory")
    ap.add_argument("--stores-b", required=True, help="repaired stores directory")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="gold keyword overlap counted as present in context")
    ap.add_argument("--limit", type=int, default=0,
                    help="questions per conversation, 0 for all")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
