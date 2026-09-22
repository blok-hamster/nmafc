"""Would ranking turns by meaning find the turns BM25 cannot?

The scan exists because a turn whose facts all rank below the cut is otherwise
unreachable, and it ranks those turns by BM25. That was the right choice for the
failure it was built for -- a concrete noun the extractor generalised away,
"Hoodies" where we answered "clothing" -- because BM25 is exactly what rewards a
rare word. It is the wrong choice for the failure that is left.

The shape of what is left, from the paired run: of the questions RAG answers and
we do not, 60% have no content word of the gold anywhere in the rendered
context. Reach on the losses sits at about 45% and every retrieval sweep run
this session plateaus there. Bounding the scan to the right conversation moved
it 1.2 points. So the gold turn is not being outranked by a distractor nearly as
often as it is not being *recognised*: the question and the turn say the same
thing in different words, and a bag of words scores that at zero.

This measures the ranker rather than building it. For each question we lose, it
finds the turn actually holding the gold and asks where each ranker puts it:

    BM25          what `_scanned_turns` uses today
    cosine        the same embedding model the store was built with
    both          the better rank of the two, which is what a blend could reach

If cosine ranks the gold turn inside the scan window on questions where BM25
does not, the lever is real and worth wiring into the router. If it does not,
turn ranking is finished as a lever and the deficit is not a retrieval problem.

Embeddings for the turns are computed once and cached beside the store, so a
re-run costs nothing. There is no generation here and no answer is scored.

Usage:
    python -u scripts/benchmarks/_screen_turn_embeddings.py --limit 178
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
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

from nmafc.integration.factory import create_embedding_provider  # noqa: E402
from nmafc.integration.grounding import source_scores  # noqa: E402

from scripts.benchmarks._screen_hydrate_pool import content_words  # noqa: E402


def turn_text(store: Path) -> dict[int, str]:
    conn = sqlite3.connect(f"file:{store / 'cold.db'}?mode=ro", uri=True)
    rows = conn.execute("SELECT turn, text FROM turn_text").fetchall()
    conn.close()
    return {int(t): x for t, x in rows}


async def embed_turns(embedder, texts: dict[int, str],
                      cache: Path) -> dict[int, list[float]]:
    """Embed every turn once and keep the result next to the store.

    Cached on the turn count and the model name, so a store that grows or a
    change of embedding model recomputes rather than quietly comparing vectors
    from two different spaces.
    """
    model = os.environ.get("NMAFC_BENCH_EMBEDDING", "unknown")
    key = f"{model}:{len(texts)}"
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        if blob.get("key") == key:
            return {int(t): v for t, v in blob["vectors"].items()}
    order = sorted(texts)
    vectors: list[list[float]] = []
    # Batched because the provider batches internally at 2048 but the request
    # body still has to fit, and a partial failure halfway through 2,957 turns
    # should cost one batch rather than all of them.
    for i in range(0, len(order), 256):
        chunk = [texts[t] for t in order[i:i + 256]]
        vectors.extend(await embedder.embed(chunk))
        print(f"  embedded {min(i + 256, len(order))}/{len(order)} turns",
              flush=True)
    out = dict(zip(order, vectors))
    cache.write_text(json.dumps({"key": key,
                                 "vectors": {str(t): v for t, v in out.items()}}),
                     encoding="utf-8")
    return out


def cosine_ranks(qv: list[float], vecs: dict[int, list[float]]) -> list[int]:
    # The provider returns unit vectors, so the dot product is the cosine and
    # normalising again would only cost time.
    scored = ((sum(a * b for a, b in zip(qv, v)), t) for t, v in vecs.items())
    return [t for _, t in sorted(scored, reverse=True)]


def rank_of(gold: set[int], ranked: list[int]) -> int | None:
    for i, t in enumerate(ranked):
        if t in gold:
            return i + 1
    return None


async def run(args: argparse.Namespace) -> None:
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    texts = turn_text(store)
    print(f"{len(texts)} turns in the store")
    vecs = await embed_turns(embedder, texts, store / "turn_vectors.json")

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    # Two result shapes exist: the run's own rows call the verdict `ok`, the
    # file reshaped for the reach screens calls it `ours_ok`.
    rows = [r for r in rows if r["category"] == args.category
            and not r.get("ok", r.get("ours_ok"))]
    if args.limit:
        rows = rows[: args.limit]

    buckets = [1, 3, 8, 20, 50]
    hits = {name: dict.fromkeys(buckets, 0) for name in ("bm25", "cos", "both")}
    only_cos = []
    n = 0
    for row in rows:
        wanted = content_words(row["gold"])
        if not wanted:
            continue
        gold = {t for t, x in texts.items() if wanted <= content_words(x)}
        if not gold:
            continue
        n += 1
        bm = source_scores(row["question"], texts)
        bm_ranked = sorted(texts, key=lambda t: (-bm.get(t, 0.0), t))
        qv = await embedder.embed_single(row["question"])
        cos_ranked = cosine_ranks(qv, vecs)
        rb, rc = rank_of(gold, bm_ranked), rank_of(gold, cos_ranked)
        for k in buckets:
            hits["bm25"][k] += bool(rb and rb <= k)
            hits["cos"][k] += bool(rc and rc <= k)
            hits["both"][k] += bool((rb and rb <= k) or (rc and rc <= k))
        if rc and rc <= 8 and (not rb or rb > 8):
            only_cos.append((row["question"], row["gold"], rb, rc))
        if n % 20 == 0:
            print(f"  {n} questions", flush=True)

    print(f"\n{n} of {len(rows)} losses have the gold in some turn. "
          f"Store is {len(texts)} turns.")
    print("\n  gold turn ranked inside the top")
    print(f"  {'ranker':<10}" + "".join(f"{k:>8}" for k in buckets))
    for name, label in (("bm25", "BM25"), ("cos", "cosine"), ("both", "better")):
        print(f"  {label:<10}" + "".join(
            f"{100 * hits[name][k] / max(n, 1):>7.1f}%" for k in buckets))
    print(f"\n  The scan takes {args.scan}, so the column that decides this is "
          f"{args.scan}.")
    print(f"  Questions cosine reaches at {args.scan} and BM25 does not: "
          f"{len(only_cos)}")
    for q, g, rb, rc in only_cos[: args.show]:
        print(f"\n    Q  {q}")
        print(f"    A  {g}")
        print(f"       BM25 rank {rb or 'never'}, cosine rank {rc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/od_full_asresults.json")
    ap.add_argument("--category", default="open-domain")
    ap.add_argument("--limit", type=int, default=178)
    ap.add_argument("--scan", type=int, default=8)
    ap.add_argument("--show", type=int, default=6)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
