"""Mark finished stores as finished, so a rerun answers without re-extracting.

The checkpoint file is deleted once a conversation finishes ingesting -- correct
for its original job, since a stale file over a complete store would make the
next run resume from `exchanges_done` and duplicate the tail. The side effect is
that a *successfully* ingested store is indistinguishable from an empty
directory, so the next run wipes it and pays the full extraction cost again.

That cost is the entire reason a prompt experiment feels expensive. Ingestion is
one LLM call per exchange and takes hours; answering is one call per question
and takes minutes. Any change that lives in the answer prompt or in retrieval
settings needs none of the ingestion work redone, and this script says so
explicitly by writing the state file back:

    exchanges_done = every exchange in the conversation
    turn           = the same number, since process_turn advances the clock once
                     per exchange

`prepare_store` then reopens the store, restores the clock, and returns a
`start_at` past the end of the list, so `ingest_conversation` skips every
exchange without calling a model.

Safety, in order of how badly each would corrupt a result:

  * The fingerprint is computed by the same function the runner uses, from the
    flags the rerun will use. A mismatch means the store is left alone and the
    rerun re-ingests, which is slow rather than wrong.
  * Stores are checked for content first. Seeding an empty directory would
    produce a run that answers 1,986 questions against nothing and reports it as
    a memory result.
  * `--dry-run` is the default posture in practice: run it, read the table,
    then pass --write.

This is deliberately a separate script rather than a runner flag. Reusing a
store is a claim that the store matches the experiment, and that claim should be
made by hand.

Usage:
    python scripts/benchmarks/_seed_ingest_state.py --run scripts/benchmarks/results/full_v3
    python scripts/benchmarks/_seed_ingest_state.py --run ... --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmarks import ingest_checkpoint
from scripts.benchmarks.arms.base import build_exchanges
from scripts.benchmarks.datasets.locomo_loader import load_locomo

# Arms that hold a store worth reusing. `raw_llm` keeps no store at all, and
# `rag` re-chunks from scratch in a single batched embed with no LLM call, so
# neither has anything to skip.
RESUMABLE_ARMS = ("neuromorphic", "neuromorphic_tuned", "stateful_nodecay")


def store_has_content(store_dir: Path) -> bool:
    """Does this directory hold an actual ingested store?

    Checks the archive rather than the vector table: Cold ROM is a single SQLite
    file that is only created once something is written, whereas the LanceDB
    directory is created at store construction and exists even when empty.
    """
    cold = store_dir / "cold.db"
    return cold.exists() and cold.stat().st_size > 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory holding stores/")
    ap.add_argument("--write", action="store_true",
                    help="actually write the state files (default: report only)")
    ap.add_argument("--arms", default=",".join(RESUMABLE_ARMS),
                    help="comma-separated arm names to seed")
    ap.add_argument("--beta", type=float, default=None,
                    help="must match the --beta the rerun will use, if any")
    args = ap.parse_args()

    store_root = Path(args.run) / "stores"
    if not store_root.is_dir():
        raise SystemExit(f"no stores directory under {args.run}")

    # Only ingestion-affecting overrides reach the fingerprint, so retrieval
    # flags the rerun may add do not need to be mirrored here.
    overrides: dict = {}
    if args.beta is not None:
        overrides["beta"] = args.beta

    conversations = {c.sample_id: c for c in load_locomo()}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(f"run   : {args.run}")
    print(f"mode  : {'WRITE' if args.write else 'dry run, nothing written'}\n")
    print(f"  {'arm':22s} {'conversation':14s} {'exchanges':>9s}  status")

    seeded = skipped = 0
    for arm_name in arms:
        fp = ingest_checkpoint.fingerprint(arm_name, overrides)
        for sample_id, conv in sorted(conversations.items()):
            store_dir = ingest_checkpoint.store_dir_for(store_root.parent,
                                                        arm_name, sample_id)
            if not store_dir.is_dir():
                continue
            if not store_has_content(store_dir):
                print(f"  {arm_name:22s} {sample_id:14s} {'-':>9s}  "
                      f"EMPTY, left alone")
                skipped += 1
                continue

            total = len(build_exchanges(conv.get_flat_history()))
            if args.write:
                ingest_checkpoint.write(
                    store_dir,
                    ingest_checkpoint.IngestState(
                        conversation_id=sample_id,
                        exchanges_done=total,
                        turn=total,
                        fingerprint=fp,
                    ),
                )
                status = "seeded"
            else:
                status = "would seed"
            print(f"  {arm_name:22s} {sample_id:14s} {total:9d}  {status}")
            seeded += 1

    print(f"\n  {seeded} store(s) {'seeded' if args.write else 'ready to seed'}, "
          f"{skipped} skipped")
    if not args.write:
        print("  re-run with --write to apply")
    else:
        print("\n  the rerun must pass --store-root/--checkpoint the same way the "
              "original did, and must use the same --beta, or the fingerprint "
              "will not match and ingestion will start over.")


if __name__ == "__main__":
    main()
