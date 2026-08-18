"""Mid-conversation ingestion checkpoints, so a dropped link costs minutes.

The run-level checkpoint in `run_locomo` is per *conversation*: a conversation
counts as done only once every one of its questions has been answered. Ingestion
is the expensive half -- 20 to 30 minutes of one LLM extraction call per
exchange -- and nothing about it was ever written down. An interruption at
exchange 168 of 211 therefore threw away 168 exchanges of work, and the resumed
run started that conversation again from nothing.

That is not a hypothetical. A connection drop at 00:21 on 2026-08-18 cost a
complete overnight run: both conversations were within four minutes of finishing
ingestion, and both went back to zero.

The store itself was never the problem. Hot RAM and Cold ROM are already on
disk and already hold every fact extracted so far; what was missing was any
record of *how far through* the conversation that store had got, and a way to
reopen it instead of wiping it. This module is that record.

Two things make the record safe rather than merely useful:

* **The fingerprint.** A store half-built under `beta=0.5` must never be
  resumed into a `beta=0` run: the result would be one arm silently ingested
  under two different decay configurations, which is a corrupted experiment that
  still produces a plausible-looking number. Any setting that changes what gets
  written -- decay overrides, the extractor prompt variant, the arm -- goes into
  a hash, and a mismatch means start over.

* **The turn clock.** `NeuromorphicMemory._current_turn` lives in memory only
  and starts at 0. Reopening a store without restoring it would leave records
  stamped `created_at_turn=168` in a world that believes the time is 0, and
  every decay and reinforcement calculation from then on would be wrong. The
  clock is stored alongside the exchange count and restored with it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

STATE_FILENAME = "ingest_state.json"

# Bumped when the meaning of a stored field changes. An older file is treated as
# a mismatch rather than migrated: re-ingesting costs minutes, and a
# misinterpreted checkpoint costs a whole experiment silently.
STATE_VERSION = 1


@dataclass(frozen=True)
class IngestState:
    """How far a store got through one conversation, and under what settings."""

    conversation_id: str
    exchanges_done: int
    turn: int
    fingerprint: str


def fingerprint(arm_name: str, decay_overrides: dict | None) -> str:
    """Hash every setting that changes what ingestion writes.

    The extractor variant is read from the environment rather than passed in
    because that is where it lives -- `default_extraction_prompt` resolves it at
    call time, so a run's prompt is a property of the environment and not of any
    object the runner holds. Leaving it out would let a `tiered` store be
    resumed into a `permissive` run, which is exactly the class of mistake the
    fingerprint exists to prevent.
    """
    payload = {
        "version": STATE_VERSION,
        "arm": arm_name,
        "decay": {k: decay_overrides[k] for k in sorted(decay_overrides or {})},
        "extractor_variant": os.environ.get(
            "NMAFC_EXTRACTOR_VARIANT", "tiered"
        ).strip().lower(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def store_dir_for(root: Path, arm_name: str, conversation_id: str) -> Path:
    """Deterministic store location for one (arm, conversation).

    Deterministic is the whole point: the previous behaviour handed every arm a
    fresh `tempfile.mkdtemp`, so even though the data survived a crash, the next
    process had no way to find it. Workers are pooled and a conversation is not
    pinned to a worker, so the path keys on the conversation rather than on
    which worker slot happened to pick it up.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in conversation_id)
    return root / "stores" / f"{arm_name}__{safe}"


def read(store_dir: Path, conversation_id: str, fp: str) -> IngestState | None:
    """Return usable prior progress, or None to start this conversation over.

    None is returned for every failure mode -- absent, unreadable, malformed,
    wrong conversation, wrong settings -- because they all have the same correct
    response. An unreadable checkpoint is not an error worth stopping a
    twelve-hour benchmark for; it is a reason to spend twenty minutes
    re-ingesting.
    """
    path = store_dir / STATE_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if raw.get("version") != STATE_VERSION:
        return None
    if raw.get("conversation_id") != conversation_id:
        return None
    if raw.get("fingerprint") != fp:
        return None

    try:
        done = int(raw["exchanges_done"])
        turn = int(raw["turn"])
    except (KeyError, TypeError, ValueError):
        return None
    if done <= 0 or turn < 0:
        return None

    return IngestState(
        conversation_id=conversation_id,
        exchanges_done=done,
        turn=turn,
        fingerprint=fp,
    )


def write(store_dir: Path, state: IngestState) -> None:
    """Record progress, atomically, never raising.

    Written to a temporary file and moved into place: the process this guards
    against is one that dies without warning, and a half-written state file
    would be read back as a valid checkpoint pointing at an exchange the store
    never reached. `os.replace` is atomic on both POSIX and Windows.

    Failures are swallowed on purpose. This is bookkeeping running inside the
    ingestion loop; a full disk should cost the resume, not the run.
    """
    path = store_dir / STATE_FILENAME
    tmp = store_dir / (STATE_FILENAME + ".tmp")
    payload = {
        "version": STATE_VERSION,
        "conversation_id": state.conversation_id,
        "exchanges_done": state.exchanges_done,
        "turn": state.turn,
        "fingerprint": state.fingerprint,
    }
    try:
        store_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def clear(store_dir: Path) -> None:
    """Drop the progress record, leaving the store alone.

    Called once a conversation finishes ingesting. Without it the file would
    still claim partial progress over a store that is actually complete, and the
    next run would resume from `exchanges_done` and re-ingest the tail -- adding
    duplicate facts to a finished store rather than saving any work.
    """
    try:
        (store_dir / STATE_FILENAME).unlink()
    except OSError:
        pass
