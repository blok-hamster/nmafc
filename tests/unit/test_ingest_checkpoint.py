"""Mid-conversation ingestion checkpoints.

The behaviour under test is a safety property, not a feature: a checkpoint that
is merely absent costs twenty minutes, but one that is wrongly *accepted* costs
an experiment -- a store half-built under one decay configuration, finished
under another, reporting a number that looks fine and means nothing. Most of
these tests are therefore about refusing to resume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmarks import ingest_checkpoint as ic  # noqa: E402

FP = "0123456789abcdef"


def _state(conv: str = "conv-26", done: int = 50, turn: int = 50) -> ic.IngestState:
    return ic.IngestState(
        conversation_id=conv, exchanges_done=done, turn=turn, fingerprint=FP
    )


def test_roundtrip(tmp_path: Path) -> None:
    ic.write(tmp_path, _state())
    got = ic.read(tmp_path, "conv-26", FP)
    assert got is not None
    assert got.exchanges_done == 50
    assert got.turn == 50


def test_missing_file_reads_as_none(tmp_path: Path) -> None:
    assert ic.read(tmp_path, "conv-26", FP) is None


def test_wrong_conversation_is_refused(tmp_path: Path) -> None:
    """A store built from conv-26 must never be resumed as conv-30.

    Worker slots are pooled and reused, so an arm instance meets many
    conversations; the store directory is the only thing keying them apart.
    """
    ic.write(tmp_path, _state(conv="conv-26"))
    assert ic.read(tmp_path, "conv-30", FP) is None


def test_wrong_fingerprint_is_refused(tmp_path: Path) -> None:
    """The case that would silently corrupt a result rather than crash.

    Resuming a beta=0.5 store into a beta=0 run yields one arm ingested under
    two decay configurations. Nothing would fail; the run would simply report a
    number that answers no question anyone asked.
    """
    ic.write(tmp_path, _state())
    assert ic.read(tmp_path, "conv-26", "deadbeefdeadbeef") is None


def test_corrupt_file_is_refused_not_raised(tmp_path: Path) -> None:
    """A truncated file must cost minutes, not kill a multi-hour benchmark."""
    (tmp_path / ic.STATE_FILENAME).write_text('{"version": 1, "exch', encoding="utf-8")
    assert ic.read(tmp_path, "conv-26", FP) is None


def test_old_version_is_refused(tmp_path: Path) -> None:
    payload = {
        "version": ic.STATE_VERSION - 1,
        "conversation_id": "conv-26",
        "exchanges_done": 50,
        "turn": 50,
        "fingerprint": FP,
    }
    (tmp_path / ic.STATE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    assert ic.read(tmp_path, "conv-26", FP) is None


@pytest.mark.parametrize("done,turn", [(0, 0), (-1, 5), (5, -1)])
def test_nonsense_progress_is_refused(tmp_path: Path, done: int, turn: int) -> None:
    """Zero exchanges is not a resume point, and negatives are corruption."""
    payload = {
        "version": ic.STATE_VERSION,
        "conversation_id": "conv-26",
        "exchanges_done": done,
        "turn": turn,
        "fingerprint": FP,
    }
    (tmp_path / ic.STATE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    assert ic.read(tmp_path, "conv-26", FP) is None


def test_clear_removes_progress(tmp_path: Path) -> None:
    """After a conversation finishes, the record must not claim partial work.

    Left behind, it would make the next run resume from `exchanges_done` and
    re-ingest the tail into a store that already holds it -- adding duplicate
    facts rather than saving any work.
    """
    ic.write(tmp_path, _state())
    ic.clear(tmp_path)
    assert ic.read(tmp_path, "conv-26", FP) is None
    ic.clear(tmp_path)  # absent file must not raise


def test_write_leaves_no_temp_file(tmp_path: Path) -> None:
    """The write is atomic, so a crash cannot leave a half-written state."""
    ic.write(tmp_path, _state())
    assert not (tmp_path / (ic.STATE_FILENAME + ".tmp")).exists()


def test_write_never_raises_on_bad_path(tmp_path: Path) -> None:
    """Bookkeeping inside the ingestion loop must not be able to kill the run."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    ic.write(blocker / "nested", _state())  # must not raise


class TestFingerprint:
    def test_decay_settings_change_it(self) -> None:
        assert ic.fingerprint("a", {"beta": 0.0}) != ic.fingerprint("a", {"beta": 0.5})

    def test_arm_changes_it(self) -> None:
        assert ic.fingerprint("neuromorphic", {}) != ic.fingerprint(
            "neuromorphic_tuned", {}
        )

    def test_key_order_does_not_change_it(self) -> None:
        """Otherwise a resume would depend on dict insertion order."""
        assert ic.fingerprint("a", {"beta": 0.5, "max_hops": 2}) == ic.fingerprint(
            "a", {"max_hops": 2, "beta": 0.5}
        )

    def test_none_and_empty_agree(self) -> None:
        assert ic.fingerprint("a", None) == ic.fingerprint("a", {})

    def test_extractor_variant_changes_it(self, monkeypatch) -> None:
        """A `tiered` store must not be resumed into a `permissive` run.

        The variant lives in the environment rather than in any object the
        runner holds, so it has to be read the same way `default_extraction_prompt`
        reads it or the guard would have a hole in exactly the place that cost
        eight accuracy points to discover.
        """
        monkeypatch.setenv("NMAFC_EXTRACTOR_VARIANT", "tiered")
        tiered = ic.fingerprint("a", {})
        monkeypatch.setenv("NMAFC_EXTRACTOR_VARIANT", "permissive")
        assert ic.fingerprint("a", {}) != tiered


class TestStoreDir:
    def test_is_deterministic(self, tmp_path: Path) -> None:
        """The point of the whole design: a later process can find the store."""
        assert ic.store_dir_for(tmp_path, "arm", "conv-26") == ic.store_dir_for(
            tmp_path, "arm", "conv-26"
        )

    def test_separates_conversations_and_arms(self, tmp_path: Path) -> None:
        paths = {
            ic.store_dir_for(tmp_path, "arm", "conv-26"),
            ic.store_dir_for(tmp_path, "arm", "conv-30"),
            ic.store_dir_for(tmp_path, "other", "conv-26"),
        }
        assert len(paths) == 3

    def test_sanitises_path_separators(self, tmp_path: Path) -> None:
        """A conversation id is dataset-supplied and must not escape the root."""
        got = ic.store_dir_for(tmp_path, "arm", "../../etc/passwd")
        assert ".." not in got.name
        assert got.parent == tmp_path / "stores"
