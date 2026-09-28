"""Tests for `nmafc eval locomo` — the Phase 3.2 reproducibility harness.

The mission-critical behaviour is `regenerate`: a third party with a repo
checkout can reproduce every figure in the README results section from the
committed per-question JSON with no API calls, no store access, and no
dependencies beyond the standard library. That is covered here as a real
end-to-end run. The three forwarding commands are covered with a stubbed
subprocess so the live-LLM runner is never invoked.
"""

from __future__ import annotations

import sys

import pytest

from nmafc import cli

REPO_ROOT = cli._REPO_ROOT


@pytest.fixture
def recorder(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv: list[str], cwd) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(cli, "_run", fake_run)
    return calls


def _expect_systemexit(func, code: int):
    with pytest.raises(SystemExit) as exc:
        func()
    assert exc.value.code == code


def test_regenerate_reproduces_readme_table(capfd):
    _expect_systemexit(lambda: cli.main(["eval", "locomo", "regenerate"]), 0)
    out = capfd.readouterr().out

    assert "Paired LoCoMo run, 1985 questions" in out
    assert "SCORED (4)" in out
    assert "ALL (5)" in out
    assert "adversarial" in out
    # The published headline figures must still be there, else the README and
    # this harness have drifted apart.
    assert "+6.8" in out      # ours vs RAG, 4 scored categories
    assert "-14.6" in out     # adversarial stays in the paper, not dropped


def test_run_forwards_flags_verbatim(recorder):
    _expect_systemexit(
        lambda: cli.main([
            "eval", "locomo", "run",
            "--", "--arms", "raw,rag", "--conversations", "2", "--skip-judge",
        ]),
        0,
    )
    assert len(recorder) == 1
    argv = recorder[0]
    assert argv[:1] == [sys.executable]
    assert argv[1].endswith("scripts/benchmarks/run_locomo.py")
    assert argv[2:] == ["--arms", "raw,rag", "--conversations", "2", "--skip-judge"]


def test_summarise_forwards_run_and_baseline(recorder):
    _expect_systemexit(
        lambda: cli.main([
            "eval", "locomo", "summarise",
            "--run", "results/fresh", "--baseline", "results/paired_2026_09_10",
        ]),
        0,
    )
    assert len(recorder) == 1
    argv = recorder[0]
    assert argv[1].endswith("scripts/benchmarks/_summarise_locomo.py")
    assert argv[2:] == ["--run", "results/fresh", "--baseline", "results/paired_2026_09_10"]


def test_summarise_without_baseline(recorder):
    _expect_systemexit(
        lambda: cli.main(["eval", "locomo", "summarise", "--run", "results/fresh"]),
        0,
    )
    assert len(recorder) == 1
    assert recorder[0][2:] == ["--run", "results/fresh"]


def test_run_missing_scripts_errors_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_REPO_ROOT", cli._REPO_ROOT / "does-not-exist")
    _expect_systemexit(lambda: cli.main(["eval", "locomo", "run"]), 1)
    err = capsys.readouterr().err
    assert "run_locomo.py not found" in err


def test_methodology_prints_paired_design(capsys):
    cli.main(["eval", "locomo", "methodology"])
    out = capsys.readouterr().out
    assert "McNemar" in out
    assert "adversarial" in out
    assert "nmafc eval locomo regenerate" in out
