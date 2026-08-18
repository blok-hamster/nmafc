"""Resuming a run must not redo finished work or forget what it measured.

Two failures found the hard way on a 1,986-question run:

1. The judge phase re-graded every restored answer, because it filtered only on
   "did this question error", never on "does this row already have a verdict".
   A resume that had nothing left to answer still spent ~40 minutes and a full
   arm's worth of quota per completed arm recomputing verdicts already on disk.

2. Restored arms reported 0 ms latency, 0 context tokens and 0 total tokens,
   because cost samples lived only on the worker objects and a resumed arm's
   workers answer nothing. The headline compression claim -- 431 context tokens
   against 20,006 -- is exactly the number that went missing.

Both are invisible in a normal run and only appear after a restart, which is
why they get pinned here rather than left to the next long run to rediscover.
"""

from __future__ import annotations

import pytest

from scripts.benchmarks.run_locomo import (
    _rebuild_metrics_from_rows,
    run_judge_phase,
)


class _ExplodingJudge:
    """Any use at all is a failure: these rows were already graded."""

    async def chat_with_extraction(self, *args, **kwargs):
        raise AssertionError("judge was called for already-graded answers")


def _row(**overrides) -> dict:
    row = {
        "question": "q",
        "gold_answer": "a",
        "predicted": "a",
        "category": "single-hop",
        "f1": 1.0,
        "judge_correct": True,
        "latency_ms": 1234.5,
        "context_tokens": 400,
    }
    row.update(overrides)
    return row


class TestJudgePhaseSkipsGradedAnswers:
    @pytest.mark.asyncio
    async def test_fully_graded_arm_is_not_regraded(self):
        rows = [_row() for _ in range(50)]
        await run_judge_phase(rows, _ExplodingJudge(), concurrency=4)
        assert all(r["judge_correct"] is True for r in rows)

    @pytest.mark.asyncio
    async def test_a_false_verdict_counts_as_graded(self):
        # False is a real verdict -- "the judge marked this wrong" -- and a
        # falsy check here would re-grade every wrong answer on every resume,
        # which on this benchmark is over half the set.
        rows = [_row(judge_correct=False) for _ in range(10)]
        await run_judge_phase(rows, _ExplodingJudge(), concurrency=4)
        assert all(r["judge_correct"] is False for r in rows)

    @pytest.mark.asyncio
    async def test_ungraded_rows_are_still_sent_to_the_judge(self):
        # The skip must not become a blanket skip: a row whose judge call
        # failed carries None and has to get another attempt.
        rows = [_row(), _row(judge_correct=None)]
        seen: list[dict] = []

        async def _fake_batch(items, **kwargs):
            seen.extend(items)
            return [type("V", (), {"correct": True})() for _ in items]

        import scripts.benchmarks.run_locomo as runner

        original = runner.judge_batch
        runner.judge_batch = _fake_batch
        try:
            await run_judge_phase(rows, _ExplodingJudge(), concurrency=4)
        finally:
            runner.judge_batch = original

        assert len(seen) == 1  # only the ungraded one
        assert all(r["judge_correct"] is True for r in rows)


class TestMetricsSurviveARestart:
    def test_latency_and_context_are_recovered_from_rows(self):
        rows = [_row(latency_ms=100.0, context_tokens=10),
                _row(latency_ms=300.0, context_tokens=30)]

        metrics = _rebuild_metrics_from_rows(rows)

        assert metrics["latencies"] == [100.0, 300.0]
        assert metrics["context_tokens"] == [10, 30]

    def test_unknown_token_counts_stay_empty_rather_than_guessed(self):
        # Prompt and completion counts were never written to the rows. Leaving
        # them empty makes the gap visible in the report; inventing a plausible
        # number from context size would make a cost claim out of an estimate.
        metrics = _rebuild_metrics_from_rows([_row()])

        assert metrics["prompt_tokens"] == []
        assert metrics["completion_tokens"] == []
