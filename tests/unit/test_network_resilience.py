"""A dropped link must not be turned into a permanent failure.

A ~45s network dropout during a benchmark run exhausted the six-retry budget --
which spans about two minutes of exponential backoff -- and killed four
conversations mid-ingestion, discarding roughly 2.5 hours of extraction work
each. Retries were not missing; they were spent too fast on a failure that only
time could fix.

The fix separates the two budgets: server-side failures (429, 5xx) spend the
retry *count*, while link failures spend wall-clock *time* and do not consume
the count. These tests pin that separation, since the difference is invisible
until an outage lasts longer than the retry budget.
"""

from __future__ import annotations

import pytest

from scripts.benchmarks import resilience
from scripts.benchmarks.resilience import (
    RetryingEmbeddingProvider,
    RetryingLLMProvider,
    is_network_error,
)


class _Status(Exception):
    """An exception that carries an HTTP status, i.e. the server replied."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """Collapse backoff so the tests measure policy, not wall-clock."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(resilience.asyncio, "sleep", _instant)


class _FlakyLLM:
    """Fails with `error` for the first `failures` calls, then succeeds."""

    def __init__(self, failures: int, error: Exception) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    async def chat_with_extraction(self, messages, system_prompt):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok", []


class _FlakyEmbedder:
    def __init__(self, failures: int, error: Exception) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return [[0.0, 0.0, 0.0] for _ in texts]


class TestNetworkErrorClassification:
    def test_connection_error_is_a_network_error(self):
        assert is_network_error(Exception("Connection error."))

    def test_dns_failure_is_a_network_error(self):
        assert is_network_error(Exception("getaddrinfo failed"))

    def test_rate_limit_is_not_a_network_error(self):
        # 429 means the service answered. It belongs on the retry-count budget,
        # not the outage budget, or a throttled run would hang for 20 minutes
        # per call instead of backing off and moving on.
        assert not is_network_error(_Status("RateLimitReached", 429))

    def test_status_bearing_error_is_never_network(self):
        # A 503 body can literally contain the word "connection"; the presence
        # of a status code is what settles it, not the wording.
        assert not is_network_error(_Status("upstream connection reset", 503))


class TestOutageSurvival:
    @pytest.mark.asyncio
    async def test_llm_survives_more_failures_than_the_retry_count(self):
        # This is the discriminating case: 20 consecutive connection failures
        # against max_retries=6. The old count-based loop gave up at 6 and
        # returned an empty answer.
        inner = _FlakyLLM(failures=20, error=Exception("Connection error."))
        provider = RetryingLLMProvider(inner, limiter=None, max_retries=6)

        answer, _ = await provider.chat_with_extraction(
            messages=[{"role": "user", "content": "hi"}], system_prompt=""
        )

        assert answer == "ok"
        assert inner.calls == 21
        assert provider.failures == 0

    @pytest.mark.asyncio
    async def test_embedder_survives_and_does_not_raise(self):
        # The embedder re-raises rather than degrading, so an outage here is
        # what actually propagated up and killed the whole conversation.
        inner = _FlakyEmbedder(failures=15, error=Exception("Connection error."))
        provider = RetryingEmbeddingProvider(inner, limiter=None, max_retries=6)

        vectors = await provider.embed(["a"])

        assert vectors == [[0.0, 0.0, 0.0]]
        assert inner.calls == 16

    @pytest.mark.asyncio
    async def test_outage_budget_is_bounded(self, monkeypatch):
        # Surviving an outage must not mean retrying forever: a permanently
        # dead link has to surface as a failure rather than hang the run.
        monkeypatch.setattr(resilience, "NETWORK_RETRY_BUDGET_S", 0.0)
        inner = _FlakyLLM(failures=10_000, error=Exception("Connection error."))
        provider = RetryingLLMProvider(inner, limiter=None, max_retries=6)

        answer, _ = await provider.chat_with_extraction(
            messages=[{"role": "user", "content": "hi"}], system_prompt=""
        )

        assert answer == ""
        assert provider.failures == 1
        assert inner.calls == 1  # zero budget means the first failure is final


class TestRateLimitBudgetUnchanged:
    @pytest.mark.asyncio
    async def test_429s_still_give_up_after_max_retries(self):
        # The outage budget must not silently turn a throttled deployment into
        # an unbounded wait. 429s still stop at max_retries.
        inner = _FlakyLLM(failures=10_000, error=_Status("429 rate limit", 429))
        provider = RetryingLLMProvider(inner, limiter=None, max_retries=3)

        answer, _ = await provider.chat_with_extraction(
            messages=[{"role": "user", "content": "hi"}], system_prompt=""
        )

        assert answer == ""
        assert inner.calls == 4  # initial attempt + 3 retries
        assert provider.failures == 1

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(self):
        inner = _FlakyLLM(failures=10_000, error=_Status("400 bad request", 400))
        provider = RetryingLLMProvider(inner, limiter=None, max_retries=6)

        answer, _ = await provider.chat_with_extraction(
            messages=[{"role": "user", "content": "hi"}], system_prompt=""
        )

        assert answer == ""
        assert inner.calls == 1
