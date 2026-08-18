"""Rate-limit governance and retry for long benchmark runs.

Measured quota for this deployment (read from x-ratelimit-* response headers,
which disagreed with the published quota table in both directions):

    500,000 tokens / 60s
    500 requests / 60s

At ~2,258 tokens per call averaged over the planned workload, TPM allows about
221 calls/min. With ~2.5s mean latency that is a sustained concurrency near 9 —
so the request cap is not what binds, and pushing concurrency past ~10 buys
nothing except 429s and wasted retry budget.

Two layers, both applied by wrapping the provider objects rather than editing
the library, so extraction calls made deep inside NeuromorphicMemory are
covered along with answering, judging, and embedding:

  RateLimiter     — a shared token+request bucket that paces calls *before*
                    they are sent, so we mostly avoid 429s instead of
                    recovering from them.
  Retrying*       — exponential backoff with jitter for the 429s and transient
                    5xx that get through anyway. Honors Retry-After.

Both are shared across arms: the quota is per-deployment, so every arm running
concurrently draws from the same bucket.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time

from nmafc.integration.base import EmbeddingProvider, LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate

TPM_LIMIT = int(os.environ.get("NMAFC_BENCH_TPM_LIMIT", "500000"))
RPM_LIMIT = int(os.environ.get("NMAFC_BENCH_RPM_LIMIT", "500"))
# Fraction of quota we actually aim to use. Token counts are estimated from
# characters, so leave headroom for the estimate running low.
SAFETY = float(os.environ.get("NMAFC_BENCH_QUOTA_SAFETY", "0.85"))
MAX_RETRIES = int(os.environ.get("NMAFC_BENCH_MAX_RETRIES", "6"))

# Wall-clock budget for riding out a dead network link, as opposed to a busy
# server. These are different failures and deserve different budgets: a 429
# means the service is up and answering, so six quick retries is the right
# response, while "connection error" means the link is gone and no number of
# fast retries helps -- only waiting does.
#
# Sized from the actual failure on this machine: a dropout at 23:55 exhausted
# the 6-retry budget (which spans ~2 minutes) and killed four conversations
# mid-ingestion, discarding ~2.5 hours of extraction work each. 20 minutes
# covers a router reboot or an ISP blip with room to spare, and costs nothing
# when the network is healthy because the budget only starts ticking on the
# first network error.
NETWORK_RETRY_BUDGET_S = float(
    os.environ.get("NMAFC_BENCH_NETWORK_RETRY_BUDGET_S", "1200")
)

_RETRYABLE = ("429", "500", "502", "503", "504", "timeout", "timed out",
              "connection", "overloaded", "rate limit", "ratelimit")

# Failures of the link itself rather than of the service behind it.
_NETWORK = ("connection", "timed out", "timeout", "unreachable", "reset by peer",
            "name resolution", "getaddrinfo", "ssl", "socket", "network")


def is_network_error(exc: Exception) -> bool:
    """True when the link is down, rather than the server being busy or angry.

    An exception carrying a status_code got a real HTTP response back, which
    means the connection worked -- that is a server-side condition and belongs
    on the ordinary retry budget, however it happens to be worded.
    """
    if getattr(exc, "status_code", None) is not None:
        return False
    return any(t in str(exc).lower() for t in _NETWORK)


def is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return any(t in str(exc).lower() for t in _RETRYABLE)


def retry_after_seconds(exc: Exception) -> float | None:
    """Pull a server-specified wait out of headers or message text."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-tokens"):
        val = headers.get(key)
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    match = re.search(r"retry after (\d+(?:\.\d+)?)\s*s", str(exc), re.I)
    return float(match.group(1)) if match else None


class RateLimiter:
    """Sliding-window limiter over both tokens and requests per 60s."""

    def __init__(
        self,
        tpm: int = TPM_LIMIT,
        rpm: int = RPM_LIMIT,
        safety: float = SAFETY,
    ) -> None:
        self._tpm = max(1, int(tpm * safety))
        self._rpm = max(1, int(rpm * safety))
        self._events: list[tuple[float, int]] = []  # (timestamp, tokens)
        self._lock = asyncio.Lock()
        self.throttle_waits = 0
        self.throttled_seconds = 0.0

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        self._events = [e for e in self._events if e[0] > cutoff]

    async def acquire(self, tokens: int) -> None:
        """Block until this call fits inside both windows, then reserve it."""
        # A single call larger than the window can never fit; let it through
        # rather than deadlock, and let retry handle the 429 if it comes.
        tokens = min(tokens, self._tpm)
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                used_tokens = sum(t for _, t in self._events)
                if used_tokens + tokens <= self._tpm and len(self._events) < self._rpm:
                    self._events.append((now, tokens))
                    return
                oldest = self._events[0][0] if self._events else now
                wait = max(0.05, 60.0 - (now - oldest))
                self.throttle_waits += 1
                self.throttled_seconds += wait
            await asyncio.sleep(wait)

    def stats(self) -> dict:
        return {
            "throttle_waits": self.throttle_waits,
            "throttled_seconds": round(self.throttled_seconds, 1),
        }


class _Backoff:
    """Retry bookkeeping shared by the LLM and embedding wrappers.

    Keeps two independent budgets. Ordinary retryable failures (429, 5xx) spend
    the `max_retries` count. Network failures spend wall-clock time instead and
    do *not* consume that count, so a long outage cannot be converted into a
    permanent give-up by a handful of fast reconnect attempts -- which is
    exactly how four conversations were lost mid-run.
    """

    def __init__(self, max_retries: int, label: str) -> None:
        self._max_retries = max_retries
        self._label = label
        self._attempt = 0
        self._net_attempt = 0
        self._deadline: float | None = None

    async def wait(self, exc: Exception) -> bool:
        """Sleep for the appropriate backoff. False means stop retrying."""
        if is_network_error(exc):
            now = time.monotonic()
            if self._deadline is None:
                self._deadline = now + NETWORK_RETRY_BUDGET_S
                print(f"    [net] {self._label}: link down "
                      f"({type(exc).__name__}); will keep retrying for up to "
                      f"{NETWORK_RETRY_BUDGET_S / 60:.0f} min")
            if now >= self._deadline:
                return False
            delay = min(30.0, 5.0 * 2 ** min(self._net_attempt, 3))
            self._net_attempt += 1
            await asyncio.sleep(delay + random.uniform(0, 0.5 * delay))
            return True

        if self._attempt >= self._max_retries or not is_retryable(exc):
            return False
        delay = retry_after_seconds(exc)
        if delay is None:
            delay = min(60.0, 2.0**self._attempt)
        self._attempt += 1
        await asyncio.sleep(delay + random.uniform(0, 0.5 * delay + 0.25))
        return True

    def recovered(self) -> None:
        """Call after a success, so a later outage gets a fresh budget."""
        if self._deadline is not None:
            print(f"    [net] {self._label}: link back")
            self._deadline = None
            self._net_attempt = 0


def estimate_tokens(messages: list[dict], system_prompt: str) -> int:
    chars = len(system_prompt) + sum(len(m.get("content") or "") for m in messages)
    return chars // 4 + 256  # + expected completion


class RetryingLLMProvider(LLMProvider):
    """Wraps any LLMProvider with pacing, backoff, and failure accounting."""

    def __init__(
        self,
        inner: LLMProvider,
        limiter: RateLimiter | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._inner = inner
        self._limiter = limiter
        self._max_retries = max_retries
        self.calls = 0
        self.retries = 0
        self.failures = 0

    async def chat_with_extraction(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[str, list[MemoryStateUpdate]]:
        last: Exception | None = None
        backoff = _Backoff(self._max_retries, "llm")
        while True:
            if self._limiter:
                await self._limiter.acquire(estimate_tokens(messages, system_prompt))
            try:
                result = await self._inner.chat_with_extraction(
                    messages=messages, system_prompt=system_prompt
                )
                self.calls += 1
                backoff.recovered()
                return result
            except Exception as exc:  # noqa: BLE001 - classified below
                last = exc
                if not await backoff.wait(exc):
                    break
                self.retries += 1

        self.failures += 1
        # A dead call must not abort a multi-hour run. Return empty so the
        # question is scored as a miss and the failure shows up in the report.
        print(f"    [retry] giving up: {type(last).__name__}: {str(last)[:140]}")
        return "", []

    def stats(self) -> dict:
        return {"calls": self.calls, "retries": self.retries, "failures": self.failures}


class RetryingEmbeddingProvider(EmbeddingProvider):
    """Same treatment for embeddings.

    Local Ollama embeddings draw no API quota, so this skips the limiter and
    only guards against transient local failures.
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        limiter: RateLimiter | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._inner = inner
        self._limiter = limiter
        self._max_retries = max_retries
        self.calls = 0
        self.retries = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        last: Exception | None = None
        backoff = _Backoff(self._max_retries, "embed")
        while True:
            if self._limiter:
                await self._limiter.acquire(sum(len(t) for t in texts) // 4)
            try:
                result = await self._inner.embed(texts)
                self.calls += 1
                backoff.recovered()
                return result
            except Exception as exc:  # noqa: BLE001 - classified below
                last = exc
                if not await backoff.wait(exc):
                    raise
                self.retries += 1

    def stats(self) -> dict:
        return {"calls": self.calls, "retries": self.retries}
