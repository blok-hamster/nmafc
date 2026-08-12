"""Operational metrics aggregation for benchmark arms.

Tracks latency, token cost, storage footprint — the non-accuracy metrics
that prove the neuromorphic architecture's efficiency advantages.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class ArmResponse:
    """Response from a single question answered by a benchmark arm."""
    answer: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    context_tokens: int


@dataclass
class ArmMetrics:
    """Accumulated operational metrics for a benchmark arm."""
    arm_name: str
    _latencies: list[float] = field(default_factory=list)
    _prompt_tokens: list[int] = field(default_factory=list)
    _completion_tokens: list[int] = field(default_factory=list)
    _context_tokens: list[int] = field(default_factory=list)
    hot_storage_records: int = 0
    cold_storage_events: int = 0

    def record(self, response: ArmResponse) -> None:
        """Record metrics from a single response."""
        self._latencies.append(response.latency_ms)
        self._prompt_tokens.append(response.prompt_tokens)
        self._completion_tokens.append(response.completion_tokens)
        self._context_tokens.append(response.context_tokens)

    @property
    def count(self) -> int:
        return len(self._latencies)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(self._prompt_tokens)

    @property
    def total_completion_tokens(self) -> int:
        return sum(self._completion_tokens)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def avg_context_tokens(self) -> float:
        return statistics.mean(self._context_tokens) if self._context_tokens else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self._latencies) if self._latencies else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return statistics.median(self._latencies) if self._latencies else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> dict:
        """Serialize metrics to a dictionary for JSON output."""
        return {
            "arm_name": self.arm_name,
            "questions_answered": self.count,
            "tokens": {
                "total_prompt": self.total_prompt_tokens,
                "total_completion": self.total_completion_tokens,
                "total": self.total_tokens,
                "avg_context_per_question": round(self.avg_context_tokens, 1),
            },
            "latency_ms": {
                "avg": round(self.avg_latency_ms, 1),
                "p50": round(self.p50_latency_ms, 1),
                "p95": round(self.p95_latency_ms, 1),
            },
            "storage": {
                "hot_records": self.hot_storage_records,
                "cold_events": self.cold_storage_events,
            },
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one arm on one dataset."""
    arm_name: str
    dataset: str
    variant: str = ""

    # Accuracy metrics
    overall_accuracy: float = 0.0
    overall_f1: float = 0.0
    accuracy_by_category: dict[str, float] = field(default_factory=dict)
    f1_by_category: dict[str, float] = field(default_factory=dict)

    # Operational metrics
    metrics: ArmMetrics | None = None

    # Per-question details
    question_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON output."""
        return {
            "arm_name": self.arm_name,
            "dataset": self.dataset,
            "variant": self.variant,
            "accuracy": {
                "overall": round(self.overall_accuracy, 4),
                "overall_f1": round(self.overall_f1, 4),
                "by_category": {
                    k: round(v, 4) for k, v in self.accuracy_by_category.items()
                },
                "f1_by_category": {
                    k: round(v, 4) for k, v in self.f1_by_category.items()
                },
            },
            "operational": self.metrics.to_dict() if self.metrics else {},
            "question_results": self.question_results,
        }


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "claude-haiku",
) -> float:
    """Estimate USD cost based on token counts and model pricing."""
    pricing = {
        "claude-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
        "claude-sonnet": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.0 / 1_000_000},
    }

    rates = pricing.get(model, pricing["claude-haiku"])
    return prompt_tokens * rates["input"] + completion_tokens * rates["output"]
