"""Scoring and evaluation module for benchmark results."""

from __future__ import annotations
from dataclasses import dataclass
from scripts.benchmarks.runners import RunResult


@dataclass
class ScoredResult:
    framework: str
    case_id: str
    category: str
    query: str
    ground_truth: str
    response: str
    accuracy_score: float
    context_collision: bool
    latency_sec: float
    token_count: int
    cost_usd: float


def evaluate_result(result: RunResult) -> ScoredResult:
    """Score a single run result for accuracy and context collision."""
    gt_lower = result.ground_truth.lower()
    resp_lower = result.response.lower()

    # Calculate accuracy score
    gt_words = [w.strip(".,!?") for w in gt_lower.split() if len(w) > 3]
    if not gt_words:
        score = 1.0 if len(resp_lower) > 0 else 0.0
    else:
        matches = sum(1 for w in gt_words if w in resp_lower)
        score = max(0.0, min(1.0, matches / len(gt_words)))

    # Special false premise handling
    if "never mentioned" in gt_lower or "no bicycle" in gt_lower or "didn't mention" in gt_lower:
        if "don't know" in resp_lower or "didn't mention" in resp_lower or "not mentioned" in resp_lower or "no mention" in resp_lower:
            score = 1.0
        else:
            score = 0.2

    # Context collision check (e.g. old 3PM time vs new 4:30PM time)
    collision = False
    if result.category == "Temporal Update":
        if "3:00" in resp_lower or "3pm" in resp_lower:
            collision = True
            score = 0.0

    return ScoredResult(
        framework=result.framework,
        case_id=result.case_id,
        category=result.category,
        query=result.query,
        ground_truth=result.ground_truth,
        response=result.response,
        accuracy_score=score,
        context_collision=collision,
        latency_sec=result.latency_sec,
        token_count=result.token_count,
        cost_usd=result.cost_usd,
    )
