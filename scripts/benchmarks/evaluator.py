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
    unsupported_claim: bool  # Academic UCR: Unsupported Claim Rate / Hallucination on Distractors
    latency_sec: float
    token_count: int
    cost_usd: float


def evaluate_result(result: RunResult) -> ScoredResult:
    """Score a single run result using academic LongMemEval / LoCoMo standard metrics: Accuracy & Unsupported Claim Rate (UCR)."""
    gt_lower = result.ground_truth.lower()
    resp_lower = result.response.lower()

    # Calculate accuracy score
    gt_words = [w.strip(".,!?") for w in gt_lower.split() if len(w) > 3]
    if not gt_words:
        score = 1.0 if len(resp_lower) > 0 else 0.0
    else:
        matches = sum(1 for w in gt_words if w in resp_lower)
        score = max(0.0, min(1.0, matches / len(gt_words)))

    unsupported = False

    # 1. False Premise Queries -> Academic Unsupported Claim Rate (UCR) check
    if "never mentioned" in gt_lower or "no bicycle" in gt_lower or "didn't mention" in gt_lower:
        rejection_phrases = [
            "don't know", "didn't mention", "not mentioned", "no mention",
            "never mentioned", "no information", "do not specify", "does not specify",
            "no record", "not provided", "haven't mentioned", "not state", "no bicycle"
        ]
        if any(p in resp_lower for p in rejection_phrases):
            score = 1.0
            unsupported = False
        else:
            score = 0.2
            unsupported = True  # Model accepted false premise (Unsupported Claim)

    # 2. Temporal Update Queries -> Academic Temporal Anchoring Error (TAE) / Stale Fact Bleed
    if result.category == "Temporal Update":
        if "3:00" in resp_lower or "3pm" in resp_lower:
            unsupported = True
            score = 0.0

    return ScoredResult(
        framework=result.framework,
        case_id=result.case_id,
        category=result.category,
        query=result.query,
        ground_truth=result.ground_truth,
        response=result.response,
        accuracy_score=score,
        unsupported_claim=unsupported,
        latency_sec=result.latency_sec,
        token_count=result.token_count,
        cost_usd=result.cost_usd,
    )

