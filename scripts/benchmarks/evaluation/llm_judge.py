"""LLM-as-Judge evaluation following Zep/LongMemEval protocol.

Uses an LLM (GPT-4o or Claude) to judge whether a predicted answer is correct
given the gold answer. This approach has high correlation with human evaluators
as demonstrated in the Zep paper (Rasmussen et al., 2025).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from nmafc.integration.base import LLMProvider

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a conversational memory benchmark.
Your job is to determine whether a predicted answer correctly addresses the question
given a gold (reference) answer.

Rules:
- The predicted answer does NOT need to be word-for-word identical to the gold answer.
- It IS correct if it conveys the same factual information, even if phrased differently.
- It IS correct if it contains the gold answer as part of a longer response.
- It is INCORRECT if it contradicts the gold answer, provides wrong information,
  says "I don't know" when an answer exists, or answers a different question.
- For temporal questions: dates/times must match (approximate is OK if within reason).
- For adversarial questions: correctly identifying unanswerable questions counts as correct.

Respond with ONLY a JSON object:
{"correct": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""

JUDGE_USER_TEMPLATE = """Question: {question}

Gold Answer: {gold_answer}

Predicted Answer: {predicted_answer}

Is the predicted answer correct? Respond with JSON only."""


@dataclass
class JudgeResult:
    correct: bool
    confidence: float
    reasoning: str
    latency_ms: float


async def judge_answer(
    question: str,
    predicted: str,
    gold_answer: str,
    judge_provider: LLMProvider,
) -> JudgeResult:
    """Use an LLM to judge whether the predicted answer is correct."""
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted,
    )

    start = time.perf_counter()

    response_text, _ = await judge_provider.chat_with_extraction(
        messages=[{"role": "user", "content": user_msg}],
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    result = _parse_judge_response(response_text)
    result.latency_ms = latency_ms
    return result


async def judge_batch(
    items: list[dict[str, str]],
    judge_provider: LLMProvider,
    concurrency: int = 5,
    retry_on_error: int = 2,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[JudgeResult]:
    """Judge a batch of answers with controlled concurrency.

    Each item should have keys: question, predicted, gold_answer

    `on_progress(done, total)` is called as verdicts land. Judging ~2,000
    answers is a single silent gather that can run 20+ minutes, which is
    indistinguishable from a hang in a log file; the callback is what makes it
    observable. Counting completions is safe under asyncio's single-threaded
    scheduling -- no lock is needed for `done += 1`.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[JudgeResult] = []
    done = 0
    # Roughly ten updates per batch, whatever its size: a fixed stride prints
    # nothing at all on a short pilot batch and floods a full 2,000-answer run.
    step = max(1, min(100, len(items) // 10))

    async def _judge_one(item: dict[str, str]) -> JudgeResult:
        nonlocal done
        try:
            return await _judge_inner(item)
        finally:
            done += 1
            if on_progress is not None and (done % step == 0 or done == len(items)):
                on_progress(done, len(items))

    async def _judge_inner(item: dict[str, str]) -> JudgeResult:
        async with semaphore:
            for attempt in range(retry_on_error + 1):
                try:
                    return await judge_answer(
                        question=item["question"],
                        predicted=item["predicted"],
                        gold_answer=item["gold_answer"],
                        judge_provider=judge_provider,
                    )
                except Exception:
                    if attempt == retry_on_error:
                        return JudgeResult(
                            correct=False,
                            confidence=0.0,
                            reasoning="Judge evaluation failed after retries",
                            latency_ms=0.0,
                        )
                    await asyncio.sleep(2 ** attempt)
            return JudgeResult(
                correct=False, confidence=0.0,
                reasoning="unreachable", latency_ms=0.0,
            )

    tasks = [_judge_one(item) for item in items]
    results = await asyncio.gather(*tasks)
    return list(results)


def _parse_judge_response(response: str) -> JudgeResult:
    """Parse the judge's JSON response, handling common formatting issues."""
    response = response.strip()

    # Strip markdown code fences if present
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join(
            line for line in lines
            if not line.startswith("```")
        )

    try:
        data = json.loads(response)
        return JudgeResult(
            correct=bool(data.get("correct", False)),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=str(data.get("reasoning", "")),
            latency_ms=0.0,
        )
    except (json.JSONDecodeError, ValueError):
        # Fallback: look for true/false in the response
        lower = response.lower()
        correct = "true" in lower and "false" not in lower
        return JudgeResult(
            correct=correct,
            confidence=0.3,
            reasoning=f"Parsed from raw response: {response[:100]}",
            latency_ms=0.0,
        )


def compute_judge_accuracy(results: list[JudgeResult]) -> dict[str, float]:
    """Compute aggregate accuracy from judge results."""
    if not results:
        return {"accuracy": 0.0, "avg_confidence": 0.0, "count": 0}

    correct_count = sum(1 for r in results if r.correct)
    avg_confidence = sum(r.confidence for r in results) / len(results)

    return {
        "accuracy": correct_count / len(results),
        "avg_confidence": avg_confidence,
        "count": len(results),
        "correct": correct_count,
        "incorrect": len(results) - correct_count,
    }
