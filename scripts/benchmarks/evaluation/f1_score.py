"""F1 scoring following the LoCoMo paper protocol.

Implements normalized token-level F1 partial match, which is the standard
evaluation metric used in LoCoMo (Maharana et al., 2024) for QA accuracy.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize answer text for fair comparison.

    Following LoCoMo paper: lowercase, remove articles, remove punctuation,
    collapse whitespace.
    """
    text = str(text).lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse whitespace
    text = " ".join(text.split())

    return text.strip()


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between normalized prediction and ground truth.

    Returns 0.0 if either is empty after normalization.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match after normalization. Returns 1.0 or 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_batch_f1(
    predictions: list[str],
    ground_truths: list[str],
) -> dict[str, float]:
    """Compute aggregate F1 and EM over a batch of predictions."""
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have same length")

    if not predictions:
        return {"f1": 0.0, "exact_match": 0.0, "count": 0}

    f1_scores = [
        compute_f1(p, g) for p, g in zip(predictions, ground_truths)
    ]
    em_scores = [
        compute_exact_match(p, g) for p, g in zip(predictions, ground_truths)
    ]

    return {
        "f1": sum(f1_scores) / len(f1_scores),
        "exact_match": sum(em_scores) / len(em_scores),
        "count": len(predictions),
    }
