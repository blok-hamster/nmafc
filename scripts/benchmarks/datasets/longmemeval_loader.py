"""LongMemEval dataset loader.

Loads the real LongMemEval benchmark from HuggingFace.
Source: xiaowu0162/longmemeval-cleaned

Dataset contains 500 questions across 6 types designed to evaluate
long-term conversational memory (used by Zep paper, Rasmussen et al., 2025).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from huggingface_hub import hf_hub_download

REPO_ID = "xiaowu0162/longmemeval-cleaned"

VARIANT_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

QUESTION_TYPES = [
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]


@dataclass
class LongMemEvalQuestion:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_sessions: list[list[dict]]
    haystack_dates: list[str] = field(default_factory=list)
    answer_session_ids: list[str] = field(default_factory=list)

    @property
    def total_turns(self) -> int:
        return sum(len(s) for s in self.haystack_sessions)

    @property
    def num_sessions(self) -> int:
        return len(self.haystack_sessions)

    def get_flat_history(self) -> list[dict]:
        """Flatten all haystack sessions into a single turn list."""
        turns = []
        for session in self.haystack_sessions:
            for turn in session:
                turns.append({
                    "role": turn["role"],
                    "content": turn["content"],
                })
        return turns


def load_longmemeval(
    variant: str = "oracle",
    cache_dir: str | None = None,
) -> list[LongMemEvalQuestion]:
    """Download and parse LongMemEval from HuggingFace.

    Variants:
        oracle: 500 questions, minimal haystack (shortest context per question)
        s: 500 questions, short haystack (more distractor sessions)
        m: 500 questions, medium haystack (longest context)
    """
    if variant not in VARIANT_FILES:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANT_FILES.keys())}")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=VARIANT_FILES[variant],
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    with open(path) as f:
        raw_data = json.load(f)

    questions = []
    for item in raw_data:
        questions.append(LongMemEvalQuestion(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            answer=item["answer"],
            question_date=item.get("question_date", ""),
            haystack_sessions=item.get("haystack_sessions", []),
            haystack_dates=item.get("haystack_dates", []),
            answer_session_ids=item.get("answer_session_ids", []),
        ))

    return questions


def iter_by_type(
    questions: list[LongMemEvalQuestion],
    qtype: str | None = None,
) -> list[LongMemEvalQuestion]:
    """Filter questions by type."""
    if qtype is None:
        return questions
    return [q for q in questions if q.question_type == qtype]


def get_dataset_stats(questions: list[LongMemEvalQuestion]) -> dict:
    """Return summary statistics for the loaded dataset."""
    from collections import Counter

    type_counts = Counter(q.question_type for q in questions)
    total_turns = sum(q.total_turns for q in questions)
    avg_sessions = sum(q.num_sessions for q in questions) / len(questions) if questions else 0

    return {
        "total_questions": len(questions),
        "total_turns": total_turns,
        "avg_sessions_per_question": round(avg_sessions, 1),
        "questions_by_type": dict(type_counts.most_common()),
    }
