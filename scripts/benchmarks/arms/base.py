"""Abstract base class for benchmark arms.

Each arm represents a different memory strategy being evaluated.
All arms share the same LLM for generation and answer questions
from the same dataset — they differ only in how they manage memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..evaluation.metrics import ArmMetrics, ArmResponse


# Appended to every arm's answer prompt. LoCoMo scores token-level F1 against
# gold answers that are typically 1-4 words ("by dancing", "19 January, 2023"),
# so a correct but conversational reply scores near zero: an answer of
# "Both Jon and Gina like to destress through dancing..." earned F1=0.042
# against gold "by dancing" while the LLM judge marked it correct. That gap is
# a formatting artifact, not a memory result, and it makes the headline F1
# meaningless unless the model is held to short-form output.
#
# Applied identically to all arms, so it changes the absolute numbers without
# advantaging any arm over another.
SHORT_ANSWER_RULES = """

ANSWER FORMAT (strict — your answer is scored by token overlap against a short reference):
- Output ONLY the answer itself. No preamble, no explanation, no markdown, no quotes.
- Never begin with "Based on", "According to", "The conversation shows", or similar.
- Be as short as possible — usually 1-5 words.
- "When"/date questions: answer with the date as written in the transcript, e.g. "19 January, 2023".
- Yes/no questions: answer exactly "Yes" or "No".
- If the answer is genuinely absent, reply exactly: No information available."""


def build_exchanges(turns: list[dict]) -> list[str]:
    """Group a transcript into speaker-labelled exchanges for memory ingestion.

    Each user turn is paired with the assistant turn that follows it, and both
    are labelled with their speaker, producing one block per exchange:

        Caroline: I adopted a golden retriever named Biscuit.
        Melanie: Wonderful! How is Biscuit settling in?

    Why this matters for a fair comparison: the raw-LLM arm receives the entire
    transcript, both speakers included. The memory arms previously ingested only
    turns where role == "user", so every fact stated by the other speaker was
    invisible to them — they were being scored on a strictly smaller slice of
    the evidence than the baseline they are compared against. On LoCoMo, where
    speaker_b maps to "assistant", that discards roughly half the content.

    Cost is unchanged: one exchange still means one process_turn call, so this
    adds no LLM calls relative to the previous user-turns-only loop.

    Passing the exchange as user_msg (rather than via conversation_history) is
    deliberate. StateExtractor appends user_msg to whatever context it is
    handed, so the old `conversation_history=[turn]` duplicated the user message
    verbatim in the extraction prompt.

    Each exchange is prefixed with its session timestamp where the dataset
    provides one, so extracted facts can carry a date. Without it, the temporal
    question category is unanswerable no matter how good the memory is.
    """
    exchanges: list[str] = []
    pending: list[str] = []
    pending_date: str | None = None

    def label(turn: dict) -> str:
        speaker = turn.get("speaker") or turn.get("role", "unknown")
        return f"{speaker}: {turn['content']}"

    def flush() -> None:
        if not pending:
            return
        header = f"[Session — {pending_date}]\n" if pending_date else ""
        exchanges.append(header + "\n".join(pending))

    for turn in turns:
        if turn.get("role") == "user":
            flush()
            pending = [label(turn)]
            pending_date = turn.get("date")
        else:
            # Assistant turns preceding any user turn still carry facts; keep
            # them rather than dropping them on the floor.
            pending.append(label(turn))
            if pending_date is None:
                pending_date = turn.get("date")
    flush()
    return exchanges


class BenchmarkArm(ABC):
    """Base class for all benchmark arms."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self.metrics = ArmMetrics(arm_name=name)

    @abstractmethod
    async def ingest_conversation(self, turns: list[dict]) -> None:
        """Feed conversation history into the memory system.

        Args:
            turns: List of {role: "user"|"assistant", content: str} dicts
        """

    @abstractmethod
    async def answer_question(self, question: str) -> ArmResponse:
        """Answer a question using the memory system.

        The arm should:
        1. Retrieve relevant context from its memory
        2. Generate an answer using the LLM with that context
        3. Return the answer with timing/token metrics
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all state between conversations/questions."""

    def update_storage_metrics(self) -> None:
        """Update hot/cold storage counts in metrics. Override in subclasses."""
        pass
