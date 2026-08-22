"""Abstract base class for benchmark arms.

Each arm represents a different memory strategy being evaluated.
All arms share the same LLM for generation and answer questions
from the same dataset — they differ only in how they manage memory.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable

from ..evaluation.metrics import ArmMetrics, ArmResponse


_PREAMBLE_PATTERNS = re.compile(
    r"^(?:"
    r"based on (?:the |my )?(?:facts|memories|information)\b[^.]*?[,.:]\s*"
    r"|according to (?:the |my )?(?:facts|memories)\b[^.]*?[,.:]\s*"
    r"|(?:looking|going) (?:at|through) (?:the |my )?(?:facts|memories)\b[^.]*?[,.:]\s*"
    r"|(?:from|reviewing|examining) (?:the |my )?(?:facts|memories)\b[^.]*?[,.:]\s*"
    r"|the facts (?:show|indicate|suggest|provided|state) (?:that )?"
    r"|i (?:don't|do not|cannot|can't) (?:have|find|see|determine)\b[^.]*?[.]\s*(?:however|but)[,:]?\s*"
    r"|i'?ve? reviewed [^.]*?[.]\s*(?:however|but)[,:]?\s*"
    r"|there is no (?:record|info|information|mention|data)[^.]*?[.]\s*(?:however|but|looking)[,:]?\s*"
    r"|let me (?:check|look|review)[^\n]*?\n"
    r"|i can see (?:from |that )[^.]*?(?:that |[:,]\s*)"
    r")",
    re.IGNORECASE,
)

_TRAILING_NOISE = re.compile(
    r"[\s.]*(?:\(Valid:.*?\)|\n.*)",
    re.DOTALL,
)


def strip_answer(raw: str) -> str:
    """Post-process LLM response to extract just the bare answer.

    Removes common preamble patterns, trailing explanations, and markdown.
    """
    text = raw.strip()
    if not text:
        return text

    # Remove markdown bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)

    # Strip numbered lists — take just the items
    lines = text.split("\n")
    if len(lines) > 1 and re.match(r"^\d+[.)]\s", lines[0]):
        items = []
        for line in lines:
            item = re.sub(r"^\d+[.)]\s*", "", line).strip()
            if item and not item.startswith("("):
                items.append(item)
        if items:
            text = ", ".join(items)

    # Strip preamble
    text = _PREAMBLE_PATTERNS.sub("", text).strip()

    # Split on sentence boundaries and drop meta/negation sentences
    if ". " in text and len(text) > 40:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if re.match(r"(?i)^(however|but|note|also|additionally|i |there is no|no information|looking)", s):
                continue
            if re.search(r"(?i)(no (?:info|record|mention|data|fact)|not (?:available|found|specified|provided))", s):
                continue
            if re.search(r"(?i)^(the only mention|the (?:facts|memories) (?:don|do not|only))", s):
                continue
            kept.append(s.rstrip("."))
        if kept:
            text = kept[0] if len(kept) == 1 else ". ".join(kept)

    # Strip trailing validity annotations or explanations after newline
    text = _TRAILING_NOISE.sub("", text).strip()

    # Strip framing clauses: "X is Y" → "Y" when X is a meta-reference
    text = re.sub(
        r"(?i)^(?:the only mention of \w+ (?:researching|doing|is) (?:is |that )?|"
        r"\w+ has participated in:?\s*|"
        r"\w+ (?:appears|seems) to be\s+)",
        "", text,
    ).strip()

    # Remove leading/trailing quotes and periods
    text = text.rstrip(".")
    if text.startswith(("'", '"')) and text.endswith(("'", '"')):
        text = text[1:-1]

    return text


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

CRITICAL: You are completing a fill-in-the-blank quiz. Your response is scored by EXACT TOKEN OVERLAP with a 1-5 word reference answer. Every extra word LOWERS your score.

Rules:
- Reply with ONLY the bare answer. No sentences, no explanation, no context.
- Maximum 5 words unless listing items.
- For lists: comma-separated nouns only (e.g. "running, pottery, camping").
- For dates: just the date (e.g. "19 January 2023" or "July 2023").
- For yes/no: just "Yes" or "No".
- For names: just the name (e.g. "Sweden" or "Oliver, Luna, Bailey").
- For "would/could/likely" questions: answer "Yes" or "No" then at most 3 words of reason.
- NEVER start with "Based on", "According to", "The facts show", "Looking at", "I don't have", or any preamble.
- NEVER say "No information available" — always attempt an answer from the facts, even if uncertain.

Examples of CORRECT responses:
Q: When did she go camping? → June 2023
Q: What are her pets' names? → Oliver, Luna, Bailey
Q: What does she do to relax? → running, pottery
Q: Would she enjoy classical music? → Yes, she likes Bach and Mozart
Q: What is his job? → software engineer
Q: Where did she move from? → Sweden"""


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

    # Whether this arm can reopen a partly-ingested store and carry on from the
    # middle of a conversation. False for the stateless arms, which hold their
    # context in memory and have nothing on disk to resume from, and for which
    # re-ingesting is nearly free anyway. The runner checks this before offering
    # a resume, so an arm that cannot honour `start_at` is never handed one.
    supports_ingest_resume: bool = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.metrics = ArmMetrics(arm_name=name)

    def prepare_store(
        self, store_dir: str, conversation_id: str, fingerprint: str
    ) -> int:
        """Point the arm at `store_dir` and report exchanges already ingested.

        The default is the old behaviour: throw away all state and start the
        conversation from the beginning. Arms with a durable store override this
        to reopen one whose recorded conversation and settings match, and return
        how far it got.

        Returns:
            Number of leading exchanges already in the store; 0 to ingest all.
        """
        self.reset()
        return 0

    @abstractmethod
    async def ingest_conversation(
        self,
        turns: list[dict],
        start_at: int = 0,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Feed conversation history into the memory system.

        Args:
            turns: List of {role: "user"|"assistant", content: str} dicts
            start_at: Skip this many leading exchanges, already ingested by an
                interrupted earlier run. Only meaningful when
                `supports_ingest_resume` is True.
            on_progress: Called after each exchange with
                (exchanges_done, turn_clock), so the caller can checkpoint.
                Exchanges are counted from the start of the conversation, not
                from `start_at`, so the number means the same thing on a resumed
                run as on a fresh one.
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
