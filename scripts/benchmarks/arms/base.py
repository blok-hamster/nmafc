"""Abstract base class for benchmark arms.

Each arm represents a different memory strategy being evaluated.
All arms share the same LLM for generation and answer questions
from the same dataset — they differ only in how they manage memory.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from ..evaluation.metrics import ArmMetrics, ArmResponse
from ..resilience import throttled_seconds


def timer_start() -> tuple[float, float]:
    """Mark the start of a timed region, capturing the quota-wait baseline."""
    return time.perf_counter(), throttled_seconds()


def timer_split(mark: tuple[float, float]) -> tuple[float, float]:
    """Split elapsed time since `mark` into (work_ms, throttle_ms).

    Every arm calls a rate-limited provider inside its own stopwatch, so the
    raw elapsed time contains however long the shared deployment quota made it
    wait. That wait is not a property of the arm: the raw_llm arm spends ~20,000
    tokens per question and the memory arms ~650, all drawing on one bucket, so
    the arm that happens to be running when the bucket drains absorbs the
    delay. Reporting the two apart is what makes latency comparable at all.

    Clamped at zero because the throttle account is charged from a different
    clock (`time.monotonic`) than the stopwatch (`time.perf_counter`), and
    nothing guarantees their rounding agrees to the microsecond.
    """
    started, throttle_at_start = mark
    elapsed_ms = (time.perf_counter() - started) * 1000
    throttle_ms = (throttled_seconds() - throttle_at_start) * 1000
    return max(0.0, elapsed_ms - throttle_ms), throttle_ms


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
#
# Revised 3 September 2026 after auditing where the F1 actually goes. The
# earlier version of these rules was written as if the failure were verbosity in
# general, and it is not: the median answer is already 4 words against a gold
# median of 3, and stripping trailing hedges off the finished run moved mean F1
# by +0.001. The loss is concentrated in a 27% tail scoring 0.207 against 0.601
# for the rest, and 250 of those 418 answers are open-domain questions -- the
# largest scored category, and the one case these rules said nothing about.
# Every worked example above covered dates, lists, names or yes/no, so on
# "why did she..." the model had no format to copy and wrote a sentence.
#
# The second half of that tail is vocabulary rather than length. Against gold
# "personal style and customer comfort" the run answered "chose furniture that
# reflects her own style while making customers feel cozy" -- a fair paraphrase
# that shares one scoring token, because "comfort" was restated as "cozy".
# Token overlap does not reward understanding, it rewards the same words, so the
# rules now ask for the source wording to be reused rather than improved on.
SHORT_ANSWER_RULES = """

CRITICAL: You are completing a fill-in-the-blank quiz. Your response is scored by EXACT TOKEN OVERLAP with a 1-5 word reference answer. Every extra word LOWERS your score.

Rules:
- Reply with ONLY the bare answer. No sentences, no explanation, no context.
- Maximum 5 words unless listing items.
- For lists: comma-separated nouns only (e.g. "running, pottery, camping"). Name EVERY item the facts support, not just the best-matching one. A question asking what someone does, has, likes, owns or has taken part in is asking for all of them; answering with one is the single most common error on these.
- For dates: just the date (e.g. "19 January 2023" or "July 2023"). Give day, month and year whenever the facts support all three — a bare year is scored as a miss. If the facts date something only in relation to another date ("the Sunday before 25 May 2023", "the week before 9 June 2023"), answer in that same relative form rather than converting it.
- For yes/no: just "Yes" or "No".
- For names: just the name (e.g. "Sweden" or "Oliver, Luna, Bailey").
- For "would/could/likely" questions: answer "Yes" or "No" then at most 3 words of reason.
- For "why", "what does X think/feel", "what motivated X", "how did X" questions: give the reason as a bare phrase, not a sentence. Do not repeat the person's name, do not restate the question, do not add what happened next. "wanted a career change" — not "Gina decided to leave because she wanted a career change."
- COPY THE WORDING FROM THE FACTS. Reuse the exact nouns and adjectives that appear in the retrieved facts rather than substituting your own. A synonym scores as a miss: if the facts say "delighted", answer "delighted" and not "very happy"; if they say "apartment", answer "apartment" and not "flat".
- Once you have given the answer, STOP. Never append a caveat about what the facts do or do not confirm, and never dispute a date in the question. If the facts support the answer for a nearby date, give the answer plainly.
- NEVER start with "Based on", "According to", "The facts show", "Looking at", "I don't have", or any preamble.
- NEVER say "No information available" — always attempt an answer from the facts, even if uncertain.

Examples of CORRECT responses:
Q: When did she go camping? → June 2023
Q: What are her pets' names? → Oliver, Luna, Bailey
Q: What does she do to relax? → running, pottery
Q: Would she enjoy classical music? → Yes, she likes Bach and Mozart
Q: What is his job? → software engineer
Q: Where did she move from? → Sweden
Q: Why did he take up the guitar? → wanted a creative outlet
Q: What did she consider when picking the flat? → the light and the commute
Q: What motivated her to keep training? → seeing her own progress
Q: How is the new job going? → busy but rewarding"""


def build_exchanges(turns: list[dict]) -> list[str]:
    """The exchange text alone, for callers that do not carry dates.

    Kept as the exchange list it always was so that resume arithmetic, the
    checkpoint fingerprint and every diagnostic that replays turn order keep
    indexing the same way.
    """
    return [text for text, _ in build_dated_exchanges(turns)]


def build_dated_exchanges(turns: list[dict]) -> list[tuple[str, str | None]]:
    """Group a transcript into speaker-labelled exchanges, each with its date.

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

    The date is also returned alongside the text, rather than only baked into
    the header, because the header alone was not enough. It reaches the
    extractor, but what the store kept was a turn number, so facts came back to
    the model as "(Valid: turn 220 - present)" and dated questions were
    unanswerable at the far end instead of the near one. Handing the date to
    the memory as well lets it be read back as a date.
    """
    exchanges: list[tuple[str, str | None]] = []
    pending: list[str] = []
    pending_date: str | None = None

    def label(turn: dict) -> str:
        speaker = turn.get("speaker") or turn.get("role", "unknown")
        return f"{speaker}: {turn['content']}"

    def flush() -> None:
        if not pending:
            return
        header = f"[Session — {pending_date}]\n" if pending_date else ""
        exchanges.append((header + "\n".join(pending), pending_date))

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

    def compact_storage(self) -> bool:
        """Run between-conversation store maintenance. Override in subclasses.

        Called by the runner once a conversation's questions are answered, which
        is the only safe moment for it: compaction costs seconds up front and
        repays them across later reads, so it belongs between conversations
        rather than inside one. Arms with no durable store have nothing to do
        here and return False.
        """
        return False
