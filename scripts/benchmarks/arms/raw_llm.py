"""Arm 1: Raw LLM with full context window stuffing.

This baseline represents the simplest approach — stuff the entire conversation
history into the context window and let the LLM answer directly.
No memory system, no retrieval, no state management.

This is equivalent to the "Base" and "Long-context" conditions in the LoCoMo paper
and the "Full-conversation" baseline in the Zep paper.
"""

from __future__ import annotations

import time

from nmafc.integration.base import LLMProvider

from ..evaluation.metrics import ArmResponse
from .base import BenchmarkArm, SHORT_ANSWER_RULES

ANSWER_SYSTEM_PROMPT = (
    "You are a conversational AI assistant with access to the full "
    "conversation history below.\n"
    "Answer the user's question based ONLY on information from the "
    "conversation history.\n"
    "If the answer is not in the history, say \"I don't know\" or "
    "\"This information is not available.\"\n"
    "Be concise — answer in a few words or a short phrase when possible."
    + SHORT_ANSWER_RULES
)

CONTEXT_PREFIX = "=== CONVERSATION HISTORY ===\n"
CONTEXT_SUFFIX = "\n=== END HISTORY ===\n\n"


class RawLLMArm(BenchmarkArm):
    """Full conversation history stuffed into context window."""

    def __init__(self, llm_provider: LLMProvider, max_context_chars: int = 200_000) -> None:
        super().__init__(name="raw_llm")
        self._llm = llm_provider
        self._history: list[dict] = []
        self._max_chars = max_context_chars

    async def ingest_conversation(self, turns: list[dict]) -> None:
        """Store conversation turns verbatim."""
        self._history.extend(turns)

    async def answer_question(self, question: str) -> ArmResponse:
        """Answer by injecting full history into system prompt."""
        history_text = self._format_history()
        context_tokens = len(history_text) // 4  # Approximate token count

        system = ANSWER_SYSTEM_PROMPT + "\n\n" + CONTEXT_PREFIX + history_text + CONTEXT_SUFFIX

        start = time.perf_counter()
        response_text, _ = await self._llm.chat_with_extraction(
            messages=[{"role": "user", "content": question}],
            system_prompt=system,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        prompt_tokens = (len(system) + len(question)) // 4
        completion_tokens = len(response_text) // 4

        response = ArmResponse(
            answer=response_text.strip(),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_tokens=context_tokens,
        )
        self.metrics.record(response)
        return response

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []

    def _format_history(self) -> str:
        """Format history as text, truncating from the start if too long.

        Session timestamps are emitted as headers whenever the date changes, so
        temporal questions ("When did X happen?") are answerable at all, and
        turns are attributed to real speaker names rather than User/Assistant,
        since the questions refer to people by name.
        """
        lines = []
        current_date = None
        for turn in self._history:
            date = turn.get("date")
            if date and date != current_date:
                lines.append(f"\n[Session — {date}]")
                current_date = date
            speaker = turn.get("speaker") or turn["role"].capitalize()
            lines.append(f"{speaker}: {turn['content']}")

        full_text = "\n".join(lines)

        if len(full_text) > self._max_chars:
            full_text = full_text[-self._max_chars:]
            # Find first complete line after truncation
            newline_idx = full_text.find("\n")
            if newline_idx > 0:
                full_text = "[...truncated...]\n" + full_text[newline_idx + 1:]

        return full_text
