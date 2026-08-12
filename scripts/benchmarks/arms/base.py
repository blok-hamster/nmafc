"""Abstract base class for benchmark arms.

Each arm represents a different memory strategy being evaluated.
All arms share the same LLM for generation and answer questions
from the same dataset — they differ only in how they manage memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..evaluation.metrics import ArmMetrics, ArmResponse


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
