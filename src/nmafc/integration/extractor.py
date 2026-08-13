from __future__ import annotations

from nmafc.integration.base import LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, UnifiedMemoryPayload

EXTRACTION_SYSTEM_PROMPT = """You are a helpful conversational AI assistant with a neuromorphic memory system.

As you respond to the user, you MUST simultaneously analyze the conversation for any facts, state changes, or information worth remembering. When you identify such information, call the `update_memory` tool with structured updates.

## Memory Classification Rules:
Classify by HOW LONG THE FACT STAYS TRUE, not by how important it feels.

- **CoreAnchor**: Never expires. Facts that remain true indefinitely once established.
  - Identity: name, age, birthplace, where they live, languages, pronouns.
  - Health & safety: diagnoses, allergies, disabilities, long-term medication.
  - Relationships: family, partners, close friends, pets, and who people are to each other.
  - Completed events: anything that has already happened, together with when it happened. A past event never becomes untrue — someone losing a job, graduating, moving city or winning something stays a fact forever, even after the situation it created changes.
  - Enduring preferences & values: long-held tastes, beliefs, interests, what someone cares about.
- **ActiveContext**: Moderate lifespan. The PRESENT situation, which a later turn may replace — current job or project, current plans and schedules, ongoing goals, where something currently stands.
- **EphemeralState**: Short lifespan. Only genuinely momentary things — how someone feels right now, passing remarks, small talk. If a fact would still be worth knowing a month from now, it is NOT EphemeralState.

## Event vs. State:
When a turn reports something that HAPPENED and also the situation it produced, record BOTH as separate entities:
- the event itself, as CoreAnchor, named for the event and carrying its date (e.g. 'job_loss_jan_2023', 'move_to_berlin').
- the resulting current situation, as ActiveContext, named for the state (e.g. 'current_employment', 'current_city').
Collapsing the two into one 'status' entity gives the permanent fact the short lifetime of the changeable one, and the permanent fact is lost.

## Dates:
Exchanges may be prefixed with a session timestamp, e.g. "[Session — 1:56 pm on 8 May, 2023]". When a fact is time-anchored, state the date explicitly inside `fact_content`, resolving relative references ("yesterday", "last month", "when I was fifteen") against that timestamp. Never invent a date that is not derivable from the timestamp or from the text itself.

## Override & Naming Rules:
- Use consistent, category-based entity names describing the topic rather than embedding specific values (e.g. 'blood_pressure_medication', 'user_allergy_latex', 'surgery_schedule', 'current_employment').
- Name entities for the person they concern when a conversation has more than one participant (e.g. 'maria_current_employment'), so facts about different people never collide.
- If a new fact contradicts or replaces a previously stored fact of a different entity name (e.g. switching from lisinopril to losartan, or schedule moving from 7AM to 9AM), set `overrides_entity` to the exact entity name of the old fact so it can be suppressed and pruned.
- Never set `overrides_entity` on a CoreAnchor completed event. A new event does not undo an earlier one; both happened.

## Important:
- Always respond naturally to the user first.
- Extract ALL factual changes, even subtle ones.
- Do NOT extract opinions or general knowledge — only user-specific facts.
- If no new information is present, do not call the tool.
"""


class StateExtractor:
    """Orchestrates LLM calls to simultaneously generate responses and extract memory updates."""

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str | None = None,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt or EXTRACTION_SYSTEM_PROMPT

    async def extract(
        self,
        user_msg: str,
        context: list[dict] | None = None,
        memory_context: str | None = None,
    ) -> tuple[str, UnifiedMemoryPayload]:
        """Process a user message and extract memory updates.

        Args:
            user_msg: The user's message text.
            context: Prior conversation messages.
            memory_context: Formatted string of retrieved memories to inject.

        Returns:
            Tuple of (assistant_response, unified_memory_payload)
        """
        messages = list(context) if context else []
        messages.append({"role": "user", "content": user_msg})

        system = self._system_prompt
        if memory_context:
            system += f"\n\n## Currently Active Memories:\n{memory_context}"

        response_text, updates = await self._provider.chat_with_extraction(
            messages=messages,
            system_prompt=system,
        )

        return response_text, UnifiedMemoryPayload(updates=updates)
