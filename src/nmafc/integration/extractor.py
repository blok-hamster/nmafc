from __future__ import annotations

from nmafc.integration.base import LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, UnifiedMemoryPayload

EXTRACTION_SYSTEM_PROMPT = """You are a helpful conversational AI assistant with a neuromorphic memory system.

As you respond to the user, you MUST simultaneously analyze the conversation for any facts, state changes, or information worth remembering. When you identify such information, call the `update_memory` tool with structured updates.

## Memory Classification Rules:
- **CoreAnchor**: Permanent identity facts (name, medical conditions, allergies, life-safety info). These never expire.
- **ActiveContext**: Current state that may change (schedules, active goals, ongoing tasks, relationships). Moderate lifespan.
- **EphemeralState**: Transient information (current mood, passing comments, small talk). Short lifespan.

## Override & Naming Rules:
- Use consistent, category-based entity names describing the topic rather than embedding specific values (e.g. 'blood_pressure_medication', 'user_allergy_latex', 'surgery_schedule', 'user_job').
- If a new fact contradicts or replaces a previously stored fact of a different entity name (e.g. switching from lisinopril to losartan, or schedule moving from 7AM to 9AM), set `overrides_entity` to the exact entity name of the old fact so it can be suppressed and pruned.

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
