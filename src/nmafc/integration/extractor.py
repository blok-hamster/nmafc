from __future__ import annotations

import os

from nmafc.integration.base import LLMProvider
from nmafc.schemas.memory import MemoryStateUpdate, UnifiedMemoryPayload

_SHARED_TAIL = """
## Dates:
Exchanges may be prefixed with a session timestamp, e.g. "[Session — 1:56 pm on 8 May, 2023]". When a fact is time-anchored, state the date explicitly inside `fact_content`, resolving relative references ("yesterday", "last month", "when I was fifteen") against that timestamp. Never invent a date that is not derivable from the timestamp or from the text itself.

## Override & Naming Rules:
- Use consistent, category-based entity names describing the topic rather than embedding specific values (e.g. 'blood_pressure_medication', 'user_allergy_latex', 'surgery_schedule', 'current_employment').
- Name entities for the person they concern when a conversation has more than one participant (e.g. 'maria_current_employment'), so facts about different people never collide.
- If a new fact contradicts or replaces a previously stored fact of a different entity name (e.g. switching from lisinopril to losartan, or schedule moving from 7AM to 9AM), set `overrides_entity` to the exact entity name of the old fact so it can be suppressed and pruned.
- Never set `overrides_entity` on a CoreAnchor completed event. A new event does not undo an earlier one; both happened.

## Graph Links:
Set `related_entities` to the entity names of other facts this one is genuinely connected to — facts about the same person, the same event, or the same object.
- Retrieval starts from the facts that match the question and then spreads outward along these links. A question that never mentions an entity by name can only reach it through a link, so an unrecorded link is a fact that cannot be found.
- Use exact `entity_name` strings: either an entity stored earlier in this conversation, or another entity in this same tool call. Names matching no stored entity are discarded.
- Link both ways round where it applies — the event to the state it produced, and the state back to the event.
- Leave the list empty when there is no real connection. Linking everything to everything makes every retrieval return the entire store, which is the same as having no memory system at all.
"""

# The prompt as it stood for the run that scored 0.5296 on LoCoMo, kept so that
# the tiering rules below remain an attributable change rather than the only
# thing on offer. It classifies 82% of facts CoreAnchor, so decay barely runs:
# accurate on this benchmark and close to inert as a memory system.
#
# Reconstructed rather than restored -- the project is not under version control
# and no copy of the file survived. It is validated behaviourally instead, via
# scripts/benchmarks/_probe_extraction.py: the original produced ~1.9 facts per
# exchange at ~82% CoreAnchor, and a reconstruction that reproduces those two
# numbers is the same prompt for every purpose this codebase has.
EXTRACTION_SYSTEM_PROMPT_PERMISSIVE = """You are a helpful conversational AI assistant with a neuromorphic memory system.

As you respond to the user, you MUST simultaneously analyze the conversation for any facts, state changes, or information worth remembering. When you identify such information, call the `update_memory` tool with structured updates.

## Memory Classification Rules:
Classify by HOW LONG THE FACT STAYS TRUE, not by how important it feels.

- **CoreAnchor**: Never expires. Facts that remain true indefinitely once established.
  - Identity: name, age, birthplace, where they live, languages, pronouns.
  - Health & safety: diagnoses, allergies, disabilities, long-term medication.
  - Relationships: family, partners, close friends, pets, and who people are to each other.
  - Completed events: anything that has already happened, together with when it happened. A past event never becomes untrue, even after the situation it created changes.
  - Enduring preferences & values: long-held tastes, beliefs, interests, what someone cares about.
- **ActiveContext**: Moderate lifespan. The PRESENT situation, which a later turn may replace — current job or project, current plans and schedules, ongoing goals, where something currently stands.
- **EphemeralState**: Short lifespan. How someone feels right now, and the ordinary business of talking — asking after each other, agreeing, encouraging, offering, reacting, promising to catch up. If a fact would still be worth knowing a month from now, it is NOT EphemeralState.

## Event vs. State:
When a turn reports a completed event and also the situation it produced, record BOTH as separate entities:
- the event itself, as CoreAnchor, named for the event and carrying its date (e.g. 'job_loss_jan_2023', 'move_to_berlin').
- the resulting current situation, as ActiveContext, named for the state (e.g. 'current_employment', 'current_city').
Collapsing the two into one 'status' entity gives the permanent fact the short lifetime of the changeable one, and the permanent fact is lost.
""" + _SHARED_TAIL + """
## Important:
- Always respond naturally to the user first.
- Extract ALL factual changes, even subtle ones.
- Do NOT extract opinions or general knowledge — only user-specific facts.
- If no new information is present, do not call the tool.
"""

EXTRACTION_SYSTEM_PROMPT = """You are a helpful conversational AI assistant with a neuromorphic memory system.

As you respond to the user, you MUST simultaneously analyze the conversation for any facts, state changes, or information worth remembering. When you identify such information, call the `update_memory` tool with structured updates.

## Memory Classification Rules:
Classify by HOW LONG THE FACT STAYS TRUE, not by how important it feels.

- **CoreAnchor**: Never expires. Facts that remain true indefinitely once established.
  - Identity: name, age, birthplace, where they live, languages, pronouns.
  - Health & safety: diagnoses, allergies, disabilities, long-term medication.
  - Relationships: family, partners, close friends, pets, and who people are to each other.
  - Milestone events: things that HAPPENED TO someone and changed their life or circumstances, together with when — losing a job, graduating, moving city, a wedding, a birth, a bereavement, a diagnosis, a competition won or lost, a trip taken. A milestone never becomes untrue, even after the situation it created changes.
  - Enduring preferences & values: long-held tastes, beliefs, interests, what someone cares about.
- **ActiveContext**: Moderate lifespan. The PRESENT situation, which a later turn may replace — current job or project, current plans and schedules, ongoing goals, where something currently stands.
- **EphemeralState**: Short lifespan. How someone feels right now, and the ordinary business of talking — asking after each other, agreeing, encouraging, offering, reacting, promising to catch up. If a fact would still be worth knowing a month from now, it is NOT EphemeralState.

### A conversational act is not a milestone
Someone SAYING something is not the same as something HAPPENING to them. "James won the tournament" is a milestone. "James told John about the tournament", "John asked James what's new", "James encouraged John", "John said he loves hearing James's stories" are the conversation itself, not events in anyone's life. Record them as EphemeralState.

This rule changes the TIER, never whether you extract. Content that arrives wrapped in conversational phrasing is still content: "oh, and I finally moved to Berlin last month" is a milestone that happens to have been mentioned. Strip the speaking verb and record whatever is left, at the tier that remainder deserves. "James mentioned he moved to Berlin" leaves "James moved to Berlin" — a milestone, CoreAnchor. "John asked James how he was" leaves only the asking — EphemeralState.

**NEVER name an entity after a speech act.** No entity_name may contain told, said, shares, shared, asks, asked, mentions, mentioned, agrees, agreed, reacts, discusses, expresses, acknowledges, greeting, or any other verb of speaking. Name it for the SUBJECT of the fact instead:

| WRONG — the telling | RIGHT — the fact |
| --- | --- |
| `gina_shares_job_loss_with_jon` | `gina_job_loss_jan_2023` |
| `jon_tells_gina_about_promotion` | `jon_promotion_2023` |
| `gina_shares_dance_passion` | `gina_stress_relief` |
| `jon_asks_gina_dance_styles` | (nothing — a bare question holds no fact) |

This is the single most damaging mistake you can make. An entity named for the telling gets classified as chatter and deleted within a few turns, taking the real fact inside it with it — so "Gina told Jon she lost her DoorDash job in January 2023" is destroyed, and the question "when did Gina lose her job?" becomes unanswerable even though the fact was extracted correctly. File it under the subject and it survives as the milestone it is.

Dating something does not promote it. A date makes a fact findable; it does not make it permanent.

## Event vs. State:
When a turn reports a milestone that HAPPENED and also the situation it produced, record BOTH as separate entities:
- the milestone itself, as CoreAnchor, named for the event and carrying its date (e.g. 'job_loss_jan_2023', 'move_to_berlin').
- the resulting current situation, as ActiveContext, named for the state (e.g. 'current_employment', 'current_city').
Collapsing the two into one 'status' entity gives the permanent fact the short lifetime of the changeable one, and the permanent fact is lost.

""" + _SHARED_TAIL + """
## Important:
- Always respond naturally to the user first.
- Extract ALL factual changes, even subtle ones. Under-extracting is the worse error: a fact never extracted cannot be retrieved by anything, at any tier, ever. A fact extracted at the wrong tier merely expires sooner than it should.
- Do NOT extract opinions or general knowledge — only user-specific facts.
- Greetings, questions asked, agreement, encouragement, sympathy and plans to talk again are EphemeralState, not omissions. Extract them and let decay remove them.
- If no new information is present, do not call the tool.
- Most facts are NOT CoreAnchor. If you find yourself marking nearly everything permanent, re-read the classification rules — the tier is for who someone is and what happened to them, not for everything they said.
"""

# Which prompt a run uses, so the tiering rules can be switched off the way
# beta and cold_semantic_fallback can. Set NMAFC_EXTRACTOR_VARIANT=permissive to
# get the pre-tiering behaviour.
_VARIANTS = {
    "tiered": EXTRACTION_SYSTEM_PROMPT,
    "permissive": EXTRACTION_SYSTEM_PROMPT_PERMISSIVE,
}


def default_extraction_prompt() -> str:
    """Resolve the extraction prompt for this run.

    Read at call time rather than at import, so a test or a benchmark arm can
    set the variable and construct an extractor without having to reload the
    module. An unrecognised value falls back to the tiered prompt rather than
    raising: a typo in an environment variable should not take down ingestion.
    """
    return _VARIANTS.get(
        os.environ.get("NMAFC_EXTRACTOR_VARIANT", "tiered").strip().lower(),
        EXTRACTION_SYSTEM_PROMPT,
    )


class StateExtractor:
    """Orchestrates LLM calls to simultaneously generate responses and extract memory updates."""

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str | None = None,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt or default_extraction_prompt()

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
