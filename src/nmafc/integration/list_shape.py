"""Is this question asking for a list, or for one thing?

The real LoCoMo run put single-hop at 50.9% against RAG's 45.6%, the only
category whose lead is indistinguishable from noise. Split by how long the gold
answer is, it is not one category:

    gold 1-3 words        138 questions   we are right 68.1%
    gold 4-8 words        103 questions   we are right 35.9%
    gold 9+ words          40 questions   we are right 30.0%

Reading the long-gold failures shows one question shape over and over. "What
activities does Melanie partake in" wants four. "What events for veterans has
John participated in" wants five. "What exercises has John done" wants four.
The gold is a list, the context holds part of it, and a partial list is scored
wrong exactly like a wrong answer. The reach diagnosis agrees: on the questions
both arms fail, 55% arrive as most-of-it and only 20% as nothing at all.

Width was the obvious fix and it does not work. MEASURED, 10 September: every
question this gate fires on, answered both ways in one window, 322 of them.
Wide scores 50.9% against the shipped 49.4%, +17/-12, p=0.458, for 324 extra
tokens a question. Single-hop, the category the gate was built off, goes
backwards at -1.4. An earlier read on 108 open-domain questions had it at +5.6
and that was nine coin flips landing one way; it regressed the moment there
were more of them.

So nothing here is wired into a shipping path, and this module is kept for two
reasons. The detection is sound and cheap, and the shape it finds is real: we
score 50.2% on the questions it fires on against 66.0% on the ones it does not.
That gap is worth attacking. Handing those questions more context is not how.

Same shape as `answer_type.gate` and for the same reason: a regex over the
question, no embedding, no model call, no retrieval. The difference is what it
qualifies. `answer_type` changes the prompt, this changes how much is fetched.

Deliberately narrow. Firing on a single-answer question buys nothing and costs
tokens, so the rule is a *plural head noun in the wh-phrase* and a small set of
explicit list phrasings, not a guess at what the asker had in mind.
"""

from __future__ import annotations

import re

# Words ending in "s" that are not plural nouns. Without this, "What is Joanna
# allergic to", "What has Melanie done" and "What does Jon offer" all match a
# bare `\w+s` head and the gate fires on most of the category, which is the
# same as not having a gate.
_NOT_PLURAL = frozenset(
    "is was has does his its this thus us yes less unless whereas perhaps "
    "always sometimes across towards besides plus versus".split()
)

# The head of the wh-phrase, allowing a couple of modifiers in between:
# "what LGBTQ+ events", "what musical artists", "which of Joanna's screenplays".
# The head must be the last word before the verb, so the modifiers are
# non-greedy and the noun is anchored on a word boundary.
#
# The modifiers admit an apostrophe and the head does not, and that asymmetry
# is load-bearing rather than tidy. "What are Joanna's hobbies?" offers
# "Joanna's" as the first word ending in s, and a head class that accepted it
# would capture the possessive, have it rejected as a non-plural, and then find
# nothing else: `finditer` has already consumed the wh-word, so "hobbies" is
# never reached and a plainly list-shaped question comes back quiet. Barring
# the apostrophe from the head makes the engine backtrack past the possessive
# and take it as the modifier it is.
_PLURAL_HEAD = re.compile(
    r"\b(?:what|which)\s+"
    r"(?:(?:kind|kinds|type|types|sort|sorts)\s+of\s+)?"
    r"(?:(?:some|any|all|the|of|a|an)\s+)*"
    r"(?:[\w'+-]+\s+){0,3}?"
    r"([\w+-]+s)\b",
    re.I,
)

# Phrasings that ask for everything without a plural noun to key on. "In what
# ways" is the clearest and appears verbatim in the failures; the rest are the
# ordinary ways English says "all of them".
_EXPLICIT = re.compile(
    r"\b(?:in\s+what\s+ways|what\s+are\s+some|list\s+(?:all|the|every)|"
    r"name\s+(?:all|every)|all\s+of\s+the\s+\w+s\b|"
    r"what\s+all\b|each\s+of\s+the\b)",
    re.I,
)

# A possessive is a modifier, not the head. `_PLURAL_HEAD` already keeps the
# common "'s" form out of the capture, so this is the belt to that braces: it
# catches the plural possessive "the girls' names", where the head genuinely
# ends in a bare s and the apostrophe is the only thing saying so.
_POSSESSIVE = re.compile(r"'s$|s'$")


def _is_plural_noun(word: str) -> bool:
    lowered = word.lower()
    if lowered in _NOT_PLURAL or _POSSESSIVE.search(lowered):
        return False
    # "ss" endings are almost never plurals in this corpus -- "business",
    # "illness", "progress" -- and each one that slips through widens a
    # single-answer question at full token cost.
    return len(lowered) > 3 and not lowered.endswith("ss")


def wants_list(question: str) -> bool:
    """True when the question asks to enumerate rather than to name one thing.

    False on anything it is unsure about. The cost of a miss is a question that
    stays as hard as it is today; the cost of a false fire is real tokens on a
    question that never needed them, and the whole reason this exists is that
    the wide configuration cannot be afforded everywhere.
    """
    if not question:
        return False
    if _EXPLICIT.search(question):
        return True
    for match in _PLURAL_HEAD.finditer(question):
        if _is_plural_noun(match.group(1)):
            return True
    return False
