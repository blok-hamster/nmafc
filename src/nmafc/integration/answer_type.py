"""What kind of thing is the question asking for?

A question often names the category of its own answer -- "in which *state*",
"what *console*", "how *old*" -- and when it does, that category is a constraint
the retrieved evidence usually cannot supply on its own. Asked which state a
shelter is in, the store held "Stamford" and the answer given was "Stamford".
Asked which country a pendant was bought in, the store held "Paris" and the
answer given was "Paris". Both are evidence for the answer rather than the
answer, and both were counted wrong.

That is not a retrieval failure and no amount of extra context fixes it: the
right fact was already in the window. It is caused by the answering rules, which
say to reuse the exact wording of the facts because a synonym scores as a miss.
On nearly every question that is correct and worth a large part of the score. On
a question that names the kind of answer it wants it is exactly wrong, because
"Connecticut" is not a synonym for "Stamford" -- it is the level the question
asked for, and "Stamford" is the reason to believe it.

Measured over 1,535 paired questions before this existed: on the 238 that name a
type we scored 56.7% against RAG's 61.8%, and on the remaining 1,297 we scored
66.7% against 64.9%. The whole of the deficit lives in the typed questions, and
both arms fail them together, which is what a shared prompt rule looks like from
the outside.

Deliberately narrow, for two reasons that are really one:

  * **Only when the question names the type.** A directive derived from a
    question that named nothing would be a guess about what the asker meant, and
    a guess in the prompt is indistinguishable from a hallucination in the
    answer.
  * **Only a level shift, never an invention.** The directive licenses naming
    the state a retrieved town sits in. It does not license naming a state when
    nothing retrieved points at one, and it says so, because the failure mode
    this trades against is answering "Connecticut" from no evidence at all.

The cost is a short line on the ~15% of questions that name a type, and nothing
at all on the rest. No embedding, no model call, no retrieval: it is a regex over
the question, which is already in the prompt.
"""

from __future__ import annotations

import re

# Categories whose *level* is fixed. A question naming one of these is asking
# for a specific rung of a hierarchy, which is the whole mechanism: the evidence
# sits on a different rung and has to be climbed.
#
# "place", "thing", "way" and "reason" are excluded on purpose. They name no
# level, so a question containing them constrains nothing, and emitting a
# directive for them would put words in the model's mouth about a question that
# asked for none.
_TYPE_NOUNS = (
    # geography, the clearest hierarchy and the one that failed most
    "state|country|city|town|village|province|county|region|continent|nation|"
    # calendar, where a bare year against a gold month is the same error
    "month|year|day|date|weekday|season|decade|century|"
    # made things, where maker and model are different rungs
    # "make" is absent on purpose. It earns its place only in "what make of
    # car", which nobody in this corpus asks, and it is the commonest verb in
    # the corpus, which is how "What did Maria make for her home" came to be
    # tagged as a question about manufacturers.
    "console|device|gadget|appliance|vehicle|brand|model|manufacturer|"
    "company|firm|employer|team|band|network|platform|app|website|"
    # people and groups
    "language|nationality|religion|breed|species|"
    # work and study
    "job|profession|occupation|career|role|degree|major|field|discipline|"
    "subject|qualification|certification|"
    # pastimes and works, where a series and one of its books are different
    "sport|instrument|genre|dish|cuisine|drink|meal|ingredient|"
    "book|novel|movie|film|show|series|song|album|game|colour|color"
)

# Words that may sit between the wh-word and the noun without being part of the
# type. "which US state" wants "US state" and the qualifier matters; "what are
# the breeds" wants "breed" and "are the" is grammar. Without this the directive
# reads "answer with a thing of kind: are the breeds", which is noise dressed as
# a constraint.
_NOT_QUALIFIERS = frozenset(
    "a an the is are was were do does did has have had will would can could "
    "of in on at to for other others else about your his her their my our its "
    "some any all these those this that".split()
)

_NAMED = re.compile(
    rf"\b(?:which|what)\s+"
    rf"(?:kind\s+of\s+|sort\s+of\s+|type\s+of\s+)?"
    rf"((?:[\w'-]+\s+){{0,3}}?({_TYPE_NOUNS})s?)\b",
    re.I,
)

# Wh-phrases that fix a type without naming a noun. "How old" admits an age and
# nothing else. These would be invisible to a noun list, and "a number" was the
# single most-failed demanded type in the measurement above.
_IMPLIED: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bhow\s+old\b", re.I), "an age"),
    (re.compile(r"\bhow\s+(?:many|much)\b", re.I), "a number"),
    (re.compile(r"\bhow\s+(?:often|frequently)\b", re.I), "a frequency"),
    (re.compile(r"\bhow\s+long\b", re.I), "a length of time"),
)

# Split in two, and the split is the whole reason this fits.
#
# The first version put the reasoning and the category into one line and emitted
# it per question. At 51 tokens it fired on a third of multi-hop and pushed the
# mean context from 992 to 1,009, which breaks the sub-1,000 claim to fix a
# class worth a few questions. Bad trade, and avoidable: the reasoning is
# identical on every question, so paying for it once and quoting only the
# category per question costs 6 tokens where it fires and nothing where it does
# not.
#
# TYPE_RULE goes in the arm's system prompt, where RAG's own instructions live,
# and is not retrieved context. `type_tag` is the only part charged per
# question, and it is derived from the question rather than fetched from the
# store, so it is not context either -- but it is counted anyway, because a
# mechanism that hides its cost in a place the metric does not look is a
# mechanism nobody can check.
TYPE_RULE = (
    "\nWhen an 'Asked for:' line names the kind of answer wanted, give a thing "
    "of that kind. If the facts give something narrower that fixes it, a town "
    "when a state is asked, name what was asked for and not the evidence for "
    "it. If nothing in the facts fixes one, answer from the facts as they "
    "stand rather than inventing one."
)

# Written with a colon and no article so that the detector never has to choose
# one. "a US state" and "an umbrella" cannot both be got right by looking at the
# first letter, and an article is worth nothing here anyway: the model is being
# handed a category, not reading a sentence.
_TAG = "Asked for: {}."


def _qualified(span: str) -> str | None:
    """The type named by a matched wh-phrase, or None if it is not one.

    The span runs from the wh-word to the type noun, so in "What are the breeds"
    it is "are the breeds" and the auxiliary and article have to go. The first
    version dropped every word it did not like and kept the rest, which is why
    the paired run tagged "What did Maria make for her home" as asking for a
    "Maria make" and "What happened to John's job" as asking for a "happened
    John's job". Both are verb clauses that happen to end on a noun the list
    knows.

    Dropping only a *leading* run of them tells the two apart. Auxiliaries and
    articles come before the type and mean nothing: "are the breeds" is a
    demand for breeds. One appearing after something else means the noun is not
    the head of the wh-phrase but the object of a verb, and the question is
    asking what happened rather than what kind of thing it was.
    """
    words = span.split()
    lead = 0
    while lead < len(words) and words[lead].lower() in _NOT_QUALIFIERS:
        lead += 1
    rest = words[lead:]
    if not rest or any(w.lower() in _NOT_QUALIFIERS for w in rest):
        return None
    return " ".join(rest)


def demanded_type(question: str) -> str | None:
    """The kind of answer this question names, or None if it names none.

    A bare category, no article: `US state`, `books`, `video games`. The implied
    types carry their own wording because there is no noun to quote.

    Plurality is preserved rather than normalised away, because a question
    asking for `books` is asking for all of them and the answering rules already
    treat a list differently from a single name.
    """
    if not question:
        return None

    for match in _NAMED.finditer(question):
        found = _qualified(match.group(1))
        if found:
            return found

    for pattern, label in _IMPLIED:
        if pattern.search(question):
            return label
    return None


def type_tag(question: str) -> str:
    """The per-question line, or "" when the question names no type.

    Empty on roughly four questions in five, which is the point twice over: a
    rule that fires everywhere is a rule the model stops reading, and charging
    every question for a class that appears in one in five is how a cheap fix
    becomes an expensive one.

    Meaningless without `TYPE_RULE` in the system prompt, which is what tells
    the model what an "Asked for:" line is. The two ship together or not at all.
    """
    demanded = demanded_type(question)
    return _TAG.format(demanded) if demanded else ""


def gate(question: str) -> tuple[str, str]:
    """The rule and the tag for one question, or two empty strings.

    The first version of this arm put `TYPE_RULE` in the system prompt on every
    question and let the tag decide whether it applied. That is one instruction
    too many on the four questions in five that name no type, and the paired run
    charged for it: the untyped controls came back 1.8 points down, 3 fixed
    against 7 broken. Not significant, and not nothing either -- a rule about
    naming the level asked for is a strange thing to read when nothing has been
    asked for at a level.

    Gating the rule on the tag removes that cost rather than measuring it. The
    system prompt is built per call, so on a question the detector does not fire
    on, a gated arm sends the shipped prompt unchanged -- byte-identical, same
    question, same retrieved context, the same call the shipped arm makes. There
    is no difference left to cost anything.

    Which also means the untyped half of any future A/B here is not worth paying
    for. Two identical prompts differ only by the model's own sampling, and
    `_ab_answer_type.py` copies arm A's answer across rather than buying a
    second sample of it.

    Returns `(rule, tag)`. The caller decides where the rule goes, because that
    depends on the wording it is qualifying.
    """
    tag = type_tag(question)
    return (TYPE_RULE, tag) if tag else ("", "")
