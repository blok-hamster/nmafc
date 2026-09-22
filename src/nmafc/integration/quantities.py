"""Numbers, treated as content rather than as short words.

`cue_words` tokenises on `[a-z0-9']+` and keeps what is longer than two
characters. Every number below 100 is therefore invisible to it: `42`, `7`, `$5`
and `12%` all vanish, while `2023` survives by being four digits long. That is an
accident of a length filter written for stopwords, and it has two consequences
that matter once a conversation carries quantities.

The first is a correctness bug in pattern separation. "He paid 42 dollars" and
"He paid 47 dollars" reduce to the same content words, `{paid, dollars}`, so
containment is 1.0 and one of them is dropped as a restatement of the other. They
are not restatements. They are two different amounts, and the one that survives
is decided by which happened to rank higher.

The second is a ranking bug in hydration. When a turn is trimmed to its
best-matching lines, the line carrying the figure the question asks about scores
no better than any other line in the turn, because the figure is not in the cue.

Both are fixed by the same thing: pull numbers out separately, normalise them,
and compare them as their own class of evidence. It costs no API call and no
context tokens -- these are decisions about what to keep, not text that is added.

What this deliberately does not do is arithmetic. It does not know that half of
eight is four, or that 1.5 hours and 90 minutes are the same span. A memory
should return the number that was said; deciding what follows from it is the
reading model's job, and a memory that quietly converted units would be inventing
evidence rather than storing it.
"""

from __future__ import annotations

import re

# A number as it is written in prose: digits, optional thousands separators,
# optional decimal part. Signs, currency symbols and percent marks are left out
# on purpose -- `$42` and `42` are the same quantity for the purpose of deciding
# whether two facts disagree, and carrying the symbol would make them differ.
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Small numbers are as often written out as they are typed, and "two weeks" and
# "2 weeks" are the same quantity. Only the range that actually turns up in
# conversation is mapped; beyond that the written form is rare enough that a
# longer table would be more surface than it is worth.
WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    # Ordinals, because "the third time" and "the 3rd time" are one fact.
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}

WORD_TOKEN = re.compile(r"[a-z]+")


def normalise(raw: str) -> str:
    """One canonical spelling per quantity, so equal numbers compare equal.

    `1,200` and `1200` are the same number; so are `3.50` and `3.5`, and `7.0`
    and `7`. Without this the comparison is a string match on how the number
    happened to be typed, which would report two identical amounts as a
    disagreement and defeat the point of the check.
    """
    text = raw.replace(",", "")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def quantity_cues(text: str) -> set[str]:
    """Every number in the text, normalised. Digits and small written numbers.

    Returned as a set because the question is only ever whether a quantity is
    present, never how many times it was said.
    """
    if not text:
        return set()
    lowered = text.lower()
    found = {normalise(m) for m in NUMBER.findall(lowered)}
    found.update(WORDS[w] for w in WORD_TOKEN.findall(lowered) if w in WORDS)
    return found


def disagrees(a: str, b: str) -> bool:
    """Do these two texts state quantities that are not the same?

    True only when both carry numbers and the numbers are not identical. A fact
    with no numbers in it disagrees with nothing -- silence about a quantity is
    not a contradiction of it, and treating it as one would stop a general
    statement from ever being recognised as covered by a specific one.
    """
    left, right = quantity_cues(a), quantity_cues(b)
    return bool(left) and bool(right) and left != right


def unverified_quantities(fact: str, source: str) -> list[str]:
    """Numbers the fact asserts that its own source turn never said.

    This is the audit that a summarising memory needs and does not otherwise
    have. Extraction rewrites a turn into a sentence with a language model, and a
    model that drops a digit or rounds a figure produces a fact that reads as
    confidently as a correct one. The source turn is stored verbatim, so the
    check is free and exact: every number in the summary should be a number that
    was actually said.

    It reports rather than repairs, and nothing calls it during retrieval. A
    quantity absent from the source is usually an extraction error but is
    sometimes a legitimate rewrite -- "a dozen" written back as `12` -- and
    silently discarding facts on a signal with a known false-positive mode is
    exactly what the supersession detector was measured doing when it killed
    correct facts at 17.9%.
    """
    said = quantity_cues(source)
    return sorted(quantity_cues(fact) - said, key=lambda n: (len(n), n))
