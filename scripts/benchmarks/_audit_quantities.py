"""How often does extraction change a number that was said?

The framework summarises each turn into facts with a language model, and the
summary is what gets stored, ranked and shown. A model that drops a digit,
rounds a figure or turns "two cats" into "a dog and a cat" produces a fact that
reads exactly as confidently as a correct one, and nothing downstream can tell
the difference -- the wrong number is now the memory.

That failure had been noticed by hand: 18 of 101 current answers are not in the
store, and one named cause was counts being dropped. Three remembered examples
is an impression, not a rate. This turns it into a rate, because the check is
free and exact: `turn_text` keeps every turn verbatim, so every number a fact
asserts can be compared against the turn it was extracted from.

Pure SQLite. No LanceDB, no embedding, no generation, no API key, and read-only
-- it opens the database in immutable mode, so it cannot disturb a store that a
benchmark is reading at the same time and cannot leave reinforcement writes
behind the way an unguarded `close()` does.

What a hit means, and what it does not:

  - A number in the fact that is not in its source turn is *suspicious*, not
    proven wrong. "a dozen eggs" summarised as `12 eggs` is a faithful rewrite
    and will be flagged; so will a fact drawing on context from an earlier turn.
  - The reverse case needs its own test, and gets one below. A turn saying "two
    cats" summarised as "has cats" has lost the count, which is the failure that
    started this, and it cannot be found by looking for numbers in the fact --
    there are none. It is found by looking for a count in the *source* that is
    attached to a noun the fact went on to discuss without it.

So read the samples before acting. This names candidates for a prompt revision;
it is not a filter, and wiring it into retrieval would repeat the mistake the
supersession detector made when it killed correct facts at 17.9%.

Usage:
    python -u scripts/benchmarks/_audit_quantities.py \
        --store C:/nmafc_ab/haystack/stores/neuromorphic_tuned__haystack
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nmafc.integration.quantities import (  # noqa: E402
    WORDS,
    quantity_cues,
    unverified_quantities,
)
from nmafc.integration.query_router import STOPWORDS, split_turn  # noqa: E402


# A count and the thing counted, as it is said out loud: "two cats", "3 kids",
# "five years". Written numbers included, because at these magnitudes people
# write them out more often than they type them.
#
# The lookbehind refuses a match that starts inside a longer number. Without it
# "the 2016 Finals" reads as a count of "16 finals", which was reported as a
# dropped count in the first run of this and is nothing of the kind.
WRITTEN = "|".join(sorted(WORDS, key=len, reverse=True))
COUNTED = re.compile(rf"(?<![\d.])\b(\d{{1,2}}|{WRITTEN})\s+([a-z]{{3,}})\b")

# Nouns that carry a number in nearly every sentence they appear in, where the
# number is a unit of time rather than a count of anything the memory is about.
# "married five years" losing the five is a real loss; "at 8 pm" is not the
# failure being looked for and would swamp everything that is.
#
# Month names are in here for a reason found by running it without them: the
# first version reported 925 dropped counts and every one of the twelve worst
# nouns was a month, because "14 August" in a session header is a date and the
# fact saying "mid-August 2023" has not dropped a count. That is measurement
# noise standing exactly where the signal would be.
UNITS = frozenset(
    "am pm year years month months week weeks day days hour hours minute "
    "minutes second seconds time times oclock morning evening night "
    "january february march april may june july august september october "
    "november december".split()
)

# The word after a number is not always the thing being counted. "the second one
# is always ready" gives "2 one", and "those moments stay with you" gives
# "1 with", neither of which is a count of anything. Excluding the function
# words and the number words themselves leaves nouns, which is what a dropped
# count has to be attached to.
NOT_A_NOUN = UNITS | STOPWORDS | frozenset(WORDS) | frozenset(
    "about after again all also always any because before being between both "
    "each even ever every following got just like made make more most much "
    "new now old only other over own same some still such than these those "
    "through too under until up very way well went".split()
)


def stem(word: str) -> str:
    """Crude singular. `cats` and `cat` are the same noun for this purpose."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def dropped_counts(fact: str, source: str) -> list[tuple[str, str]]:
    """Counts the source attached to a noun that the fact discusses without one.

    This is the "two cats became a cat" failure, stated precisely enough to
    test. Both halves of the condition matter: the fact has to be talking about
    the same noun, or every fact from a turn would be blamed for every number
    anywhere in it, and the fact has to carry no number of its own for that
    noun's neighbourhood, or a correct restatement counts as a loss.
    """
    said = quantity_cues(fact)
    words = {stem(w) for w in re.findall(r"[a-z]{3,}", fact.lower())}
    out: list[tuple[str, str]] = []
    for raw, noun in COUNTED.findall(source.lower()):
        if noun in NOT_A_NOUN or stem(noun) not in words:
            continue
        value = WORDS.get(raw, raw.lstrip("0") or "0")
        if value not in said:
            out.append((value, noun))
    return out


# "Caroline: Hey Melanie!" -- the speaker label the ingester writes on each line.
SPEAKER = re.compile(r"^\s*([A-Z][a-z]+):")


def speakers(body: list[str]) -> set[str]:
    """Who actually spoke in this turn."""
    return {m.group(1) for line in body if (m := SPEAKER.match(line))}


def misattributed(fact: str, body: list[str], cast: set[str]) -> list[str]:
    """People the fact talks about who are nowhere in the turn it came from.

    The third named extraction failure, and the same read-only method answers it.
    Every source line is labelled with its speaker, so the set of people involved
    in a turn is known exactly. A fact that names someone who neither spoke in
    that turn nor was mentioned in it is attributing something to a person who
    was not there.

    `cast` is every speaker name seen anywhere in the store, which is what makes
    this safe: without it any capitalised word in a fact would be treated as a
    person, and the check would fire on place names and brand names constantly.
    """
    present = speakers(body) | {
        name for name in cast
        if any(name.lower() in line.lower() for line in body)
    }
    named = {name for name in cast if re.search(rf"\b{name}\b", fact)}
    return sorted(named - present)


def rows(db: Path):
    """Every fact beside the verbatim turn it was extracted from.

    Opened immutable so this cannot write, cannot take a lock, and cannot be
    blamed for anything a concurrent benchmark sees.
    """
    conn = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
    try:
        return conn.execute("""
            SELECT e.turn, e.entity_name, e.fact_content, t.text
            FROM memory_event_log e
            JOIN turn_text t
              ON t.agent_id = e.agent_id
             AND t.conversation_id = e.conversation_id
             AND t.turn = e.turn
            WHERE e.is_active = 1
        """).fetchall()
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True,
                    help="Store directory containing cold.db")
    ap.add_argument("--show", type=int, default=25,
                    help="How many flagged facts to print in full")
    args = ap.parse_args()

    db = Path(args.store) / "cold.db"
    if not db.exists():
        sys.exit(f"no cold.db under {args.store}")

    facts = rows(db)
    if not facts:
        sys.exit("no facts with a stored source turn -- this store predates "
                 "turn_text, so there is nothing to check against")

    # Every name that ever appears as a speaker. Built in one pass before the
    # checks, because "is this word a person" is only answerable against the cast
    # of the whole store, not against one turn.
    cast: set[str] = set()
    for _, _, _, source in facts:
        cast |= speakers(split_turn(source)[1])
    print(f"cast of {len(cast)} speakers: "
          f"{', '.join(sorted(cast)[:14])}{' ...' if len(cast) > 14 else ''}\n")

    with_numbers = 0
    flagged: list[tuple[int, str, str, list[str], str]] = []
    lost: list[tuple[int, str, str, list[tuple[str, str]], str]] = []
    wrong: list[tuple[int, str, str, list[str], str]] = []
    for turn, entity, fact, source in facts:
        # Only what somebody said. The bracketed session header is metadata the
        # ingester wrote, so a number in it was never uttered and cannot have
        # been dropped from a summary of the utterances.
        _, body = split_turn(source)
        spoken = "\n".join(body)
        strangers = misattributed(fact, body, cast)
        if strangers:
            wrong.append((turn, entity, fact, strangers, source))
        gone = dropped_counts(fact, spoken)
        if gone:
            lost.append((turn, entity, fact, gone, source))
        if not quantity_cues(fact):
            continue
        with_numbers += 1
        missing = unverified_quantities(fact, source)
        if missing:
            flagged.append((turn, entity, fact, missing, source))

    print(f"facts with a stored source turn: {len(facts)}")
    print(f"  of those, carrying a number:   {with_numbers} "
          f"({100 * with_numbers / len(facts):.1f}%)")
    if not with_numbers:
        return
    print(f"  numbers not in the source:     {len(flagged)} "
          f"({100 * len(flagged) / with_numbers:.1f}% of facts with a number, "
          f"{100 * len(flagged) / len(facts):.1f}% of all facts)")

    # Which numbers go unverified says more than how many. A tail of years and
    # large figures is mostly the extractor resolving a date from elsewhere in
    # the conversation; a pile of small counts is the failure this was written
    # for, because a count is nearly always said outright in the turn it
    # describes.
    counts = Counter(n for _, _, _, missing, _ in flagged for n in missing)
    small = sum(c for n, c in counts.items()
                if n.isdigit() and len(n) <= 2 and int(n) <= 20)
    print(f"  of the unverified numbers, {small} are counts of 20 or under, "
          f"which is the shape a dropped or altered count takes")
    print("\n  most common unverified numbers: "
          + ", ".join(f"{n}x{c}" for n, c in counts.most_common(12)))

    print(f"\n  first {min(args.show, len(flagged))} flagged, "
          f"read them before acting on the rate:\n")
    for turn, entity, fact, missing, source in flagged[:args.show]:
        said = " / ".join(line.strip() for line in source.split("\n")
                          if line.strip())[:180]
        print(f"  turn {turn}  [{entity}]  missing {missing}")
        print(f"    fact:   {fact[:150]}")
        print(f"    source: {said}\n")

    # The other direction, and the one the extraction complaint was actually
    # about: a count that was said and did not survive into the summary.
    print("=" * 72)
    print(f"\ncounts said about a noun and dropped from a fact about that same "
          f"noun: {len(lost)} ({100 * len(lost) / len(facts):.1f}% of all facts)")
    if lost:
        nouns = Counter(noun for _, _, _, gone, _ in lost for _, noun in gone)
        print("  most affected nouns: "
              + ", ".join(f"{n}x{c}" for n, c in nouns.most_common(12)))

        # Grouped by noun so a prompt revision can be aimed. One noun accounting
        # for a large share is a specific instruction worth writing; a flat
        # spread means the extractor drops counts generally and the instruction
        # has to be general too.
        by_noun: dict[str, list] = defaultdict(list)
        for turn, entity, fact, gone, source in lost:
            for value, noun in gone:
                by_noun[noun].append((turn, entity, fact, value, source))
        print(f"\n  worst {min(6, len(by_noun))} nouns, with an example each:\n")
        for noun, hits in sorted(by_noun.items(), key=lambda kv: -len(kv[1]))[:6]:
            turn, entity, fact, value, source = hits[0]
            said = " / ".join(line.strip() for line in source.split("\n")
                              if line.strip())[:180]
            print(f"  {noun}: {len(hits)} facts. e.g. turn {turn} [{entity}] "
                  f"lost \"{value} {noun}\"")
            print(f"    fact:   {fact[:150]}")
            print(f"    source: {said}\n")

    # The third named extraction failure: a fact about somebody who was not in
    # the turn it came from.
    print("=" * 72)
    print(f"\nfacts naming a person absent from their own source turn: "
          f"{len(wrong)} ({100 * len(wrong) / len(facts):.1f}% of all facts)")
    if not wrong:
        return
    who = Counter(name for _, _, _, names, _ in wrong for name in names)
    print("  most often attributed to: "
          + ", ".join(f"{n}x{c}" for n, c in who.most_common(12)))
    print(f"\n  first {min(8, len(wrong))}, read before believing the rate:\n")
    for turn, entity, fact, names, source in wrong[:8]:
        said = " / ".join(line.strip() for line in source.split("\n")
                          if line.strip())[:180]
        print(f"  turn {turn}  [{entity}]  absent: {names}")
        print(f"    fact:   {fact[:150]}")
        print(f"    source: {said}\n")


if __name__ == "__main__":
    main()
