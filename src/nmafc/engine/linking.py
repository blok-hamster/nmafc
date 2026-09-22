"""Resolve extracted graph links onto entities that actually exist.

The extraction prompt tells the model that link targets must be real: "Use exact
`entity_name` strings ... Names matching no stored entity are discarded." The
first half is an instruction the model routinely disobeys, and the second half
was never implemented -- `related_entities` went from the model straight into
storage with nothing checking it.

Measured over the ten-store LoCoMo run, across all 11,902 links ever written:

    target alive today                       6,696   56.3%
    target names an entity never stored      4,615   38.8%
    target was stored and later pruned         591    5.0%

So the sparse graph is not decay's doing -- pruning accounts for 5% of it. Two
links in five were dead the moment they were written, pointing at plausible
names the model produced without ever creating the matching fact:

    caroline_adoption_agency_research   (written)
    caroline_adoption_research          (what actually exists)
    melanie_lake_sunrise_painting_2022  (written)
    melanie_lake_sunrise_painting       (what actually exists)

These are near misses, not inventions, which is what makes them recoverable:
78% of them match a stored entity at a token overlap of 0.5 or better. The model
is describing the right fact and spelling it differently, usually by adding or
dropping one qualifier.

Matching is on the underscore-separated parts of the name rather than on
characters. Entity names here are generated as `person_topic_detail`, so the
parts are meaningful units and their overlap says something; edit distance over
the raw string would score `caroline_adoption_research` against
`caroline_adoption_agency` almost identically to `caroline_adoption_dream`,
which are a match and a different fact respectively.
"""

from __future__ import annotations

from collections.abc import Iterable

# Token overlap (Jaccard over name parts) at or above which a written link is
# treated as naming a stored entity. 0.5 is where the population separates:
# below it the pairs are facts about the same person rather than the same fact
# ("caroline_adoption_dream" against "caroline_adoption_research" scores 0.5 on
# two shared parts of four, and is exactly the boundary case), and a threshold
# loose enough to catch the remaining 13% starts linking anything sharing a
# first name, which is the failure the prompt already warns about: linking
# everything to everything is the same as having no memory system.
DEFAULT_MIN_OVERLAP = 0.5


def _parts(name: str) -> frozenset[str]:
    return frozenset(p for p in name.lower().split("_") if p)


def resolve_link_targets(
    links: Iterable[str],
    known_entities: Iterable[str],
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> list[str]:
    """Map written link names onto stored entity names, dropping the unmatched.

    Returns the resolved targets in their original order with duplicates
    removed, each one guaranteed to name something in `known_entities`.

    An exact match always wins and is returned in the stored entity's own
    casing, because traversal compares lowercased while storage does not, and a
    link that differs from its target only in case is a link that resolves in
    one place and dangles in the other.

    Links that match nothing are dropped rather than kept. A link naming no
    stored entity cannot be traversed, so keeping it only inflates the link
    count -- which is precisely what made the graph look healthier on paper than
    it was.
    """
    canonical = {}
    indexed = []
    for entity in known_entities:
        key = entity.lower()
        if key in canonical:
            continue
        canonical[key] = entity
        indexed.append((key, _parts(entity)))

    if not canonical:
        return []

    resolved: list[str] = []
    seen: set[str] = set()

    for link in links:
        key = link.lower()
        target = canonical.get(key)

        if target is None:
            wanted = _parts(link)
            if not wanted:
                continue
            best_score = 0.0
            for candidate_key, candidate_parts in indexed:
                shared = len(wanted & candidate_parts)
                if not shared:
                    continue
                # Jaccard rather than coverage: scoring by how much of the
                # written name is covered would match every short name against
                # every long one that contains it, so "caroline_adoption" would
                # resolve to whichever adoption fact happened to be checked
                # first.
                score = shared / len(wanted | candidate_parts)
                if score > best_score:
                    best_score = score
                    target = canonical[candidate_key]
            if target is None or best_score < min_overlap:
                continue

        if target.lower() not in seen:
            seen.add(target.lower())
            resolved.append(target)

    return resolved
