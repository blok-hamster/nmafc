"""Pattern separation over the rendered fact list.

`separate_facts` is the dentate-gyrus half of the context: it removes the
restatements of a fact so the rendered slots hold facts that differ. These tests
pin the three decisions that make it safe to switch on -- which of a redundant
pair survives, that ordering is otherwise untouched, and that it can never grow
the list it was given.
"""

from __future__ import annotations

from nmafc.integration.query_router import separate_facts
from nmafc.schemas.memory import MemoryRecord, MemoryType


def fact(text: str) -> MemoryRecord:
    return MemoryRecord(
        entity_name="caroline",
        fact_content=text,
        memory_type=MemoryType.ACTIVE_CONTEXT,
    )


def contents(records: list[MemoryRecord]) -> list[str]:
    return [r.fact_content for r in records]


class TestSeparateFacts:
    def test_keeps_facts_that_differ(self):
        records = [fact("Caroline plays saxophone"),
                   fact("Melanie adopted two cats")]
        assert contents(separate_facts(records, 0.8)) == [
            "Caroline plays saxophone", "Melanie adopted two cats"]

    def test_drops_a_restatement_of_an_earlier_fact(self):
        records = [fact("Caroline plays saxophone in a jazz quartet"),
                   fact("Caroline plays saxophone")]
        assert contents(separate_facts(records, 0.8)) == [
            "Caroline plays saxophone in a jazz quartet"]

    def test_the_more_specific_fact_survives_even_when_ranked_lower(self):
        """Specificity beats rank, and the survivor takes the better position.

        Keeping the vaguer fact because it happened to rank higher would throw
        away information in order to preserve an ordering. The kept fact sits
        where the dropped one sat, so nothing is promoted past a fact it lost to.
        """
        records = [fact("Caroline plays saxophone"),
                   fact("Melanie adopted two cats"),
                   fact("Caroline plays saxophone in a jazz quartet")]
        assert contents(separate_facts(records, 0.8)) == [
            "Caroline plays saxophone in a jazz quartet",
            "Melanie adopted two cats"]

    def test_threshold_of_one_still_merges_a_total_restatement(self):
        """At 1.0 only a fact wholly restated inside another is removed.

        "Caroline plays saxophone" is entirely contained in the longer fact, so
        containment is exactly 1.0 and the pair merges even at the strictest
        setting. A pair that merely shares a subject does not.
        """
        total = [fact("Caroline plays saxophone in a jazz quartet"),
                 fact("Caroline plays saxophone")]
        assert len(separate_facts(total, 1.0)) == 1

        partial = [fact("Caroline plays saxophone on Sundays"),
                   fact("Caroline plays piano")]
        assert len(separate_facts(partial, 1.0)) == 2

    def test_stopword_only_facts_are_kept_and_never_absorb_anything(self):
        """A fact with no content words has an empty cue.

        An empty set is contained in every set, so comparing against it would
        delete the entire list. Both directions are skipped.
        """
        records = [fact("It was."), fact("Caroline plays saxophone"),
                   fact("And so on.")]
        assert len(separate_facts(records, 0.8)) == 3

    def test_never_returns_more_than_it_was_given(self):
        records = [fact(f"Caroline visited city number {i}") for i in range(20)]
        assert len(separate_facts(records, 0.8)) <= len(records)

    def test_empty_input(self):
        assert separate_facts([], 0.8) == []

    def test_partial_overlap_below_threshold_is_two_facts(self):
        """"Caroline plays saxophone" and "Caroline visited Chicago" share only
        the name, which is 1 of 2 content words in each. Containment is 0.5, so a
        0.8 threshold has to keep both."""
        records = [fact("Caroline saxophone"), fact("Caroline Chicago")]
        assert len(separate_facts(records, 0.8)) == 2
