"""Numbers as content: normalisation, disagreement, and the two call sites.

The point being defended throughout is narrow and worth stating once: a memory
may compress what was said, but it may not change a number or lose one. Every
test here is a case where the word-level machinery would have done exactly that.
"""

from __future__ import annotations

from nmafc.integration.quantities import (
    disagrees,
    normalise,
    quantity_cues,
    unverified_quantities,
)
from nmafc.integration.query_router import best_lines, separate_facts
from nmafc.schemas.memory import MemoryRecord, MemoryType


def fact(content: str, turn: int = 1) -> MemoryRecord:
    return MemoryRecord(entity_name="e", fact_content=content,
                        memory_type=MemoryType.ACTIVE_CONTEXT,
                        created_at_turn=turn)


class TestNormalise:
    def test_thousands_separators_are_not_part_of_the_number(self):
        assert normalise("1,200") == normalise("1200") == "1200"

    def test_trailing_zeros_after_a_point_do_not_make_a_new_number(self):
        assert normalise("3.50") == normalise("3.5") == "3.5"
        assert normalise("7.0") == normalise("7") == "7"

    def test_a_number_that_is_all_zeros_survives_the_stripping(self):
        assert normalise("0") == "0"
        assert normalise("0.0") == "0"


class TestQuantityCues:
    def test_small_numbers_are_found_where_cue_words_loses_them(self):
        from nmafc.integration.query_router import cue_words
        assert cue_words("he paid 42 dollars") == {"paid", "dollars"}
        assert quantity_cues("he paid 42 dollars") == {"42"}

    def test_currency_and_percent_marks_are_not_part_of_the_quantity(self):
        assert quantity_cues("$42 and 42% and 42") == {"42"}

    def test_written_small_numbers_count(self):
        assert quantity_cues("two weeks") == quantity_cues("2 weeks") == {"2"}

    def test_ordinals_written_either_way_agree(self):
        assert quantity_cues("the third time") == quantity_cues("the 3rd time")

    def test_a_date_contributes_its_numbers(self):
        assert quantity_cues("11 August 2023") == {"11", "2023"}

    def test_text_with_no_numbers_gives_nothing(self):
        assert quantity_cues("she liked the film") == set()
        assert quantity_cues("") == set()


class TestDisagrees:
    def test_different_amounts_disagree(self):
        assert disagrees("He paid 42 dollars", "He paid 47 dollars")

    def test_the_same_amount_written_differently_does_not(self):
        assert not disagrees("It cost $1,200", "It cost 1200 dollars")

    def test_silence_about_a_number_is_not_a_contradiction(self):
        """Otherwise a general statement could never be recognised as covered by
        a specific one, and pattern separation would stop separating anything."""
        assert not disagrees("He plays saxophone", "He plays 2 instruments")
        assert not disagrees("He plays 2 instruments", "He plays saxophone")

    def test_neither_carrying_a_number_does_not_disagree(self):
        assert not disagrees("He plays saxophone", "He plays saxophone in a band")


class TestSeparationKeepsDifferentNumbers:
    def test_two_amounts_are_not_one_fact(self):
        """The bug this exists for. Content words are identical, containment is
        1.0, and without the veto the second amount is deleted."""
        records = [fact("He paid 42 dollars"), fact("He paid 47 dollars")]
        assert len(separate_facts(records, 0.8)) == 2

    def test_a_restatement_carrying_the_same_number_is_still_merged(self):
        records = [fact("Rent is 900"), fact("Rent is 900 a month now")]
        kept = separate_facts(records, 0.8)
        assert len(kept) == 1
        assert kept[0].fact_content == "Rent is 900 a month now"

    def test_a_quantity_free_restatement_is_still_merged(self):
        records = [fact("Caroline plays saxophone"),
                   fact("Caroline plays saxophone in a jazz quartet")]
        assert len(separate_facts(records, 0.8)) == 1

    def test_a_third_fact_may_still_merge_with_an_earlier_one(self):
        """A vetoed pair must not stop the record matching something else. The
        veto skips a candidate; it does not end the search."""
        records = [fact("He paid 42 dollars"),
                   fact("She runs daily"),
                   fact("She runs daily before work")]
        kept = separate_facts(records, 0.8)
        assert [r.fact_content for r in kept] == [
            "He paid 42 dollars", "She runs daily before work"]

    def test_separation_never_loses_a_quantity(self):
        """The guarantee, stated as a property rather than a case."""
        records = [fact("Rent was 900"), fact("Rent was 950"),
                   fact("Rent was 1,000"), fact("Rent went up")]
        kept = separate_facts(records, 0.5)
        before = set().union(*(quantity_cues(r.fact_content) for r in records))
        after = set().union(*(quantity_cues(r.fact_content) for r in kept))
        assert before == after


class TestBestLinesRanksByQuantity:
    BODY = [
        "Alice: rent has been on my mind a lot lately",
        "Bob: rent is such a difficult subject for everyone",
        "Alice: the rent went up to 950 last month",
    ]

    def test_the_line_with_the_number_wins(self):
        picked = best_lines(self.BODY, {"rent"}, 1, {"950"})
        assert picked == ["Alice: the rent went up to 950 last month"]

    def test_without_quantities_the_word_ordering_is_unchanged(self):
        assert (best_lines(self.BODY, {"rent"}, 1)
                == best_lines(self.BODY, {"rent"}, 1, set()))

    def test_a_quantity_outranks_more_shared_words(self):
        body = ["Bob: rent difficult subject money lately",
                "Alice: it was 950"]
        cue = {"rent", "difficult", "subject", "money", "lately"}
        assert best_lines(body, cue, 1, {"950"}) == ["Alice: it was 950"]

    def test_a_short_turn_is_returned_whole_either_way(self):
        assert best_lines(self.BODY, {"rent"}, 5, {"950"}) == self.BODY


class TestUnverifiedQuantities:
    def test_a_number_the_source_never_said_is_reported(self):
        assert unverified_quantities("He paid 47 dollars",
                                     "Alice: I paid 42 dollars for it") == ["47"]

    def test_a_faithful_summary_reports_nothing(self):
        assert unverified_quantities("He paid 42 dollars",
                                     "Alice: I paid 42 dollars for it") == []

    def test_a_summary_with_no_numbers_reports_nothing(self):
        assert unverified_quantities("He paid for it", "Alice: I paid 42") == []

    def test_formatting_differences_are_not_reported_as_errors(self):
        assert unverified_quantities("Rent is $1,200", "Alice: rent is 1200") == []

    def test_a_written_number_rewritten_as_a_digit_is_not_flagged(self):
        assert unverified_quantities("It lasted 2 weeks",
                                     "Alice: it went on two weeks") == []
