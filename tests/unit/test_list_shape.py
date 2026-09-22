"""The list-shape detector, against the questions that motivated it.

Real LoCoMo single-hop questions taken from the paired run of 10 September,
quoted with their own spelling and spacing. The detector decides how much
context a question is worth, so a false fire costs tokens on a question that
never needed them and a miss costs nothing but the status quo. The tests are
weighted that way: precision is asserted harder than recall.

Measured on the run these came from: fires on 129 of 281 single-hop questions,
94.6% of which have list-shaped gold, and that half scores 35.7% against the
quiet half's 63.8%.
"""

from __future__ import annotations

import pytest

from nmafc.integration.list_shape import wants_list


class TestQuestionsThatAskForAList:
    """The class the gate exists for. Every gold here is four or more items."""

    @pytest.mark.parametrize("question", [
        "What activities does Melanie partake in?",
        "What books has Melanie read?",
        "What LGBTQ+ events has Caroline participated in?",
        "What activities has Melanie done with her family?",
        "What symbols are important to Caroline?",
        "What musical artists/bands has Melanie seen?",
        "What events for veterans has John participated in?",
        "What exercises has John done?",
        "What causes has John done events for?",
        "What desserts has Maria made?",
        "What are Joanna's hobbies?",
        "What kind of writings does Joanna do?",
        "What book recommendations has Joanna given to Nate?",
        "Which events has Jon participated in to promote his business venture?",
    ])
    def test_the_gate_fires(self, question):
        assert wants_list(question) is True


class TestQuestionsThatAskForOneThing:
    """Widening these buys nothing and costs the whole gain elsewhere.

    A clause in the prompt is free; extra facts and extra hydrated turns are
    not. Screened, the width that lifts list questions costs 238 tokens a
    question, so the gate paying out on a single-answer question is the failure
    mode that matters.
    """

    @pytest.mark.parametrize("question", [
        "What is his job?",
        "Where did she move from?",
        "Why did he take up the guitar?",
        "When did she go camping?",
        "How old is Jolene?",
        "Who supports Caroline?",
        "Did Nate enjoy the tournament?",
        "",
    ])
    def test_the_gate_stays_quiet(self, question):
        assert wants_list(question) is False


class TestWordsEndingInSThatAreNotPlurals:
    """The whole detector is "is the head of the wh-phrase a plural noun", and
    English is full of words that end in s and are not.

    Without the exclusions the gate fires on most of the category, which is the
    same as not having a gate: the widening is applied everywhere, the token
    mean goes well past 1,000, and the reason for gating in the first place is
    gone.
    """

    @pytest.mark.parametrize("question", [
        "What is Joanna allergic to?",
        "What has Melanie done this year?",
        "What does Jon's dance studio offer?",
        "What was grandma's gift to Melanie?",
    ])
    def test_an_auxiliary_verb_is_not_a_plural_noun(self, question):
        assert wants_list(question) is False

    def test_a_double_s_ending_is_not_a_plural(self):
        # "business", "illness", "progress". Each one that slipped through
        # would widen a single-answer question at full token cost.
        assert wants_list("What business did Gina start?") is False
        assert wants_list("What progress has Jon made?") is False

    def test_a_possessive_is_a_modifier_and_not_the_head(self):
        # "Joanna's" ends in s once the apostrophe is ignored, and the thing
        # asked for is singular.
        assert wants_list("What is Joanna's hobby?") is False


class TestPhrasingsWithNoPluralNounToKeyOn:
    """English asks for everything in ways that never reach a plural head."""

    @pytest.mark.parametrize("question", [
        "In what ways is Caroline participating in the LGBTQ community?",
        "What are some changes Caroline has faced during her transition?",
        "List all the places Maria has volunteered.",
    ])
    def test_an_explicit_enumeration_fires(self, question):
        assert wants_list(question) is True


class TestTheKnownMisses:
    """Questions whose gold is a list and which the gate does not catch.

    Recorded rather than fixed. Each one would need a rule keyed on something
    other than the question's grammar -- "Where has Maria made friends?" is
    singular in every respect and has a three-item gold -- and a rule loose
    enough to catch it fires on the single-answer questions above. The cost of
    a miss is a question that stays as hard as it is today, which is the
    cheaper of the two errors.
    """

    @pytest.mark.parametrize("question", [
        "Where has Melanie camped?",
        "What does Melanie do to destress?",
        "How did Gina promote her clothes store?",
        "Why did Gina decide to start her own clothing store?",
    ])
    def test_these_are_missed_on_purpose(self, question):
        assert wants_list(question) is False


class TestTheGateCostsNothingToAsk:
    """No embedding, no model call, no store access. The same property that
    makes `answer_type.gate` affordable, and the reason both are regexes."""

    def test_it_is_a_pure_function_of_the_string(self):
        question = "What activities does Melanie partake in?"
        assert wants_list(question) == wants_list(question)

    def test_it_survives_the_empty_string_and_junk(self):
        for question in ("", "   ", "???", "s"):
            assert wants_list(question) is False
