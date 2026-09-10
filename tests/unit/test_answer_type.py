"""The demanded-type detector, against the questions that motivated it.

Most of these are real LoCoMo questions taken from the paired run, quoted with
their own spelling and spacing. A detector tested only on sentences its author
wrote will pass on the grammar its author had in mind, and the questions that
broke the first version -- "What are the breeds of Audrey's dogs?" -- read
nothing like the examples.
"""

from __future__ import annotations

import pytest

from nmafc.integration.answer_type import (TYPE_RULE, demanded_type, gate,
                                           type_tag)


class TestQuestionsThatNameAType:
    """The class the mechanism exists for: both arms answered a rung too low."""

    @pytest.mark.parametrize("question,expected", [
        ("In which state is the shelter from which James adopted the puppy?",
         "state"),
        ("Which country were Jolene and her mother visiting in 2010?",
         "country"),
        ("What Console does Nate own?", "Console"),
        ("Which city was Calvin visiting in August 2023?", "city"),
        ("Which US state do Audrey and Andrew potentially live in?",
         "US state"),
        ("What language does Tim know besides German?", "language"),
        ("What kind of game did John have a career-high in assists in?",
         "game"),
    ])
    def test_the_named_category_is_recovered(self, question, expected):
        assert demanded_type(question) == expected


class TestQuestionsThatNameNoType:
    """No directive at all, rather than a vague one. A guess about what the
    asker meant is a hallucination that happens to live in the prompt."""

    @pytest.mark.parametrize("question", [
        "Why didn't John want to go to Starbucks?",
        "What pets wouldn't cause any discomfort to Joanna?",
        "How do Evan and Sam use creative outlets to cope with life's challenges?",
        "Would John be open to moving to another country?",
        "What is something that Andrew could do to fit birdwatching in?",
        "",
    ])
    def test_nothing_is_demanded(self, question):
        assert demanded_type(question) is None
        assert type_tag(question) == ""

    def test_a_type_noun_outside_a_wh_phrase_demands_nothing(self):
        # "country" appears, but the question is a yes/no about willingness,
        # not a request to name one. Matching on the noun alone would fire here.
        assert demanded_type("Would John move to another country?") is None


class TestGrammarThatBrokeTheFirstVersion:
    """Auxiliaries and articles sit where a qualifier would, and the first
    version read them as part of the type."""

    def test_auxiliaries_are_not_part_of_the_type(self):
        assert demanded_type("What are the breeds of Audrey's dogs?") == "breeds"

    def test_a_real_qualifier_is_kept(self):
        # "US" changes what a correct answer is, so dropping it would loosen the
        # constraint the question actually set.
        assert demanded_type("Which US state did Jolene visit?") == "US state"

    def test_possessives_are_dropped_but_the_noun_survives(self):
        assert demanded_type("What is his job?") == "job"


class TestVerbClausesThatEndOnATypeNoun:
    """Found in the paired run, not by reading the regex.

    Both of these tagged, and the tags were nonsense: `Asked for: Maria make.`
    and `Asked for: happened John's job.` A question ending on a noun the list
    knows is not the same as a question asking for one of that noun, and the
    first version could not tell the difference because it deleted the words
    that would have told it.
    """

    def test_a_verb_that_happens_to_be_a_type_noun_does_not_fire(self):
        assert demanded_type(
            "What did Maria make for her home to remind her of England?"
        ) is None

    def test_a_noun_reached_through_a_preposition_does_not_fire(self):
        assert demanded_type("What happened to John's job?") is None

    def test_the_same_noun_still_fires_when_it_is_what_is_asked_for(self):
        # The fix must not silence the class it was built for.
        assert demanded_type("What job does John have?") == "job"

    def test_a_later_phrase_is_still_found_after_a_rejected_one(self):
        # Rejecting the first match must not abandon the question: the search
        # continues rather than returning on the first thing that looks close.
        assert demanded_type(
            "What happened to John's job, and which country did he move to?"
        ) == "country"


class TestPluralityIsPreserved:
    """A plural asks for every one of them, and the answering rules already
    treat a list differently. Normalising to the singular would erase that."""

    @pytest.mark.parametrize("question,expected", [
        ("What books has Tim read?", "books"),
        ("Which bands has Dave enjoyed listening to?", "bands"),
        # "video games" rather than "games": the qualifier is part of what was
        # asked for, and Nate also plays board games.
        ("What video games does Nate play?", "video games"),
        ("What book did she finish?", "book"),
    ])
    def test_number_survives(self, question, expected):
        assert demanded_type(question) == expected


class TestNoArticleIsEverChosen:
    """The detector quotes the question's own noun and adds nothing. An article
    picked from the first letter gets "an US state" wrong, and the directive
    reads as a category rather than as a sentence, so none is needed."""

    def test_a_named_type_comes_back_bare(self):
        assert demanded_type("What state?") == "state"
        assert demanded_type("What instrument does she play?") == "instrument"

    def test_only_the_implied_types_carry_wording_of_their_own(self):
        # There is no noun in "how old" to quote, so these are the one place a
        # phrase is supplied rather than taken from the question.
        assert demanded_type("How old is Jolene?") == "an age"
        assert demanded_type("How many tournaments has Nate won?") == "a number"


class TestImpliedTypes:
    """Wh-phrases that fix a type without naming a noun. `a number` was the
    most-failed demanded type in the measurement, and a noun list misses it."""

    @pytest.mark.parametrize("question,expected", [
        ("How many car shows has Dave attended?", "a number"),
        ("How much did the pendant cost?", "a number"),
        ("How often does Sam get health checkups?", "a frequency"),
        ("How long did it take for Jon to open his studio?",
         "a length of time"),
        ("How old is Jolene?", "an age"),
    ])
    def test_the_phrase_fixes_the_type(self, question, expected):
        assert demanded_type(question) == expected

    def test_how_many_reports_a_number_and_not_the_unit(self):
        # "How many months" could yield "months", and it is left as "a number"
        # on purpose. The failures in this class answered "Less than 4 months"
        # against a gold of "three months" and "once or twice" against "2" --
        # the unit was already right and the count was wrong, so naming the unit
        # would add tokens against an error nobody made.
        assert demanded_type(
            "How many months passed between Andrew adopting Toby and Buddy?"
        ) == "a number"


class TestTheTag:
    def test_it_names_the_type_it_found(self):
        assert "Asked for: state." in type_tag(
            "In which state is the shelter?")

    def test_the_guard_against_inventing_one_lives_in_the_shared_rule(self):
        # The tag names a category and nothing else, so on its own it reads as
        # "produce a country" whether or not the facts support one -- which is a
        # worse failure than the one being fixed. The guard is in TYPE_RULE
        # because it is identical on every question and paying for it per
        # question is what broke the budget.
        assert "rather than inventing one" in TYPE_RULE
        assert "If nothing in the facts fixes one" in TYPE_RULE

    def test_the_rule_explains_what_an_asked_for_line_is(self):
        # Ship one without the other and the tag is an unexplained fragment.
        assert "Asked for:" in TYPE_RULE

    def test_it_stays_short_enough_to_be_free(self):
        # Charged to the budget of every question that fires it. Multi-hop
        # measures 992 tokens of context against a 1,000 ceiling and fires on a
        # third of its questions, so anything above about 22 tokens per firing
        # breaks the ceiling on the mean by itself.
        longest = max(
            len(type_tag(q)) // 4
            for q in ("Which US state do Audrey and Andrew potentially live in?",
                      "How long did it take for Jon to open his studio?",
                      "What are the breeds of Audrey's dogs?")
        )
        assert longest <= 12, f"{longest} tokens is too much to add per question"

    def test_it_is_empty_when_nothing_is_demanded(self):
        assert type_tag("Why did he leave?") == ""


class TestTheRuleIsGatedOnTheTag:
    """A question that names no type must produce no change to the prompt.

    This is the property the whole cost argument rests on. If `gate` returns so
    much as a newline on an untyped question, the arm is no longer identical to
    the shipped one there, and the untyped slice goes back to being something
    that has to be paid for and measured instead of reasoned about.
    """

    UNTYPED = (
        "Why did he leave?",
        "What did Maria make for her home to remind her of a trip to England?",
        "Tell me about Deborah's garden.",
        "Did Nate enjoy the tournament?",
    )

    @pytest.mark.parametrize("question", UNTYPED)
    def test_an_untyped_question_changes_nothing_at_all(self, question):
        rule, tag = gate(question)
        assert rule == ""
        assert tag == ""

    def test_a_typed_question_gets_both_halves(self):
        rule, tag = gate("In which state is the shelter James used?")
        assert rule == TYPE_RULE
        assert tag == "Asked for: state."

    @pytest.mark.parametrize("question", UNTYPED + (
        "In which state is the shelter James used?",
        "How old is Calvin's daughter?",
    ))
    def test_the_two_halves_are_never_shipped_apart(self, question):
        # The tag is an unexplained fragment without the rule, and the rule is
        # dead text without the tag. Neither is ever emitted alone.
        rule, tag = gate(question)
        assert bool(rule) == bool(tag)

    def test_the_tag_is_the_one_the_detector_gives(self):
        # Gating decides whether to speak, never what to say. If these two ever
        # disagree, a question is being conditioned on a type nothing detected.
        for question in self.UNTYPED + ("What console did Nate buy?",):
            assert gate(question)[1] == type_tag(question)
