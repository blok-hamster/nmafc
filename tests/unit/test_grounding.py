"""BM25 of a question against the turns behind its candidate facts.

The worked cases are the real open-domain losses. A scorer tested only on
invented text passes on the separation its author built into the fixture, and
the point of this one is that the deciding word is a single adjective sitting in
a turn that otherwise reads exactly like its competitor.
"""

from __future__ import annotations

import math

from nmafc.engine.reranking import rerank
from nmafc.integration.grounding import source_scores, terms
from nmafc.integration.query_router import QueryRouter
from nmafc.schemas.memory import (DecayConfig, MemoryRecord, MemoryType,
                                  SearchCandidate)


class TestTerms:
    def test_stopwords_go(self):
        assert "the" not in terms("the novel")
        assert "novel" in terms("the novel")

    def test_short_numbers_survive_the_length_filter(self):
        # `cue_words` drops every token of three characters or fewer, which is
        # how every quantity under 100 became invisible to the whole system.
        got = terms("two weeks before 11 August")
        assert "11" in got
        assert "2" in got, "the written form should normalise to a digit"

    def test_repeats_are_kept(self):
        # BM25 is defined on term frequency; a set would throw it away.
        assert terms("gripping gripping").count("gripping") == 2

    def test_empty_text_is_no_terms(self):
        assert terms("") == []


class TestTheAdjectiveThatDecides:
    """`What novel is Evan reading that he finds gripping?` -- both turns say
    Evan is reading a novel, and one word tells them apart."""

    QUERY = "What novel is Evan reading that he finds gripping?"
    TEXTS = {
        4: "Evan: I started The Last Devil to Die last night, a nice easy read.",
        9: "Evan: I am reading The Great Gatsby and it is absolutely gripping.",
    }

    def test_the_turn_with_the_qualifier_ranks_first(self):
        scores = source_scores(self.QUERY, self.TEXTS)
        assert scores[9] > scores[4]

    def test_removing_the_qualifier_removes_the_separation(self):
        # Not a tautology worth skipping: it shows the ordering comes from the
        # deciding word and not from some incidental difference in the turns.
        neutral = "What novel is Evan reading?"
        scores = source_scores(neutral, self.TEXTS)
        assert scores[9] == scores[4] or abs(scores[9] - scores[4]) < 1.0


class TestRarityIsMeasuredInThePool:
    def test_a_word_in_every_turn_decides_nothing(self):
        # Zero, not merely equal. Equal-but-positive orders the pool just as
        # well, but it would mean the all-zero return -- the caller's only way
        # of telling that this signal has no opinion -- could never be reached
        # by a question that shared any word at all with its candidates.
        texts = {1: "Evan is reading.", 2: "Evan is reading.",
                 3: "Evan is reading."}
        assert set(source_scores("What is Evan reading?", texts).values()) \
            == {0.0}

    def test_a_word_in_one_turn_of_many_decides_everything(self):
        texts = {n: "Tim suggested a city for the trip." for n in range(1, 10)}
        texts[9] = "Tim suggested Edinburgh for the team trip next month."
        scores = source_scores(
            "What city did Tim suggest for the team trip next month?", texts)
        assert scores[9] == max(scores.values())
        assert scores[9] > scores[1]

    def test_no_pool_word_matches_and_everything_scores_zero(self):
        texts = {1: "Deborah walked the dog.", 2: "Deborah cooked dinner."}
        assert set(source_scores("What console does Nate own?", texts).values()) \
            == {0.0}


class TestLengthNormalisation:
    def test_a_long_rambling_turn_does_not_win_on_length_alone(self):
        # Without `b`, the longest turn wins on the chance of containing a rare
        # word rather than on being about the question.
        short = "Maria: the puppy is called Coco."
        padding = " ".join(f"word{i}" for i in range(300))
        texts = {1: short, 2: f"Maria: we talked about lots of things. {padding}"}
        scores = source_scores("What is the name of Maria's puppy?", texts)
        assert scores[1] > scores[2]


class TestDegenerateInputs:
    """All of these are ordinary, not exceptional: most questions are not
    disambiguation questions, and the caller reads all-zero as "no opinion"."""

    def test_an_empty_pool_gives_an_empty_result(self):
        assert source_scores("anything", {}) == {}

    def test_an_empty_question_scores_every_turn_zero(self):
        texts = {1: "some text", 2: "other text"}
        assert source_scores("", texts) == {1: 0.0, 2: 0.0}

    def test_a_turn_of_pure_stopwords_does_not_divide_by_zero(self):
        texts = {1: "the and of to", 2: "Evan read The Great Gatsby"}
        scores = source_scores("What did Evan read?", texts)
        assert scores[1] == 0.0
        assert scores[2] > 0.0

    def test_every_turn_empty_is_survivable(self):
        assert source_scores("q", {1: "", 2: ""}) == {1: 0.0, 2: 0.0}

    def test_a_single_turn_pool_scores_zero_because_nothing_is_rare(self):
        # One turn cannot be rarer than itself. Returning a large score here
        # would let the signal outvote the others whenever only one candidate
        # carried a turn number.
        scores = source_scores("What novel is gripping?", {1: "gripping novel"})
        assert scores[1] == 0.0


class TestScoresAreNeverNegative:
    def test_a_common_word_cannot_push_a_turn_down(self):
        # Unfloored, BM25's IDF goes negative for a term in more than half the
        # pool, which would punish a turn for being on topic.
        texts = {1: "Evan reading novel", 2: "Evan reading novel",
                 3: "Evan reading novel gripping"}
        scores = source_scores("Evan reading novel gripping", texts)
        assert all(s >= 0.0 for s in scores.values())
        assert scores[3] > scores[1]


class TestTheMathsIsBm25:
    def test_idf_matches_the_probabilistic_form(self):
        # One rare term, one turn of one word, so every normalisation factor is
        # 1 and the score reduces to idf alone. Pinned so a future edit to the
        # constants is a deliberate change and not a silent drift.
        texts = {1: "gripping", 2: "other", 3: "other", 4: "other"}
        expected = math.log(1.0 + (4 - 1 + 0.5) / (1 + 0.5))
        assert source_scores("gripping", texts)[1] == expected


# --------------------------------------------------------------------------
# The wiring. The scorer above can be right while the router hands it the wrong
# turns, ignores what it says, or admits raw turns into the candidate pool --
# and that last one would quietly spend the retrieval budget the module's own
# docstring promises it does not touch.
#
# The shape being tested changed once, and the reason is worth keeping. This
# first went in as a sixth list fused by RRF. Screened free against the 69
# open-domain losses that have an answering record, it reached three more of
# them and reshuffled the top facts of 54 of 60 questions already answered
# correctly -- because RRF weights every list alike, so "the question's words
# are in the turn this fact came from" counted for as much as topping vector
# search. The diagnosis was near-ties, so the fix is a tie-break: an additive
# boost, sized like the gap between adjacent ranks rather than like a whole
# list.
# --------------------------------------------------------------------------


class FakeCold:
    """A cold store that answers `text_for_turns` and records what was asked."""

    def __init__(self, texts: dict[int, str],
                 whole: dict[int, str] | None = None) -> None:
        self.texts = texts
        self.asked: list[list[int]] = []
        # The whole conversation, for `all_turn_text`. Defaulting to nothing
        # keeps every test written before the scan existed asserting what it
        # asserted: a scan over an empty conversation returns nothing, so the
        # feature is inert unless a test hands it a conversation to scan.
        self.whole = whole or {}
        self.scans = 0

    def text_for_turns(self, turns):
        self.asked.append(list(turns))
        return {t: self.texts[t] for t in turns if t in self.texts}

    def all_turn_text(self):
        self.scans += 1
        return dict(self.whole)


def candidate(entity: str, turn: int, source: str = "hot_vector",
              rank: int = 0) -> SearchCandidate:
    return SearchCandidate(
        record=MemoryRecord(entity_name=entity, fact_content=f"{entity} fact",
                            memory_type=MemoryType.ACTIVE_CONTEXT,
                            created_at_turn=turn),
        score=1.0 - rank / 100, source=source, rank_in_source=rank,
    )


def router(texts: dict[int, str], weight: float = 0.001) -> QueryRouter:
    return QueryRouter(hot=None, cold=FakeCold(texts), embedder=None,
                       config=DecayConfig(source_grounding=weight))


READING = {
    4: "Evan: I am reading The Last Devil to Die, a nice easy read.",
    9: "Evan: I am reading The Great Gatsby and it is absolutely gripping.",
    12: "Deborah: the garden looks lovely at this time of year.",
}
QUESTION = "What novel is Evan reading that he finds gripping?"


def pool() -> list[SearchCandidate]:
    """Two facts a question cannot tell apart, and one that is off-topic."""
    return [candidate("devil", 4), candidate("gatsby", 9),
            candidate("garden", 12)]


class TestTheShares:
    def test_the_best_matching_turn_scores_one(self):
        assert router(READING)._grounded(QUESTION, pool())["gatsby"] == 1.0

    def test_a_worse_match_scores_between_zero_and_one(self):
        share = router(READING)._grounded(QUESTION, pool())["devil"]
        assert 0.0 < share < 1.0

    def test_a_turn_matching_nothing_is_absent_rather_than_zero(self):
        # Absent and zero behave the same in the fusion, but absent is the
        # honest encoding: the signal has no opinion about that entity.
        assert "garden" not in router(READING)._grounded(QUESTION, pool())

    def test_an_entity_spanning_turns_keeps_its_best(self):
        # One entity can hold facts from several turns. The question is whether
        # any turn behind it matches, so the best one stands for the entity.
        r = router(READING)
        cands = [candidate("evan", 4), candidate("evan", 9),
                 candidate("garden", 12)]
        assert r._grounded(QUESTION, cands)["evan"] == 1.0

    def test_it_is_off_unless_asked_for(self):
        assert DecayConfig().source_grounding == 0.0


class TestOnlyTurnsAlreadyReachedAreRead:
    def test_no_turn_is_read_that_no_candidate_pointed_at(self):
        # The promise that this costs no retrieval budget rests entirely on
        # this: turns are reached through facts that already won a slot, never
        # searched for.
        cold_texts = dict(READING) | {77: "Evan: a gripping thriller, actually."}
        r = router(cold_texts)
        r._grounded(QUESTION, [candidate("devil", 4), candidate("gatsby", 9)])
        assert r._cold.asked == [[4, 9]]

    def test_no_new_entity_is_ever_introduced(self):
        r = router(READING)
        cands = pool()
        known = {c.record.entity_name for c in cands}
        assert set(r._grounded(QUESTION, cands)) <= known


class TestWhenItSaysNothing:
    def test_one_turn_gets_no_boost(self):
        # Otherwise the only fact carrying a turn number takes a full boost for
        # winning a contest it was the only entrant in.
        r = router(READING)
        assert r._grounded(QUESTION, [candidate("gatsby", 9)]) == {}

    def test_facts_with_no_source_turn_are_skipped_not_crashed_on(self):
        r = router(READING)
        assert "floating" not in r._grounded(
            QUESTION, pool() + [candidate("floating", 0)])

    def test_a_cold_store_without_turn_text_is_survivable(self):
        # Hydration has not always existed, and a store built before it has no
        # turn text to read. Retrieval should carry on unchanged, not fail.
        r = QueryRouter(hot=None, cold=object(), embedder=None,
                        config=DecayConfig(source_grounding=0.001))
        assert r._grounded(QUESTION, pool()) == {}

    def test_a_question_matching_nothing_hands_out_no_boost(self):
        r = router(READING)
        assert r._grounded("What console does Nate own?", pool()) == {}


class TestTheBoostIsSizedLikeATieBreak:
    """The number that decides whether this is a tie-break or a takeover.

    Two facts adjacent in one fused list differ by about 0.0003, and belonging
    to an extra list is worth about 1/61. A boost between the two moves a fact
    past its near-copy and leaves a fact that won on merit where it was.
    """

    def test_a_near_tie_is_broken_by_the_evidence(self):
        # Both facts are found by vector search alone, adjacent in rank, which
        # is exactly the "Evan is reading X / Evan is reading Y" case.
        cands = [candidate("devil", 4, rank=0), candidate("gatsby", 9, rank=1)]
        plain = [r.entity_name for r in rerank(list(cands), DecayConfig(), 0)]
        assert plain == ["devil", "gatsby"]

        boosted = [r.entity_name for r in rerank(
            list(cands), DecayConfig(source_grounding=0.001), 0,
            grounding={"gatsby": 1.0, "devil": 0.4})]
        assert boosted == ["gatsby", "devil"]

    def test_a_fact_that_won_on_merit_is_not_displaced(self):
        # `winner` is in three lists, `other` in one. That is a real margin,
        # not a tie, and a tie-break must not overturn it however well the
        # loser's turn happens to read.
        cands = [candidate("winner", 4, source=s, rank=0)
                 for s in ("hot_vector", "cold_semantic", "bfs_hot")]
        cands.append(candidate("other", 9, source="hot_vector", rank=1))
        boosted = [r.entity_name for r in rerank(
            list(cands), DecayConfig(source_grounding=0.001), 0,
            grounding={"other": 1.0})]
        assert boosted[0] == "winner"

    def test_at_a_full_list_weight_it_does_overturn_that_margin(self):
        # Not a bug, a boundary: it shows the weight is what limits the signal,
        # so any small value is meaningfully different from switching it on
        # hard, and the earlier list-shaped version was the hard case.
        cands = [candidate("winner", 4, source=s, rank=0)
                 for s in ("hot_vector", "cold_semantic", "bfs_hot")]
        cands.append(candidate("other", 9, source="hot_vector", rank=1))
        boosted = [r.entity_name for r in rerank(
            list(cands), DecayConfig(source_grounding=0.05), 0,
            grounding={"other": 1.0})]
        assert boosted[0] == "other"

    def test_no_grounding_map_changes_nothing(self):
        assert [r.entity_name for r in rerank(pool(), DecayConfig(), 0)] == \
            [r.entity_name for r in rerank(
                pool(), DecayConfig(source_grounding=0.001), 0, grounding=None)]


# --------------------------------------------------------------------------
# Choosing which turns get hydrated, using the same scorer.
#
# Ranking a fact higher and reading the turn behind it are two decisions, and
# the shipped code made only the first of them with the question in hand.
# `_source_turns` took `records[:hydrate_top_k]` and hydrated whatever turns
# those facts came from, so the question chose which *lines* of a turn survived
# and never which turns were fetched. On the bucket that decides open-domain --
# 47 of 116 losses where the answer was already in the prompt and the model
# picked a competing value -- the deciding word is in a turn, and which turn
# gets read is the whole question.
#
# The discipline these tests exist to pin: it changes which turns, never how
# many. On a 1,000-token budget against RAG's 1,430 there is no free slot, so
# every turn this chooses is a turn it dropped.
# --------------------------------------------------------------------------


def rec(entity: str, turn: int) -> MemoryRecord:
    return MemoryRecord(entity_name=entity, fact_content=f"{entity} fact",
                        memory_type=MemoryType.ACTIVE_CONTEXT,
                        created_at_turn=turn)


def picker(texts: dict[int, str], pool_size: int = 40) -> QueryRouter:
    return QueryRouter(hot=None, cold=FakeCold(texts), embedder=None,
                       config=DecayConfig(hydrate_pool=pool_size))


# Ranked worst-first on purpose: turn 4 is the fact ranking's favourite and turn
# 9 is the one holding the word the question turns on. Any test below that
# passes with an empty return would also pass if the feature did nothing, so
# the ordering is what makes them able to fail.
RANKED = [rec("devil", 4), rec("gatsby", 9), rec("garden", 12)]


class TestChoosingWhichTurnsToHydrate:
    def test_it_takes_the_turn_the_question_points_at(self):
        # Fact rank would have hydrated turn 4 and then 12. `gripping` is only
        # in turn 9, and the question reaches past 12 to get it.
        assert picker(READING)._turns_by_question(QUESTION, RANKED, 2) == [4, 9]

    def test_the_top_ranked_turn_is_never_dropped(self):
        # Four of the five regressions in the paired open-domain run were the
        # same shape: a gold spread over two turns, where the shipped arm
        # answered "showed him his work and gave him advice" and this one
        # answered "showed him his work". Concentrating every slot on the best
        # question match drops the turn holding the rest of the answer.
        wider = dict(READING) | {
            15: "Evan: the weather has been grim all week.",
            18: "Evan: I finished the gripping one last night, what a novel.",
        }
        ranked = RANKED + [rec("weather", 15), rec("finished", 18)]
        for want in (2, 3):
            assert picker(wider)._turns_by_question(
                QUESTION, ranked, want)[0] == 4

    def test_with_one_slot_the_shipped_behaviour_stands(self):
        # The consequence of the rule above, stated rather than discovered: a
        # question hydrating a single turn has no slot left to spend, so this
        # cannot help there. Better than the alternative, which is that the
        # single-turn case is exactly where concentrating on one match is
        # riskiest.
        assert picker(READING)._turns_by_question(QUESTION, RANKED, 1) == [4]

    def test_it_never_takes_more_than_it_was_given(self):
        chosen = picker(READING)._turns_by_question(QUESTION, RANKED, 2)
        assert len(chosen) == 2

    def test_it_is_off_unless_asked_for(self):
        assert DecayConfig().hydrate_pool == 0
        off = QueryRouter(hot=None, cold=FakeCold(READING), embedder=None,
                          config=DecayConfig())
        assert off._turns_by_question(QUESTION, RANKED, 1) == []

    def test_a_pool_no_wider_than_the_budget_is_left_alone(self):
        # Nothing to choose from: rank would hydrate all three anyway, so
        # overriding it could only reorder for no gain.
        assert picker(READING)._turns_by_question(QUESTION, RANKED, 3) == []

    def test_a_question_it_has_no_opinion_about_is_left_alone(self):
        # All-zero is not a preference for the first turns. Falling back to the
        # measured behaviour is the only honest thing to do with no signal.
        silent = "Which flux capacitor calibrates the quantum manifold?"
        assert picker(READING)._turns_by_question(silent, RANKED, 1) == []

    def test_no_question_is_left_alone(self):
        assert picker(READING)._turns_by_question(None, RANKED, 1) == []
        assert picker(READING)._turns_by_question("", RANKED, 1) == []

    def test_the_pool_bounds_what_is_even_considered(self):
        # A width of one fact cannot reach turn 9, so there is nothing to
        # choose and the shipped behaviour stands.
        assert picker(READING, pool_size=1)._turns_by_question(
            QUESTION, RANKED, 1) == []

    def test_fact_rank_breaks_a_tie(self):
        # Two turns the question cannot separate fall back to the order that
        # was measured, not to whatever order the dict happened to hold.
        same = "Evan: I am reading a novel and finding it gripping."
        texts = {4: same, 9: same, 12: "Deborah: the garden looks lovely."}
        assert picker(texts)._turns_by_question(QUESTION, RANKED, 1) == [4]
        flipped = [rec("gatsby", 9), rec("devil", 4), rec("garden", 12)]
        assert picker(texts)._turns_by_question(QUESTION, flipped, 1) == [9]

    def test_it_only_reads_turns_it_might_choose(self):
        # Hydration is the expensive half of the prompt. Reading the whole
        # store to decide which three turns to print would make the saving
        # theoretical.
        r = picker(READING, pool_size=2)
        r._turns_by_question(QUESTION, RANKED, 1)
        assert r._cold.asked == [[4, 9]]

    def test_a_cold_store_without_turn_text_falls_back(self):
        # Stores written before `turn_text` existed hydrate nothing, and this
        # must degrade to the shipped path rather than raising.
        r = QueryRouter(hot=None, cold=object(), embedder=None,
                        config=DecayConfig(hydrate_pool=40))
        assert r._turns_by_question(QUESTION, RANKED, 1) == []


# The failure the scan was built for, in one fixture. Turn 30 holds the answer's
# own noun and no fact in RANKED points anywhere near it, so under `hydrate_pool`
# at any width turn 30 is unreachable: the pool is drawn from the retrieved list
# and nothing in the retrieved list nominates it.
#
# This is the shape of 36 of the 55 open-domain questions still lost, where the
# gold has no content word anywhere in the rendered context and the missing word
# is a concrete noun the extractor generalised away.
CONVERSATION = dict(READING) | {
    30: "Gina: I made a limited edition line of hoodies for the shop.",
    31: "Gina: the shop has been quiet all week.",
    33: "Gina: there was a limited edition print run as well.",
}
LINE = "What did Gina make a limited edition line of?"


def scanner(scan: int, pool_size: int = 40) -> QueryRouter:
    return QueryRouter(
        hot=None, cold=FakeCold(CONVERSATION, whole=CONVERSATION),
        embedder=None,
        config=DecayConfig(hydrate_pool=pool_size, hydrate_scan=scan))


class TestScanningPastWhatRetrievalFound:
    def test_a_turn_no_fact_points_at_can_now_be_hydrated(self):
        # The whole point. Turn 30 is in no record, and it is chosen.
        assert scanner(8)._turns_by_question(LINE, RANKED, 2) == [4, 30]

    def test_the_scan_is_off_unless_asked_for(self):
        assert DecayConfig().hydrate_scan == 0
        off = QueryRouter(hot=None,
                          cold=FakeCold(CONVERSATION, whole=CONVERSATION),
                          embedder=None, config=DecayConfig(hydrate_pool=40))
        assert 30 not in off._turns_by_question(LINE, RANKED, 2)

    def test_the_reserved_top_ranked_turn_survives_the_scan(self):
        # Same rule as `hydrate_pool`, and it has to hold against a signal that
        # can now outrank every retrieved turn at once. A scan free to take
        # every slot would be a different mechanism wearing this one's budget.
        for want in (2, 3):
            assert scanner(8)._turns_by_question(LINE, RANKED, want)[0] == 4

    def test_it_still_never_hydrates_more_turns_than_before(self):
        # `want` is the number of turns fact rank would have read. The scan
        # changes which turns are read and must never change how many, because
        # every extra turn is tokens against a 1,000-token ceiling.
        for want in (1, 2, 3):
            assert len(scanner(8)._turns_by_question(LINE, RANKED, want)) == want

    def test_a_turn_sharing_nothing_with_the_question_is_never_taken(self):
        # A zero-scoring turn is not a preference. Spending a hydration slot on
        # an arbitrary turn is strictly worse than leaving it with the fact that
        # earned it, so no signal means no change.
        silent = "Which flux capacitor calibrates the quantum manifold?"
        assert scanner(8)._turns_by_question(silent, RANKED, 2) == []

    def test_the_setting_caps_how_many_may_compete(self):
        # Turn 33 also matches -- "limited edition" -- and at a scan of one only
        # the better of the two is allowed in at all.
        assert 33 not in scanner(1)._turns_by_question(LINE, RANKED, 3)
        assert 33 in scanner(8)._turns_by_question(LINE, RANKED, 3)

    def test_it_never_offers_a_turn_retrieval_already_found(self):
        chosen = scanner(8)._turns_by_question(LINE, RANKED, 3)
        assert len(chosen) == len(set(chosen))

    def test_the_scan_alone_is_enough(self):
        # `hydrate_pool` off, scan on: the two are independent mechanisms and
        # the newer one must not need the older one configured to work.
        chosen = scanner(8, pool_size=0)._turns_by_question(LINE, RANKED, 2)
        assert chosen == [4, 30]

    def test_the_conversation_is_read_once_per_question(self):
        r = scanner(8)
        r._turns_by_question(LINE, RANKED, 2)
        assert r._cold.scans == 1

    def test_a_cold_store_that_cannot_scan_falls_back(self):
        # Backends with no turn text at all, and stores written before this
        # existed, must degrade to the shipped path rather than raising.
        r = QueryRouter(hot=None, cold=object(), embedder=None,
                        config=DecayConfig(hydrate_pool=40, hydrate_scan=8))
        assert r._turns_by_question(LINE, RANKED, 1) == []
        blind = QueryRouter(hot=None, cold=FakeCold(CONVERSATION),
                            embedder=None,
                            config=DecayConfig(hydrate_scan=8))
        assert 30 not in blind._turns_by_question(LINE, RANKED, 2)


# The failure that motivated bounding the scan, in one fixture. The store holds
# more than one conversation laid end to end, and turn 900 belongs to strangers
# who happen to use the question's rarest word. It outscores the turn that
# answers the question, because BM25 rewards the rare word and knows nothing
# about who is speaking.
#
# This was read off a real rendered context: asked which novel Evan finds
# gripping, four of the five hydrated turns came from other people's
# conversations, matched on "gripping" or "novel". Most of the source budget
# went on distractors, and the model answered from a confident wrong candidate.
FAR = dict(CONVERSATION) | {
    900: "Nadia: a limited edition line, another limited edition line, and a "
         "third limited edition line.",
    905: "Nadia: limited edition line, limited edition line, limited line.",
}


STRANGERS = {900, 905}


def bounded(span_facts: int, margin: int, scan: int = 8) -> QueryRouter:
    return QueryRouter(
        hot=None, cold=FakeCold(FAR, whole=FAR), embedder=None,
        config=DecayConfig(hydrate_pool=40, hydrate_scan=scan,
                           scan_span_facts=span_facts,
                           scan_span_margin=margin))


class TestBoundingTheScanToWhereRetrievalPointed:
    def test_a_far_turn_wins_a_slot_when_the_scan_is_unbounded(self):
        # The bug, stated as a passing test so the fix has something to reverse.
        # RANKED points at turns 4, 9 and 12; the strangers' turns are in no
        # record and in no conversation of theirs, and one is hydrated anyway.
        chosen = bounded(0, 0)._turns_by_question(LINE, RANKED, 3)
        assert STRANGERS & set(chosen)

    def test_bounding_the_span_keeps_it_out(self):
        chosen = bounded(3, 80)._turns_by_question(LINE, RANKED, 3)
        assert not STRANGERS & set(chosen)

    def test_the_turn_the_scan_was_built_for_still_arrives(self):
        # Bounding must not undo the mechanism it bounds. Turn 30 is nominated
        # by no fact, sits inside the span, and is still chosen.
        assert 30 in bounded(3, 80)._turns_by_question(LINE, RANKED, 3)

    def test_it_is_off_unless_asked_for(self):
        assert DecayConfig().scan_span_facts == 0
        assert DecayConfig().scan_span_margin == 0
        assert STRANGERS & set(bounded(0, 80)._turns_by_question(
            LINE, RANKED, 3))

    def test_the_margin_is_what_admits_a_neighbouring_turn(self):
        # Turn 33 is 21 turns past the last ranked fact's turn. A margin of 8
        # cannot reach it and a margin of 80 can, which is the knob doing
        # exactly what it says and nothing else.
        assert 33 not in bounded(3, 8)._turns_by_question(LINE, RANKED, 3)
        assert 33 in bounded(3, 80)._turns_by_question(LINE, RANKED, 3)

    def test_only_the_top_facts_define_the_span(self):
        # A span drawn from the whole retrieved list is nearly the whole store,
        # because the tail of that list is where off-subject facts sit. One
        # stray fact at turn 900 must not widen the span to include turn 905.
        strays = RANKED + [rec("sneakers", 900)]
        assert 905 not in bounded(3, 80)._turns_by_question(LINE, strays, 4)
        assert 905 in bounded(4, 80)._turns_by_question(LINE, strays, 4)

    def test_it_still_never_hydrates_more_turns_than_before(self):
        for want in (1, 2, 3):
            assert len(bounded(3, 80)._turns_by_question(
                LINE, RANKED, want)) == want

    def test_the_reserved_top_ranked_turn_survives_the_bound(self):
        # The first turn defines the span, so excluding it would be incoherent,
        # and it is the turn fact rank was surest about.
        assert bounded(3, 80)._turns_by_question(LINE, RANKED, 2)[0] == 4

    def test_a_retrieved_fact_outside_the_span_loses_its_slot_too(self):
        # A fact from another conversation is off-subject for the same reason a
        # scanned turn from one is, and it arrives with rank behind it, which
        # makes it likelier to win a slot rather than less.
        strays = RANKED + [rec("sneakers", 900)]
        assert 900 not in bounded(3, 80)._turns_by_question(LINE, strays, 4)

    def test_rarity_is_still_judged_against_the_whole_store(self):
        # The bound says which turns may be taken, deliberately not which are
        # scored. Narrowing the corpus would make a word common in one
        # conversation look rare, which is the opposite of what this is for.
        r = bounded(3, 80)
        r._turns_by_question(LINE, RANKED, 2)
        assert r._cold.scans == 1
