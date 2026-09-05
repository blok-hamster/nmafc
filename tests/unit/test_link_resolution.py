"""Resolving extracted link names onto entities that actually exist."""

from __future__ import annotations

from nmafc.engine.linking import resolve_link_targets


def test_exact_match_passes_through():
    assert resolve_link_targets(["a_b"], ["a_b", "c_d"]) == ["a_b"]


def test_exact_match_returns_the_stored_casing():
    """Traversal lowercases but storage does not, so the stored form must win."""
    assert resolve_link_targets(["caroline_x"], ["Caroline_X"]) == ["Caroline_X"]


def test_near_miss_resolves_to_the_stored_entity():
    """The measured failure: one extra qualifier on an otherwise correct name."""
    stored = ["melanie_lake_sunrise_painting", "melanie_work_stress"]
    assert resolve_link_targets(
        ["melanie_lake_sunrise_painting_2022"], stored
    ) == ["melanie_lake_sunrise_painting"]


def test_dropped_qualifier_also_resolves():
    stored = ["caroline_adoption_research"]
    assert resolve_link_targets(
        ["caroline_adoption_agency_research"], stored
    ) == ["caroline_adoption_research"]


def test_unrelated_name_is_dropped_not_guessed():
    """A link naming nothing stored cannot be traversed, so it is not kept."""
    assert resolve_link_targets(["zebra_quantum_thing"], ["caroline_pottery"]) == []


def test_sharing_only_a_first_name_is_not_a_match():
    """Otherwise every fact about one person links to every other."""
    stored = ["caroline_pottery_class_progress"]
    assert resolve_link_targets(["caroline_marathon_training_plan"], stored) == []


def test_best_candidate_wins_not_the_first_seen():
    stored = ["caroline_adoption_dream", "caroline_adoption_agency_research"]
    assert resolve_link_targets(["caroline_adoption_agency"], stored) == [
        "caroline_adoption_agency_research"
    ]


def test_duplicates_collapse_after_resolution():
    """Two different misspellings of one entity are still a single link."""
    stored = ["melanie_lake_painting"]
    out = resolve_link_targets(
        ["melanie_lake_painting_2022", "melanie_lake_painting"], stored
    )
    assert out == ["melanie_lake_painting"]


def test_order_is_preserved():
    stored = ["a_one", "b_two", "c_three"]
    assert resolve_link_targets(["c_three", "a_one"], stored) == ["c_three", "a_one"]


def test_no_known_entities_means_no_links():
    assert resolve_link_targets(["a_b"], []) == []


def test_empty_link_name_is_skipped():
    assert resolve_link_targets(["", "___"], ["a_b"]) == []


def test_threshold_of_one_admits_only_exact_matches():
    stored = ["melanie_lake_painting"]
    assert resolve_link_targets(
        ["melanie_lake_painting_2022"], stored, min_overlap=1.0
    ) == []
    assert resolve_link_targets(
        ["melanie_lake_painting"], stored, min_overlap=1.0
    ) == ["melanie_lake_painting"]


def test_threshold_is_inclusive_at_the_boundary():
    """Two shared parts of four scores exactly 0.5 and must be admitted."""
    stored = ["caroline_adoption_research_notes"]
    assert resolve_link_targets(
        ["caroline_adoption"], stored, min_overlap=0.5
    ) == ["caroline_adoption_research_notes"]
