"""Rewiring links around pruned records, so forgetting costs a node not a region.

Covers plan_rewiring directly rather than through prune_cycle: the interesting
behaviour is entirely in choosing the replacement links, and driving it through
storage would test LanceDB batching instead.
"""

from __future__ import annotations

import pytest

from nmafc.engine.pruning import plan_rewiring
from nmafc.schemas.memory import MemoryRecord, MemoryType


def rec(entity: str, links: list[str] | None = None,
        mtype: MemoryType = MemoryType.ACTIVE_CONTEXT) -> MemoryRecord:
    return MemoryRecord(
        entity_name=entity,
        fact_content=f"fact about {entity}",
        memory_type=mtype,
        related_entities=links or [],
        created_at_turn=1,
        last_reinforced_turn=1,
    )


def rewire(records, doomed_entities, **kw):
    """Plan rewiring given entity names rather than record ids, for readability."""
    doomed_ids = {r.id for r in records if r.entity_name in doomed_entities}
    return dict(plan_rewiring(records, doomed_ids, kw.pop("max_links", 8), **kw))


def test_link_through_a_pruned_record_is_rerouted_not_lost():
    """A -> B -> C with B pruned should leave A -> C, not A -> nothing."""
    a, b, c = rec("a", ["b"]), rec("b", ["c"]), rec("c")
    plan = rewire([a, b, c], {"b"})
    assert plan[a.id] == ["c"]


def test_chain_of_pruned_records_is_walked_through():
    """Small talk decays in runs, so the replacement is often several hops out."""
    a, b, c, d = rec("a", ["b"]), rec("b", ["c"]), rec("c", ["d"]), rec("d")
    plan = rewire([a, b, c, d], {"b", "c"})
    assert plan[a.id] == ["d"]


def test_walk_stops_at_max_depth():
    """An uncapped walk would eventually connect a conversation to itself."""
    records = [rec("a", ["b"]), rec("b", ["c"]), rec("c", ["d"]),
               rec("d", ["e"]), rec("e")]
    doomed = {"b", "c", "d"}
    assert rewire(records, doomed, max_depth=3)[records[0].id] == ["e"]
    # Two levels of pruned nodes is one short of reaching e, so there is
    # nothing to attach and the link is dropped rather than invented.
    assert rewire(records, doomed, max_depth=2)[records[0].id] == []


def test_surviving_links_are_never_displaced_by_rewired_ones():
    """An inferred connection must not cost a stated one when the cap binds."""
    hub = rec("hub", ["x1", "x2", "x3"])
    a = rec("a", ["keep1", "keep2", "hub"])
    records = [a, hub, rec("keep1"), rec("keep2"),
               rec("x1"), rec("x2"), rec("x3")]
    plan = rewire(records, {"hub"}, max_links=3)
    assert plan[a.id][:2] == ["keep1", "keep2"]
    assert len(plan[a.id]) == 3


def test_fanout_is_capped():
    """A doomed hub must not hand its whole neighbourhood to every neighbour."""
    hub = rec("hub", [f"n{i}" for i in range(12)])
    a = rec("a", ["hub"])
    records = [a, hub] + [rec(f"n{i}") for i in range(12)]
    plan = rewire(records, {"hub"}, max_links=4)
    assert len(plan[a.id]) == 4


def test_records_untouched_by_pruning_are_not_rewritten():
    """Only records that actually pointed at something doomed produce updates."""
    a, b, c = rec("a", ["c"]), rec("b", ["c"]), rec("c")
    doomed_ids = {b.id}
    assert plan_rewiring([a, b, c], doomed_ids, max_links=8) == []


def test_doomed_records_are_not_given_new_links():
    """Rewriting a row that is about to be deleted is wasted work."""
    a, b = rec("a", ["b"]), rec("b", ["a"])
    doomed_ids = {a.id, b.id}
    assert plan_rewiring([a, b], doomed_ids, max_links=8) == []


def test_self_links_are_not_created():
    """A -> B -> A collapses to nothing, not to a self-reference."""
    a, b = rec("a", ["b"]), rec("b", ["a"])
    plan = rewire([a, b], {"b"})
    assert plan[a.id] == []


def test_duplicate_targets_are_collapsed():
    """Two doomed waypoints onto one survivor is still a single link."""
    a = rec("a", ["b", "c"])
    records = [a, rec("b", ["z"]), rec("c", ["z"]), rec("z")]
    plan = rewire(records, {"b", "c"})
    assert plan[a.id] == ["z"]


def test_a_rewired_target_already_linked_is_not_duplicated():
    a = rec("a", ["z", "b"])
    records = [a, rec("b", ["z"]), rec("z")]
    plan = rewire(records, {"b"})
    assert plan[a.id] == ["z"]


def test_nothing_doomed_means_nothing_to_do():
    a = rec("a", ["b"])
    assert plan_rewiring([a, rec("b")], set(), max_links=8) == []


def test_zero_cap_disables_rewiring_entirely():
    a, b, c = rec("a", ["b"]), rec("b", ["c"]), rec("c")
    assert plan_rewiring([a, b, c], {b.id}, max_links=0) == []


@pytest.mark.parametrize("cased", ["B", "b", "  b"])
def test_matching_is_case_insensitive_like_the_traversal(cased):
    """QueryRouter compares entity names lowercased; rewiring must agree."""
    a, b, c = rec("a", [cased.strip()]), rec("B", ["c"]), rec("c")
    plan = rewire([a, b, c], {"B"})
    assert plan[a.id] == ["c"]
