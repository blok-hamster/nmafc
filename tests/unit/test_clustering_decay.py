"""Clustering protection: densely interlinked facts decay more slowly.

Motivated by Lin et al. (2026, Science 393, eaee7004), where hippocampal memory
survived the elimination of most synapses and what survived was *clustered*
connectivity. Their anaesthesia control is the reason these tests insist on
clustering rather than degree: synapse density recovered to naive levels while
the memory stayed impaired, so a count of connections predicted nothing.

The mapping from spatial clustering on a dendrite to graph clustering over
entity names is an analogy, and these tests only pin the arithmetic.
"""

from __future__ import annotations

import math

import pytest

from nmafc.engine.decay import (
    build_entity_graph,
    clustering_coefficient,
    compute_lambda,
    decay_all,
    decay_record,
)
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType


def record(entity: str, related: list[str], **kwargs) -> MemoryRecord:
    defaults = {
        "entity_name": entity,
        "fact_content": f"Fact about {entity}",
        "memory_type": MemoryType.ACTIVE_CONTEXT,
        "weight": 1.0,
        "consolidation_index": 0,
        "created_at_turn": 0,
        "last_reinforced_turn": 0,
        "related_entities": related,
    }
    defaults.update(kwargs)
    return MemoryRecord(**defaults)


class TestBuildEntityGraph:
    def test_links_are_undirected(self):
        graph = build_entity_graph([record("a", ["b"])])
        assert graph["a"] == {"b"}
        assert graph["b"] == {"a"}

    def test_names_are_lowercased_to_match_traversal(self):
        """QueryRouter compares entities case-insensitively; the graph must agree."""
        graph = build_entity_graph([record("Alice", ["BOB"])])
        assert graph["alice"] == {"bob"}
        assert graph["bob"] == {"alice"}

    def test_self_links_are_dropped(self):
        graph = build_entity_graph([record("a", ["a", "b"])])
        assert graph["a"] == {"b"}

    def test_unlinked_records_are_still_nodes(self):
        graph = build_entity_graph([record("lonely", [])])
        assert graph["lonely"] == set()


class TestClusteringCoefficient:
    def test_triangle_scores_one(self):
        graph = build_entity_graph(
            [record("a", ["b", "c"]), record("b", ["c"])]
        )
        assert clustering_coefficient(graph, "a") == pytest.approx(1.0)

    def test_star_scores_zero(self):
        """A hub whose neighbours ignore each other earns nothing."""
        graph = build_entity_graph([record("hub", ["x", "y", "z", "w"])])
        assert clustering_coefficient(graph, "hub") == 0.0

    def test_degree_alone_confers_nothing(self):
        """The paper's control: connection count recovered, memory did not.

        A node with eight unrelated neighbours must not outscore a node with two
        that reference each other, or the measure has collapsed back into degree.
        """
        star = build_entity_graph([record("hub", [f"n{i}" for i in range(8)])])
        triangle = build_entity_graph([record("a", ["b", "c"]), record("b", ["c"])])
        assert clustering_coefficient(star, "hub") < clustering_coefficient(
            triangle, "a"
        )

    def test_partial_interlinking_scores_between(self):
        # a-b, a-c, a-d with only b-c linked: 1 of 3 neighbour pairs.
        graph = build_entity_graph(
            [record("a", ["b", "c", "d"]), record("b", ["c"])]
        )
        assert clustering_coefficient(graph, "a") == pytest.approx(1 / 3)

    def test_fewer_than_two_neighbours_has_no_pairs(self):
        graph = build_entity_graph([record("a", ["b"])])
        assert clustering_coefficient(graph, "a") == 0.0

    def test_unknown_entity_is_zero_not_an_error(self):
        assert clustering_coefficient({}, "missing") == 0.0


class TestBetaIsOffByDefault:
    def test_default_config_disables_the_mechanism(self):
        assert DecayConfig().beta == 0.0

    def test_lambda_unchanged_when_beta_is_zero(self):
        config = DecayConfig()
        clustered = compute_lambda(MemoryType.ACTIVE_CONTEXT, 0, config, 1.0)
        plain = compute_lambda(MemoryType.ACTIVE_CONTEXT, 0, config, 0.0)
        assert clustered == plain

    def test_decay_all_ignores_the_graph_when_beta_is_zero(self):
        """beta = 0 is the ablation control and must reproduce the old path exactly."""
        config = DecayConfig()
        records = [record("a", ["b", "c"]), record("b", ["c"])]
        graph = build_entity_graph(records)

        with_graph = decay_all(records, 10, config, graph)
        without = decay_all(records, 10, config)

        assert [w for _, w in with_graph] == [w for _, w in without]


class TestClusteringSlowsDecay:
    def test_clustered_record_outlives_an_isolated_one(self):
        config = DecayConfig(beta=0.5)
        records = [
            record("a", ["b", "c"]),
            record("b", ["c"]),
            record("lonely", []),
        ]
        graph = build_entity_graph(records)

        weights = dict(decay_all(records, 20, config, graph))
        clustered = weights[records[0].id]
        isolated = weights[records[2].id]

        assert clustered > isolated

    def test_protection_matches_the_closed_form(self):
        config = DecayConfig(beta=0.5)
        rec = record("a", ["b", "c"])
        graph = build_entity_graph([rec, record("b", ["c"])])
        c = clustering_coefficient(graph, "a")

        got = decay_record(rec, 10, config, c)

        expected = math.exp(-config.lambda_active_context * (1 - 0.5 * c) * 10)
        assert got == pytest.approx(expected, rel=1e-9)

    def test_beta_scales_the_protection(self):
        rec = record("a", ["b", "c"])
        graph = build_entity_graph([rec, record("b", ["c"])])

        weak = decay_all([rec], 20, DecayConfig(beta=0.2), graph)[0][1]
        strong = decay_all([rec], 20, DecayConfig(beta=0.9), graph)[0][1]

        assert strong > weak

    def test_protection_can_never_stop_decay(self):
        """beta < 1 is enforced by the schema, so lambda stays strictly positive.

        At the ceiling this slows decay by 100x -- a fully clustered
        ActiveContext record needs ~4,600 turns to reach the prune threshold
        rather than ~46 -- but it never becomes a second class of anchor.
        """
        config = DecayConfig(beta=0.99)
        rec = record("a", ["b", "c"])
        graph = build_entity_graph([rec, record("b", ["c"])])

        assert decay_all([rec], 5000, config, graph)[0][1] < config.w_prune

    def test_beta_of_one_is_rejected(self):
        with pytest.raises(ValueError):
            DecayConfig(beta=1.0)

    def test_core_anchors_are_still_skipped(self):
        config = DecayConfig(beta=0.5)
        anchor = record("identity", ["b", "c"], memory_type=MemoryType.CORE_ANCHOR)
        graph = build_entity_graph([anchor, record("b", ["c"])])

        assert decay_all([anchor], 50, config, graph) == []
