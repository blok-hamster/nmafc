from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping

from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType

EntityGraph = Mapping[str, set[str]]


def get_lambda_base(memory_type: MemoryType, config: DecayConfig) -> float:
    return config.get_lambda_base(memory_type)


def build_entity_graph(records: Iterable[MemoryRecord]) -> dict[str, set[str]]:
    """Undirected adjacency over entity names, from every record's related_entities.

    Built over *all* records rather than the mutable ones alone. Core anchors
    never decay, but they are still nodes: a fact linked to the user's identity
    sits in a denser neighbourhood than one that is not, and dropping anchors
    from the graph would hide exactly the links most worth counting.

    Names are lowercased to match QueryRouter's traversal, which compares
    entities case-insensitively.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    for record in records:
        entity = record.entity_name.lower()
        adjacency[entity]  # nodes with no links still exist, with degree 0
        for related in record.related_entities:
            neighbour = related.lower()
            if neighbour == entity:
                continue  # a self-link is not connectivity
            adjacency[entity].add(neighbour)
            adjacency[neighbour].add(entity)
    return dict(adjacency)


def clustering_coefficient(graph: EntityGraph, entity: str) -> float:
    """Local clustering coefficient of one node, in [0, 1].

    The fraction of a node's neighbour pairs that are themselves linked. A fact
    whose entities all reference each other scores 1; a fact that fans out to
    unrelated entities scores 0, however many of them there are.

    Degree alone is deliberately not used. Lin et al.'s anaesthesia control
    recovered synapse density to naive levels without recovering the memory, so
    the count of connections is precisely the measure their data rules out.
    Nodes with fewer than two neighbours have no pairs and score 0, which means
    a highly-connected hub with no interlinking earns no protection either.
    """
    neighbours = [n for n in graph.get(entity.lower(), ()) if n != entity.lower()]
    degree = len(neighbours)
    if degree < 2:
        return 0.0

    links = 0
    for i in range(degree):
        adjacent = graph.get(neighbours[i], ())
        for j in range(i + 1, degree):
            if neighbours[j] in adjacent:
                links += 1

    return (2.0 * links) / (degree * (degree - 1))


def compute_alpha(k: int, eta: float) -> float:
    """Consolidation coefficient: e^{-eta * k}.

    Shrinks exponentially as retrieval count k increases,
    making frequently-recalled facts resistant to decay.
    """
    return math.exp(-eta * k)


def compute_lambda(
    memory_type: MemoryType,
    k: int,
    config: DecayConfig,
    clustering: float = 0.0,
) -> float:
    """Effective decay rate: lambda_base(type) * alpha(k) * (1 - beta * C).

    `clustering` is the local clustering coefficient of the record's entity. At
    the default beta of 0.0 the third term is 1 and this reduces exactly to the
    original two-factor rate, so callers that know nothing about the graph --
    and every existing test -- are unaffected.
    """
    base = get_lambda_base(memory_type, config)
    alpha = compute_alpha(k, config.eta)
    protection = 1.0 - config.beta * clustering
    return base * alpha * protection


def compute_weight(w0: float, lambda_val: float, delta_t: int) -> float:
    """Synaptic weight after delta_t turns: w0 * e^{-lambda * delta_t}."""
    if lambda_val == 0.0:
        return w0
    return w0 * math.exp(-lambda_val * delta_t)


def decay_record(
    record: MemoryRecord,
    current_turn: int,
    config: DecayConfig,
    clustering: float = 0.0,
) -> float:
    """Compute the decayed weight for a single record at the current turn.

    `record.weight` is read as w0. Callers that persist the result must advance
    record.last_reinforced_turn with it, or the elapsed term keeps growing
    against a weight that has already absorbed it -- see
    HotStorage.apply_weight_updates.
    """
    if record.memory_type == MemoryType.CORE_ANCHOR:
        return record.weight

    delta_t = current_turn - record.last_reinforced_turn
    if delta_t <= 0:
        return record.weight

    lam = compute_lambda(
        record.memory_type, record.consolidation_index, config, clustering
    )
    return compute_weight(record.weight, lam, delta_t)


def decay_all(
    records: list[MemoryRecord],
    current_turn: int,
    config: DecayConfig,
    graph: EntityGraph | None = None,
) -> list[tuple[str, float]]:
    """Compute new weights for all mutable records.

    Returns list of (record_id, new_weight) pairs.
    Core anchors are skipped (they never decay).

    `graph` supplies the connectivity used for clustering protection, and should
    be built over every record rather than just the ones passed here (see
    build_entity_graph). Omitting it scores every record at C = 0, which is also
    what beta = 0 produces, so the mechanism stays off unless both are supplied.
    Coefficients are memoised per entity: several records commonly share one
    entity, and the calculation is quadratic in that entity's degree.
    """
    results = []
    coefficients: dict[str, float] = {}

    for record in records:
        if record.memory_type == MemoryType.CORE_ANCHOR:
            continue

        clustering = 0.0
        if graph is not None and config.beta > 0.0:
            entity = record.entity_name.lower()
            if entity not in coefficients:
                coefficients[entity] = clustering_coefficient(graph, entity)
            clustering = coefficients[entity]

        new_weight = decay_record(record, current_turn, config, clustering)
        results.append((record.id, new_weight))
    return results
