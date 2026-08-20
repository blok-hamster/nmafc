"""Reciprocal Rank Fusion (RRF) reranker for multi-source retrieval.

Fuses heterogeneous result lists — vector hits, keyword matches, and graph
traversal — into a single relevance-ranked output. Controls fan-out naturally:
BFS can discover 35 records, but only the top_k most relevant survive fusion.
"""

from __future__ import annotations

from collections import defaultdict

from nmafc.schemas.memory import DecayConfig, MemoryRecord, SearchCandidate


def reciprocal_rank_fusion(
    candidate_lists: dict[str, list[SearchCandidate]],
    k: int = 60,
    top_k: int = 15,
    current_turn: int = 0,
    recency_boost: float = 0.0,
    weight_signal: float = 0.0,
) -> list[MemoryRecord]:
    """Compute RRF scores across all source lists, return top_k records.

    RRF_score(d) = sum(1 / (k + rank_in_list_i)) for each list containing d.

    Deduplication: when the same entity_name appears from multiple sources,
    the Hot-sourced version takes priority; otherwise the highest-scoring one wins.

    Optional additive modifiers applied after fusion:
    - recency_boost: score += recency_boost * (1 - age/max_age)
    - weight_signal: score += weight_signal * record.weight
    """
    # Assign ranks (1-indexed) within each source list
    for source, items in candidate_lists.items():
        if source in ("hot_vector", "cold_semantic"):
            items.sort(key=lambda c: -(c.score or 0))
        elif source == "cold_keyword":
            items.sort(key=lambda c: c.rank_in_source)
        else:
            items.sort(key=lambda c: (c.hop_distance, -(c.record.weight)))

    # Compute RRF score per unique entity, keeping the best record per entity
    entity_scores: dict[str, float] = defaultdict(float)
    entity_records: dict[str, MemoryRecord] = {}
    entity_source_priority: dict[str, int] = {}

    source_priority = {"hot_vector": 0, "bfs_hot": 1, "cold_semantic": 2, "cold_keyword": 3, "bfs_cold": 4}

    for source, items in candidate_lists.items():
        for rank_idx, candidate in enumerate(items):
            entity_key = candidate.record.entity_name.lower()
            rrf_contribution = 1.0 / (k + rank_idx + 1)
            entity_scores[entity_key] += rrf_contribution

            # Keep the record from the highest-priority source
            src_pri = source_priority.get(source, 5)
            if entity_key not in entity_records or src_pri < entity_source_priority[entity_key]:
                entity_records[entity_key] = candidate.record
                entity_source_priority[entity_key] = src_pri

    # Apply optional modifiers
    if recency_boost > 0 and current_turn > 0:
        max_age = max(
            (current_turn - rec.created_at_turn for rec in entity_records.values()),
            default=1,
        )
        if max_age > 0:
            for entity_key, rec in entity_records.items():
                age = current_turn - rec.created_at_turn
                entity_scores[entity_key] += recency_boost * (1.0 - age / max_age)

    if weight_signal > 0:
        for entity_key, rec in entity_records.items():
            entity_scores[entity_key] += weight_signal * rec.weight

    # Sort by RRF score descending, take top_k
    ranked = sorted(entity_scores.keys(), key=lambda e: -entity_scores[e])
    return [entity_records[e] for e in ranked[:top_k]]


def rerank(
    candidates: list[SearchCandidate],
    config: DecayConfig,
    current_turn: int = 0,
) -> list[MemoryRecord]:
    """Main entry point: group candidates by source, apply RRF, return top_k."""
    if not candidates:
        return []

    # Group by source
    lists: dict[str, list[SearchCandidate]] = defaultdict(list)
    for c in candidates:
        lists[c.source].append(c)

    return reciprocal_rank_fusion(
        candidate_lists=dict(lists),
        k=config.rrf_k,
        top_k=config.rerank_top_k,
        current_turn=current_turn,
        recency_boost=config.recency_boost,
        weight_signal=config.weight_signal,
    )
