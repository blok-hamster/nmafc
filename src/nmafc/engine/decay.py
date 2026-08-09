from __future__ import annotations

import math

from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType


def get_lambda_base(memory_type: MemoryType, config: DecayConfig) -> float:
    return config.get_lambda_base(memory_type)


def compute_alpha(k: int, eta: float) -> float:
    """Consolidation coefficient: e^{-eta * k}.

    Shrinks exponentially as retrieval count k increases,
    making frequently-recalled facts resistant to decay.
    """
    return math.exp(-eta * k)


def compute_lambda(memory_type: MemoryType, k: int, config: DecayConfig) -> float:
    """Effective decay rate: lambda_base(type) * alpha(k)."""
    base = get_lambda_base(memory_type, config)
    alpha = compute_alpha(k, config.eta)
    return base * alpha


def compute_weight(w0: float, lambda_val: float, delta_t: int) -> float:
    """Synaptic weight after delta_t turns: w0 * e^{-lambda * delta_t}."""
    if lambda_val == 0.0:
        return w0
    return w0 * math.exp(-lambda_val * delta_t)


def decay_record(record: MemoryRecord, current_turn: int, config: DecayConfig) -> float:
    """Compute the decayed weight for a single record at the current turn."""
    if record.memory_type == MemoryType.CORE_ANCHOR:
        return record.weight

    delta_t = current_turn - record.last_reinforced_turn
    if delta_t <= 0:
        return record.weight

    lam = compute_lambda(record.memory_type, record.consolidation_index, config)
    return compute_weight(record.weight, lam, delta_t)


def decay_all(
    records: list[MemoryRecord], current_turn: int, config: DecayConfig
) -> list[tuple[str, float]]:
    """Compute new weights for all mutable records.

    Returns list of (record_id, new_weight) pairs.
    Core anchors are skipped (they never decay).
    """
    results = []
    for record in records:
        if record.memory_type == MemoryType.CORE_ANCHOR:
            continue
        new_weight = decay_record(record, current_turn, config)
        results.append((record.id, new_weight))
    return results
