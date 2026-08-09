"""
Rigorous verification of all mathematical formulas against the spec:
MATH_AND_ARCHETURE_SPEC.md §2.2–2.4

Each test computes expected values by hand and compares to implementation output.
"""

import math

import pytest

from nmafc.engine.decay import compute_alpha, compute_lambda, compute_weight, decay_record
from nmafc.engine.pruning import apply_suppression, detect_override, identify_prunable
from nmafc.engine.reinforcement import reinforce
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryStateUpdate, MemoryType


# ============================================================================
# §2.1 Memory State Representation
# S_i = <v_i, w_i(t), τ_i, k_i>
# ============================================================================

class TestStateRepresentation:
    """Verify MemoryRecord maps to the spec's state tuple S_i."""

    def test_state_tuple_fields_present(self):
        record = MemoryRecord(
            entity_name="test",
            fact_content="fact",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=0.75,
            consolidation_index=3,
        )
        # v_i: embedding is stored externally in LanceDB, not in the record
        # w_i(t): weight ∈ [0, 1.0]
        assert 0.0 <= record.weight <= 1.0
        # τ_i: memory_type ∈ {CoreAnchor, ActiveContext, EphemeralState}
        assert record.memory_type in MemoryType
        # k_i ∈ ℕ₀
        assert record.consolidation_index >= 0


# ============================================================================
# §2.2 Cognitive Decay Equation
# w_i(t) = w_i(t₀) · e^{-λ_i · (t - t₀)}
# where λ_i = λ_base(τ_i) · α(k_i)
# ============================================================================

class TestCognitiveDecayEquation:
    """Verify: w_i(t) = w_i(t₀) · e^{-λ_i · (t - t₀)}"""

    @pytest.fixture
    def config(self) -> DecayConfig:
        return DecayConfig()

    # --- Base decay rates λ_base(τ_i) ---

    def test_lambda_base_core_anchor_is_zero(self, config: DecayConfig):
        """Spec: λ_base(CoreAnchor) = 0.0 ⟹ e^0 = 1 ⟹ w_i(t) = 1.0"""
        assert config.get_lambda_base(MemoryType.CORE_ANCHOR) == 0.0

    def test_lambda_base_active_context(self, config: DecayConfig):
        """Spec: λ_base(ActiveContext) = 0.05"""
        assert config.get_lambda_base(MemoryType.ACTIVE_CONTEXT) == 0.05

    def test_lambda_base_ephemeral(self, config: DecayConfig):
        """Spec: λ_base(EphemeralState) = 0.69"""
        assert config.get_lambda_base(MemoryType.EPHEMERAL_STATE) == 0.69

    # --- Full decay equation numerical checks ---

    def test_core_anchor_weight_never_changes(self, config: DecayConfig):
        """CoreAnchor: λ=0 ⟹ w(t) = w(t₀) for all t."""
        record = MemoryRecord(
            entity_name="identity",
            fact_content="Name is Alice",
            memory_type=MemoryType.CORE_ANCHOR,
            weight=1.0,
            consolidation_index=0,
            last_reinforced_turn=0,
        )
        for t in [1, 10, 100, 1000, 10000]:
            assert decay_record(record, current_turn=t, config=config) == 1.0

    def test_active_context_k0_t10(self, config: DecayConfig):
        """
        ActiveContext, k=0, t₀=0, t=10:
        λ = 0.05 * α(0) = 0.05 * e^{-0.15*0} = 0.05 * 1.0 = 0.05
        w(10) = 1.0 * e^{-0.05 * 10} = e^{-0.5} ≈ 0.60653
        """
        record = MemoryRecord(
            entity_name="schedule",
            fact_content="Meeting at 3PM",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0,
            consolidation_index=0,
            last_reinforced_turn=0,
        )
        expected = math.exp(-0.5)  # 0.60653...
        actual = decay_record(record, current_turn=10, config=config)
        assert abs(actual - expected) < 1e-10

    def test_active_context_k0_t20(self, config: DecayConfig):
        """
        ActiveContext, k=0, t₀=0, t=20:
        w(20) = 1.0 * e^{-0.05 * 20} = e^{-1.0} ≈ 0.36788
        """
        record = MemoryRecord(
            entity_name="task",
            fact_content="Finish report",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0,
            consolidation_index=0,
            last_reinforced_turn=0,
        )
        expected = math.exp(-1.0)
        actual = decay_record(record, current_turn=20, config=config)
        assert abs(actual - expected) < 1e-10

    def test_ephemeral_k0_t1(self, config: DecayConfig):
        """
        EphemeralState, k=0, t₀=0, t=1:
        λ = 0.69 * 1.0 = 0.69
        w(1) = 1.0 * e^{-0.69*1} ≈ 0.50158
        """
        record = MemoryRecord(
            entity_name="mood",
            fact_content="Happy",
            memory_type=MemoryType.EPHEMERAL_STATE,
            weight=1.0,
            consolidation_index=0,
            last_reinforced_turn=0,
        )
        expected = math.exp(-0.69)
        actual = decay_record(record, current_turn=1, config=config)
        assert abs(actual - expected) < 1e-10

    def test_ephemeral_k0_t5(self, config: DecayConfig):
        """
        EphemeralState, k=0, t₀=0, t=5:
        w(5) = e^{-0.69*5} = e^{-3.45} ≈ 0.03175
        This is well below w_prune=0.1
        """
        record = MemoryRecord(
            entity_name="mood",
            fact_content="Tired",
            memory_type=MemoryType.EPHEMERAL_STATE,
            weight=1.0,
            consolidation_index=0,
            last_reinforced_turn=0,
        )
        expected = math.exp(-3.45)
        actual = decay_record(record, current_turn=5, config=config)
        assert abs(actual - expected) < 1e-10
        assert actual < 0.1  # Below prune threshold

    def test_decay_uses_last_reinforced_turn_not_created(self, config: DecayConfig):
        """
        Spec: t - t₀ where t₀ is "last reinforcement or write event"
        Record created at turn 5, last reinforced at turn 8, current turn 12:
        delta_t = 12 - 8 = 4 (NOT 12 - 5)
        """
        record = MemoryRecord(
            entity_name="fact",
            fact_content="Something",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0,
            consolidation_index=0,
            created_at_turn=5,
            last_reinforced_turn=8,
        )
        expected = math.exp(-0.05 * 4)  # delta_t = 12 - 8 = 4
        actual = decay_record(record, current_turn=12, config=config)
        assert abs(actual - expected) < 1e-10

    def test_decay_with_non_unit_initial_weight(self, config: DecayConfig):
        """
        Spec: w_i(t) = w_i(t₀) · e^{-λ·Δt}
        If w₀=0.7, λ=0.05, Δt=6:
        w(t) = 0.7 * e^{-0.3} ≈ 0.7 * 0.74082 ≈ 0.51857
        """
        record = MemoryRecord(
            entity_name="x",
            fact_content="y",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=0.7,
            consolidation_index=0,
            last_reinforced_turn=0,
        )
        expected = 0.7 * math.exp(-0.05 * 6)
        actual = decay_record(record, current_turn=6, config=config)
        assert abs(actual - expected) < 1e-10


# ============================================================================
# §2.2 continued: λ_i = λ_base(τ_i) · α(k_i)
# where α(k_i) = e^{-η·k_i}  (§2.3)
# ============================================================================

class TestLambdaComputation:
    """Verify: λ_i = λ_base(τ_i) · α(k_i) = λ_base · e^{-η·k}"""

    @pytest.fixture
    def config(self) -> DecayConfig:
        return DecayConfig()

    def test_alpha_k0(self, config: DecayConfig):
        """α(0) = e^{-0.15*0} = 1.0"""
        assert compute_alpha(0, config.eta) == 1.0

    def test_alpha_k1(self, config: DecayConfig):
        """α(1) = e^{-0.15*1} = e^{-0.15} ≈ 0.86071"""
        expected = math.exp(-0.15)
        assert abs(compute_alpha(1, config.eta) - expected) < 1e-10

    def test_alpha_k5(self, config: DecayConfig):
        """α(5) = e^{-0.15*5} = e^{-0.75} ≈ 0.47237"""
        expected = math.exp(-0.75)
        assert abs(compute_alpha(5, config.eta) - expected) < 1e-10

    def test_alpha_k10(self, config: DecayConfig):
        """α(10) = e^{-0.15*10} = e^{-1.5} ≈ 0.22313"""
        expected = math.exp(-1.5)
        assert abs(compute_alpha(10, config.eta) - expected) < 1e-10

    def test_lambda_active_k0(self, config: DecayConfig):
        """λ(ActiveContext, k=0) = 0.05 * 1.0 = 0.05"""
        assert compute_lambda(MemoryType.ACTIVE_CONTEXT, 0, config) == 0.05

    def test_lambda_active_k5(self, config: DecayConfig):
        """λ(ActiveContext, k=5) = 0.05 * e^{-0.75} ≈ 0.02362"""
        expected = 0.05 * math.exp(-0.75)
        actual = compute_lambda(MemoryType.ACTIVE_CONTEXT, 5, config)
        assert abs(actual - expected) < 1e-10

    def test_lambda_ephemeral_k3(self, config: DecayConfig):
        """λ(Ephemeral, k=3) = 0.69 * e^{-0.45} ≈ 0.69 * 0.63763 ≈ 0.43996"""
        expected = 0.69 * math.exp(-0.45)
        actual = compute_lambda(MemoryType.EPHEMERAL_STATE, 3, config)
        assert abs(actual - expected) < 1e-10

    def test_consolidated_fact_decays_slower(self, config: DecayConfig):
        """
        A fact retrieved 10 times (k=10) should decay MUCH slower than k=0.

        ActiveContext, Δt=20:
          k=0:  w = e^{-0.05*20} = e^{-1.0} ≈ 0.368
          k=10: λ = 0.05 * e^{-1.5} ≈ 0.01116
                w = e^{-0.01116*20} = e^{-0.2231} ≈ 0.800

        The consolidated fact retains 80% vs 37% — that's the spaced repetition effect.
        """
        record_k0 = MemoryRecord(
            entity_name="a", fact_content="f", memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0, consolidation_index=0, last_reinforced_turn=0,
        )
        record_k10 = MemoryRecord(
            entity_name="b", fact_content="g", memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0, consolidation_index=10, last_reinforced_turn=0,
        )
        w_k0 = decay_record(record_k0, current_turn=20, config=config)
        w_k10 = decay_record(record_k10, current_turn=20, config=config)

        expected_k0 = math.exp(-0.05 * 1.0 * 20)
        expected_k10 = math.exp(-0.05 * math.exp(-1.5) * 20)

        assert abs(w_k0 - expected_k0) < 1e-10
        assert abs(w_k10 - expected_k10) < 1e-10
        assert w_k10 > 0.75  # Well above prune threshold
        assert w_k0 < 0.40   # Significantly decayed


# ============================================================================
# §2.3 Spaced Repetition (Reinforcement on Read)
# 1. w_i(t_now) = 1.0
# 2. k_i ← k_i + 1
# 3. α(k_i) = e^{-η·k_i}  (flattening)
# ============================================================================

class TestSpacedRepetition:
    """Verify the three LTP operations from §2.3."""

    def test_weight_reset_to_one(self):
        """Spec §2.3.1: w_i(t_now) = 1.0"""
        record = MemoryRecord(
            entity_name="x", fact_content="y", memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=0.3, consolidation_index=2, last_reinforced_turn=5,
        )
        reinforced = reinforce(record, current_turn=10)
        assert reinforced.weight == 1.0

    def test_consolidation_increment(self):
        """Spec §2.3.2: k_i ← k_i + 1"""
        record = MemoryRecord(
            entity_name="x", fact_content="y", memory_type=MemoryType.ACTIVE_CONTEXT,
            consolidation_index=7,
        )
        reinforced = reinforce(record, current_turn=10)
        assert reinforced.consolidation_index == 8

    def test_decay_flattening_after_reinforcement(self):
        """
        Spec §2.3.3: After k increments, α(k) shrinks ⟹ λ shrinks ⟹ slower decay.

        Scenario: fact reinforced 5 times, then left for 20 turns.
        λ = 0.05 * e^{-0.15*5} = 0.05 * e^{-0.75} ≈ 0.02362
        w(20) = e^{-0.02362*20} = e^{-0.4724} ≈ 0.6236

        Compare to never-reinforced (k=0):
        w(20) = e^{-0.05*20} = e^{-1.0} ≈ 0.3679
        """
        config = DecayConfig()

        # Simulate 5 reinforcements
        record = MemoryRecord(
            entity_name="x", fact_content="y", memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0, consolidation_index=0, last_reinforced_turn=0,
        )
        for turn in range(1, 6):
            record = reinforce(record, current_turn=turn)

        assert record.consolidation_index == 5
        assert record.weight == 1.0
        assert record.last_reinforced_turn == 5

        # Now decay for 20 turns from last reinforcement (turn 5 → turn 25)
        w = decay_record(record, current_turn=25, config=config)
        lam = 0.05 * math.exp(-0.15 * 5)
        expected = math.exp(-lam * 20)
        assert abs(w - expected) < 1e-10
        assert w > 0.6  # Still healthy after 20 turns


# ============================================================================
# §2.4 State Overrides & Active Synaptic Pruning
# w_old(t_now) = w_old(t⁻) · γ  where γ ∈ [0, 0.1]
# If w_i(t) < w_prune → evict from Hot RAM
# ============================================================================

class TestSynapticPruning:
    """Verify override suppression and pruning constraint."""

    def test_suppression_formula(self):
        """
        Spec §2.4.2: w_old(t_now) = w_old(t⁻) · γ
        With w=1.0, γ=0.1: w_new = 1.0 * 0.1 = 0.1
        With w=0.8, γ=0.1: w_new = 0.8 * 0.1 = 0.08
        """
        r1 = MemoryRecord(
            entity_name="a", fact_content="f", memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0,
        )
        r2 = MemoryRecord(
            entity_name="b", fact_content="g", memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=0.8,
        )
        s1 = apply_suppression(r1, gamma=0.1)
        s2 = apply_suppression(r2, gamma=0.1)
        assert abs(s1.weight - 0.1) < 1e-10
        assert abs(s2.weight - 0.08) < 1e-10

    def test_pruning_constraint(self):
        """
        Spec §2.4.3: If w_i(t) < w_prune (0.1), evict.

        w=0.1 → NOT pruned (spec says < not ≤)
        w=0.09 → pruned
        w=0.05 → pruned
        """
        records = [
            MemoryRecord(
                entity_name="at_threshold", fact_content="f",
                memory_type=MemoryType.ACTIVE_CONTEXT, weight=0.1,
            ),
            MemoryRecord(
                entity_name="below_threshold", fact_content="g",
                memory_type=MemoryType.ACTIVE_CONTEXT, weight=0.09,
            ),
            MemoryRecord(
                entity_name="well_below", fact_content="h",
                memory_type=MemoryType.ACTIVE_CONTEXT, weight=0.05,
            ),
        ]
        prunable = identify_prunable(records, w_prune=0.1)
        # Only records with w < 0.1 (strictly less than)
        assert len(prunable) == 2
        prunable_names = {r.entity_name for r in records if r.id in prunable}
        assert "at_threshold" not in prunable_names
        assert "below_threshold" in prunable_names
        assert "well_below" in prunable_names

    def test_suppression_then_prune_scenario(self):
        """
        Full scenario from spec §2.4:
        1. Old fact at w=1.0
        2. New contradicting fact arrives
        3. Suppression: w_old = 1.0 * 0.1 = 0.1
        4. Pruning check: w=0.1 is NOT < 0.1, so NOT pruned yet
        5. One decay tick later (ephemeral): w = 0.1 * e^{-0.69} ≈ 0.0502
        6. Now w < 0.1 → pruned
        """
        config = DecayConfig()

        old_record = MemoryRecord(
            entity_name="medication_schedule", fact_content="Aspirin at 9AM",
            memory_type=MemoryType.ACTIVE_CONTEXT, weight=1.0,
            consolidation_index=0, last_reinforced_turn=10,
        )

        # Step 3: Suppress
        suppressed = apply_suppression(old_record, gamma=config.gamma)
        assert abs(suppressed.weight - 0.1) < 1e-10

        # Step 4: Not prunable yet
        assert suppressed.weight >= config.w_prune
        prunable = identify_prunable([suppressed], config.w_prune)
        assert len(prunable) == 0

        # Step 5: After 1 more decay tick
        suppressed_record = suppressed.model_copy(update={"last_reinforced_turn": 10})
        w_after_decay = decay_record(suppressed_record, current_turn=11, config=config)
        # w = 0.1 * e^{-0.05*1} ≈ 0.0951
        expected = 0.1 * math.exp(-0.05 * 1)
        assert abs(w_after_decay - expected) < 1e-10
        assert w_after_decay < config.w_prune  # Now prunable

    def test_override_detection_entity_match(self):
        """
        Spec §2.4.1: system identifies existing node sharing same entity identifier.
        """
        existing = [
            MemoryRecord(
                entity_name="medication_schedule", fact_content="9AM",
                memory_type=MemoryType.ACTIVE_CONTEXT,
            ),
            MemoryRecord(
                entity_name="unrelated", fact_content="other",
                memory_type=MemoryType.ACTIVE_CONTEXT,
            ),
        ]
        update = MemoryStateUpdate(
            entity_name="medication_schedule",
            fact_content="11AM",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            overrides_entity="medication_schedule",
        )
        targets = detect_override(update, existing)
        assert len(targets) == 1
        assert targets[0].entity_name == "medication_schedule"


# ============================================================================
# Edge cases and boundary conditions
# ============================================================================

class TestEdgeCases:
    def test_weight_cannot_go_negative(self):
        """Exponential decay is always positive: e^{-x} > 0 for all x."""
        config = DecayConfig()
        record = MemoryRecord(
            entity_name="x", fact_content="y", memory_type=MemoryType.EPHEMERAL_STATE,
            weight=0.001, consolidation_index=0, last_reinforced_turn=0,
        )
        w = decay_record(record, current_turn=1000, config=config)
        assert w > 0.0

    def test_multiple_suppressions_compound(self):
        """If suppressed twice (shouldn't normally happen), weight compounds."""
        record = MemoryRecord(
            entity_name="x", fact_content="y", memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0,
        )
        s1 = apply_suppression(record, gamma=0.1)
        s2 = apply_suppression(s1, gamma=0.1)
        assert abs(s2.weight - 0.01) < 1e-10

    def test_reinforcement_after_decay_resets_fully(self):
        """After decay drops weight, reinforcement brings it back to 1.0."""
        config = DecayConfig()
        record = MemoryRecord(
            entity_name="x", fact_content="y", memory_type=MemoryType.ACTIVE_CONTEXT,
            weight=1.0, consolidation_index=0, last_reinforced_turn=0,
        )
        # Decay to 0.6
        decayed_weight = decay_record(record, current_turn=10, config=config)
        assert decayed_weight < 0.7

        # Reinforce
        record_decayed = record.model_copy(update={"weight": decayed_weight})
        reinforced = reinforce(record_decayed, current_turn=10)
        assert reinforced.weight == 1.0
        assert reinforced.consolidation_index == 1
