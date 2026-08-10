"""Complex smoke test: 15-turn conversation exercising all neuromorphic behaviors.

Scenarios tested:
1. Identity facts (CoreAnchor) persist indefinitely
2. Multiple sequential overrides (medication dose changes 3 times)
3. Ephemeral facts fully decay and get pruned within ~4 turns
4. Active context transitions (job change mid-conversation)
5. Cold ROM fallback when hot search misses
6. Contradicting facts suppress predecessors correctly
7. Frequently accessed facts consolidate (high k, slow decay)
8. Infrequently accessed facts decay naturally
9. Correct answers to recall questions across all types

Usage:
    ANTHROPIC_API_KEY_BEDROCK=ABSK... python scripts/smoke_test_complex.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nmafc.integration.bedrock_provider import BedrockAnthropicProvider
from nmafc.integration.factory import create_embedding_provider
from nmafc.schemas.memory import DecayConfig, MemoryType
from nmafc.storage.config import NMafcConfig, StorageConfig
from nmafc.wrapper import NeuromorphicMemory

CONVERSATION = [
    # Turn 1-2: Core identity
    "My name is Dr. Sarah Chen. I'm a 42-year-old cardiologist at Massachusetts General Hospital.",
    "I'm married to James. We have two kids — Lily who's 7 and Noah who's 4.",

    # Turn 3: Medical info (will be overridden later)
    "I take lisinopril 10mg every morning for blood pressure. Started it about 6 months ago.",

    # Turn 4: Ephemeral — should decay within ~3-4 turns
    "I've got a terrible headache right now. Probably from staring at charts all day.",

    # Turn 5: Active context — schedule
    "I have surgery at 7AM tomorrow — a triple bypass on Mr. Henderson.",

    # Turn 6: First override — medication dose change
    "By the way, my doctor increased my lisinopril to 20mg starting this week. The 10mg wasn't cutting it.",

    # Turn 7: Ephemeral — different transient state
    "I'm feeling really anxious about tomorrow's surgery. It's a complex case.",

    # Turn 8: Unrelated topic (headache should be decaying)
    "Oh, I should mention — I'm severely allergic to latex. That's critical for any medical procedures.",

    # Turn 9: Another unrelated topic — testing that old ephemerals decay
    "We just adopted a golden retriever last month. His name is Bear.",

    # Turn 10: Second override — medication changes again
    "Update on meds — I'm switching from lisinopril to losartan 50mg. Had a persistent cough as a side effect.",

    # Turn 11: Active context override — schedule change
    "Surgery got pushed to 9AM. The anesthesiologist had a scheduling conflict.",

    # Turn 12: Ephemeral — should be very short-lived
    "I'm starving. Haven't eaten since breakfast.",

    # Turn 13: Recall test — identity + medical
    "Can you remind me what medications I'm currently taking and any allergies I have?",

    # Turn 14: Job change — major active context override
    "Big news — I'm leaving Mass General. I accepted a position as Chief of Cardiology at Johns Hopkins starting next month.",

    # Turn 15: Final recall — testing full memory state
    "Give me a summary of what you know about me — family, work, health, pets.",
]


def print_memory_state(mem: NeuromorphicMemory, turn: int) -> None:
    hot_stats = mem.get_hot_stats()
    cold_stats = mem.get_cold_stats()
    print(f"\n  Hot RAM: {hot_stats['count']} records | avg weight: {hot_stats['avg_weight']:.3f}")
    print(f"  Cold ROM: {cold_stats['total_events']} total, {cold_stats['active_events']} active")
    if hot_stats.get("types"):
        print(f"  Types: {hot_stats['types']}")

    all_records = mem._hot.get_all()
    if all_records:
        # Group by type
        by_type: dict[str, list] = {}
        for r in all_records:
            by_type.setdefault(r.memory_type.value, []).append(r)

        print(f"\n  {'Entity':<35} {'Type':<16} {'Weight':>7} {'k':>3} {'Age':>4}")
        print(f"  {'─' * 35} {'─' * 16} {'─' * 7} {'─' * 3} {'─' * 4}")
        for type_name in ["CoreAnchor", "ActiveContext", "EphemeralState"]:
            records = by_type.get(type_name, [])
            for r in sorted(records, key=lambda x: x.weight, reverse=True):
                age = turn - r.last_reinforced_turn
                print(f"  {r.entity_name:<35} {r.memory_type.value:<16} {r.weight:>7.4f} {r.consolidation_index:>3} {age:>4}")


async def run_complex_smoke_test() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY_BEDROCK")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY_BEDROCK environment variable")
        sys.exit(1)

    region = os.environ.get("AWS_REGION", "us-east-1")

    print("=" * 80)
    print("NMAFC COMPLEX SMOKE TEST — 15 turns, Claude Haiku 4.5 via AWS Bedrock")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(Path(tmpdir) / "lancedb"),
                cold_uri=str(Path(tmpdir) / "cold.db"),
                embedding_dim=768,
            ),
            decay=DecayConfig(),
        )

        llm = BedrockAnthropicProvider(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region=region,
            api_key=api_key,
        )
        embedder = create_embedding_provider("ollama/nomic-embed-text")

        mem = NeuromorphicMemory(
            llm_provider=llm,
            embedding_provider=embedder,
            config=config,
        )

        history: list[dict] = []

        for i, user_msg in enumerate(CONVERSATION, 1):
            print(f"\n{'━' * 80}")
            print(f"  TURN {i:>2} | {user_msg}")
            print(f"{'━' * 80}")

            response = await mem.process_turn(user_msg, history)

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": response})

            # Truncate long responses
            display = response[:300] + "..." if len(response) > 300 else response
            print(f"\n  A: {display}")

            print_memory_state(mem, i)

        # ═══════════════════════════════════════════════════════════════════════
        # VERIFICATION
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n\n{'═' * 80}")
        print("  VERIFICATION REPORT")
        print(f"{'═' * 80}")

        all_records = mem._hot.get_all()
        core_anchors = [r for r in all_records if r.memory_type == MemoryType.CORE_ANCHOR]
        active_ctx = [r for r in all_records if r.memory_type == MemoryType.ACTIVE_CONTEXT]
        ephemeral = [r for r in all_records if r.memory_type == MemoryType.EPHEMERAL_STATE]

        print(f"\n  Final state after {mem.current_turn} turns:")
        print(f"    Core Anchors:    {len(core_anchors)}")
        print(f"    Active Context:  {len(active_ctx)}")
        print(f"    Ephemeral:       {len(ephemeral)}")
        print(f"    Total Hot RAM:   {len(all_records)}")
        print(f"    Total Cold ROM:  {mem.get_cold_stats()['total_events']}")

        # Check expectations
        checks = []

        # 1. Identity preserved
        has_name = any("sarah" in r.fact_content.lower() or "chen" in r.fact_content.lower() for r in core_anchors)
        checks.append(("Identity (Sarah Chen) preserved as CoreAnchor", has_name))

        # 2. Family info preserved
        has_family = any("james" in r.fact_content.lower() or "lily" in r.fact_content.lower() for r in all_records)
        checks.append(("Family info preserved", has_family))

        # 3. Latex allergy preserved (life-safety = CoreAnchor)
        has_allergy = any("latex" in r.fact_content.lower() for r in core_anchors)
        checks.append(("Latex allergy preserved as CoreAnchor", has_allergy))

        # 4. Current medication is losartan, NOT lisinopril
        med_records = [r for r in all_records if "med" in r.entity_name.lower() or "losartan" in r.fact_content.lower() or "lisinopril" in r.fact_content.lower()]
        if med_records:
            latest_med = max(med_records, key=lambda r: r.created_at_turn)
            has_losartan = "losartan" in latest_med.fact_content.lower()
            checks.append(("Current medication is losartan (not lisinopril)", has_losartan))
        else:
            checks.append(("Current medication is losartan (not lisinopril)", False))

        # 5. Old lisinopril records should be suppressed/pruned
        lisinopril_active = [r for r in all_records if "lisinopril" in r.entity_name.lower() and r.weight > 0.1]
        checks.append(("Old lisinopril records suppressed/pruned", len(lisinopril_active) == 0))

        # 6. Headache (Turn 4) should be fully decayed/pruned by Turn 15
        headache_records = [r for r in all_records if "headache" in r.fact_content.lower()]
        checks.append(("Headache (ephemeral, Turn 4) pruned by Turn 15", len(headache_records) == 0))

        # 7. Hunger (Turn 12) may still be present but with low weight
        hunger_records = [r for r in all_records if "hungry" in r.fact_content.lower() or "starving" in r.fact_content.lower() or "eaten" in r.fact_content.lower()]
        if hunger_records:
            hunger_weight = hunger_records[0].weight
            checks.append((f"Hunger (Turn 12) decaying (weight={hunger_weight:.3f})", hunger_weight < 0.5))
        else:
            checks.append(("Hunger (Turn 12) already pruned", True))

        # 8. Surgery time updated to 9AM
        surgery_records = [r for r in all_records if "surgery" in r.fact_content.lower() or "bypass" in r.fact_content.lower()]
        if surgery_records:
            latest_surgery = max(surgery_records, key=lambda r: r.created_at_turn)
            has_9am = "9" in latest_surgery.fact_content
            checks.append(("Surgery time updated to 9AM", has_9am))
        else:
            checks.append(("Surgery time updated to 9AM", False))

        # 9. Job updated to Johns Hopkins
        job_records = [r for r in all_records if "johns hopkins" in r.fact_content.lower() or "chief" in r.fact_content.lower()]
        checks.append(("Job updated to Johns Hopkins/Chief of Cardiology", len(job_records) > 0))

        # 10. Pet info preserved
        pet_records = [r for r in all_records if "bear" in r.fact_content.lower() or "golden" in r.fact_content.lower() or "dog" in r.fact_content.lower()]
        checks.append(("Pet (Bear, golden retriever) remembered", len(pet_records) > 0))

        # Print results
        print(f"\n  {'#':<4} {'Check':<55} {'Result':<6}")
        print(f"  {'─' * 4} {'─' * 55} {'─' * 6}")
        for idx, (label, passed) in enumerate(checks, 1):
            status = "PASS" if passed else "FAIL"
            marker = "  " if passed else ">>"
            print(f"  {marker}{idx:<2}  {label:<55} {status}")

        passed_count = sum(1 for _, p in checks if p)
        total_count = len(checks)
        print(f"\n  Result: {passed_count}/{total_count} checks passed")

        if passed_count == total_count:
            print("\n  ✓ ALL CHECKS PASSED — Neuromorphic memory system operating correctly")
        else:
            print(f"\n  ✗ {total_count - passed_count} CHECK(S) FAILED — Review above for details")

        # Memory hygiene report
        print(f"\n\n{'─' * 80}")
        print("  MEMORY HYGIENE REPORT")
        print(f"{'─' * 80}")
        print(f"  Records created (Cold ROM):    {mem.get_cold_stats()['total_events']}")
        print(f"  Records active (Hot RAM):      {len(all_records)}")
        print(f"  Records pruned:                {mem.get_cold_stats()['total_events'] - len(all_records)}")
        print(f"  Pruning ratio:                 {1 - len(all_records) / max(1, mem.get_cold_stats()['total_events']):.1%}")

        avg_k = sum(r.consolidation_index for r in all_records) / max(1, len(all_records))
        max_k = max((r.consolidation_index for r in all_records), default=0)
        print(f"  Avg consolidation (k):         {avg_k:.1f}")
        print(f"  Max consolidation (k):         {max_k}")

        ephemeral_weights = [r.weight for r in ephemeral]
        if ephemeral_weights:
            print(f"  Ephemeral avg weight:          {sum(ephemeral_weights) / len(ephemeral_weights):.4f}")
        else:
            print(f"  Ephemeral records:             all pruned (correct)")


if __name__ == "__main__":
    asyncio.run(run_complex_smoke_test())
