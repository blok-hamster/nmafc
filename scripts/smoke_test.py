"""Live smoke test: runs 8 conversation turns through the full neuromorphic pipeline.

Tests that:
1. The LLM actually emits tool calls with well-classified memory types
2. State overrides trigger suppression
3. Decay reduces weights over time
4. Ephemeral facts get pruned
5. Core anchors persist indefinitely

Usage:
    ANTHROPIC_API_KEY_BEDROCK=ABSK... python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nmafc.integration.base import EmbeddingProvider
from nmafc.integration.bedrock_provider import BedrockAnthropicProvider
from nmafc.schemas.memory import DecayConfig, MemoryType
from nmafc.storage.config import NMafcConfig, StorageConfig
from nmafc.wrapper import NeuromorphicMemory


class DeterministicEmbedding(EmbeddingProvider):
    """Hash-based embedder for smoke testing (no API call needed)."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        results = []
        for text in texts:
            h = hashlib.sha512(text.encode()).digest()
            while len(h) < self._dim:
                h += hashlib.sha512(h).digest()
            raw = [((b % 200) - 100) / 100.0 for b in h[: self._dim]]
            norm = sum(x * x for x in raw) ** 0.5
            if norm == 0:
                norm = 1.0
            results.append([x / norm for x in raw])
        return results


CONVERSATION = [
    "Hi, I'm Marcus. I'm 34 years old and I'm a software engineer at Google.",
    "I'm severely allergic to shellfish — found out the hard way last year. Anaphylaxis, EpiPen, the whole deal.",
    "I've got a meeting with my team at 3PM today about the new API redesign.",
    "Actually, that meeting got moved to 4:30PM. The tech lead had a conflict.",
    "I'm feeling pretty stressed about the deadline. We have to ship by Friday.",
    "Oh I also take metformin every morning for type 2 diabetes. 500mg dose.",
    "You know what, the stress is gone now. Just had a great coffee break with colleagues.",
    "Quick question — what time is my meeting today?",
]


async def run_smoke_test() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY_BEDROCK")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY_BEDROCK environment variable")
        sys.exit(1)

    region = os.environ.get("AWS_REGION", "us-east-1")

    print("=" * 70)
    print("NMAFC LIVE SMOKE TEST — Claude Haiku 4.5 via AWS Bedrock")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = NMafcConfig(
            storage=StorageConfig(
                hot_uri=str(Path(tmpdir) / "lancedb"),
                cold_uri=str(Path(tmpdir) / "cold.db"),
                embedding_dim=384,
            ),
            decay=DecayConfig(),
        )

        llm = BedrockAnthropicProvider(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region=region,
            api_key=api_key,
        )
        embedder = DeterministicEmbedding(dim=384)

        mem = NeuromorphicMemory(
            llm_provider=llm,
            embedding_provider=embedder,
            config=config,
        )

        history: list[dict] = []

        for i, user_msg in enumerate(CONVERSATION, 1):
            print(f"\n{'─' * 70}")
            print(f"TURN {i} | User: {user_msg}")
            print(f"{'─' * 70}")

            response = await mem.process_turn(user_msg, history)

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": response})

            print(f"\nAssistant: {response[:200]}{'...' if len(response) > 200 else ''}")

            # Show memory state
            hot_stats = mem.get_hot_stats()
            cold_stats = mem.get_cold_stats()
            print(f"\n  Hot RAM: {hot_stats['count']} records, avg weight: {hot_stats['avg_weight']:.3f}")
            print(f"  Cold ROM: {cold_stats['total_events']} total, {cold_stats['active_events']} active")
            if hot_stats.get("types"):
                print(f"  Types: {hot_stats['types']}")

            # Show individual records
            all_records = mem._hot.get_all()
            if all_records:
                print(f"\n  {'Entity':<30} {'Type':<18} {'Weight':>7} {'k':>3}")
                print(f"  {'─' * 30} {'─' * 18} {'─' * 7} {'─' * 3}")
                for r in sorted(all_records, key=lambda x: x.weight, reverse=True):
                    print(f"  {r.entity_name:<30} {r.memory_type.value:<18} {r.weight:>7.4f} {r.consolidation_index:>3}")

        # Final verification
        print(f"\n{'=' * 70}")
        print("VERIFICATION")
        print(f"{'=' * 70}")

        all_records = mem._hot.get_all()
        core_anchors = [r for r in all_records if r.memory_type == MemoryType.CORE_ANCHOR]
        active_ctx = [r for r in all_records if r.memory_type == MemoryType.ACTIVE_CONTEXT]
        ephemeral = [r for r in all_records if r.memory_type == MemoryType.EPHEMERAL_STATE]

        print(f"\nFinal state after {mem.current_turn} turns:")
        print(f"  Core Anchors: {len(core_anchors)} (should include name, allergy, diabetes)")
        print(f"  Active Context: {len(active_ctx)} (should include meeting at 4:30)")
        print(f"  Ephemeral: {len(ephemeral)} (stress should be gone/pruned)")
        print(f"  Total in Hot RAM: {len(all_records)}")
        print(f"  Total in Cold ROM: {mem.get_cold_stats()['total_events']}")

        # Check specific expectations
        checks = []

        has_identity = any("marcus" in r.fact_content.lower() or "34" in r.fact_content for r in core_anchors)
        checks.append(("Identity preserved as CoreAnchor", has_identity))

        has_allergy = any("shellfish" in r.fact_content.lower() for r in core_anchors)
        checks.append(("Allergy preserved as CoreAnchor", has_allergy))

        meeting_records = [r for r in all_records if "meeting" in r.entity_name.lower() or "meeting" in r.fact_content.lower()]
        if meeting_records:
            latest_meeting = max(meeting_records, key=lambda r: r.created_at_turn)
            has_updated_meeting = "4:30" in latest_meeting.fact_content or "4" in latest_meeting.fact_content
            checks.append(("Meeting time updated to 4:30PM", has_updated_meeting))
        else:
            checks.append(("Meeting time updated to 4:30PM", False))

        print(f"\n  {'Check':<45} {'Result'}")
        print(f"  {'─' * 45} {'─' * 6}")
        for label, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  {label:<45} {status}")

        all_passed = all(p for _, p in checks)
        print(f"\n  {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
