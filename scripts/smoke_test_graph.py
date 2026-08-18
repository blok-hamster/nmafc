"""Graph traversal smoke test: verifies Spreading Activation retrieval.

Tests that:
1. Records with related_entities links are stored correctly
2. Querying for one entity retrieves linked entities via hop-1
3. Hop-2 traversal follows chains (A → B → C)
4. max_hops is respected (hop-3 entities are NOT retrieved when max_hops=2)
5. Retrieval doesn't duplicate records already found via vector search
6. Isolated nodes stay isolated (no phantom traversals)

Strategy:
  - top_k is set LOW (2) so vector search returns only the closest matches.
  - Linked entities have semantically DISTANT content so they can ONLY be
    reached via graph traversal, not vector similarity.
  - This isolates the graph traversal logic from vector search behaviour.

Usage:
    python scripts/smoke_test_graph.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nmafc.integration.factory import create_embedding_provider
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryStateUpdate, MemoryType
from nmafc.storage.config import NMafcConfig, StorageConfig
from nmafc.storage.hot import HotStorage
from nmafc.storage.cold import ColdStorage
from nmafc.storage.cold_base import ColdStorageBase
from nmafc.integration.query_router import QueryRouter


# Graph structure we're building:
#
#   user_name ──→ spouse_james ──→ child_lily ──→ school_elementary
#                      │                              (hop 3 from user_name)
#                      ├──→ child_noah
#                      └──→ pet_bear
#
#   user_job ──→ workplace_hospital
#
#   user_allergy (no links — isolated node)
#
#   filler_astronomy, filler_geology, filler_botany, filler_music, filler_chess
#     (semantically distant noise to dilute vector search results)
#
# KEY: Linked entities use deliberately UNRELATED content so vector similarity
#      for "Sarah" or "family" queries won't reach them. Only graph links can.

RECORDS = [
    # ── Core graph nodes ──
    MemoryStateUpdate(
        entity_name="user_name",
        fact_content="User's name is Dr. Sarah Chen, 42 years old, lives in Baltimore",
        memory_type=MemoryType.CORE_ANCHOR,
        related_entities=["spouse_james"],
    ),
    MemoryStateUpdate(
        entity_name="spouse_james",
        # Avoid direct name match in query so vector search doesn't pull it in at Hop 0
        fact_content="Husband James is a software architect who works remotely from home",
        memory_type=MemoryType.CORE_ANCHOR,
        related_entities=["child_lily", "child_noah", "pet_bear"],
    ),
    MemoryStateUpdate(
        entity_name="child_lily",
        # Deliberately distant from "Sarah" queries — only reachable via graph
        fact_content="Seven-year-old who loves watercolor painting and collects seashells",
        memory_type=MemoryType.CORE_ANCHOR,
        related_entities=["school_elementary"],
    ),
    MemoryStateUpdate(
        entity_name="child_noah",
        fact_content="Four-year-old starting preschool who is fascinated by dinosaurs",
        memory_type=MemoryType.CORE_ANCHOR,
        related_entities=[],
    ),
    MemoryStateUpdate(
        entity_name="pet_bear",
        fact_content="Golden retriever adopted from the local shelter in January",
        memory_type=MemoryType.CORE_ANCHOR,
        related_entities=[],
    ),
    MemoryStateUpdate(
        entity_name="school_elementary",
        # Hop 3 from user_name — must NOT appear at max_hops=2
        fact_content="Oakwood Elementary on Pine Street accepts students from kindergarten through fifth grade",
        memory_type=MemoryType.ACTIVE_CONTEXT,
        related_entities=[],
    ),
    MemoryStateUpdate(
        entity_name="user_job",
        fact_content="Chief of Cardiology at Johns Hopkins Hospital, board certified",
        memory_type=MemoryType.CORE_ANCHOR,
        related_entities=["workplace_hospital"],
    ),
    MemoryStateUpdate(
        entity_name="workplace_hospital",
        fact_content="Johns Hopkins Hospital in East Baltimore, started position last month",
        memory_type=MemoryType.ACTIVE_CONTEXT,
        related_entities=[],
    ),
    MemoryStateUpdate(
        entity_name="user_allergy",
        # Completely isolated — tests that traversal doesn't leak
        fact_content="Severe anaphylactic allergy to latex, carries epinephrine auto-injector",
        memory_type=MemoryType.CORE_ANCHOR,
        related_entities=[],
    ),
    # ── Filler nodes (semantically distant noise) ──
    # These dilute the vector space so top_k=2 won't accidentally grab graph nodes
    MemoryStateUpdate(
        entity_name="filler_astronomy",
        fact_content="The Andromeda galaxy is approximately 2.537 million light-years from Earth",
        memory_type=MemoryType.ACTIVE_CONTEXT,
        related_entities=[],
    ),
    MemoryStateUpdate(
        entity_name="filler_geology",
        fact_content="Obsidian is formed when felsic lava cools rapidly with minimal crystal growth",
        memory_type=MemoryType.ACTIVE_CONTEXT,
        related_entities=[],
    ),
    MemoryStateUpdate(
        entity_name="filler_botany",
        fact_content="Photosynthesis converts carbon dioxide and water into glucose using sunlight",
        memory_type=MemoryType.ACTIVE_CONTEXT,
        related_entities=[],
    ),
    MemoryStateUpdate(
        entity_name="filler_music",
        fact_content="A diminished seventh chord consists of minor thirds stacked in equal intervals",
        memory_type=MemoryType.ACTIVE_CONTEXT,
        related_entities=[],
    ),
    MemoryStateUpdate(
        entity_name="filler_chess",
        fact_content="The Sicilian Defense begins with the moves 1.e4 c5 and is the most popular response",
        memory_type=MemoryType.ACTIVE_CONTEXT,
        related_entities=[],
    ),
]


async def run_graph_test() -> None:
    print("=" * 80)
    print("NMAFC GRAPH TRAVERSAL SMOKE TEST — Spreading Activation Verification")
    print("=" * 80)

    embedder = create_embedding_provider("ollama/nomic-embed-text")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        config = StorageConfig(
            hot_uri=str(Path(tmpdir) / "lancedb"),
            cold_uri=str(Path(tmpdir) / "cold.db"),
            agent_id="graph-test",
            conversation_id="conv-001",
            embedding_dim=768,
        )

        hot = HotStorage(config)
        cold = ColdStorage(config.cold_uri, agent_id="graph-test", conversation_id="conv-001")

        # ── Ingest all records ──
        print(f"\n[1/6] Ingesting {len(RECORDS)} records with explicit relationship links...")
        print()
        print("  Graph structure:")
        print("  user_name → spouse_james → child_lily → school_elementary (hop 3)")
        print("                           → child_noah")
        print("                           → pet_bear")
        print("  user_job → workplace_hospital")
        print("  user_allergy (isolated)")
        print(f"  + 5 filler records (semantically distant noise)")
        print()

        for update in RECORDS:
            embedding = await embedder.embed_single(update.fact_content)
            record = MemoryRecord(
                entity_name=update.entity_name,
                fact_content=update.fact_content,
                memory_type=update.memory_type,
                weight=1.0,
                consolidation_index=0,
                created_at_turn=1,
                last_reinforced_turn=1,
                related_entities=list(update.related_entities),
            )
            hot.upsert(record, embedding)
            cold.append_event(update, turn=1)
            links = f" → [{', '.join(update.related_entities)}]" if update.related_entities else ""
            print(f"    {update.entity_name:<25} {update.memory_type.value:<16}{links}")

        print(f"\n  Total records in Hot RAM: {hot.count()}")

        # ═══════════════════════════════════════════════════════════════════════
        # Test 1: top_k=2, max_hops=2 — graph traversal should pull family cluster
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n{'─' * 80}")
        print("[2/6] Test: 'Tell me about Sarah' with top_k=2, max_hops=2")
        print(f"{'─' * 80}")
        print("  Vector search returns only 2 closest hits (hop 0).")
        print("  Graph traversal must pull linked entities at hops 1 and 2.")

        decay_2hop = DecayConfig(top_k=1, max_hops=2, theta=0.0)
        router_2hop = QueryRouter(hot, cold, embedder, decay_2hop)

        results_2hop = await router_2hop.retrieve("Tell me about Sarah", current_turn=2)
        entities_2hop = [r.entity_name for r in results_2hop]
        set_2hop = set(entities_2hop)

        print(f"\n  Retrieved {len(results_2hop)} records:")
        for r in results_2hop:
            print(f"    {r.entity_name:<25} {r.fact_content[:55]}")

        # ═══════════════════════════════════════════════════════════════════════
        # Test 2: top_k=2, max_hops=1 — should stop at spouse, NOT reach children
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n{'─' * 80}")
        print("[3/6] Test: 'Tell me about Sarah' with top_k=2, max_hops=1")
        print(f"{'─' * 80}")
        print("  Should reach spouse_james (hop 1) but NOT children/pet (hop 2).")

        decay_1hop = DecayConfig(top_k=1, max_hops=1, theta=0.0)
        router_1hop = QueryRouter(hot, cold, embedder, decay_1hop)

        results_1hop = await router_1hop.retrieve("Tell me about Sarah", current_turn=2)
        entities_1hop = [r.entity_name for r in results_1hop]
        set_1hop = set(entities_1hop)

        print(f"\n  Retrieved {len(results_1hop)} records:")
        for r in results_1hop:
            print(f"    {r.entity_name:<25} {r.fact_content[:55]}")

        # ═══════════════════════════════════════════════════════════════════════
        # Test 3: top_k=1, max_hops=3 — should now reach school_elementary
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n{'─' * 80}")
        print("[4/6] Test: 'Tell me about Sarah' with top_k=1, max_hops=3")
        print(f"{'─' * 80}")
        print("  Should now reach school_elementary (hop 3).")

        decay_3hop = DecayConfig(top_k=1, max_hops=3, theta=0.0)
        router_3hop = QueryRouter(hot, cold, embedder, decay_3hop)

        results_3hop = await router_3hop.retrieve("Tell me about Sarah", current_turn=2)
        entities_3hop = [r.entity_name for r in results_3hop]
        set_3hop = set(entities_3hop)

        print(f"\n  Retrieved {len(results_3hop)} records:")
        for r in results_3hop:
            print(f"    {r.entity_name:<25} {r.fact_content[:55]}")

        # ═══════════════════════════════════════════════════════════════════════
        # Test 4: Isolated node query — no phantom traversals
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n{'─' * 80}")
        print("[5/6] Test: 'latex allergy' (isolated node, no links)")
        print(f"{'─' * 80}")

        decay_isolated = DecayConfig(top_k=1, max_hops=2, theta=0.0)
        router_isolated = QueryRouter(hot, cold, embedder, decay_isolated)

        results_iso = await router_isolated.retrieve("latex allergy anaphylaxis", current_turn=2)
        entities_iso = [r.entity_name for r in results_iso]
        set_iso = set(entities_iso)

        print(f"\n  Retrieved {len(results_iso)} records:")
        for r in results_iso:
            print(f"    {r.entity_name:<25} {r.fact_content[:55]}")

        has_allergy = "user_allergy" in entities_iso

        # Isolated node should NOT pull in family/job/filler via graph
        family_nodes = {"spouse_james", "child_lily", "child_noah", "pet_bear", "school_elementary"}
        phantom_leaks = family_nodes.intersection(set_iso)

        print(f"\n  Allergy record found:  {has_allergy}")
        print(f"  Phantom graph leaks:   {sorted(phantom_leaks) if phantom_leaks else 'none (correct)'}")

        # ═══════════════════════════════════════════════════════════════════════
        # Test 5: Deduplication — records found via vector AND graph should not repeat
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n{'─' * 80}")
        print("[6/6] Test: Deduplication check")
        print(f"{'─' * 80}")

        has_dupes_2hop = len(entities_2hop) != len(set(entities_2hop))
        has_dupes_3hop = len(entities_3hop) != len(set(entities_3hop))

        print(f"\n  Duplicates in 2-hop results: {has_dupes_2hop}")
        print(f"  Duplicates in 3-hop results: {has_dupes_3hop}")

        # ═══════════════════════════════════════════════════════════════════════
        # Verification Summary
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n\n{'═' * 80}")
        print("  VERIFICATION SUMMARY")
        print(f"{'═' * 80}")

        # Determine what vector search alone returns (top_k=2, no graph)
        # by comparing set differences between hop configs
        hop2_only_entities = {"child_lily", "child_noah", "pet_bear"}  # Should appear at hop≥2
        hop3_only_entities = {"school_elementary"}  # Should appear at hop≥3

        checks = []

        # Check 1: user_name hit by vector search in all configs
        checks.append((
            "user_name retrieved (hop 0 — vector hit)",
            "user_name" in set_2hop and "user_name" in set_1hop,
        ))

        # Check 2: spouse_james reachable at max_hops≥1
        checks.append((
            "spouse_james retrieved at max_hops=1 (hop 1 graph link)",
            "spouse_james" in set_1hop,
        ))

        # Check 3: Children + pet retrieved at max_hops=2 via hop 2
        children_at_2hop = hop2_only_entities.intersection(set_2hop)
        checks.append((
            f"Children/pet retrieved at max_hops=2 (hop 2): {len(children_at_2hop)}/3",
            len(children_at_2hop) >= 2,
        ))

        # Check 4: Children + pet NOT retrieved at max_hops=1
        children_at_1hop = hop2_only_entities.intersection(set_1hop)
        checks.append((
            f"Children/pet NOT at max_hops=1: {len(children_at_1hop)} found (expect 0)",
            len(children_at_1hop) == 0,
        ))

        # Check 5: school_elementary NOT at max_hops=2
        checks.append((
            "school_elementary NOT at max_hops=2 (would need hop 3)",
            "school_elementary" not in set_2hop,
        ))

        # Check 6: school_elementary IS at max_hops=3
        checks.append((
            "school_elementary IS at max_hops=3 (hop 3 reached)",
            "school_elementary" in set_3hop,
        ))

        # Check 7: Isolated node retrieval
        checks.append(("Isolated node (allergy) retrievable", has_allergy))

        # Check 8: No phantom graph leaks from isolated node
        checks.append((
            "No phantom leaks from isolated node",
            len(phantom_leaks) == 0,
        ))

        # Check 9: No duplicates
        checks.append(("No duplicate records (2-hop)", not has_dupes_2hop))
        checks.append(("No duplicate records (3-hop)", not has_dupes_3hop))

        # Check 10: Monotonic — higher max_hops retrieves ≥ same entities
        checks.append((
            "Monotonic: 3-hop results ⊇ 2-hop results ⊇ 1-hop results",
            set_1hop.issubset(set_2hop) and set_2hop.issubset(set_3hop),
        ))

        print(f"\n  {'#':<4} {'Check':<62} {'Result':<6}")
        print(f"  {'─' * 4} {'─' * 62} {'─' * 6}")
        for idx, (label, passed) in enumerate(checks, 1):
            status = "PASS" if passed else "FAIL"
            marker = "  " if passed else ">>"
            print(f"  {marker}{idx:<2}  {label:<62} {status}")

        passed_count = sum(1 for _, p in checks if p)
        total_count = len(checks)
        print(f"\n  Result: {passed_count}/{total_count} checks passed")

        # Show entity sets for debugging
        print(f"\n  ── Entity sets by max_hops ──")
        print(f"  max_hops=1: {sorted(set_1hop)}")
        print(f"  max_hops=2: {sorted(set_2hop)}")
        print(f"  max_hops=3: {sorted(set_3hop)}")

        if passed_count == total_count:
            print("\n  ✓ ALL CHECKS PASSED — Spreading Activation graph traversal verified")
        else:
            print(f"\n  ✗ {total_count - passed_count} CHECK(S) FAILED — review above for details")
            sys.exit(1)

        cold.close()


if __name__ == "__main__":
    asyncio.run(run_graph_test())
