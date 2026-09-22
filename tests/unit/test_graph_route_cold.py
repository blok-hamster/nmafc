import tempfile
from pathlib import Path
from types import SimpleNamespace

from nmafc.schemas.memory import MemoryRecord, MemoryStateUpdate, MemoryType
from nmafc.storage.cold import ColdStorage
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage
from nmafc.web.routes.graph import get_entity_detail, get_entity_graph


def make_store(tmpdir: str):
    storage_config = StorageConfig(
        hot_uri=str(Path(tmpdir) / "hot"),
        cold_uri=str(Path(tmpdir) / "cold.db"),
        embedding_dim=3,
    )
    hot = HotStorage(storage_config)
    cold = ColdStorage(storage_config.cold_uri)
    return hot, cold


class TestColdTierInGraph:
    def test_cold_only_entity_appears_with_edge_and_detail(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            hot, cold = make_store(tmpdir)
            hot.upsert(
                MemoryRecord(
                    entity_name="user_health",
                    fact_content="Cholesterol at 5.2 mmol/L",
                    memory_type=MemoryType.ACTIVE_CONTEXT,
                    related_entities=[],
                ),
                [1.0, 1.0, 0.0],
            )
            cold.append_event(
                MemoryStateUpdate(
                    entity_name="user_allergy",
                    fact_content="Allergic to peanuts",
                    memory_type=MemoryType.ACTIVE_CONTEXT,
                    related_entities=["user_health"],
                ),
                turn=1,
            )
            mem = SimpleNamespace(_hot=hot, _cold=cold)

            graph = get_entity_graph(mem)

            node_ids = {n["id"] for n in graph["nodes"]}
            assert "user_allergy" in node_ids
            allergy = next(n for n in graph["nodes"] if n["id"] == "user_allergy")
            assert allergy["record_count"] == 1
            assert {"source": "user_allergy", "target": "user_health"} in graph["edges"]

            detail = get_entity_detail("user_allergy", mem)
            assert detail["entity_name"] == "user_allergy"
            assert [r["entity_name"] for r in detail["records"]] == ["user_allergy"]
            assert "user_health" in detail["neighbors"]

            cold.close()

    def test_archive_fact_already_in_hot_is_not_double_counted(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            hot, cold = make_store(tmpdir)
            hot.upsert(
                MemoryRecord(
                    entity_name="hub",
                    fact_content="Lives at 14 Maple Street",
                    memory_type=MemoryType.CORE_ANCHOR,
                    related_entities=["user_allergy"],
                ),
                [1.0, 0.0, 0.0],
            )
            # The archive mirrors the hot fact; the graph must not count it twice.
            cold.append_event(
                MemoryStateUpdate(
                    entity_name="hub",
                    fact_content="Lives at 14 Maple Street",
                    memory_type=MemoryType.CORE_ANCHOR,
                    related_entities=[],
                ),
                turn=1,
            )
            mem = SimpleNamespace(_hot=hot, _cold=cold)

            graph = get_entity_graph(mem)
            hub = next(n for n in graph["nodes"] if n["id"] == "hub")
            assert hub["record_count"] == 1

            cold.close()

    def test_mem_without_cold_store_still_works(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            hot, _cold = make_store(tmpdir)
            hot.upsert(
                MemoryRecord(
                    entity_name="user_health",
                    fact_content="Cholesterol at 5.2 mmol/L",
                    memory_type=MemoryType.ACTIVE_CONTEXT,
                    related_entities=[],
                ),
                [1.0, 1.0, 0.0],
            )
            mem = SimpleNamespace(_hot=hot)

            graph = get_entity_graph(mem)
            assert {"id": "user_health", "record_count": 1} in [
                {k: n[k] for k in ("id", "record_count")} for n in graph["nodes"]
            ]
