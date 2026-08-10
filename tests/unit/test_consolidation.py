import pytest
import tempfile
from pathlib import Path
from nmafc.engine.consolidation import MemoryConsolidator
from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryType
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage
from nmafc.storage.cold import ColdStorage


def test_rem_consolidation_elevation_and_pruning():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage_config = StorageConfig(
            hot_uri=str(Path(tmpdir) / "hot"),
            cold_uri=str(Path(tmpdir) / "cold.db"),
            embedding_dim=3,
        )
        hot = HotStorage(storage_config)
        cold = ColdStorage(storage_config.cold_uri)
        decay_config = DecayConfig()
        consolidator = MemoryConsolidator(hot, cold, decay_config)

        # 1. Record with consolidation_index k >= 10 should be elevated to CORE_ANCHOR
        rec1 = MemoryRecord(
            entity_name="user_project",
            fact_content="Working on AI Memory Engine",
            memory_type=MemoryType.ACTIVE_CONTEXT,
            consolidation_index=12,
            related_entities=["dead_pointer"],
        )
        hot.upsert(rec1, [1.0, 0.0, 0.0])

        # Run REM sleep consolidation pass
        count = consolidator.consolidate(current_turn=5)

        updated_records = hot.get_all()
        assert len(updated_records) == 1
        record = updated_records[0]

        # Assert elevation to CORE_ANCHOR
        assert record.memory_type == MemoryType.CORE_ANCHOR
        # Assert broken relation pointer "dead_pointer" was pruned
        assert record.related_entities == []

        cold.close()
