from __future__ import annotations

from typing import Callable
from nmafc.schemas.memory import DecayConfig, MemoryType
from nmafc.storage.cold import ColdStorage
from nmafc.storage.hot import HotStorage


class MemoryConsolidator:
    """Offline REM Sleep Memory Consolidation Engine.

    Scans active memories and Cold ROM logs to:
    1. Consolidate redundant episodic state changes.
    2. Elevate high-consolidation (high k) Active Contexts to Core Anchors.
    3. Prune broken or stale related_entities pointers.
    """

    def __init__(self, hot: HotStorage, cold: ColdStorage, config: DecayConfig) -> None:
        self._hot = hot
        self._cold = cold
        self._config = config

    def consolidate(self, current_turn: int) -> int:
        """Run consolidation pass over Hot RAM records.

        Returns count of records updated or consolidated.
        """
        records = self._hot.get_all()
        consolidated_count = 0

        # 1. Elevate highly consolidated ActiveContext (k >= 10) to CoreAnchor
        for rec in records:
            if rec.memory_type == MemoryType.ACTIVE_CONTEXT and rec.consolidation_index >= 10:
                elevated = rec.model_copy(update={"memory_type": MemoryType.CORE_ANCHOR, "weight": 1.0})
                self._hot.update_weight(elevated.id, 1.0)
                consolidated_count += 1

        # 2. Clean up dead relation pointers
        active_entities = {r.entity_name.lower() for r in self._hot.get_all()}
        for rec in self._hot.get_all():
            if not rec.related_entities:
                continue
            cleaned_relations = [rel for rel in rec.related_entities if rel.lower() in active_entities]
            if len(cleaned_relations) != len(rec.related_entities):
                updated = rec.model_copy(update={"related_entities": cleaned_relations})
                results = self._hot._table.search().where(f"id = '{rec.id}'").limit(1).to_list()
                if results:
                    row = results[0]
                    row["related_entities"] = cleaned_relations
                    row.pop("_distance", None)
                    self._hot.delete(rec.id)
                    self._hot._table.add([row])
                    consolidated_count += 1

        return consolidated_count
