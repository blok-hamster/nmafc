from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from nmafc.schemas.memory import DecayConfig, MemoryType
from nmafc.storage.cold_base import ColdStorageBase
from nmafc.storage.hot import HotStorage

if TYPE_CHECKING:
    from nmafc.storage.event_log import EventLog


class MemoryConsolidator:
    """Offline REM Sleep Memory Consolidation Engine.

    Scans active memories and Cold ROM logs to:
    1. Consolidate redundant episodic state changes.
    2. Elevate high-consolidation (high k) Active Contexts to Core Anchors.
    3. Prune broken or stale related_entities pointers.
    """

    def __init__(self, hot: HotStorage, cold: ColdStorageBase, config: DecayConfig) -> None:
        self._hot = hot
        self._cold = cold
        self._config = config

    def consolidate(self, current_turn: int, event_logger: EventLog | None = None) -> int:
        """Run consolidation pass over Hot RAM records.

        Returns count of records updated or consolidated.

        When `event_logger` is provided, CONSOLIDATION events are emitted
        for each record elevated from ActiveContext to CoreAnchor.
        """
        records = self._hot.get_all()
        consolidated_count = 0

        # 1. Elevate highly consolidated ActiveContext (k >= 10) to CoreAnchor
        for rec in records:
            if rec.memory_type == MemoryType.ACTIVE_CONTEXT and rec.consolidation_index >= 10:
                results = self._hot._table.search().where(f"id = '{rec.id}'").limit(1).to_list()
                if results:
                    row = results[0]
                    row["memory_type"] = MemoryType.CORE_ANCHOR.value
                    row["weight"] = 1.0
                    row.pop("_distance", None)
                    self._hot.delete(rec.id)
                    self._hot._table.add([row])
                    consolidated_count += 1

                    if event_logger is not None:
                        from nmafc.schemas.events import EventType, MemoryEvent

                        event_logger.log(
                            MemoryEvent(
                                event_type=EventType.CONSOLIDATION,
                                turn=current_turn,
                                record_id=rec.id,
                                entity_name=rec.entity_name,
                                old_weight=rec.weight,
                                new_weight=1.0,
                                old_memory_type=rec.memory_type.value,
                                new_memory_type=MemoryType.CORE_ANCHOR.value,
                            )
                        )

        # 2. Clean up dead relation pointers
        active_entities = {r.entity_name.lower() for r in self._hot.get_all()}
        for rec in self._hot.get_all():
            if not rec.related_entities:
                continue
            cleaned_relations = [rel for rel in rec.related_entities if rel.lower() in active_entities]
            if len(cleaned_relations) != len(rec.related_entities):
                results = self._hot._table.search().where(f"id = '{rec.id}'").limit(1).to_list()
                if results:
                    row = results[0]
                    row["related_entities"] = cleaned_relations
                    row.pop("_distance", None)
                    self._hot.delete(rec.id)
                    self._hot._table.add([row])
                    consolidated_count += 1

        return consolidated_count
