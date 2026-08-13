from __future__ import annotations

from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa

from nmafc.schemas.memory import MemoryRecord, MemoryType, SearchResult
from nmafc.storage.config import StorageConfig

TABLE_NAME = "memory_vectors"

SCHEMA = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), -1)),
    pa.field("id", pa.string()),
    pa.field("agent_id", pa.string()),
    pa.field("conversation_id", pa.string()),
    pa.field("entity_name", pa.string()),
    pa.field("fact_content", pa.string()),
    pa.field("memory_type", pa.string()),
    pa.field("weight", pa.float64()),
    pa.field("consolidation_index", pa.int32()),
    pa.field("created_at_turn", pa.int32()),
    pa.field("last_reinforced_turn", pa.int32()),
    pa.field("related_entities", pa.list_(pa.string())),
])


class HotStorage:
    """LanceDB-backed vector store (Hot RAM).

    Stores only the active, high-salience memory vectors (~1% of total data).
    Provides fast similarity search with metadata for decay calculations.
    """

    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        self._agent_id = config.agent_id
        self._conversation_id = config.conversation_id
        if not config.is_cloud:
            Path(config.hot_uri).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(config.hot_uri)
        self._ensure_table()

    def _ensure_table(self) -> None:
        if TABLE_NAME not in self._db.table_names():
            schema = pa.schema([
                pa.field("vector", pa.list_(pa.float32(), self._config.embedding_dim)),
                pa.field("id", pa.string()),
                pa.field("agent_id", pa.string()),
                pa.field("conversation_id", pa.string()),
                pa.field("entity_name", pa.string()),
                pa.field("fact_content", pa.string()),
                pa.field("memory_type", pa.string()),
                pa.field("weight", pa.float64()),
                pa.field("consolidation_index", pa.int32()),
                pa.field("created_at_turn", pa.int32()),
                pa.field("last_reinforced_turn", pa.int32()),
                pa.field("related_entities", pa.list_(pa.string())),
            ])
            self._db.create_table(TABLE_NAME, schema=schema)
        self._table = self._db.open_table(TABLE_NAME)

    @property
    def _scope_filter(self) -> str:
        """WHERE clause that scopes all queries to this agent + conversation."""
        return f"agent_id = '{self._agent_id}' AND conversation_id = '{self._conversation_id}'"

    def upsert(self, record: MemoryRecord, embedding: list[float]) -> None:
        existing = self._table.search().where(f"id = '{record.id}'").limit(1).to_list()
        if existing:
            self.delete(record.id)

        row = {
            "vector": embedding,
            "id": record.id,
            "agent_id": self._agent_id,
            "conversation_id": self._conversation_id,
            "entity_name": record.entity_name,
            "fact_content": record.fact_content,
            "memory_type": record.memory_type.value,
            "weight": record.weight,
            "consolidation_index": record.consolidation_index,
            "created_at_turn": record.created_at_turn,
            "last_reinforced_turn": record.last_reinforced_turn,
            "related_entities": list(record.related_entities),
        }
        self._table.add([row])

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[SearchResult]:
        results = (
            self._table.search(query_embedding)
            .where(self._scope_filter)
            .limit(top_k)
            .to_list()
        )
        search_results = []
        for row in results:
            distance = row.get("_distance", 0.0)
            score = max(0.0, min(1.0, 1.0 - distance))
            record = self._row_to_record(row)
            search_results.append(SearchResult(record=record, score=score, hops=0))
        return search_results

    def get_by_entity(self, entity_name: str) -> list[MemoryRecord]:
        results = (
            self._table.search()
            .where(f"{self._scope_filter} AND entity_name = '{entity_name}'")
            .limit(100)
            .to_list()
        )
        return [self._row_to_record(r) for r in results]

    def get_by_entities(self, entity_names: list[str]) -> list[MemoryRecord]:
        if not entity_names:
            return []
        quoted = ", ".join(f"'{name}'" for name in set(entity_names))
        results = (
            self._table.search()
            .where(f"{self._scope_filter} AND entity_name IN ({quoted})")
            .limit(500)
            .to_list()
        )
        return [self._row_to_record(r) for r in results]

    def update_weight(self, record_id: str, new_weight: float) -> None:
        results = self._table.search().where(f"id = '{record_id}'").limit(1).to_list()
        if not results:
            return
        row = results[0]
        row["weight"] = new_weight
        row.pop("_distance", None)
        self.delete(record_id)
        self._table.add([row])

    def apply_weight_updates(self, updates: list[tuple[str, float]]) -> None:
        """Apply many weight changes in a single delete + add.

        Semantically identical to calling update_weight() in a loop, but the
        per-record version costs a scan, a delete and an add *each*. The decay
        pass rewrites every mutable record every turn, so that made per-turn
        cost grow linearly with stored memories — quadratic over a
        conversation, and the dominant cost of ingestion.
        """
        if not updates:
            return

        weights = dict(updates)  # last write wins, as in the sequential loop
        quoted = ", ".join(f"'{record_id}'" for record_id in weights)
        rows = self._table.search().where(f"id IN ({quoted})").limit(10000).to_list()
        if not rows:
            return

        for row in rows:
            row.pop("_distance", None)
            row["weight"] = weights[row["id"]]

        self._table.delete(f"id IN ({quoted})")
        self._table.add(rows)

    def delete_many(self, record_ids: list[str]) -> None:
        """Delete several records in one predicate instead of one call each."""
        if not record_ids:
            return
        quoted = ", ".join(f"'{record_id}'" for record_id in set(record_ids))
        self._table.delete(f"id IN ({quoted})")

    def update_reinforcement(self, record_id: str, new_k: int, turn: int) -> None:
        results = self._table.search().where(f"id = '{record_id}'").limit(1).to_list()
        if not results:
            return
        row = results[0]
        row["weight"] = 1.0
        row["consolidation_index"] = new_k
        row["last_reinforced_turn"] = turn
        row.pop("_distance", None)
        self.delete(record_id)
        self._table.add([row])

    def apply_reinforcements(
        self, updates: list[tuple[str, int]], turn: int
    ) -> None:
        """Apply many LTP reinforcements in a single delete + add.

        Semantically identical to calling update_reinforcement() in a loop, and
        the same optimisation apply_weight_updates() makes for the decay pass.
        Retrieval reinforces every record that Spreading Activation surfaces,
        which after two hops is routinely dozens per question, so the per-record
        version made a single answer cost dozens of table rewrites.
        """
        if not updates:
            return

        new_ks = dict(updates)  # last write wins, as in the sequential loop
        quoted = ", ".join(f"'{record_id}'" for record_id in new_ks)
        rows = self._table.search().where(f"id IN ({quoted})").limit(10000).to_list()
        if not rows:
            return

        for row in rows:
            row.pop("_distance", None)
            row["weight"] = 1.0
            row["consolidation_index"] = new_ks[row["id"]]
            row["last_reinforced_turn"] = turn

        self._table.delete(f"id IN ({quoted})")
        self._table.add(rows)

    def delete(self, record_id: str) -> None:
        self._table.delete(f"id = '{record_id}'")

    def get_all_mutable(self) -> list[MemoryRecord]:
        results = (
            self._table.search()
            .where(f"{self._scope_filter} AND memory_type != '{MemoryType.CORE_ANCHOR.value}'")
            .limit(10000)
            .to_list()
        )
        return [self._row_to_record(r) for r in results]

    def get_all(self) -> list[MemoryRecord]:
        results = self._table.search().where(self._scope_filter).limit(10000).to_list()
        return [self._row_to_record(r) for r in results]

    def count(self) -> int:
        return self._table.count_rows()

    def clear(self) -> None:
        """Remove all records scoped to this agent + conversation."""
        self._table.delete(self._scope_filter)

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        results = self._table.search().where(f"id = '{record_id}'").limit(1).to_list()
        if not results:
            return None
        return self._row_to_record(results[0])

    def _row_to_record(self, row: dict) -> MemoryRecord:
        rel = row.get("related_entities")
        if rel is None:
            rel_list = []
        elif hasattr(rel, "tolist"):
            rel_list = rel.tolist()
        else:
            rel_list = list(rel)

        return MemoryRecord(
            id=row["id"],
            entity_name=row["entity_name"],
            fact_content=row["fact_content"],
            memory_type=MemoryType(row["memory_type"]),
            weight=row["weight"],
            consolidation_index=row["consolidation_index"],
            created_at_turn=row["created_at_turn"],
            last_reinforced_turn=row["last_reinforced_turn"],
            related_entities=rel_list,
        )

