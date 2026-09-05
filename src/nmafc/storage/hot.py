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
    pa.field("valid_at", pa.int32()),
    pa.field("invalid_at", pa.int32()),
    pa.field("valid_at_text", pa.string()),
])

_TEMPORAL_COLUMNS = {"valid_at", "invalid_at", "valid_at_text"}


def _sql_str(value: str) -> str:
    """Render a Python string as a SQL string literal for a LanceDB filter.

    Entity names come from the extractor, which means they are model-generated
    English rather than identifiers: "melanie's family camping trip" is a name
    it produces routinely. Interpolating that between bare single quotes ends
    the literal at the apostrophe and hands the parser "s family camping trip",
    which fails the whole query -- and because entity lookup is the graph
    traversal step, the failure takes the retrieval down with it.

    Doubling the quote is the SQL-standard escape and the only one DataFusion
    accepts here; there is no parameter binding on this API to use instead.
    """
    return "'" + value.replace("'", "''") + "'"


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
                pa.field("valid_at", pa.int32()),
                pa.field("invalid_at", pa.int32()),
                pa.field("valid_at_text", pa.string()),
            ])
            self._db.create_table(TABLE_NAME, schema=schema)
        self._table = self._db.open_table(TABLE_NAME)
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add temporal columns to tables created before they existed.

        Uses LanceDB's own column addition, which alters the table schema and
        backfills nulls. The obvious-looking alternative -- read every row,
        delete the table's contents, add the rows back with the new key set --
        cannot work: `add` validates incoming rows against the schema already on
        disk, so the extra key is rejected with "Field not found in target
        schema" and an existing store fails to open at all. The rewrite also
        capped at whatever limit it read with, and silently dropped anything
        past it.
        """
        existing_names = {f.name for f in self._table.schema}
        missing = [f for f in SCHEMA if f.name in _TEMPORAL_COLUMNS - existing_names]
        if missing:
            self._table.add_columns(missing)

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
            "valid_at": record.valid_at,
            "invalid_at": record.invalid_at,
            "valid_at_text": record.valid_at_text,
        }
        self._table.add([row])

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        exclude_invalidated: bool = True,
    ) -> list[SearchResult]:
        # Cosine, not the LanceDB default of L2. `score` below is consumed as a
        # similarity in [0, 1] and compared against DecayConfig.theta, and only
        # cosine distance gives that: it is 1 - cos_sim, so 1 - distance is the
        # cosine similarity itself. Under (squared) L2 the same arithmetic is
        # meaningless -- unit-norm embedding pairs land at distance 1.4-1.6, so
        # 1 - distance clamps to 0.0 for every hit and no threshold above zero
        # is ever reachable.
        where = self._scope_filter
        if exclude_invalidated:
            where += " AND invalid_at IS NULL"
        results = (
            self._table.search(query_embedding)
            .distance_type("cosine")
            .where(where)
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
            .where(f"{self._scope_filter} AND entity_name = {_sql_str(entity_name)}")
            .limit(100)
            .to_list()
        )
        return [self._row_to_record(r) for r in results]

    def get_by_entities(
        self, entity_names: list[str], exclude_invalidated: bool = True
    ) -> list[MemoryRecord]:
        if not entity_names:
            return []
        quoted = ", ".join(_sql_str(name) for name in set(entity_names))
        where = f"{self._scope_filter} AND entity_name IN ({quoted})"
        if exclude_invalidated:
            where += " AND invalid_at IS NULL"
        results = (
            self._table.search()
            .where(where)
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

    def apply_weight_updates(
        self, updates: list[tuple[str, float]], decay_turn: int | None = None
    ) -> None:
        """Apply many weight changes in a single delete + add.

        Semantically identical to calling update_weight() in a loop, but the
        per-record version costs a scan, a delete and an add *each*. The decay
        pass rewrites every mutable record every turn, so that made per-turn
        cost grow linearly with stored memories — quadratic over a
        conversation, and the dominant cost of ingestion.

        `decay_turn` advances the decay clock alongside the weight, and callers
        applying a decay pass must pass it. decay_record() reads the stored
        weight as w0 and multiplies by e^{-lambda * (current_turn -
        last_reinforced_turn)}. Writing the decayed weight back without moving
        last_reinforced_turn leaves an already-decayed value sitting in the w0
        slot while the elapsed term keeps growing, so turn n applies n turns of
        decay to a weight that has already absorbed n-1 of them. Over a
        conversation that compounds to w0 * e^{-lambda*n(n+1)/2} rather than
        w0 * e^{-lambda*n}: at lambda = 0.05 an ActiveContext record reaches the
        0.1 prune threshold at turn 10 instead of turn 46, and an EphemeralState
        record at turn 3 instead of turn 5.

        The weight is still carried forward rather than recomputed from 1.0,
        because suppression (pruning.apply_suppression) multiplies the stored
        weight by gamma when a newer fact contradicts an older one. Restarting
        decay from 1.0 would erase that penalty on the next pass.
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
            if decay_turn is not None:
                row["last_reinforced_turn"] = decay_turn

        self._table.delete(f"id IN ({quoted})")
        self._table.add(rows)

    def delete_many(self, record_ids: list[str]) -> None:
        """Delete several records in one predicate instead of one call each."""
        if not record_ids:
            return
        quoted = ", ".join(f"'{record_id}'" for record_id in set(record_ids))
        self._table.delete(f"id IN ({quoted})")

    def update_reinforcement(self, record_id: str, new_k: int, turn: int) -> None:
        """One LTP writeback. See apply_reinforcements for the clock guard."""
        results = self._table.search().where(f"id = '{record_id}'").limit(1).to_list()
        if not results:
            return
        row = results[0]
        row["weight"] = 1.0
        row["consolidation_index"] = new_k
        row["last_reinforced_turn"] = max(turn, row["last_reinforced_turn"])
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

        The clock only ever moves forward. A caller that reopens a store without
        restoring its turn counter starts at 0 and reinforces at turn 1, which
        without this guard stamps `last_reinforced_turn=1` onto records created
        at turn 200 -- reinforced, on the record, 199 turns before they existed.
        Every later decay then measures elapsed time from that floor and reads
        the whole conversation as having passed since. Found on the full_v3
        stores, where 3,477 of 3,807 records carry a reinforcement turn earlier
        than their creation turn.
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
            row["last_reinforced_turn"] = max(turn, row["last_reinforced_turn"])

        self._table.delete(f"id IN ({quoted})")
        self._table.add(rows)

    def apply_relation_updates(
        self, updates: list[tuple[str, list[str]]]
    ) -> None:
        """Rewrite `related_entities` for many records in a single delete + add.

        Used by the prune cycle to reroute links around a fact it is about to
        remove, so the graph loses a node rather than a region. Same batched
        shape as apply_reinforcements: pruning touches every record that pointed
        at anything doomed, which on a busy conversation is hundreds at once.
        """
        if not updates:
            return

        new_links = dict(updates)  # last write wins
        quoted = ", ".join(_sql_str(record_id) for record_id in new_links)
        rows = self._table.search().where(f"id IN ({quoted})").limit(10000).to_list()
        if not rows:
            return

        for row in rows:
            row.pop("_distance", None)
            row["related_entities"] = list(new_links[row["id"]])

        self._table.delete(f"id IN ({quoted})")
        self._table.add(rows)

    def set_invalid_at_many(self, updates: list[tuple[str, int]]) -> None:
        """Mark records as temporally invalidated in a single batch operation."""
        if not updates:
            return
        invalidations = dict(updates)
        quoted = ", ".join(f"'{rid}'" for rid in invalidations)
        rows = self._table.search().where(f"id IN ({quoted})").limit(10000).to_list()
        if not rows:
            return
        for row in rows:
            row.pop("_distance", None)
            row["invalid_at"] = invalidations[row["id"]]
        self._table.delete(f"id IN ({quoted})")
        self._table.add(rows)

    def compact(self) -> bool:
        """Merge accumulated table versions back into a compact layout.

        Lance is append-only: `delete` tombstones rows and `add` writes a new
        fragment, so every reinforcement and every decay pass leaves another
        version behind rather than overwriting in place. Retrieval reinforces
        on every query, so the count climbs for the life of a conversation --
        stores from the full_v3 run had accumulated 4,025 versions -- and each
        later scan has to read across the fragments that produced them.

        Measured on a copy of one such store: traversal fell from 48.2 ms to
        14.4 ms and vector search from 32.4 ms to 23.3 ms after a single call,
        which cost 6.4 seconds once. That trade only makes sense between
        conversations, never inside one, so this is exposed for a caller to
        schedule rather than being called from the write path.

        Returns False when the installed LanceDB exposes no optimize entry
        point, so a caller can log the miss instead of failing a long run over
        a maintenance step that is an optimisation and nothing more.
        """
        optimize = getattr(self._table, "optimize", None)
        if optimize is None:
            return False
        try:
            optimize()
        except Exception:  # noqa: BLE001 - maintenance must never break a run
            return False
        return True

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
        results = self._table.search().where(self._scope_filter).to_list()
        return len(results)

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
            valid_at=row.get("valid_at"),
            invalid_at=row.get("invalid_at"),
            valid_at_text=row.get("valid_at_text"),
        )

