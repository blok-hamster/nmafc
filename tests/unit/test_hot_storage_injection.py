"""Regression tests for record-id escaping in LanceDB predicates.

Before this file, `HotStorage` escaped `agent_id`, `conversation_id` and
`entity_name` through `_sql_str()`, but interpolated record ids raw:

    .where(f"{self._scope_filter} AND id = '{record_id}'")

DataFusion has no parameter binding on this API, so the id is spliced into a
string literal. `record_id` reaches that predicate straight from the HTTP path
parameter on `GET / PATCH / DELETE /v1/memories/{memory_id}`.

Every tenant shares ONE `hot_uri` and ONE `memory_vectors` table -- isolation is
achieved by the `agent_id AND conversation_id` filter, not by separate tables.
That is why the filter is the security boundary, and why breaking out of it is
cross-tenant access. The tests below reproduce the production topology exactly:
one shared database, many `HotStorage` handles differing only in scope.

The exploit is ordinary SQL precedence. Given the attacker scoped to
`agent_id='attacker'`, the id `x' OR '1'='1` produces:

    agent_id = 'attacker' AND conversation_id = 'c' AND id = 'x' OR '1'='1'

AND binds tighter than OR, so this parses as
`(agent_id=... AND conversation_id=... AND id='x') OR ('1'='1')` -- the trailing
clause is a standalone tautology and every row in the table matches.

ONE ORDERING TRAP, learned the hard way
---------------------------------------
A `HotStorage` handle must be constructed AFTER the rows it should not see are
written. Lance table handles are version snapshots: a handle opened before an
upsert may not observe it, which silently makes a cross-tenant test pass even
against vulnerable code. `_attacker()` below takes the victim store first and
builds the attacker handle second. If you change that ordering, these tests
become vacuous -- there is a specific assertion below that guards against it.
"""

import tempfile
from pathlib import Path

import pytest

from nmafc.schemas.memory import MemoryRecord, MemoryType
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import HotStorage

EMBED_DIM = 8

# Payloads that try to break out of the string literal and neutralise the
# tenant predicate. The first is the working cross-tenant read; the rest probe
# quote doubling, comment syntax, whitespace, and trailing-quote rebalancing.
INJECTION_IDS = [
    "x' OR '1'='1",
    "x' OR '1'='1' --",
    "x' OR '1'='1' /*",
    "x' OR 1=1 --",
    "x'; DROP TABLE memory_vectors; --",
    "x\" OR \"1\"=\"1",
    "x'OR'1'='1",
    "x' or ''='",
    "x' AND '1'='1",
    "x'; UPDATE memory_vectors SET fact_content='pwned'; --",
    "x' UNION SELECT * FROM memory_vectors --",
    "x'||(SELECT 1)||'",
    "x' AND id != '",
    "x' AND 1=1 AND id = 'y",
    "x\n' OR '1'='1",
    "x' OR 1=1 OR id = 'z",
    "'''' OR ''1''=''1",
    "*",
    "x`) OR (`1`='1",
    "v1' OR '1'='1",
]

VICTIM_FACT = "TENANT-B PHI: SSN 999-88-7777"


def _embedding(seed: float = 0.1) -> list[float]:
    return [seed + i * 0.01 for i in range(EMBED_DIM)]


def _record(entity: str = "test_entity", record_id: str | None = None, **kw) -> MemoryRecord:
    defaults = {
        "entity_name": entity,
        "fact_content": f"Fact about {entity}",
        "memory_type": MemoryType.ACTIVE_CONTEXT,
        "weight": 1.0,
        "consolidation_index": 0,
        "created_at_turn": 1,
        "last_reinforced_turn": 1,
    }
    if record_id is not None:
        defaults["id"] = record_id
    defaults.update(kw)
    return MemoryRecord(**defaults)


def _handle(tmp: str, agent_id: str, conversation_id: str = "conv-1") -> HotStorage:
    """A tenant's handle onto the SHARED hot table. Same path for everyone."""
    config = StorageConfig(
        hot_uri=str(Path(tmp) / "lancedb"),
        embedding_dim=EMBED_DIM,
        agent_id=agent_id,
        conversation_id=conversation_id,
    )
    return HotStorage(config)


def _seeded(tmp: str) -> HotStorage:
    """Victim tenant, one record written. Returns the victim handle."""
    victim = _handle(tmp, "tenant-victim")
    victim.upsert(
        _record("ssn", record_id="victim-1", fact_content=VICTIM_FACT,
                memory_type=MemoryType.CORE_ANCHOR),
        _embedding(),
    )
    return victim


def _attacker(tmp: str) -> HotStorage:
    """Attacker handle, opened AFTER the victim wrote. See module docstring."""
    return _handle(tmp, "tenant-attacker")


class TestSharedTableTopology:
    """Guard the premise the whole file depends on."""

    def test_tenants_share_one_table_and_isolation_is_by_filter(self):
        tmp = tempfile.mkdtemp()
        victim = _seeded(tmp)
        attacker = _attacker(tmp)

        assert victim.count() == 1
        assert attacker.count() == 0, "tenant scoping must hide the victim's row"

        # The row is physically in the shared table, reachable only by filter.
        assert victim.get_record("victim-1") is not None
        assert attacker.get_record("victim-1") is None

    def test_stale_handle_cannot_see_later_writes_which_is_the_ordering_trap(self):
        """Documents WHY attacker handles are built second in this file.

        A Lance handle is a version snapshot. One opened before the victim's
        write cannot observe it -- not even through a predicate that matches
        every row. So a cross-tenant test that opens the attacker handle first
        would return None and pass, against vulnerable code as well as fixed.

        This asserts the snapshot property directly, using a tautological
        predicate so tenant scoping cannot be what is hiding the row.

        If this ever fails, Lance began auto-refreshing handles: the ordering
        constraint is then obsolete and the rest of the file is merely
        conservative, which is harmless.
        """
        tmp = tempfile.mkdtemp()
        stale = _handle(tmp, "tenant-attacker")     # opened BEFORE the write
        victim = _seeded(tmp)                       # written AFTER

        assert victim.count() == 1
        assert victim.get_record("victim-1") is not None
        assert stale.get_record("x' OR '1'='1") is None, (
            "stale handle observed a later write; the ordering note in this "
            "module's docstring is out of date and should be revisited"
        )


class TestRecordIdCannotWidenScope:
    """The core finding: an adversarial id must not cross the tenant boundary."""

    def test_injection_returns_victim_record(self):
        """Plain-language statement of the vulnerability, for the record."""
        tmp = tempfile.mkdtemp()
        _seeded(tmp)
        attacker = _attacker(tmp)
        assert attacker.count() == 0

        leaked = attacker.get_record("x' OR '1'='1")
        assert leaked is None, f"cross-tenant disclosure: got {leaked.fact_content!r}"
        assert VICTIM_FACT not in str(leaked)

    @pytest.mark.parametrize("payload", INJECTION_IDS)
    def test_get_record_returns_none(self, payload: str):
        tmp = tempfile.mkdtemp()
        _seeded(tmp)
        leaked = _attacker(tmp).get_record(payload)
        assert leaked is None, (
            f"payload {payload!r} crossed tenant scope, returned "
            f"{leaked.fact_content if leaked else None!r}"
        )

    @pytest.mark.parametrize("payload", INJECTION_IDS)
    def test_update_weight_does_not_mutate_other_tenants(self, payload: str):
        tmp = tempfile.mkdtemp()
        victim = _seeded(tmp)
        _attacker(tmp).update_weight(payload, 0.0)

        after = victim.get_record("victim-1")
        assert after is not None, f"payload {payload!r} deleted another tenant's record"
        assert after.weight == pytest.approx(1.0), (
            f"payload {payload!r} mutated another tenant's weight"
        )

    @pytest.mark.parametrize("payload", INJECTION_IDS)
    def test_update_reinforcement_does_not_mutate_other_tenants(self, payload: str):
        tmp = tempfile.mkdtemp()
        victim = _seeded(tmp)
        _attacker(tmp).update_reinforcement(payload, new_k=99, turn=5)

        after = victim.get_record("victim-1")
        assert after is not None, f"payload {payload!r} deleted another tenant's record"
        assert after.consolidation_index == 0

    @pytest.mark.parametrize("payload", INJECTION_IDS)
    def test_delete_does_not_remove_other_tenants(self, payload: str):
        tmp = tempfile.mkdtemp()
        victim = _seeded(tmp)
        _attacker(tmp).delete(payload)
        assert victim.get_record("victim-1") is not None, (
            f"payload {payload!r} deleted another tenant's record"
        )

    @pytest.mark.parametrize("payload", INJECTION_IDS)
    def test_delete_many_does_not_remove_other_tenants(self, payload: str):
        """The `id IN (...)` builders carried the same bug in batch form."""
        tmp = tempfile.mkdtemp()
        victim = _seeded(tmp)
        _attacker(tmp).delete_many([payload])
        assert victim.get_record("victim-1") is not None

    @pytest.mark.parametrize("payload", INJECTION_IDS)
    def test_apply_weight_updates_does_not_mutate_other_tenants(self, payload: str):
        tmp = tempfile.mkdtemp()
        victim = _seeded(tmp)
        _attacker(tmp).apply_weight_updates({payload: 0.0})
        after = victim.get_record("victim-1")
        assert after is not None
        assert after.weight == pytest.approx(1.0)

    @pytest.mark.parametrize("payload", INJECTION_IDS)
    def test_apply_reinforcements_does_not_mutate_other_tenants(self, payload: str):
        tmp = tempfile.mkdtemp()
        victim = _seeded(tmp)
        _attacker(tmp).apply_reinforcements({payload: 50}, turn=9)
        after = victim.get_record("victim-1")
        assert after is not None
        assert after.consolidation_index == 0

    @pytest.mark.parametrize("payload", INJECTION_IDS)
    def test_set_invalid_at_many_does_not_mutate_other_tenants(self, payload: str):
        tmp = tempfile.mkdtemp()
        victim = _seeded(tmp)
        _attacker(tmp).set_invalid_at_many([(payload, 7)])
        after = victim.get_record("victim-1")
        assert after is not None, f"payload {payload!r} deleted another tenant's record"
        assert after.invalid_at is None, (
            f"payload {payload!r} invalidated another tenant's record"
        )

    @pytest.mark.parametrize("payload", INJECTION_IDS)
    def test_upsert_does_not_overwrite_other_tenant(self, payload: str):
        """upsert keyed on `id`; an injected id must not match the victim's row."""
        tmp = tempfile.mkdtemp()
        victim = _seeded(tmp)
        _attacker(tmp).upsert(_record("attacker-owned", record_id=payload), _embedding(0.5))
        after = victim.get_record("victim-1")
        assert after is not None
        assert after.fact_content == VICTIM_FACT


class TestInjectionDoesNotCorruptStore:
    """A rejected payload must not damage the tenant's own rows or raise."""

    @pytest.mark.parametrize("payload", INJECTION_IDS)
    def test_own_records_survive(self, payload: str):
        tmp = tempfile.mkdtemp()
        store = _handle(tmp, "tenant-a")
        store.upsert(_record("mine", record_id="mine-1"), _embedding())

        for call in (lambda: store.get_record(payload), lambda: store.delete(payload)):
            try:
                call()
            except Exception as exc:  # noqa: BLE001 - raising is the failure
                pytest.fail(
                    f"payload {payload!r} raised instead of matching nothing: "
                    f"{type(exc).__name__}: {str(exc)[:160]}"
                )
        assert store.get_record("mine-1") is not None
        assert store.count() == 1

    def test_table_remains_readable_after_every_payload(self):
        tmp = tempfile.mkdtemp()
        store = _handle(tmp, "tenant-a")
        store.upsert(_record("mine", record_id="mine-1"), _embedding())

        for payload in INJECTION_IDS:
            try:
                store.get_record(payload)
            except Exception as exc:  # noqa: BLE001
                pytest.fail(
                    f"payload {payload!r} raised a parser error: {type(exc).__name__}: "
                    f"{str(exc)[:160]}"
                )
        assert store.count() == 1


class TestLegitimateIdsStillWork:
    """Guard against over-escaping regressing the happy path."""

    def test_uuid_roundtrips(self):
        tmp = tempfile.mkdtemp()
        store = _handle(tmp, "tenant-a")
        rec = _record("mine")
        store.upsert(rec, _embedding())
        assert store.get_record(rec.id) is not None

    def test_punctuation_bearing_ids_roundtrip(self):
        tmp = tempfile.mkdtemp()
        store = _handle(tmp, "tenant-a")
        ids = ["rec:123", "rec/456", 'rec"789', "rec-with-dash", "rec_0", "rec 1"]
        for i, rid in enumerate(ids):
            store.upsert(_record(f"e{i}", record_id=rid), _embedding())
        for rid in ids:
            assert store.get_record(rid) is not None, f"legitimate id {rid!r} not found"
        assert store.count() == len(ids)

    def test_batch_paths_still_work(self):
        """The IN(...) builders must still function after escaping."""
        tmp = tempfile.mkdtemp()
        store = _handle(tmp, "tenant-a")
        recs = [_record(f"e{i}", record_id=f"r{i}") for i in range(3)]
        for r in recs:
            store.upsert(r, _embedding())
        assert store.count() == 3

        store.apply_weight_updates({r.id: 0.5 for r in recs})
        assert all(
            store.get_record(r.id).weight == pytest.approx(0.5) for r in recs
        )

        store.apply_reinforcements({r.id: 3 for r in recs}, turn=4)
        assert all(
            store.get_record(r.id).consolidation_index == 3 for r in recs
        )

        store.set_invalid_at_many([(r.id, 6) for r in recs])
        assert all(store.get_record(r.id).invalid_at == 6 for r in recs)

        store.delete_many([r.id for r in recs])
        assert store.count() == 0

    def test_apostrophe_in_entity_name_still_works(self):
        """The original reason _sql_str exists must keep working."""
        tmp = tempfile.mkdtemp()
        store = _handle(tmp, "tenant-a")
        store.upsert(_record("melanie's family trip"), _embedding())
        assert store.get_by_entity("melanie's family trip")
