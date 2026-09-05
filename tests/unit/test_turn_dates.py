"""Facts reach the model dated, not stamped with a turn number.

Retrieved facts used to be presented as "(Valid: turn 220 - present)". An
ordinal is not a date, so every question asking when something happened was
unanswerable however good the retrieval had been, and the run's own answers say
so in as many words: "it was mentioned in turn 220 and is valid from turn 220
onward, but no specific date". Twenty answers in that run quoted a turn number
back to the user as if it were a date.

Two sources of dates, in priority order. A per-fact date the extractor read off
the conversation wins, because facts are routinely stated well after the event
("last year we drove to the coast"). Otherwise the date of the turn that
mentioned it. Turn numbers survive only as the fallback of last resort, for
stores that were never given any dates at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nmafc.integration.query_router import QueryRouter
from nmafc.schemas.memory import (
    DecayConfig,
    MemoryRecord,
    MemoryStateUpdate,
    MemoryType,
)
from nmafc.storage.cold import ColdStorage
from nmafc.storage.config import StorageConfig
from nmafc.storage.hot import TABLE_NAME, HotStorage


@pytest.fixture
def cold(tmp_path: Path):
    store = ColdStorage(str(tmp_path / "cold.db"))
    yield store
    store.close()


@pytest.fixture
def router(tmp_path: Path, cold: ColdStorage):
    hot = HotStorage(StorageConfig(
        hot_uri=str(tmp_path / "hot"),
        cold_uri=str(tmp_path / "cold.db"),
        embedding_dim=4,
    ))
    yield QueryRouter(hot, cold, embedder=None, config=DecayConfig())


def record(turn: int = 5, text: str | None = None, invalid_at: int | None = None):
    return MemoryRecord(
        entity_name="john_road_trip",
        fact_content="John drove to the Pacific Northwest",
        memory_type=MemoryType.ACTIVE_CONTEXT,
        created_at_turn=turn,
        valid_at=turn,
        valid_at_text=text,
        invalid_at=invalid_at,
    )


# --- the archive's turn -> date table -------------------------------------

def test_turn_timestamps_round_trip(cold: ColdStorage):
    cold.record_turn_timestamp(3, "15 May 2023")
    assert cold.turn_timestamps() == {3: "15 May 2023"}


def test_recording_the_same_turn_twice_corrects_it(cold: ColdStorage):
    """Backfilling a finished store must not fail on the primary key."""
    cold.record_turn_timestamp(3, "15 May 2023")
    cold.record_turn_timestamp(3, "16 May 2023")
    assert cold.turn_timestamps() == {3: "16 May 2023"}


def test_dates_are_scoped_to_the_tenant(tmp_path: Path):
    a = ColdStorage(str(tmp_path / "c.db"), agent_id="a", conversation_id="one")
    b = ColdStorage(str(tmp_path / "c.db"), agent_id="a", conversation_id="two")
    try:
        a.record_turn_timestamp(1, "15 May 2023")
        assert a.turn_timestamps() == {1: "15 May 2023"}
        assert b.turn_timestamps() == {}
    finally:
        a.close()
        b.close()


def test_an_archive_predating_the_table_still_opens(tmp_path: Path):
    """The table is created on open, so old archives gain it rather than break."""
    first = ColdStorage(str(tmp_path / "cold.db"))
    first.append_event(
        MemoryStateUpdate(
            entity_name="e", fact_content="f", memory_type=MemoryType.ACTIVE_CONTEXT
        ),
        turn=1,
    )
    first.close()

    second = ColdStorage(str(tmp_path / "cold.db"))
    try:
        assert second.turn_timestamps() == {}
        second.record_turn_timestamp(1, "15 May 2023")
        assert second.turn_timestamps() == {1: "15 May 2023"}
    finally:
        second.close()


# --- what the model is shown ----------------------------------------------

def test_a_store_with_no_dates_still_falls_back_to_turns(router: QueryRouter):
    assert "turn 5" in router.format_context([record(turn=5)])


def test_a_dated_turn_is_shown_as_a_date(router: QueryRouter, cold: ColdStorage):
    cold.record_turn_timestamp(5, "15 May 2023")
    context = router.format_context([record(turn=5)])
    assert "15 May 2023" in context
    assert "turn 5" not in context


def test_an_undated_turn_borrows_the_session_it_sits_in(
    router: QueryRouter, cold: ColdStorage
):
    """Dates are recorded per turn but a conversation is dated per session."""
    cold.record_turn_timestamp(4, "15 May 2023")
    cold.record_turn_timestamp(9, "2 June 2023")
    assert "15 May 2023" in router.format_context([record(turn=6)])


def test_a_turn_before_every_known_date_is_not_given_one(
    router: QueryRouter, cold: ColdStorage
):
    """Better a turn number than a date the store cannot justify."""
    cold.record_turn_timestamp(9, "2 June 2023")
    assert "turn 2" in router.format_context([record(turn=2)])


def test_the_extractors_own_date_beats_the_turns(
    router: QueryRouter, cold: ColdStorage
):
    """"Last year we drove to the coast" is not dated by when it was said."""
    cold.record_turn_timestamp(5, "15 May 2023")
    context = router.format_context([record(turn=5, text="summer 2022")])
    assert "summer 2022" in context
    assert "15 May 2023" not in context


def test_an_invalidated_fact_gets_a_dated_end(router: QueryRouter, cold: ColdStorage):
    cold.record_turn_timestamp(5, "15 May 2023")
    cold.record_turn_timestamp(11, "2 June 2023")
    context = router.format_context([record(turn=5, invalid_at=11)])
    assert "15 May 2023 - 2 June 2023" in context


def test_a_live_fact_still_reads_as_present(router: QueryRouter, cold: ColdStorage):
    cold.record_turn_timestamp(5, "15 May 2023")
    assert "- present" in router.format_context([record(turn=5)])


def test_new_dates_are_picked_up_after_a_reset(router: QueryRouter, cold: ColdStorage):
    """The map is cached, so ingestion after a retrieval must invalidate it."""
    assert "turn 5" in router.format_context([record(turn=5)])
    cold.record_turn_timestamp(5, "15 May 2023")
    router.reset_turn_dates()
    assert "15 May 2023" in router.format_context([record(turn=5)])


# --- what the validity range costs ----------------------------------------

@pytest.fixture
def compact_router(tmp_path: Path, cold: ColdStorage):
    hot = HotStorage(StorageConfig(
        hot_uri=str(tmp_path / "hot_compact"),
        cold_uri=str(tmp_path / "cold.db"),
        embedding_dim=4,
    ))
    yield QueryRouter(
        hot, cold, embedder=None, config=DecayConfig(compact_validity=True)
    )


def test_compact_drops_the_label_and_the_open_end(
    compact_router: QueryRouter, cold: ColdStorage
):
    """Measured at rerank_top_k=40, this suffix was 21% of the prompt and every
    one of its 10,000 instances ended "- present"."""
    cold.record_turn_timestamp(5, "15 May 2023")
    context = compact_router.format_context([record(turn=5)])
    assert "(15 May 2023)" in context
    assert "Valid:" not in context
    assert "present" not in context


def test_compact_still_shows_an_end_where_there_is_one(
    compact_router: QueryRouter, cold: ColdStorage
):
    """The range is dropped because it is uninformative, not because it is
    unwanted: a fact that really was superseded still says so."""
    cold.record_turn_timestamp(5, "15 May 2023")
    cold.record_turn_timestamp(11, "2 June 2023")
    context = compact_router.format_context([record(turn=5, invalid_at=11)])
    assert "(15 May 2023 to 2 June 2023)" in context


def test_compact_is_off_unless_asked_for(router: QueryRouter, cold: ColdStorage):
    cold.record_turn_timestamp(5, "15 May 2023")
    assert "(Valid: 15 May 2023 - present)" in router.format_context([record(turn=5)])


# --- the verbatim evidence layer ------------------------------------------

@pytest.fixture
def hydrating_router(tmp_path: Path, cold: ColdStorage):
    hot = HotStorage(StorageConfig(
        hot_uri=str(tmp_path / "hot_hydrate"),
        cold_uri=str(tmp_path / "cold.db"),
        embedding_dim=4,
    ))
    yield QueryRouter(hot, cold, embedder=None, config=DecayConfig(hydrate_top_k=2))


def test_turn_text_round_trips(cold: ColdStorage):
    cold.record_turn_text(3, "John: I drove to the coast two weekends ago.")
    assert cold.text_for_turns([3]) == {
        3: "John: I drove to the coast two weekends ago."
    }


def test_only_the_turns_asked_for_come_back(cold: ColdStorage):
    """A conversation holds hundreds of turns and a prompt hydrates a handful."""
    cold.record_turn_text(1, "first")
    cold.record_turn_text(2, "second")
    assert cold.text_for_turns([2]) == {2: "second"}
    assert cold.text_for_turns([]) == {}


def test_re_ingesting_a_turn_corrects_its_text(cold: ColdStorage):
    cold.record_turn_text(3, "old wording")
    cold.record_turn_text(3, "new wording")
    assert cold.text_for_turns([3]) == {3: "new wording"}


def test_turn_text_is_scoped_to_the_tenant(tmp_path: Path):
    a = ColdStorage(str(tmp_path / "c.db"), agent_id="a", conversation_id="one")
    b = ColdStorage(str(tmp_path / "c.db"), agent_id="a", conversation_id="two")
    try:
        a.record_turn_text(1, "mine")
        assert a.text_for_turns([1]) == {1: "mine"}
        assert b.text_for_turns([1]) == {}
    finally:
        a.close()
        b.close()


def test_the_source_turn_reaches_the_prompt(
    hydrating_router: QueryRouter, cold: ColdStorage
):
    """The wording extraction threw away is what several categories are scored
    on: the gold "two weekends before 17 July 2023" lives in the utterance,
    while the fact carries a date the extractor resolved for itself."""
    cold.record_turn_text(5, "Melanie: we went camping two weekends ago.")
    context = hydrating_router.format_context([record(turn=5)])
    assert "<SOURCE>" in context
    assert "two weekends ago" in context
    # The fact is still the thing being ranked and shown; this is added to it.
    assert "John drove to the Pacific Northwest" in context


def test_one_turn_is_hydrated_once_however_many_facts_came_off_it(
    hydrating_router: QueryRouter, cold: ColdStorage
):
    """An exchange routinely yields six facts. Billing per fact would pay for
    the same text repeatedly."""
    cold.record_turn_text(5, "Melanie: we went camping two weekends ago.")
    context = hydrating_router.format_context([record(turn=5), record(turn=5)])
    assert context.count("two weekends ago") == 1


def test_hydration_stops_at_the_budget(
    hydrating_router: QueryRouter, cold: ColdStorage
):
    """Budget is two, so the third fact's source is not worth the tokens."""
    for turn, text in ((1, "first turn"), (2, "second turn"), (3, "third turn")):
        cold.record_turn_text(turn, text)
    context = hydrating_router.format_context(
        [record(turn=1), record(turn=2), record(turn=3)]
    )
    assert "first turn" in context
    assert "second turn" in context
    assert "third turn" not in context


def test_a_store_with_no_turn_text_renders_as_before(
    hydrating_router: QueryRouter, cold: ColdStorage
):
    """Stores built before the table existed must keep working, silently
    without hydration rather than failing an answer."""
    context = hydrating_router.format_context([record(turn=5)])
    assert "<SOURCE>" not in context
    assert "John drove to the Pacific Northwest" in context


def test_hydration_is_on_by_default(router: QueryRouter, cold: ColdStorage):
    """The default configuration is the one the published numbers came from.

    `hydrate_top_k` was 0 while the measurement that justifies it (64.4% ->
    66.1%, p = 0.021) was taken at 5, so the shipped default contradicted the
    result it was meant to carry.
    """
    cold.record_turn_text(5, "Melanie: we went camping two weekends ago.")
    context = router.format_context([record(turn=5)])
    assert "<SOURCE>" in context
    assert "two weekends ago" in context


def test_hydration_can_be_switched_off(tmp_path: Path, cold: ColdStorage):
    """Zero is the ablation, and it has to reach the prompt as facts alone."""
    hot = HotStorage(StorageConfig(
        hot_uri=str(tmp_path / "hot_dry"),
        cold_uri=str(tmp_path / "cold.db"),
        embedding_dim=4,
    ))
    dry = QueryRouter(hot, cold, embedder=None, config=DecayConfig(hydrate_top_k=0))
    cold.record_turn_text(5, "Melanie: we went camping two weekends ago.")
    context = dry.format_context([record(turn=5)])
    assert "<SOURCE>" not in context
    assert "two weekends ago" not in context
    assert "John drove to the Pacific Northwest" in context


# --- the field that carries it --------------------------------------------

def test_hot_storage_round_trips_the_extractors_date(tmp_path: Path):
    hot = HotStorage(StorageConfig(
        hot_uri=str(tmp_path / "hot"),
        cold_uri=str(tmp_path / "cold.db"),
        embedding_dim=4,
    ))
    hot.upsert(record(turn=5, text="January 2023"), [0.1, 0.2, 0.3, 0.4])
    assert hot.get_all()[0].valid_at_text == "January 2023"


def test_opening_a_store_without_the_column_keeps_every_row(tmp_path: Path):
    """Migration must add the column, not empty the table doing it.

    The previous migration read the rows, called delete("true"), then added them
    back with the new key. `add` validates against the schema already on disk,
    so the extra key was rejected -- but the delete had already committed. A
    real store lost all 494 of its facts to that ordering, recoverable only
    because LanceDB keeps versions. Adding the column in place has no such
    window.
    """
    import lancedb
    import pyarrow as pa

    uri = str(tmp_path / "hot")
    old_schema = pa.schema([
        f for f in pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 4)),
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
        ])
    ])
    db = lancedb.connect(uri)
    table = db.create_table(TABLE_NAME, schema=old_schema)
    table.add([{
        "vector": [0.1, 0.2, 0.3, 0.4],
        "id": "abc",
        "agent_id": "default",
        "conversation_id": "default",
        "entity_name": "john_road_trip",
        "fact_content": "John drove to the Pacific Northwest",
        "memory_type": "ActiveContext",
        "weight": 1.0,
        "consolidation_index": 0,
        "created_at_turn": 5,
        "last_reinforced_turn": 5,
        "related_entities": [],
        "valid_at": 5,
        "invalid_at": None,
    }])

    hot = HotStorage(StorageConfig(
        hot_uri=uri, cold_uri=str(tmp_path / "cold.db"), embedding_dim=4,
    ))
    records = hot.get_all()
    assert len(records) == 1
    assert records[0].fact_content == "John drove to the Pacific Northwest"
    assert records[0].valid_at_text is None


def test_a_fact_with_no_date_stores_none(tmp_path: Path):
    hot = HotStorage(StorageConfig(
        hot_uri=str(tmp_path / "hot"),
        cold_uri=str(tmp_path / "cold.db"),
        embedding_dim=4,
    ))
    hot.upsert(record(turn=5), [0.1, 0.2, 0.3, 0.4])
    assert hot.get_all()[0].valid_at_text is None
