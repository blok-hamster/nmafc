import re
from types import SimpleNamespace

from nmafc.schemas.memory import MemoryRecord, MemoryType
from nmafc.web.routes.memory import get_record_source

_CLOCK_PREFIX = re.compile(r"^\d{1,2}:\d{2}\s*[ap]\.?m\.?\s+on\s+", re.I)


class RouterStub:
    """Enough of QueryRouter's date resolution for the route to run alone."""

    def __init__(self, timestamps: dict[int, str]):
        self._timestamps = timestamps

    def _date_for(self, turn):
        if not self._timestamps:
            return None
        earlier = [t for t in self._timestamps if t <= turn]
        return self._timestamps[max(earlier)] if earlier else None

    @staticmethod
    def _day(stamp: str) -> str:
        return _CLOCK_PREFIX.sub("", stamp)


def make_mem(record, timestamps, texts):
    hot = SimpleNamespace(get_record=lambda _id: record if _id == record.id else None)
    cold = SimpleNamespace(
        turn_timestamps=lambda: timestamps,
        text_for_turns=lambda turns: {t: texts[t] for t in turns if t in texts},
    )
    return SimpleNamespace(_hot=hot, _cold=cold, _router=RouterStub(timestamps))


def make_record(turn: int, valid_at=None, invalid_at=None, valid_at_text=None) -> MemoryRecord:
    return MemoryRecord(
        entity_name="melanie_sunrise_painting",
        fact_content="Melanie painted a sunrise in 2022.",
        memory_type=MemoryType.CORE_ANCHOR,
        created_at_turn=turn,
        valid_at=valid_at,
        invalid_at=invalid_at,
        valid_at_text=valid_at_text,
    )


class TestRecordSource:
    def test_dated_store_resolves_dates_and_source_text(self):
        rec = make_record(turn=5, valid_at=3, invalid_at=7)
        mem = make_mem(
            rec,
            timestamps={
                3: "7:18 pm on 27 May, 2023",
                5: "8:02 pm on 28 May, 2023",
                7: "9:00 pm on 30 May, 2023",
            },
            texts={3: "first mention", 5: "the turn it was said", 7: "superseded here"},
        )

        result = get_record_source(rec.id, mem)

        assert result["record"]["id"] == rec.id
        assert result["record"]["created_date"] == "28 May, 2023"
        assert result["validity"]["has_dates"] is True
        # The extractor's wording wins over the turn's date.
        assert result["validity"]["valid_at_text"] is None
        assert result["validity"]["valid_date"] == "27 May, 2023"
        assert result["validity"]["invalid_date"] == "30 May, 2023"
        assert [s["turn"] for s in result["source_turns"]] == [3, 5, 7]
        assert result["source_turns"][1]["text"] == "the turn it was said"
        assert result["source_turns"][1]["timestamp"] == "8:02 pm on 28 May, 2023"

    def test_extractor_wording_is_kept_in_validity(self):
        rec = make_record(turn=5, valid_at=5, valid_at_text="last summer")
        mem = make_mem(
            rec,
            timestamps={5: "7:18 pm on 27 May, 2023"},
            texts={5: "a turn"},
        )

        result = get_record_source(rec.id, mem)

        assert result["validity"]["valid_at_text"] == "last summer"
        assert result["validity"]["valid_date"] == "27 May, 2023"

    def test_undated_store_falls_back_to_nulls(self):
        rec = make_record(turn=12)
        mem = make_mem(rec, timestamps={}, texts={})

        result = get_record_source(rec.id, mem)

        assert result["validity"]["has_dates"] is False
        assert result["validity"]["valid_date"] is None
        assert result["record"]["created_date"] is None
        assert result["source_turns"] == [{"turn": 12, "timestamp": None, "text": None}]

    def test_missing_record_returns_error(self):
        rec = make_record(turn=1)
        mem = make_mem(rec, timestamps={}, texts={})

        result = get_record_source("does-not-exist", mem)

        assert result == {"error": "Record not found"}