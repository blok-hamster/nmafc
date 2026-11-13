"""Anchored notes: stored as written, exact staleness, lexical recall, hard budget.

Everything here is offline. A note costs no model call and no embedding, so
there is nothing to mock: a scratch repository and a temp notes file are the
whole fixture.
"""

from __future__ import annotations

import pathlib

import pytest

from nmafc.code.notes import FRESH, MISSING, STALE, NoteStore, default_notes_file, render_notes
from nmafc.code.render import tokens

A_PY = (
    "def foo():\n"
    "    return 1\n"
    "\n"
    "\n"
    "class Box:\n"
    "    def get(self):\n"
    "        return foo()\n"
)


@pytest.fixture
def store(tmp_path: pathlib.Path) -> NoteStore:
    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / "a.py").write_text(A_PY, encoding="utf-8")
    (repo / "data.txt").write_text("rows\n", encoding="utf-8")
    return NoteStore(repo, file=tmp_path / "notes.jsonl")


def test_text_is_kept_verbatim_and_survives_reload(store: NoteStore):
    text = "The `patch` column is REVERSED for swesmith rows: 63/150, fix = reverse(patch)."
    note = store.add(text, "a.py", "foo")
    assert note.text == text
    assert note.anchor == "a.py::foo"

    again = NoteStore(store.root, file=store.file)
    assert [n.text for n in again.notes.values()] == [text]
    assert again.status(again.notes[note.id]) == FRESH


def test_symbol_anchor_ignores_edits_elsewhere_in_the_file(store: NoteStore):
    note = store.add("foo is the single source of truth for the default.", "a.py", "foo")

    # Insert lines above and change a different symbol: foo's own bytes are untouched,
    # even though its line numbers moved.
    (store.root / "a.py").write_text("import os\n\n" + A_PY.replace("return foo()", "return 2"),
                                     encoding="utf-8")
    assert store.status(note) == FRESH

    (store.root / "a.py").write_text(A_PY.replace("return 1", "return 99"), encoding="utf-8")
    assert store.status(note) == STALE


def test_file_anchor_goes_stale_on_any_change_and_works_for_non_python(store: NoteStore):
    note = store.add("Header row is missing in this file.", "data.txt")
    assert store.status(note) == FRESH
    (store.root / "data.txt").write_text("rows\nmore\n", encoding="utf-8")
    assert store.status(note) == STALE


def test_missing_when_the_anchor_is_gone(store: NoteStore):
    by_symbol = store.add("get delegates to foo.", "a.py", "Box.get")
    by_file = store.add("About the data file.", "data.txt")
    (store.root / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (store.root / "data.txt").unlink()
    assert store.status(by_symbol) == MISSING
    assert store.status(by_file) == MISSING


def test_a_note_that_cannot_be_anchored_is_refused(store: NoteStore):
    with pytest.raises(ValueError, match="not found"):
        store.add("x", "nope.py")
    with pytest.raises(ValueError, match="not found"):
        store.add("x", "a.py", "no_such_symbol")
    with pytest.raises(ValueError, match="text"):
        store.add("   ", "a.py")
    assert store.notes == {}


def test_bare_method_name_is_accepted_when_unambiguous(store: NoteStore):
    note = store.add("Returns foo() unchanged.", "a.py", "get")
    assert store.status(note) == FRESH


def test_search_is_lexical_and_returns_nothing_for_an_unrelated_question(store: NoteStore):
    store.add("Retry ladder caps at sixty seconds.", "a.py", "foo")
    store.add("Box.get must not be called before the index is built.", "a.py", "Box.get")

    hits = store.search("what is the retry cap?")
    assert [n.symbol for n, _, _ in hits] == ["foo"]

    assert store.search("kubernetes ingress certificates") == []
    assert store.search("the of and") == []          # stop words only


def test_identifier_subwords_match(store: NoteStore):
    store.add("has_text_frame reads the wrong attribute.", "a.py", "foo")
    assert len(store.search("hasTextFrame")) == 1
    assert len(store.search("text frame")) == 1


def test_fresh_notes_rank_before_stale_ones(store: NoteStore):
    stale = store.add("retry retry retry cap", "a.py", "foo")          # the better lexical match
    fresh = store.add("retry cap is set in Box.get", "a.py", "Box.get")
    (store.root / "a.py").write_text(A_PY.replace("return 1", "return 99"), encoding="utf-8")

    rows = store.search("retry cap")
    assert [(n.id, status) for n, status, _ in rows] == [(fresh.id, FRESH), (stale.id, STALE)]


def test_filter_by_anchor_without_a_query(store: NoteStore):
    store.add("about foo", "a.py", "foo")
    store.add("about get", "a.py", "Box.get")
    store.add("about data", "data.txt")
    assert len(store.search(path="a.py")) == 2
    assert [n.text for n, _, _ in store.search(symbol="get")] == ["about get"]
    assert len(store.search()) == 3


def test_remove_is_durable(store: NoteStore):
    note = store.add("temporary", "a.py")
    assert store.remove(note.id) is True
    assert store.remove(note.id) is False
    assert NoteStore(store.root, file=store.file).notes == {}


def test_render_marks_stale_and_respects_a_hard_budget(store: NoteStore):
    for i in range(20):
        store.add(f"finding number {i} about the retry ladder " + "x" * 80, "a.py", "foo")
    (store.root / "a.py").write_text(A_PY.replace("return 1", "return 99"), encoding="utf-8")
    rows = store.search("retry ladder", top_k=20)
    assert len(rows) == 20

    block, shown = render_notes(rows, budget_tokens=200)
    assert 0 < shown < 20
    assert tokens(block) <= 200
    assert block.startswith("<NOTES>") and block.endswith("</NOTES>")
    assert "[STALE" in block

    assert render_notes(rows, budget_tokens=0) == ("", 0)
    assert render_notes([], budget_tokens=500) == ("", 0)


def test_an_oversized_first_note_is_truncated_not_dropped(store: NoteStore):
    store.add("retry " + "y" * 4000, "a.py", "foo")
    block, shown = render_notes(store.search("retry"), budget_tokens=60)
    assert shown == 1
    assert tokens(block) <= 60
    assert block.rstrip().endswith("</NOTES>")


def test_default_file_lives_in_the_data_dir_not_the_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("NMAFC_DATA_DIR", str(tmp_path / "data"))
    repo = tmp_path / "proj"
    repo.mkdir()
    target = default_notes_file(repo)
    assert str(tmp_path / "data") in str(target)
    assert repo not in target.parents
    # Two repositories never share a notes file.
    other = tmp_path / "other"
    other.mkdir()
    assert default_notes_file(other) != target
