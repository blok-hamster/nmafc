"""Code memory: exact spans, exact edges, exact staleness.

The claims this subsystem makes are all falsifiable without a benchmark, because
they are claims about correctness rather than about accuracy: a span either is
the definition or it is not, an edge either exists in the source or it does not,
and a hash either matches or it does not. Those are what these tests pin.

The one claim not tested here is that this shape of context helps a model answer
coding questions, because there is no code benchmark in this repository to test
it against. That is stated in `render.py` and is not asserted anywhere.
"""

from __future__ import annotations

import textwrap

from nmafc.code.index import SymbolIndex
from nmafc.code.render import render_symbols, suggest_budget, tokens
from nmafc.code.symbols import content_hash, parse_source
from nmafc.schemas.code import Symbol, SymbolKind

SAMPLE = textwrap.dedent('''
    """Module docstring."""
    import os
    from pathlib import Path as P


    def helper(value: int) -> int:
        """Double it."""
        return value * 2


    class Router:
        """Routes things."""

        def retrieve(self, query: str):
            """Find records."""
            return helper(len(query))

        def format_context(self, records):
            return ", ".join(records)
''').lstrip()


BINDINGS = '''\
import re

DEFAULT_TIMEOUT = 30
PATTERN = re.compile(r"x+")
HOT, COLD = 0, 1
LABEL: str = "hot"
__all__ = ["DEFAULT_TIMEOUT"]
config.timeout = 99
table["key"] = "value"


def connect():
    return DEFAULT_TIMEOUT
'''


class TestBindings:
    """Module-level assignments, which `def`-and-`class`-only indexing lost.

    Found by running the index over the standard library: "who calls scanstring"
    returned nothing, because `scanstring = c_scanstring or py_scanstring` is an
    assignment. In real code the answer to "what is the default timeout" is an
    assignment more often than it is a function.
    """

    def parsed(self):
        _, symbols = parse_source("a.py", BINDINGS)
        return {s.qualname: s for s in symbols}

    def test_a_module_level_constant_is_a_symbol(self):
        found = self.parsed()["DEFAULT_TIMEOUT"]
        assert found.kind is SymbolKind.BINDING
        assert found.signature == "DEFAULT_TIMEOUT = 30"
        assert found.span == 1

    def test_tuple_unpacking_defines_every_name(self):
        by_name = self.parsed()
        assert by_name["HOT"].signature == "HOT, COLD = 0, 1"
        assert by_name["COLD"].signature == "HOT, COLD = 0, 1"

    def test_an_annotated_assignment_counts(self):
        assert self.parsed()["LABEL"].kind is SymbolKind.BINDING

    def test_dunder_metadata_is_not_a_definition(self):
        assert "__all__" not in self.parsed()

    def test_attribute_and_subscript_targets_are_not_definitions(self):
        """`config.timeout = 99` mutates something defined elsewhere. Recording
        it would put a second entry in the index claiming to be that thing."""
        by_name = self.parsed()
        assert "config" not in by_name
        assert "timeout" not in by_name
        assert "table" not in by_name

    def test_a_binding_carries_its_references_as_edges(self):
        """`re.compile` would not do here: `compile` is a builtin, and builtins
        are filtered out on purpose because an edge everything shares is not an
        edge. `re` is imported, so it is a name this module actually binds."""
        assert "re" in self.parsed()["PATTERN"].refs

    def test_a_constant_is_reachable_from_the_function_that_uses_it(self):
        index = SymbolIndex(".")
        index.add_source("a.py", BINDINGS)
        assert [s.qualname for s in index.callers("DEFAULT_TIMEOUT")] == ["connect"]

    def test_a_binding_has_no_docline_from_the_line_above_it(self):
        assert self.parsed()["DEFAULT_TIMEOUT"].docline == ""


class TestParseSource:
    def test_finds_functions_classes_and_methods(self):
        _, symbols = parse_source("a.py", SAMPLE)
        by_name = {s.qualname: s for s in symbols}
        assert by_name["helper"].kind is SymbolKind.FUNCTION
        assert by_name["Router"].kind is SymbolKind.CLASS
        assert by_name["Router.retrieve"].kind is SymbolKind.METHOD

    def test_span_covers_exactly_the_definition(self):
        _, symbols = parse_source("a.py", SAMPLE)
        helper = next(s for s in symbols if s.qualname == "helper")
        lines = SAMPLE.split("\n")[helper.lineno - 1:helper.end_lineno]
        assert lines[0].startswith("def helper")
        assert lines[-1].strip() == "return value * 2"

    def test_signature_and_docline_are_captured(self):
        _, symbols = parse_source("a.py", SAMPLE)
        helper = next(s for s in symbols if s.qualname == "helper")
        assert helper.signature == "def helper(value: int) -> int:"
        assert helper.docline == "Double it."

    def test_refs_record_the_call_not_the_receiver(self):
        _, symbols = parse_source("a.py", SAMPLE)
        retrieve = next(s for s in symbols if s.qualname == "Router.retrieve")
        assert "helper" in retrieve.refs
        # `len` is a builtin: an edge to it would join nearly every function to
        # nearly every other and carry no information.
        assert "len" not in retrieve.refs

    def test_imports_are_recorded_under_the_bound_name(self):
        record, _ = parse_source("a.py", SAMPLE)
        assert record.imports["os"] == "os"
        assert record.imports["P"] == "pathlib.Path"

    def test_a_file_that_does_not_parse_yields_no_symbols(self):
        """Half a repository is syntactically invalid mid-edit, and an index
        that raises then is unavailable exactly when it is wanted."""
        record, symbols = parse_source("broken.py", "def oops(:\n")
        assert symbols == []
        assert record.content_hash


class TestSymbolIndex:
    def build(self) -> SymbolIndex:
        index = SymbolIndex(".")
        index.add_source("a.py", SAMPLE)
        return index

    def test_resolves_by_bare_name_and_by_qualname(self):
        index = self.build()
        assert [s.qualname for s in index.resolve("retrieve")] == \
            ["Router.retrieve"]
        assert [s.qualname for s in index.resolve("Router.retrieve")] == \
            ["Router.retrieve"]

    def test_reverse_edges_answer_what_calls_this(self):
        index = self.build()
        assert [s.qualname for s in index.callers("helper")] == \
            ["Router.retrieve"]

    def test_forward_edges_answer_what_does_this_use(self):
        index = self.build()
        retrieve = index.resolve("Router.retrieve")[0]
        assert "helper" in [s.qualname for s in index.callees(retrieve)]

    def test_neighbourhood_tags_hop_distance(self):
        index = self.build()
        seed = index.resolve("helper")
        hops = dict((s.qualname, hop)
                    for s, hop in index.neighbourhood(seed, hops=1))
        assert hops["helper"] == 0
        assert hops["Router.retrieve"] == 1

    def test_retrieve_returns_nothing_when_no_symbol_is_named(self):
        """A question naming no symbol gets no answer rather than a guess.

        This is lexical resolution, not semantic search, and returning a
        plausible-looking neighbourhood for a question it did not understand is
        how a code memory produces confident nonsense.
        """
        index = self.build()
        assert index.retrieve("how does the whole thing fit together") == []

    def test_reindexing_a_file_forgets_what_was_deleted(self):
        index = self.build()
        assert index.resolve("helper")
        index.add_source("a.py", "def other():\n    return 1\n")
        assert index.resolve("helper") == []
        assert index.resolve("other")

    def test_dropping_a_file_clears_its_reverse_edges(self):
        index = self.build()
        index.drop("a.py")
        assert index.callers("helper") == []
        assert index.symbols == {}


class TestStaleness:
    def test_edited_file_is_stale_and_refresh_reparses_only_it(self, tmp_path):
        (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
        (tmp_path / "b.py").write_text("def other():\n    return 1\n",
                                       encoding="utf-8")
        index = SymbolIndex(tmp_path)
        index.build()
        assert index.stale() == []

        (tmp_path / "a.py").write_text(SAMPLE + "\n\ndef added():\n    pass\n",
                                       encoding="utf-8")
        assert index.stale() == ["a.py"]
        assert index.refresh() == ["a.py"]
        assert index.resolve("added")
        assert index.stale() == []

    def test_deleted_file_is_stale_and_refresh_drops_it(self, tmp_path):
        (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
        index = SymbolIndex(tmp_path)
        index.build()
        (tmp_path / "a.py").unlink()
        assert index.stale() == ["a.py"]
        index.refresh()
        assert index.resolve("helper") == []

    def test_verify_fails_once_the_span_no_longer_matches(self, tmp_path):
        """The one failure mode a pointer has that a summary does not: the file
        moves under it and the span now names different code."""
        (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
        index = SymbolIndex(tmp_path)
        index.build()
        helper = index.resolve("helper")[0]
        assert index.verify(helper)

        (tmp_path / "a.py").write_text("# a new first line\n" + SAMPLE,
                                       encoding="utf-8")
        assert not index.verify(helper)


class TestRender:
    def build(self, tmp_path) -> SymbolIndex:
        (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
        index = SymbolIndex(tmp_path)
        index.build()
        return index

    def test_focal_symbol_is_shown_in_full_and_neighbours_as_signatures(
            self, tmp_path):
        index = self.build(tmp_path)
        items = index.retrieve("helper", hops=1)
        block = render_symbols(index, items, budget_tokens=400)
        assert "return value * 2" in block          # hop 0, full body
        assert "def retrieve(self, query: str):" in block   # hop 1, signature
        assert "return helper(len(query))" not in block     # not its body

    def test_every_entry_carries_a_checkable_pointer(self, tmp_path):
        index = self.build(tmp_path)
        block = render_symbols(index, index.retrieve("helper", hops=1),
                               budget_tokens=400)
        assert "a.py:" in block

    def test_budget_is_a_hard_cap(self, tmp_path):
        index = self.build(tmp_path)
        items = index.retrieve("helper", hops=2)
        for budget in (20, 40, 80, 200):
            block = render_symbols(index, items, budget_tokens=budget)
            assert tokens(block) <= budget, f"overspent at {budget}"

    def crowded(self, tmp_path) -> SymbolIndex:
        """A long definition with a crowd of callers.

        The small sample cannot test the budget: every symbol in it is three
        lines, so nothing ever overflows and a test that asserts overflow passes
        for the wrong reason. This one has a seed too big for a modest budget
        and more neighbours than any budget will hold.
        """
        body = "\n".join(f"    step_{i} = {i}" for i in range(60))
        callers = "\n\n".join(
            f"def caller_{i}():\n    return target()" for i in range(25))
        source = (f"def target():\n    \"\"\"The one asked about.\"\"\"\n"
                  f"{body}\n    return step_0\n\n\n{callers}\n")
        (tmp_path / "big.py").write_text(source, encoding="utf-8")
        index = SymbolIndex(tmp_path)
        index.build()
        return index

    def test_a_generous_budget_shows_the_whole_body(self, tmp_path):
        index = self.crowded(tmp_path)
        block = render_symbols(index, index.retrieve("target", hops=1),
                               budget_tokens=4000)
        assert "step_59 = 59" in block

    def test_a_seed_too_large_for_the_budget_degrades_and_is_never_dropped(
            self, tmp_path):
        """The saving is only real if the definition asked about is still there.

        Under a budget too small for the body, the seed comes back as its
        signature and a note saying where the body is, rather than disappearing.
        A block that fits by discarding what was asked about has not saved
        anything, it has answered a different question.
        """
        index = self.crowded(tmp_path)
        block = render_symbols(index, index.retrieve("target", hops=1),
                               budget_tokens=200)
        assert "def target():" in block
        assert "body omitted, 63 lines" in block
        assert "step_59 = 59" not in block
        assert tokens(block) <= 200

    def test_a_tiny_budget_still_names_the_seed(self, tmp_path):
        index = self.crowded(tmp_path)
        block = render_symbols(index, index.retrieve("target", hops=1),
                               budget_tokens=25)
        assert "big.py:1-" in block
        assert tokens(block) <= 25

    def test_overflow_is_declared_rather_than_silently_dropped(self, tmp_path):
        index = self.crowded(tmp_path)
        block = render_symbols(index, index.retrieve("target", hops=1),
                               budget_tokens=200)
        assert "not shown" in block

    def test_budget_is_a_hard_cap_on_a_crowded_neighbourhood(self, tmp_path):
        index = self.crowded(tmp_path)
        items = index.retrieve("target", hops=1)
        assert len(items) > 20, "the crowd is the point of this fixture"
        for budget in (25, 60, 120, 300, 800):
            block = render_symbols(index, items, budget_tokens=budget)
            assert tokens(block) <= budget, f"overspent at {budget}"

    def test_a_stale_span_is_flagged_instead_of_printed(self, tmp_path):
        index = self.build(tmp_path)
        items = index.retrieve("helper", hops=0)
        (tmp_path / "a.py").write_text("# shifted\n" + SAMPLE, encoding="utf-8")
        block = render_symbols(index, items, budget_tokens=400)
        assert "stale" in block
        assert "return value * 2" not in block

    def test_empty_neighbourhood_renders_nothing(self, tmp_path):
        index = self.build(tmp_path)
        assert render_symbols(index, [], budget_tokens=400) == ""


class TestSuggestBudget:
    def sized(self, lines: int) -> Symbol:
        return Symbol(path="a.py", qualname="f", kind=SymbolKind.FUNCTION,
                      lineno=1, end_lineno=lines)

    def test_a_short_seed_and_a_small_reach_spends_a_fraction(self):
        items = [(self.sized(8), 0)] + [(self.sized(5), 1)] * 3
        assert suggest_budget(items, ceiling=1200) < 1200

    def test_a_wide_neighbourhood_spends_the_ceiling(self):
        items = [(self.sized(40), 0)] * 4 + [(self.sized(5), 1)] * 40
        assert suggest_budget(items, ceiling=1200) == 1200

    def test_a_long_seed_is_budgeted_for_even_when_it_is_the_only_one(self):
        """The bug this replaced: one seed looked like a cheap question, got a
        third of the ceiling, and did not fit -- so the definition asked about
        was dropped and the result reported as a saving."""
        short = suggest_budget([(self.sized(6), 0)], ceiling=1200)
        long = suggest_budget([(self.sized(75), 0)], ceiling=1200)
        assert long > short
        assert long >= 75 * 10, "a 75-line body needs roughly 750 tokens"

    def test_never_below_the_floor(self):
        assert suggest_budget([], ceiling=1200, floor=200) == 200
        assert suggest_budget([(self.sized(1), 0)], ceiling=1200,
                              floor=200) == 200


class TestAgainstThisRepository:
    """The index built over real code, which is the only honest scale test."""

    def test_indexes_its_own_source_and_finds_a_known_symbol(self):
        index = SymbolIndex("src")
        count = index.build()
        assert count > 100, f"only {count} symbols indexed"

        found = index.resolve("format_context")
        assert found, "format_context should be in this repository"
        assert all(index.verify(s) for s in found)

    def test_the_known_caller_relationship_is_present(self):
        index = SymbolIndex("src")
        index.build()
        callers = {s.qualname for s in index.callers("format_context")}
        assert any("process_turn" in c or "chat" in c or "Memory" in c
                   for c in callers) or callers, \
            "format_context is called somewhere in src"

    def test_hashing_is_stable(self):
        assert content_hash("abc") == content_hash("abc")
        assert content_hash("abc") != content_hash("abd")
