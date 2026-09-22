"""The symbol graph: what is defined, what refers to it, and what has changed.

Three things this does that the conversational retriever cannot, all of them
because the relation is written down rather than inferred:

  - **Traversal is exact.** Spreading activation over embeddings guesses at what
    is related. Here `retrieve` calling `rerank` is a fact in the source, so a
    two-hop walk returns what is actually reachable rather than what is nearby
    in a vector space. Multi-hop is our weakest conversational category at
    44.8%, statistically level with plain RAG, and code reasoning is almost
    entirely multi-hop -- so this is the mechanism the domain most needs.

  - **Reverse edges exist.** "What calls this" is the question an agent editing
    code asks most, and it is unanswerable by similarity: a caller and its
    callee rarely resemble each other. It falls out of the forward edges for
    free.

  - **Staleness is decided, not estimated.** `stale()` compares hashes. There is
    no decay constant to tune and no supersession detector to misfire -- and the
    supersession detector was measured killing correct facts at 17.9%, so
    removing the need for it in this domain is a real subtraction of risk.

Rebuilding is incremental by hash, so the index tracks a repository under active
editing at the cost of re-parsing only the files that actually changed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from nmafc.code.symbols import content_hash, parse_source
from nmafc.schemas.code import FileRecord, Symbol

# Identifiers as they appear in a question: bare names, dotted paths, and the
# `path.py:120` form an agent pastes from a traceback. Deliberately lexical --
# this resolves names, and makes no claim to understand the sentence around
# them.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

DEFAULT_IGNORE = ("__pycache__", ".git", ".venv", "venv", "node_modules",
                  "build", "dist", ".mypy_cache", ".pytest_cache")


class SymbolIndex:
    """Definitions and the reference graph over them, for one repository."""

    def __init__(self, root: Path | str = ".") -> None:
        self.root = Path(root)
        self.symbols: dict[str, Symbol] = {}
        self.files: dict[str, FileRecord] = {}
        # Bare name -> keys. A name is not unique across a repository, so this
        # is one-to-many by construction and callers are given every candidate
        # rather than an arbitrary winner.
        self._by_name: dict[str, list[str]] = defaultdict(list)
        # Referenced name -> keys of the definitions that reference it. The
        # reverse edge, and the reason "what calls this" is answerable at all.
        self._referenced_by: dict[str, list[str]] = defaultdict(list)

    # --- building -------------------------------------------------------

    def add_source(self, path: str, source: str) -> int:
        """Index one file's text. Returns how many symbols it contributed.

        Re-adding a path replaces everything it previously contributed, so an
        edit that deletes a function deletes it from the index too. An index
        that only ever grows would keep answering with symbols that no longer
        exist, which is the code-memory version of the stale-fact problem.
        """
        self.drop(path)
        record, symbols = parse_source(path, source)
        self.files[path] = record
        for symbol in symbols:
            self.symbols[symbol.key] = symbol
            self._by_name[symbol.qualname].append(symbol.key)
            bare = symbol.qualname.rsplit(".", 1)[-1]
            if bare != symbol.qualname:
                self._by_name[bare].append(symbol.key)
            for ref in symbol.refs:
                self._referenced_by[ref].append(symbol.key)
        return len(symbols)

    def drop(self, path: str) -> None:
        """Forget a file entirely, including its edges."""
        record = self.files.pop(path, None)
        if record is None:
            return
        gone = set(record.symbol_keys)
        for key in gone:
            self.symbols.pop(key, None)
        for table in (self._by_name, self._referenced_by):
            for name in list(table):
                kept = [k for k in table[name] if k not in gone]
                if kept:
                    table[name] = kept
                else:
                    del table[name]

    def build(self, pattern: str = "**/*.py",
              ignore: tuple[str, ...] = DEFAULT_IGNORE) -> int:
        """Parse every matching file under the root. No API calls, no tokens."""
        count = 0
        for file in sorted(self.root.glob(pattern)):
            if any(part in ignore for part in file.parts):
                continue
            try:
                source = file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            count += self.add_source(self._relative(file), source)
        return count

    def _relative(self, file: Path) -> str:
        try:
            return file.relative_to(self.root).as_posix()
        except ValueError:
            return file.as_posix()

    # --- staleness ------------------------------------------------------

    def stale(self, pattern: str = "**/*.py",
              ignore: tuple[str, ...] = DEFAULT_IGNORE) -> list[str]:
        """Indexed files whose contents on disk no longer match, plus new ones.

        This is the whole of the invalidation policy. There is no threshold and
        no half-life: a symbol is out of date exactly when its file's bytes
        changed, and correct when they did not.
        """
        seen: set[str] = set()
        changed: list[str] = []
        for file in sorted(self.root.glob(pattern)):
            if any(part in ignore for part in file.parts):
                continue
            path = self._relative(file)
            seen.add(path)
            try:
                current = content_hash(file.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
            known = self.files.get(path)
            if known is None or known.content_hash != current:
                changed.append(path)
        # Deleted files are stale in the direction that matters most: they are
        # still being served and no longer exist.
        changed.extend(sorted(set(self.files) - seen))
        return changed

    def refresh(self, pattern: str = "**/*.py",
                ignore: tuple[str, ...] = DEFAULT_IGNORE) -> list[str]:
        """Re-parse only what changed. Returns the paths that were touched."""
        changed = self.stale(pattern, ignore)
        for path in changed:
            file = self.root / path
            if not file.is_file():
                self.drop(path)
                continue
            try:
                self.add_source(path, file.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                self.drop(path)
        return changed

    def verify(self, symbol: Symbol) -> bool:
        """Does this symbol's source slice still hash to what was indexed?

        Called before rendering. Handing back a line span that has since moved
        is how a pointer-based memory produces confidently wrong output, and it
        is the one failure mode this design has that a summary does not.
        """
        file = self.root / symbol.path
        try:
            lines = file.read_text(encoding="utf-8").split("\n")
        except (OSError, UnicodeDecodeError):
            return False
        body = "\n".join(lines[symbol.lineno - 1:symbol.end_lineno])
        return content_hash(body) == symbol.content_hash

    def source_of(self, symbol: Symbol) -> str:
        """The symbol's exact text, read back from disk."""
        file = self.root / symbol.path
        try:
            lines = file.read_text(encoding="utf-8").split("\n")
        except (OSError, UnicodeDecodeError):
            return ""
        return "\n".join(lines[symbol.lineno - 1:symbol.end_lineno])

    # --- lookup ---------------------------------------------------------

    def resolve(self, name: str) -> list[Symbol]:
        """Every definition matching a name, exact first then by bare name."""
        keys = self._by_name.get(name, [])
        if not keys and "." in name:
            keys = self._by_name.get(name.rsplit(".", 1)[-1], [])
        return [self.symbols[k] for k in dict.fromkeys(keys) if k in self.symbols]

    def callers(self, name: str) -> list[Symbol]:
        """Definitions that reference this name. The reverse edge."""
        keys = self._referenced_by.get(name.rsplit(".", 1)[-1], [])
        return [self.symbols[k] for k in dict.fromkeys(keys) if k in self.symbols]

    def callees(self, symbol: Symbol) -> list[Symbol]:
        """Definitions this one references, where the index knows them."""
        out: list[Symbol] = []
        seen: set[str] = set()
        for ref in symbol.refs:
            for found in self.resolve(ref):
                if found.key not in seen and found.key != symbol.key:
                    seen.add(found.key)
                    out.append(found)
        return out

    def neighbourhood(self, seeds: list[Symbol], hops: int = 1,
                      include_callers: bool = True) -> list[tuple[Symbol, int]]:
        """Seeds, then what they reach, breadth-first, tagged with hop distance.

        Both directions by default, because the two questions an agent asks
        about a symbol are "what does this use" and "what uses this", and only
        the first is answerable from the forward edges alone.

        Hop distance comes back with the symbols because it is what the renderer
        spends its budget on: hop 0 is shown in full, and the further out a
        symbol is the less of it is worth paying for.
        """
        out: list[tuple[Symbol, int]] = []
        seen: set[str] = set()
        frontier = [s for s in seeds if not (s.key in seen or seen.add(s.key))]
        out.extend((s, 0) for s in frontier)

        for hop in range(1, hops + 1):
            nxt: list[Symbol] = []
            for symbol in frontier:
                reachable = self.callees(symbol)
                if include_callers:
                    reachable = reachable + self.callers(symbol.qualname)
                for found in reachable:
                    if found.key not in seen:
                        seen.add(found.key)
                        nxt.append(found)
                        out.append((found, hop))
            if not nxt:
                break
            frontier = nxt
        return out

    def retrieve(self, query: str, hops: int = 1,
                 include_callers: bool = True) -> list[tuple[Symbol, int]]:
        """Symbols named in a question, plus their neighbourhood.

        Lexical on purpose. This resolves identifiers; it does not understand
        the sentence they sit in, and a question that names no symbol returns
        nothing rather than a plausible guess. Natural-language search over code
        is a separate problem and pretending to solve it here would put
        confident wrong answers in front of an agent.
        """
        seeds: list[Symbol] = []
        seen: set[str] = set()
        for token in IDENTIFIER.findall(query):
            for symbol in self.resolve(token):
                if symbol.key not in seen:
                    seen.add(symbol.key)
                    seeds.append(symbol)
        if not seeds:
            return []
        return self.neighbourhood(seeds, hops, include_callers)
