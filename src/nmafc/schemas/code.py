"""Records for code memory: what a symbol is, and where it actually lives.

The conversational half of this system stores a summary of what was said. That
is the right shape for conversation, where the gist is the answer, and the wrong
shape for code, where the answer is the bytes. A summary reading "the retry
ladder caps at sixty seconds" is not the retry ladder, and an agent acting on it
writes something that does not compile.

So a code memory stores no prose at all. It stores a pointer -- file, symbol,
line span, content hash -- and the source is read back from the file when it is
needed. That is the hippocampal index idea the conversational side already wins
with, applied where it is exactly correct rather than merely useful: a fact
about a conversation can only point at a turn approximately, but a fact about a
function points at bytes on disk that either match the hash or do not.

Two consequences follow, and both are the reason this is cheap:

  - Nothing is embedded and nothing is summarised, so building the index costs
    no tokens and no API calls. It is a parse.
  - Staleness is exact. A conversational memory has to guess whether it has gone
    out of date, which is what decay and supersession are for and why both were
    measured to misfire. A code memory knows: the file's hash changed or it did
    not.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    # A module-level name bound by assignment: a constant, a compiled regex, a
    # configured logger, an alias like `scanstring = c_scanstring or
    # py_scanstring`. Indexing only `def` and `class` was tested against the
    # standard library and lost exactly these -- a question naming one got
    # nothing back, which is honest but not useful, because in real code the
    # answer to "what is the default timeout" is an assignment.
    BINDING = "binding"


class Symbol(BaseModel):
    """One definition, addressed by where it is rather than by what it says."""

    path: str = Field(description="Repository-relative path, forward slashes")
    qualname: str = Field(
        description="Dotted name within the module, e.g. QueryRouter.retrieve"
    )
    kind: SymbolKind
    lineno: int = Field(ge=1, description="First line of the definition")
    end_lineno: int = Field(ge=1, description="Last line, inclusive")
    signature: str = Field(
        default="",
        description="The def or class line, whitespace-normalised, no body",
    )
    docline: str = Field(
        default="",
        description="First line of the docstring, if there is one",
    )
    # Names referenced in the body: calls, base classes, decorators. These are
    # the edges of the association graph, and they are exact where a vector
    # neighbourhood is a guess -- a caller and its callee usually look nothing
    # alike, and two functions that do the same thing look identical.
    refs: list[str] = Field(default_factory=list)
    content_hash: str = Field(
        default="",
        description="sha256 of this symbol's source slice, for exact staleness",
    )

    @property
    def key(self) -> str:
        """Unique across the repository. `qualname` alone is not: every module
        with a `main` would collide, and collisions in an index that claims to
        be exact are worse than a missing entry."""
        return f"{self.path}::{self.qualname}"

    @property
    def span(self) -> int:
        return self.end_lineno - self.lineno + 1


class FileRecord(BaseModel):
    """A parsed file and the hash that says whether the parse is still valid."""

    path: str
    content_hash: str
    # Module-level `import x` / `from y import z`, as the names they bind. Used
    # to resolve a reference to the file that defines it before falling back to
    # a repository-wide name lookup.
    imports: dict[str, str] = Field(
        default_factory=dict,
        description="Bound name -> dotted module path it came from",
    )
    symbol_keys: list[str] = Field(default_factory=list)
