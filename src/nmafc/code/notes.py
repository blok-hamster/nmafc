"""Anchored notes: findings about code, kept as written, valid until the code moves.

The two stores this package already has leave a gap between them. A symbol
record holds no prose at all, and conversational memory holds prose it has
rewritten into facts about a person. What an agent learns while working on a
repository is neither: "this function decides X", "the `patch` column is
reversed for these rows", "do not call this before the index is built". That is
prose, it has to survive verbatim because the detail is the value, and it is
about bytes on disk rather than about anyone.

So a note is stored exactly as it was written and anchored to what it is about:
a file, and optionally one symbol inside it, together with the hash of that
anchor at the time of writing. Three things follow, and they are the same three
the rest of the code-memory path was built for:

  - **Writing costs nothing.** No extraction call and no embedding, so a note is
    durable in the time it takes to append a line. The conversational
    `remember` path waits on a retrieval and a model call; this does neither.
  - **Staleness is exact.** A note is fresh while its anchor still hashes to
    what it did, stale once it does not, and missing if the anchor is gone.
    Nothing decays with time or with turn count, because a finding about a
    function does not get less true on Tuesday. It gets less true on an edit.
  - **A stale note is flagged, never hidden.** Knowing that something was once
    established here, and that the ground has since moved, is exactly what an
    agent needs before it decides whether to re-check. Dropping the note would
    throw that away; serving it unmarked would be the confidently wrong answer
    this package exists to avoid.

A symbol anchor hashes the symbol's own source slice, so edits elsewhere in the
same file, including ones that shift its line numbers, leave the note fresh. A
file anchor hashes the whole file and goes stale on any change to it.

Retrieval is lexical on purpose, for the reason `SymbolIndex.retrieve` is: a
query that shares no word with any note returns nothing rather than the nearest
plausible guess. Whether notes recalled this way make an agent cheaper or more
accurate on real coding tasks has not been measured in this repository.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from nmafc.code.render import tokens
from nmafc.code.symbols import content_hash, parse_source

FRESH = "fresh"
STALE = "stale"
MISSING = "missing"

_WORD = re.compile(r"[A-Za-z][a-z0-9]+|[A-Z]+(?![a-z])|\d+")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Words that carry no signal in a question about code. Kept short: an
# over-eager stop list silently drops identifiers that happen to be English.
_STOP = frozenset(
    "the a an and or of to in is it for on with this that be as at by from "
    "not are was does do how what why when where which".split()
)


class Note(BaseModel):
    """One finding, and the bytes it was true of when it was written."""

    id: str
    text: str
    path: str = Field(description="Repository-relative path, forward slashes")
    symbol: str = Field(default="", description="Qualified name inside the file, or empty")
    anchor_hash: str = Field(description="Hash of the anchor when the note was written")
    created_at: str = ""

    @property
    def anchor(self) -> str:
        return f"{self.path}::{self.symbol}" if self.symbol else self.path


def _fold(word: str) -> str:
    """Fold a plural onto its singular, so `caps` matches `cap` and `patches`, `patch`.

    Deliberately crude: it only has to be applied the same way to notes and to
    questions. A real stemmer would conflate more, and conflating identifiers
    is how a lexical search starts returning things nobody asked for.
    """
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        if word.endswith(("ches", "shes", "xes", "zes")):
            return word[:-2]
        return word[:-1]
    return word


def _terms(text: str) -> list[str]:
    """Identifier sub-words, so `has_text_frame` and `hasTextFrame` both match `frame`."""
    out: list[str] = []
    for ident in _IDENT.findall(text):
        for part in _WORD.findall(ident):
            part = part.lower()
            if len(part) > 1 and part not in _STOP:
                out.append(_fold(part))
    return out


# A note is returned only if it covers at least this share of the question's
# words, weighted by how rare each word is across the notes. Without it, one
# common word in common ("patch", "test") was enough to return several
# unrelated notes for a question the store knew nothing about. 0.3 comes from
# a small probe (12 questions over 16 notes: relevant notes covered 0.33-1.00,
# loose matches on unanswerable questions 0.11-0.21), so treat it as a starting
# point and not as a tuned constant.
DEFAULT_MIN_COVERAGE = 0.3


def default_notes_file(root: Path | str) -> Path:
    """Per-repository file under the NMAFC data directory, never inside the repo."""
    # Imported here so `nmafc.code` stays importable without the storage stack.
    from nmafc.storage.config import default_data_dir

    resolved = str(Path(root).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return default_data_dir() / "code_notes" / f"{digest}.jsonl"


class NoteStore:
    """Notes for one repository, in an append-only JSONL file."""

    def __init__(self, root: Path | str = ".", file: Path | str | None = None) -> None:
        self.root = Path(root)
        self.file = Path(file) if file is not None else default_notes_file(self.root)
        self.notes: dict[str, Note] = {}
        self._load()

    # --- persistence ----------------------------------------------------

    def _load(self) -> None:
        if not self.file.is_file():
            return
        for line in self.file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("deleted"):
                self.notes.pop(row.get("id", ""), None)
                continue
            try:
                note = Note(**row)
            except (TypeError, ValueError):
                continue
            self.notes[note.id] = note

    def _append(self, row: dict) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- anchors --------------------------------------------------------

    def _relative(self, path: str) -> str:
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(self.root.resolve()).as_posix()
            except ValueError:
                return candidate.as_posix()
        return candidate.as_posix()

    def _anchor_hash(self, path: str, symbol: str) -> str | None:
        """Hash of the anchor as it is on disk now, or None if it is not there."""
        try:
            source = (self.root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        if not symbol:
            return content_hash(source)
        try:
            _, symbols = parse_source(path, source)
        except (SyntaxError, ValueError):
            return None
        for found in symbols:
            if found.qualname == symbol:
                return found.content_hash
        bare = [s for s in symbols if s.qualname.rsplit(".", 1)[-1] == symbol]
        # A bare name is accepted only when it is unambiguous in the file.
        return bare[0].content_hash if len(bare) == 1 else None

    def status(self, note: Note) -> str:
        current = self._anchor_hash(note.path, note.symbol)
        if current is None:
            return MISSING
        return FRESH if current == note.anchor_hash else STALE

    # --- writing --------------------------------------------------------

    def add(self, text: str, path: str, symbol: str = "") -> Note:
        """Store a finding against a file, or against one symbol in it.

        Raises ValueError when the anchor does not exist. A note that cannot be
        checked later is a note that can never be flagged stale, and that is
        the one kind this store must not hold.
        """
        text = text.strip()
        if not text:
            raise ValueError("A note needs some text.")
        rel = self._relative(path)
        anchor_hash = self._anchor_hash(rel, symbol)
        if anchor_hash is None:
            what = f"symbol {symbol!r} in {rel}" if symbol else f"file {rel}"
            raise ValueError(f"Cannot anchor a note to {what}: it was not found under {self.root}.")
        note = Note(
            id=uuid.uuid4().hex[:12],
            text=text,
            path=rel,
            symbol=symbol,
            anchor_hash=anchor_hash,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.notes[note.id] = note
        self._append(note.model_dump())
        return note

    def remove(self, note_id: str) -> bool:
        if self.notes.pop(note_id, None) is None:
            return False
        self._append({"id": note_id, "deleted": True})
        return True

    # --- reading --------------------------------------------------------

    def search(self, query: str = "", path: str | None = None, symbol: str | None = None,
               top_k: int = 8,
               min_coverage: float = DEFAULT_MIN_COVERAGE) -> list[tuple[Note, str, float]]:
        """Notes matching a question and/or an anchor, as (note, status, score).

        Fresh notes come before stale and missing ones, then by score. With no
        query the anchor filter alone decides, newest first. A note that covers
        less than `min_coverage` of the question's words is not returned.
        """
        pool = list(self.notes.values())
        if path:
            rel = self._relative(path)
            pool = [n for n in pool if n.path == rel]
        if symbol:
            pool = [n for n in pool
                    if n.symbol == symbol or n.symbol.rsplit(".", 1)[-1] == symbol]
        if not pool:
            return []

        wanted = _terms(query)
        if wanted:
            docs = [_terms(f"{n.text} {n.path} {n.symbol}") for n in pool]
            scores = _bm25(docs, wanted)
            cover = _coverage(docs, wanted)
            ranked = [(n, s) for n, s, c in zip(pool, scores, cover)
                      if s > 0 and c >= min_coverage]
        elif query.strip():
            ranked = []          # a query made only of stop words matches nothing
        else:
            ranked = [(n, 0.0) for n in pool]

        order = {FRESH: 0, STALE: 1, MISSING: 2}
        out = [(n, self.status(n), s) for n, s in ranked]
        if wanted:
            out.sort(key=lambda row: (order[row[1]], -row[2]))
        else:
            out.sort(key=lambda row: (order[row[1]], _neg_time(row[0].created_at)))
        return out[:max(top_k, 0)]


def _neg_time(stamp: str) -> float:
    try:
        return -datetime.fromisoformat(stamp).timestamp()
    except ValueError:
        return 0.0


def _coverage(docs: list[list[str]], query: list[str]) -> list[float]:
    """Share of the question each note covers, weighting rare words above common ones.

    A word that appears in no note at all carries the most weight, which is the
    point: a question whose most specific word is unknown to the store should
    not be answered by notes that merely share its ordinary words.
    """
    n = len(docs)
    sets = [set(doc) for doc in docs]
    df: Counter[str] = Counter()
    for seen in sets:
        df.update(seen)
    words = set(query)
    weight = {w: math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in words}
    total = sum(weight.values()) or 1.0
    return [sum(weight[w] for w in words & seen) / total for seen in sets]


def _bm25(docs: list[list[str]], query: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    n = len(docs)
    avg = (sum(len(d) for d in docs) / n) if n else 1.0
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    out: list[float] = []
    for doc in docs:
        tf = Counter(doc)
        score = 0.0
        for term in set(query):
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(doc) / (avg or 1.0)))
        out.append(score)
    return out


def render_notes(rows: list[tuple[Note, str, float]], budget_tokens: int) -> tuple[str, int]:
    """A `<NOTES>` block under a hard token budget. Returns (block, notes shown).

    The budget is a cap, as it is for symbols: rows are added in rank order and
    the block stops before the one that would cross the line. The first row is
    truncated rather than dropped if it alone is too large, because an empty
    answer to a question that has one is the worse failure.
    """
    if not rows or budget_tokens <= 0:
        return "", 0
    def block(entries: list[str]) -> str:
        return "\n".join(["<NOTES>", *entries, "</NOTES>"])

    # The cap is checked against the block as it will actually be returned.
    # Summing per-line estimates drifts by a token or two from the rounding,
    # and a budget that is exceeded by one is not a hard budget.
    lines: list[str] = []
    for note, status, _ in rows:
        mark = ""
        if status == STALE:
            mark = " [STALE: anchor changed since this was written]"
        elif status == MISSING:
            mark = " [MISSING: anchor no longer exists]"
        entry = f"- ({note.anchor}){mark} {note.text}"
        if tokens(block([*lines, entry])) > budget_tokens:
            if lines:
                break
            # tokens() is len // 4, so a block fits iff len <= budget * 4 + 3.
            room = budget_tokens * 4 + 3 - len(block([""])) - 4
            if room <= 0:
                break
            entry = entry[:room].rstrip() + " ..."
            if tokens(block([entry])) > budget_tokens:
                break
        lines.append(entry)
    if not lines:
        return "", 0
    return block(lines), len(lines)
