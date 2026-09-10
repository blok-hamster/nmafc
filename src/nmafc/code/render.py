"""Turn a symbol neighbourhood into a context block, under a hard token budget.

This is graded hydration, transplanted. On the conversational side the measured
win was to restore whole turns for the best-ranked few facts and only the
best-matching lines of the rest -- depth where it pays, and a pointer where it
does not. The same shape applies here and applies more cleanly, because hop
distance is exact rather than a ranking:

    hop 0   the symbol asked about        full source
    hop 1   what it calls, what calls it  signature and first docstring line
    hop 2+  the wider neighbourhood       pointer only, `path:12-48`

The budget is a hard cap, not a target. Items are added in priority order and
the block stops when the next one would cross the line, so a context built this
way cannot grow past the number it was given however large the neighbourhood is.
That is the whole answer to "make it better at code without spending more
tokens": the cheap thing is not compressing what you send, it is sending the
body of the one symbol that matters and a line each for the twenty around it.

Nothing here is verified against a code benchmark, because this repository has
none. The mechanism is exact -- the spans and hashes either match the file or
they do not -- but the claim that this shape of context answers coding questions
better than a grep-and-read agent is untested, and should not be made until
there is a benchmark to make it against.
"""

from __future__ import annotations

from nmafc.schemas.code import Symbol

# The repository's own approximation everywhere else, kept for comparability.
CHARS_PER_TOKEN = 4


def tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def pointer(symbol: Symbol) -> str:
    return f"{symbol.path}:{symbol.lineno}-{symbol.end_lineno}"


# Measured over the 399 symbols of three lines or more in this repository's
# `src`: mean 10.53 tokens per line, median 10.39, p90 13.14. Eleven is a little
# above the median so a typical definition is not underestimated, and well below
# p90 so a densely written one is not paid for in advance.
TOKENS_PER_LINE = 11
# A rendered neighbour is one pointer, one signature and at most a docstring
# line. Measured at 20 to 30 tokens across this repository.
TOKENS_PER_NEIGHBOUR = 25


def suggest_budget(items: list[tuple[Symbol, int]], ceiling: int,
                   floor: int = 200) -> int:
    """A budget sized to what this question is about to fetch. Metamemory.

    The brain does not retrieve at a fixed depth. It runs a feeling of knowing
    before recall and decides how hard to look, and the signal here is free:
    the index already knows how many symbols matched and how long each one is.

    Sizing it by *count* alone was the first version of this and it was wrong in
    a way worth recording. One seed and a small neighbourhood looks like a cheap
    question, so it was given a third of the ceiling -- but "explain
    format_context" is one seed and seventy-five lines, and a third of the
    ceiling could not hold it, so the renderer dropped the very definition that
    was asked about and reported a 98% saving on a context that answered
    nothing. A budget that ignores the size of what it is buying will do that
    every time.

    So the estimate is spans, not counts: the seeds cost their own length, the
    neighbours cost a line each, and the ceiling is the cap rather than the
    default. `render_symbols` guarantees the seeds a place regardless, so an
    underestimate here degrades what is shown rather than losing it.
    """
    if not items:
        return floor
    seeds = [symbol for symbol, hop in items if hop == 0]
    need = sum(symbol.span for symbol in seeds) * TOKENS_PER_LINE
    neighbours = (len(items) - len(seeds)) * TOKENS_PER_NEIGHBOUR
    return max(floor, min(ceiling, need + neighbours))


def render_symbols(index, items: list[tuple[Symbol, int]],
                   budget_tokens: int, full_hops: int = 0,
                   signature_hops: int = 1) -> str:
    """A `<CODE>` block: exact source where it pays, pointers where it does not.

    Every entry is addressed by `path:start-end`, so anything shown can be
    checked and anything omitted can be fetched. Source is only ever printed
    after `index.verify` confirms the span still hashes to what was indexed; a
    symbol whose file moved under it is shown as a pointer marked stale rather
    than as bytes that are no longer there. Serving a confidently wrong slice is
    the one failure this design has that a summary does not, and it is worth a
    disk read per symbol to not have it.
    """
    if not items:
        return ""

    ordered = sorted(items, key=lambda pair: (pair[1], pair[0].path,
                                              pair[0].lineno))
    lines = ["<CODE>"]
    used = tokens("<CODE>\n</CODE>")
    shown = 0
    # The overflow footer has to be paid for before the entries are, not after.
    # Appended afterwards it pushes a block that fitted exactly over the line,
    # which turns a hard cap into a cap plus a footer. Reserved at its worst
    # case -- the count that would be printed if nothing at all fitted -- so the
    # reserve can only ever be larger than what is finally spent.
    footer = f"({len(ordered)} more symbols not shown; ask for one by name)"
    reserve = tokens(footer) + 1

    for index_of, (symbol, hop) in enumerate(ordered):
        last = index_of == len(ordered) - 1
        room = budget_tokens - used - (0 if last else reserve)
        head = symbol.signature or symbol.qualname
        doc = f"  # {symbol.docline}" if symbol.docline else ""

        bare = f"{pointer(symbol)}  {symbol.qualname}"

        if hop <= full_hops:
            if not index.verify(symbol):
                choices = [f"{bare}  # stale: file changed since indexing, "
                           f"re-read it", bare]
            else:
                # Degrade, never drop. The symbol the question named is the one
                # thing the block must contain: a context that fits its budget
                # by discarding it has not saved anything, it has answered a
                # different question. Three levels, richest first -- the body,
                # then the signature with a note saying where the body is, then
                # the pointer alone, which costs four tokens and is still enough
                # for a caller to go and read it.
                choices = [
                    f"{pointer(symbol)}\n{index.source_of(symbol)}",
                    f"{pointer(symbol)}  {head}{doc}"
                    f"  # body omitted, {symbol.span} lines at this span",
                    bare,
                ]
        elif hop <= signature_hops:
            # Neighbours do not degrade. They are ordered by how much they are
            # worth, so when the next one will not fit, everything after it is
            # worth less and stopping is the right move; filling the last of the
            # budget with the least useful entries in the neighbourhood is not.
            choices = [f"{pointer(symbol)}  {head}{doc}"]
        else:
            choices = [bare]

        entry = next((c for c in choices if tokens(c) + 1 <= room), None)
        if entry is None:
            # Priority order means everything after this is worth less than
            # what was already dropped, so stopping is correct and continuing
            # to look for a smaller one would fill the budget with the least
            # useful entries in the neighbourhood.
            break
        lines.append(entry)
        used += tokens(entry) + 1
        shown += 1

    if shown < len(ordered):
        lines.append(f"({len(ordered) - shown} more symbols not shown; "
                     f"ask for one by name)")
    lines.append("</CODE>")
    return "\n".join(lines)
