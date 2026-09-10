"""Code memory: exact pointers, an exact graph, and exact staleness.

This is deliberately a separate subsystem from the conversational memory rather
than a new record type inside it, because the three mechanisms that make the
conversational side work are all wrong here:

  - **Summarisation.** Conversation is answered from the gist. Code is answered
    from the bytes, and a paraphrase of a function is not the function.
  - **Decay.** A memory of a conversation genuinely does become less reliable
    with time. A function does not, and a theorem never does. What makes a code
    memory stale is an edit, which is checkable exactly.
  - **Vector association.** Spreading activation over embeddings is a reasonable
    guess at what is related in a conversation. In a repository the relation is
    written down -- this calls that, this imports that, this test covers that --
    and guessing at what is recorded is strictly worse than reading it.

What carries across is the part that was measured to win: hydration. A fact
points at a turn and the turn is restored on demand; a symbol points at a line
span and the source is restored on demand. The difference is that the pointer is
exact and the restore can be verified against a hash.

Nothing in here calls a language model or an embedding provider. Indexing a
repository is a parse, so it costs no tokens, finishes in milliseconds per file,
and can be re-run on every edit rather than nightly.
"""

from nmafc.code.index import SymbolIndex
from nmafc.code.render import render_symbols
from nmafc.code.symbols import parse_source

__all__ = ["SymbolIndex", "parse_source", "render_symbols"]
