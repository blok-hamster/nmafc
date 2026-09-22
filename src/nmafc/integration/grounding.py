"""Rank facts by the evidence behind them, not by the summary in front of them.

Open-domain is the category that decides the benchmark, and 47 of the 116
questions lost there had the gold answer sitting in the prompt already. The
model read it and chose a competing value. That is not retrieval, not decay and
not extraction recall, and no amount of extra context fixes it -- the answer was
in the window.

Reading them says what it is. Each question carries a qualifier that decides
between several equally good stored facts:

    What novel is Evan reading that he finds *gripping*?
      gold The Great Gatsby        ours The Last Devil to Die
    What is the name of Maria's puppy she got *two weeks before 11 Aug 2023*?
      gold Coco                    ours Shadow
    What city did Tim suggest for the team trip *next month*?
      gold Edinburgh, Scotland     ours Nashville, Austin, Miami, Chicago...

Extraction wrote "Evan is reading The Great Gatsby" and dropped *gripping*, so
both novels are stored as equally plain reading facts and nothing in the ranking
can tell them apart. RAG strips nothing, so its chunk still holds the word that
decides the answer -- which is most of why it wins this category.

The word is not lost, though. It is in the turn the fact came from, and
`turn_text` has kept every turn since hydration was built. What has been missing
is that the *query* never meets that text: reranking fuses vector rank, keyword
rank and graph distance, and after retrieval the question's own wording plays no
part at all.

So this scores each candidate's source turn against the question and hands the
ordering to RRF as one more list to fuse.

Two design points, and the second is the one that keeps this honest:

  * **Rarity is measured within the candidate pool, not against a corpus.**
    A word that appears in one of thirty candidate turns decides something; a
    word in all thirty decides nothing and is scored as nothing. That is IDF,
    and computing it over the
    pool needs no index, no statistics kept on disk and no second pass -- and it
    self-normalises per question, so a name that is rare in one conversation and
    everywhere in another is weighted correctly in both.

  * **Turns never enter the candidate pool.** `cold.py` says why raw turns are
    deliberately left out of `memory_fts`: indexed, they would compete with
    facts for the retrieval budget. Nothing here indexes them or admits them as
    candidates. A turn is reachable only through a fact that already won a slot,
    exactly as before; all that changes is the order those facts are read in.

Costs nothing in context. It reorders a list that was already going to be
printed, and reads turns already on disk.
"""

from __future__ import annotations

import math
import re

from nmafc.integration.quantities import quantity_cues

_WORD = re.compile(r"[a-z0-9']+")

# Deliberately the same small list `cue_words` uses, and for the same reason: a
# long stop list starts deciding which content words matter, and IDF over the
# pool already demotes anything that appears everywhere. Words like "reading"
# and "puppy" are not on it and do not need to be -- they turn up in most of the
# candidate turns for their own question, so the pool prices them at nearly zero
# without anybody deciding they are unimportant.
_STOPWORDS = frozenset(
    "a an and are as at be been but by can did do does for from had has have "
    "he her him his how i if in into is it its me my not of on or our she that "
    "the their them then there they this to was we were what when where which "
    "who why will with would you your".split()
)

# BM25's usual constants. `b` at 0.75 is the standard partial length
# normalisation, and it earns its place here: turns range from one line to a
# dozen, and without it the longest turn in the pool wins on the chance of
# containing a rare word rather than on being about the question.
_K1 = 1.5
_B = 0.75


def terms(text: str) -> list[str]:
    """Content words and numbers, in order, repeats kept.

    Repeats are kept because BM25 is defined on term frequency, and numbers are
    kept because `cue_words` drops every token of three characters or fewer --
    which silently removed every quantity below 100 from every lexical decision
    in the system until `quantities.py` found it. A question asking about "two
    weeks before 11 August" is almost all short tokens.
    """
    if not text:
        return []
    lowered = text.lower()
    words = [w for w in _WORD.findall(lowered)
             if w not in _STOPWORDS and (len(w) > 2 or w.isdigit())]
    return words + sorted(quantity_cues(lowered))


def source_scores(query: str, texts: dict[int, str]) -> dict[int, float]:
    """BM25 of the question against each candidate turn, scored within the pool.

    `texts` maps turn number to the verbatim turn. The return maps the same turn
    numbers to a score, and turns that share no rare wording with the question
    score zero rather than being omitted, so a caller can rank the whole pool
    without deciding what a missing entry means.

    An empty pool, an empty question, or a question whose every word appears in
    every turn all give zeros throughout, which the caller should read as "this
    signal has nothing to say here" -- and it is the right answer, not a
    degenerate one. Most questions are not disambiguation questions.
    """
    asked = set(terms(query))
    if not asked or not texts:
        return {turn: 0.0 for turn in texts}

    tokenised = {turn: terms(text) for turn, text in texts.items()}
    lengths = {turn: len(ws) for turn, ws in tokenised.items()}
    total = sum(lengths.values())
    if not total:
        return {turn: 0.0 for turn in texts}
    average = total / len(tokenised)

    n = len(tokenised)
    scores: dict[int, float] = {}
    # Document frequency within the pool. Computed once over the whole pool
    # rather than per turn, because a term's rarity is a property of the pool.
    document_freq = {
        term: sum(1 for ws in tokenised.values() if term in ws)
        for term in asked
    }

    for turn, words in tokenised.items():
        present = set(words)
        score = 0.0
        for term in asked:
            if term not in present:
                continue
            df = document_freq[term]
            if df >= n:
                # A term in every candidate turn separates nothing, and the
                # smoothed IDF below never quite reaches zero for it. Left as
                # written it pays a small score to every turn alike, which is
                # harmless for ordering but makes the all-zero return -- the one
                # signal a caller has that this mechanism has no opinion --
                # unreachable whenever the question and the pool share a word.
                # It also covers a pool of one, where nothing can be rare.
                continue
            # The usual BM25 probabilistic IDF, floored at zero. Unfloored it
            # goes negative for a term in more than half the pool, which would
            # let a turn be pushed *down* for containing a common word the
            # question also used -- punishing a turn for being on topic.
            idf = max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))
            freq = words.count(term)
            norm = 1.0 - _B + _B * lengths[turn] / average
            score += idf * freq * (_K1 + 1.0) / (freq + _K1 * norm)
        scores[turn] = score
    return scores
