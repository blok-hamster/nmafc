# Paired run, 10 September — per-question data

Everything the main README's results section claims is recomputable from these
files. Start with:

```bash
python scripts/benchmarks/results/paired_2026_09_10/summarise.py
```

Answering `azure_v1/DeepSeek-V4-Pro`, embeddings
`azure_v1/text-embedding-3-small`, judge `azure_v1/Kimi-K2.6`.

## Why the rows hold two arms

Both arms answered every question in the same window against the same persisted
stores, and both predictions were written to the same row. That is what makes
McNemar exact applicable: the 1,473 questions where the arms agree carry no
information about which is better, and the 512 where they disagree are the whole
of the evidence. Running two arms separately and subtracting the totals throws
that resolution away and imports run-to-run drift on top.

The drift is small but not zero. Re-running our own arm against its saved output
moved 0.1 points at n=841, gross movement +21/−22 — that is, 43 individual
answers changed to produce a one-tenth-of-a-point difference in the total. Net
figures hide that; gross figures are reported throughout for the same reason.

## Files

| file | n | what it is |
|---|---|---|
| `locomo_all.json` | 1,985 | the headline run. Ours and RAG, every category |
| `commit_short_all.json` | 1,984 | prompt clause pushing the model to commit rather than refuse |
| `commit_exact.json` | 1,984 | second attempt at the same, with the wording blamed for the damage removed |
| `od_ship.json` | 830 | open-domain, shipped retrieval settings |
| `od_ground.json` | 830 | open-domain, grounding floor 0.003 |
| `od_wide.json` | 830 | open-domain, grounding 0.003 **and** hydration 10 |
| `lg_ship.json` | 322 | the questions `list_shape.wants_list` fires on, shipped settings |
| `lg_wide.json` | 322 | the same questions, wide settings |

## Columns

`locomo_all.json` carries both arms per row:

```
conv, question, category, gold, first,
ours_pred, ours_ok, ours_chars, ours_ms, ours_throttle,
rag_pred,  rag_ok,  rag_chars,  rag_ms,  rag_throttle
```

The single-arm files carry `category, chars, conv, gold, ok, pred, question,
rag_chars, rag_ok, rag_pred, shipped_ok, tag`. Note `shipped_ok`: each row
records what the shipped configuration did on that same question, so an arm can
be compared against the baseline without joining to another file. Joining by
question text against `locomo_all.json` is a trap the column names invite —
`ours_ok` there is `shipped_ok` here.

## Reading these honestly

- `ours_chars // 4` is the token approximation. Crude, applied identically to
  both arms.
- `*_throttle` records time spent waiting on the rate limiter. Latency
  comparisons across arms are close to meaningless without subtracting it, and
  arms that did not run interleaved should not be compared on latency at all.
- Adversarial is 446 of the 1,985 and we lose it by 14.6 points. It is included
  here and in every table. Published comparisons drop it; dropping the category
  you lose is not a measurement decision.
- Nothing below roughly 800 questions in this benchmark has survived being
  re-measured at full scale. Four separate results were mispriced by a small
  sample before that rule was adopted.
