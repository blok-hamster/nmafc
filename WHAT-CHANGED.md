# What changed on `muna`

A summary of the work on this branch, why each piece exists, and what it was
worth. Every number here was measured on the standard LoCoMo protocol (the four
scored categories, 1,539–1,540 questions depending on the run) with an LLM judge,
unless it says otherwise. The adversarial category is excluded from that
denominator to stay comparable with published work, and is separately measured
and reported rather than quietly dropped.

---

## The short version

The framework started 5.9 points behind a plain RAG baseline. It now beats it by
6.8 on a third less context.

| date | run | ours | RAG | our context |
|---|---|---|---|---|
| 14 Aug | first full run | 54.35% | 60.26% | 440 tok |
| 19 Aug | `full_v2` | 59.55% | – | 1,304 tok |
| 21 Aug | `full_v3` | 63.31% | 64.22% | 656 tok |
| 5 Sept | `full_v3` + hydration + turn dates | 66.95% | 66.17% | 1,137 tok |
| **10 Sept** | **paired run, answer-type gate + grounding** | **71.4%** | **64.6%** | **964 tok** |

That is **+17.0 points** overall on our own arm, and the lead over RAG is
**+6.8, McNemar exact p = 5.5e-08, gross movement +238/−133**.

The last row is different in kind from the ones above it, and the difference
matters more than the number. **Both arms answered every question in the same
window against the same stores**, with both predictions written to the same row.
Earlier rows are two runs subtracted from each other, which imports cross-run
drift into every comparison and makes McNemar inapplicable. Everything from 10
September onward is paired.

Pairing also settled how much drift there actually is. Re-running our own arm
against its own saved output moved **0.1 points at n=841** — gross +21/−22, so
43 individual answers changed to produce a tenth of a point. That is far tighter
than the 1.5 to 2 points earlier sections of this file assume, and it is why the
warning that used to sit here ("do not claim superiority over RAG on 0.78
points") no longer applies to a paired result. It still applies to any two
numbers taken from different runs.

**The overall column is still the least interesting thing here.** The two systems
do not fail in the same places:

| category | n | ours | RAG | delta | McNemar exact |
|---|---|---|---|---|---|
| **temporal** | 321 | **71.7%** | 46.1% | **+25.5** | p = 1.1e-14 |
| **multi-hop** | 96 | **57.3%** | 44.8% | **+12.5** | p = 0.0075 |
| single-hop | 281 | 50.9% | 45.6% | +5.3 | p = 0.133 |
| open-domain | 841 | 79.8% | 80.3% | −0.5 | p = 0.804 |
| adversarial | 446 | 35.9% | 50.4% | **−14.6** | p = 4.1e-08 |

Temporal and multi-hop are the result. A 25-point lead on "when did it happen"
is exactly the category the design was argued for, and multi-hop clears
significance despite n=96. Single-hop and open-domain are ties — the +5.3 is not
a win at p=0.133 and the −0.5 is not a loss either, and neither should be quoted
as one.

Two things this table does that the earlier ones did not. It reports
**adversarial**, which published comparisons drop and which we lose badly;
dropping the category you lose is not a measurement decision. And it reports
McNemar per category, so a reader can see which rows are results and which are
noise without having to know the sample sizes by heart.

Two findings held across every experiment on the branch:

**Everything that gave the model more or better material helped, and everything
that tried to take material away hurt.** Five separate experiments, no
exceptions — until width, which gave the model more material on open-domain and
bought +1.6 at p=0.111 for 332 extra tokens. More material has stopped paying.
The remaining wins came from telling the model what *shape* of answer to give
and from ranking evidence better, not from handing it more.

**Nothing under roughly 800 questions in this benchmark has survived being
re-measured at full scale.** Four separate results were mispriced by a small
sample: a 6-question token estimate 292 tokens high, an 8-question clause read, a
108-question gate result that read +5.6 and settled at +1.6, and a 281-question
precision screen that could not see the miss it caused.

---

## Background: what this system is

Two tiers, both append-only.

- **Hot RAM** is a LanceDB vector store holding extracted *facts*. Facts carry a
  weight that decays over time. Below a threshold they are pruned.
- **Cold ROM** is a SQLite event log holding everything, forever. Nothing is
  ever deleted from it. It supports keyword search (FTS5) and dense vector
  search over the same embeddings Hot RAM already computed.

Pruning moves a fact from the fast store to the slow one. It does not destroy
it. Measured on `conv-26`: 627 facts in the event log, 294 still in Hot RAM,
333 pruned, and every pruned fact's own words still return a Cold ROM hit. That
is recall failing while recognition survives, which is the behaviour we want.

---

## The changes that paid

### 1. Graph search across both tiers, and the decay fix (+9.0 points)

14 Aug to 21 Aug, 54.35% → 63.31%. By far the largest single contribution, and
the only change that made the system **better and cheaper at the same time**:
average context fell from 1,304 to 656 tokens while accuracy rose.

Two things were wrong before. Graph traversal only walked Hot RAM, so any chain
of reasoning that passed through a pruned fact simply stopped. And link targets
were stored as whatever the extractor happened to call an entity, so a large
share of links pointed at nothing.

What landed:

- Graph expansion now runs over both tiers. `ColdStorage.get_events_for_entities`
  and `semantic_search` were added so the archive can answer by meaning, not
  only by shared words. A question about someone's *aunt* cannot keyword-match a
  fact stored as *mother's sister*; the archive held the answer and could not
  reach it.
- `src/nmafc/engine/linking.py` (new) resolves extracted link names against
  stored entities and drops the ones that match nothing. Config:
  `resolve_link_targets` (on by default), `link_match_threshold`.
- `rewire_pruned_links` reroutes links *around* a pruned record instead of
  severing them. Off by default, deliberately, so the ablation arm can still
  demonstrate that forgetting disconnects the graph.

Category effects were largest exactly where you would expect: multi-hop
38.5% → 52.1%, temporal 51.4% → 61.4%.

### 2. Hydration (+1.75 points, p = 0.021)

**This is the novel piece and the one worth reading closely.**

The problem: extraction is lossy, and nothing else in the store survives it. Hot
RAM and the event log both keep only the summarised fact, so once a turn is
processed its actual wording is gone forever. LoCoMo asks questions in the
dataset's own phrasing, like *"two weekends before 17 July 2023"*. That phrasing
lives in the utterance. The stored fact carries a date the extractor resolved
for itself, sometimes inconsistently.

The fix, borrowed from fuzzy-trace theory's split between **gist** and
**verbatim** memory: keep both, and use each for what it is good at.

- A new `turn_text` table stores what was actually said, verbatim, one row per
  turn.
- It is deliberately **outside** the FTS index and carries **no embedding**.
  Facts are what decays and what gets ranked. Turn text is *evidence*, reachable
  only through a fact that has already won a slot, by its turn number.
- At retrieval, once the ranking is settled, the source turns behind the top
  *k* facts are attached to the prompt inside a `<SOURCE>` block.

Why keep it out of the index: if raw turns were searchable they would compete
with facts for the retrieval budget. That is the one way storing them could cost
anything.

Measured effect, on the same stores, A against B:

```
A  facts only          :   990   64.4%     656 context tokens
B  + hydrate top 5     :  1017   66.1%   1,045 context tokens
change                 :   +27   +1.8%   (p = 0.021)
```

The retrieval-side picture is the same and larger, because reachability is
measured without generation getting in the way. Restoring the top five facts'
source turns moved strict answer reachability from 52.7% to 59.4% overall, and
from 50.5% to 59.8% on temporal. Per category (`_sweep_precision.py`), reach
rose +5.2 on multi-hop and +2.5 on open-domain, and on open-domain strict reach
rose +8.0. So the answer is reaching the prompt far more often than the accuracy
gain alone suggests, which means there is headroom left in generation rather
than in retrieval.

**`hydrate_top_k` now defaults to 5**, so this is on unless you switch it off.
It shipped at `0` for a while, which meant the library's default configuration
was not the configuration its own numbers came from. Set it to `0` for the
ablation. Stores written before `turn_text` existed have nothing to hydrate from
and degrade silently to facts alone, which is the old behaviour exactly, so
turning it on cannot break an old store.

### 3. Wider retrieval budget (+1.5 points)

64.8% at `rerank_top_k=20` → 66.3% at 40. Honest, but unglamorous: it is just
retrieving more. Worth knowing that it is roughly break-even against hydration
while costing more tokens, and the two are additive.

The A/B prints the mechanism rather than just the delta:

```
gained by the wider budget : 75
lost to distraction        : 52
```

So extra facts help about 1.4 times as often as they distract. That ratio gets
worse as the budget grows, which is why 30 was only +0.5.

### 4. Real dates instead of turn numbers (+1.3 points)

Turn numbers were reaching the model in place of dates. Facts were rendered as
`(Valid: turn 220 - present)`, and an ordinal is not a date, so answers came
back saying *"it was mentioned in turn 220 but no specific date is given"*,
which is a correct reading of what the model was shown.

A `turn_timestamps` table now records when each turn actually happened, one row
per turn rather than per fact, because the date belongs to the conversation and
not to each thing said in it.

This one has a placebo control. `_ab_noise.py` runs the same experiment with the
dates scrambled and came back at **−0.3%**, so the +1.3% is a real effect and
not the A/B harness flattering itself.

**The table was empty on every store until 5 September.** `turn_timestamps` had
zero rows across all ten `full_v3` stores, so this improvement existed in the
code and in an A/B and in no published number. `_backfill_turn_dates.py` filled
it: 2,957 turns, 100% dated, no LLM calls, because the date of a turn is a
property of the transcript rather than of anything a model produced. If you are
working from an older store, run that script before you measure anything
temporal.

### 5. The answer-type gate

The last change that paid, and the first one that did not work by giving the
model more material.

A question often names the category of its own answer. "In which **state** is the
shelter" — the store held `Stamford`, the answer given was "Stamford", and it was
marked wrong. "In which **country** was the pendant bought" — the store held
`Paris`, the answer given was "Paris", marked wrong. Both are evidence *for* the
answer rather than the answer.

No amount of retrieval fixes this, because the right fact was already in the
window. The cause is our own answering rules, which say to reuse the exact
wording of the retrieved facts because a synonym scores as a miss. That is
correct nearly everywhere and worth a large part of the score. On a question that
names the kind of answer it wants it is exactly backwards: "Connecticut" is not a
synonym for "Stamford", it is the level the question asked for, and "Stamford" is
the reason to believe it.

Measured over 1,535 paired questions *before* the gate existed:

| | n | ours | RAG |
|---|---|---|---|
| questions naming a type | 238 | 56.7% | 61.8% |
| everything else | 1,297 | 66.7% | 64.9% |

The whole of the deficit lived in the typed questions, and both arms failed them
together — which is what a shared prompt rule looks like from the outside.

`integration/answer_type.gate()` emits one short line, and only when the question
names a type. It is a regex over a string already in the prompt: no embedding, no
model call, no store access, and no cost on the ~85% of questions it stays quiet
on. It is deliberately a *level shift and never an invention* — it licenses
naming the state a retrieved town sits in, and says explicitly that it does not
license naming a state when nothing retrieved points at one, because the failure
it trades against is answering "Connecticut" from no evidence at all.

---

## The changes that did not pay

All are measured, all are reproducible from the scripts in this branch, and they
are as much a result as the positives are.

| experiment | effect | script |
|---|---|---|
| Rewriting the answer prompt | **−1.04**, and +0.5 at p=0.774 when retried under deep hydration: no effect | `_ab_prompt.py` |
| Decay as a ranking signal | **−5.3** reachability | `_sweep_weight_signal.py` |
| Contradiction pruning | **−5.9** | `_detect_supersessions.py`, `_apply_invalidation.py` |
| Context compaction | +0.5, inside noise | `_ab_budget.py --compact-b` |
| Noise control (placebo) | −0.3, as designed | `_ab_dates.py`, `_ab_noise.py` |
| Width on open-domain | +1.6 at p=0.111, for 332 tokens, and unshippable | `_run_open_domain_full.py` |
| The list-shape gate | +1.6 at p=0.458 over 322 questions | `_screen_list_gate.py`, `list_shape.py` |
| `commit_short` clause | adversarial +5.4, temporal −4.0, net **+0.0** | `_ab_budget.py --variant` |
| `commit_exact` clause | adversarial +0.2, temporal −3.4 | `_ab_budget.py --variant` |

Several are worth explaining, because each one closes off a line of work.

### Decay cannot reach the score through ranking

`DecayConfig.weight_signal` adds an RRF boost proportional to a record's decay
weight. It defaults to `0.0`, and it turns out **no run has ever raised it**. So
the decay weight has always been computed, persisted, and then ignored by the
reranker. The only channel from decay to a score was pruning.

Testing it needed prepared stores, because 97.8% of live weights sit at exactly
1.0 and a boost proportional to a constant reorders nothing.
`_inject_weights.py` copies the stores and recomputes weight from record age,
which is the widest spread available (68.7% at 1.0, 30.3% graded between 0.1
and 1.0).

Swept over 1,529 questions, retrieval only, no generation:

```
 weight_signal   reach strict   reach loose   churn vs 0
          0.0          52.1%         54.2%         0.0%
         0.05          48.7%         50.9%        43.1%
          0.1          47.5%         49.8%        45.3%
          0.2          47.2%         49.5%        46.3%
          0.4          46.8%         49.1%        47.0%
```

Monotonically worse, −5.3 at the top, negative in **every** category including
temporal, where a recency signal should have helped most. Churn of 43–47%
confirms the ranking genuinely moved, so this is a real negative and not a
wiring failure. Always read churn alongside reach: flat reach with zero churn
means the setting never took effect.

### Contradiction pruning fails on the detector, not the plumbing

The idea was retrieval-induced forgetting: when a later fact makes an earlier
one untrue, withdraw the earlier one.

`detect_override` has fired zero times across 8,276 facts, because it matches on
identical `entity_name`, and the extractor names entities per event.
`jon_job_loss_jan_2023` and `jon_new_shop_july_2023` are the same subject under
two names, so they never collide.

So we matched on *what a fact is about* instead: pairs close in vector space and
separated in time, adjudicated by an LLM. At cosine 0.7 that is 7,073 candidate
pairs across ten stores rather than the ~800,000 a full pairwise scan gives.
354 supersessions were found over 223 facts, and applied to copies of the
stores.

Scored on 101 mined belief-update questions, both arms re-run on the same code:

```
                        BASELINE      INVALIDATED
correct                 76  75.2%     70  69.3%     -5.9
gold reached prompt     80  79.2%     76  75.2%
stale reached prompt    63  62.4%     65  64.4%     went UP
broke 7, fixed 1
```

The reason is decisive. Of the stored facts carrying a **stale** answer, 19.6%
were invalidated. Of the facts carrying the **correct** answer, **17.9%** were
invalidated. The detector strips right and wrong facts at essentially the same
rate. It is a coin flip. Vector similarity plus an LLM verdict cannot tell
*"overtaken by events"* from *"restated differently"* on this extractor's
output.

**If you build an invalidation mechanism, measure the gold-invalidation rate
next to the stale-invalidation rate.** A mechanism that removes both at the same
rate is noise, no matter how good its headline accuracy looks.

### Rewriting the answer prompt

Adding an explanation of the `<SOURCE>` block to the system prompt lost 1.04
points and was negative in every category (multi-hop worst, −5.2). Answer
instructions are not where the headroom is.

Tried a second time, because the first measurement was taken when `<SOURCE>`
held five source turns beside all twenty facts and was a minor part of the
prompt. Under the configuration that block is now the larger half of, on 400
open-domain questions with both prompts generated in one session:

```
n=400   A 77.0%   B 77.5%   +0.5 points
fixed 7  broke 5  p = 0.774
```

Twelve discordant pairs in four hundred questions. The two prompts agree almost
everywhere, which says the model was already using `<SOURCE>` without being told
what it was, so naming it changes nothing in either direction. The −1.04 does
not survive either; read the effect as zero. Closed at both configurations.

### Width on open-domain: real, and still not shippable

Open-domain is the one scored category we do not lead, so it got the most
attention. Three arms, one window, 830 paired questions, splitting the two knobs
that had previously only been moved together:

| arm | accuracy | tokens | vs shipped | vs RAG |
|---|---|---|---|---|
| wide (hydrate 10, grounding 0.003) | 81.3% | 1,300 t | +1.6, +35/−22, p = 0.111 | +1.2, p = 0.426 |
| grounding only (0.003) | 80.6% | 1,047 t | +0.8, +31/−24, p = 0.419 | +0.5, p = 0.797 |
| shipped | 79.8% | 968 t | baseline | −0.4, p = 0.867 |
| RAG | 80.1% | 1,464 t | | |

Splitting the knobs was worth doing on its own: **facts are about twice as
token-efficient as hydrated turns.** Grounding alone gets half the gain for a
sixth of the token cost. Marginal render cost is roughly 30 tokens per printed
fact against 55 per hydrated turn, and the accuracy per token follows it.

Neither arm ships, and significance is not the reason. **Applying width to
open-domain only requires knowing the category, and the category is not knowable
at inference time.** It is a column in the benchmark file, not a property of the
question. A policy keyed on it is a benchmark artefact, and building one would
have been fitting to the test set through the back door.

### The list-shape gate: built, wired, measured, dead

The obvious repair for the previous problem. If the category cannot be known but
the *shape of the question* can, gate on that instead.
`integration/list_shape.wants_list()` decides from the question string alone
whether it asks to enumerate — "What activities does Melanie partake in" wants
four things — using a plural head noun in the wh-phrase plus a small set of
explicit list phrasings. Regex only, same cost profile as the answer-type gate.

The signal it finds is real. It fires on 327 of 1,985 questions, and we score
**50.2% where it fires against 66.0% where it does not**. A 16-point gap keyed on
something knowable at inference is exactly what the width work needed.

Handing those questions more context does not close it. Every firing question
answered both ways, 322 of them:

| category | n | delta | discordant | p |
|---|---|---|---|---|
| open-domain | 108 | +5.6 | +8/−2 | 0.109 |
| single-hop | 141 | **−1.4** | +6/−8 | 0.791 |
| adversarial | 58 | +1.7 | +3/−2 | 1.000 |
| multi-hop | 13 | +0.0 | | |
| temporal | 2 | +0.0 | | |
| **all** | **322** | **+1.6** | **+17/−12** | **0.458** |

Wide 50.9% against the shipped 49.4%, for 324 extra tokens a question. Single-hop
— the category the gate was built from — goes backwards. Applied as a policy over
all 1,985 it moves 63.4% to 63.66% and 963 tokens to 1,017.

The module is kept, tested and documented, wired into nothing. The detection is
sound and the 16-point gap is worth attacking; more context is not how.

**This is also where the sample-size rule came from.** An earlier read of this
same gate, on the 108 open-domain questions above, reported **+5.6** and looked
like the best result of the cycle. It was nine coin flips landing one way. It
regressed the moment there were more of them.

### Two prompt clauses for adversarial, both dead

Adversarial is the clear loss: 35.9% against RAG's 50.4%, −14.6, p = 4.1e-08. The
obvious theory is that we refuse too often. Measured, that theory does not hold —
RAG refuses at a rate between comparable to ours and higher than ours, depending
which refusal detector you use, and still scores 14.5 points above us. Roughly a
third of the gap is recoverable refusal. The rest is wrong answers on questions
we did commit to, which no instruction reaches.

Two clauses were built and measured over all 1,984:

| clause | adversarial | temporal | all five |
|---|---|---|---|
| `commit_short` | **+5.4**, p = 0.0022 | **−4.0**, p = 0.0024 | +0.0, +72/−72 |
| `commit_exact` | +0.2 | **−3.4**, p = 0.0074 | strictly worse than both |

`commit_short` buys adversarial and pays for it exactly, one question for one
question, out of temporal and multi-hop. `commit_exact` was an attempt to keep
the gain and drop the damage by deleting the wording blamed for it. It deleted
the gain and kept the damage: its refusal rate came back identical to the
shipped configuration, meaning its second sentence had cancelled its first.

The diagnosis that produced `commit_exact` was wrong, and worth recording as
wrong. It claimed one word was doing the temporal damage. Only 6 of the 13
temporal breakages overlapped with `commit_short`'s 15, so the two clauses were
not breaking the same questions and the single-word theory never had support.

The conclusion is structural rather than about wording. **The behaviour that
answers an adversarial question and the behaviour that dates a temporal one are
welded together in this model, and the join is not in the prompt.**

---

## Cost, not score

Two things that buy nothing on the leaderboard and matter anyway.

### The prompt is 12.4% smaller for nothing

Broken down over 200 prompts on the finished stores, the context was not mostly
facts:

```
tokens per prompt      1,124
  fact text              498   44.3%
  "(Valid: ... )"        227   20.2%
  SOURCE block + tags    389   34.6%
```

A fifth of it was the validity span, and **4,000 of 4,000 spans in that sample
ended `- present`**, because nothing in these stores has ever been invalidated.
Every line was paying for a word that was true of every line. The clock time on
a session timestamp is the same kind of cost: 11.2 characters of a 25.8-
character date, on a benchmark where nothing asks the hour.

`compact_validity` drops the label, the open end and the clock, and takes the
context to 982 tokens. The date survives in full, and a fact that really was
superseded still renders its end date.

Tested rather than assumed, because "shorter" and "better" are not the same
claim. Paired A/B, 1,537 questions, both arms on the same stores with the same
retrieval and the same judge:

```
                    correct              context
A  verbose span    1018   66.23%        1,137 tok
B  compact span    1016   66.10%          997 tok
   delta             -2   -0.13 pts       -141 tok  (-12.4%)

fixed 37   broke 39   McNemar p = 0.91
```

Read it as "any real effect is inside ±1.2 points", not as "identical". On by
default because 12.4% is a certain saving against an unmeasurable cost. The
SOURCE block keeps its headers exactly as stored: verbatim that has been edited
is not verbatim.

The 997 above is this A/B's own measurement. In the 10 September paired run the
shipped configuration renders **963 tokens against RAG's 1,461**, which is where
the "a third less context" headline comes from. Marginal render cost, useful for
budgeting any further change: about **30 tokens per printed fact and 55 per
hydrated turn**.

### Latency, and a number that was never what it looked like

**The framework's own work is about 56 ms per question.** Split over 60
questions:

```
region          mean      p50      p90    share
embed            263      225      402   14.1%    <- network
retrieve         319      312      469   17.0%
format             0        0        1    0.0%
generate        1289      765     1583   68.9%    <- network
```

The embedding call happens inside `retrieve`, so retrieve is 319 ms of which
263 ms is waiting on the network. Everything this codebase controls — vector
search, cold search, graph traversal, reranking, the hydration lookup,
rendering — is the remainder. **83% of a question is waiting for a provider.**

The one real saving available was `defer_reinforcement_writes`. Reinforcement
rewrites every surviving record through a delete and an add, which on an
append-only store appends a table version per query. Measured over 240
retrievals against a *copy* of a finished store, same machine, same minute, one
flag changed:

```
                mean      1-40   41-80   81-160  161-240
immediate       501 ms     420     515      474      561   <- rises
deferred        300 ms     307     287      308      294   <- flat
```

About 260 ms of both figures is the embedding call, so this removes roughly six
times the rest of the local work put together, and it is the only part that
grows as a run goes on. Now the default, with `reinforcement_buffer_limit` so a
caller that only ever retrieves cannot accumulate writes that never land.

**Do not read the run-to-run latency numbers as an improvement.** `full_v2`
reports 21,306 ms per question and `full_v3` 2,159 ms, and almost all of that
gap is a measurement change, not a code change: the runs before 4 September have
no `throttle_ms` column at all, so their latency includes however long the
shared provider quota made them queue. `full_v2` also ran two arms against one
quota, and the 14 August run ran four including `raw_llm` at 20,000 tokens a
question. The giveaway is the shape — a program getting slower rises across a
run, and `full_v2` falls (29,660 → 12,398 ms) as the traffic clears, while
`full_v3` is flat at ~2,100 ms throughout.

So the honest claim is 200 ms of real saving, plus the end of a tenfold
mismeasurement. Not a tenfold speedup.

---

## Correctness fixes (no score effect, but they were real bugs)

### The clock guard

`HotStorage.apply_reinforcements` and `update_reinforcement` now do
`last_reinforced_turn = max(turn, existing)`.

Without it, a caller that reopens a store without restoring its turn counter
starts at 0 and reinforces at turn 1, stamping `last_reinforced_turn=1` onto
records created at turn 200. Reinforced, on the record, 199 turns before they
existed. Every later decay then measures elapsed time from that floor and reads
the whole conversation as having passed.

Found on the `full_v3` stores, where **3,477 of 3,807 records** carried a
reinforcement turn earlier than their creation turn.

### Reading the memory was writing to the memory

Related, and worth knowing before you trust anything read off disk. One
`retrieve` call with the default config changed 14 records on disk, because
`defer_reinforcement_writes` defaults to `False`. Every benchmark question was
reinforcing ~14 records and persisting it.

Two consequences: `consolidation_index` counts benchmark questions rather than
genuine recalls (values of 70, 277, 352 are eval artefacts), and
`last_reinforced_turn` is stamped backwards as described above.

The clock guard stops the corruption spreading. It does not repair existing
stores. **`defer_reinforcement_writes` now defaults to `True`**, so a script
that touches a store no longer has to remember to ask, and the buffer flushes
itself at `reinforcement_buffer_limit` records as well as on decay, `maintain()`
and `close()`. Pass `--no-defer-reinforcement` for the ablation.

`created_at_turn` and `memory_type` are the only clock fields the harness leaves
intact.

### Cold ROM now honours `invalid_at`

The column existed but was never written and never filtered on, while the router
queries the archive on every retrieval. So invalidating a fact removed it from
Hot RAM and the archive handed it straight back on the next query. Measured over
101 belief-update questions, Hot-only invalidation left the stale value in the
prompt 64 times out of 101, which is **one more** than doing nothing at all.

What landed:

- `ColdStorage.invalidate_facts()`, matching on `(entity_name, fact_content)`
  because the archive autoincrements its own primary key and does not carry the
  Hot record's UUID.
- `AND invalid_at IS NULL` on `keyword_search`, `semantic_search`'s row fetch,
  and `_load_vectors`.
- A default no-op on `ColdStorageBase` so the Postgres backend still runs.
- Propagation from the wrapper's override path.

**Pruning deliberately does not propagate to Cold ROM.** A superseded fact is
*wrong*, so it should go. A pruned fact is merely *weak* and is still true, so
it keeps its place in the archive. That distinction is the whole point of having
two tiers.

The row itself is never deleted in either case. It stays in the append-only log
with `invalid_at` set, so the audit trail and any as-of-turn query still see it.

---

## New configuration

All in `src/nmafc/schemas/memory.py`. Defaults preserve existing behaviour.

| setting | default | what it does |
|---|---|---|
| `hydrate_top_k` | `5` | Attach source turns behind this many top-ranked facts. **Set to 0 for the ablation.** |
| `resolve_link_targets` | `True` | Match extracted link names to stored entities; drop unmatchable links |
| `link_match_threshold` | `0.5` | How close a name has to be to count as a match |
| `rewire_pruned_links` | `False` | Reroute links around pruned records instead of severing them |
| `rewire_max_links` | `8` | Cap on rewiring fan-out |
| `rewire_max_depth` | `2` | Cap on rewiring search depth |
| `defer_reinforcement_writes` | `True` | Buffer LTP writebacks instead of writing on every query. 501 ms per retrieval immediate and rising, 300 ms deferred and flat |
| `reinforcement_buffer_limit` | `256` | Flush the buffer once this many records are held, so a caller that only retrieves cannot lose writes |
| `compact_validity` | `True` | Render validity as `(27 May 2023)` rather than `(Valid: 7:18 pm on 27 May, 2023 - present)`. 12.4% off the prompt |
| `weight_signal` | `0.0` | RRF boost proportional to decay weight. Leave at 0, see above. |
| `always_search_cold` | `True` | Search Cold ROM in parallel with Hot on every query, ignoring `theta`. This is what replaced the gated fallback |
| `rerank_top_k` | `20` | How many records survive RRF fusion. Screening found this is the **only** retrieval setting that moves reachability |
| `hydrate_full_turns` | `0` | How many top-ranked turns come back whole before `hydrate_lines` trims the rest |
| `hydrate_lines` | `None` | Keep only the N speaker lines of a hydrated turn that best match the question. `None` = the whole turn |
| `hydrate_pool` | `0` | Choose hydrated turns by question match across this many candidates instead of by fact rank. Same number of turns, different turns |
| `hydrate_scan` | `0` | How wide the pool search reads before ranking it |
| `source_grounding` | `0.0` | Score floor below which a fact is not worth rendering. Raising it to `0.003` bought +0.8 on open-domain for 79 tokens, the best accuracy-per-token of anything measured — and still not significant |

`source_grounding` is the one to reach for first if you are trying to buy
accuracy. It is not enabled by default because +0.8 at p=0.419 is not a result,
and turning on a setting whose evidence is a non-significant gain is how a
benchmark artefact becomes a default.

---

## The scripts

Everything under `scripts/benchmarks/_*.py` is experiment tooling. They are all
read-only against the real stores unless the docstring says otherwise, and the
ones that mutate work on copies.

**A/B experiments** (two configs, same stores, same questions, paired)

- `_ab_budget.py` – retrieval budget, compaction, hydration, prompt clauses.
  Also the home of `close_readonly()` and the `SCORED` category tuple
- `_ab_dates.py` – real dates against turn numbers
- `_ab_prompt.py` – answer-prompt variants
- `_ab_link_repair.py` – link resolution
- `_ab_vs_rag.py` – ours against the RAG arm, both in one window
- `_run_open_domain_full.py` – full-scale paired runner. Carries `--gate-only`
  and `--list-gate` so a question subset can be isolated by a property of the
  *question string* rather than by its category label

**Sweeps and screens** (retrieval only, no generation, no cost)

- `_sweep_weight_signal.py` – decay as a ranking signal
- `_sweep_context_budget.py`, `_sweep_precision.py`, `_sweep_hydration.py`
- `_screen_grounding.py`, `_screen_hydrate_pool.py`, `_screen_ranking.py`,
  `_screen_list_gate.py`, `_screen_oracle_reach.py`

**Probes and audits**

- `_retrieval_funnel.py` – where answers are lost between store and prompt
- `_error_taxonomy.py`, `_probe_answered_wrong.py`, `_probe_pruned.py`,
  `_probe_verbatim.py`, `_tier_audit.py`, `_context_anatomy.py`,
  `_ingest_audit.py`, `_hot_latency_probe.py`

**Belief-update pipeline**

- `_mine_updates.py` – mine questions where something changed mid-conversation
- `_detect_supersessions.py` – find superseded facts
- `_apply_invalidation.py` – write the verdicts onto **copies** of the stores
- `_test_updates.py` – score current answer against stale answer

**Maintenance**

- `_backfill_turn_dates.py`, `_backfill_turn_text.py` – add the new tables to
  existing stores without re-ingesting
- `_repair_graph_links.py`, `_seed_ingest_state.py`, `_trim_hedges.py`
- `_supervise.sh` – watchdog that restarts a long run after a crash

### A screening trick worth knowing

Full scoring is ~87 minutes per config because of generation. Most retrieval
settings can be screened in ~4 minutes by measuring **reachability** instead:
does the gold answer appear anywhere in the retrieved facts? No generation, no
judge. `_sweep_weight_signal.py` works this way. Only promote a config to full
scoring once reachability says it is worth it.

### Reproducing a result

The headline table costs nothing to reproduce, because the per-question data is
committed:

```bash
# every figure in the results section, from the committed JSON. No API calls.
python scripts/benchmarks/results/paired_2026_09_10/summarise.py
```

To re-measure rather than recompute:

```bash
# hydration, the headline
python scripts/benchmarks/_ab_budget.py --hydrate-a 0 --hydrate-b 5 \
    --out /tmp/hydrate.json

# decay as a ranking signal, the headline negative. Retrieval only, ~4 minutes.
python scripts/benchmarks/_sweep_weight_signal.py --out /tmp/weights.json

# the list gate, on the questions it fires on and only those
python scripts/benchmarks/_run_open_domain_full.py --list-gate fires \
    --out /tmp/lg.json

# belief updates
python scripts/benchmarks/_test_updates.py \
    --questions /path/to/updates.json --out /tmp/updates.json
```

All of them read `NMAFC_BENCH_PROVIDER` and the rest of the configuration from
`.env`. Nothing is hardcoded and `.env` is gitignored.

Two things to know before running any of them against the real stores. **Reading
a store writes to it** unless the script calls `close_readonly()` — retrieval
reinforces what it returns, so one question mutates about 14 records. And
**nothing mutates a store in place**: every experiment that changes state copies
first, because the ten indexed stores cost ~5.3 hours of paid ingestion to
rebuild and are treated as immutable inputs.

---

## Caveats, honestly stated

- **Cross-run drift is smaller than this file used to assume, and the reason to
  pair is not drift anyway.** The 1.5-to-2-point figure quoted in earlier
  revisions came from small samples. Re-running our own arm against its saved
  output at n=841 moved **0.1 points**, gross +21/−22. The real argument for
  pairing is statistical rather than defensive: on 1,985 questions the two arms
  agree 1,473 times, and those agreements carry no information about which is
  better. Comparing totals discards the 512 disagreements that are the entire
  evidence. Pair, and report gross movement (+N/−M) rather than the net, because
  a net of +0.0 can hide 72 questions fixed and 72 broken — which is exactly what
  `commit_short` did.

- **68.4% of the facts that survive in Hot RAM are classified `CoreAnchor`**,
  whose base decay rate is 0, so decay barely runs at all in practice. At
  extraction the share is 35%; the gap is decay doing its job on the other
  tiers. Sampling says roughly half the anchors are dated one-off episodes
  ("Melanie took her kids to the museum on 5 July 2023") that should not be
  anchors at all. This caps every decay-tuning result on the branch. Fixing it
  will most likely *lower* the LoCoMo score, since every measured result here
  says removing material from the prompt costs points.

- **The `full_v3` stores carry corrupted `consolidation_index` and
  `last_reinforced_turn`** from before the clock guard. Do not read decay
  history off them. `created_at_turn` and `memory_type` are intact.

- **The 5 September run used `--defer-reinforcement`; the 21 August one did
  not.** Reinforcement writebacks are buffered rather than committed per query.
  At `weight_signal = 0`, which is the default and what both runs used, weight
  does not enter ranking, so the two are equivalent for retrieval. It changes
  what the stores look like afterwards, not what the model saw.

- **Latency: settled, and the mean was never the right statistic.** This used to
  read "we have never measured our latency against RAG's under the same
  conditions", because every comparison available was two runs on two nights
  against a shared quota — two runs of the *identical* RAG arm once came out 91%
  apart. The paired run settles it, both arms interleaved in one window:

  | | p50 | p75 | p90 | p99 | mean |
  |---|---|---|---|---|---|
  | ours | 1,741 ms | 2,407 ms | 3,559 ms | 26,100 ms | 2,721 ms |
  | RAG | 1,769 ms | 2,398 ms | 3,346 ms | 12,019 ms | 2,295 ms |

  **Indistinguishable through p90** — we are marginally faster at the median and
  marginally slower at p90, and neither gap is worth a sentence. The 19% gap in
  the *mean* is entirely tail, and both arms top out at ~93.8 s, which is a
  provider retry ladder rather than anything either architecture computes. Quote
  the median. An earlier revision of the README claimed we were "roughly 6×
  slower" off exactly this mean, on two unpaired runs; that claim was wrong and
  has been removed.

- The belief-update questions were mined from LoCoMo, which was not built to
  test belief updates. LongMemEval is the right dataset for that and has not
  been run.

---

## Where to look next

Ranked by expected value, given everything above.

~~**Turn hydration on in the arm configs.**~~ Done, 5 September. It was the whole
of the 63.31% → 66.95% jump, together with the dates backfill.

~~**One run with both arms in it.**~~ Done, 10 September. It settled accuracy
(+6.8 on the scored set, p = 5.5e-08), latency (a tie through p90), and the size
of cross-run drift (0.1 points) in one window. Everything above is now paired.

~~**Close the open-domain gap.**~~ Closed, though not the way this list expected.
Open-domain is 79.8% against RAG's 80.3%, a tie at p=0.804, and it got there from
better ranking rather than more hydration. Raising `hydrate_top_k` — the "cheap
thing to try first" — was tried at full scale and bought +1.6 at p=0.111 for 332
extra tokens, which is not affordable under the 1,000-token budget.

Still open, ranked by expected value:

1. **The list-shaped questions.** We score 50.2% on the 327 questions
   `wants_list` fires on against 66.0% on the rest, and that gap is keyed on
   something knowable at inference time. Width has been tried and does not close
   it. What has *not* been tried is changing the answering behaviour rather than
   the retrieval: a partial list scores like a wrong answer, so the question is
   whether the model can be got to enumerate what it holds instead of naming the
   first item.
2. **Fix `CoreAnchor` over-classification in the extractor.** Nothing about
   forgetting can be evaluated properly until decay actually runs. Needs full
   re-ingestion (~5.3h) because `memory_type` is set at write time, so batch it
   with every other write-time change. Expect the LoCoMo number to go *down*.
   Note the cheap half of this was already checked: **re-labelling tiers on
   existing stores takes about 4 minutes**, and doing so disproved the theory
   that the 68% anchor share was a bug. Only changes that must happen at write
   time justify the re-ingest.
3. **LongMemEval.** LoCoMo rewards retrieval breadth, which is why breadth is
   what wins here. A dataset built around belief updates would test the parts of
   this design that LoCoMo cannot.
4. **Adversarial, if at all, from somewhere other than the prompt.** Two clauses
   were measured and both traded temporal for adversarial one question for one
   question. Anything further needs a different mechanism, not different wording.
5. **Not** more work on the supersession detector. The failure there is not a
   threshold or a prompt, it is that the signal does not exist in the
   embeddings.
6. **Not** re-ingestion for its own sake. The ten stores were last written on 10
   September after roughly 6,000 questions were answered against them, with no
   stale WAL files. Re-ingesting replaces the artefact every number in this file
   was measured on, and is only warranted if the extractor, the chunker, the
   embedding model or the fact schema changes.

---

## Tests

```bash
python -m pytest tests/unit -q
# 547 passed, 3 skipped
```

New: `test_link_resolution.py`, `test_link_rewiring.py`, `test_turn_dates.py`,
`test_deferred_reinforcement.py` -- the last of these because `defer_reinforcement_writes`
had no test at all, which is why flipping its default broke nothing.

Then `test_answer_type.py`, `test_list_shape.py`, `test_grounding.py`,
`test_quantities.py` for the question-inspection modules. These are cheap to test
properly because they are pure functions of a string: no store, no network, no
model. `test_list_shape.py` is worth reading even though the module it covers is
not wired in. It asserts precision harder than recall, and it carries a
`TestTheKnownMisses` class of list-shaped questions the gate deliberately does
*not* catch, recorded rather than fixed — a rule loose enough to catch "Where has
Maria made friends?" fires on the single-answer questions too, and a miss costs
only the status quo while a false fire costs real tokens.
