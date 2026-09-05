# What changed on `muna`

A summary of the work on this branch, why each piece exists, and what it was
worth. Every number here was measured on the standard LoCoMo protocol (1,540
questions, adversarial category excluded) with an LLM judge, unless it says
otherwise.

---

## The short version

The framework started 5.9 points behind a plain RAG baseline. It now matches
RAG while using 28% less context.

| date | run | ours | RAG | our context |
|---|---|---|---|---|
| 14 Aug | first full run | 54.35% | 60.26% | 440 tok |
| 19 Aug | `full_v2` | 59.55% | – | 1,304 tok |
| 21 Aug | `full_v3` | 63.31% | 64.22% | 656 tok |
| Sept | + hydration (A/B on `full_v3` stores) | **66.56%** | 66.20% | 1,045 tok |

That is **+12.2 points** overall. The interesting part is not the total, it is
which changes produced it, because the pattern is consistent and it was not the
pattern we expected.

**Everything that gave the model more or better material helped. Everything that
tried to take material away hurt.** That holds across five separate experiments
and it is the main finding of the branch.

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

**Note for anyone running this:** `hydrate_top_k` defaults to **0**, so
hydration is off unless you turn it on. It is not enabled in any of the arm
configs under `scripts/benchmarks/arms/`. The +1.75 above was measured through
the A/B harness (`_ab_budget.py --hydrate-b 5`). If you want it in a normal run
you have to set it.

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

---

## The changes that did not pay

All five are measured, all are reproducible from the scripts in this branch, and
they are as much a result as the positives are.

| experiment | effect | script |
|---|---|---|
| Rewriting the answer prompt | **−1.04** | `_ab_prompt.py` |
| Decay as a ranking signal | **−5.3** reachability | `_sweep_weight_signal.py` |
| Contradiction pruning | **−5.9** | `_detect_supersessions.py`, `_apply_invalidation.py` |
| Context compaction | +0.5, inside noise | `_ab_budget.py --compact-b` |
| Noise control (placebo) | −0.3, as designed | `_ab_dates.py`, `_ab_noise.py` |

Three of these are worth explaining, because each one closes off a line of work.

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
stores. **Any script that touches these stores should pass
`defer_reinforcement_writes=True`.** The `_ab_*` and `_sweep_*` helpers do. The
`neuromorphic_tuned` arm does not, which is a known outstanding item.

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
| `hydrate_top_k` | `0` | Attach source turns behind this many top-ranked facts. **Set to 5 for the +1.75.** |
| `resolve_link_targets` | `True` | Match extracted link names to stored entities; drop unmatchable links |
| `link_match_threshold` | `0.5` | How close a name has to be to count as a match |
| `rewire_pruned_links` | `False` | Reroute links around pruned records instead of severing them |
| `rewire_max_links` | `8` | Cap on rewiring fan-out |
| `rewire_max_depth` | `2` | Cap on rewiring search depth |
| `defer_reinforcement_writes` | `False` | Buffer LTP writebacks until `flush_reinforcements()`. **Set `True` for read-only analysis.** |
| `compact_validity` | `False` | Render validity as `(date)` rather than `(Valid: date - present)` |
| `weight_signal` | `0.0` | RRF boost proportional to decay weight. Leave at 0, see above. |

---

## The scripts

Everything under `scripts/benchmarks/_*.py` is experiment tooling. They are all
read-only against the real stores unless the docstring says otherwise, and the
ones that mutate work on copies.

**A/B experiments** (two configs, same stores, same questions, paired)

- `_ab_budget.py` – retrieval budget, compaction, hydration
- `_ab_dates.py` – real dates against turn numbers
- `_ab_prompt.py` – answer-prompt variants
- `_ab_link_repair.py` – link resolution

**Sweeps**

- `_sweep_weight_signal.py` – decay as a ranking signal
- `_sweep_context_budget.py`, `_sweep_precision.py`

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

```bash
# hydration, the headline
python scripts/benchmarks/_ab_budget.py --hydrate-a 0 --hydrate-b 5 \
    --out /tmp/hydrate.json

# decay as a ranking signal, the headline negative
python scripts/benchmarks/_sweep_weight_signal.py --out /tmp/weights.json

# belief updates
python scripts/benchmarks/_test_updates.py \
    --questions /path/to/updates.json --out /tmp/updates.json
```

All of them read `NMAFC_BENCH_PROVIDER` and the rest of the configuration from
`.env`. Nothing is hardcoded and `.env` is gitignored.

---

## Caveats, honestly stated

- **Cross-run drift is about 1.5 to 2 points.** An unchanged arm scored 66.12%
  and then 66.56% on consecutive runs. Any cross-run claim smaller than 2 points
  is not safe. This is why every result above that matters was measured as a
  paired A/B on the same stores, not by comparing two full runs.

- **82% of facts are classified `CoreAnchor`**, whose base decay rate is 0. So
  decay barely runs at all in practice, which caps every decay-tuning result on
  this branch. Fixing the extractor's classification is the prerequisite for any
  serious work on forgetting.

- **The `full_v3` stores carry corrupted `consolidation_index` and
  `last_reinforced_turn`** from before the clock guard. Do not read decay
  history off them. `created_at_turn` and `memory_type` are intact.

- **Hydration is implemented but off by default** and not enabled in the arm
  configs. See the note in its section.

- The belief-update questions were mined from LoCoMo, which was not built to
  test belief updates. LongMemEval is the right dataset for that and has not
  been run.

---

## Where to look next

Ranked by expected value, given everything above.

1. **Turn hydration on in the arm configs.** It is measured, it is significant,
   and it is currently costing 1.75 points for no reason.
2. **Fix `CoreAnchor` over-classification in the extractor.** Nothing about
   forgetting can be evaluated properly until decay actually runs.
3. **LongMemEval.** LoCoMo rewards retrieval breadth, which is why breadth is
   what wins here. A dataset built around belief updates would test the parts of
   this design that LoCoMo cannot.
4. **Not** more work on the supersession detector. The failure there is not a
   threshold or a prompt, it is that the signal does not exist in the
   embeddings.

---

## Tests

```bash
python -m pytest tests/unit -q
# 360 passed, 3 skipped
```

New: `test_link_resolution.py`, `test_link_rewiring.py`, `test_turn_dates.py`.
