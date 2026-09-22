# RECENT_OPS

Running handoff log. If a session ends, another one can pick up from here
without re-reading anything. Newest work at the top. `WHAT-CHANGED.md` holds
the long-form history; this file holds the live state.

## Ground rules that survive every session

- Nothing is hardcoded. Endpoints, deployment names, keys and concurrency all
  live in `.env` and are read through `os.environ`. No exceptions.
- Nothing is committed and nothing is pushed by an assistant. Commands are
  handed over and run by hand.
- Stores are never written to in place. Anything that mutates copies first.
- Comparisons are paired inside one session. Cross-run numbers drift 1.5-2
  points on accuracy and up to 91% on latency, so a table built from two
  different days is measuring the provider's queue.

## Working copy moved to C:/nmafc_muna, docs rewritten, ready to push

**The repo now lives at `C:/nmafc_muna`.** Deliberately outside OneDrive,
because OneDrive sync corrupts `.git`. The original at `.../new nmac/nmafc` is
untouched and still holds the 31 Bigbrown files.

Copied with `tar` rather than `git clone`, because a clone takes only committed
state and every line of the September work is uncommitted. Verified identical:
branch `muna`, HEAD `69f651f`, `.env` present, `git fsck` clean.

Cleaned, deletions only:

- 31 `*-Bigbrown.*` OneDrive sync-conflict copies. Already gitignored at
  `.gitignore:53`, so this is local hygiene -- nobody opens the stale
  `query_router-Bigbrown.py` by accident. src .py count 69 -> 51.
- `next` and `web-ui@0.1.0`: empty 0-byte files from shell-redirect typos on
  19 Aug. **Both tracked**, so they are on GitHub now and this removes them.
- `.pytest_cache`.

Suite after cleanup: **547 passed, 3 skipped** in `tests/unit`. The drop from
595 is three duplicate Bigbrown test files leaving; the live counterparts
(`test_cold_fallback.py`, `test_hot_storage.py`, `test_pruning.py`) still pass
their 45 tests. No coverage lost.

Docs rewritten against the 10 September paired run:

- `README.md` -- new "Where it stands" block at the top; the whole "Current
  results" section replaced (it still reported `full_v2` **losing** to RAG,
  0.5955 against 0.6026); new sections on latency, negative results and
  screening; the two contradictory Cold ROM sections resolved into the
  `always_search_cold` reality; the Spreading Activation section collapsed from
  two superseded readings into one; Project Structure refreshed with
  `answer_type`, `grounding`, `quantities`, `list_shape`, `linking`,
  `schemas/code.py` and `code/`.
- `WHAT-CHANGED.md` -- short version rewritten, the answer-type gate added as
  the fifth change that paid, four new negative results, the latency caveat
  resolved, the config table extended, "Where to look next" re-ranked.
- `scripts/benchmarks/results/paired_2026_09_10/` **is new and committed**: the
  8 per-question JSON files plus `summarise.py`, which regenerates every table
  in the README from them with no API calls. Needed a two-step un-ignore in
  `.gitignore` (directory, then files) because `results/**` is excluded.

**Nothing is staged, committed or pushed.** The single command was handed over.

### Latency, settled

Both arms interleaved in one window, from `locomo_all.json`:

```
        p50      p75      p90       p99      mean
  ours  1,741    2,407    3,559    26,100    2,721
  RAG   1,769    2,398    3,346    12,019    2,295
```

Indistinguishable through p90. The 19% gap in the mean is entirely tail, and
both arms max at ~93.8 s, which is the provider's retry ladder. **Quote the
median.** The README's old "roughly 6x slower" came from the mean of two
unpaired runs and has been removed.

### Cross-run drift is 0.1, not 1.5-2

Re-running our own arm against its saved output at n=841 moved 0.1 points,
gross +21/-22. The ground rule above still says pair everything, but the reason
is now statistical rather than defensive: on 1,985 questions the arms agree
1,473 times and those agreements carry no information, so comparing totals
throws away the 512 disagreements that are the whole evidence.

## The list gate: built, wired, measured, dead

Four arms, every question the gate fires on measured both ways, one window.
`od_wide.json` + `od_ship.json` cover open-domain; `lg_wide.json` +
`lg_ship.json` cover the other four categories. 322 firing questions in total.

```
  every firing question, paired      n=322
    wide      50.9%   1,287 t
    shipped   49.4%     963 t
    effect     +1.6   +17/-12   discordant 29   p=0.458
```

Broken out, and this is the whole story:

```
  open-domain  n=108   +5.6   +8/-2    p=0.109
  single-hop   n=141   -1.4   +6/-8    p=0.791
  adversarial  n= 58   +1.7   +3/-2    p=1.000
  multi-hop    n= 13   +0.0   +0/-0    p=1.000
  temporal     n=  2   +0.0   +0/-0    p=1.000
```

**It goes backwards on the category it was built for.** The gate was designed off
single-hop, where it fires on half the questions and where the long-gold failures
were read one by one. Single-hop is the one category where width makes things
worse.

**The open-domain +5.6 did not survive.** On 108 questions it was +8/-2 and
looked like the best thing measured all day. Across 322 it is +1.6 at p=0.458.
Eight fixes against two breaks is nine coin flips; it regressed to the mean the
moment there were more of them. Fourth time in this project a small sample has
mispriced a retrieval change, after the 6-question 1,592t smoke, the 8-question
1,027t clause reading, and the 281-question precision screen that could not see a
miss.

**Policy arithmetic, for the record.** Width where the gate fires, shipped
elsewhere, over all 1,985: accuracy 63.4% -> 63.7%, context 963t -> 1,017t.
A quarter of a point for 54 tokens, resting on a p=0.458 effect, and it puts the
mean over 1k.

Not shipped. `list_shape.py`, its 37 unit tests and the `--list-gate` filter all
stay in the tree: the detector is correct, cheap and well tested, and the finding
is about width, not about detection. Anyone reviving this should note that the
gate does identify hard questions -- 50.2% correct where it fires against 66.0%
where it does not -- so the shape is real. Width is just not the answer to it.

## Width on open-domain: measured properly, positive, and not shippable

Three arms, 830 paired questions, one window, 10 September. RAG's saved answers
pair all three. `od_wide.json`, `od_ground.json`, `od_ship.json`.

```
  arm                     acc    tokens    vs shipped              vs RAG
  wide  hyd10 gr0.003   81.3%   1,300t   +1.6  +35/-22  p=0.111   +1.2  p=0.426
  grounding   gr0.003   80.6%   1,047t   +0.8  +31/-24  p=0.419   +0.5  p=0.797
  shipped   (baseline)  79.8%     968t          --                -0.4  p=0.867
  RAG                   80.1%   1,464t
```

**The drift check is the most valuable number here.** Arm 3 is the shipping
configuration rerun from scratch, and it landed at 79.7% against the saved run's
79.8%, +21/-22. Our-arm cross-run drift on open-domain at n=841 is a tenth of a
point, not the 0.7 the harness docstring assumes and nowhere near the 1.2
measured at n=250. Every same-category comparison in this benchmark is more
trustworthy than we have been treating it.

**Facts are cheaper than turns.** Loosening grounding alone buys half the gain
for a quarter of the tokens, +0.8 for 79t against +1.6 for 332t. Neither clears
significance: only 55 to 57 of 830 questions change answer at all.

**Why the cheap arm still cannot ship, and this is the part that matters.**
Category is a benchmark label, not a property of an incoming question. Nothing at
inference time says "this is open-domain", so grounding 0.003 is global or it is
nothing. Global puts it on temporal, which is our largest lead at +25.5 and is
the category most sensitive to which facts clear the bar. Measuring that is
another 1,144 questions and was not run. Any future session tempted by "apply
the wide setting to open-domain only" should stop here: that configuration does
not exist outside the benchmark.

**`--ceiling` is reporting only.** It compares the mean against a number and
prints "under" or "OVER". It trims nothing. That is why the wide arm printed
"widest single question 2019t (under the 1800t ceiling)". No run in this project
was ever clipped by it, so past measurements stand, but the flag's name promises
enforcement it does not provide.

Also recorded: a 6-question smoke of the wide arm reported 1,592t and the full
841 came in at 1,300t. Small samples in this benchmark draw from one
conversation and misprice width. Third time this has happened.

## Adversarial cannot be fixed from the prompt. Two clauses, both dead.

`commit_exact` was the fix for the wording bug diagnosed below. It failed, and
it failed informatively: 767 questions, temporal plus adversarial, our arm only,
one session.

```
                       n    exact  shipped   diff   +fixed/-broke        p
  adversarial        446    36.1%    35.9%   +0.2      +28 / -27          1
  temporal           321    68.2%    71.7%   -3.4       +2 / -13    0.00739
  both               767    49.5%    50.8%   -1.3      +30 / -40      0.282
```

Strictly worse than shipping and strictly worse than `commit_short`. It gave up
the entire adversarial gain and kept most of the temporal damage.

**The two behaviours are welded together, and this is the finding.** Refusal
rate on adversarial, same detector across all four arms:

```
                  refuses   correct
  shipped          20.9%     35.9%
  commit_short     15.5%     41.1%
  commit_exact     20.9%     36.1%
  RAG              19.5%     50.4%
```

`commit_exact` refuses on 93 of 446, which is the shipped count to the question.
Its second sentence -- "this licenses nothing about the answer itself" -- did not
narrow the first, it cancelled it. The refusals came straight back: "The facts
do not mention Caroline running a charity race", "Melanie hasn't chosen an
adoption agency". Meanwhile the temporal damage barely moved, 13 broken against
`commit_short`'s 15, and six are the same questions. So the word "nearest" was
not the cause. Any instruction strong enough to stop the model refusing also
makes it commit to a date it should have left relative; soften it enough to
protect the dates and the refusals return. There is no third setting.

**And refusal was never the whole gap anyway.** RAG refuses at 19.5%, almost
exactly our rate, and still scores 14.5 points higher. Removing 24 refusals
bought `commit_short` 5.2 points, so the refusal deficit is worth about 5 points
and is recoverable. The other 9 points are wrong answers on questions where we
did commit, and no prompt clause touches them. That is a retrieval-shape
problem: compaction discards the conversational hedging RAG uses to spot a false
premise.

**Decision: stop prompt work on adversarial.** Ship the stack as measured. The
write-up reports +2.0 over RAG across all 1,985 questions, +6.8 across the four
non-adversarial categories, and names adversarial as the category we lose with
the mechanism above as the reason. `commit`, `commit_short` and `commit_exact`
stay in `VARIANTS` as the record of what was tried, none of them wired into any
shipping path.

No confirmatory paired run is needed for this. The shipping stack's numbers are
already paired and already measured.

## `commit_short` works, costs, and the cost looks like a wording bug

Finished, all 1,984 answered questions, our arm only, one session. The partial
read at 1,235 rows called every sign correctly and every magnitude loosely,
which is what a partial run is worth.

```
  category        n   commit_short  shipped   diff   +fixed/-broke        p
  adversarial   445      41.1%       35.7%    +5.4      +41 / -17    0.00223
  temporal      321      67.6%       71.7%    -4.0       +2 / -15    0.00235
  multi-hop      96      52.1%       57.3%    -5.2       +1 /  -6      0.125
  single-hop    281      51.2%       50.9%    +0.4      +13 / -12          1
  open-domain   841      79.0%       79.8%    -0.8      +15 / -22      0.324
  ALL          1984      63.4%       63.4%    +0.0      +72 / -72          1
```

Against RAG the overall is +2.0, exactly the shipping stack's number. Context
mean 964t against the shipped 964t, so the clause is free on the budget metric
and the only question is accuracy.

An exact wash, and not a quiet one: the adversarial gain and the temporal loss
are both significant, the same size, and opposite. Two things then say the
trade is not forced.

**It is not the dropped `qualifier`.** Only 13.1% of temporal is gated, and on
those 28 questions the run scores exactly what the shipped stack scores,
60.7% to 60.7%. Seven of the eight breakages are on the plain path, where
`qualifier` never was. So the one-clause-at-a-time confound is not the cause,
and the stacking fix will not rescue temporal on its own.

**It is the word "nearest".** What broke does not look like anything a premise
clause should be able to reach:

```
  gold "The week before 21 January 2022"   ->  "21 January 2022"
  gold "The week before 3 July 2023"       ->  "late June 2023, 11 August 2023"
  gold "a handwritten letter"              ->  "a cute note"
  gold "cozy and comfortable"              ->  "comfortable and invited"
  gold "Rome"                              ->  "Paris"
```

The finished run says the same thing on the questions RAG gets and we do not:

```
  gold "The week before 23 August 2023"    ->  "August 2023"
  gold "Becoming Nicole"                   ->  "the book Caroline recommended"
```

The second is the clause caught in the act. The title is in the facts, and the
model handed back a description of it instead.

`Answer from the nearest thing the facts hold` licenses two separate things and
only one was wanted. Answering despite a wrong premise in the *question* is the
entire point. Approximating the *answer* is not, and it fights two shipped
rules head-on: copy the wording from the facts, and leave a relative date in its
relative form. The model was obeying the clause.

`commit_exact` says the first without the second and re-asserts exact wording
inside the clause that was undermining it. Queued behind the screen: 767
questions, temporal plus adversarial, our arm only. Temporal is where the damage
is and adversarial is where the gain is, so that slice tests both sides of the
trade for 40% of the questions.

## The next paid run, ready to launch but NOT recommended

SUPERSEDED by the section above. Both clauses died, so there is nothing to
confirm and this run would spend three hours reproducing the shipping numbers we
already have paired. Kept because the command is smoke-tested and is the right
shape for any future confirmatory run: both arms, one session, all five
categories, clauses stacked rather than swapped. Roughly three hours at
concurrency 8. Swap the `--variant` value for whatever is actually being tested.

```
python -u scripts/benchmarks/_ab_vs_rag.py --shipping \
  --run C:/nmafc_ab/locomo_real \
  --categories temporal,multi-hop,single-hop,open-domain,adversarial \
  --variant "qualifier@gated,commit_short@everywhere" \
  --concurrency 8 --restart --out C:/nmafc_ab/confirm_all.json
```

Smoke-tested at `--limit 1` on 10 September: the stack opens, both clauses land
on the paths asked for, and the startup line names them. `--limit` is per
conversation, not per run.

Why this run and not the screen it follows. The screen regenerates our arm only
and pairs it against RAG answers made hours earlier in another process. This
project's own ground rule is that comparisons are paired inside one session,
because cross-run drift runs 1.5-2 points and every effect here is that size.
Reusing RAG is defensible -- RAG's arm is untouched by anything we changed --
but this is the table a paper prints, and the shortcut is easier to spend three
hours avoiding than to defend afterwards.

**Re-ingestion is not on this list, deliberately.** Checked on 10 September
rather than assumed: all ten `cold.db` files were last written at 09:35-09:36,
the turn-vector indexing pass, and the write-ahead logs are 0 bytes. Since then
the 1,539-question paired run, the adversarial control, the reach diagnostics,
three width screens, two clause screens and a 1,985-question run have all read
them and written nothing. `close_readonly` drops the reinforcement buffer
before `close()` and it works.

Nothing in this session runs at ingestion time either: `list_shape.py` is a
regex over the question, `commit_short` is a prompt clause, the widths are
query-time config. So re-ingesting would not validate anything. It would spend
5.3 hours producing different embeddings and different extractions, and every
number measured today would stop being about the stores that then existed --
including the ones a re-run would be checking against. Re-ingest only if the
extractor, the chunker, the embedding model or the fact schema changes.

## 10 September 2026: adversarial measured at last, and the honest headline

### The number that changes the headline

`_ab_vs_rag.py --categories adversarial` on the shipping stack, both arms in
one session, 446 questions:

```
  adversarial   446   ours 35.9%   RAG 50.4%   -14.6   p=4.1e-08   RAG
  context: ours 960 tokens, RAG 1486
  latency: paired median -38 ms, sign test p=0.276, no reliable difference
```

Two corrections come out of this and both matter more than the number.

**The cross-run historicals were wrong and were making us look worse.** The
figures carried into this session, ours 15.25% from `full_v3` against RAG's
32.29% from `full_v4_rag`, are from two different runs on two different days
under two different configurations. Paired in one session the same category is
35.9% against 50.4%. The gap is 14.6 points, not 17, and our own score is more
than double what the stale pair said. This is the fourth time a cross-run
comparison here has been misleading in a way a paired run fixed.

**The whole-benchmark table has to fold this in.** `SCORED` in `_ab_budget.py`
excludes adversarial, so every table this project has produced is over 1,539
questions and there are 1,985. Folded in:

```
                 scored 1539     adversarial 446     all 1985
  ours              71.4%             35.9%            63.4%
  RAG               64.6%             50.4%            61.4%
  lead              +6.8             -14.6             +2.0
```

`+2.0` is the honest headline, not `+6.8`. We still win overall and still win
on a third less context, but the margin is a fifth of what the reported table
says. Do not quote `+6.8` without saying what it excludes.

### Adversarial is a refusal problem, not a retrieval problem

Reach diagnosis over the 103 questions RAG answers and we do not:

```
  nothing arrived                4    3.9%
  a fragment                     4    3.9%
  most of it, a piece missing   17   16.5%
  all of it                     78   75.7%
```

Three quarters already have every gold content word in our window. And we
refuse constantly: 137 of 446 answers are a refusal of some kind, and we are
scored right on 3 of them. RAG refuses on 92 and is right on 1. So refusing is
worth about nothing and costs about 134 questions, 45 of which RAG answers.

The awkward part is that the shipped prompt already bans it. It says `NEVER say
"No information available"` and `Always prefer giving an answer over refusing`,
and the model refuses on a quarter of the category anyway. The ban names one
wording; the failures come in two. The second is a premise denial -- "Caroline
did not go camping with her family", "Melanie is not pursuing counseling" --
which obeys the ban to the letter and refuses all the same. LoCoMo's
adversarial questions swap the person or the object, and the model objects to
the attribution rather than answering from the fact it can plainly see.

Two clauses in `_ab_conversion.py`, screened on the 137 refusals rather than all
446, because that is the population either one could move. Our arm regenerated,
RAG's saved answers reused, `_run_open_domain_full.py --store per-conv`:

```
                    n    ours   shipped   fixed  broken        p
  commit          137    7.3%     2.2%      +10     -3     0.0923
  commit_short    137   19.7%     2.2%      +27     -3     8.4e-06
```

The shorter clause is nearly three times the longer one. `commit` explains why
refusing is wrong; `commit_short` names the swap the question performs and
stops. Explaining apparently gives the model somewhere to put a better-worded
refusal, which is what the smoke test showed it doing.

**A confound to know about before reading the all-category run.** `--variant`
appended one clause and only one, so `--variant commit_short` did not add
`commit_short` to the shipping stack, it swapped `qualifier` out for it. The
137-question screens above and the 1,985-question run launched at 14:35 are
therefore comparisons of two whole configurations, not isolations of one
clause. `qualifier` fires only on gated questions, about 15% of them, so most
of what moves is `commit_short` -- but "most" is not "all" and the run cannot
prove which.

Fixed rather than argued around. `parse_clauses` now lives in
`_ab_conversion.py` beside `VARIANTS`, both harnesses import it, and
`--variant qualifier@gated,commit_short@everywhere` is two clauses on two
different paths. A bare name still takes `--variant-where`, so every command
already written still means what it meant.

The fixes are exact and not judge noise: `necklace`, `The Lean Startup`,
`self-doubt`, `custom controller decorations`, `fashion department of an
international company`. The 3 broken are the same 3 questions where refusing
was scored right in the control, so the clause costs precisely what it should.
72 of the 137 still refuse, so half the refusals are still on the table.

`commit_short` was written on the assumption that the clause would be charged
against the context budget, and it is not. Measured, the `commit` run comes
back at 965 tokens against the control's 960 on a different subset of the same
category: the clause goes in the system prompt, and this project already
decided in `answer_type.py` that a system-prompt rule is the arm's own
instructions rather than retrieved context, on the grounds that RAG has a
system prompt too and nobody counts that either. Consistent, and defensible,
and worth saying out loud rather than leaving in a measurement: the clause is
real tokens sent to a real model, it is simply not tokens the budget metric was
ever defined to cover. `commit_short` is still worth running, but as the
cheaper wording rather than as the affordable one.

**`answer_first` is dead and should not be paid for.** It was written off
`full_v3`, where 22.8% of wrong adversarial answers contained every gold word
and wrong answers ran 25.1 words. Under the shipping stack wrong answers run
5.2 words against 3.2 for right ones and 2.4% contain every gold word. The
length rules already removed the thing it was written to remove.

### The list gate: a good detector whose fix costs real tokens

`src/nmafc/integration/list_shape.py`, `wants_list(question)`. A regex, no
embedding and no model call, the same shape as `answer_type.gate`. On
single-hop it fires on 141 of 281 and splits the category almost exactly along
the fault line found the day before:

```
              n    list-shaped gold   mean gold words   ours    RAG
  fires     141         95.0%               7.8        36.9%   33.3%
  quiet     140         37.1%               4.5        65.0%   57.9%
```

So the weak half is identified for free and with 95.0% precision. The fix is
not free. Thirteen widths screened on the 247 questions it fires on across all
scored categories, wins carried through beside losses, no generation:

```
  configuration          loss reach  win reach  tokens  overall mean
  shipping                  56.3%      89.8%      972       964
  30 facts, 6 turns         60.8%      90.6%     1210      1002
  30 facts, 3 turns         57.5%      89.8%     1042       975
  35 facts, 2 turns         58.0%      89.8%     1039       975
  40 facts, 0 turns         56.6%      87.9%      915       955
  20 facts, 10 turns        58.6%      90.8%     1196      1000
  16 facts, 10 turns        55.4%      90.2%     1077       981
  12 facts, 10 turns        52.8%      89.1%      956       961
```

The trade does not exist. Facts do not substitute for hydrated turns and turns
do not substitute for facts: every configuration that pays for one by cutting
the other lands at or below shipping's reach. The only thing that lifts reach
is more of both, and that is `30 facts, 6 turns` at an overall mean of 1,002.

So the list gate is a decision rather than a result. It buys 4.5 points of
reach and puts the mean over 1,000. It has not been converted to accuracy and
reach is not accuracy. Do not ship it on the strength of the reach table.

**The width table above was screened before a detector bug was fixed, and the
overall column is now optimistic.** `tests/unit/test_list_shape.py` caught it:
"What are Joanna's hobbies?" came back quiet. `Joanna's` was the first word
ending in s, so it was captured as the head, rejected as a possessive, and then
`finditer` moved past the wh-word it had already consumed and never reached
`hobbies`. Barring the apostrophe from the head class makes the engine backtrack
and take the possessive as the modifier it is.

Fixing it made the detector better on every axis and the arithmetic slightly
worse. Fire rate over the scored set goes 16.0% to 17.5%, single-hop coverage
129 to 141 of 281, precision 94.6% to 95.0%, and the split sharpens from
35.7/63.8 to 36.9/65.0. But the widening is now applied to 17.5% of questions
rather than 16.0%, so `30 facts, 6 turns` costs about 42 tokens on the mean
rather than 38: call it 1,006 and not 1,002. The reach column is unaffected --
it was measured on the questions that fired, and the twelve newly-caught ones
are the same shape as the rest -- but re-screen before shipping rather than
trusting that.

Worth keeping as a lesson about where these bugs hide. The detector was
measured against 281 real questions and looked excellent, and the measurement
could not see the miss because a miss is invisible in a precision figure. The
unit test found it in three minutes by asking a question the screen had no
reason to isolate.

## 10 September 2026: single-hop diagnosed, and the category we do not report

### Single-hop is two categories wearing one label

Split the 281 questions by how long the gold answer is:

```
  gold 1-3 words   138 questions   we are right 68.1%
  gold 4-8 words   103 questions   we are right 35.9%
  gold 9+ words     40 questions   we are right 30.0%
```

The lookup half is healthy. The list half is not, and it is half the category.
The four-cell split says the same thing from another direction: 36.3% of
single-hop is failed by **both** arms, against 11.3% on open-domain. A cell that
size is the task shape, not a disadvantage against RAG.

On losses we cover 29.2% of the gold's content words and 40.6% of what we say
is in the gold. On wins those are 69.3% and 68.5%. So we both miss items and
invent them: gold "Nike, Gatorade, Moxie, outdoor gear", ours "Nike, Gatorade,
Under Armour, outdoor gear".

Reach on the 102 both-wrong questions: 19.6% nothing arrived, 15.7% a fragment,
54.9% most of it with a piece missing, 9.8% all of it. **Only that last 9.8% is
a prompt problem.** A prompt fix caps out at about ten questions in 281.

### The width screen, free, no generation

`_screen_single_hop.py`, 138 losses and 143 wins carried through every
configuration:

```
  configuration          loss reach  long-gold  win reach   tokens
  shipping                  54.9%      61.9%      77.9%      979
  no fact separation        54.6%      61.5%      78.1%      968
  30 facts                  58.5%      65.7%      79.4%     1213
  30 facts, no sep          58.5%      65.7%      79.4%     1223
  30 facts, 10 turns        59.7%      67.4%      80.2%     1446
  rerank 60, 30 facts       58.5%      65.7%      79.4%     1213
```

- **The fact-separation theory is dead.** `fact_overlap_max` is not deleting
  list items: turning it off moves reach by 0.3 points the wrong way. It was a
  good theory and it is wrong.
- **Reranking wider does nothing.** 60 is identical to 40 at every figure, so
  the candidate pool is not the constraint; what is printed from it is.
- **Width is the only lever, and it costs tokens.** 20 to 30 facts buys +3.6
  loss reach and +3.8 on long gold for 234 tokens. Hydration 6 to 10 adds
  another +1.2 for a further 233. Both keep single-hop under RAG's 1,486 on the
  category, so neither breaks the "cost no more than RAG" constraint, but both
  break the self-imposed 1,000.

Untested and next: a **list gate**, the mirror of the length-rule gate that
already exists. The gate detects short-answer questions and asks for brevity;
nothing detects "enumerate everything" questions. Widening on those alone would
be close to token-neutral over the category, because the 1-3 word half can stay
narrow. Nobody has built or measured it.

### The category we have not been reporting

`SCORED` excludes adversarial, so every whole-benchmark table in this file
covers four of LoCoMo's five categories. Adversarial is 446 questions, larger
than temporal. The only measurements that exist are cross-run and predate the
shipping stack:

```
  adversarial   ours (full_v3)  15.25%   RAG (full_v4_rag)  32.29%
```

Folding those rates into the paired run's 1,539 would take overall from
ours 71.4% / RAG 64.6% to roughly **58.8% / 57.3%**, turning a +6.8 lead into
about +1.5. That arithmetic mixes runs and configurations and is indicative
only, but the direction is not in doubt and the size of it is not small.

**Correction, same day.** The first version of this section said adversarial
questions are unanswerable and the fix is to decline more often, and that doing
so would trade against the categories we win. That was wrong, written from the
category's name rather than from its contents, and it would have sent the next
session in exactly the wrong direction.

Reading the questions: adversarial carries a **false premise**, usually a
detail attached to the wrong person or the wrong object, and LoCoMo's gold is
still a real answer. "What hobby is a therapy for Tim when away from the
court?" has gold "Cooking". Declining is scored wrong. Correcting the premise
is scored wrong. Answering is scored right.

What our 378 wrong answers in `full_v3` actually do:

```
  our answer contains every gold content word, still wrong    86   22.8%
  we corrected the premise instead of answering              116   30.7%
  we declined to answer                                       88   23.3%

  mean length when wrong   25.1 words
  mean length when right   15.5 words
```

So the largest single bucket is answers that **already hold the gold** and lose
on framing: we lead with the correction, bury the answer, and run to 25 words
where a right answer runs to 15. That is a prompt problem worth up to 86
questions before touching retrieval at all, and the abstention story would have
found none of it.

Paired in-session measurement of the category is running as `adv_control.json`;
`SCORED` no longer acts as a ceiling in `_ab_vs_rag.py`, so `--categories
adversarial` now selects something instead of silently selecting nothing.

**No writeup should claim a whole-benchmark lead until this lands.**

## 10 September 2026: the real LoCoMo run, both arms in one session

`C:/nmafc_ab/locomo_real.json`, log beside it. 1,539 paired questions, one
dropped by the content filter from both arms. Ten per-conversation stores,
concurrency 8, roughly two hours end to end. This is the first head-to-head on
the standard per-conversation task rather than the merged haystack.

```
  category          n   ours     RAG    diff   gross      ourt   RAGt  saving
  OVERALL        1539   71.4%   64.6%   +6.8  +238/-133    964   1454    34%

  temporal        321   71.7%   46.1%  +25.5  +101/-19     943   1422    34%
  multi-hop        96   57.3%   44.8%  +12.5  +15/-3       961   1410    32%
  single-hop      281   50.9%   45.6%   +5.3  +51/-36      980   1486    34%
  open-domain     841   79.8%   80.3%   -0.5  +71/-75      967   1461    34%
```

McNemar exact: overall p=5.49e-08, temporal p=1.08e-14, multi-hop p=0.00754,
single-hop p=0.133, open-domain p=0.804.

Latency: paired median +29 ms against us, ours faster on 751 questions and RAG
on 788, sign test p=0.359. No reliable difference. Absolute medians are 1,830
and 1,832 ms and both are inflated by concurrency 8; only the paired difference
is meaningful at that setting.

### What this run settles

- **The haystack was not flattering us.** Every category lands within a couple
  of points of its merged-store figure, so ten conversations in one store is a
  harder task that produces the same verdicts. Both measurements now agree.
- **Open-domain is a tie, twice, independently.** -1.2 merged and -0.5 here,
  and 71 fixed against 75 broken. Two task shapes, same answer.
- **Multi-hop moved from marginal to real.** +3.1 merged, +12.5 here at
  p=0.0076, with 15 fixed against 3 broken. The per-conversation task is where
  hop-following pays; on the merged store the extra nine conversations of
  distractors were eating it.
- **Single-hop is still the weak category.** +5.3 and p=0.133 is not a win, and
  both arms sit near 50%.
- **Latency is level.** An interim read at n=1,265 showed RAG faster at
  p=0.0156; it did not survive the full sample. Worth remembering as a case of
  a p-value under 0.05 on a partial run meaning nothing.

Prep, all of which is reusable: stores copied to `C:/nmafc_ab/locomo_real`
rather than written in place, turn index built into all ten (2,957 of 2,957
verified), 20-question smoke test before spending.

## 10 September 2026: the real LoCoMo harness can now express the shipping stack

Everything measured since August has been on the merged haystack: ten
conversations in one store, 1,535 questions, every question searching ten
people's lives instead of one. Real LoCoMo is the per-conversation task, and
`_ab_vs_rag.py` is the harness for it -- both arms generated in the same worker,
seconds apart, so neither drift nor the provider's queue can enter.

It could not express the current stack. It was written before semantic turn
ranking, the hydration pool, source grounding, fact separation and the
length-rule prompt existed, and had flags for none of them. Now wired, all free,
no generation:

- **`--shipping`** sets the whole settled configuration at once: rerank 40,
  20 facts, pool 40, grounding 0.03, scan 8, hydrate 6, semantic 12.0 floor
  0.30, overlap 0.8, deduped headers, gated answer-type prompt, `qualifier` on
  the gated path only. Deliberately one flag: the configuration is nine numbers
  and a prompt, and typing eight of them right produces a run that looks valid
  and measures something nobody chose. Any flag given explicitly still wins.
- **Historical defaults are untouched.** A bare invocation still reproduces the
  configuration this harness was written against, so old comparisons stay
  reproducible. Verified both ways by running with a nonexistent `--run`, which
  prints the resolved configuration and spends nothing.
- **The resume guard from `_run_open_domain_full.py` is now here too.** Resume
  was keyed on the question and not the configuration, so a complete results
  file would answer nothing and report the old numbers as the new stack's.
- **A missing turn index is now named per store**, because semantic ranking is
  the largest single retrieval gain and it fails silently without one.

### What the ten stores actually hold

Two copies exist and they are not the same. Check before using either:

```
  scripts/benchmarks/results/full_v3/stores/   facts, 2,957 turn_text, NO turn_vectors
  C:/nmafc_ab/stores/                          facts only -- no turn_text at all
```

`full_v3` is the usable set. `C:/nmafc_ab/stores/` predates hydration entirely,
and a run there would print facts with an empty source block and report it as
the shipping stack.

`_backfill_turn_text.py` was written for the case where turn text is missing. It
turned out not to be needed on `full_v3`, and it is kept because the recovery it
does is worth having: turn N's text is the Nth exchange from
`build_dated_exchanges`, a pure function of the transcript, so the text is
rebuildable from the dataset for free instead of by 5.3 hours of re-ingestion.
It refuses to write unless the dates it rebuilds match the store's own
`turn_timestamps` turn for turn, because a turn_text table that is off by one
hydrates confident and wrong.

### What is still needed before the run

1. `_build_turn_index.py --store <path>` on each of the ten `full_v3` stores.
   2,957 short embeddings in total, no LLM extraction.
2. The paired run itself: 1,540 scored questions (adversarial is not scored),
   both arms plus both judgements, so about four LLM calls per question and
   roughly 3.7x the generation of one 839-question open-domain run.

Neither has been spent. Nothing is approved.

## 10 September 2026: four prompt theories, four deaths, and the honest ceiling

The question was whether open-domain can be won at 957 tokens by any means
other than spending more. Everything below was tried and measured. None of it
worked, and the pattern across them is more useful than any of them.

### The losses were diagnosed properly first, and two popular theories died free

On `od_ship.json`, the shipping run:

- **Judge noise is real, tiny, and symmetric.** On the 413 questions where our
  answer and RAG's share every content word, the judge disagreed 5 times: 3 for
  RAG, 2 for us. Net one question. There is no scoring artifact to exploit and
  the gap is not a measurement error.
- **We are not systematically terse.** Ours averages 3.1 content words, RAG 3.2,
  gold 3.3. The "our answers are too short" theory is wrong in aggregate.

What did survive:

```
  losses where our answer is a strict subset of RAG's winning answer : 12
  wins   where RAG's answer is a strict subset of ours               :  4
```

Net 8 questions against a 10-question gap, and **10 of the 12 are single-value
questions**, not compound ones. RAG wins them by naming every candidate and
letting the judge score containment -- gold "a cactus in the desert", RAG "the
sunset painting and the cactus painting", marked right while leading with the
wrong one.

### `hedge`: built from that, and it does not work

Ungated path, 713 questions, with its control generated in the same window
rather than borrowed:

```
                ours     RAG    diff
  control      79.0%   79.5%    -0.6
  hedge        78.3%   79.5%    -1.3      fixed 22  broke 26  net -4
```

The instruction was obeyed -- answers naming several candidates went from 153 to
188 -- and it still **fixed 2 of the 11 questions it was built for** while
breaking 26 elsewhere. The containment asymmetry is real in the losses and does
not convert when you ask for it.

### The pattern, which is the actual finding

Four instructions have now been tested on the ungated path:

```
  assemble     on both paths   broke the short-answer questions it should not have touched
  assemble2    ungated         +17 / -23
  qualifier    ungated         +18 / -28
  hedge        ungated         +22 / -26
```

**Every one of them churns 40 to 50 verdicts and nets zero or negative.** That
is not four bad instructions; that is the answer being determined by the
context and the instruction adding noise on top. The only instruction that ever
netted positive is `qualifier` on the gated path, +4/-0, where the gold is exact
and the model's job is to choose rather than to phrase.

Token-neutral budget routing was also tested and is flat: gated reach is 68.8%
at the current 20 facts / 6 turns against 65.6% at every trade, and the paid
version measured +2/-0 in session but showed no gain when spliced (79.26%
against 79.40% already measured).

### The honest ceiling at 957 tokens

**Open-domain cannot be won at this context size with anything found here.**
Measured state: 79.4% against 80.6%, ten questions, p=0.477 -- statistically
indistinguishable, not a win, and it should never be written up as one.

The one route with evidence behind it needs more context and was discarded for
breaching a self-imposed limit rather than a competitive one: at hydrate 10 and
20 facts, **1,265 tokens, loss reach was 51.7% against 47.5%** at the shipping
setting. RAG spends **1,459** on this category, so that configuration is still
cheaper than the thing it has to beat. If the binding rule is "cost less than
RAG" rather than "under 1,000", that is the experiment to run and its expected
effect is well clear of the noise floor, which is true of nothing else left.

## FINAL, 9 September 2026: the whole benchmark, one configuration, under 1k

All four categories regenerated under the shipping configuration in one window.
`od_ship.json`, `cat_temporal.json`, `cat_single-hop.json`, `cat_multi-hop.json`.

```
  category         n    ours     RAG    diff  shipped  our tok  RAG tok
  open-domain    839   79.4%   80.6%    -1.2    74.3%      957     1459
  temporal       321   67.9%   46.1%   +21.8    64.8%      933     1424
  single-hop     279   47.7%   43.4%    +4.3    45.2%      964     1477
  multi-hop       96   49.0%   45.8%    +3.1    44.8%      951     1394
  -----------------------------------------------------------
  WHOLE BENCH   1535   69.3%   64.4%    +4.9    65.1%      953     1451
```

**+4.9 points over RAG across 1,535 questions at 34% of its context cost.** We
answer 232 questions RAG misses; it answers 157 we miss; sign test p=0.000168.
Every category is under the 1,000-token constraint, the widest single question
in any category is 1,500t, and the mean is 953.

**Open-domain is the one category RAG still leads, by 1.2 points, p=0.477.**
That is not a tie on the point estimate -- it is 10 questions -- but on 839
questions the two are not statistically distinguishable. Called that way and no
stronger. It was -1.8 at the start of the session on 1,031 tokens.

**Every category also beats the shipped stack**, which is the fairer test of the
session's work: +5.1 open-domain (p=8.72e-05), +3.1 temporal, +2.5 single-hop,
+4.2 multi-hop.

### The shipping configuration

```
  rerank_top_k              40
  context_facts_top_k       20
  hydrate_pool              40
  source_grounding        0.03
  hydrate_scan               8
  hydrate_top_k              6
  scan_semantic             12       <- new this session
  scan_semantic_floor     0.30       <- never binds, kept off by default
  scan_span_facts            0       <- built, near-neutral, off
  dedupe_source_headers   true
  overlap                  0.8
  answer prompt: gated answer-type + `qualifier` on the gated path only
```

Requires `turn_vectors` in the store. Build it with `_build_turn_index.py`; a
run without it prints `NO INDEX, semantic ranking is inert` and is not this
configuration.

### What was tried and did not work, so nobody repeats it

- `assemble` on both paths: broke the short-answer questions it should not have
  touched.
- `assemble2` on the ungated path: +17/-23, inside drift, null.
- `qualifier` on the ungated path: +18/-28, negative. The same clause is +4/-0
  on the gated path. **Prompt clauses are path-specific; a variant with no
  targeting is a variant that has not been tested.**
- Span-bounding the turn scan: correct, near-neutral, and not additive with
  semantic ranking. Off.
- Retrieval tuning generally: grounding, hydration depth, fact count, line
  trimming, pattern separation, adaptive hydration, scan scoping, semantic
  ranking. The last of these is the only one that paid.

## State as of 9 September 2026

### PAID: `qualifier` everywhere. It reproduces on the short path and it costs on the long one

All 839, `--variant qualifier --variant-everywhere`, run A as the control.
78.2% overall against A's 79.0%, so as a whole configuration it is worse. Split
by path, which is the reason the run was worth making:

```
                n     A    qualifier    RAG     fixed  broke   net
  gated       123   81.3%     84.6%    86.2%      4      0     +4
  ungated     705   78.7%     77.3%    79.4%     18     28    -10
```

**The instruction is right where it was aimed and wrong everywhere else**, which
is the same lesson `assemble2` taught from the other side. Telling the model to
hunt for the detail that discriminates between similar facts helps when the
answer is one value and several candidates compete for it; on an open question
it narrows an answer that needed to be broad.

**The gated effect replicated.** Two independent samples of the same 123
questions, generated hours apart in different runs:

```
  run 1, gated-only        fixed 3   broke 0
  run 2, everywhere        fixed 4   broke 0
```

Seven fixes, no breakage, across both. Two of the five questions were fixed in
both runs and three in only one, so part of it is drift -- but drift breaks
things as well as fixing them, and **nothing broke in either run**. That is the
part that does not look like noise.

### Best configuration on record, and the gap is now 7 questions

`qualifier` on the gated path only, candidate retrieval, spliced with A's
ungated answers, which the change cannot touch because their prompt is
byte-identical:

```
                                             ours     RAG    gap
  A alone                                   79.11%  80.43%  -1.33
  + qualifier on gated (run 1 sample)       79.47%  80.43%  -0.97
  + qualifier on gated (run 2 sample)       79.59%  80.43%  -0.85
  qualifier everywhere                      78.38%  80.43%  -2.05
```

**About 7 questions behind, at 958 tokens against RAG's 1,459.** Started the
session at -1.8 on 1,031 tokens.

The ungated path is where the remainder sits, and it has now taken three
prompt changes -- `assemble`, `assemble2`, `qualifier` -- all null or negative,
on top of an exhausted retrieval search. Nothing free is left there.

**One clean run would settle the headline**: `--variant qualifier
--variant-where gated` over all 839 gives the shipping configuration unspliced,
in one window, instead of a number assembled from two. Not run.

### PAID: `qualifier` on the short-answer path is the first prompt change that works

Three runs of the 126 gated questions only, launched together, with a control
generated in the same window rather than borrowed from an earlier run. Running
the path under test alone rather than all 839 costs less than half a category
run and removes the 61-verdict drift that swamped every earlier prompt control.

```
  gated only, n=123 paired      ours     fixed  broke     vs RAG 86.2%
  control                      81.3%         -      -           -4.9
  qualifier      on gated      83.7%         3      0           -2.5
  discriminate   on gated      82.9%         2      0           -3.3
```

**Three fixed, nothing broken.** That is the first one-sided result any prompt
variant has produced. Every earlier attempt traded roughly evenly and died in
the drift: `assemble` +12/-5 then -3 on a control that could not resolve it,
`assemble2` +17/-23 on the ungated path. The difference is targeting -- the
instruction now lands only on the questions whose failure it describes.

Spliced onto run A's untouched 713 ungated answers, which the change cannot
have affected because the prompt on that path is byte-identical:

```
  ours 79.43%   RAG 80.50%   gap -1.08     (run A alone was -1.44)
```

**Honest size.** Three questions. A 3-0 split has a two-sided sign-test p of
0.25, so this is direction without significance, and the gated set is 126
questions so a bigger sample of it does not exist. What makes it worth keeping
despite that is the zero: a change that fixes 3 and breaks 0 cannot be costing
anything, which is not true of any earlier variant.

`discriminate` is the same effect slightly weaker and is the strictly larger
instruction, so `qualifier` is preferred: fewer words, same sign, no breakage.

Files: `C:/nmafc_ab/gated_control.json`, `gated_qualifier.json`,
`gated_discriminate.json`.

### The remaining gap is 9 questions, split evenly between the two paths

Under the candidate plus `qualifier` on gated:

```
  gated    n=126   ours 83.7%   RAG 86.5%   contributes -0.38 of the total gap
  ungated  n=713   ours 78.7%   RAG 79.5%   contributes -0.68 of the total gap
```

**`qualifier` has never been tested on the ungated path.** Only `assemble` and
`assemble2` have, and both are completeness instructions; `qualifier` is a
discrimination instruction and is a different change. One run of 839 with
`--variant qualifier --variant-everywhere` would test the ungated path and take
a second sample of the gated one at the same time, with run A as its control,
and either path can be spliced out afterwards. That is the decisive remaining
experiment. Not run, not approved.

### PAID, both approved and both run: the candidate is the best yet and still 1.5 behind

Two full 839-question open-domain runs, launched together so they share a time
window and the RAG verdicts are the same saved answers in both.

```
                              ours     RAG    diff   win  lose       p   tokens
  A  retrieval candidate     79.0%   80.6%    -1.5    73    86   0.341      958
  B  A + assemble2, ungated  78.4%   80.6%    -2.1    77    95   0.195      958
```

`C:/nmafc_ab/od_cand.json`, `C:/nmafc_ab/od_variant.json`. Configuration in
both: rerank 40, facts 20, pool 40, grounding 0.03, scan 8, hydrate 6, semantic
12 / floor 0.30, deduped headers, overlap 0.8, gated answer-type. 2,957 turns
indexed, confirmed in both logs.

**A is the best result this project has measured, on both axes at once.** 79.0%
against the previous high of 78.8%, at 958 tokens against 1,031, and RAG spends
1,459 for its 80.6%, so we are **34% cheaper**. Against the shipped stack A is
+4.8 points, +75 fixed against -35 broken, p=0.000172.

**It is not a tie and it should not be called one.** The point estimate is 13
questions behind. p=0.341 means the two are not statistically distinguishable on
839 questions, which is a weaker claim than a tie and is the honest one.

**B closes the prompt lever properly this time.** Targeted at the 705 ungated
questions where it applies, `assemble2` fixed 17 and broke 23 -- inside the
61-verdict drift floor, so the reading is no effect, not harm. Two attempts,
two prompt variants, both targetings; the instruction does not convert reached
gold into right answers. Stop paying for this line.

### Correction: the gate figures that justified B's targeting were wrong

I reported the length gate as ours 93.2% / RAG 90.5% on the gated path, and used
"we already beat RAG where a rule fires" as the reason to keep the instruction
off it. Recomputed from the results files with the same `gate()`:

```
  file            gated n=126   ours     RAG        ungated n=713   ours     RAG
  od_full.json                 84.9%   86.5%                       77.7%   79.5%
  od_full2.json                83.3%   86.5%                       77.1%   79.5%
  od_cand.json                 81.0%   86.5%                       78.7%   79.5%
```

**RAG is ahead on both paths, not just one.** The 93.2/90.5 figure does not
reproduce from any saved file and should be treated as an error of mine, not as
a result. B's targeting was defensible anyway -- the ungated path is where 73 of
the 86 losses are -- but the stated reason for it was false.

The corrected table says something more useful. Under the candidate the
**ungated path is nearly level, -0.8 points**, and the gap has concentrated in
the 126 short-answer questions, where we are **-5.5**.

### The new retrieval helps long answers and hurts short ones

Paired against the last paid run, question by question:

```
  gated     n=123    fixed  1   broke  6
  ungated   n=705    fixed 34   broke 28
```

The ungated movement is inside drift. The gated movement is small in absolute
terms, 5 questions, but it is one-sided and it points the same way as the table
above: more hydrated dialogue and one fewer turn is a good trade for a compound
answer and a bad one for a single value, where extra dialogue supplies extra
plausible candidates to choose wrongly between.

The 13 gated losses, diagnosed by coverage (`--gate gated`, new flag):

```
  nothing arrived        2   15.4%
  a piece missing        7   53.8%
  all of it              4   30.8%
```

They are discrimination failures, not completeness failures -- the wrong one of
several similar candidates:

```
  Q  What novel is Evan reading that he finds gripping?
     gold  The Great Gatsby            ours  The Last Devil to Die
  Q  What book recommendation did Tim give to John for the trip?
     gold  A fantasy novel by Patrick Rothfuss    ours  The Alchemist
  Q  Which movie's theme is Tim's favorite to play on the piano?
     gold  Harry Potter and the Philosopher's Stone   ours  Star Wars
```

Two more are abstentions -- "unknown", "not specified in the facts" -- on
questions where half the gold had arrived.

**This is the failure the `qualifier` and `discriminate` variants were written
for, and neither has ever been tested on the gated path alone.** That is the
one prompt experiment left with a stated reason, and it is the exact opposite
targeting of the one that just failed. It is also small: 13 questions is 1.5
points, so it could close the whole gap and could equally be noise on 126
questions. Not run. Not approved.

## State as of 8 September 2026

### The turn scan was searching ten conversations at once

Printing one rendered context found this. Asked which novel Evan is reading that
he finds gripping, the top fact was factually wrong, the gold sat at rank 2, and
**four of the five hydrated turns came from other people's conversations** --
John and Maria on houses, Joanna on a screenplay, Tim on a different novel --
each matched on "gripping" or "novel". Most of the SOURCE budget went on
distractors.

The cause is structural. `_build_haystack.py` lays ten conversations end to end,
and the merge flattens them onto one agent and one conversation id: querying the
store directly returns exactly one group, `('default', 'default', 2957, 1,
2957)`. So `all_turn_text()` hands the scan all 2,957 turns of all ten and BM25
ranks across the lot, where a rare word in a stranger's dialogue beats a common
word in the right conversation.

The boundary survives as a **turn range**, because the merge offsets turns per
source. `scan_span_facts` and `scan_span_margin` bound the scan to the range of
the top retrieved facts' turns. That is inference from retrieval's own output,
not knowledge of which conversation the question came from, so it holds on a
store whose sources were never labelled -- which is this one.

Free screen on 90 losses: the span of the top 5 facts widened by 80 turns
excludes **36% of hydrated turns** while still containing the gold turn in
**98%** of the questions where the gold is in a turn at all.

### The scoping is correct and it buys almost nothing

Reach, on the 178 real losses and 178 wins, hydrate 7, dedupe, overlap 0.8:

```
  config                        losses          wins    tokens
  40:18:40:0.03:8               44.9%          88.2%       942   <- unscoped
  40:18:40:0.03:8:5:150         46.1%          88.2%       944
  40:18:40:0.03:8:12:20         44.9%          88.8%       945
  40:18:40:0.03:8:3:80          44.9%          87.6%       946
  40:18:40:0.03:8:5:20          44.9%          88.2%       946
  40:18:40:0.03:8:5:80          44.4%          87.6%       944
  40:18:40:0.03:8:5:0           41.6%          87.6%       949
```

Best is +1.2 points of loss reach at 2 tokens. Tightening the span past that
hurts: at margin 0 reach falls 3.3 points.

**The free screen promised 87% and delivered 1.2 points, and the reason is
worth keeping.** That screen measured what reach *would* be if every slot freed
from an out-of-span turn went to the right turn. They don't. They go to a
different wrong turn. An upper bound is not a gain -- the same mistake shape as
pricing pattern separation as a token saving, and the screen's own printed
caveat said so before the number was believed.

Kept anyway, because reach cannot price what it does: removing a **wrong**
candidate is invisible to a measure that only asks whether the right one
arrived, and the rank-1 fact in that Evan context was wrong.

### Words find the turn half the time; meaning finds it two thirds

This is the largest free finding of the session. The deficit is not mostly
precision: 55% of the losses have no gold word in context at all, so the
turn holding the answer is never hydrated. (**That 55% is measured wrong and the
corrected figure is 13.4%** -- see "The 55% was an all-or-nothing test" below.
The work it prompted still stands on its own numbers; the premise does not.)
The scan ranks turns by BM25, which
is exactly right for the failure it was built for -- a concrete noun the
extractor generalised away, "Hoodies" where we answered "clothing" -- and blind
to the failure that is left, where question and turn say the same thing in
different words.

Both rankers scored against the 106 of 178 losses whose answer is in some turn,
asking where each puts that turn:

```
  gold turn ranked inside the top
  ranker           1       3       8      20      50
  BM25         23.6%   39.6%   50.0%   59.4%   68.9%
  cosine       32.1%   47.2%   65.1%   77.4%   90.6%
  better       37.7%   55.7%   68.9%   81.1%   94.3%
```

The scan takes 8, so the column that decides it is 8: **50.0% against 65.1%**,
and 68.9% for whichever ranker is better, so there is something in each and the
blend is worth having rather than a straight swap. Twenty questions are reachable
at 8 by meaning and not by words:

```
  Q  How does John feel while surfing?
  A  super exciting and free-feeling      BM25 rank 834, cosine rank 5

  Q  How is Maria's new puppy adjusting to its new home?
  A  doing great - learning commands and house training
                                          BM25 rank 299, cosine rank 1
```

Built as `scan_semantic` (weight of meaning, fused on rank because a BM25 score
and a cosine share no scale) and `scan_semantic_floor`. The floor is needed
because the old guard -- drop a turn sharing no word with the question -- would
throw away precisely the turns this exists to reach, and a cosine is never zero,
so without a floor every turn would qualify.

Vectors live in a new `turn_vectors` table, written by
`_build_turn_index.py`, never searched to answer a query, only to order turns
competing for a hydration slot a fact already won. All 2,957 turns indexed from
the screen's cache, so nothing was embedded twice. `cold.db` backed up to
`C:/nmafc_ab/cold.db.bak` first.

### It works, and the weight plateaus at 12

Reach, 178 real losses and 178 wins, rerank 40, facts 18, pool 40, grounding
0.03, scan 8, hydrate 7, dedupe, overlap 0.8:

```
  scan_semantic   floor      losses     wins   tokens
        0 (off)       -       44.9%    88.2%      942
            1.0    0.00       46.6%    88.2%      948
            1.0    0.30       46.6%    88.2%      948
            3.0    0.30       47.2%    88.2%      950
            6.0    0.20       48.3%    88.8%      952
            6.0    0.30       48.3%    88.8%      952
           12      0.30       48.9%    88.8%      953   <- settled here
           25      0.30       48.9%    88.8%      954
           60      0.30       48.9%    88.8%      955
          200      0.30       48.9%    88.8%      954
```

**+4.0 points of loss reach and +0.6 of win reach for 11 tokens.** This is the
best retrieval configuration measured in this project: at 953 tokens it beats
the 1,219-token config that was the previous high (47.8% / 88.8%), so it is not
a token trade at all.

Weight 12 is the settling point because 12, 25, 60 and 200 are identical and 12
keeps the most word signal of the four.

**`scan_semantic_floor` has never bound and should be treated as untested.**
Floors of 0.0, 0.20 and 0.30 all give the identical row at weight 12, and 0.0
disables the clause entirely, so no turn has ever been taken on meaning alone --
a turn sharing no word with the question ranks near-last in the word half and
never reaches the top 8 of the fusion, even at weight 200 where meaning
outweighs words sixteen-fold. The whole gain is **re-ordering turns that already
qualified**, not admitting new ones. The knob is kept, defaulted off, because
the guard it replaces would be wrong on a store where meaning-only turns do
surface; it earns nothing on this one.

Widening the scan is worse, so it stays at 8: scan 12 gives 48.3% and scan 20
gives 47.8%, against 48.9% at 8. Ranking turns better is not the same as ranking
more turns, and past 8 the extra candidates are noise the fusion has to beat.

Turn budget, at weight 12, and this is where the token headroom is:

```
  hydrate      losses     wins   tokens
        5       46.6%    88.2%      840
        6       48.3%    88.8%      898
        7       48.9%    88.8%      953
```

Hydrate 5 with semantic ranking reaches **more gold at 840 tokens than the
shipped candidate reaches at 942** without it. Hydrate 6 costs 0.6 points
against 7 and frees 55 tokens, which matters because the paid run showed the
fact cut from 20 to 18 cost accuracy the reach screen could not see.

Spending those freed tokens on facts instead of turns is the better trade, and
it settles the configuration:

```
  facts  hydrate     losses     wins   tokens
     18        7      48.9%    88.8%      953
     20        6      49.4%    88.8%      958   <- settled candidate
     22        6      49.4%    88.8%     1016
     22        5      47.8%    88.2%      958
     24        5      48.9%    88.2%     1012
```

**Facts 20 / hydrate 6 / semantic 12 / floor 0.30, at 958 tokens, is the
configuration to run.** It restores the fact count the paid run showed was
costly to cut, beats the previous 1,219-token high (47.8% / 88.8%) at a fifth of
the token price, and sits under the 1k ceiling with 42 tokens to spare. Facts 22
buys nothing for 58 more tokens; at hydrate 5 the same 958 tokens buy 1.6 points
less. It is now the default in `_run_open_domain_full.py`.

**It has never been generated against.** Every accuracy number in this file
comes from a different retrieval configuration. 49.4% is reach, not accuracy.

**Span scoping is not additive with this and is dropped from the candidate.**
Bounding the scan on top of semantic ranking costs 0.5 to 0.6 points at every
weight tried (48.3 -> 47.8 at weight 6, 48.9 -> 48.3 at weight 12). Both fix the
same failure, and meaning does it better: a turn from a stranger's conversation
is off-subject by meaning too, so the span has nothing left to exclude and only
removes turns the fusion wanted. The code stays, defaulted off.

### The arithmetic says this still does not close the gap, and it should have been done first

The deficit is 1.8 points on 839 questions, about 15 answers. The reach screen
runs on 178 losses, so **one point of loss reach is 1.8 questions**, and newly
reached gold converts maybe half the time. Closing 1.8 points therefore needs
roughly **17 points of loss reach**. The whole session's sweeps have moved it 4.0,
and every retrieval configuration ever screened sits under 49%.

Retrieval tuning cannot get there. That is now measured, not suspected, across
grounding, hydration depth, fact count, line trimming, pattern separation,
adaptive hydration, scan scoping and semantic turn ranking.

What is left is the other half of the deficit: the questions RAG wins where the
gold is in our context already and we fail to assemble it. `assemble` measured
+12/-5 on that fixable set and was parked on a 90-question control that showed
-3, but 1 fix against 3 breaks has p=0.625 and a 95% band from -53 to +20.
**That lever was never resolved, only under-measured**, and it is the only
tested change that moves the failure shape the deficit is actually made of.
Resolving it needs a control sample large enough to see a breakage rate near 1%,
which is paid. Not spent; awaiting a decision.

### The 55% was an all-or-nothing test, and it hid where the deficit actually is

Everything above was built on `present()`, which asks whether **every** content
word of the gold reached the context and returns one bit. That pools "we gave
nothing" with "we gave half", which are opposite problems with opposite fixes,
and the paid experiment was about to be chosen without knowing which one is
bigger. Replaced with **coverage** -- the fraction of the gold's content words
that reached -- in `_diagnose_partial_reach.py`. No generation, so it was free.

The 97 questions RAG answers and we do not, rendered under the settled candidate
(facts 20, hydrate 6, semantic 12):

```
  nothing arrived                  13   13.4%
  a fragment                        5    5.2%
  most of it, a piece missing      23   23.7%
  all of it                        56   57.7%

  a retrieval problem (something arrived, not all)   28   28.9%
  a prompt problem (all of it arrived)               56   57.7%
```

**13.4%, not 55%.** The premise that the answer is usually absent is wrong under
this retrieval, and the reason the old number was so much larger is that a
question missing one word of a nine-word gold scored the same as one missing all
nine. Two changes are entangled here -- the better measure and the better
retrieval -- and this run cannot separate them, but either way the conclusion
holds: **retrieval is not where the remaining deficit lives.**

What the losses look like is under-answering. The gold is a list and we return
part of it:

```
  A  carving out some me-time each day for running, reading, or playing the violin
  we said  running, reading, playing violin                         (92% reached)

  A  Zelda BOTW for Switch, Animal Crossing: New Horizons, Overcooked 2
  we said  Zelda: Breath of the Wild, Animal Crossing: New Horizons (89% reached)

  A  breaking tasks into smaller pieces and setting goals, using planners
  we said  breaking tasks into smaller pieces                       (89% reached)
```

Those are not retrieval failures. Every piece was in the context.

### Why `assemble` failed, and it was not the instruction

The answer prompt has a length gate: rules that fire on question shape and tell
the model how long to answer. Counting where it fires, across all 839:

```
                        n     ours     RAG
  a length rule fires   126   93.2%   90.5%
  no rule fires         713   77.1%   79.5%
```

**We beat RAG by 2.7 where a rule fires and lose by 2.4 where none does, and 84
of the 97 losses are ungated.** The gated questions are the short-answer ones --
a name, a date, one value -- and they are nearly solved.

`assemble` was appended to **both** prompts unconditionally, so an instruction
saying "give every part of the answer" landed hardest on the 126 questions that
are single-value and already at 93.2%, where the only thing it can do is add
words to answers that were right. That is a targeting bug, not evidence the
instruction is wrong, and the 90-question control that killed it was sampled
across both paths.

Made targeted rather than re-argued: `--variant` now applies only where no
length rule fires, with `--variant-everywhere` to restore the old behaviour.
Mean gold length on the losses is 5.3 words against 4.8 on the wins, which is
the same signal from the other side.

### The drift floor is 61 verdicts, and it means the control that killed `assemble` measured nothing

The two full paid runs share 828 questions and near-identical configurations.
Comparing them question by question:

```
  run 1 right   654    run 2 broke 33   (5.0%)
  run 1 wrong   174    run 2 fixed 28   (16.1%)
```

**61 verdicts churn between two runs that should agree.** A 90-question control
showing -3 is inside that, comfortably. Any future control has to clear this
floor before it means anything, and the cheapest way to clear it is to test on
the path where the change is supposed to act rather than diluting it across 839.

### The under-1k config was cut on the wrong axis, and the sweep says which one

The paid cheap arm (968t, 79.1%) cut two things at once: facts 20 -> 16 and
turns 10 -> 7. Reading the 11 questions it broke says only one of those cuts was
the expensive one. Their gold answers average **6.9 words against 4.9 for the
category as a whole** -- these are compound answers assembled out of several
stored facts, and the fact cap is what took the pieces away.

So the whole cut should come from turns and none of it from facts. The sweep, at
`rerank 40 : pool 40 : grounding 0.003 : scan 8`, deduped headers, on 120 losses
and 120 wins, no generation:

```
              20 facts                18 facts                16 facts
  h10   1265t  51.7 / 85.0  +4/-0                      1143t
   h8   1139t  50.0 / 85.0  +4/-0    1077t             1018t  49.2 / 84.2
   h7   1079t  49.2 / 85.0  +4/-0    1018t  49.2/84.2   957t  49.2 / 84.2  +4/-1
   h6   1020t  47.5 / 85.0  +4/-0     960t  47.5/84.2
   h5    955t  45.8 / 85.0  +4/-0     895t  45.8/84.2
```

Two things fall out of that table and they point the same way.

**Every drop from 20 facts to 18 costs a win and buys nothing.** The `+4/-1`
appears at 18 in every row and never at 20, and loss reach does not move. The
fact cap is a pure loss, which is exactly what the broken-question reading
predicted.

**Turns are the cheap axis.** Going 10 -> 5 turns holds win reach at 85.0% with
zero displaced wins, and costs 5.9 points of loss reach for 310 tokens. Since
the category runs about 194 wins to 55 losses, holding the win side flat is
worth roughly 3.5x whatever the loss side gives up.

**h5 / 20 facts is 955t at 45.8 / 85.0 with +4/-0** -- under the ceiling already,
and strictly better on the win side than the 968t config that was paid for.

### PAID RUN: all 839 open-domain questions. RAG is still ahead by 1.8

8 September 2026. `C:/nmafc_ab/od_full.json`, one arm, paired against the RAG
answers saved in `haystack.json`. Config: rerank 40, 18 facts, pool 40,
grounding 0.003, scan 8, hydrate 7, deduped headers, overlap 0.8, gated prompt.

```
                          n      ours    theirs    diff    win   lose        p
  vs RAG                839     78.8%     80.6%    -1.8     72     87    0.267
  vs shipped stack      839     78.8%     74.3%    +4.5     75     37 0.000421

  context: ours 1,031t, RAG 1,459t -- we spend 29% less
```

**The retrieval work is real and the RAG win is not.** +75 fixed against -37
broken at p=0.0004 is a solid four-and-a-half point gain over the shipped stack,
holding across the whole category rather than a sample of it. But we are behind
RAG by 1.8, and at n=839 the noise band is about 0.7, so that deficit is
probably not noise even though p=0.267 on the sign test.

**The +1.6 measured against RAG on 250 questions was an artefact.** Split the
same run by whether a question was in the old paid sample:

```
                            n     ours      RAG   shipped   diff vs RAG
  the 250-q paid sample   252    79.8%    81.0%     75.8%          -1.2
  the other 590           587    78.4%    80.4%     73.6%          -2.0
  all 839                 839    78.8%    80.6%     74.3%          -1.8
```

Two separate things were flattering the earlier number, and they compound. The
sample is mildly easy for us on both the old stack (75.8 against 73.6) and the
new one (79.8 against 78.4). And **250 questions cannot resolve 1.8 points.**
Four paid runs were spent on samples that could not answer the question they
were run to answer. Do not run another 250-question A/B against RAG.

### The fact cap costs accuracy the reach screen cannot see

The candidate scored **79.8% on the 250-question sample where the 1,280t arm
scored 81.1% and 82.3%**. Same questions, same store, same prompt. So dropping
20 facts to 18 cost something -- and the reach screen had said their win-side
reach was *identical* at 85.0%, with no win displaced.

That is the third free screen this session to misprice a change, and the three
misprice for one reason. `present()` asks whether every content word of the gold
appears somewhere in the render. It cannot see whether the model can **assemble**
the answer out of what is there. For a compound gold spread across several facts,
dropping one fact leaves every word still present in some other fact and the
answer no longer constructible.

The rule that follows: **a reach screen prices reach. It does not price a
change that alters what the model has to do with what it reached.** Prompt
changes and fact-count changes both fall in that class.

### The remaining deficit is one failure shape, and it is discrimination

The 87 questions RAG answers and we do not, bucketed by how our answer relates
to the gold:

```
    52   59.8%   no overlap at all -- right topic, wrong instance
    16   18.4%   we have most of it, missing a piece
    16   18.4%   we have a fragment
     2    2.3%   gold inside our answer, judge said no anyway
```

Sixty percent is not a coverage failure. `What novel is Evan reading that he
finds gripping` returns a different novel he also mentioned. `Maria's puppy she
got two weeks before August 11` returns a different pet. We retrieve the right
subject and choose the wrong one of several candidates, because the word that
discriminates -- *gripping*, *two weeks before* -- was dropped by extraction, so
every competing fact scores identically.

More facts cannot fix that; it adds distractors. Ranking can, and the mechanism
already exists.

### source_grounding was sized as a tiebreak and it should not have been

`source_grounding` scores each fact by how well **the turn behind it** matches
the question -- which is exactly where the dropped discriminating word still
lives. It shipped at 0.003, deliberately "sized like adjacent ranks". Swept at
h7 / 18 facts / overlap 0.8, against the 178 questions the paid run actually
loses:

```
  grounding     losses     wins    ctx tokens
    0.003       43.3%     87.6%       1024
    0.01        43.8%     88.2%       1019
    0.03        44.4%     88.2%        945     <- knee
    0.1         43.3%     87.1%        906
    0.3         43.3%     87.1%        893
```

**0.03 is worth 137 tokens.** It reaches as well as 20 facts at 1,082t and
better on the win side, for 945. It renders nothing itself; the saving is
indirect, because grounded ranking picks shorter and more on-topic turns.
Past 0.03 the weight starts overriding relevance and both slices fall.

### Facts beat turns at equal cost, which reverses the earlier reading

Every earlier sweep moved facts only *downward* from 20. A printed fact costs
about 30 tokens and a hydrated turn about 55, so the untested corner was more
facts and fewer turns. At hydrate 5, grounding 0.003, overlap 0.8:

```
  facts     losses     wins    ctx tokens
    24      44.4%     87.1%      1078
    28      45.5%     87.1%      1166
    32      47.8%     88.8%      1219    +19/-6 and +14/-0
```

Set against the turn-heavy arm at the same price:

```
  h10, 20 facts, overlap 0.8   44.9%   88.2%   1231
  h5,  32 facts, overlap 0.8   47.8%   88.8%   1219
```

**Three points of loss reach, for nothing, at the same width** -- and 32 facts
at 5 turns displaces no win at all. The turn-heavy configuration that two paid
runs were built around was the wrong shape.

### We under-answer, and it is the whole deficit

Two independent measurements, neither of them a screen.

```
  mean answer length, words        gold    ours    RAG
  all 839                           4.8     4.2     4.4
  questions we win                  4.5     4.1     4.2
  questions RAG wins and we lose    5.7     4.5     5.7

  of those 87   RAG's answer contains every gold content word    56%
                ours does                                          3%
```

On the questions we lose, the gold is 5.7 words, **RAG matches it exactly and
we come in 1.2 words short.** Where we win, the lengths match. This is not a
retrieval fact and no retrieval config addresses it.

### Conversion, tested on 80 questions instead of 839

`_ab_conversion.py`, 8 September. Census of the 80 losses whose gold is already
in the context, plus 90 of the 661 wins, four prompts, control re-generated in
the same session.

```
                on the 80 fixable        on 90 of the wins
  control            22.5%                    96.7%
  assemble           31.2%   +12/-5           94.4%   +1/-3
  qualifier          25.0%    +9/-7           93.3%   +0/-3
  discriminate       21.2%    +6/-7           96.7%   +1/-1
```

**`assemble` is +7 on a census of exactly the questions a prompt can fix.** The
harness projected it negative, and that projection should not be believed: it
multiplies the control net by 661/90 = 7.3, and 1 fix against 3 breaks has
p=0.625. The 95% band on the scaled control term is roughly -53 to +20 answers.
**A 90-question control slice cannot resolve a breakage rate small enough to
matter, and no affordable slice can.** The only way to price the win side is to
run the win side.

`qualifier` and `discriminate` are dropped: both spend their target gain on
control breakage and neither beats `assemble` on the slice that matters.

### What `assemble` actually does, which is not what it was written to do

Reading its 12 fixes and 5 breaks changed the design. It was written for
compound answers, but it recovered `The Great Gatsby` and `Apex Legends` --
single-value answers where the win came from surveying every candidate instead
of taking the first plausible fact. And **all five breaks are short golds**:
`when she was 10`, `at a festival`, `action and sci-fi`, where surveying turned
into padding.

So the behaviour to keep is the survey and the behaviour to stop is the padding,
and they separate cleanly in the instruction. `assemble2` says read everything
before answering, give every part when the question has parts, and give exactly
one value and stop when it does not. That is the version in the full run.

### Retrieval is settled at 945 tokens, and the arithmetic says stop tuning it

The whole landscape at or under the ceiling, screened against the 178 questions
the 839-question paid run actually loses. All have scan 8, overlap 0.8, deduped
headers, pool matched to rerank:

```
  hydrate : facts : grounding      tokens    losses     wins
        7 :    18 :      0.003       1024     43.3%    87.6%   (what was paid for)
        7 :    18 :      0.03         945     44.4%    88.2%   <- ship this
        3 :    28 :      0.03         995     42.1%    85.4%
        2 :    32 :      0.03        1003     44.4%    84.8%
        3 :    32 :      0.03        1047     45.5%    86.0%
        4 :    32 :      0.03        1094     47.8%    87.1%
        5 :    32 :      0.003       1219     47.8%    88.8%
       10 :    20 :      0.8ov       1231     44.9%    88.2%
```

**h7 / 18 facts / grounding 0.03 dominates everything at or under 1,000**:
equal or better on both slices than anything cheaper, and cheaper than anything
that reaches better. The facts-heavy corner is genuinely better *above* the
ceiling and genuinely worse below it, because buying loss reach means giving up
win reach and the win side is 661 questions against 178.

**And the arithmetic says this is the end of retrieval as a lever.** We are 15
answers short of RAG on 839. Moving loss reach from 44.4% to 47.8% reaches six
more questions, which convert at the historical third: two answers, for 149
extra tokens. Every remaining retrieval config trades one slice against the
other for single-digit counts. There is no config that closes 15.

### Where the 15 answers have to come from

Of the 178 questions the paid config loses, **77 already have the gold in the
rendered context.** Those are the only questions a prompt can fix, and 77 is
enough to matter: converting a fifth of them closes the gap.

That is also what the deficit shape says. 60% of the questions RAG answers and
we do not are "right topic, wrong instance" -- the model choosing badly among
candidates it can already see. Not a reach failure by construction.

**`_ab_conversion.py` exists to test prompt levers without paying category
prices.** A prompt change can only fix a question whose gold is already in the
context and can only break a question we currently win, so it generates exactly
those two slices: a census of the 77, and a 90-question sample of the wins. Four
prompts across both slices is about 1,340 calls instead of the ~6,700 a
full-category A/B of four prompts would cost.

The control prompt is **re-generated rather than read from `od_full.json`**. The
run-to-run drift is 1.2 points; comparing a variant against answers generated an
hour earlier would measure the provider, not the prompt.

Three variants, each aimed at a measured bucket rather than invented:
`qualifier` (name the discriminating word and use it), `discriminate`
(qualifier, plus permission to check the dialogue for it -- explicitly not the
"prefer SOURCE" rule that measured -2.0), and `assemble` (aimed at the 37% of
the deficit that is a fragment or a near-miss of a multi-part answer).

### The candidate, and the screening is finished

Everything below converges on one configuration. It is the best reach profile
measured at or near the ceiling, and nothing further in the render is free:

```
  rerank 40 : context_facts_top_k 18 : hydrate_pool 40 : grounding 0.003
  hydrate_scan 8, hydrate 7 turns, dedupe_source_headers on, fact_overlap_max 0.8

  1,026t     losses 49.2%  (+19/-5)     wins 85.0%  (+4/-0)
```

The fallback, if 1,000 is treated as hard rather than nominal, is the same
thing at 6 turns: **967t, 47.5 / 85.0, +4/-0**. That is 1.7 points of loss reach
for 59 tokens, and it is strictly better than the 968t config that was actually
paid for, which reached 47.5 / **83.3**. Whatever else happens, the cheap arm
should not be re-run as it was.

For scale, the arm that scored 81.1-82.3% in both paid runs is **1,265t at
51.7 / 85.0**. The candidate gives up 2.5 points of loss reach, holds the win
side exactly, and costs 239 fewer tokens.

### Line-trimmed turns: dead, and worth writing down why

The one lever that could have broken the reach-versus-cost tradeoff. A whole
turn costs about 55 tokens; trimmed to its best-matching lines it should cost
far less, so ten trimmed turns might fit where seven whole ones do. At h8 with
20 facts, overlap 0.8, deduped:

```
  lines : whole      losses    wins     ctx tokens
      4 : 2          50.8%     85.0%       1144
      3 : 2          50.8%     85.0%       1132
      3 : 1          50.8%     85.0%       1131
      2 : 1          50.8%     85.0%       1130
```

Going from four lines per turn to two moves the render by **14 tokens** and
moves reach not at all. The reason is that the turns the question scan picks are
already short -- LoCoMo exchanges are a line or two -- so there is nothing to
trim. `hydrate_lines` is a real mechanism against long documents and a no-op
against dialogue. It is off, and the axis is closed.

That leaves turns-versus-tokens as a straight trade with no trick in it, which
is why the candidate above is a knee and not a waypoint.

### Pattern separation is a reach gain, not a token saving

`fact_overlap_max` was parked in an earlier session as "17 tokens, presence
flat". It was re-screened at h6 / 20 facts because 1,020t needed 20 tokens
shaving. It does not shave them:

```
  overlap    shipped baseline    40:20:40:0.003:8
  off              1056t         1020t   47.5 / 85.0  +17/-5  +4/-0
  0.8              1038t         1028t   48.3 / 85.0  +18/-5  +4/-0
  0.7              1024t         1032t   47.5 / 85.0  +18/-5  +4/-0
  0.6              1004t         1040t   47.5 / 85.0  +18/-5  +5/-0
```

The baseline gets cheaper and the capped config gets **dearer**, because overlap
runs before the print limit. Dropping a redundant fact does not shorten the
block; it promotes the next distinct fact into the freed slot. On an uncapped
render that is a saving. On a render capped at 20 facts it is a swap.

The swap is worth having on its own terms: **0.8 buys +0.8 points of loss reach
for +8 tokens with the win side untouched.** 0.7 and 0.6 cost more and buy less,
so 0.8 is the setting. But it is not the lever that gets h6 under 1,000, and
nothing else in the render is.

### Two levers killed for free before anything was spent

**Deeper retrieval does nothing.** Holding the render fixed at 968t and moving
only retrieval depth, which prints no tokens at all:

```
  rerank:facts:pool:scan      gold in ctx, losses    wins
  40:16:40:8                        48.3%             84.2%
  60:16:60:8                        47.5%             84.2%
  60:16:60:20                       45.8%             84.2%
  80:16:80:12                       45.8%             84.2%
  80:16:120:30                      45.0%             84.2%
```

Monotonically worse. A wider pool gives the question scan more turns to rank and
it ranks the wrong ones. **40 / 40 / 8 is already the optimum**, and there is no
free accuracy hiding in retrieval depth.

**Adaptive turn budgets have no signal to run on.** The ceiling is a *mean*, so
a question answered by its facts alone could fund a question that needs twelve
turns. That needs the two told apart before generation. The candidate signal was
`covered` -- the share of the question's content words already in the FACTS
block, which costs nothing because FACTS is rendered anyway:

```
                          n     mean covered
  gold in FACTS         ...        0.787
  gold in SOURCE only   ...        0.750
  gold in neither       ...        0.796
```

The group that needs turns is *between* the other two. At every threshold from
0.2 to 0.75 the precision sits on the 16.8% base rate. There is no separation,
so adaptive hydration would be a coin toss wearing a mechanism's clothes.
`scripts/benchmarks/_screen_adaptive_signal.py` exists to record that, and the
mechanism was not built.

### The ceiling on retrieval, and where RAG's open-domain lead actually lives

Every retrieval change in this file moves the same number a few points and
nobody had asked what the maximum was. `_screen_oracle_reach.py` asks it: for
each open-domain question arm B loses, is every content word of the gold
present in **any** turn of the raw transcript. No ranking, no budget. That is
the best any turn-retrieval method could ever do.

```
  Gold present in SOME turn:
    questions we get wrong       36 of 55      65.5%
    questions we get right      151 of 194     77.8%

  By gold length, on the questions we get wrong:
    1-2 content words            17 of 20      85.0%
    3-5 content words            14 of 26      53.8%
    6+ content words              5 of 9       55.6%
```

**19 of 55 golds are in no turn at all** -- "friend's advice", "Hoodies", "a
photo of a man standing on a rock". They are the dataset's paraphrase of the
dialogue, not a quotation of it. No retrieval change of any kind reaches those,
and the length split is the evidence for why: short golds are quotations and
score 85%, long ones are paraphrases and score barely half.

Then the cross-tabulation the whole session turned on. RAG reads the same
transcript we do, so a gold in no turn is out of its reach too:

```
  Of the questions WE get wrong, how does RAG do?
                                  n   RAG right
    gold is in some turn         36     21   58%
    gold is in no turn           19      6   32%
```

**RAG's lead is concentrated in the reachable half.** It wins 58% of the
questions where the words are there and 32% where they are not. So retrieval
work is the right lever -- and the ceiling above says exactly how much is left
in it.

Caveat that limits what this proves: the haystack is one merged store, so
`scope()` searches all 2,957 turns and a gold can be matched by a turn from
another transcript. That makes the 65.5% generous. A low number would still be
decisive; a high one proves less.

### Retrieval is now at three quarters of its ceiling, and that is most of it

`_screen_scan_rank.py` asks the follow-up: when the gold IS in a turn and we
still miss it, what rank does BM25 give that turn out of all 2,957?

```
    top 8                    18   50.0%
    top 24                    6   16.7%
    top 100                   6   16.7%
    deeper than 100           5   13.9%
    unranked (score 0)        1    2.8%
```

Half the reachable golds are already ranked inside the budget the scan spends.
The deep-ranked groups are largely the merged-store artefact -- reading them
shows "happy" matching a turn about something else -- so they are smaller than
they look. **BM25 is not the bottleneck.** A semantic turn index was the
obvious next build and this is the measurement that says not to bother.

Running the diagnosis at the config that follows from it, `--scan 8
--hydrate 10`:

```
                             n      conversion           reach
  all B losses              55        27  49.1%        28  50.9%
  date-qualified             8         3  37.5%         5  62.5%
  no date                   47        24  51.1%        23  48.9%
  type-demanding             7         4  57.1%         3  42.9%
```

Conversion-eligible went 19 -> 23 -> 27 of 55 as the scan and the hydration
budget widened, against the ceiling of 36. **Three quarters of what search can
ever deliver is now delivered.**

Read 27 as an upper bound, not a count. The detector fires when the gold's words
appear anywhere in the rendered context, and this config doubles the dialogue in
that context, so some of 19 -> 27 is a larger haystack rather than a better one.
The ceiling of 36 is unaffected by that.

### The config is cheaper than RAG in every category, and damages none

`40:20:40:0.003:8` at `--hydrate 10`, screened on all four:

```
  category      shipped   config   RAG    losses      wins
  multi-hop      1328     1261    1395    +0/-0      +2/-0
  single-hop     1346     1275    1478    +5/-3      +0/-1
  temporal       1307     1233    1425    +7/-3      +3/-3
  open-domain    1268     1268    1459   +15/-5      +4/-0
```

Both columns are at `--hydrate 10`; the shipped stack in production runs
`--hydrate 5` at 989 tokens. So this trades 989 -> ~1,259 mean, still under
RAG's tightest category (1,395, multi-hop) and well under its mean of 1,430.
The standing constraint is that we cannot cost more than RAG, and we do not.

Gross movement, never net: single-hop wins -1 is the only regression and
temporal churns 3 for 3.

### What the remaining 27 look like, and why more retrieval will not fix them

Reading all 27 conversion failures, the shape is not vagueness and not
truncation. It is **the wrong specific item**:

```
  gold  sunflowers                     ours  her four dogs
  gold  Witcher 3                      ours  childhood sketches
  gold  Coco                           ours  Shadow
  gold  tree pose                      ours  Dancer Pose (Natarajasana)
  gold  A fantasy novel by Rothfuss    ours  The Alchemist by Paulo Coelho
```

In eleven of the 27, arms A and B give the identical answer. Retrieval config
does not decide those questions at all.

This is what RAG is beating us at, and the mechanism is legible: RAG puts the
question-matched dialogue chunk in front of the model as its primary unit. We
put 20 extracted facts first and the prompt says "COPY THE WORDING FROM THE
FACTS", so the dialogue we hydrate is material the model is told to treat as
secondary. Extraction dropped the qualifier the question turns on, several
stored facts now match equally well, and the model picks the salient one.

The three conversion levers priced below are all still dead. The one NOT yet
priced is ordering and precedence between the two blocks, and it cannot be
screened for free -- it changes what the model says, not what it sees, so it
needs generation. That is the next paid decision, not a next free one.

### PAID RUN: under 1,000 tokens is reachable, but not at the same accuracy

250 open-domain questions, 7 September 2026. `C:/nmafc_ab/od_cheap.json`.

    A = hydrate 10, 20 facts, headers as stored   (the config that scored 82.3%)
    B = hydrate 7, 16 facts, deduped headers      (the 957t config off the sweep)

```
                    n        A        B    diff   fix   brk        p   A ctx   B ctx
  ALL             250    81.2%    79.2%    -2.0     6    11    0.332    1280     968
  B widest single question 1423t, mean 968t (under the 1000t ceiling)
```

Paired against RAG's saved answers on the same 249:

```
  A  h10/20f     81.1%   RAG 80.7%   diff +0.4   win 19 lose 18  p=1.000  1280t
  B  h7/16f+dd   79.1%   RAG 80.7%   diff -1.6   win 22 lose 26  p=0.665   968t
```

**Under 1,000 tokens costs about 2 points and puts us back behind RAG.** Both
differences are inside noise, so neither is a proven effect, but the point
estimate moved the wrong way on both measures and there is no reading of this
where the cheap config is the better one.

### THE MEASUREMENT FACT THAT LIMITS EVERY NUMBER ABOVE

**Arm A is byte-identical in both paid runs today and scored 82.3% then 81.1%
on the same 249 questions.** Same store, same settings, same prompt, same judge.
That is 1.2 points of run-to-run drift from generation and judging alone.

Every effect chased today is 1.6 to 2.6 points. The drift is 1.2. So single
250-question runs cannot resolve them, and any config picked on one run is
partly picked on noise. This is the reason to stop tuning open-domain by single
runs rather than a reason to run more of them:

    +1.6 (A vs RAG, run 1)      inside drift
    +0.4 (A vs RAG, run 2)      inside drift
    -1.6 (B vs RAG)             inside drift
    -2.0 (B vs A)               inside drift

The honest summary of both runs together is that A and RAG are indistinguishable
on open-domain and A costs 13% fewer tokens. That is a real and defensible
claim. "We beat RAG on open-domain" is not.

### PAID RUN: open-domain is ahead of RAG for the first time, and the SOURCE rule is dead

250 open-domain questions, both arms generated together, 7 September 2026.
`C:/nmafc_ab/od_source.json`, log at `od_source.log`.

    A = scan 8, hydrate 10, rerank 40, facts 20, pool 40, grounding 0.003, gated prompt
    B = A + SOURCE_RULE

```
                    n        A        B    diff   fix   brk        p
  ALL             250    82.4%    80.4%    -2.0     2     7     0.18
    typed          32    81.2%    81.2%    +0.0     0     0     1.00
    untyped       218    82.6%    80.3%    -2.3     2     7     0.18
```

**The SOURCE rule loses, 7 broken against 2 fixed.** Not significant at p=0.18,
but the direction is unambiguous and there is nothing to buy by re-running it.
Dropped. `SOURCE_RULE` and `--source-rule` stay in the tree as a recorded
negative, wired off.

Why the screen pointed the wrong way, because this is the third time a free
screen has mispriced a prompt change: the screen measures **where the gold is**,
and the rule changes **which block the model trusts when both look plausible**.
A question whose gold is in SOURCE only is not the same set as a question the
model would newly answer from SOURCE. The +4.8 upside was real and irrelevant.

Paired against RAG's own saved answers on the same 249 questions:

```
  arm A   82.3%   RAG 80.7%   diff +1.6   win 22 lose 18  p=0.636   1282t vs 1468t
  arm B   80.3%   RAG 80.7%   diff -0.4   win 22 lose 23  p=1.000   1281t vs 1468t
```

**Read this precisely.** The point estimate is ahead by 1.6 and p=0.636, so it
is a statistical tie with a favourable point estimate, NOT a win. 22 questions
won against 18 lost is a coin flip. What can be claimed is that the deficit is
gone: open-domain was -2.7 against RAG and is now +1.6, a 4.3-point swing, and
it costs 1,282 tokens against RAG's 1,468.

The whole swing is retrieval -- `hydrate_scan` plus hydrate 10. That is more
than the ~+1 projected from the reach screens, so the reach-to-conversion rate
on newly reached golds was better than the historical third.

Still outstanding before "we beat RAG" is a sentence anyone can say: the other
three categories have only ever been generated under shipped retrieval. Free
reach screens say the new config damages none of them (multi-hop wins +2/-0,
temporal losses +7/-3, single-hop wins -1) but reach is not accuracy.

### The SOURCE lever was priced on a context that no longer exists

`_screen_source_block.py` killed "tell the model SOURCE exists" earlier this
session at +1.2 upside against -8.8 exposure. That was measured at hydrate 5
with no scan. The scan changes what SOURCE *contains* -- turns chosen by the
question instead of by fact rank -- so the price was stale for any configuration
that turns it on. The screen now takes `--scan`, re-run at `--scan 8
--hydrate 10`:

```
                    we won   we lost     total
  source only           22        12        34
  both                 111         8       119
  facts only            25         7        32
  neither               36        28        64

  Upside if a source-aware prompt converted every one: +4.8 points (12 questions)
  Downside if it lost every facts-only win:           -10.0 points (25 questions)
  SOURCE is 51% of the rendered context by characters.
```

**The prize went from 3 questions to 12 while the exposure went from 22 to 25.**
The ratio moved from 1:7.3 to 1:2.1, and the gap to RAG is 2.7 points against an
upside of 4.8. The earlier kill was correct on the earlier context; quoting it
against this one would be wrong.

`SOURCE_RULE` in `arms/base.py` is written as a **tiebreak, not a precedence**:
answer from the facts whenever a fact names what was asked, and go to SOURCE
only when no fact names the specific thing. The 25 exposed wins are exactly the
questions a fact already answers, so conditioning this way leaves them alone.
Worded as "prefer SOURCE" it is a bad bet on its own numbers.

Wired to `_ab_open_domain.py --source-rule`, arm B only.

### Hydration could never reach a turn retrieval missed. Now it can

The store's own schema comment states the design: turn text is "reachable only
through a fact that already won a slot". That is a real hole and it is where
most of the remaining open-domain gap lives.

A turn whose facts all rank below the cut cannot be hydrated at any
`hydrate_pool` setting, because the pool is drawn from the retrieved list and
nothing in the retrieved list nominates it. If the extractor generalised the
answer's noun away, the fact carrying it never competes, and the turn holding it
is unreachable no matter how well it answers the question.

`hydrate_scan` (default 0, off) adds the top N turns of the whole conversation
by BM25 of the question, ranked alongside the fact-derived pool. **It takes no
retrieval slot and prints no extra turn**: the number hydrated is still
`hydrate_top_k`, the turn fact rank was surest about is still reserved, and a
turn scoring zero is never taken. It answers the schema comment's objection --
turns competing with facts for the retrieval budget -- by not doing that. It
competes for the hydration budget, which was already being spent.

Free screen, 120 open-domain losses and 120 wins, gold-in-rendered-context:

```
  rerank:facts:pool:ground:scan   losses            wins              ctx
  shipped                      45  37.5%  +0/-0   98  81.7%  +0/-0    995t
  40:20:40:0.003:0             52  43.3%  +10/-3  98  81.7%  +1/-1    981t
  40:20:40:0.003:3             55  45.8%  +15/-5  100 83.3%  +2/-0    964t
  40:20:40:0.003:8             55  45.8%  +15/-5  102 85.0%  +4/-0    963t
  40:20:40:0.003:20            55  45.8%  +15/-5  102 85.0%  +4/-0    963t
  20:none:0:0:20               55  45.8%  +14/-4  101 84.2%  +4/-1    972t
```

Three things to take from that table.

The knee is at or below 8, so 8 is the setting. Above it nothing changes and the
BM25 pass gets longer.

**It is the first mechanism here that adds golds to the questions we already win
without displacing any**: +4/-0 on the wins, where every previous retrieval
change traded one for one. And it is cheaper -- 963 tokens against the shipped
arm's 995 -- because the turns the question picks are shorter than the turns
fact rank picked.

The last row is the uncomfortable one. Scan alone, on **shipped** retrieval with
no `rerank_top_k=40`, no `context_facts_top_k`, no `hydrate_pool` and no
`source_grounding`, gets 45.8% and 84.2% at 972 tokens. It reproduces almost the
entire stack that took three mechanisms and a paid run to establish. The stack is
still marginally better and cheaper and it is already bought, so it stays -- but
if anything here ever needs simplifying, that is the row to start from.

### Three conversion levers priced for free, all three bad bets

The paid stack left 55 of 249 open-domain questions wrong. Splitting them by
whether the gold reached the rendered context (`_diagnose_open_domain.py`): 19
conversion, 36 reach. Each candidate fix was then priced against its own risk
before spending anything, and none of them survived.

**Tell the model the SOURCE block exists.** No prompt in the repo mentions it.
The system prompt says "Answer the question using these facts" and the rules say
"COPY THE WORDING FROM THE FACTS", while 37% of the tokens are verbatim dialogue
the model is never told to use and one rule points away from. That reads like a
bug, and it is not: rendering the two blocks separately and asking where each
gold is (`_screen_source_block.py`) gives **+1.2 points of upside against -8.8 of
downside**. Only 3 lost golds are in SOURCE and not in FACTS; 22 wins are in
FACTS and not in SOURCE. The facts block is doing the work. Do not reopen.

**A completeness rule.** Golds are often multi-part and our answers often stop
early -- "Savana, Sleep" answered as "Savana". Counting how many losing answers
are a proper subset of the gold's content words (`_screen_partial_answers.py`):
5 of 55, **+2.0 upside against -9.6**, because 24 wins already carry extra words
the judge tolerated. Do not reopen.

**A specificity rule.** The golds name particular things where we name classes:
"Hoodies" answered as "clothing". Counting golds naming something ours does not:
22 of 55 losses -- and **62 of 194 wins**. The same base rate on both sides, so
it does not discriminate and a rule built on it would be firing on a third of
the questions we already get right. Reading the examples explains why: most are
not vaguer answers but different competing facts. "Photography" against our
"fixing cars" is not a specificity failure.

That is the value of screening. Three plausible prompt changes, three sets of
numbers, no credit spent, and the one mechanism that did survive is free of
tokens rather than costing them.

### The whole-benchmark picture, and why it reframes what is left to do

Computed free from the saved `haystack.json`, 1,535 paired questions, no
generation. This is the number the project should be quoted at, and it had not
been looked at as a whole before.

```
category         n   ours    RAG   diff  win lose         p
multi-hop       96   44.8   45.8   -1.0    9   10         1
open-domain    839   74.3   80.6   -6.3   63  116  9.12e-05
single-hop     279   45.2   43.4   +1.8   50   45     0.682
temporal       321   64.8   46.1  +18.7   93   33  8.46e-08
ALL           1535   65.1   64.4   +0.7  215  204     0.625

context     ours 976-999t in every category, mean 994
            RAG  1394-1477t,                mean 1451
```

The live configuration already sits **under 1,000 tokens in every category** and
ties RAG overall at 31% less context, with a large temporal win. Multi-hop's
-1.0 is 9 against 10 and is noise. The 1,300t candidate buys about 2.5 points of
open-domain, breaks sub-1k, and still loses open-domain by 3.2, so under
"win every round and never cost more" it is a bad trade and should not ship.

Do we beat Mem0, Zep, Letta or A-Mem? **Unknown, and not currently claimable.**
There is no such code in this repository. The arms are `raw_llm`,
`stateful_nodecay`, `neuromorphic`, `neuromorphic_tuned` and `rag`, and the
README describes `stateful` as a *simulation* of the MemGPT/Zep philosophy, not
an implementation of it. Comparing against a number from someone else's paper
would be comparing across judge, prompt and baseline at once. The defensible
claim is parity accuracy at 31% less context plus a large temporal win.

### The deficit is one question shape, and it is not a retrieval failure

Free, from `haystack.json`, via `_probe_answer_type.py`. Split the 1,535 paired
questions by whether the wh-phrase names the kind of answer wanted -- "which
state", "what console", "how old":

```
                        n    ours     RAG    diff
type-demanding        238    56.7    61.8    -5.0
everything else     1,297    66.7    64.9    +1.8
```

**The whole of the deficit sits in questions that name a type, and both arms
fail them together.** That is the signature of a shared prompt rule, not of
retrieval: `SHORT_ANSWER_RULES` says "COPY THE WORDING FROM THE FACTS ... a
synonym scores as a miss", so asked for a state both arms return the town the
facts name, asked for a country both return the city. Multi-hop is lost entirely
in this class -- 32.1 against 35.7 on its 28 typed questions, and dead level
50.0 against 50.0 on the other 68.

The four-cell split says the same thing from the other side. Multi-hop's -1.0 is
9 wins against 10 losses; of the 10, two are judge noise (identical wording, split
verdicts) and three are answer-breadth. The headline cannot move. The 43
both-wrong is the only cell with room in it.

### Answer-type conditioning: +5.1 on the class, p=0.015, still under budget

`src/nmafc/integration/answer_type.py`, 38 unit tests. Split in two on purpose:
`TYPE_RULE` (79t, static, sits in the system prompt) plus a per-question tag of
about 5t, `Asked for: state.` A single 51t per-question directive was tried
first and pushed three categories over 1,000 -- the split is what keeps it
affordable.

Paired A/B, 472 questions, one session, one store, RAG deliberately not re-run:

```
                        n       A       B    diff  fix  brk       p   B ctx
typed (as run)        272    57.7    61.8    +4.0   19    8   0.052    968t
untyped controls      200    75.0    73.5    -1.5    3    6   0.508   1020t
```

The controls are not optional: `TYPE_RULE` sits in the prompt on every question,
so testing only the questions the tag fires on would measure the gain and be
structurally blind to the cost. The -1.5 is 3 against 6 and is noise, but it is
the number to watch if this is ever widened.

**Reading the regressions found a detector bug worth 18 questions.** The first
version deleted words it did not like from the matched phrase and kept the rest,
so "What did Maria make for her home" tagged as `Asked for: Maria make.` -- a
verb clause ending on a noun the list knew. 18 of the 272 were this, all of them
`make`, and they were actively harmful: 0 fixed, 2 broken, -11.1 on that slice.
Fixed by rejecting rather than trimming (a stop word may lead the phrase, never
follow content) and by dropping `make` from the noun list. Re-scored free from
the saved answers:

```
                        n     fix   brk       p
typed (fixed)         254      19     6   0.015
untyped controls      200       3     6   0.508
projected overall     472      22    12   0.121

typed, by category    multi-hop    31   35.5 -> 45.2   +9.7   952t
                      temporal     41   48.8 -> 56.1   +7.3   975t
                      single-hop   56   37.5 -> 42.9   +5.4   978t
                      open-domain 126   71.4 -> 74.6   +3.2   956t
```

Significant on the class it targets, every category under 1,000 tokens, and the
largest single gain is multi-hop -- the category the whole session was aimed at.
Not yet significant overall (p=0.121), because four fifths of questions do not
name a type and the tag correctly stays silent on them.

**Caveat, stated because it is easy to miss:** the fixed-detector numbers above
are a re-score of answers generated with the buggy detector. The 18 false fires
are treated as reverting to arm A, which is what an untagged question is, but
`TYPE_RULE` was still in their prompt. The typed slice is untouched by this and
is a real measurement; the projected overall is an estimate.

### The caveat is now closed, for 36 calls instead of 1,880

`_ab_answer_type_repair.py`. The repair is cheap because of an asymmetry worth
remembering for any future prompt A/B: **arm A never sees the tag.** Its prompt
is the shipped one, unchanged by the detector fix, so every A answer and every A
verdict in the file is still exactly what A produces. And arm B only changes
where the tag changes. So the honest repair is to recompute the tag for the 472
questions already asked, regenerate B for the ones that differ, and report the
whole set relabelled -- a paired A/B over a fixed question set, both arms valid.

18 differed, all of them losing a bogus tag, none gaining one. That fact alone
was free and settled half the caveat: the 254 typed questions were generated
from byte-identical prompts either way, so the +5.1 on them was never an
estimate. Regenerating the 18 as untyped controls closed the rest:

```
                        n       A       B    diff  fix  brk       p   B ctx
typed                 254    55.9    61.0    +5.1   19    6  0.0146    964t
untyped controls      218    75.7    73.9    -1.8    3    7   0.344   1020t
ALL                   472    65.0    66.9    +1.9   22   13   0.175    989t
```

### Gating the rule removes the control cost rather than measuring it

The -1.8 on the controls is caused by `TYPE_RULE` sitting in the system prompt
on questions that name no type. That is avoidable, and the fix is free: the
system prompt is built per call, so include the rule only when the detector
fires. On an untyped question a gated arm then sends the shipped prompt
**byte-identical** -- same system prompt, same question, same retrieved context,
the same call arm A makes. There is no difference left to cost anything.

`answer_type.gate` (12 unit tests, the important one being that an untyped
question returns two empty strings and not so much as a newline). Two
consequences follow:

* The gated arm's score can be read off the existing file -- untyped questions
  take arm A's answer because that *is* the gated arm's answer. Not an estimate;
  the same generated text under a different label. `_score_answer_type.py`.
* The untyped half of any future run here is not worth buying. Two identical
  prompts differ only by the provider's sampling. `_ab_answer_type.py` now
  copies A across instead, roughly halving what a rerun costs.

```
gated vs shipped     n   shipped   gated  diff  fix  brk       p    ctx
ALL                472     65.0    67.8   +2.8   19    6  0.0146   989t
  multi-hop         39     38.5    46.2   +7.7    4    1
  temporal          84     59.5    63.1   +3.6    4    1
  single-hop        91     48.4    51.6   +3.3    5    2
  open-domain      258     76.7    78.3   +1.6    6    2
```

Every flip is a typed question, by construction. +2.8 overall at p=0.0146,
against +1.9 at p=0.175 ungated -- the same mechanism, with its one measured
cost designed out.

### Projected onto the full benchmark: three categories won, open-domain is the last

The 472 is a stratified sample (every typed question plus 200 controls), so its
totals are not benchmark numbers. The gated arm changes typed questions only, so
the projection is the measured per-category typed rate applied to that category's
typed count in the full 1,535, and nothing else moves:

```
                typed   net/typed   ours now   projected     RAG     gap
multi-hop          31       +9.7       44.8       47.9      45.8    +2.1
single-hop         56       +5.4       45.2       46.2      43.4    +2.9
temporal           41       +7.3       64.8       65.7      46.1   +19.6
open-domain       126       +3.2       74.3       74.7      80.6    -5.8
ALL               254       +5.1       65.1       66.0      64.4    +1.6
```

at 989 tokens of context against RAG's 1,430.

**Multi-hop is the goal this session was set and the projection wins it**, 47.9
against 45.8, having been 44.8 against 45.8. Three of four categories ahead.

**Read the significance honestly.** Only the 254-question aggregate is
significant (p=0.0146). Multi-hop's +9.7 is 4 fixed against 1 broken on 31
questions, p=0.375 -- directional, not established. The aggregate is solid, the
split between categories is not, and multi-hop's +2.1 lead therefore rests on
five flips. Anyone quoting the category table should quote this next to it.

**Open-domain is the only remaining loss and it is 839 of 1,535 questions**, so
it sets the overall. Answer-type conditioning moves it +3.2 on its typed 126,
which is real and nowhere near the -5.8 gap. This is exactly the target of the
standing instruction, and `source_grounding` below was screened against precisely
these losses.

### The question never chose which turns to read, and that is the open-domain bug

`hydrate_pool` in `DecayConfig`, `QueryRouter._turns_by_question`, 10 new tests
in `tests/unit/test_grounding.py`. Off by default (`0`).

The largest bucket in the whole benchmark is the 47 of 116 open-domain losses
where **the answer was already in the prompt** and the model chose a competing
value. The questions carry a qualifier extraction dropped -- `gripping`, `two
weeks before 11 August`, `next month` -- so several stored facts match equally
and nothing in the prompt separates them. The dropped word is still in the turn.

Hydration is the only mechanism that puts turn text in the prompt, and as
shipped it chose turns by fact rank alone: `records[:hydrate_top_k]`, hydrate
whatever turns those facts came from. The question was used to choose *lines
within* a turn and never *which turns*. The one signal that knows what is being
asked was spent on the smaller of the two decisions.

This is also, and not incidentally, the principle every framework that beats RAG
on open-domain follows. Parent-document retrieval, HippoRAG's "the graph routes,
the passages answer", cross-encoder reranking over raw text: all three say the
structured layer is an index, not the answer. Ours had become the answer. The
brain framing says the same thing more strongly -- hippocampal index theory has
the index in the hippocampus and the content in cortex, with recall as cue-driven
reinstatement. Retrieving the index and then reading the index is the part that
was never brain-like; hydrating on the cue is pattern completion.

**The first attempt measured nothing, and the reason is the useful part.** At the
shipped `rerank_top_k=20` and about six facts per exchange, the retrieved pool is
three to six distinct turns, while fact rank already hydrates three or four. There
was nothing to choose from. The three settings have to move together:

```
  rerank_top_k       20 -> 40    retrieval depth, renders nothing, free
  context_facts_top_k   -> 20    caps printed facts, so the widening stays free
  hydrate_pool          -> 40    choose turns by question match, same count
```

`_source_turns` is passed the full retrieved list rather than the printed one, so
the wider pool reaches hydration while the FACTS block stays the size it was
measured at. `want` is passed into `_turns_by_question` rather than decided
there: it is the number of turns fact rank would have taken, so this changes
*which* turns are read and never *how many*. On a 1,000-token budget there is no
free slot; every turn chosen is a turn dropped.

Screened free over 240 open-domain questions, gold-answer-in-context, both
halves (`_screen_hydrate_pool.py`):

```
  rerank:facts:pool:grounding    losses           wins            ctx
  shipped                     45  37.5%  +0/-0   98  81.7%  +0/-0   995t
  20:none:0:0.003             50  41.7%  +6/-1   98  81.7%  +0/-0   984t
  40:20:40:0                  51  42.5%  +9/-3   98  81.7%  +1/-1   994t
  40:20:40:0.003              52  43.3%  +10/-3  98  81.7%  +1/-1   981t
```

**+9/-3 on the losses, +1/-1 on the wins, and a token cheaper.** Pool 80 is
identical to pool 40, so 40 is the setting. The instrument reports gross movement
rather than a net, because 98 before and 98 after is either nothing moving or
three golds displaced by three others, and only one of those is safe to ship.

**The two retrieval mechanisms overlap rather than add.** 52 together against 51
for hydration alone: they read the same BM25 signal, so it is largely the same
questions being fixed by two different decisions. Worth knowing before either is
sold as independent. Both together is still the configuration to run, because it
reaches the most and costs the least (981t), at the same risk profile as
hydration alone. Grounding alone is the conservative choice: +6/-1 and nothing
displaced on the wins at all.

The mechanism that should *not* overlap is answer-type conditioning, because it
attacks the other half of the problem. These two change whether the answer
reaches the prompt; conditioning changes whether an answer already in the prompt
is the one chosen. The 47-of-116 bucket is a conversion failure by definition --
the text was there and lost anyway -- so reach could never have closed it alone.
`_ab_open_domain.py` runs all three together for that reason.

Two checks on the free detector, since the standing rule is to read examples
before believing a rate. It counts a hit when every content word of the gold
appears in the rendered context, and on the shipped arm it says 37.5%, against
the paid judge's 47 of 116 (40.5%) for the same quantity. Close enough to be
measuring the same thing. And the questions it says were fixed read correctly by
hand -- "What book did Caroline recommend to Melanie?" where the stored fact is
"the book you recommended a while ago" and the title is in the turn.

Not generation-tested. Gold in the context is not gold in the answer.

### Source grounding: built, screened, redesigned, wired off

`src/nmafc/integration/grounding.py`, 32 unit tests, `source_grounding` in
`DecayConfig` defaulting to `0.0`.

The gap it targets: RRF fuses vector rank, keyword rank and graph distance, so
**after retrieval the question's own wording never participates in ranking at
all.** 47 of the 116 open-domain losses already had the gold answer in the
prompt; the model read it and chose a competing value, because extraction wrote
"Evan is reading The Great Gatsby" and dropped *gripping*, leaving two stored
reading facts nothing can tell apart. The word survives in `turn_text`, which
`cold.py` deliberately keeps out of `memory_fts` so raw turns never compete with
facts for the retrieval budget. This honours that: turns are reached only
through facts that already won a slot, and only the order changes.

BM25 of the question against each candidate's source turn, IDF computed within
the candidate pool -- no index, no corpus statistics, self-normalising per
question, zero context tokens.

**The first design was wrong and the free screen said so before anything was
paid for.** It went in as a sixth list fused by RRF. But RRF weights every list
alike, so "the question's words appear in the turn this fact came from" counted
for as much as topping vector search:

```
                   in pool      top 20       top 6     up   down    churn/60
  off             40   58%     40   58%     30   43%      0      0
  as a list, 2    42   61%     42   61%     33   48%     11     11        27
  as a list, 3    43   62%     43   62%     33   48%     14     13        45
  as a list, 5    43   62%     43   62%     33   48%     17     13        54
  as a list, 8    43   62%     43   62%     34   49%     20     10        58
```

Three more of 69 losses reached, against the top facts of **54 of 60 questions
already answered correctly** being reshuffled. Up and down are near equal, which
is what a coin looks like. Bad trade, and it does not match the diagnosis: the
problem was two facts the ranking *cannot separate*, which is a tie, and a tie
wants a tie-break.

Rebuilt as an additive boost beside `recency_boost` and `weight_signal`:
`score += source_grounding * share`, where share is the fact's source-turn BM25
scaled so the best turn in the pool is 1.0. The scale is what makes this
different, not the idea: two facts adjacent in one fused list differ by about
0.0003, and a whole extra list is worth about 0.016. The list version was the
second number. Weights around the first move a fact past its near-copy and leave
a fact that won on merit alone -- pinned by
`test_a_fact_that_won_on_merit_is_not_displaced`.

Screened the same way, and it dominates the list form on every column at once:

```
                   in pool      top 20       top 6     up   down    churn/60
  off             40   58%     40   58%     30   43%      0      0
  w=0.0001        40   58%     40   58%     30   43%      3      2           5
  w=0.0003        40   58%     40   58%     30   43%      3      2           6
  w=0.001         42   61%     42   61%     31   45%     12      5          15
  w=0.003         44   64%     44   64%     33   48%     18      5          25
  w=0.01          45   65%     45   65%     34   49%     21      7          45
  as a list, 5    43   62%     43   62%     33   48%     17     13          54
```

**w=0.003 is the candidate.** Against the list form it reaches one more record,
turns 17-up/13-down into 18-up/5-down -- a ratio instead of a coin -- and halves
the churn. Below 0.001 the boost is too small to move anything; above 0.01 it
stops being a tie-break. Nothing here is generation-tested: it says the
answering record is in the prompt more often, not that the model uses it.

Two things worth keeping from the instrument:

* **Off-vs-off is 0% churn.** Run that control before believing any churn
  number; it is what says retrieval is reproducible across handles and the
  churn is the mechanism rather than the harness. `--weights 0`.
* **A gain-only screen would have been structurally blind to all of this.**
  Every headline number above is unchanged or better; the whole verdict is in
  the churn column, which only exists because the questions we already win were
  measured too.

`_screen_grounding.py` reuses the `ranking_targets.json` the judge already paid
for, so screening any ranking change is now free: 69 answerable open-domain
losses, rank measured with and without, plus churn on 60 questions we win.

### Both cheap open-domain levers are now closed

Source-aware answer prompt: 400 questions, +0.5, p=0.774. More facts at
`12:24:1:1:4`: 300 questions, +0.3, p=1, and it was at the 1,395 ceiling. Do not
reopen either without a new mechanism behind it.

### Pattern separation, from the dentate gyrus

`fact_overlap_max` in `DecayConfig`, `separate_facts` in `query_router`. Drops a
fact whose content words are contained in a better-ranked fact's above the
threshold. Containment rather than Jaccard, because the redundant case is a short
summary wholly restated inside a longer one and Jaccard scores that pair as
barely similar. Applied to the rendered facts only -- `_source_turns` still reads
the unseparated list, so hydration is byte-identical and context can only shrink.

Cost screen, 839 open-domain, no generation, `C:/nmafc_ab/separation_cost.json`:

```
setting                   mean t   p90 t   max t   vs base   shrunk
20:5                        1001    1178    1523        +0        0
20:5:0:0:0:90                992    1170    1523       -10      293
20:5:0:0:0:80                983    1159    1523       -19      461
20:5:0:0:0:70                968    1146    1500       -34      600
20:5:0:0:0:60                944    1125    1461       -58      744
```

Presence is the other half and is also free. Run over the same 839 with the
quantity veto in place -> `C:/nmafc_ab/separation_presence.json`:

```
        setting  present    share  vs shipped   context
           20:5      657    78.3%       +0.0     1001t
  20:5:0:0:0:80      657    78.3%       +0.0      984t
  20:5:0:0:0:90      655    78.1%       -0.2      992t
  20:5:0:0:0:70      655    78.1%       -0.2      972t
  20:5:0:0:0:60      653    77.8%       -0.5      952t
```

**The verdict is: it works, and it is not worth switching on.** Overlap 80 holds
presence exactly while saving 17 tokens, which is the flat-line-at-a-lower-price
result that was wanted. But 17 tokens is 1.7% of a context already 457 under
RAG's binding ceiling, and the spread across the whole column is four questions
in 839 -- overlap 90 losing two while the more aggressive overlap 80 loses none
is noise, not a curve. Buying 1.7% by adding a live mechanism whose risk presence
cannot see (it asks only whether the gold string is in the prompt, not whether a
fact the model was reasoning from was removed) is a bad trade at this margin.

**So `fact_overlap_max` stays None, which is the default and is what every
result to date was measured under.** It is built, tested and screened; 80 is the
setting if a future configuration ever needs the headroom. Do not spend a
generation A/B on 17 tokens.

The quantity veto is separate and is unconditional, because it is a correctness
fix rather than a tuning knob. Its cost is visible in the table: the same
settings screened at 983/968/944 before the veto and 984/972/952 after, so it
keeps a few facts that were previously being merged away, at a price of one to
eight tokens.

### Numbers were invisible to every lexical decision in the system

`cue_words` tokenises on `[a-z0-9']+` and keeps what is longer than two
characters, so every number below 100 disappears: `42`, `7`, `$5`, `12%`. Only
`2023` survived, by being four digits long. That is an accident of a length
filter written for stopwords, and it had teeth.

The sharp case is a correctness bug in the pattern separation directly above.
"He paid 42 dollars" and "He paid 47 dollars" both reduce to `{paid, dollars}`,
so containment is 1.0 and one of the two amounts is **deleted as a restatement of
the other**, with the survivor decided by whichever ranked higher. Two facts that
state different numbers are two facts.

`src/nmafc/integration/quantities.py` pulls numbers out as their own class of
evidence: normalised so `1,200`, `1200` and `1200.0` are one quantity and `two`
and `2` are one quantity, with a small written-number and ordinal table. Two
uses, neither of which adds a token to any context:

- **`separate_facts` may not merge two facts that state different numbers.** The
  guarantee is a property, tested as one: separation cannot lose a quantity.
- **`best_lines` ranks a shared quantity above any number of shared words.** A
  number is the most specific thing a line can share with a question -- several
  lines of a turn discuss the rent, one says what it was. Inactive while
  `hydrate_lines` is None, which is the shipped setting, so this is groundwork.

`unverified_quantities(fact, source)` is an audit and nothing calls it during
retrieval: numbers a summary asserts that its own source turn never said. It
reports rather than repairs, because "a dozen" written back as `12` is a
legitimate rewrite, and acting automatically on a signal with a known
false-positive mode is what the supersession detector was measured doing when it
killed correct facts at 17.9%.

Note this changes the separation cost numbers in the table above slightly upward,
since the veto keeps facts that were previously merged. The presence run now in
flight was restarted against the current code for that reason.

### The quantity audit, run over all 8,276 facts. Both premises were wrong

`scripts/benchmarks/_audit_quantities.py`, pure SQLite, read-only, immutable
mode, no generation and no API key. It compares every number in a stored fact
against the verbatim turn it was extracted from, in both directions. It costs
nothing to re-run.

**Direction one, numbers the fact asserts that the source never said: 493 facts,
19.5% of the 2,522 facts carrying a number.** That rate looks alarming and is
not an error rate. Reading the sample settles it in about a minute -- almost
every hit is the extractor doing *arithmetic*, and doing it correctly:

```
"I painted that lake sunrise last year"            -> "last year (2022)"
"5 years already!"          (June 2023 session)    -> "married around June 2018"
"my 18th birthday ten years ago"                   -> "Caroline is 28 years old"
"known these friends for 4 years, since I moved"   -> "moved around June 2019"
```

That is derived content, and **it is the mechanism behind the +18.7 temporal
win.** The extractor resolves a relative expression into an absolute date at
write time, so the stored fact answers "when" outright. This is a feature that
had never been named, and the audit is the thing that named it. It also marks
the risk precisely: derived arithmetic can be wrong and nothing checks it. Every
derivation spot-checked here was correct.

**Direction two, counts said aloud and dropped from the summary: 55 facts, 0.7%
of the store, and most of those are artefacts of the detector.** This is the
direction the extraction-recall item was actually complaining about, and it does
not survive contact with the whole store. The genuine class is small -- "four is
enough for now" becoming "her dogs" -- and it is the only one that is real.

**Direction three, facts naming a person who is absent from their own source
turn: 63 facts, 0.8%** -- and reading them tells the same story as direction one.

```
turn: "Jon: Thanks for the kind words!"
fact: "Jon thanked Gina for her kind words on 23 March 2023."

turn: "John: Yup, we rock as a team! Glad to have you."
fact: "John told Maria he's glad to have her."
```

Every one inspected is *correct*. The extractor resolves an implicit addressee to
a named person using who it knows is in the conversation, which is what makes a
fact usable standing alone: "John told Maria" can be retrieved and read, "he told
her" cannot. State the detector's limit plainly -- many stored turns carry only
one speaker's line, so "absent from the turn" frequently just means the partner
did not happen to speak in that turn. The 0.8% is an upper bound on a problem
that appears not to exist.

**So do not rewrite the extraction prompt for counts or for attribution.** The
next-steps item was built on three remembered examples. Over 8,276 facts, two of
its three named causes measure under one percent and dissolve on inspection into
the extractor doing something right, and 32 minutes of re-ingestion to chase that
is not a good trade. **Destinations dropped is the one cause still standing, and
it is still unmeasured** -- it has no free detector yet, because "Chicago exists
only as somebody's suggestion, never as a place visited" is a claim about
modality rather than about a token being present, and no lexical check sees it.

**Getting to that number took three passes, and the two corrections are the
lesson.** The first run reported 925 dropped counts, 11.2% of the store, and
every one of the twelve worst nouns was a month: `14 August` in the session
header, which the ingester wrote and nobody said. Stripping the header and
excluding month names took it to 79. The second run still counted `the 2016
Finals` as a count of "16 finals" and `the second one` as a count of "2 one";
a lookbehind refusing a match inside a longer number, and a function-word filter,
took it to 55. **A detector reporting 11.2% and a detector reporting 0.7% were
the same idea; the difference was entirely in what counted as a hit.** Read the
examples before believing the rate, every time.

### Code memory: an exact symbol index, no embeddings, no tokens to build

`src/nmafc/code/` and `src/nmafc/schemas/code.py`. Python `ast` gives the
definitions, their spans, their content hashes and the references between them.
Three things this does that the conversational retriever cannot, all because the
relation is written down rather than inferred: traversal is exact, reverse edges
("what calls this") exist and are free, and staleness is decided by hash rather
than estimated by a decay constant -- which removes the need for the supersession
detector in this domain entirely.

`render_symbols` is graded hydration transplanted: hop 0 in full, hop 1 as
signature and docstring line, hop 2+ as a pointer, under a hard token cap.
`suggest_budget` is metamemory -- a feeling-of-knowing run before recall decides
how hard to look, and the signal is free because the index already knows how many
symbols matched and how long each is.

**Two bugs found here are worth keeping, because both produced confident wrong
output that looked like a win:**

1. `ast.walk` does not prune subtrees, so a `ClassDef` absorbed every reference
   its methods made and `callers("helper")` answered with both the calling method
   and the class it happened to sit in. Fixed with an explicit descent that skips
   nested definitions. This is the everything-to-everything edge, and the
   docstring had warned against it.
2. `suggest_budget` sized by symbol *count*, so a one-seed question looked cheap
   and got a third of the ceiling -- but "explain format_context" is one seed and
   seventy-five lines, so the renderer **dropped the definition that was asked
   about** and reported a 98% saving on a context that answered nothing. Fixed by
   sizing on spans (measured: 10.53 tokens per line mean, 10.39 median, 13.14
   p90 over 399 symbols) and by letting a seed degrade -- body, then signature
   plus "body omitted, N lines at this span", then a bare pointer -- rather than
   ever being dropped.

Measured against the fair baseline, which is grep-and-read-the-file rather than
stuffing the repository in. `python -u scripts/benchmarks/_measure_code_context.py --root src`:

```
indexed 540 symbols in 66 files, 216 ms, 0 API calls, 0 tokens
8 real questions, every one reporting seed kept: yes
total 5217t against 54163t, 90% less than reading the files
a question naming no symbol returns 0 symbols, by design
staleness check over 66 files: 13 ms, 0 changed
```

The saving moved from 91% to 90% when bindings were added, and that is the right
direction. More real edges means more genuinely relevant neighbours in the block,
not padding -- a constant a function reads is now reachable from it.

Run against a repository it was not developed on -- the Python 3.12 standard
library, 862 files of code nobody here wrote, with decorators, metaclasses and
generated modules:

```
36,193 symbols in 862 files, 10.0 s
malformed spans: 0
"what does JSONDecoder do"  97 symbols, block 1165t, within budget, seed kept
"explain HTTPConnection"    46 symbols, block  778t, within budget, seed kept
"who calls scanstring"      no symbol
```

**That last line was a real limitation, and it is now fixed.** `scanstring` is a
module-level assignment, `scanstring = c_scanstring or py_scanstring`, and the
index recorded only `def` and `class`, so aliases, constants and configured
loggers were all invisible. `SymbolKind.BINDING` covers them: 2,823 more symbols
in the standard library, 39,020 total, still no malformed spans, and the question
now answers with the exact line plus the caller that uses it.

**Two things had to be right for a binding to be worth indexing, and both were
wrong on the first attempt:**

1. A constant is referenced by *name*, never by call, and `_refs` recorded only
   calls and attributes. A binding therefore had no reverse edges at all, which
   is most of the point of indexing it. Bare names are now recorded, but only
   when the module actually binds or imports that name -- recording every bare
   name would make a local variable called `config` in forty functions join all
   forty to a module named config, which is the everything-to-everything edge
   twice burned already.
2. Reads only. Counting the assignment target made `DEFAULT_TIMEOUT = 30` a user
   of itself and put every binding at the head of its own caller list.

Attribute and subscript targets are deliberately not definitions: `config.timeout
= 99` mutates something defined elsewhere, and recording it would put a second
entry in the index claiming to be that thing.

**There is no accuracy column and there must not be one until there is a code
benchmark.** The mechanism is exact -- spans and hashes either match the file or
they do not -- but the claim that this shape of context answers coding questions
better than a competent grep-and-read agent is untested, and scoring it against
questions written by the person who built it would not be a test.

Tests: 451 passed, 3 skipped. Nothing committed, nothing pushed.

## State as of 6 September 2026

### The decisive A/B has run. Cheaper, faster, and still behind on open-domain

Both arms generated in the same session, all 839 open-domain questions, the
candidate context at `6:28:1:1:4` with `top_k` 20 and `rerank_top_k` 40.
-> `C:/nmafc_ab/candidate_open.json`

```
  ours  76.8%    1302 tokens    median 2456 ms
  RAG   80.0%    1460 tokens    median 2767 ms
  -3.2, p=0.0426, ours faster p=2.45e-05
```

Against the saved shipped run on the same questions. RAG itself moved only -0.7
between the two sessions (p=0.307), so for once the cross-session read is
licensed:

```
  ours   candidate 76.7%   shipped 74.2%   +2.5, net +21 questions, p=0.031
  margin over RAG    -3.2  <-  -6.4
  context            1303t <-  1000t       RAG 1460t
```

And temporal at the same configuration, 120 questions, both arms today: 67.5%
against RAG's 43.3%, +24.2, p=0.000154, at 1,262 tokens against RAG's 1,431.
-> `C:/nmafc_ab/candidate_temporal.json`

So, against the standing goal of winning every round: cost won, latency won,
temporal won, **open-domain still lost**. The gap halved and did not close.

#### The remaining gap is 26 questions the six-fact cut broke

Of the 116 old open-domain losses, 40 are now right. But 26 questions that were
right are now wrong, and 26 questions is 3.1 points against a 3.2-point gap.

```
  old loss bucket          fixed
  gold was in the prompt   17 / 40    42%
  stored, never reached    12 / 29    41%
  never extracted          11 / 47    23%
```

Two things in that table are worth keeping. The middle row says the ranking
screen was right: `top_k` 10 -> 20 converts. The bottom row is the surprise --
eleven questions whose gold the judge said was never extracted came back correct
anyway, because the raw source turns carry what extraction dropped. Hydration
partly covers for the write-time failure, which nothing predicted and which
lowers the value of the extraction work below where it was ranked.

The 26 breakages are the price of rendering six facts instead of twenty, and
they are affordable to undo. Comparing per category, RAG spends 1,460 on
open-domain against our 1,303 and 1,425 on temporal against our 1,262, so there
are roughly 160 tokens of headroom and a fact costs about 31.

Priced across every question of every category, against RAG's own per-category
spend. Multi-hop is the binding constraint at 1,395, not open-domain at 1,459:

```
   setting        multi-hop   single-hop   temporal   open-dom
   RAG spends        1395        1478         1425       1459
   6:28:1:1:4        1285        1320         1276       1303   <- measured above
   10:24:1:1:4       1298        1343         1311       1334
   14:20:1:1:4       1323        1367         1339       1353
   8:28:1:1:4        1342        1381         1336       1365
   12:24:1:1:4       1360        1404         1371       1395   <- richest legal
   10:28:1:1:4       1402        1442         1397       1427   over on multi-hop
   12:28:1:1:4       1463        1503         1457       1488   over on three
```

#### `format_context` was never given the question

Fixed, in `wrapper.py` and every benchmark arm and harness. `hydrate_lines`
keeps the N best-matching lines of each turn, matched against the cue -- and
`format_context(records)` without a query built that cue from the retrieved
facts alone. `_sweep_hydration.py` was the only caller in the repo that passed
the question, which means every presence number was measured under a
configuration the answer path could not produce.

Measured before claiming it mattered: over 60 current open-domain losses the
prompt changes for 7 of them, 12%, and the gold text's presence moves on none.
The retrieved facts already carry most of the question's words, so the union
adds little. A real bug, worth fixing, and **not** the explanation for the -3.2.
Do not re-run the A/B for it alone.

#### The answer prompt has never been told what `<SOURCE>` is

`ANSWER_SYSTEM_PROMPT` opens "You have a knowledge graph of facts, shown in
`<FACTS>` tags. Answer the question using these facts", never mentions `<SOURCE>`
at all, and later instructs "COPY THE WORDING FROM THE FACTS". At the shipped
configuration that was reasonable. At the candidate it is not: six facts is about
186 tokens and 28 source turns about 1,100, so `<SOURCE>` is roughly 85% of the
prompt and the model is directed at the other 15% -- the summaries whose dropped
qualifiers are the diagnosed cause of these very losses.

`_ab_prompt.py` already exists and already writes the fix, and WHAT-CHANGED
records it at **-1.04**, in the "did not pay" table. That measurement was taken
at `--hydrate 5` on the single-conversation stores with all twenty facts
rendered, where `<SOURCE>` was a minor block and pointing at it cost a point.
The configuration it was refuted under no longer exists, so it was re-run at
`6:28:1:1:4` on the merged store, both prompts in one session, 400 open-domain
questions:

```
  n=400   A 77.0%   B 77.5%   +0.5 points
  fixed 7  broke 5  p=0.774
```

**Null, and decisively so.** Twelve discordant pairs in four hundred questions
means the two prompts agree almost everywhere: the model was already reading
`<SOURCE>` without being told what it was, so the instruction was never the
bottleneck. Neither the -1.04 nor a win survives; the effect is nothing.
`_ab_prompt.py` now takes the merged store, the context flags, `--categories`,
`--sample` and `--compact` (which was hardcoded off and did not match the arm
being measured). **Close this line.** -> `C:/nmafc_ab/prompt_candidate.json`

The useful consequence is negative but real: the remaining open-domain gap is
about what is *in* the context, not about how the context is labelled or read.
That leaves the fact count and extraction, and nothing cheaper.

### The most important result of the session: it is not a retrieval problem

`_probe_stored_vs_reached.py`, one embedding and one judging call per question,
over the 101. **Read the instrument note below before quoting these numbers**;
two earlier versions of this script produced different splits and both were
wrong.

```
  gold stored somewhere       83   82.2%     <- the ceiling
  gold reached the prompt     81   80.2%     <- what retrieval achieves

  of the 20 misses:
    stored but not surfaced    2   10.0%   ACCESS failure
    never extracted at all    18   90.0%   WRITE failure

  the decisive stale-only block: 19, of which gold was stored in only 2
```

**Retrieval is operating two points below its own ceiling.** It surfaces almost
everything that exists to be surfaced. That single fact explains every negative
result on this page: decay, demote, expand, `weight_signal`, `recency_boost` all
reorder the retrieved set, and there is almost nothing left in the store for a
better ordering to reach. The 14.9% stale-only block is not one problem but two,
and nearly all of it -- 17 of 19 -- has no gold fact in the store at all.

This is the engram question, asked of our own system, and the answer comes out
opposite to the mouse case. Ryan et al. induced amnesia, found the trace intact
and access broken, and recovered the memory by stimulating the tagged neurons
directly. Our failures are the other way round: they are traces that were never
laid down. The system is not amnesic, it is inattentive at encoding.

Consequence for where effort goes: **the remaining points are at write time, in
extraction recall, not at query time.** 18 of 101 current answers were never
extracted, which is 17.8% of the whole test lost before any query is asked. That
is larger than every retrieval-side margin measured in this project.

Hand-checked, and the failures are specific rather than diffuse. "How many cats
does Melanie have?" answers two; the store holds "Melanie has a dog and a cat",
so the count was dropped. "What city did John visit?" answers Chicago; the only
fact mentioning Chicago is Tim listing it as a suggestion, never John going. The
store holds eleven Prius facts and not one connects the Prius to the Jasper road
trip. Extraction is recording topics and losing quantities, destinations and
which of two people did the thing.

#### Why the instrument, not the store, was the hard part

Three versions, two of them wrong, and the failures are worth keeping because
each would have been believed:

1. **Whole-word matching** (`present` from `_test_updates.py`) reported 12 of 21
   misses never extracted. It cannot handle inflection, so a stored "John is
   trying to get his car fixed" did not match a gold "tried to get it fixed". It
   also passes any short answer automatically: gold "two" is whole-word present
   in every store containing the word "two" anywhere. Wrong in both directions
   at once.
2. **Judge with an overlap shortlist.** The judge cannot be shown 8,000 facts,
   so candidates were ranked by content-word overlap with the gold. For a
   one-word gold every fact containing that word ties at 1.0 and the top eight
   are arbitrary. This made the judge look harsher than whole-word matching and
   reported a 73.7% ceiling under a 78.9% reach -- impossible, since every
   retrieved record came out of the store.
3. **Judge with an embedding shortlist**, the current one. Candidates are the
   nearest store vectors to the question and answer together, with everything
   retrieval surfaced seeded in front so "stored" is a real superset of
   "reached". Validated against seven cases inspected by hand: the five wording
   variants come out stored, the two genuine absences come out not stored.

The general lesson, and it applies to every probe in this repo: **an instrument
that fails on both sides of a comparison does not cancel out.** It moved the
access-versus-write split from 57/43 to 10/90.

### The haystack exists, and it cost nothing

`_build_haystack.py` merges the ten finished stores into one. Every store
already tags its records `agent_id='default'` and `conversation_id='default'`,
and `HotStorage` filters searches on exactly those two fields, so records from
ten stores dropped into one table are visible to every query with no code
change. Vectors are already computed, facts already extracted: no embedding
call, no LLM call, no ingestion.

```
merged: 2,957 turns   cold 8,276 facts   hot 3,807 facts   no id collisions
offsets: 26:1-211  30:212-397  41:398-733  42:734-1047  43:1048-1392
         44:1393-1730  47:1731-2074  48:2075-2415  49:2416-2672  50:2673-2957
```

Turn numbers are the only thing that could not be copied as they were, since
each store counts from 1 and hydration would otherwise fetch whichever turn 50
it found first. `memory_event_log.id` is renumbered on the way in and
`memory_fts` is rebuilt whole, because it is an external-content FTS5 index
keyed on those ids.

This matters because everything measured up to now was measured on easy ground:
one conversation per store, so a question only ever competed against ~830 facts
from its own transcript. The +0.5 overall, the +17.1 temporal and the +7.9 on
mylocomoeval were all taken in that setting. `_ab_haystack.py` runs both arms
against the merged pile, with RAG indexing all ten transcripts so neither arm
answers from a smaller world. Prediction stated in advance: both arms score
worse; the question is which degrades faster.

### The haystack has been run, and it refutes the prediction

1,535 paired questions, both arms, one merged store, one RAG index over all ten
transcripts. 1,524 of them were also answered on the single-conversation stores,
so the two runs are paired question by question.

```
                      one conversation          haystack (10x)
  category      n     ours    RAG   marg      ours    RAG   marg    shift
  OVERALL    1524    66.1%  65.4%   +0.7     65.2%  64.2%   +0.9     +0.2
  single-hop  279    44.4%  47.3%   -2.9     45.2%  43.4%   +1.8     +4.7
  temporal    321    64.8%  47.7%  +17.1     64.8%  46.1%  +18.7     +1.6
  multi-hop    96    49.0%  46.9%   +2.1     44.8%  45.8%   -1.0     -3.1
  open-domain 828    76.0%  80.6%   -4.6     74.4%  80.4%   -6.0     -1.4
```

**Neither arm degraded.** McNemar on each arm across the two runs: ours lost 55
and gained 40, p=0.151; RAG lost 68 and gained 50, p=0.117. Ten times the pile
and both arms are where they were, inside the 1.5-2 point cross-run drift.

The honest reading is that this is a null result about the *test*, not a win.
The prediction was that both arms would suffer, and it was wrong, and the reason
is visible once looked for: LoCoMo's ten transcripts have disjoint casts. A
question naming Melanie cannot be confused by conv-50's people because none of
them are called Melanie, so nine tenths of the haystack is not competing for the
answer at all. The density scan in `_sweep_capacity.py` says the same thing from
the other side -- merging ten stores produced almost exactly the crowding of the
ten stores summed, so the merge added volume without adding contested space.

So this does not establish that the margin survives a real haystack. It
establishes that it survives *this* one, and that this one is easier than
intended. A genuine LongMemEval haystack, where distractor sessions are chosen
to be confusable, is still unrun and still costs roughly 250 hours of ingestion.
Do not quote the haystack numbers as evidence of scale-robustness without the
disjoint-casts caveat attached.

What does hold up: temporal is +18.7 with p=8.5e-08 on 321 questions, our
largest and most durable win. Open-domain is -6.0 with p=9.1e-05 and is now the
only category costing us the overall result.

### Open-domain is the whole benchmark, and the failure is not what we assumed

Open-domain is 839 of 1,535 scored questions. We lose 116 and win 63. At parity
the overall goes 65.1% to 68.6% against RAG's 64.4%, turning a +0.7 no test can
distinguish from noise into a +4.2. Every other category is rounding by
comparison, and this is now the first place to spend.

`_sweep_hydration.py` formats the same retrieved pool at several hydration
depths and asks a judge whether the gold answer is present, over the 116 we
lose:

```
  hydrate   gold present    vs shipped   context
        0     43   37.1%        -3.4       619t
        5     47   40.5%        +0.0      1003t
       12     53   45.7%        +5.2      1463t
       20     60   51.7%       +11.2      1870t
```

**47 of the 116 already had the answer in the prompt at shipped settings.** The
model read it and answered wrong. That is the largest single bucket and it is
not retrieval, not forgetting and not extraction recall.

Reading the predictions says what it is. Only 10 of 116 say we do not know. The
rest are confident, correctly shaped and wrong in one consistent way: they name
a real fact about the right subject that answers a different question.

```
  What novel is Evan reading that he finds gripping?
    gold The Great Gatsby       ours The Last Devil to Die
  What is the name of Maria's puppy she got two weeks before 11 Aug 2023?
    gold Coco                   ours Shadow
  What city did Tim suggest to John for the team trip next month?
    gold Edinburgh, Scotland    ours Nashville, Austin, Miami, Chicago...
```

Each question carries a qualifier -- *gripping*, *two weeks before 11 August*,
*next month* -- and extraction stripped it out, so several stored facts match
the question equally well and nothing in the prompt distinguishes them. RAG
strips nothing, so its chunk still contains the word that decides the answer.
This is the same defect the stored-versus-reached probe found, seen from the
other side: there it dropped counts and destinations, here it drops the
qualifiers that disambiguate.

Sizing the 116:

```
  47  (41%)  answer present, competing value chosen        disambiguation
  13  (11%)  reachable only by hydrating deeper, 5 -> 20   free, config only
  56  (48%)  not present at any depth                      not yet split
```

The 56 have now been split, by pointing `_probe_stored_vs_reached.py` at them
with `--store haystack` so it reads the same pile the run retrieved from. 55 of
them survived the filter:

```
  gold stored somewhere    22   40.0%
  gold reached the prompt   1    1.8%
  of 54 misses:  21 stored but not surfaced (39%),  33 never stored (61%)
```

**This revises the headline claim, and the revision matters.** Retrieval sits 2
questions under its ceiling on mylocomoeval, and that was written up as "no rule
that removes, reorders or inserts records can win more than 2 questions". That
holds for mylocomoeval and does not generalise. On open-domain there are 21
questions whose answer is in the store and never surfaced -- real retrieval
headroom, in the category that decides the benchmark. The two tests disagree
because they ask different things: mylocomoeval asks which of two known values
is current, where the pool almost always contains at least one of them, while
open-domain asks about an arbitrary detail that may sit anywhere in 12,000
facts.

So the selection faults flagged in `_sweep_precision.py` are back on the table
for open-domain specifically, and the settled-negative list applies to
mylocomoeval only.

The mechanism for the first two buckets already exists and is turned down:
`_source_turns` in `query_router.py` fetches the verbatim turns behind the
best-ranked facts, deduped by turn, and `hydrate_top_k` is 5. We also spend 994
context tokens against RAG's 1,451, so there is room to read more source and
stay the cheaper arm -- though not at depth 20, which costs 1,870.

#### Deep hydration works. First significant win over RAG.

`_ab_hydration.py`, all 839 open-domain questions, `hydrate_top_k` 5 -> 20,
nothing else changed and no re-ingestion. Drift gate passed first: the shipped
configuration scored 78.0% when saved and 79.0% today on the same 100 questions,
+1.0, same verdict on 95, so the saved shipped and RAG answers are a usable
baseline.

```
  open-domain, 839 questions
    shipped  hydrate 5     74.3%     999 tokens
    deep     hydrate 20    78.2%    1847 tokens
    RAG                    80.6%    1459 tokens

    deep vs shipped   won  50  lost  17  net +33  p=6.74e-05  better
    deep vs RAG       won  69  lost  89  net -20  p=0.13      no difference
    shipped vs RAG    won  63  lost 116  net -53  p=9.12e-05  worse
```

The category stops costing us. It does not become a win -- we neutralise RAG
there rather than beating it -- and that is enough, because open-domain was the
only category losing.

Carried to the whole benchmark, deep hydration on open-domain and shipped
settings everywhere else:

```
  overall          65.1%  ->  67.3%     RAG 64.4%
  margin            +0.7  ->   +2.9
  vs RAG        won 221  lost 177  p=0.031   <- first significant win
  context          994t  ->  1457t     RAG 1451t
```

**At the same context cost as RAG.** Open-domain is 55% of the test, so
hydrating only that category lands the average within 6 tokens of RAG's.

Three things this does NOT yet show, and none of them should be glossed:

1. **It is a per-category configuration, and categories are not known at
   inference.** The 67.3% assumes hydrate 20 on open-domain and 5 elsewhere.
   Shipping it means either hydrating everything at 20, which is untested on the
   other 696 questions, or routing on a predicted question type, which is a new
   component with its own error rate.
2. **The risk sits on temporal.** It is our largest win, +18.7, and it has never
   been run at depth 20. If deeper source text dilutes the dated structure that
   wins it, the global change could cost more than it earns. Test temporal
   before anything else; 321 questions, one arm, about 0.9M tokens.
3. **Depth 20 alone costs 1,847 tokens** against RAG's 1,459. The average only
   works out because the other categories stay shallow. Trimming the fact budget
   to pay for source text is untested.

Point 3 is now tested, and it is the good news below.

#### The fact list was almost entirely duplication

Deep hydration was more accurate and more expensive, which is not a win. The way
out is that the prompt pays twice for the same information: every fact is a
summary of a turn, and at a budget of 20 the facts cost about 619 tokens and the
turns behind them about 1,250.

The router could not express the trade. `_source_turns` hydrates
`records[:hydrate_top_k]` out of the same list `format_context` renders, so
cutting facts cut the source with them and the two knobs behaved as one. New
config field `context_facts_top_k` separates them: render the top N facts,
hydrate out of the full retrieved list. Unset renders all of them, which is what
every earlier measurement did, so no saved number moves.

`_sweep_hydration.py`, rebuilt to sweep `facts:lines`, on the same 116
open-domain losses:

```
   facts  lines  present    share  vs shipped   context
       4     20       61    52.6%      +12.1     1375t
      20     20       60    51.7%      +11.2     1865t   OVER BUDGET
       8     20       60    51.7%      +11.2     1499t   OVER BUDGET
       6     20       60    51.7%      +11.2     1434t
       0     20       57    49.1%       +8.6     1246t
       0     16       49    42.2%       +1.7     1066t
      20      5       47    40.5%       +0.0     1002t   <- shipped
```

Six facts and twenty source turns holds the gold exactly as often as twenty and
twenty, for 431 fewer tokens, and lands under RAG's 1,459. Fourteen of the
twenty facts were costing tokens and adding no reachable information.

They may have been costing accuracy too. The 47 losses whose answer was already
in the prompt were answered with a real fact about the right subject that
answers a different question, which is what being distracted by a competing
summary looks like. Deleting fourteen summaries while keeping every source turn
removes the thing that was picked instead of the source. That is a generation
effect and presence cannot see it, so it is being A/B'd:
`_ab_hydration.py --deep-facts 6`, 839 questions, deep arm only, drift-gated.

#### The drift gate fired, and it was right to

That A/B did not run. The gate re-ran the shipped configuration on the same 100
questions and got 81.0% against the 78.0% saved in `haystack.json` -- +3.0,
where the tolerance is 2.0 and the effect being looked for is about 2.4. The
same check on the same seed read +1.0 in the session that produced the depth-20
result, so the provider has moved about two points since. Nothing was spent on
the deep arm.

Two consequences, and the second is the expensive one:

1. **The depth-20 result stands as reported.** It was generated in a session
   measured at +1.0 against its own baseline, inside tolerance, which is exactly
   what the gate is for.
2. **Every generation A/B from now on has to run both arms in one session.**
   `haystack.json`'s RAG column is no longer a fair opponent, and no correction
   factor makes it one -- RAG answers with the same model, so it drifts too, and
   comparing today's arm to a saved opponent flatters us by about three points.
   For open-domain that is roughly 3.0M tokens for the pair.

Costing is also settled, and it caps hydration for good. `_sweep_hydration.py
--cost-only` prices settings with no generation and no judge:

```
   facts  lines   context  headroom vs RAG's 1459
       6     20     1435t       +24t
       0     24     1415t       +44t
       4     20     1375t       +84t
       0     20     1246t      +213t
      20      5     1000t      +459t   <- shipped
       2     24     1482t       -23t   OVER
       0     28     1535t       -76t   OVER
```

Nothing past twenty source turns is affordable. So hydration's ceiling is the
depth-20 presence figure, and the depth-20 accuracy figure was a tie with RAG.
Hydration alone cannot produce a win.

#### It is also a saving, if a saving is what is wanted

Asked for a configuration under 1,000 tokens rather than under RAG's 1,459, and
the trade turns out not to be a trade at all at that end of the range:

```
   facts  lines  present    share   context
       0     14       51    44.0%     958t
       4     12       50    43.1%     972t
       6     10       47    40.5%     918t
      20      5       45    38.8%     999t   <- shipped
```

No facts and fourteen source turns is cheaper than shipped and holds the answer
for about six more of the 116. The twenty-fact list costs 630 tokens and adds
nothing the turns it summarises do not already say.

Read that +5.2 as +4 to +6. Shipped scored 47 in the sweep above and 45 in this
one, same configuration and same questions, so the judge's run-to-run noise is
about two questions and the gain is a few times that rather than ten times it.

The whole ladder, cheapest useful configuration to most accurate affordable one:

```
   facts  lines   present   context
       6     20     51.7%    1432t    best under RAG's 1459
       0     20     49.1%    1242t
       0     14     44.0%     958t    best under 1000
      20      5     38.8%     999t    shipped
```

Which end to ship is a budget decision, not a measurement one. Under 1,000 is
better than today on both axes and still loses to RAG; 6:20 is the only shape
with a plausible route to beating it. `top_k` 10 -> 20 applies to either, costs
no tokens, and should be in whichever is chosen.

### Recall was episodic, and it did not need to be

Asked how the brain retains things so cheaply, and the answer turned out to be
an actual defect in the code rather than an analogy.

Hydration restored **whole turns**. A turn is a session header plus both
speakers' lines -- an episode -- and the answer is usually one clause of one
line. At roughly 50 tokens a turn and twenty turns hydrated, that is about 1,250
tokens spent to deliver a few dozen useful ones. The session header
`[Session - 1:56 pm on 8 May, 2023]` is identical on every turn of a session, so
twenty turns can carry the same twelve tokens a dozen times.

The hippocampus does not replay episodes to recall a detail. The index points at
a cortical pattern and reinstates the part the cue calls for. Three fields, none
of which costs an API call:

- `hydrate_lines` -- keep the N speaker lines of a turn that best match the cue.
  The cue is the question **plus the facts extracted from that same turn**. The
  facts alone would be circular: extraction is what dropped the qualifier, so a
  fact-only cue reliably keeps the line the fact already covers and discards the
  one that completes it.
- `dedupe_source_headers` -- print a repeated header once per run of turns.
  Lossless, because turns are emitted in order.
- `hydrate_full_turns` -- the best-ranked turns come back whole, the tail comes
  back as single lines. Graded reinstatement: strong cue, rich detail; weak cue,
  fragment. Counted by rank, printed by date.

Priced first, on 40 questions, nothing judged:

```
   setting     context    what it is
      6:20       1393t    twenty whole turns
  6:20:2:1       1330t    two lines each -- most turns only have two
  6:20:1:1        910t    one line each, headers deduped
  0:20:1:1        721t
```

A 35% cut from line trimming, and header dedup is 51 tokens of it. Two lines per
turn saves almost nothing, which is the tell that a turn is two lines: the
choice is one line or the whole thing.

Judged on all 116:

```
   setting     present   share    context
      6:20          61   52.6%     1433t
  6:40:1:1          57   49.1%     1169t
  6:32:1:1          56   48.3%     1123t
  6:20:1:1          51   44.0%      910t
  0:20:1:1          48   41.4%      721t
      20:5          45   38.8%      999t   <- shipped
```

Two results worth keeping. **`6:20:1:1` is cheaper than shipped and holds six
more answers** -- 910 tokens against 999, 51 against 45. And forty sparse turns
nearly match twenty dense ones for 264 fewer tokens, which says the win is reach
rather than detail.

But sparse does not beat dense at the top: 57 against 61. Hence
`hydrate_full_turns`, and it is the best thing found this session:

```
     setting     present   share    context
  6:40:1:1:8          63   54.3%     1427t
        6:20          61   52.6%     1432t
  6:40:1:1:5          60   51.7%     1333t
  0:40:1:1:5          53   45.7%     1144t
        20:5          46   39.7%      998t   <- shipped
```

Forty turns of reach with the best eight returned whole costs the same as twenty
whole turns and covers twice the ground. Read 63 against 61 as "at least as
good" rather than better -- see the noise note below -- and the point still
holds, because the reach is free at that price.

One measurement caveat that applies to this whole section. Shipped has now been
judged four times on the same 116 questions and scored 47, 45, 45 and 46, so the
judge's own noise is about two questions. Differences of two are not
differences. The +14.7 is seven times that and the +5.2 is three times it; the
+1.7 rows are not results.

#### Every price in the two tables above is understated

`_sweep_hydration.py` selects the questions RAG answered and we did not, because
that is what presence is a question about. It is the wrong sample for money. A
question is lost partly *because* its source block is wide, so the losses are the
worst case and not the bill. `--select all` was added to price against every
question of a category instead, and it moves the numbers by more than the
headroom they were being judged against:

```
   setting        temporal (321)   open-domain (839)   RAG's tightest is 1395
   6:40:1:1:8          1597t             1659t          over on both
   6:40:1:1:5          1500t             1561t          over on both
   6:32:1:1:5          1388t             1424t          over on open-domain
   0:40:1:1:5          1318t             1373t          under
   6:28:1:1:4          1275t             1303t          under, ~100t spare
   6:24:1:1:4          1190t             1211t          under
   6:20:1:1             964t              974t          under, and under shipped
```

RAG's own per-category spend, from `haystack.json`: open-domain 1,459, temporal
1,425, single-hop 1,478, multi-hop 1,395, all 1,451. Winning every round means
clearing the tightest of those, so 1,395 is the ceiling, not 1,459.

`6:40:1:1:8` -- the configuration the presence sweep argued for -- costs 1,659
tokens on open-domain, 264 over. It was never shippable and the loss-only
pricing hid that. A four-question smoke run at the candidate showed 1,564 against
RAG's 1,499 and the full 321 confirmed it, which is what the cheap safety step
was for.

#### The configuration this argues for

```
  top_k                  10 -> 20      free, 6 more answering records reached
  rerank_top_k           20 -> 40      retrieval only, no tokens rendered
  context_facts_top_k   all ->  6
  hydrate_top_k           5 -> 28
  hydrate_lines        whole ->  1
  hydrate_full_turns       0 ->  4
  dedupe_source_headers false -> true

  context     open-domain   999t -> 1303t   RAG 1459t
                 temporal   977t -> 1275t   RAG 1425t
```

`0:40:1:1:5` is 70 tokens cheaper and held more of the gold on the losses, but
zero facts means no FACTS block at all, and the dated structure that wins
temporal by +18.7 is rendered in that block. Spending the one known risk to save
70 tokens is a bad trade.

And the frugal variant, for a budget under 1,000 rather than under RAG's:
`6:20:1:1` at 974 tokens -- cheaper than what ships today and holding six more
answers than it.

Neither is an accuracy claim. Presence says the answer is in front of the model,
not that it is chosen, and the only sound accuracy figure remains deep hydration
at 78.2% against RAG's 80.6%, p = 0.13, from before the drift. Stop tuning
presence: the remaining differences are inside the noise floor, and the only
experiment that can settle whether this beats RAG is the paired A/B with both
arms in one session.

#### That A/B now has a harness

`_ab_haystack.py` already answers both arms back to back in the same worker, so
drift cannot enter it and no gate is needed; what it could not do was express a
new context setting. It now takes `--facts`, `--hydrate-lines`,
`--hydrate-full-turns`, `--dedupe-headers`, plus `--categories` and `--sample`
for running one category or a fair random subset. Every one of them defaults to
what produced `haystack.json`, so running it bare still reproduces the baseline.
`_ab_vs_rag.py` took the same flags for the single-conversation condition.

This is what `_ab_hydration.py` should have been. That script scores a candidate
against *saved* answers at half the price, and the saved baseline had drifted 3.0
points by the time it ran. Half price for an untrustworthy answer is not a
saving.

### Ranking: 29 answers are in the store and never retrieved

`_screen_ranking.py`, new, and the cheapest instrument built this session. The
trick is that the judge is always being asked the same thing -- is the answering
record in this pool -- and the answering record does not change when the
configuration does. So identify it once by text, then score every configuration
by looking for that text. One judge pass of about 116 calls, then arithmetic.

Stage 1, on the 116 open-domain losses:

```
  69/116 losses have an answering record in the store
  40 of those are in the shipped pool of 20
  29 are not retrieved at all
```

**29 questions is 3.5 points of open-domain accuracy, and the gap to RAG is
2.4.** This lever is large enough on its own and costs nothing at inference.

The picks were hand-checked and they are exact. They also read as three
different failures rather than one: "The Great Gatsby" sits at rank 3, was
rendered as a fact, and we answered with a different novel; "Edinburgh" sits at
rank 14, was rendered, and we answered with the earlier brainstorm list; "Coco"
is not retrieved at all. The first two are selection, the third is ranking.

Rank 14 is the case that ties the two threads together. At six facts Edinburgh
stops being a competing summary and becomes a hydrated source turn instead,
which is precisely the change `context_facts_top_k` was added to make.

Stage 2 sweeps configurations for free -- one embedding per question per
configuration, no generation:

```
   hot  cold  pool  hops   in pool   top 20   top 6
    10    20    20     2    40   58%   40  58%  30  43%   <- shipped
    20    20    20     2    46   67%   46  67%  31  45%
    40    20    20     2    46   67%   46  67%  31  45%
    10    40    20     2    40   58%   40  58%  29  42%
    10    80    20     2    38   55%   38  55%  29  42%
    20    40    20     2    43   62%   43  62%  29  42%
    10    20    20     3    40   58%   40  58%  30  43%
    10    20    20     1    41   59%   41  59%  30  43%
    10    20    40     2    49   71%   40  58%  30  43%
```

One knob moves: hot `top_k` 10 -> 20 puts six more answering records in the
pool, and the pool size does not change, so the context costs exactly the same.
Widening cold makes it worse. Hops still do nothing. A pool of 40 reaches 49
records but only 40 of them rank inside 20, so the extras are never hydrated and
buy nothing at a budget we can afford.

Refining around the one knob that moved:

```
   hot  cold  pool  hops   in pool
    10    20    20     2    40   58%   <- shipped
    15    20    20     2    45   65%
    20    10    20     2    45   65%
    20    20    20     2    46   67%   <- peak
    25    20    20     2    45   65%
    30    20    20     2    46   67%
    20    20    20     1    46   67%
    30    10    20     2    43   62%
```

20 is a peak rather than a trend: 15 and 25 both give 45 and 30 gives no more
than 20 does. That matters, because a monotone "more is better" curve usually
means the knob is a proxy for something else, and this one is not.

The unreached targets are worth reading as a set. They are dated open-domain
questions ("last weekend before April 10, 2023", "in November 2022") and
thematic ones ("what are Caroline's plans", "what motivates John's team"),
against a merged 12,083-fact store where the same first names recur across all
ten transcripts. Both shapes rank badly by embedding similarity alone.

### Capacity-triggered forgetting: negative, but it identified the right signal

Asked directly: can decay fire when there is too much information rather than
when time has passed? `_sweep_capacity.py` implements it. The trigger is a
fact's own neighbourhood -- how many other facts sit within `tau` cosine of it.
Under `cap` neighbours nothing happens at any age. Inside a crowded
neighbourhood the newest is kept and the rest suppressed, so age only ever
counts when something newer overlaps.

Run on the merged 12,083-fact store, full 101, retrieval only, each policy
against two matched-volume controls: OLDEST drops the same number of facts by
age alone, RANDOM by seeded coin.

```
  policy                        gold   stale  only-stale    vs shipped     silenced  in pool
  shipped                      79.2%   59.4%    13.9%                            0        0
  capacity  tau=0.8  cap=2     70.3%   58.4%    18.8%     -8.9  -1.0  +5.0    2064      941
    control oldest             63.4%   45.5%    15.8%    -15.8 -13.9  +2.0    1415      552
    control random             65.3%   57.4%    16.8%    -13.9  -2.0  +3.0    1936      867
  capacity  tau=0.85 cap=2     73.3%   61.4%    15.8%     -5.9  +2.0  +2.0     899      475
    control oldest             70.3%   52.5%    14.9%     -8.9  -6.9  +1.0     612      261
    control random             72.3%   60.4%    14.9%     -6.9  +1.0  +1.0     874      414
  capacity  tau=0.85 cap=5     78.2%   59.4%    13.9%     -1.0  +0.0  +0.0     361      185
  capacity  tau=0.9  cap=5     79.2%   59.4%    13.9%     +0.0  +0.0  +0.0      78       72
  capacity  tau=0.9  cap=10    79.2%   59.4%    13.9%     +0.0  +0.0  +0.0       7        1
```

Two things are true at once and both matter.

**The signal is real.** At every setting where anything fires, crowding damages
gold less than either control at the same volume: -8.9 against -15.8 for age and
-13.9 for chance. Density is a better selector than the clock, which is the
first time any forgetting rule in this project has beaten its own controls. The
mechanism is sound.

**The action is still wrong.** It never buys anything. Stale barely moves (-1.0
at its most aggressive) and only-stale, the case that decides correctness, goes
*up* at every setting that fires. Where the rule is free it has not fired on
anything reaching a prompt: at tau=0.9 cap=10 it silences 7 facts and removes 1
from a pool in 101 questions.

The `in pool` column is why this is conclusive rather than merely
disappointing. Several settings score exactly level with shipped, and it would
be easy to record those as "safe". They are inert. Silencing 100 facts and
touching 8 pool records is not evidence of safety.

Note also what OLDEST does: it is the only control that meaningfully removes
stale material, -13.9, and it pays -15.8 gold for it. That is decay's failure
shape reproduced exactly, on a different store, at matched volume.

The conclusion is the same one the stored-versus-reached probe reaches from the
other direction. Retrieval sits 2 questions under its own ceiling. **No rule
that removes, reorders or inserts records can win more than 2 questions,**
because that is all that is left in the store to win. Capacity triggering was
the best-designed member of that family and it still lost.

### The decay replacement is a second clean negative

Two designs were screened on the full 101, retrieval only, against the shipped
top-20. DEMOTE pushes a record down when a newer record says the same thing
(subject-word overlap above a threshold). EXPAND follows each leading fact to
whatever was said about it later, inserting the neighbour immediately behind its
parent.

```
  policy                  gold    stale   only-stale   fires    vs shipped
  shipped                79.2%   62.4%     14.9%
  demote t=0.15          73.3%   59.4%     17.8%             -5.9  -3.0  +3.0
  demote t=0.2           70.3%   61.4%     16.8%             -8.9  -1.0  +2.0
  demote t=0.3           75.2%   61.4%     15.8%             -4.0  -1.0  +1.0
  expand sim 0.75        78.2%   63.4%     15.8%    20.4%    -1.0  +1.0  +1.0
  expand sim 0.85        79.2%   62.4%     14.9%     4.8%    +0.0  +0.0  +0.0
  expand sim 0.90        79.2%   62.4%     14.9%     1.2%    +0.0  +0.0  +0.0
  expand+demote t=0.3    74.3%   60.4%     15.8%             -5.0  -2.0  +1.0
```

Same shape as decay, milder. Demoting costs more gold than it saves stale at
every threshold.

Expansion needed a fix before it could be judged at all. The first version
keyed its index on record id, and a record the router pulled out of the archive
**has no id**: `_cold_row_to_record` deliberately invents none, so `MemoryRecord`
falls back to a fresh uuid4. An id-keyed index misses every archived candidate
silently -- it does not fail, it returns nothing. A trace showed 23 of 38 pool
records present in the index and the missing 15 were exactly the archived ones.
The index was also built from Hot alone, 3,807 facts, ignoring Cold's 8,276.
`FactIndex` now loads both tiers and keys on fact text, and the `fires` column
above exists so that "no effect" can never again be read off a policy that
never ran.

With the mechanism verified working the answer is unchanged. Where expansion
fires often enough to matter it pulls in more stale than gold; where the floor
is tight enough not to hurt, it barely fires. There is no floor at which it
wins.

Read together with the decay result: reordering the shipped top-20, or
inserting into it, loses on every signal tried -- age, subject-overlap
supersession, and forward vector neighbourhood. Removing information is not how
this improves, and neither is adding more of it at the margin.

### mylocomoeval's question set is reproducible again

The filter that took the miner's 211 candidates down to the 101 that were
actually used was run inline and never saved, so the whole result rested on a
file nobody could regenerate. `_clean_updates.py` is that filter, reconstructed
against the saved intermediates and reproducing both stages exactly: **211 ->
101, zero disagreements**. Three rules, and each drops questions that would make
a measurement meaningless rather than merely noisy.

```
  absence stale    81   "no pet mentioned", "not specified before turn 31" --
                        nothing earlier exists to retrieve, so stale-in-prompt
                        can only read false and the score inflates silently
  turn-anchored    19   "as of turn 302" -- both arms are asked once, after the
                        last turn, and can never stand at turn 302
  repeated update   8   three questions, one update: all of "which game is
                        James playing" answer Cyberpunk 2077 over The Witcher 3
```

Two items are excluded by name rather than by rule ("natural", "previous job"),
so the judgement is visible instead of bent into a regex.

The important consequence: **101 is the honest supply from this mining pass,
not an over-aggressive cut.** More questions have to come from a better mining
prompt -- one that stops producing absence-stales and turn-anchored questions
in the first place -- not from relaxing this filter.

### Read-only close, rolled out

`close_readonly()` is now used by every harness that measures rather than uses a
store: `_ab_vs_rag.py`, `_mylocomoeval.py`, `_test_updates.py`,
`_sweep_recovery.py`, `_sweep_precision.py`, `_sweep_context_budget.py`,
`_probe_verbatim.py`, `_sweep_decay_signal.py`, `_sweep_supersession.py`,
`_ab_haystack.py`. The docstrings that claimed deferral alone made a run
read-only have been corrected in place. All eight edited files import-check
clean.

## State as of 5 September 2026

### Where we stand against RAG

Same session, same judge, same 1,537 questions, `_ab_vs_rag.py`:

```
ACCURACY
  OVERALL      ours 66.1%  RAG 65.6%   +0.5   p=0.725     tie
  temporal     ours 64.8%  RAG 47.7%  +17.1   p=1.7e-06   OURS
  open-domain  ours 75.8%  RAG 80.7%   -4.9   p=0.00154   RAG
  single-hop   ours 44.5%  RAG 47.3%   -2.8   p=0.466     tie
  multi-hop    ours 49.0%  RAG 46.9%   +2.1   p=0.774     tie
LATENCY (work only, quota waiting excluded)
  ours faster on 849, RAG on 688, sign test p=4.41e-05 -> ours reliably faster
  median paired difference -54 ms
CONTEXT
  ours 995 tokens   RAG 1,453 tokens
```

Read it plainly: we win temporal by a mile, we lose open-domain, everything
else is a tie, and we are faster on a third less context. Overall is a tie.

### mylocomoeval (new this session)

The knowledge-update test. LoCoMo cannot see forgetting in either direction --
it asks everything after the last turn about things said once -- so decay tuning
there has never bought more than about 1.65 points. LongMemEval can see it, but
its haystacks cost roughly 250 hours of ingestion. mylocomoeval is the middle:
`_mine_updates.py` pulls the real moments in the LoCoMo transcripts where
something stops being true, keeping both the answer that is now right and the
one it replaced. No re-ingestion, no invented data.

- questions: `C:/nmafc_ab/updates_final.json`, 101 vetted (211 mined,
  128 after a first clean)
- runner: `scripts/benchmarks/_mylocomoeval.py`, both arms in one loop
- results: `C:/nmafc_ab/mylocomoeval.json`

```
                        NMAFC    RAG (control)
correct                 72.3%    64.4%     +7.9, p=0.215 -> not significant
answered stale           8.9%     9.9%
gold in prompt          79.2%    69.3%     +9.9
stale in prompt         62.4%    52.5%     we surface MORE stale than RAG
both in prompt          47.5%    35.6%     wrong 14.6% vs 19.4%
only stale in prompt    14.9%    16.8%     wrong 100%  vs 82.4%
discordant: ours-only 20, RAG-only 12
```

Three things follow, and the second is the important one.

1. Our retrieval finds the current fact more often than RAG's: 79.2% against
   69.3%. That is where the 7.9 points come from.
2. The forgetting machinery contributes nothing. RAG has no supersession at
   all, and RAG puts stale facts in the prompt *less* often than we do, 52.5%
   against 62.4%. Invalidation is not removing what it exists to remove.
   Claim 5 fails against a control, not just against a target.
3. When both versions reach the prompt the model picks correctly about 85% of
   the time for either arm, so the reasoning is fine. The damage is the
   stale-only case, where we are wrong 100% of the time. It is a retrieval
   failure and cannot be prompted around.

The +7.9 is directionally ours but does not clear significance at n=101. The
same ratio would need roughly 40/24 discordant to reach p=0.06 and 60/36 to
reach p=0.018, so about two to three times the questions. Re-mining toward the
full 211 is the cheap way to settle it.

Retrieval reproduced the earlier single-arm run to the question (79.2 / 62.4 /
47.5 / 14.9 identical), which confirms the config is the shipped one. Only the
`correct` figure moved, 75.2% to 72.3%, which is ordinary generation and judge
drift.

### Decay, finally measured properly (5 September 2026)

`weight_signal` had only ever been measured on LoCoMo, where forgetting cannot
help, so the 5.3-point loss there proved nothing. Measured on mylocomoeval
instead, and the first sweep came back **perfectly flat** at every value from
0.0 to 1.0. Chasing that flatness found four separate reasons decay's score
never reaches a ranking, each one sufficient on its own:

1. `decay_record` returns early for CoreAnchor unconditionally. 80% of Hot RAM.
2. `delta_t = current_turn - last_reinforced_turn`, and retrieval reinforces
   what it touches. The median `last_reinforced_turn` in conv-26 is 212 against
   a final turn of 211, so the median fact reads as reinforced *after* the
   conversation ended and takes the `delta_t <= 0` early return.
3. `consolidation_index` has a median of 62 and a maximum of 1401, again from
   reads rather than from anyone repeating themselves. `alpha = e^(-0.15k)`, so
   at k=62 the fade rate is multiplied by about 0.0001.
4. Net effect: 290 of 294 stored weights are exactly 1.0, and adding a constant
   to every entity's score reorders nothing.

`_recompute_weights.py` rebuilds the weights from `created_at_turn`, which no
read touches, with `--reset-k` and `--core-lambda` to lift restrictions 1 and 3.
That spreads weights properly (mean 0.99 to 0.50, 200+ distinct values,
essentially nothing pinned). Re-swept on those stores:

```
  config      gold    stale   only-stale  neither      vs shipped
  w=0.0      79.2%   62.4%     14.9%       5.9%
  w=0.1      60.4%   51.5%     19.8%      19.8%    -18.8  -10.9  +5.0
  w=1.0      59.4%   51.5%     19.8%      20.8%    -19.8  -10.9  +5.0
```

**Decay works and it is harmful.** It does remove stale facts, 62.4% to 51.5%.
But it removes gold facts nearly twice as fast, 79.2% to 60.4%, and the number
that matters gets worse: stale-only-in-prompt rises from 14.9% to 19.8%. The
"neither" column more than triples. It is not selecting, it is deleting.

The reason is that decay's only input is age, and age is a bad proxy for
staleness. A fact about who someone is stays true for the whole conversation
while getting steadily older. Superseded facts are older than what replaced
them, but not by enough to beat the collateral damage.

This was measured on the benchmark most favourable to decay, after repairing
the two contaminations that had been masking it. It still loses. Treat the
question as closed.

### Harness bug found and fixed the same day

`defer_reinforcement_writes=True` does **not** make a run read-only. It only
buffers; `NeuromorphicMemory.close()` then calls `flush_reinforcements()` and
commits, resetting `weight` to 1.0 and advancing `consolidation_index` and
`last_reinforced_turn` on every record the queries touched. Several docstrings
in `scripts/benchmarks/` claimed "deferred and never flushed, so the stores are
left exactly as they were". That claim was wrong.

It showed up as a sweep whose second and later configurations read a store its
first configuration had already flattened. `close_readonly()` in `_ab_budget.py`
drops the buffer before closing. **`_sweep_decay_signal.py` uses it. The other
harnesses still call `close()` and should be switched over.** Accuracy results
already taken are not invalidated -- reinforcement changes weight and k, and
with `weight_signal=0` neither reaches ranking -- but any future run that reads
weight must use the read-only close.

### Settled negatives, do not redo these

- **Tier misclassification is not the problem.** `_relabel_tiers.py` re-ran the
  classifier over all 3,807 facts using the rules sliced from the live
  extraction prompt. CoreAnchor went *up*, 68.4% to 70.5%, and 82.2% of labels
  were unchanged. 68% is what those rules correctly produce on chat about
  people's lives.
- **Keeping invalidated facts is a dud.** `_sweep_recovery.py` over all 1,540
  questions: `exclude_invalidated=False` buys +0.3 points of strict
  reachability. Noise. It stays True.
- **Hydration is a slow climb.** hyd-5 (shipped) 59.9% strict, hyd-8 60.6%,
  hyd-12 61.4%. hyd-12 costs 449 extra tokens, which hands back the whole
  context saving over RAG. Only 5 to 8 is worth an accuracy A/B, and it has not
  been run.
- **Decay's output never reaches ranking.** `weight_signal=0.0`, so the score
  decay computes is not read by retrieval. `recency_boost=0.0`.
- **Cold is searched every time.** `always_search_cold=True`, so the hot/cold
  routing threshold never fires.

### Live configuration

```
always_search_cold=True    weight_signal=0.0     recency_boost=0.0
exclude_invalidated=True   hydrate_top_k=5       compact_validity=True
defer_reinforcement_writes=True
rerank_top_k=20  hot top_k=10  cold budget=20  max_hops=2
```

### Cost, measured not guessed

From `vs_rag.json`, `chars // 4`:

```
head-to-head 1,537 q   6,148 calls   7.1M input tokens   (already spent)
mylocomoeval   101 q     404 calls   0.47M input tokens  (6.6% of the above)
```

Ingestion rate is 1,110 turns/hour, which is why re-ingestion is the thing to
avoid. Throttle was 0 ms across the whole head-to-head, so the Azure rate limit
is not binding -- any shortage is spend, not speed.

LongMemEval for reference: oracle 9.9 h, full S about 247.8 h, the
knowledge-update 78 with the haystack trimmed 50 to 15 about 11.6 h. That last
one is the only practical version and mylocomoeval exists to avoid needing it.

## Scripts added, none committed

| file | what it does |
| --- | --- |
| `scripts/benchmarks/_ab_vs_rag.py` | paired head-to-head, accuracy and latency in one session |
| `scripts/benchmarks/_sweep_recovery.py` | retrieval-only screen of invalidation and hydration, no generation |
| `scripts/benchmarks/_relabel_tiers.py` | re-classify existing facts without re-reading conversations |
| `scripts/benchmarks/_mylocomoeval.py` | knowledge-update, both arms paired |
| `scripts/benchmarks/_sweep_decay_signal.py` | retrieval-only sweep of `weight_signal` and `recency_boost` |
| `scripts/benchmarks/_recompute_weights.py` | rebuild decay weights offline from `created_at_turn` |
| `scripts/benchmarks/_build_haystack.py` | merge the ten stores into one, no API calls |
| `scripts/benchmarks/_sweep_supersession.py` | screen decay replacements (demote, expand) |
| `scripts/benchmarks/_ab_haystack.py` | both arms against the merged pile |
| `scripts/benchmarks/_clean_updates.py` | the 211 -> 101 filter, written down and verified |
| `scripts/benchmarks/_probe_stored_vs_reached.py` | is a miss an access failure or a write failure |
| `scripts/benchmarks/_sweep_capacity.py` | crowding-triggered forgetting, with matched-volume controls |
| `scripts/benchmarks/_sweep_hydration.py` | is the answer in the source turns at depth, retrieval only |
| `scripts/benchmarks/_ab_hydration.py` | deep hydration generation A/B, with a drift gate on the saved baseline |
| `scripts/benchmarks/_screen_ranking.py` | name the answering record once, then price every retrieval config for free |
| `scripts/benchmarks/_measure_code_context.py` | symbol-index context against grep-and-read-the-file, with a seed-kept column |
| `scripts/benchmarks/_audit_quantities.py` | read-only SQLite audit: numbers and speakers a fact asserts against the turn it came from |
| `scripts/benchmarks/_diagnose_multihop.py` | split any category into both-right / we-win / we-lose / both-wrong and print the questions in full |
| `scripts/benchmarks/_probe_answer_type.py` | size the type-demanding class from saved answers, free |
| `scripts/benchmarks/_ab_answer_type.py` | paired prompt A/B for `TYPE_RULE`, gated, untyped controls copied from A |
| `scripts/benchmarks/_ab_answer_type_repair.py` | regenerate only the arm whose prompt changed; `--dry-run` prices it first |
| `scripts/benchmarks/_score_answer_type.py` | score the gated arm and project it onto the full 1,535, free |
| `scripts/benchmarks/_screen_hydrate_pool.py` | gold-in-context off vs on, gains and displacements, free |
| `scripts/benchmarks/_ab_open_domain.py` | paired A/B of the whole open-domain stack, both arms one session. Arm A is now configurable, so it can be run as "previous best vs previous best plus one change" |
| `scripts/benchmarks/_diagnose_open_domain.py` | splits the remaining losses into reach and conversion, free |
| `scripts/benchmarks/_screen_source_block.py` | where each gold actually is, FACTS vs SOURCE, and what a source-aware prompt could win or cost. Free |
| `scripts/benchmarks/_screen_partial_answers.py` | is a wrong answer a piece of the right one, or a different one? Pure text, no store, no spend |
| `scripts/benchmarks/_screen_oracle_reach.py` | the ceiling: is the gold in ANY turn at all, and does RAG win the reachable half. Reads cold ROM directly, no spend |
| `scripts/benchmarks/_screen_scan_rank.py` | when the gold IS in a turn and we miss it, what rank does BM25 give that turn? Decides lexical-vs-semantic. No spend |
| `scripts/benchmarks/_screen_grounding.py` | price `source_grounding` off the paid targets, plus churn on questions we already win |
| `scripts/benchmarks/_screen_adaptive_signal.py` | can a query-time signal tell which questions need turns? Answer: no. Kept as the record of a mechanism not built |
| `scripts/benchmarks/_run_open_domain_full.py` | one arm, all 839 open-domain questions, paired against RAG's saved answers. Questions come from the saved file so pairing cannot drift; reports vs RAG, vs shipped as gross counts, and both context widths |
| `scripts/benchmarks/_ab_conversion.py` | prompt variants generated only on the questions a prompt could fix (gold already in context, we lose) plus a sample of the wins it could break. Re-generates the control so drift cancels; projects onto 839 showing its own arithmetic |
| `scripts/benchmarks/_screen_scan_scope.py` | how much of the SOURCE block goes to turns from other conversations, and whether the gold turn survives bounding the scan. Sweeps span width and margin off one retrieval pass. Prints its own upper bound as an upper bound |
| `scripts/benchmarks/_screen_turn_embeddings.py` | where BM25 and cosine each rank the turn that actually holds the gold. The screen that found meaning beats words 65.1% to 50.0% inside the scan window. Caches turn vectors beside the store |
| `scripts/benchmarks/_build_turn_index.py` | writes `turn_vectors` into a store's cold.db, reusing the screen's cached vectors so nothing is embedded twice. `--check` reports without writing |
| `scripts/benchmarks/_diagnose_partial_reach.py` | how much of each gold answer reached the context, as a fraction rather than a yes/no. Buckets the losses into nothing / a fragment / a piece missing / all of it, and prints which piece went missing. The screen that showed 57.7% of losses already have the whole answer in context. No generation |

Modified, not new: `_ab_budget.py` gained `close_readonly()` and a `facts=`
argument; seven harnesses switched to `close_readonly` and had their read-only
claims corrected.

**One framework change, not just harnesses.** `context_facts_top_k` in
`schemas/memory.py`, honoured in `query_router.format_context`. It renders the
top N retrieved facts while `_source_turns` still hydrates out of the full
list, which is the only way to spend fact tokens on source tokens. Unset means
render everything, so every number taken before it existed is unaffected, and
the ten tests covering context formatting pass unchanged.

**New framework code, 7 September.** `fact_overlap_max` and `separate_facts`
(pattern separation); `src/nmafc/integration/quantities.py` and the quantity veto
inside `separate_facts`, plus the quantity argument to `best_lines`; and the
whole of `src/nmafc/code/` with `src/nmafc/schemas/code.py`. New tests:
`tests/unit/test_pattern_separation.py`, `tests/unit/test_quantities.py`,
`tests/unit/test_code_index.py`. `_sweep_hydration.py` parses a sixth setting
field, so the notation is now `facts:turns:lines:dedupe:whole:overlap`.

**New framework code, 7 September, second batch.**
`src/nmafc/integration/answer_type.py` (`TYPE_RULE`, `demanded_type`,
`type_tag`) and `src/nmafc/integration/grounding.py` (`terms`,
`source_scores`), plus `source_grounding` in `DecayConfig`,
`QueryRouter._grounded`, and `grounding` / `grounding_weight` arguments on
`rerank` and `reciprocal_rank_fusion`. New tests:
`tests/unit/test_answer_type.py` (50, after the gate), `tests/unit/test_grounding.py`
(32). `answer_type.gate` was added later the same day, with
`_ab_answer_type_repair.py` (repairs a paired run in place when only one arm's
prompt changed) and `_score_answer_type.py` (scores the gated arm and projects it
onto the full benchmark, both free). Suite is 526 passed, 3 skipped.

Nothing in that batch is on by default. `source_grounding` is `0.0`, and with
no weight the fusion is byte-for-byte what it was; `answer_type` is imported
only by its A/B harness and is not yet in any shipped arm prompt.

Push by hand, 7 September second batch:

```
git add src/nmafc/schemas/memory.py src/nmafc/integration/answer_type.py src/nmafc/integration/grounding.py src/nmafc/integration/query_router.py src/nmafc/engine/reranking.py src/nmafc/storage/cold.py src/nmafc/storage/cold_base.py tests/unit/test_answer_type.py tests/unit/test_grounding.py scripts/benchmarks/_diagnose_multihop.py scripts/benchmarks/_probe_answer_type.py scripts/benchmarks/_ab_answer_type.py scripts/benchmarks/_ab_answer_type_repair.py scripts/benchmarks/_score_answer_type.py scripts/benchmarks/_screen_grounding.py scripts/benchmarks/_screen_hydrate_pool.py scripts/benchmarks/_ab_open_domain.py scripts/benchmarks/_diagnose_open_domain.py scripts/benchmarks/_screen_source_block.py scripts/benchmarks/_screen_partial_answers.py scripts/benchmarks/_screen_oracle_reach.py scripts/benchmarks/_screen_scan_rank.py scripts/benchmarks/_screen_adaptive_signal.py scripts/benchmarks/_run_open_domain_full.py scripts/benchmarks/_ab_conversion.py scripts/benchmarks/_screen_scan_scope.py scripts/benchmarks/_screen_turn_embeddings.py scripts/benchmarks/_build_turn_index.py scripts/benchmarks/_diagnose_partial_reach.py scripts/benchmarks/_ab_budget.py scripts/benchmarks/arms/base.py RECENT_OPS.md
git commit -m "retrieval: rank facts by the turns behind them, so the question's own wording reaches the ranking for the first time; prompt: name the kind of answer a question demands, in a static rule plus a five-token tag, and reject verb clauses that merely end on a type noun"
git push origin muna
```

Push by hand, 7 September first batch:

```
git add src/nmafc/schemas/memory.py src/nmafc/schemas/code.py src/nmafc/integration/query_router.py src/nmafc/integration/quantities.py src/nmafc/code/__init__.py src/nmafc/code/symbols.py src/nmafc/code/index.py src/nmafc/code/render.py tests/unit/test_pattern_separation.py tests/unit/test_quantities.py tests/unit/test_code_index.py scripts/benchmarks/_sweep_hydration.py scripts/benchmarks/_measure_code_context.py scripts/benchmarks/_audit_quantities.py RECENT_OPS.md
git commit -m "context: pattern separation over the rendered facts, with a quantity veto so two different numbers are never merged into one fact; code: an exact ast symbol index with reverse edges, hash staleness and a hard-capped graded renderer that degrades a seed rather than dropping it"
git push origin muna
```

Push by hand, the earlier batch:

```
git add src/nmafc/schemas/memory.py src/nmafc/integration/query_router.py src/nmafc/wrapper.py scripts/benchmarks/_ab_prompt.py scripts/benchmarks/_ab_dates.py scripts/benchmarks/_ab_context.py scripts/benchmarks/arms/neuromorphic.py scripts/benchmarks/arms/neuromorphic_tuned.py scripts/benchmarks/arms/stateful_nodecay.py scripts/benchmarks/_ab_vs_rag.py scripts/benchmarks/_sweep_recovery.py scripts/benchmarks/_relabel_tiers.py scripts/benchmarks/_mylocomoeval.py scripts/benchmarks/_sweep_decay_signal.py scripts/benchmarks/_recompute_weights.py scripts/benchmarks/_build_haystack.py scripts/benchmarks/_sweep_supersession.py scripts/benchmarks/_ab_haystack.py scripts/benchmarks/_clean_updates.py scripts/benchmarks/_probe_stored_vs_reached.py scripts/benchmarks/_sweep_capacity.py scripts/benchmarks/_sweep_hydration.py scripts/benchmarks/_ab_hydration.py scripts/benchmarks/_screen_ranking.py scripts/benchmarks/_ab_budget.py scripts/benchmarks/_sweep_precision.py scripts/benchmarks/_sweep_context_budget.py scripts/benchmarks/_probe_verbatim.py scripts/benchmarks/_test_updates.py WHAT-CHANGED.md RECENT_OPS.md
git commit -m "context: separate rendered facts from hydrated source, and give format_context the question it is hydrating for; benchmarks: paired harnesses take the new context settings, price against every question rather than the losses, and compare two context configurations in one session"
git push origin muna
```

## Next, in order of what it buys

### Rewritten 9 September 2026

**Both runs below were approved and have now run. Results are in "State as of 9
September". A shipped the best numbers yet (79.0% at 958t); B is closed as
null.** The text is kept because the reasoning that chose them is worth reading
next to what they returned -- B was argued for on a gate figure that turned out
to be wrong, and it still would have been the right thing to test.

What is open now, in order:

1. **`qualifier` or `discriminate` on the gated path only.** The opposite
   targeting of B. Reason stated above: the 13 remaining gated losses are the
   wrong-candidate failure those two variants were written for. One category
   run. Not approved.
2. **Re-measure the other three categories under the candidate.** They have only
   ever been generated under shipped retrieval, and the gated result says short
   answers got worse, which is what temporal is made of. This is a safety check
   on a configuration that otherwise looks ready.
3. Nothing else free is left on open-domain retrieval.

The original text of the two decisions follows.

**A. Confirm the retrieval candidate.** Facts 20 / hydrate 6 / semantic 12 /
floor 0.30, 958 tokens, on all 839. This is +4.5 points of loss reach over the
configuration the last paid run used, and it is entirely unmeasured on accuracy.
Command in the block below. It is the safer of the two: it changes what the
model is given, not what it is asked to do, and the 178-question reach screen
says the win side does not regress (88.2% -> 88.8%).

**B. Resolve the completeness instruction, targeted this time.** `--variant`
now fires only where no length rule does. That is 713 of 839 questions, which is
where the entire 2.4-point deficit lives, and it keeps the instruction off the
126 single-value questions already at 93.2% that the previous attempt damaged.

**Use `assemble2`, not `assemble`.** All five questions `assemble` broke had
short gold answers, where being told to survey the context turned into padding.
`assemble2` keeps the survey and adds "where the question asks for a single
value, give exactly that value and stop". Gating and `assemble2` fix the same
failure from two directions -- gating keeps the instruction away from the
questions that are already short, and `assemble2` handles the short ones that
have no length rule to catch them, which is the case gating alone leaves open.

Run A and B as one comparison if only one budget is available -- the pair share
a retrieval configuration, so a single run of each gives both answers, and the
drift floor of 61 verdicts means a small control would tell us nothing anyway.

```
python -u scripts/benchmarks/_run_open_domain_full.py --out C:/nmafc_ab/od_cand.json
python -u scripts/benchmarks/_run_open_domain_full.py --variant assemble2 --out C:/nmafc_ab/od_variant.json
```

Both print their configuration and the indexed turn count on startup, so the log
records whether semantic ranking was live. The store has 2,957 turns indexed.

Two guards were added first, because the failure they prevent costs a whole
paid run and is invisible in the output:

- Semantic ranking **fails silently on a store with no `turn_vectors`** -- the
  run is just the old configuration wearing the new numbers. The startup line
  now counts indexed turns and says `NO INDEX, semantic ranking is inert`.
- Resume is keyed on the question, not the configuration, so an existing
  results file resumes under whatever settings are current, and a **complete**
  one answers nothing and reports the old run's numbers as the new run's. The
  defaults have changed twice since `od_full.json` was written. Every setting
  that changes an answer is now fingerprinted to `<out>.config.json` and a
  mismatch refuses to start, naming which settings differ. Verified: pointing
  the new defaults at `od_full.json` exits before spending anything.

**What is honestly expected.** Retrieval alone does not close 1.8 points -- the
arithmetic says a tie needs about 17 points of loss reach and the session found
4.5. B is the lever that matches the failure shape, and it is unproven. Said
before spending, not after.

**Still owed regardless of the above:** the other three categories have only
ever been generated under the shipped retrieval. If the candidate ships, they
need re-measuring, and temporal is the one to watch, since it was the category
the fact count protected.

### Older list, from 6 September, after pricing and the temporal safety run

The candidate configuration:

```
  top_k                  10 -> 20      free, 6 more answering records reached
  rerank_top_k           20 -> 40      retrieval only, no tokens rendered
  context_facts_top_k   all ->  6
  hydrate_top_k           5 -> 28
  hydrate_lines        whole ->  1
  hydrate_full_turns       0 ->  4
  dedupe_source_headers false -> true
```

Depth 28 and four whole turns, not 40 and eight. The presence sweep argued for
40:8 on a price of 1,427t; priced across every question instead of only the
losses it costs 1,659t on open-domain, 264 over RAG's tightest per-category
spend of 1,395. It was never shippable.

1. ~~**Check the configuration is safe on temporal.**~~ **Done, and it passes.**
   120 temporal questions, both arms generated in the same session:

   ```
     ours   67.5%   1262 tokens
     RAG    43.3%   1431 tokens
     +24.2, p=0.000154
   ```

   Against the saved run on those same 120, ours moved +1.7 and RAG -1.7, both
   inside the noise, and the margin held: +24.2 today against +20.8 saved. Six
   facts does not cost us the category, which was the one thing that could have
   killed the configuration outright. Cost rose from 964t to 1262t and is still
   169 under RAG. -> `C:/nmafc_ab/candidate_temporal.json`
2. ~~**One paired A/B, both arms, open-domain, ~3.0M tokens.**~~ **Done.** All
   839, both arms in session: ours 76.8% at 1302t, RAG 80.0% at 1460t, **-3.2 at
   p=0.0426**, and ours faster at p=2.45e-05. Against the saved shipped run
   (licensed, because RAG itself drifted only -0.7 at p=0.307) the candidate is
   **+2.5 points, net +21 questions, p=0.031**, and the margin against RAG
   narrows from -6.4 to -3.2. -> `C:/nmafc_ab/candidate_open.json`. Written
   before spending, and worth keeping: this is the only experiment
   that can settle it, and both arms have to run together now that the saved
   baseline has drifted. It tests the two screened changes as one configuration,
   which is the right unit: they interact, because trimming facts is what turns a
   rank-14 record from a competing summary into a source turn, and raising
   `top_k` is what puts six more records inside that window. Expected: hydration
   alone ties RAG, ranking adds up to 6 questions, and the selection effect on
   the 47 already-present losses is unmeasured in either direction. A win is
   plausible and not assured -- said before spending, not after.
3. ~~**Re-test the source-aware answer prompt at the candidate configuration.**~~
   **Done, and it is null.** 400 open-domain, both prompts in session:
   A 77.0%, B 77.5%, **+0.5 points, fixed 7 broke 5, p=0.774**. Twelve discordant
   pairs in four hundred questions -- the prompts agree almost everywhere,
   because the model was already reading `<SOURCE>` without being told what it
   was. The rule was set before spending (clear +3 or close), and it did not, so
   **this line is closed at both configurations.** The useful part is negative
   and real: the remaining gap is about what is *in* the context, not how the
   context is labelled. -> `C:/nmafc_ab/prompt_candidate.json`
4. ~~**More facts, at `12:24:1:1:4`.**~~ **Done, null, and it should not have
   been paid for.** Stopped at 300 of 400 once the answer was clear:
   A 76.0%, B 76.3%, **+0.3 points, fixed 11 broke 10, p=1**, context
   **A 1,300t, B 1,393t against the 1,395 ceiling.**

   **Read the mistake before reading the result.** The cost half of this
   experiment was already measured, for free, by the all-questions pricing sweep:
   it put `12:24:1:1:4` at 1,395 on open-domain, the ceiling exactly, with no
   headroom. Under the standing rule -- cost must not rise, and every round must
   be won -- a configuration that *ties* on cost cannot win the cost round, so no
   accuracy result could have made B shippable. The accuracy half was therefore
   worth almost nothing before it was run. **Screen cost with
   `_sweep_hydration.py --select all` first; it needs no generation, and if a
   setting is not comfortably under the ceiling there is nothing to generate.**

   The one thing worth keeping: the pricing table is now validated against live
   generation, predicting 1,395 and measuring 1,393. Trust it, and stop paying to
   confirm it. -> `C:/nmafc_ab/context_facts.json`

   Original reasoning, kept because the accuracy question itself was genuinely
   two-sided and is now answered: 26 questions broke when the fact list went from
   twenty to six, and 26 questions is the whole remaining 3.2 points. But 17 of
   the fixes came from *removing* competing summaries, so putting facts back may
   re-break them -- and that is exactly what happened, 11 fixed against 10 broken,
   the two effects cancelling almost to the question.

   `_ab_context.py` is the harness and it is sound: A is the shipped candidate
   `6:28:1:1:4`, B is `12:24:1:1:4`, both generated in one session on the merged
   store with the same answer prompt, it refuses to run two identical arms, and it
   scores cost per category against the ceiling because a sampled average can hide
   one category going over. Keep it for any future two-configuration question --
   but price the configuration first and only run it if there is real headroom.

   Operational note worth keeping: the first attempt died at 100/400 on a storm of
   `APIConnectionError` (exit 4), not on anything in the harness -- a single
   sequential chat call succeeded in 5.9s straight afterwards, so the endpoint
   refuses concurrent connections rather than being down. Those failures cost
   wall-clock, not tokens. **Drop `--restart` to resume**: rows checkpoint every
   20 and the sample is drawn before `done` is applied, so a resumed run continues
   through the same questions instead of drawing fresh ones. Dropping concurrency
   from 6 to 4 did not stop the storms but the run still made progress through
   them.
5. **The remaining categories at the candidate, ~1.4M.** Only needed to publish a
   whole-benchmark margin rather than an open-domain one. single-hop and
   multi-hop, 375 questions, both arms.
6. **Extraction recall, at write time.** 18 of 101 current answers are not in
   the store at all. The failures are specific, not diffuse: counts dropped
   ("Melanie has a dog and a cat" for a gold of two cats), destinations dropped
   (Chicago exists only as somebody's suggestion, never as a place visited), and
   attribution to the wrong speaker. Revise the extraction prompt against those
   three, re-ingest ONE conversation to a copy, and re-run
   `_probe_stored_vs_reached.py` on the copy. One conversation is about 32
   minutes of ingestion. Never in place.

   **Two of the three named causes are now measured over the whole store, and
   both are closed.** `_audit_quantities.py`, free, no generation: counts said
   aloud and dropped from the summary are **55 facts, 0.7%**, mostly detector
   artefacts; facts naming a person absent from their source turn are **63,
   0.8%**, and every one inspected is the extractor correctly resolving an
   implicit addressee. Both impressions came from a handful of remembered
   examples and neither holds at scale. **Do not revise the prompt for counts or
   attribution.** Full write-up above, including the 19.5% of numeric facts that
   assert a number the source never said, which is date arithmetic and is the
   mechanism behind the temporal win.

   **Destinations dropped is the one cause left standing, and it is still
   unmeasured.** No free detector exists for it: "Chicago exists only as
   somebody's suggestion, never as a place visited" is about modality, not about
   a missing token, so no lexical check sees it. Either find a cheap test for it
   or stop treating the 18 unreached answers as an extraction problem -- with two
   of three causes eliminated, the remaining explanation is likelier to be
   retrieval or the questions themselves.
7. **The 23 answers still unreached at `top_k` 20.** Read them as a set: dated
   open-domain questions and thematic ones, against a store where the same first
   names recur across all ten transcripts. Neither shape ranks well on embedding
   similarity to the question. A date filter for questions carrying a date is the
   obvious try and it is free at inference; query expansion is not, it is an LLM
   call per query.
8. **Grow mylocomoeval to settle the +7.9.** The 101 is the honest yield of the
   current mining pass, so this needs a better mining prompt over full
   transcripts, not a looser filter. About 10 calls, ~500k tokens.
9. **A confusable haystack, if scale robustness is to be claimed at all.** The
   merged LoCoMo pile has disjoint casts and turned out not to be hard. Nothing
   currently measured shows what happens under genuine competition.

Closed, do not reopen without new evidence: decay, demote, expand,
capacity-triggered pruning, `weight_signal`, `recency_boost`. Six rules, three
signals, all measured on the same 101, all negative. They share one assumption
-- that the answering fact is in the store and badly ranked -- and on
mylocomoeval that assumption is false 90% of the time.

Scoped deliberately: that list is closed **for mylocomoeval**. On open-domain
the same assumption holds for 29 of the 116 losses, and the screen has already
found a free setting that reaches six of them, so ranking work aimed at
open-domain is not ruled out and should be judged on open-domain accuracy, which
none of the six ever was.
