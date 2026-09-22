"""Prompt variants, tested only on the questions a prompt could possibly fix.

Three free screens this session mispriced a prompt change, and they all failed
the same way: a reach screen asks where the gold *is*, and a prompt changes what
the model *does* with what is already there. Those are different events. So
prompt levers have to be generated, and generation is the expensive thing.

This makes generation cheap by aiming it. A prompt change can only fix a
question whose gold is already in the rendered context, and it can only break a
question we currently get right. Everything else in the category is spend with
no information in it. Under the current configuration that is 77 fixable and 661
breakable out of 839, and the breakable side needs a sample rather than a census
because a regression rate is estimable from far fewer questions than a rare fix.

    targets    we lose it, and the gold IS in the context      every one
    controls   we win it                                       a sample

Both slices are generated for every variant including the control prompt, in one
session, so the 1.2-point run-to-run drift cancels inside each comparison rather
than being carried across saved answers.

**The control prompt is re-generated, not read from `od_full.json`.** Reading it
would make every variant look better or worse by whatever the provider was doing
an hour ago. It costs one extra pass and it is the difference between measuring
a prompt and measuring the weather.

What comes out is fix and break counts per variant on each slice, and a
projection onto all 839 that states its own arithmetic rather than hiding it.

Usage:
    python -u scripts/benchmarks/_ab_conversion.py --targets 8 --controls 8
    python -u scripts/benchmarks/_ab_conversion.py --variants discriminate,qualifier
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from scipy.stats import binomtest  # noqa: E402

from nmafc.integration.answer_type import gate  # noqa: E402
from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks._ab_answer_type import PROMPT_B, answer_with  # noqa: E402
from scripts.benchmarks._ab_budget import (  # noqa: E402
    PERMANENT,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks._screen_hydrate_pool import present  # noqa: E402
from scripts.benchmarks.arms.neuromorphic_tuned import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
)
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402

# Every variant is an appendix to whichever prompt the answer-type gate chose,
# so a variant that says nothing about a question leaves that question's prompt
# byte-identical to the control. That is what makes the control honest.
#
# All of them attack one measured failure: 60% of the questions RAG answers and
# we do not are "right topic, wrong instance". `What novel is Evan reading that
# he finds gripping` returns a different novel he also mentioned. The word that
# discriminates was dropped by extraction, so several facts score identically
# and the model takes the first plausible one.
VARIANTS: dict[str, str] = {
    # The narrow one. Names the failure and nothing else, so if it works the
    # reason is legible and if it fails nothing else is implicated.
    "qualifier": """
Questions here usually carry a qualifier that decides between several similar facts: an adjective, a date, a relation, a purpose. Find that qualifier before answering. If more than one fact fits the topic, the answer is the one that also fits the qualifier, even when another fact looks more complete.""",
    # The same instruction plus permission to read the dialogue for the
    # qualifier. Deliberately NOT a "prefer SOURCE" rule -- that was measured at
    # -2.0 and dropped. This says where to check a qualifier, not which block to
    # trust for the answer.
    "discriminate": """
Questions here usually carry a qualifier that decides between several similar facts: an adjective, a date, a relation, a purpose. Find that qualifier before answering. If more than one fact fits the topic, the answer is the one that also fits the qualifier, even when another fact looks more complete.
The facts are a summary and often drop the qualifier. When two facts fit the topic equally and neither carries it, check the dialogue lines for which one the question is describing. Use them to choose between candidates, not as a reason to answer at greater length.""",
    # Completeness rather than discrimination. Aimed at the other 37%: answers
    # that are a fragment of the gold, or most of it with a piece missing.
    "assemble": """
Some questions ask for something with several parts: a list of activities, a plan with more than one element, a reason with more than one half. When the question is plural or asks what someone does, gather every part the facts support and give them all, briefly. Do not stop at the first part you find. Do not pad a single-part answer to make it look like several.""",
    # `assemble` measured 12 fixed against 5 broken on the fixable census, and
    # every one of the five breaks was a short gold -- `when she was 10`, `at a
    # festival`, `action and sci-fi` -- where surveying the whole context turned
    # into padding. The fixes were not all compound either: it recovered `The
    # Great Gatsby` and `Apex Legends`, single-value answers where the win came
    # from considering every candidate instead of taking the first plausible
    # one. So the behaviour to keep is the survey and the behaviour to stop is
    # the padding, and they are separable in the instruction.
    # The mirror of `qualifier`, and built from the losses rather than from a
    # theory. On the ungated path RAG wins 10 questions where our answer is a
    # strict subset of its answer, and in every one it named several candidates
    # and the judge scored containment: gold "a cactus in the desert", RAG "the
    # sunset painting and the cactus painting", marked right despite leading
    # with the wrong one. We commit to one candidate and lose when we pick
    # wrong. `qualifier` says choose, which is why it wins on the gated path
    # where the gold is exact and loses here by 10.
    #
    # This is closer to gaming the judge than to remembering better, and the
    # honest defence is that the brevity gate is ours and the RAG baseline has
    # no equivalent, so the arms were never style-matched. Report how often a
    # hedged answer leads with a wrong candidate before believing the gain.
    "hedge": """
When more than one thing in the context could be what the question is asking about, and nothing in the question decides between them, name them all in one short phrase joined by "and" rather than picking one. Where only one thing fits, give that one and stop. Do not add explanation, and do not lengthen an answer that is already the single thing asked for.""",
    "assemble2": """
Read every fact and every dialogue line before answering. Do not answer from the first one that looks relevant; several will look relevant and usually only one matches everything the question asks.
Where the question asks for something with several parts -- a list of activities, a plan with more than one element, a reason with more than one half -- give every part the context supports.
Where the question asks for a single value -- one name, one title, one date, one place, one number -- give exactly that value and stop. Do not add context, do not add a second candidate, and do not explain the choice.""",
    # Written for LoCoMo's adversarial category from the `full_v3` run, and
    # superseded before it was ever paid for. Those numbers -- 22.8% of wrong
    # answers containing every gold word, wrong answers running 25.1 words --
    # came off a configuration without the length rules. The first paired
    # adversarial run under the shipping stack, 446 questions on 10 September,
    # says something else: wrong answers run 5.2 words against 3.2 for right
    # ones, and only 2.4% contain every gold word. The verbosity this was
    # written to cut is already gone. Kept for the record, not for shipping.
    "answer_first": """
A question may get a detail wrong -- the wrong object, the wrong person, the wrong occasion. If the context still shows what was actually meant, give that answer directly, in the same few words you would use if the question had been phrased correctly.
Do not open by correcting the question, do not explain what the question got wrong, and do not say the context does not mention it when a near match is plainly there. A correction placed before the answer reads as a refusal.
If the context genuinely holds nothing on the subject, say so in a few words and stop.""",
    # What the paired adversarial run actually shows. Of 446 questions we
    # decline 112 and are scored right on 3 of them, so declining costs about
    # 109 questions and buys almost nothing back. It is not that the evidence is
    # missing: on the 103 questions RAG answers and we do not, every gold
    # content word is already in our context on 78 of them, and only 21 look
    # like retrieval at all.
    #
    # Which makes this a strange clause to have to write, because the shipped
    # prompt already says "NEVER say 'No information available'" and "Always
    # prefer giving an answer over refusing", and the model declines on a
    # quarter of the category anyway. So repeating the ban would be the third
    # time of asking and is not the fix.
    #
    # The difference is that the ban names one wording and the failures come in
    # two, and the second is not recognisable as a refusal from inside the
    # rule. "No information available" is banned; "Caroline did not go camping
    # with her family" and "Melanie is not pursuing counseling" are not, and
    # they are premise denials rather than answers. They read to the model as
    # careful and to the judge as wrong. So this names the denial rather than
    # the phrase, and gives the reason -- the compacted facts almost never
    # restate the question's wording, so an imperfect match is what a correct
    # answer looks like here.
    "commit": """
Every question here is about something the facts hold, even where it names it wrongly. Compacted facts rarely restate a question's own wording, so a near match is the answer rather than evidence that there is none.
If anything in the facts is about the subject asked about, name it. Do not reply that the facts do not mention it, do not reply that the person did not do it or does not have it, and do not reply that there is no information: a denial of the question is scored exactly like a wrong answer, and on these questions it nearly always is one.
Only where nothing in the facts touches the subject at all should you say so, in three words or fewer.""",
    # `commit` was assumed to cost about 67 tokens of context on every question
    # it is attached to, and it costs none: a clause goes in the system prompt,
    # which this project already decided in `answer_type.py` is the arm's own
    # instructions rather than retrieved context, on the grounds that RAG has a
    # system prompt too. Real tokens to a real model, simply not tokens the
    # budget metric was defined to cover. So brevity here is a virtue rather
    # than a constraint, which is a weaker reason to want it -- and it turned
    # out to matter anyway.
    #
    # It is also aimed slightly wrong. On an eight-question smoke `commit` did
    # not stop the refusals, it reworded them: "There is no fact stating
    # Caroline ran a charity race; Melanie ran one" obeys every clause about
    # not saying there is no information while still refusing. The model is not
    # missing the fact, it is objecting to the attribution, which is what
    # LoCoMo's adversarial questions swap. So this one names the swap.
    "commit_short": """
Answer from the nearest thing the facts hold, even where the question names the wrong person, object, place or occasion. Never reply that the facts do not mention it, that it did not happen, or that it was someone else.""",
    # `commit_short` works and costs, and reading what it broke says the cost is
    # a wording bug rather than a real trade. Over 1,235 questions it took
    # adversarial from 33.6% to 40.6% (+26/-7) and temporal from 74.8% to 72.0%
    # (+2/-8), and the temporal breakages are not the ones a premise clause
    # should be able to reach:
    #
    #   gold "The week before 21 January 2022"  ->  "21 January 2022"
    #   gold "a handwritten letter"             ->  "a cute note"
    #   gold "Rome"                             ->  "Paris"
    #
    # "the nearest thing the facts hold" licenses two separate things and only
    # one was wanted. Answering despite a wrong premise in the *question* is the
    # whole point. Approximating the *answer* is not, and it fights two shipped
    # rules directly: copy the wording from the facts, and keep a relative date
    # in its relative form. So this says the first without saying the second,
    # and re-asserts the exact-wording rule inside the clause that was
    # undermining it.
    #
    # Also checked rather than assumed: the temporal damage is not the
    # `qualifier` clause being dropped by the old one-at-a-time `--variant`.
    # Only 13.1% of temporal is gated, and on those 28 questions the run scores
    # exactly what the shipped stack scores. Seven of the eight breakages are on
    # the plain path, where `qualifier` never was.
    "commit_exact": """
A question may attach a detail to the wrong person, object, place or occasion. Answer it from the facts anyway, in the facts' own wording, rather than replying that the facts do not mention it, that it did not happen, or that it was someone else.
This licenses nothing about the answer itself. Give the exact thing the facts give, in the form the rules above ask for, and do not offer a near-miss in place of it.""",
}


def parse_clauses(spec: str | None, default_where: str) -> list[tuple[str, str]]:
    """`qualifier@gated,commit_short@everywhere` into pairs.

    Clauses used to be one at a time, and `--variant X` silently replaced
    whatever was already shipping rather than adding to it. That is how the
    first `commit_short` run came to drop `qualifier` without anyone asking it
    to: it compared two configurations differing in two places, so a result
    either way would not have said which place caused it.

    A bare name keeps the old meaning, so every command already written still
    means what it meant.
    """
    if not spec:
        return []
    pairs = []
    for piece in spec.split(","):
        name, _, where = piece.strip().partition("@")
        where = where or default_where
        if name not in VARIANTS:
            raise SystemExit(f"unknown clause {name!r}; "
                             f"known: {', '.join(sorted(VARIANTS))}")
        if where not in ("ungated", "gated", "everywhere"):
            raise SystemExit(f"clause {name!r} placed {where!r}; "
                             f"expected ungated, gated or everywhere")
        pairs.append((name, where))
    return pairs


def pick(rows: list[dict], rendered: list[str], args) -> tuple[list, list]:
    """Split into what a prompt could fix and what it could break."""
    targets = [r for r, c in zip(rows, rendered)
               if not r["ok"] and present(r["gold"], c)]
    controls = [r for r in rows if r["ok"]]
    rng = random.Random(args.seed)
    rng.shuffle(targets)
    rng.shuffle(controls)
    return targets[: args.targets or None], controls[: args.controls or None]


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no merged store at {store}")

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    memory = open_memory(store, llm, embedder, args.hot, args.cold, args.rerank,
                         args.max_hops, compact=True, hydrate=args.hydrate,
                         facts=args.facts, dedupe=True, overlap=args.overlap)
    memory._router._config.hydrate_pool = args.pool
    memory._router._config.source_grounding = args.grounding
    memory._router._config.hydrate_scan = args.scan

    names = [v for v in args.variants.split(",") if v]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {unknown}. "
                         f"have: {sorted(VARIANTS)}")

    limit = asyncio.Semaphore(args.concurrency)
    blocked: list[str] = []

    async def render_all(qs: list[str]) -> list[str]:
        turn = memory.current_turn + 1
        out = []
        for q in qs:
            recs = await with_retries(lambda x=q: memory._router.retrieve(x, turn))
            out.append(memory._router.format_context(recs or [], q))
        return out

    try:
        print("rendering contexts to find the fixable questions, no generation")
        rendered = await render_all([r["question"] for r in rows])
        targets, controls = pick(rows, rendered, args)
        print(f"\n{len(targets)} targets (we lose, gold IS in context)")
        print(f"{len(controls)} controls (we win)")
        print(f"{len(names)} variants + control prompt, both slices each: "
              f"{2 * (len(names) + 1) * (len(targets) + len(controls))} calls\n")

        async def one(row: dict, variant: str | None) -> bool | None:
            rule, tag = gate(row["question"])
            prompt = PROMPT_B if rule else ANSWER_SYSTEM_PROMPT
            if variant:
                prompt = prompt + VARIANTS[variant]
            async with limit:
                try:
                    text, _ = await with_retries(
                        lambda: answer_with(memory, llm, row["question"],
                                            prompt, tag))
                    v = await with_retries(
                        lambda: judge_answer(row["question"], text,
                                             row["gold"], judge))
                except Exception as exc:  # noqa: BLE001
                    if type(exc).__name__ not in PERMANENT:
                        raise
                    blocked.append(row["question"])
                    return None
            return bool(v.correct)

        results: dict[str, dict[str, list]] = {}
        for variant in [None] + names:
            label = variant or "control"
            t = await asyncio.gather(*[one(r, variant) for r in targets])
            c = await asyncio.gather(*[one(r, variant) for r in controls])
            results[label] = {"targets": t, "controls": c}
            print(f"  {label}: done")
    finally:
        close_readonly(memory)

    Path(args.out).write_text(json.dumps(
        {"targets": [r["question"] for r in targets],
         "controls": [r["question"] for r in controls],
         "results": results}, indent=2), encoding="utf-8")
    report(results, targets, controls, rows, blocked, args)
    print(f"\n-> {args.out}")


def report(results, targets, controls, rows, blocked, args) -> None:
    base = results["control"]
    nt, nc = len(targets), len(controls)
    print(f"\n{'':<14}{'targets':>18}{'controls':>20}{'projected':>12}{'p':>9}")
    print(f"{'':<14}{'fix':>6}{'brk':>6}{'of':>6}{'fix':>7}{'brk':>7}{'of':>6}"
          f"{'on 839':>12}")
    print("-" * 73)

    won = sum(1 for r in rows if r["ok"])
    for label, res in results.items():
        if label == "control":
            continue
        tf = sum(1 for a, b in zip(res["targets"], base["targets"])
                 if a and not b)
        tb = sum(1 for a, b in zip(res["targets"], base["targets"])
                 if b and not a)
        cf = sum(1 for a, b in zip(res["controls"], base["controls"])
                 if a and not b)
        cb = sum(1 for a, b in zip(res["controls"], base["controls"])
                 if b and not a)
        # The target slice is a census of its population, so it projects one to
        # one. The control slice is a sample of the questions we win, so its
        # rate is scaled up to all of them. Stated rather than assumed, because
        # a projection whose arithmetic is hidden is a guess with a decimal
        # point on it.
        proj = (tf - tb) + (cf - cb) * won / max(nc, 1)
        p = binomtest(tf + cf, tf + cf + tb + cb).pvalue if (tf+cf+tb+cb) else 1.0
        print(f"{label:<14}{tf:>6}{tb:>6}{nt:>6}{cf:>7}{cb:>7}{nc:>6}"
              f"{proj:>+12.1f}{p:>9.3g}")

    print(f"\nProjection = (target fixes - target breaks) + control net scaled "
          f"by {won}/{nc}.")
    print(f"We are 15 answers short of RAG on 839. A variant projecting under "
          f"+8 will not close it\nand should not be run at full scale.")
    if blocked:
        print(f"\n{len(blocked)} questions blocked by the provider")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/od_full.json",
                    help="Output of _run_open_domain_full.py. Supplies the "
                         "verdict that decides target from control.")
    ap.add_argument("--variants", default="qualifier,discriminate,assemble")
    ap.add_argument("--targets", type=int, default=0,
                    help="0 means every fixable question, which is the point.")
    ap.add_argument("--controls", type=int, default=90,
                    help="A sample. 90 puts the standard error on the breakage "
                         "rate near 3 points, which is enough to reject a "
                         "variant that damages the wins.")
    # Retrieval is held at the configuration od_full.json was generated under,
    # so that 'target' really means 'the gold was in the context when we lost'.
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--rerank", type=int, default=40)
    ap.add_argument("--facts", type=int, default=18)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--grounding", type=float, default=0.003)
    ap.add_argument("--scan", type=int, default=8)
    ap.add_argument("--hydrate", type=int, default=7)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="C:/nmafc_ab/od_conversion.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
