"""One arm, every open-domain question, paired against RAG's saved answers.

Every open-domain measurement so far has been a 250-question A/B, and 250 is no
longer enough. Arm A was byte-identical in two paid runs on 7 September and
scored 82.3% then 81.1% on the same 249 questions. That is 1.2 points of
run-to-run drift out of generation and judging alone, and every effect being
chased is 1.6 to 2.6 points. Single 250-question runs cannot resolve them, so
a fifth one would buy another number inside the same band.

This is the run that can. All 839 open-domain questions, one arm, paired
question by question against the RAG answers already saved in `haystack.json`.

**Why single-arm is legitimate here and was not before.** A/B was the right
shape while the thing under test was our own prompt, because our arms drift
against each other and only same-session generation cancels it. RAG is not our
arm. Its retrieval, its prompt and its context are untouched by everything in
this session, its verdict on all 839 is saved with the question it answered, and
so the pairing is exact without spending a call on it. What single-arm cannot
cancel is drift in *our* generation, and the answer to that is n: 839 against
249 cuts the sampling term by about 1.8x, from roughly 1.2 points to 0.7, which
is finally smaller than the difference being measured.

**Questions come from the saved file, not from the loader.** Iterating
`haystack.json` guarantees that every question scored here has a RAG answer to
pair with and that no question is dropped or duplicated by a reshuffle. It also
means `ours_ok` is in hand for free, so the report can show movement against the
shipped stack as gross counts rather than a net.

Three numbers come out of it and they answer different questions:

    us vs RAG, paired      the headline. Did the deficit close, and by how much
    us vs shipped, gross   what this configuration bought, +N/-M, never a net
    context tokens         the constraint. Ours against RAG's, measured

Read-only: reinforcement writes are deferred and the handle is dropped rather
than flushed, so scoring 839 questions does not mutate the store it scored.

Usage:
    python -u scripts/benchmarks/_run_open_domain_full.py --sample 8 --out C:/nmafc_ab/od_smoke.json
    python -u scripts/benchmarks/_run_open_domain_full.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
from nmafc.integration.list_shape import wants_list  # noqa: E402
from nmafc.integration.factory import (  # noqa: E402
    create_embedding_provider,
    create_llm_provider,
)

from scripts.benchmarks._ab_answer_type import PROMPT_B, answer_with  # noqa: E402
from scripts.benchmarks._ab_conversion import (  # noqa: E402
    VARIANTS,
    parse_clauses,
)
from scripts.benchmarks._ab_budget import (  # noqa: E402
    PERMANENT,
    close_readonly,
    open_memory,
    with_retries,
)
from scripts.benchmarks.arms.neuromorphic_tuned import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
)
from scripts.benchmarks.evaluation.llm_judge import judge_answer  # noqa: E402


def _list_gate_filter(targets: list[dict], mode: str) -> list[dict]:
    """Keep only the questions the list gate does, or does not, fire on.

    Built to price width per question rather than per category. Unlike a
    category this is a property of the question string, so a policy built on it
    is one a deployed system can actually apply. It runs the firing half wide
    and splices the quiet half back in from a run that left it alone.

    The answer it produced is no. MEASURED over 322 firing questions: wide
    50.9% against the shipped 49.4%, +17/-12, p=0.458, for 324 tokens each.
    Single-hop, the category the gate was built off, goes backwards at -1.4.

    Kept because the split it performs is still the right instrument and the
    shape it isolates is real -- we score 50.2% where the gate fires against
    66.0% where it does not. Anything attacking that gap wants this flag.

    An earlier read of the same thing on 108 open-domain questions had it at
    +5.6, and that number is why this function exists. It was nine coin flips
    landing one way. Do not size an experiment in this benchmark below ~800
    questions.

    Fire rate varies enormously by category, half of single-hop against 0.6% of
    temporal, so this runs after the category filter and never instead of it.
    """
    if mode == "all":
        return targets
    want = mode == "fires"
    return [r for r in targets if wants_list(r["question"]) == want]


async def run(args: argparse.Namespace) -> None:
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])
    store = Path(args.run) / "stores" / f"{args.arm}__{args.store}"
    if not store.is_dir():
        raise SystemExit(f"no merged store at {store}")

    saved = json.loads(Path(args.results).read_text(encoding="utf-8"))
    targets = [r for r in saved if r["category"] == args.category]
    if args.gate_only != "all":
        # A prompt clause aimed at one path changes nothing on the other, so
        # regenerating the other path only adds drift to the comparison: two
        # runs of near-identical configurations churn 61 verdicts out of 828.
        # Restricting to the path under test buys a real in-session control for
        # a fraction of the questions. Splice the untouched path back in from
        # the full run when reporting a headline.
        want = args.gate_only == "gated"
        targets = [r for r in targets if bool(gate(r["question"])[0]) == want]
    targets = _list_gate_filter(targets, args.list_gate)
    if args.sample:
        targets = targets[: args.sample]

    memory = open_memory(store, llm, embedder, args.hot, args.cold, args.rerank,
                         args.max_hops, compact=True, hydrate=args.hydrate,
                         facts=args.facts, dedupe=args.dedupe,
                         overlap=args.overlap)
    memory._router._config.hydrate_pool = args.pool
    memory._router._config.source_grounding = args.grounding
    memory._router._config.hydrate_scan = args.scan
    memory._router._config.scan_semantic = args.semantic
    memory._router._config.scan_semantic_floor = args.semantic_floor

    # Resume is keyed on the question, not on the configuration, so a file left
    # by an earlier run resumes silently under whatever settings are current --
    # and if that file is complete, the run answers nothing and reports the old
    # configuration's numbers as the new one's. The defaults have changed twice
    # since `od_full.json` was written, so this is not hypothetical. Fingerprint
    # every setting that changes an answer and refuse to mix two of them.
    fingerprint = {k: getattr(args, k) for k in (
        "arm", "store", "category", "rerank", "facts", "pool", "grounding",
        "scan", "hydrate", "overlap", "dedupe", "semantic", "semantic_floor",
        "hot", "cold", "max_hops", "variant", "variant_everywhere",
        "variant_where", "gate_only", "list_gate")}
    out = Path(args.out)
    stamp = out.with_suffix(".config.json")
    rows: list[dict] = []
    if out.is_file() and not args.restart:
        previous = (json.loads(stamp.read_text(encoding="utf-8"))
                    if stamp.is_file() else None)
        if previous != fingerprint:
            if previous is None:
                what = ("was written before configurations were recorded, so "
                        "there is no way to tell whether it matches")
            else:
                differs = sorted(k for k in fingerprint
                                 if previous.get(k) != fingerprint[k])
                what = ("was written under a different configuration: "
                        + ", ".join(f"{k} {previous.get(k)!r} -> "
                                    f"{fingerprint[k]!r}" for k in differs))
            raise SystemExit(
                f"{out} {what}.\nResuming would mix two configurations in one "
                f"results file. Use --restart to overwrite, or --out to write "
                f"somewhere new.")
        rows = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(rows)} already answered\n")
    stamp.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    done = {(r["conv"], r["question"]) for r in rows}
    pending = [r for r in targets if (r["conv"], r["question"]) not in done]
    blocked: list[tuple[str, str]] = []

    print(f"store: {store}")
    print(f"rerank {args.rerank}, {args.facts} facts printed, hydrate "
          f"{args.hydrate} from pool {args.pool} scan {args.scan}, grounding "
          f"{args.grounding}, overlap {args.overlap}, "
          f"{'deduped headers' if args.dedupe else 'headers as stored'}, "
          f"gated answer-type, "
          f"{args.variant or 'no conversion rule'}"
          f"{' on ' + ('everywhere' if args.variant_everywhere else args.variant_where) if args.variant else ''}"
          f"{', ' + args.gate_only + ' questions only' if args.gate_only != 'all' else ''}")
    # Semantic turn ranking is the largest single gain in the candidate and it
    # fails silently on a store with no index -- the run would simply be the old
    # configuration wearing the new numbers. Counted rather than assumed.
    indexed = len(getattr(memory._router._cold, "all_turn_vectors", dict)())
    print(f"scan_semantic {args.semantic} floor {args.semantic_floor}, "
          f"{indexed} turns indexed"
          f"{'  <- NO INDEX, semantic ranking is inert' if not indexed else ''}")
    print(f"{len(pending)} {args.category} questions to answer, "
          f"{len(targets)} paired in total\n")

    limit = asyncio.Semaphore(args.concurrency)

    async def one(saved_row: dict):
        question, gold = saved_row["question"], saved_row["gold"]
        rule, tag = gate(question)
        prompt = PROMPT_B if rule else ANSWER_SYSTEM_PROMPT
        # Appended after the length rules rather than spliced into them,
        # because "give every part" only means the right thing once "be brief"
        # has been read.
        #
        # Not appended to the gated prompt, and that is the correction to the
        # earlier attempt. `assemble` went on both, so an instruction to give
        # every part of the answer landed on the questions asking for one name,
        # one date, one title -- exactly where a second candidate turns a right
        # answer into a wrong one, and exactly where we were already at 93.2%.
        # The measured split across all 839:
        #
        #     a length rule fires    126 questions   ours 93.2%   RAG 90.5%
        #     no rule fires          713 questions   ours 77.1%   RAG 79.5%
        #
        # 84 of the 97 questions RAG wins and we lose are in the second group,
        # where the gold averages 5.2 words and we answer in 4.5. So the whole
        # deficit is in the ungated path and none of the upside is in the gated
        # one; putting the clause on both spends the gain on breaking what
        # works. `--variant-everywhere` restores the old behaviour so the two
        # can be compared rather than assumed.
        #
        # `--variant-where gated` is the opposite targeting, and it has its own
        # reason. The 13 gated losses left under the candidate are the wrong one
        # of several similar candidates -- The Last Devil to Die for The Great
        # Gatsby, The Alchemist for a Rothfuss novel, Star Wars for Harry Potter
        # -- which is discrimination, not completeness, and is what `qualifier`
        # and `discriminate` were written for. Neither has been tried there.
        for clause, where in args.clauses:
            if where == "everywhere" or (where == "gated") == bool(rule):
                prompt = prompt + VARIANTS[clause]
        async with limit:
            try:
                text, chars = await with_retries(
                    lambda: answer_with(memory, llm, question, prompt, tag))
                verdict = await with_retries(
                    lambda: judge_answer(question, text, gold, judge))
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ not in PERMANENT:
                    raise
                blocked.append((saved_row["conv"], question))
                return None
        return {
            "conv": saved_row["conv"], "question": question,
            "category": saved_row["category"], "gold": gold, "tag": tag,
            "pred": text, "chars": chars, "ok": bool(verdict.correct),
            # Carried through so the report needs no second file and a resumed
            # run cannot pair against a different snapshot than it started on.
            "rag_pred": saved_row["rag_pred"], "rag_ok": bool(saved_row["rag_ok"]),
            "rag_chars": saved_row["rag_chars"],
            "shipped_ok": bool(saved_row["ours_ok"]),
        }

    try:
        tasks = [one(r) for r in pending]
        for start in range(0, len(tasks), args.checkpoint_every):
            batch = await asyncio.gather(
                *tasks[start:start + args.checkpoint_every])
            rows.extend(r for r in batch if r)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"  {len(rows)}/{len(targets)} answered")
    finally:
        close_readonly(memory)
    await finish(rows, targets, blocked, out, args)


async def per_conversation(args: argparse.Namespace) -> None:
    """The same thing against ten stores instead of one merged one.

    Real LoCoMo keeps a store per conversation, and a saved row carries the
    conversation it came from. Our arm is regenerated and RAG's saved answer is
    reused, which is the whole economy of this runner: a prompt clause changes
    only our side, so paying for RAG's side again would buy nothing but drift.
    """
    llm = create_llm_provider(os.environ["NMAFC_BENCH_PROVIDER"])
    judge = create_llm_provider(
        os.environ.get("NMAFC_BENCH_JUDGE", os.environ["NMAFC_BENCH_PROVIDER"])
    )
    embedder = create_embedding_provider(os.environ["NMAFC_BENCH_EMBEDDING"])

    saved = json.loads(Path(args.results).read_text(encoding="utf-8"))
    # A clause lives in the system prompt, so a deployed arm attaches it to
    # every question or to none. Measuring it on the category it was written
    # for answers half the question and the wrong half: the other half is what
    # it costs on the categories already being won. "all" is how that is asked,
    # and a comma list is how the cheaper version is: the category a clause is
    # aimed at plus the category it is suspected of damaging is most of the
    # answer for a fraction of the questions.
    wanted = set(args.category.split(","))
    targets = [r for r in saved
               if args.category == "all" or r["category"] in wanted]
    if args.gate_only != "all":
        want = args.gate_only == "gated"
        targets = [r for r in targets if bool(gate(r["question"])[0]) == want]
    targets = _list_gate_filter(targets, args.list_gate)
    if args.sample:
        targets = targets[: args.sample]

    fingerprint = {k: getattr(args, k) for k in (
        "arm", "store", "category", "rerank", "facts", "pool", "grounding",
        "scan", "hydrate", "overlap", "dedupe", "semantic", "semantic_floor",
        "hot", "cold", "max_hops", "variant", "variant_everywhere",
        "variant_where", "gate_only", "list_gate")}
    out = Path(args.out)
    stamp = out.with_suffix(".config.json")
    rows = []
    if out.is_file() and not args.restart:
        previous = (json.loads(stamp.read_text(encoding="utf-8"))
                    if stamp.is_file() else None)
        if previous != fingerprint:
            raise SystemExit(f"{out} was written under a different "
                             f"configuration. Use --restart or a new --out.")
        rows = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(rows)} already answered\n")
    stamp.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    done = {(r["conv"], r["question"]) for r in rows}
    blocked: list[tuple[str, str]] = []

    groups: dict[str, list[dict]] = {}
    for row in targets:
        if (row["conv"], row["question"]) not in done:
            groups.setdefault(row["conv"], []).append(row)

    print(f"per-conversation over {len(groups)} stores under {args.run}")
    print(f"rerank {args.rerank}, {args.facts} facts printed, hydrate "
          f"{args.hydrate} from pool {args.pool} scan {args.scan}, "
          f"{args.variant or 'no conversion rule'}"
          f"{' on ' + args.variant_where if args.variant else ''}")
    print(f"{sum(len(g) for g in groups.values())} {args.category} questions "
          f"to answer, {len(targets)} paired in total\n")

    limit = asyncio.Semaphore(args.concurrency)
    for name, group in groups.items():
        store = Path(args.run) / "stores" / f"{args.arm}__{name}"
        memory = open_memory(store, llm, embedder, args.hot, args.cold,
                             args.rerank, args.max_hops, compact=True,
                             hydrate=args.hydrate, facts=args.facts,
                             dedupe=args.dedupe, overlap=args.overlap)
        memory._router._config.hydrate_pool = args.pool
        memory._router._config.source_grounding = args.grounding
        memory._router._config.hydrate_scan = args.scan
        memory._router._config.scan_semantic = args.semantic
        memory._router._config.scan_semantic_floor = args.semantic_floor

        async def one(saved_row: dict):
            question, gold = saved_row["question"], saved_row["gold"]
            rule, tag = gate(question)
            prompt = PROMPT_B if rule else ANSWER_SYSTEM_PROMPT
            for clause, where in args.clauses:
                if where == "everywhere" or (where == "gated") == bool(rule):
                    prompt = prompt + VARIANTS[clause]
            async with limit:
                try:
                    text, chars = await with_retries(
                        lambda: answer_with(memory, llm, question, prompt, tag))
                    verdict = await with_retries(
                        lambda: judge_answer(question, text, gold, judge))
                except Exception as exc:  # noqa: BLE001
                    if type(exc).__name__ not in PERMANENT:
                        raise
                    blocked.append((saved_row["conv"], question))
                    return None
            return {
                "conv": saved_row["conv"], "question": question,
                "category": saved_row["category"], "gold": gold, "tag": tag,
                "pred": text, "chars": chars, "ok": bool(verdict.correct),
                "rag_pred": saved_row["rag_pred"],
                "rag_ok": bool(saved_row["rag_ok"]),
                "rag_chars": saved_row["rag_chars"],
                "shipped_ok": bool(saved_row["ours_ok"]),
            }

        try:
            tasks = [one(r) for r in group]
            for start in range(0, len(tasks), args.checkpoint_every):
                batch = await asyncio.gather(
                    *tasks[start:start + args.checkpoint_every])
                rows.extend(r for r in batch if r)
                out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                print(f"  {len(rows)}/{len(targets)} answered")
        finally:
            close_readonly(memory)

    await finish(rows, targets, blocked, out, args)


async def finish(rows, targets, blocked, out, args) -> None:
    """Write the results and print the report, however they were produced."""
    if not rows:
        print("nothing measured")
        return
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report(rows, blocked, args)
    print(f"\n-> {args.out}")


def paired(rows: list[dict], mine: str, theirs: str, label: str) -> None:
    """One paired comparison, printed with the counts the p-value came from.

    Accuracy alone hides the thing that matters: 82% against 81% is a different
    result when it is 40 won and 32 lost than when it is 4 won and 0 lost. The
    win/lose counts are the sample the sign test actually runs on, so they are
    printed next to it rather than left implicit.
    """
    n = len(rows)
    a = sum(r[mine] for r in rows)
    b = sum(r[theirs] for r in rows)
    win = sum(1 for r in rows if r[mine] and not r[theirs])
    lose = sum(1 for r in rows if r[theirs] and not r[mine])
    p = binomtest(win, win + lose).pvalue if win + lose else 1.0
    print(f"  {label:<22}{n:>5}{a / n * 100:>9.1f}%{b / n * 100:>9.1f}%"
          f"{(a - b) / n * 100:>+8.1f}{win:>7}{lose:>7}{p:>9.3g}")


def report(rows: list[dict], blocked, args) -> None:
    print(f"\n{'':<22}{'n':>5}{'ours':>10}{'theirs':>10}{'diff':>8}"
          f"{'win':>7}{'lose':>7}{'p':>9}")
    print("-" * 78)
    paired(rows, "ok", "rag_ok", "vs RAG")
    paired(rows, "ok", "shipped_ok", "vs shipped stack")

    # With `--category all` the two lines above are a benchmark-wide average and
    # an average is exactly what hides the thing worth knowing. A clause aimed
    # at adversarial can pay for itself there and quietly cost four points of
    # temporal, and the overall line would read as a modest win either way.
    seen = {r["category"] for r in rows}
    if len(seen) > 1:
        print()
        for cat in sorted(seen):
            group = [r for r in rows if r["category"] == cat]
            paired(group, "ok", "rag_ok", f"  {cat} vs RAG")
            paired(group, "ok", "shipped_ok", f"  {cat} vs shipped")

    ours = sum(r["chars"] for r in rows) / len(rows) / 4
    rag = sum(r["rag_chars"] for r in rows) / len(rows) / 4
    widest = max(r["chars"] for r in rows) / 4
    verdict = "under" if ours <= args.ceiling else "OVER"
    print(f"\ncontext: ours mean {ours:.0f}t, widest single question "
          f"{widest:.0f}t  ({verdict} the {args.ceiling}t ceiling)")
    print(f"         RAG mean {rag:.0f}t, so we spend "
          f"{100 * (1 - ours / rag):.0f}% less")

    # Gross, never net. A configuration that fixes 30 and breaks 30 reads as
    # zero on the accuracy line and is not the same result as one that touches
    # nothing, because the 30 it broke are 30 regressions to explain.
    fix = [r for r in rows if r["ok"] and not r["shipped_ok"]]
    brk = [r for r in rows if r["shipped_ok"] and not r["ok"]]
    print(f"\nagainst the shipped stack: +{len(fix)} fixed, -{len(brk)} broken")

    # The questions where RAG is right and we are not are the entire remaining
    # deficit, so they are the only ones worth reading by hand.
    still = [r for r in rows if r["rag_ok"] and not r["ok"]]
    print(f"\n{len(still)} questions RAG answers and we do not:")
    for r in still[: args.show]:
        print(f"\n  Q    {r['question']}")
        print(f"  gold {r['gold']}")
        print(f"  ours {r['pred']}")
        print(f"  RAG  {r['rag_pred']}")
    if blocked:
        print(f"\n{len(blocked)} questions blocked by the provider")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="C:/nmafc_ab/haystack")
    ap.add_argument("--arm", default="neuromorphic_tuned")
    ap.add_argument("--store", default="haystack")
    ap.add_argument("--results", default="C:/nmafc_ab/haystack.json",
                    help="The saved paired run. Supplies the questions, RAG's "
                         "answer to each, and the shipped stack's verdict.")
    ap.add_argument("--category", default="open-domain")
    ap.add_argument("--sample", type=int, default=0,
                    help="0 means every question in the category, which is the "
                         "entire point of this harness. Use a small number to "
                         "smoke-test the plumbing before spending.")
    # The candidate, off the free sweeps of 7-8 September: 1,026 tokens, gold
    # reaching 49.2% of losses and 85.0% of wins with no win displaced. Chosen
    # over the 968t arm that was paid for, which reached the same losses and
    # only 83.3% of wins -- the win side is roughly 194 of 249 questions, so
    # holding it is worth about 3.5x whatever the loss side gives up.
    ap.add_argument("--hot", type=int, default=10)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--rerank", type=int, default=40)
    # The settled candidate, screened on the 178 real losses and 178 wins:
    # 49.4% loss reach, 88.8% win reach, 958 tokens. Best measured in this
    # project, and it beats the old 1,219-token high of 47.8% at a fifth of the
    # cost, so it is not a token trade. Facts back at 20 because the semantic
    # ranker freed the tokens the cut to 18 was paying for, and that cut was
    # shown by the paid run to cost accuracy the reach screen could not see.
    ap.add_argument("--facts", type=int, default=20)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--grounding", type=float, default=0.03)
    ap.add_argument("--scan", type=int, default=8)
    ap.add_argument("--hydrate", type=int, default=6)
    # Rank hydrated turns by meaning as well as by words. Needs the store to
    # carry a turn index, written by `_build_turn_index.py`; without one this is
    # silently inert, so the config line printed at startup names the store.
    ap.add_argument("--semantic", type=float, default=12.0)
    ap.add_argument("--semantic-floor", type=float, default=0.30)
    ap.add_argument("--dedupe", action="store_true", default=True)
    ap.add_argument("--overlap", type=float, default=0.8)
    ap.add_argument("--ceiling", type=int, default=1026,
                    help="Our own budget. RAG spends about 1,468 on this "
                         "category; the standing constraint is 1,000 nominal, "
                         "relaxed to the candidate's screened width.")
    ap.add_argument("--variant", default=None,
                    help="Conversion clauses from _ab_conversion.py, "
                         "comma-separated, each optionally as name@where. A "
                         "bare name takes --variant-where, so every command "
                         "already written still means what it meant. Omit for "
                         "the shipped prompt.")
    ap.add_argument("--gate-only", default="all",
                    choices=("all", "gated", "ungated"),
                    help="only answer questions where a length rule does or "
                         "does not fire")
    ap.add_argument("--list-gate", default="all",
                    choices=("all", "fires", "quiet"),
                    help="only answer questions the list-shape gate does or "
                         "does not fire on. Width pays 5.6 points where it "
                         "fires and 1.0 where it does not, so this is how the "
                         "expensive half gets priced on its own.")
    ap.add_argument("--variant-where", default="ungated",
                    choices=("ungated", "gated", "everywhere"),
                    help="which path the variant clause is appended to")
    ap.add_argument("--variant-everywhere", action="store_true",
                    help="Also apply the variant to questions a length rule "
                         "already gates. This is what the earlier attempt did "
                         "and it is why it broke more than it fixed")
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="C:/nmafc_ab/od_full.json")
    args = ap.parse_args()
    # `--store per-conv` reads the conversation off each saved row and opens
    # the matching store, which is what real LoCoMo needs. Anything else names
    # one merged store, which is what the haystack needs.
    # `--variant-everywhere` predates placements and still wins when given, so
    # the two ways of saying the same thing cannot disagree.
    args.clauses = parse_clauses(
        args.variant,
        "everywhere" if args.variant_everywhere else args.variant_where)
    asyncio.run(per_conversation(args) if args.store == "per-conv"
                else run(args))


if __name__ == "__main__":
    main()
