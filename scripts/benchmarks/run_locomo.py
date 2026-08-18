"""Run the LoCoMo benchmark suite.

Evaluates 3 arms on the LoCoMo dataset using both F1 scoring
and LLM-as-judge. Follows the evaluation protocol from
Maharana et al. (2024).

Usage:
    python -m scripts.benchmarks.run_locomo \
        --provider bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
    python -m scripts.benchmarks.run_locomo \
        --arms neuromorphic --conversations 2
    python -m scripts.benchmarks.run_locomo \
        --categories 1,2 --output results/locomo_quick/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Windows terminals default to cp1252, which cannot encode the box-drawing
# characters in the progress output (nor much of the dataset text). A run that
# dies hours in on a print() is not an acceptable failure mode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from nmafc.schemas.memory import DecayConfig
from nmafc.integration.factory import (
    create_embedding_provider,
    create_llm_provider,
)

from . import ingest_checkpoint
from .arms.base import BenchmarkArm
from .arms.neuromorphic import NeuromorphicArm
from .arms.neuromorphic_tuned import NeuromorphicTunedArm
from .arms.rag import RagArm
from .arms.raw_llm import RawLLMArm
from .arms.stateful_nodecay import StatefulNoDecayArm
from .datasets.locomo_loader import (
    CATEGORY_NAMES,
    LoCoMoConversation,
    get_dataset_stats,
    load_locomo,
)
from .evaluation.f1_score import compute_f1
from .evaluation.llm_judge import judge_batch
from .evaluation.metrics import BenchmarkResult
from .resilience import (
    RateLimiter,
    RetryingEmbeddingProvider,
    RetryingLLMProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark suite")
    parser.add_argument(
        "--arms",
        default="raw,rag,neuromorphic,neuromorphic_tuned",
        help="Comma-separated arm names to evaluate",
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=None,
        help="Number of conversations to evaluate (default: all)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="QA categories to include: 1,2,3,4,5 (default: all)",
    )
    parser.add_argument(
        "--output",
        default="scripts/benchmarks/results/locomo/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("NMAFC_BENCH_PROVIDER", "ollama/llama3.2"),
        help="LLM provider string (e.g. bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0)",
    )
    parser.add_argument(
        "--judge",
        default=os.environ.get("NMAFC_BENCH_JUDGE"),
        help=(
            "Judge provider string (env: NMAFC_BENCH_JUDGE). Defaults to "
            "--provider, which makes the answering model grade its own output; "
            "prefer a different model family so the judge is independent of "
            "every arm."
        ),
    )
    parser.add_argument(
        "--embedding",
        default=os.environ.get("NMAFC_BENCH_EMBEDDING", "ollama/nomic-embed-text"),
        help="Embedding provider string",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation (F1 only)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--ingest-checkpoint-every",
        type=int,
        default=25,
        help=(
            "Save ingestion progress every N exchanges so an interrupted run "
            "resumes mid-conversation instead of re-ingesting it. 0 disables. "
            "The run-level --checkpoint only records whole finished "
            "conversations, so without this a drop at exchange 168 of 211 "
            "throws away every one of those 168 LLM extraction calls -- which "
            "is what a connection drop at 00:21 on 2026-08-18 cost. Stores go "
            "under <output>/stores/, which also makes them reusable by the "
            "_ab_* harnesses instead of being lost in a temp directory."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("NMAFC_BENCH_CONCURRENCY", "8")),
        help=(
            "Conversations evaluated in parallel. Measured quota is 500k TPM / "
            "500 RPM, which at observed latency saturates near 9; higher values "
            "mostly produce 429s."
        ),
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=int(os.environ.get("NMAFC_BENCH_JUDGE_CONCURRENCY", "12")),
        help="Parallel judge calls (judging runs as a separate phase)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap questions per conversation (pilot runs only; omit for the full set)",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=(
            int(os.environ["NMAFC_BENCH_MAX_HOPS"])
            if os.environ.get("NMAFC_BENCH_MAX_HOPS")
            else None
        ),
        help=(
            "Spreading Activation depth for the memory arms (env: "
            "NMAFC_BENCH_MAX_HOPS). 0 disables graph traversal. Omit to use the "
            "DecayConfig default. Set explicitly so the results file records "
            "which setting produced the numbers -- a paired A/B measured 2-hop "
            "traversal costing 3.5x context for no accuracy gain, so this is a "
            "parameter a reader will want to check rather than assume."
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=(
            float(os.environ["NMAFC_BENCH_BETA"])
            if os.environ.get("NMAFC_BENCH_BETA")
            else None
        ),
        help=(
            "Clustering protection strength for decay, in [0, 1) (env: "
            "NMAFC_BENCH_BETA). Scales each record's decay rate by "
            "(1 - beta * C), where C is the local clustering coefficient of its "
            "entity in the related_entities graph. 0 disables the mechanism and "
            "reproduces plain type-and-consolidation decay exactly, which makes "
            "it the ablation control for any run with beta > 0. Omit to use the "
            "DecayConfig default."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("NMAFC_BENCH_LOG_FILE"),
        help=(
            "Mirror all console output to this file, timestamped and flushed "
            "per line (env: NMAFC_BENCH_LOG_FILE). Appends, so a resumed run "
            "extends the same log rather than truncating it."
        ),
    )
    return parser.parse_args()


class _TeeStream:
    """Mirror stdout to a log file, timestamped and flushed on every line.

    A full run takes hours and prints nothing for minutes at a time during
    ingestion, so the only way to tell "working" from "hung" is a file that
    updates live. Buffering would defeat that entirely, hence the flush after
    every write on both sinks -- the cost is negligible next to an API call.

    Timestamps are prefixed per line rather than per write() because print()
    emits the text and the newline as separate calls; `_at_line_start` tracks
    which of the two we are in so a stamp never lands mid-line.
    """

    def __init__(self, stream, path: Path) -> None:
        self._stream = stream
        self._file = open(path, "a", encoding="utf-8", errors="replace")
        self._at_line_start = True

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._stream.flush()
        for i, part in enumerate(text.split("\n")):
            if i:
                self._file.write("\n")
                self._at_line_start = True
            if not part:
                continue
            if self._at_line_start:
                self._file.write(time.strftime("[%H:%M:%S] "))
                self._at_line_start = False
            self._file.write(part)
        self._file.flush()
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __getattr__(self, name):  # isatty, encoding, fileno, ...
        return getattr(self._stream, name)


async def _heartbeat(arm: BenchmarkArm, conv_id: str, total: int, every: int = 60):
    """Report progress while a conversation ingests.

    Ingestion is one LLM extraction call per exchange and can run 20+ minutes on
    a 345-exchange conversation while printing nothing. Rather than thread a
    callback through all five arms, this polls state the memory arms already
    maintain: `current_turn` counts processed exchanges and Hot RAM's count is
    the facts extracted so far. Arms without a memory (raw, rag) still get an
    elapsed-time line, which is enough to distinguish slow from stuck.
    """
    start = time.perf_counter()
    while True:
        await asyncio.sleep(every)
        mins = (time.perf_counter() - start) / 60
        memory = getattr(arm, "_memory", None)
        if memory is None:
            print(f"    [{arm.name}/{conv_id}] ingesting… {mins:.1f} min elapsed")
            continue
        try:
            done, records = memory.current_turn, memory._hot.count()
        except Exception:  # noqa: BLE001 - a heartbeat must never kill a run
            continue
        rate = done / mins if mins else 0.0
        eta = (total - done) / rate if rate else float("inf")
        print(
            f"    [{arm.name}/{conv_id}] ingesting {done}/{total} exchanges, "
            f"{records} facts, {mins:.1f} min elapsed, ~{eta:.1f} min left"
        )


def _metrics_snapshot(template: BenchmarkArm, workers: list[BenchmarkArm]) -> dict:
    """Combine the restored and in-flight metric samples into one record.

    The template carries whatever a previous run banked; the workers carry what
    this run has measured so far. Neither alone is the arm's cost, which is why
    a resumed arm reported 0 ms latency and 0 tokens: the workers had done no
    answering, and nothing had ever put the earlier samples back.
    """
    lat, prompt, completion, ctx = [], [], [], []
    for source in [template, *workers]:
        lat.extend(source.metrics._latencies)
        prompt.extend(source.metrics._prompt_tokens)
        completion.extend(source.metrics._completion_tokens)
        ctx.extend(source.metrics._context_tokens)
    return {
        "latencies": lat,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "context_tokens": ctx,
    }


def _write_checkpoint(
    output_dir: Path,
    arm_name: str,
    by_conversation: dict[str, list[dict]],
    metrics: dict | None = None,
) -> None:
    """Persist per-conversation results so a killed run can resume."""
    path = output_dir / f"checkpoint_{arm_name}.json"
    payload = {
        "arm": arm_name,
        "completed_conversations": len(by_conversation),
        # Per-call cost samples, not just their averages: p50/p95 latency cannot
        # be recomputed from a mean, and the comparison table reports both.
        "metrics": metrics or {},
        "by_conversation": by_conversation,
    }
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)  # atomic, so a crash mid-write cannot corrupt the file


def _rebuild_metrics_from_rows(rows: list[dict]) -> dict:
    """Recover what the per-question rows already record: latency and context.

    Checkpoints written before metrics were persisted still carry `latency_ms`
    and `context_tokens` on every row, so those two are exactly recoverable.
    Prompt and completion counts were only ever held in memory, so they stay
    empty here rather than being guessed -- an arm restored from an old
    checkpoint will report its true latency and context size, and a total token
    count of zero, which is visibly missing rather than quietly wrong.
    """
    return {
        "latencies": [r["latency_ms"] for r in rows if r.get("latency_ms") is not None],
        "prompt_tokens": [],
        "completion_tokens": [],
        "context_tokens": [
            r["context_tokens"] for r in rows if r.get("context_tokens") is not None
        ],
    }


def _load_checkpoints(
    output_dir: Path, arm_names: list[str]
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Read prior checkpoints so completed conversations are not re-run.

    Returns (completed conversations per arm, restored cost metrics per arm).
    """
    completed: dict[str, dict] = {}
    restored_metrics: dict[str, dict] = {}
    for name in arm_names:
        arm_label = {"raw": "raw_llm", "stateful": "stateful_nodecay"}.get(name, name)
        path = output_dir / f"checkpoint_{arm_label}.json"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            by_conv = data.get("by_conversation", {})
            # Drop conversations recorded with zero questions. A conversation
            # always has questions, so an empty list can only mean it died
            # mid-run and was checkpointed anyway (a bug fixed in the worker
            # above). Treating those as complete is what silently shrinks an
            # arm's question set; dropping them here repairs checkpoints
            # already written by the old code, so a resume retries them.
            dead = [cid for cid, rows in by_conv.items() if not rows]
            for cid in dead:
                del by_conv[cid]
            if dead:
                print(f"  {path.name}: discarding {len(dead)} empty "
                      f"conversation(s) {dead} — will re-run")
            completed[arm_label] = by_conv
            rows = [r for v in by_conv.values() for r in v]
            metrics = data.get("metrics") or {}
            if not metrics.get("latencies") and rows:
                metrics = _rebuild_metrics_from_rows(rows)
                print(f"  {path.name}: cost metrics rebuilt from "
                      f"{len(rows)} rows (checkpoint predates metric storage; "
                      f"token totals unavailable)")
            restored_metrics[arm_label] = metrics
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: ignoring unreadable checkpoint {path.name}: {exc}")
    return completed, restored_metrics


def build_decay_overrides(args) -> dict:
    """Collect run-level DecayConfig settings from the CLI.

    Only keys the user actually passed are included, so anything left off the
    command line keeps its DecayConfig default rather than being pinned to a
    value the runner invented.
    """
    overrides: dict = {}
    if args.max_hops is not None:
        overrides["max_hops"] = args.max_hops
    if args.beta is not None:
        overrides["beta"] = args.beta
    return overrides


def make_arm(
    name: str, llm_provider, embedding_provider, decay_overrides: dict | None = None
) -> BenchmarkArm | None:
    """Build one arm instance. Each parallel worker needs its own.

    Arms hold live memory state and call reset() between conversations, so a
    single shared instance cannot be evaluated on two conversations at once.

    `decay_overrides` applies to the neuromorphic arms only. The raw, rag and
    stateful arms either have no decay engine or deliberately pin their own
    settings, and passing run-level decay knobs to them would change what those
    baselines mean.
    """
    if name == "raw":
        return RawLLMArm(llm_provider=llm_provider)
    if name == "rag":
        return RagArm(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
        )
    if name == "stateful":
        return StatefulNoDecayArm(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
        )
    if name == "neuromorphic":
        return NeuromorphicArm(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            decay_overrides=decay_overrides,
        )
    if name == "neuromorphic_tuned":
        return NeuromorphicTunedArm(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            decay_overrides=decay_overrides,
        )
    print(f"WARNING: Unknown arm '{name}', skipping")
    return None


def create_arms(
    arm_names: list[str],
    llm_provider,
    embedding_provider,
    decay_overrides: dict | None = None,
) -> list[BenchmarkArm]:
    """Instantiate requested benchmark arms."""
    arms = [
        make_arm(n, llm_provider, embedding_provider, decay_overrides)
        for n in arm_names
    ]
    return [a for a in arms if a is not None]


def merge_metrics(target: BenchmarkArm, workers: list[BenchmarkArm]) -> None:
    """Fold per-worker metrics back into one arm-level view."""
    for w in workers:
        target.metrics._latencies.extend(w.metrics._latencies)
        target.metrics._prompt_tokens.extend(w.metrics._prompt_tokens)
        target.metrics._completion_tokens.extend(w.metrics._completion_tokens)
        target.metrics._context_tokens.extend(w.metrics._context_tokens)
        target.metrics.hot_storage_records += w.metrics.hot_storage_records
        target.metrics.cold_storage_events += w.metrics.cold_storage_events
    # Storage counts are per-conversation; report the average footprint rather
    # than a sum across every conversation the worker happened to handle.
    n = max(1, len(workers))
    target.metrics.hot_storage_records //= n
    target.metrics.cold_storage_events //= n


async def evaluate_arm_on_conversation(
    arm: BenchmarkArm,
    conv: LoCoMoConversation,
    categories: list[int] | None,
    max_questions: int | None = None,
    store_root: Path | None = None,
    ingest_every: int = 0,
    decay_overrides: dict | None = None,
) -> list[dict]:
    """Run one arm on one conversation, return per-question results.

    Questions stay sequential within a conversation on purpose: retrieval
    reinforces memory (LTP) for the stateful arms, so answering concurrently
    against one arm instance would let questions interleave and perturb each
    other's state. Parallelism comes from running conversations side by side,
    each on its own arm instance.

    Judging is deliberately not done here — it runs as a separate batched
    phase so judge calls can be parallelized independently of answering.
    """
    # Mid-conversation ingestion checkpointing. Off unless the runner supplies a
    # store root and an interval, so every existing call path keeps the old
    # behaviour of wiping and re-ingesting. See scripts/benchmarks/ingest_checkpoint.
    resume = bool(store_root) and ingest_every > 0 and arm.supports_ingest_resume
    start_at, on_progress, store_dir = 0, None, None
    if resume:
        store_dir = ingest_checkpoint.store_dir_for(store_root, arm.name, conv.sample_id)
        store_dir.mkdir(parents=True, exist_ok=True)
        fp = ingest_checkpoint.fingerprint(arm.name, decay_overrides)
        start_at = arm.prepare_store(str(store_dir), conv.sample_id, fp)
        if start_at:
            print(f"    [{arm.name}/{conv.sample_id}] resuming ingestion at "
                  f"exchange {start_at}")

        def on_progress(done: int, turn: int, _d=store_dir, _c=conv.sample_id,
                        _f=fp, _n=ingest_every) -> None:
            if done % _n == 0:
                ingest_checkpoint.write(
                    _d,
                    ingest_checkpoint.IngestState(
                        conversation_id=_c, exchanges_done=done,
                        turn=turn, fingerprint=_f,
                    ),
                )
    else:
        arm.reset()

    # Ingest all sessions
    history = conv.get_flat_history()
    beat = asyncio.create_task(
        _heartbeat(arm, conv.sample_id, total=sum(1 for t in history if t.get("role") == "user"))
    )
    try:
        await arm.ingest_conversation(history, start_at=start_at, on_progress=on_progress)
    finally:
        beat.cancel()
    # Clear only after ingestion completes. A surviving state file over a
    # finished store would make the next run resume from `exchanges_done` and
    # re-ingest the tail into a store that already has it, duplicating facts
    # rather than saving work.
    if resume:
        ingest_checkpoint.clear(store_dir)
    print(f"    [{arm.name}/{conv.sample_id}] ingested, answering "
          f"{len(conv.qa_pairs)} questions")

    results = []
    qa_pairs = conv.qa_pairs
    if categories:
        qa_pairs = [qa for qa in qa_pairs if qa.category in categories]
    if max_questions is not None:
        qa_pairs = qa_pairs[:max_questions]

    # Answering is the long phase for the stateless arms -- 1,986 questions run
    # strictly sequentially within a conversation (see docstring), so without
    # this the log goes silent for 8-15 minutes at a stretch and there is no way
    # to tell a slow run from a hung one. Every 25 is frequent enough to show
    # movement and rare enough not to bury the errors in the same file.
    answer_start = time.perf_counter()
    for i, qa in enumerate(qa_pairs):
        if i and i % 25 == 0:
            mins = (time.perf_counter() - answer_start) / 60
            eta = mins / i * (len(qa_pairs) - i)
            print(f"    [{arm.name}/{conv.sample_id}] answered {i}/{len(qa_pairs)}, "
                  f"{mins:.1f} min elapsed, ~{eta:.1f} min left")
        try:
            response = await arm.answer_question(qa.question)
        except Exception as e:
            print(f"    ERROR on Q{i}: {e}")
            results.append({
                "question": qa.question,
                "gold_answer": qa.answer,
                "predicted": "",
                "category": qa.category_name,
                "f1": 0.0,
                "judge_correct": None,
                "error": str(e),
            })
            continue

        results.append({
            "question": qa.question,
            "gold_answer": qa.answer,
            "predicted": response.answer,
            "category": qa.category_name,
            "f1": compute_f1(response.answer, qa.answer),
            "judge_correct": None,
            "latency_ms": response.latency_ms,
            "context_tokens": response.context_tokens,
        })

    arm.update_storage_metrics()
    return results


async def run_judge_phase(
    results: list[dict],
    judge_provider,
    concurrency: int,
) -> None:
    """Judge every unjudged answer in parallel, updating rows in place.

    Rows restored from a checkpoint already carry their verdict, so skipping
    them is what makes `--checkpoint resume` cheap. Without the second clause a
    resume re-graded every completed arm from scratch: 1,986 answers at judge
    throughput is ~40 minutes per arm, all of it spent recomputing verdicts
    already on disk, and all of it charged against the same quota the arms
    still waiting to run need.

    `is None` rather than a falsy check: `judge_correct=False` is a real
    verdict ("the judge marked this wrong") and must not be re-run, while None
    means either never judged or the judge call itself failed -- both worth
    another attempt.
    """
    scorable = [
        r for r in results
        if not r.get("error") and r.get("judge_correct") is None
    ]
    if not scorable:
        print("    judging: all answers already graded, skipping")
        return

    print(f"    judging {len(scorable)} answers (concurrency {concurrency})...")
    verdicts = await judge_batch(
        items=[
            {
                "question": r["question"],
                "predicted": r["predicted"],
                "gold_answer": r["gold_answer"],
            }
            for r in scorable
        ],
        judge_provider=judge_provider,
        concurrency=concurrency,
        on_progress=lambda done, total: print(f"      judged {done}/{total}"),
    )
    for row, verdict in zip(scorable, verdicts):
        row["judge_correct"] = verdict.correct if verdict is not None else None


async def run_benchmark(args: argparse.Namespace) -> None:
    """Main benchmark loop."""
    print("=" * 70)
    print("LoCoMo Benchmark Suite — NMAFC")
    print("=" * 70)

    # Load dataset
    print("\n[1/4] Loading LoCoMo dataset from HuggingFace...")
    conversations = load_locomo()
    stats = get_dataset_stats(conversations)
    print(f"  Loaded {stats['conversations']} conversations, "
          f"{stats['total_qa_pairs']} QA pairs, "
          f"{stats['total_turns']} turns")

    if args.conversations:
        conversations = conversations[:args.conversations]
        print(f"  (limited to first {args.conversations} conversations)")

    categories = None
    if args.categories:
        categories = [int(c) for c in args.categories.split(",")]
        print(f"  Categories: {[CATEGORY_NAMES[c] for c in categories]}")

    # Initialize providers
    print("\n[2/4] Initializing providers...")
    print(f"  LLM: {args.provider}")
    print(f"  Embedding: {args.embedding}")
    # One limiter shared by every arm and by the judge: the quota is
    # per-deployment, so concurrent arms draw from the same bucket.
    limiter = RateLimiter()
    llm_provider = RetryingLLMProvider(create_llm_provider(args.provider), limiter)
    embedding_provider = RetryingEmbeddingProvider(
        create_embedding_provider(args.embedding),
        limiter=None if args.embedding.startswith(("ollama/", "fastembed/")) else limiter,
    )
    print(f"  Concurrency: {args.concurrency} conversations, "
          f"{args.judge_concurrency} judges")

    judge_provider = None
    if not args.skip_judge:
        judge_str = args.judge or args.provider
        print(f"  Judge: {judge_str}")
        if judge_str == args.provider:
            print("  WARNING: judge and answering model are identical — the model "
                  "is grading its own output. Set NMAFC_BENCH_JUDGE to a "
                  "different model family for an independent judge.")
        # A judge on a different deployment has its own quota, so it must not
        # queue behind the answering model's limiter; sharing one bucket across
        # two providers throttles calls that were never rate-limited.
        judge_limiter = limiter if judge_str == args.provider else RateLimiter()
        judge_provider = RetryingLLMProvider(
            create_llm_provider(judge_str), judge_limiter
        )

    # Create arms
    arm_names = [n.strip() for n in args.arms.split(",")]
    print(f"\n[3/4] Arms: {arm_names}")

    # Run evaluation
    print("\n[4/4] Running evaluation...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed, restored_metrics = (
        _load_checkpoints(output_dir, arm_names) if args.checkpoint else ({}, {})
    )

    all_results: dict[str, BenchmarkResult] = {}
    decay_overrides = build_decay_overrides(args)

    for arm_name in arm_names:
        template = make_arm(arm_name, llm_provider, embedding_provider, decay_overrides)
        if template is None:
            continue

        print(f"\n{'─' * 50}")
        print(f"  ARM: {template.name}")
        print(f"{'─' * 50}")

        done_conversations = completed.get(template.name, {})
        pending = [c for c in conversations if c.sample_id not in done_conversations]
        all_question_results = [
            row for c in conversations if c.sample_id in done_conversations
            for row in done_conversations[c.sample_id]
        ]
        if all_question_results:
            print(f"  resuming: {len(done_conversations)} conversations already "
                  f"complete, {len(pending)} to go")
            # Seed the template with the banked cost samples. merge_metrics
            # later adds this run's workers on top, so the arm ends up
            # reporting every question it answered, not only the ones answered
            # after the last restart.
            banked = restored_metrics.get(template.name) or {}
            template.metrics._latencies.extend(banked.get("latencies", []))
            template.metrics._prompt_tokens.extend(banked.get("prompt_tokens", []))
            template.metrics._completion_tokens.extend(
                banked.get("completion_tokens", [])
            )
            template.metrics._context_tokens.extend(banked.get("context_tokens", []))

        # One arm instance per worker slot, reused across conversations.
        n_workers = max(1, min(args.concurrency, len(pending)))
        workers = [
            make_arm(arm_name, llm_provider, embedding_provider, decay_overrides)
            for _ in range(n_workers)
        ]
        queue: asyncio.Queue = asyncio.Queue()
        for ci, conv in enumerate(pending):
            queue.put_nowait((ci, conv))

        results_lock = asyncio.Lock()
        progress = {"done": 0}

        async def worker(slot: int) -> None:
            arm = workers[slot]
            while True:
                try:
                    ci, conv = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    rows = await evaluate_arm_on_conversation(
                        arm=arm, conv=conv, categories=categories,
                        max_questions=args.max_questions,
                        store_root=output_dir,
                        ingest_every=args.ingest_checkpoint_every,
                        decay_overrides=decay_overrides,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad conversation
                    # Do NOT checkpoint a conversation that died. The previous
                    # version stored rows=[] under its id, which marked it
                    # *complete with zero questions*: its results vanished from
                    # the arm, and because _load_checkpoints skips any id it
                    # finds, a resumed run skipped it forever rather than
                    # retrying it. A transient network blip at 23:55 took out
                    # four conversations that way -- 864 of 1,986 questions --
                    # after two and a half hours of ingestion each.
                    #
                    # Leaving it unrecorded means resume treats it as pending,
                    # which is the correct reading of "we never got an answer".
                    print(f"    ERROR conversation {conv.sample_id}: "
                          f"{type(exc).__name__}: {exc} — NOT checkpointed, "
                          f"will be retried on resume")
                    async with results_lock:
                        progress["done"] += 1
                    continue

                async with results_lock:
                    all_question_results.extend(rows)
                    done_conversations[conv.sample_id] = rows
                    progress["done"] += 1
                    print(f"  [{template.name}] conv {progress['done']}/{len(pending)} "
                          f"({conv.sample_id}) — {len(rows)} answered")
                    _write_checkpoint(
                        output_dir, template.name, done_conversations,
                        _metrics_snapshot(template, workers),
                    )

        start_arm = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(n_workers)))
        print(f"  answering finished in {(time.perf_counter() - start_arm) / 60:.1f} min")

        if not args.skip_judge and judge_provider:
            await run_judge_phase(
                all_question_results, judge_provider, args.judge_concurrency
            )
            # Judge verdicts were written into the same row objects the
            # checkpoint holds, so re-dump to persist them.
            _write_checkpoint(
                output_dir, template.name, done_conversations,
                _metrics_snapshot(template, workers),
            )

        # Must come after the writes above: the snapshot sums template and
        # workers, and merging folds the workers *into* the template, so a
        # snapshot taken afterwards would count this run's samples twice.
        merge_metrics(template, workers)

        # Aggregate results
        benchmark_result = _aggregate_results(template, all_question_results)
        all_results[template.name] = benchmark_result

        if isinstance(llm_provider, RetryingLLMProvider):
            print(f"  api: {llm_provider.stats()} | limiter: {limiter.stats()}")

        # Print summary for this arm
        _print_arm_summary(benchmark_result)

    # Save final results
    final_output = {
        "metadata": {
            "dataset": "locomo",
            "provider": args.provider,
            "embedding": args.embedding,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "conversations_evaluated": len(conversations),
            "questions_evaluated": sum(
                len(r.question_results) for r in all_results.values()
            ) // max(1, len(all_results)),
            # Recorded because it changes the result materially: a paired A/B
            # measured 2-hop traversal costing 3.5x context for no accuracy
            # gain, so a reader comparing two results files needs to know which
            # setting produced each. None means the DecayConfig default.
            "max_hops": args.max_hops,
            # Likewise: beta > 0 slows decay for facts in densely interlinked
            # neighbourhoods, so a run with it on is not comparable to one
            # without. None means the DecayConfig default, which is 0 (off).
            "beta": args.beta,
            "theta": DecayConfig().theta,
            "judge": (args.judge or args.provider) if not args.skip_judge else None,
        },
        "results": {name: result.to_dict() for name, result in all_results.items()},
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 70}")

    # Print comparison table
    _print_comparison_table(all_results)


def _aggregate_results(arm: BenchmarkArm, question_results: list[dict]) -> BenchmarkResult:
    """Aggregate per-question results into a BenchmarkResult."""
    result = BenchmarkResult(
        arm_name=arm.name,
        dataset="locomo",
        metrics=arm.metrics,
        question_results=question_results,
    )

    if not question_results:
        return result

    # Overall F1
    f1_scores = [r["f1"] for r in question_results]
    result.overall_f1 = sum(f1_scores) / len(f1_scores)

    # Overall judge accuracy
    judged = [r for r in question_results if r.get("judge_correct") is not None]
    if judged:
        result.overall_accuracy = sum(1 for r in judged if r["judge_correct"]) / len(judged)

    # Per-category breakdown
    categories_seen: dict[str, list[dict]] = {}
    for r in question_results:
        cat = r["category"]
        categories_seen.setdefault(cat, []).append(r)

    for cat, cat_results in categories_seen.items():
        result.f1_by_category[cat] = sum(r["f1"] for r in cat_results) / len(cat_results)
        cat_judged = [r for r in cat_results if r.get("judge_correct") is not None]
        if cat_judged:
            result.accuracy_by_category[cat] = (
                sum(1 for r in cat_judged if r["judge_correct"]) / len(cat_judged)
            )

    return result


def _print_arm_summary(result: BenchmarkResult) -> None:
    """Print a summary for one arm."""
    print(f"\n  Summary for {result.arm_name}:")
    print(f"    Overall F1:       {result.overall_f1:.3f}")
    if result.overall_accuracy > 0:
        print(f"    Judge Accuracy:   {result.overall_accuracy:.3f}")
    print("    Per-category F1:")
    for cat, f1 in sorted(result.f1_by_category.items()):
        acc = result.accuracy_by_category.get(cat, 0)
        print(f"      {cat:20s}: F1={f1:.3f}  Acc={acc:.3f}")
    if result.metrics:
        print(f"    Avg latency:      {result.metrics.avg_latency_ms:.0f} ms")
        print(f"    Avg context:      {result.metrics.avg_context_tokens:.0f} tokens")
        print(f"    Total tokens:     {result.metrics.total_tokens:,}")


def _print_comparison_table(results: dict[str, BenchmarkResult]) -> None:
    """Print a comparison table across all arms."""
    print(f"\n{'=' * 70}")
    print(f"{'COMPARISON TABLE':^70}")
    print(f"{'=' * 70}")
    print(f"{'Arm':<20} {'F1':>8} {'Accuracy':>10} {'Avg Ctx':>10} {'Latency':>10} {'Tokens':>12}")
    print(f"{'─' * 70}")
    for name, r in results.items():
        ctx = f"{r.metrics.avg_context_tokens:.0f}" if r.metrics else "—"
        lat = f"{r.metrics.avg_latency_ms:.0f}ms" if r.metrics else "—"
        tok = f"{r.metrics.total_tokens:,}" if r.metrics else "—"
        print(f"{name:<20} {r.overall_f1:>8.3f} {r.overall_accuracy:>10.3f} "
              f"{ctx:>10} {lat:>10} {tok:>12}")
    print(f"{'=' * 70}")


def main() -> None:
    args = parse_args()
    tee = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tee = _TeeStream(sys.stdout, log_path)
        sys.stdout = tee
        # stderr too: a traceback that only reached the console would be lost
        # the moment the terminal scrolled, which is exactly the failure a log
        # file exists to capture.
        sys.stderr = tee
        print(f"live log: {log_path.resolve()}")
    try:
        asyncio.run(run_benchmark(args))
    finally:
        if tee is not None:
            sys.stdout, sys.stderr = tee._stream, tee._stream
            tee.close()


if __name__ == "__main__":
    main()
