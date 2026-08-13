# NMAFC — Neuromorphic Memory Architecture for Conversational AI

A biologically-inspired stateful memory system for LLM agents. NMAFC gives conversational AI the ability to remember, forget, and prioritize information the way biological memory does — using exponential decay, spaced repetition, override suppression, and active pruning.

Unlike context-window stuffing or naive vector stores that grow without bound, NMAFC maintains a bounded, high-signal memory that improves with use. Frequently accessed facts become permanent. Contradicted facts are immediately suppressed. Stale information naturally decays away.

## Why NMAFC

| Problem | Current Approaches | NMAFC Solution |
|---------|-------------------|----------------|
| Context windows overflow | Truncate oldest messages | Hot RAM with bounded record count via cognitive decay |
| Contradictions persist | Old facts coexist with new ones | Override detection + gamma suppression (instant eviction) |
| Everything treated equally | Flat vector stores | Three-tier typing: CoreAnchor (permanent), ActiveContext (moderate decay), EphemeralState (aggressive decay) |
| No concept of importance | Retrieval count ignored | Spaced repetition — each retrieval strengthens retention (LTP) |
| Retrieval misses related facts | Single-hop vector search | Spreading Activation graph traversal (multi-hop entity linking) |
| No recoverability | Mutable state only | Dual-track: Hot RAM (fast, mutable) + Cold ROM (append-only event log, full rollback) |
| Expensive per-turn overhead | Separate extraction + response calls | Single LLM call with tool-use for simultaneous response + extraction |

## Core Mechanisms

### 1. Three-Tier Memory Classification

Every extracted fact is classified by the LLM into one of three tiers, each with distinct decay behavior:

| Tier | Decay Rate (lambda) | Half-life | Examples |
|------|---------------------|-----------|----------|
| **CoreAnchor** | 0.0 (never decays) | Infinite | Name, allergies, identity, relationships |
| **ActiveContext** | 0.05 per turn | ~14 turns | Current goals, schedules, projects |
| **EphemeralState** | 0.69 per turn | ~1 turn | Mood, passing comments, transient state |

### 2. Cognitive Decay (Ebbinghaus Forgetting Curve)

Each memory's synaptic weight decays exponentially over time:

```
w(t) = w(t_0) * e^(-lambda * delta_t)
```

Where `delta_t = current_turn - last_reinforced_turn` and `lambda` is the tier-specific decay rate modified by the consolidation coefficient.

### 3. Spaced Repetition (Long-Term Potentiation)

When a memory is retrieved during a query, it receives LTP reinforcement:

1. Weight resets to 1.0 (full strength)
2. Consolidation index `k` increments
3. Future decay rate slows: `effective_lambda = lambda_base * e^(-eta * k)` where `eta = 0.15`

A fact retrieved 10 times retains 80% weight after 20 turns vs. 37% for a never-retrieved fact. This naturally surfaces important information.

### 4. Override Detection & Suppression

When the LLM detects a contradicting fact (e.g., "I moved to Berlin" contradicts "I live in Paris"):

1. New fact specifies `overrides_entity` pointing to the old record
2. Old record's weight is multiplied by `gamma = 0.1` (instant suppression)
3. Next prune cycle evicts the old record (weight 0.1 <= prune threshold 0.1)

Zero hallucination for contradictions — suppressed facts cannot be retrieved.

### 5. Spreading Activation (Graph Traversal)

Retrieval goes beyond single-hop vector search:

1. **Hop 0:** Vector similarity search returns `top_k=10` results
2. **Hop 1:** Each result's `related_entities` are fetched from Hot RAM
3. **Hop 2:** Their `related_entities` are fetched (BFS continues to `max_hops=2`)

Example: Query "spouse" -> `spouse_james` (vector hit) -> `related_entities: ["brother_david"]` -> `brother_david` -> `related_entities: ["job_pilot"]` -> `job_pilot`.

This surfaces contextually related facts that pure vector similarity would miss.

### 6. REM Sleep Consolidation

Every 5 turns (configurable), a consolidation pass runs:

- **Elevation:** ActiveContext records with `consolidation_index >= 10` are promoted to CoreAnchor (permanent protection). Frequently-accessed facts earn immortality.
- **Dead pointer cleanup:** Removes `related_entities` references to entities that no longer exist in Hot RAM.

### 7. Dual-Track Storage

| Layer | Technology | Remote Option | Purpose | Mutability |
|-------|-----------|---------------|---------|------------|
| **Hot RAM** | LanceDB (embedded vector DB) | S3-compatible storage | Fast retrieval, vector search, weight updates | Mutable (decay, reinforce, delete) |
| **Cold ROM** | SQLite (WAL mode, FTS5) | PostgreSQL (tsvector + GIN) | Complete event log, keyword fallback, rollback source | Append-only |

Both layers support `agent_id` + `conversation_id` scoping for multi-tenant and multi-conversation isolation. Cold ROM enables full state reconstruction at any point in time via event replay.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │           NeuromorphicMemory                 │
                    │              (wrapper.py)                    │
                    └────────┬──────────┬──────────┬──────────────┘
                             │          │          │
                    ┌────────▼──┐  ┌────▼────┐  ┌─▼──────────────┐
                    │  Extract  │  │  Query  │  │    Engine       │
                    │  (LLM +   │  │  Router │  │  ┌───────────┐  │
                    │   Tool)   │  │         │  │  │   Decay   │  │
                    └───────────┘  │ Vector  │  │  │  Reinforce│  │
                                   │ Search  │  │  │   Prune   │  │
                                   │ + Graph │  │  │  Consol.  │  │
                                   │ + Cold  │  │  │  Rollback │  │
                                   │ Fallback│  │  └───────────┘  │
                                   └────┬────┘  └────────────────┘
                                        │
                         ┌──────────────┼──────────────┐
                         │              │              │
                    ┌────▼────┐    ┌────▼────┐    ┌───▼───┐
                    │ Hot RAM │    │Cold ROM │    │Embedder│
                    │(LanceDB)│    │(SQLite) │    │        │
                    └─────────┘    └─────────┘    └────────┘
```

## Processing Pipeline

Each call to `process_turn(user_msg)` executes this sequence:

```
1. Increment turn counter
2. Retrieve context (vector search + spreading activation + cold fallback)
3. Format retrieved memories as context string
4. LLM call with tool-use → simultaneous response + state extraction
5. For each extracted update:
   a. Log to Cold ROM (append-only)
   b. Detect overrides → suppress contradicted records (w *= gamma)
   c. Embed new fact → upsert to Hot RAM (weight=1.0)
6. Decay all mutable records: w(t) = w(t0) * e^(-lambda * dt)
7. Prune: delete records where weight <= w_prune
8. Every N turns: REM consolidation (elevation + dead pointer cleanup)
```

## Installation

```bash
# Core only
pip install nmafc

# With LLM providers (OpenAI + Anthropic SDKs)
pip install nmafc[llm]

# With AWS Bedrock support
pip install nmafc[aws]

# With PostgreSQL remote storage
pip install nmafc[postgres]

# With benchmark suite
pip install nmafc[bench]

# Everything (LLM + AWS + Postgres + Benchmarks)
pip install nmafc[all]

# Development (from source)
git clone https://github.com/blok-hamster/nmafc.git
cd nmafc
uv pip install -e ".[all]"
uv pip install --group dev
```

## Quick Start

```python
import asyncio
from nmafc.wrapper import NeuromorphicMemory
from nmafc.integration.factory import create_llm_provider, create_embedding_provider

async def main():
    # Create providers
    llm = create_llm_provider("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    embedder = create_embedding_provider("ollama/nomic-embed-text")

    # Initialize memory
    mem = NeuromorphicMemory(llm_provider=llm, embedding_provider=embedder)

    # Process conversation turns
    response = await mem.process_turn(
        user_msg="My name is Joshua and I'm allergic to shellfish.",
        conversation_history=[
            {"role": "user", "content": "My name is Joshua and I'm allergic to shellfish."}
        ],
    )
    print(response)

    # Memory now contains:
    #   [CoreAnchor] user_name: "Joshua"
    #   [CoreAnchor] user_allergy: "Allergic to shellfish"

    # Later turns can reference these facts
    response = await mem.process_turn(
        user_msg="What should I avoid eating?",
        conversation_history=[
            {"role": "user", "content": "What should I avoid eating?"}
        ],
    )
    print(response)  # Will reference shellfish allergy from memory

    # Check memory state
    stats = mem.get_hot_stats()
    print(f"Records: {stats['count']}, Types: {stats['types']}")

    mem.close()

asyncio.run(main())
```

### Using Config File

```python
from nmafc.wrapper import NeuromorphicMemory
from nmafc.storage.config import NMafcConfig

# From TOML config
mem = NeuromorphicMemory.from_config(config_path="configs/default.toml")

# Or from environment variables (NMAFC_LLM_PROVIDER_MODEL, NMAFC_EMBEDDING_PROVIDER_MODEL)
mem = NeuromorphicMemory.from_config()
```

### Override Detection

```python
# Turn 1: Store a fact
await mem.process_turn("I live in Paris.")
# Hot RAM: [ActiveContext] user_location: "Lives in Paris"

# Turn 5: Contradiction arrives
await mem.process_turn("I just moved to Berlin last week.")
# Hot RAM: [ActiveContext] user_location: "Moved to Berlin" (weight=1.0)
# The old "Paris" record is suppressed (w *= 0.1) and pruned

# Query will ONLY return Berlin, never Paris
```

### Manual Memory Injection

```python
from nmafc.schemas.memory import MemoryStateUpdate

# Inject facts without an LLM call (useful for bootstrapping)
await mem.ingest_updates([
    MemoryStateUpdate(
        entity_name="user_name",
        fact_content="User's name is Joshua",
        memory_type="CoreAnchor",
    ),
    MemoryStateUpdate(
        entity_name="user_project",
        fact_content="Currently working on NMAFC framework",
        memory_type="ActiveContext",
        related_entities=["user_role"],
    ),
])
```

### Rollback

```python
# Restore memory state to how it was at turn 10
restored_count = await mem.rollback(to_turn=10)
print(f"Restored {restored_count} records from Cold ROM")
```

## Configuration

### Default Configuration (`configs/default.toml`)

```toml
[decay]
lambda_core_anchor = 0.0       # CoreAnchor never decays
lambda_active_context = 0.05   # Moderate decay (~14 turn half-life)
lambda_ephemeral = 0.69        # Aggressive decay (~1 turn half-life)
eta = 0.15                     # Consolidation constant (higher = faster LTP effect)
gamma = 0.1                    # Override suppression multiplier
w_prune = 0.1                  # Eviction threshold (records at or below are deleted)
theta = 0.75                   # Retrieval similarity threshold
top_k = 10                     # Vector search result count
max_hops = 2                   # Spreading Activation graph depth
fallback_keyword_limit = 20    # Cold ROM FTS5 result limit
auto_consolidate_turns = 5     # REM sleep interval
time_unit = "turns"            # Decay time unit

[storage]
hot_uri = "./data/lancedb"     # LanceDB path (supports s3://)
cold_uri = "./data/cold.db"    # SQLite path
embedding_dim = 1536           # Auto-detected on init

[llm]
provider_model = "openai/gpt-4o-mini"

[embedding]
provider_model = "openai/text-embedding-3-small"
```

### Environment Variable Overrides

| Variable | Overrides | Description |
|----------|-----------|-------------|
| `NMAFC_HOT_URI` | `storage.hot_uri` | Hot RAM storage path (local or `s3://`) |
| `NMAFC_COLD_URI` | `storage.cold_uri` | Cold ROM path (local `.db` or `postgresql://`) |
| `NMAFC_AGENT_ID` | `storage.agent_id` | Agent/tenant namespace for isolation |
| `NMAFC_CONVERSATION_ID` | `storage.conversation_id` | Conversation thread isolation |
| `NMAFC_LLM_PROVIDER_MODEL` | `llm.provider_model` | Default LLM provider |
| `NMAFC_EMBEDDING_PROVIDER_MODEL` | `embedding.provider_model` | Default embedding provider |
| `NMAFC_EMBEDDING_DIM` | `storage.embedding_dim` | Vector width of the hot-storage column. **Set this whenever your embedding model is not 1536-dim** (e.g. `768` for `nomic-embed-text`) |
| `NMAFC_EMBED_PROBE_TIMEOUT` | — | Seconds to wait on the startup embedding-dimension probe before falling back to the configured value (default `30`) |
| `NMAFC_LLM_TEMPERATURE` | — | Sampling temperature sent with every completion. Unset by default; set to `0` for reproducible benchmark runs |

> **Set `NMAFC_EMBEDDING_DIM` if you use a non-1536-dim embedding model.** When it is unset, `NeuromorphicMemory` probes the provider at startup to detect the width. That probe runs the async embedder on a second event loop, so if the same provider instance has already been used on the calling loop, its connection pool is bound there and the probe blocks until the timeout expires. Setting the dimension explicitly skips the probe entirely.

## Supported Providers

### LLM Providers

| Provider | Format | API Key Env |
|----------|--------|-------------|
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| AWS Bedrock | `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0` | `ANTHROPIC_API_KEY_BEDROCK` |
| Azure OpenAI | `azure/DeepSeek-V4-Pro` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| Groq | `groq/llama-3.1-70b-versatile` | `GROQ_API_KEY` |
| OpenRouter | `openrouter/anthropic/claude-sonnet-4-20250514` | `OPENROUTER_API_KEY` |
| Together | `together/meta-llama/Llama-3-70b-chat-hf` | `TOGETHER_API_KEY` |
| Ollama | `ollama/llama3.2` | None (local) |
| LM Studio | `lmstudio/local-model` | None (local) |
| vLLM | `vllm/meta-llama/Llama-3-8b` | None (local) |

### Embedding Providers

| Provider | Format | Dimensions |
|----------|--------|-----------|
| OpenAI | `openai/text-embedding-3-small` | 1536 |
| OpenAI | `openai/text-embedding-3-large` | 3072 |
| Azure OpenAI | `azure/text-embedding-3-small` | 1536 |
| Bedrock Titan | `bedrock/amazon.titan-embed-text-v2:0` | 1024 |
| Ollama | `ollama/nomic-embed-text` | 768 |
| Ollama | `ollama/mxbai-embed-large` | 1024 |
| Together | `together/togethercomputer/m2-bert-80M-8k-retrieval` | 768 |
| FastEmbed (ONNX) | Built-in `BAAI/bge-small-en-v1.5` | 384 |

Embedding dimension is auto-detected on initialization — no manual configuration needed.

## Credential Setup

Copy the example and fill in keys for providers you use:

```bash
cp .env.example .env
```

The framework loads `.env` automatically via `python-dotenv`. See [.env.example](.env.example) for all supported variables.

## Remote Storage

By default NMAFC stores everything locally (LanceDB directory + SQLite file). For production deployments where multiple instances need shared memory, both layers support remote backends:

### Hot RAM → S3

LanceDB natively supports S3 URIs. No code changes — just set the URI:

```bash
NMAFC_HOT_URI=s3://your-bucket/nmafc/hot_lancedb
```

Requires AWS credentials (`AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`) in environment. Works with any S3-compatible store (AWS S3, MinIO, R2).

**Self-hosted (MinIO):**

```bash
NMAFC_HOT_URI=s3://nmafc-bucket/hot_lancedb
AWS_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_ALLOW_HTTP=true
```

`AWS_ENDPOINT_URL` redirects all S3 calls to your MinIO instance. `AWS_ALLOW_HTTP=true` is needed if not behind TLS. Same pattern works for Ceph, Wasabi, or Cloudflare R2.

### Cold ROM → PostgreSQL / CockroachDB

Replace SQLite with a managed PostgreSQL-compatible database for multi-writer access and remote persistence:

```bash
# Managed PostgreSQL (Supabase, Neon, RDS, AlloyDB)
NMAFC_COLD_URI=postgresql://user:pass@host:5432/nmafc

# CockroachDB (distributed, self-hosted or Cockroach Cloud)
NMAFC_COLD_URI=postgresql://root@cockroach-host:26257/nmafc?sslmode=verify-full
```

Uses `tsvector` + GIN index for full-text search (replaces SQLite FTS5). Both PostgreSQL and CockroachDB (v23.1+) fully support all required features — `TSVECTOR`, `to_tsvector()`, `plainto_tsquery()`, `@@` operator, `ts_rank()`, GIN indexes, and `GENERATED ALWAYS AS ... STORED` computed columns.

Install the optional dependency:

```bash
pip install nmafc[postgres]
```

### Deployment Examples

| Environment | `NMAFC_HOT_URI` | `NMAFC_COLD_URI` |
|-------------|-----------------|------------------|
| Local dev | `./data/lancedb` | `./data/cold.db` |
| Single server | `/var/nmafc/hot` | `/var/nmafc/cold.db` |
| Self-hosted | `s3://bucket/hot` (MinIO) | `postgresql://root@cockroach:26257/nmafc` |
| Managed cloud | `s3://bucket/nmafc/hot` (AWS) | `postgresql://user:pass@neon.tech/nmafc` |

## Multi-Tenancy & Conversation Isolation

Enterprise deployments need two levels of isolation:

1. **Agent/Tenant isolation** — separate agents or organizations sharing the same infrastructure cannot see each other's memories
2. **Conversation isolation** — within a single agent, separate conversation threads don't leak context

NMAFC handles both via `agent_id` and `conversation_id` scoping. Every read and write in Hot RAM and Cold ROM is filtered by both IDs.

### Configuration

**Via environment variables:**

```bash
NMAFC_AGENT_ID=customer-support-bot-01
NMAFC_CONVERSATION_ID=conv_abc123
```

**Via code (dynamic per-request):**

```python
from nmafc.wrapper import NeuromorphicMemory
from nmafc.storage.config import NMafcConfig, StorageConfig

config = NMafcConfig(
    storage=StorageConfig(
        agent_id="customer-support-bot-01",
        conversation_id="conv_abc123",
        hot_uri="s3://bucket/nmafc/hot",
        cold_uri="postgresql://host/nmafc",
    )
)
memory = NeuromorphicMemory.from_config(config=config)
```

### How It Works

| Layer | Isolation Mechanism |
|-------|---------------------|
| Hot RAM (LanceDB) | `agent_id` and `conversation_id` columns added to every record. All vector searches, entity lookups, and listings include `WHERE agent_id = ? AND conversation_id = ?` |
| Cold ROM (SQLite) | Same columns + compound index `(agent_id, conversation_id)`. All reads scoped. FTS results joined with scope filter |
| Cold ROM (PostgreSQL) | Same columns + composite index. Row-level filtering on all queries |

### Enterprise Pattern

```python
# API gateway generates IDs per request
agent_id = request.headers["X-Agent-ID"]       # "support-bot-prod"
conversation_id = request.headers["X-Conv-ID"]  # "conv_7f3a2b"

config = NMafcConfig(
    storage=StorageConfig(
        agent_id=agent_id,
        conversation_id=conversation_id,
        hot_uri="s3://company-bucket/nmafc/hot",
        cold_uri="postgresql://neon.tech:5432/nmafc",
    )
)
memory = NeuromorphicMemory.from_config(config=config)
response = await memory.process_turn(user_msg=body["message"])
```

Multiple workers (Lambda, ECS, K8s pods) all read/write the same remote storage — memories are shared within a scope and completely invisible across scopes.

## Benchmark Suite

Academic-grade evaluation comparing four memory approaches on real research datasets.

### Datasets

- **LoCoMo** (Maharana et al., 2024): 10 conversations, 1986 QA pairs, 5 categories
- **LongMemEval** (Wu et al., 2024 / Zep paper): 500 questions, 6 types, 3 variants

### Four Arms

1. **Raw LLM** — Full context window stuffing (baseline)
2. **RAG** — Chunked retrieval over the raw transcript (external baseline)
3. **Neuromorphic** — Full NMAFC at the published defaults (λ_active = 0.05)
4. **Neuromorphic Tuned** — identical, but λ_active = 0.005 (`neuromorphic_tuned`)

Arms 3 and 4 differ by **exactly one number**, so the gap between them isolates
the effect of the decay horizon and nothing else. Arm 2 is the external
comparison: does NMAFC beat ordinary retrieval?

#### Why the fourth arm exists

A record is pruned once its weight decays to `w_prune`, so an ActiveContext
memory that is never re-retrieved survives

```
exp(-λ·Δt) ≤ w_prune   →   Δt ≥ ln(1/w_prune) / λ
```

turns. At the defaults that is ln(10)/0.05 ≈ **46 turns**. LoCoMo conversations
run 186–345 exchanges and ask every question only *after* the last one, so
ActiveContext records are mathematically guaranteed to be gone before the first
question is asked. The reinforcement half of the design never fires either —
LTP resets weight on retrieval, and nothing is retrieved mid-ingestion — so
decay runs unopposed for the entire conversation.

λ for the tuned arm is derived from conversation length, not fitted to scores:

| | |
|---|---|
| longest conversation | 345 exchanges |
| requirement | ln(10)/λ > 345 → λ < 0.0067 |
| chosen | **0.005** → horizon ≈ 460 turns |

The principle is that the memory horizon must outlast the interaction horizon.
That depends on the deployment's conversation length, which is a property of the
workload rather than of the test set's answers. Deployments with longer-running
conversations should scale λ down further.

#### Removed arm: Stateful No-Decay

Earlier revisions carried a fifth arm, `stateful_nodecay` — NMAFC with decay and
pruning switched off entirely (MemGPT/Zep-style "keep everything") — as a
matched control against `neuromorphic`. It has been **dropped from the default
arm set**, because `neuromorphic_tuned` is a strictly better control: it differs
from `neuromorphic` by one value instead of four, and unlike the no-decay arm it
is a configuration you would actually ship. The code remains at
`arms/stateful_nodecay.py` and is still runnable with `--arms stateful` for
anyone reproducing the older comparison.

The RAG arm (`scripts/benchmarks/arms/rag.py`) chunks the transcript into
overlapping windows, embeds them once at ingest, and retrieves top-k per
question. It makes **zero LLM calls during ingestion**, so it is a genuine
retrieval baseline rather than a handicapped one. Tunable via
`NMAFC_RAG_CHUNK_TURNS`, `NMAFC_RAG_CHUNK_STRIDE`, `NMAFC_RAG_TOP_K`.

### Measured Dimensions

| Metric | Meaning |
|--------|---------|
| **F1** | Normalized token-overlap against the reference answer |
| **Judge Accuracy** | Binary correct/incorrect from an LLM-as-judge pass |
| **Avg context** | Tokens fed to the model per question — the efficiency claim |
| **Avg latency** | Milliseconds per answer |
| **Total tokens** | Cumulative cost |

Results are also broken down per question category (single-hop, multi-hop,
temporal, open-domain, adversarial).

### Fair-Comparison Notes

Several details materially affect whether the numbers mean anything:

- **All arms receive session dates and speaker names.** LoCoMo questions refer
  to people by name and ask "when" questions; a transcript flattened to
  `User:`/`Assistant:` with dates stripped makes the temporal category
  unanswerable by construction.
- **Sessions are ordered numerically, and each keeps its own timestamp.**
  `locomo_loader.py` previously sorted the `session_<n>` keys as text, which
  orders them `1, 10, 11, ... 19, 2, 20` — so the conversation was replayed out
  of chronological order. The timestamps were gathered by sorting a second list
  of `session_<n>_date_time` keys, and because the suffix changes the sort
  order, `session_10_date_time` sorted *before* `session_1_date_time` while
  `session_1` still sorted before `session_10`. The two lists disagreed and
  every session was stamped with a different session's date. One conversation
  (`conv-26`) also ships 35 date keys for 19 sessions, so any positional pairing
  drifts further. Dates are now looked up per session key, and sessions sort on
  the parsed integer. Fixing this alone moved the raw-LLM arm from F1 0.225 to
  0.475 on a 30-question slice — that arm holds no memory state, so the change
  isolates the loader defect from anything the framework does.
- **Memory arms ingest both speakers.** `build_exchanges()` pairs each user turn
  with the assistant turn that follows it, so memory arms see the same evidence
  as the raw-LLM baseline. Ingesting only user turns scores the memory arms on
  strictly less information than the baseline they are compared against.
- **Every arm gets the same answer-format rules.** LoCoMo gold answers are 1–4
  words, so a verbose but correct reply scores near-zero F1. `SHORT_ANSWER_RULES`
  in `arms/base.py` is appended to every arm's answer prompt.

### Memory Classification Prompt

`EXTRACTION_SYSTEM_PROMPT` in `integration/extractor.py` decides which tier each
extracted fact lands in, and therefore how long it survives. It was revised
after the first benchmark run, for a reason visible in the stored records rather
than in the scores:

```
CoreAnchor    jon_job_loss_catalyst  "Jon lost his job, which gave him the push..."
ActiveContext jon_job_status         "Jon lost his job as a banker yesterday"
```

The same event was recorded twice in two tiers, and the copy carrying the date
was the one scheduled to expire. The taxonomy offered only "permanent identity
fact" or "current state that may change", and a completed event is neither, so
the model reached for the state-shaped label. With `lambda_active_context =
0.05` and `w_prune = 0.1`, an ActiveContext record is pruned after
`ln(10)/0.05 ≈ 46` turns; LoCoMo conversations run 186–345 turns and ask every
question at the end, so those records are gone before the first question. In the
first run 79% of stored memories were ActiveContext or EphemeralState.

The revision adds five domain-general rules: completed events are permanent and
belong in CoreAnchor; an event and the state it produced are recorded as
separate entities so the permanent half does not inherit the mutable half's
lifetime; CoreAnchor covers identity, relationships, milestones and enduring
preferences rather than only the clinical examples it previously listed;
EphemeralState is restricted to genuinely momentary things; and time-anchored
facts must resolve relative references against the session timestamp rather than
inventing a date.

Two caveats a reader should weigh:

- **These results are not comparable to previously published NMAFC numbers.**
  Classification drives decay, decay drives what survives to question time.
- **The prompt was written once, from the failure mechanism, without iterating
  against benchmark scores.** Tuning a prompt until a test-set number improves is
  fitting to the test set. The same prompt is used by every memory arm, so the
  comparison between them stays matched.

### Reproducibility

Set `NMAFC_LLM_TEMPERATURE=0` to pin sampling. Note that this reduces but does
not eliminate run-to-run variance: mixture-of-experts models served behind a
batching proxy can return different completions for identical requests even at
temperature 0. Treat small-sample runs as plumbing checks, not results, and
quote a margin of error derived from repeating a run rather than assuming
determinism.

### Reliability & Throughput

`scripts/benchmarks/resilience.py` wraps both providers with:

- a sliding-window rate limiter (`NMAFC_BENCH_TPM_LIMIT`, `NMAFC_BENCH_RPM_LIMIT`,
  `NMAFC_BENCH_QUOTA_SAFETY`)
- exponential backoff with jitter honouring `Retry-After` (`NMAFC_BENCH_MAX_RETRIES`)

One failure mode this does **not** cover: Ollama unloads an idle model after
about five minutes. The raw-LLM arm requests no embeddings, so on a multi-arm
run the embedder can be evicted while it runs, and the next arm fails with
`dial tcp 127.0.0.1:<port>: connection refused` against Ollama's internal runner
port — a 400 from the client's perspective, so the retry logic treats it as a
bad request rather than a transient fault. Keep the model resident for the
duration of a run:

```bash
# server-side: set before `ollama serve`
OLLAMA_KEEP_ALIVE=-1
```

#### Batched LTP reinforcement

Retrieval reinforces every record Spreading Activation surfaces, which after two
hops is routinely dozens per question. That was applied one record at a time,
and each `update_reinforcement` cost a scan, a delete and an add — a LanceDB
fragment rewrite per record. Cost grew superlinearly, because every rewrite
fragmented the table further for the next one.

`HotStorage.apply_reinforcements()` now performs the whole set in a single
delete + add, the same optimisation `apply_weight_updates()` already made for
the decay pass. Measured on real benchmark data (330 records, 1536-dim vectors,
40 reinforced):

| | Per-record loop | Batched |
|---|---|---|
| 40 reinforcements | 51,068 ms | **1,322 ms** |

Isolated on synthetic records the gap widens with set size — 10.7× at 10
records, 27.9× at 30, 64.8× at 60 — confirming the superlinear shape rather than
a constant overhead. This dominated answer latency: the memory arms measured
~28 s per answer against the raw arm's 2.6 s, despite issuing one LLM call each.

#### Judge independence

`--judge` defaults to `--provider`, which makes the answering model grade its
own output. Models favour their own phrasing, so this inflates whichever arm
shares the judge's family. Set `NMAFC_BENCH_JUDGE` to a different family; the
runner warns when the two match, and gives a judge on a separate deployment its
own rate limiter so it does not queue behind the answering model's quota.

The benchmark answers with `azure_v1/DeepSeek-V4-Pro` and judges with
`azure_v1/Kimi-K2.6` (Moonshot), which is the only non-DeepSeek family the
Foundry gateway will serve. The Claude deployments on the same resource list as
`status: succeeded` but return `404 api_not_supported` on every chat route —
verified on `claude-haiku-4-5` and `claude-opus-5` across the `/openai/v1` route,
the classic `/openai/deployments` route on three api-versions, the
`services.ai.azure.com` inference host and the Anthropic-native `/messages`
shape. Two deployments created four days apart both fail, so this is a routing
limitation rather than propagation lag.

Kimi is a reasoning model: it spends completion tokens on `reasoning_content`
before emitting `content`. Nothing in the codebase sets `max_tokens`, so the API
default applies and this is fine — but adding a small `max_tokens` would make
judgements come back empty with `finish_reason: length`. It costs ~4 s per
judgement against DeepSeek-V4-Flash's ~0.5 s, or roughly 33 min of judging for a
full four-arm 1,986-question run at `--judge-concurrency 16`.

Note that F1 and the judge disagree systematically, and the disagreement is not
noise. LoCoMo gold answers are terse noun phrases, so a correct but
conversational answer is scored down by token overlap while the judge accepts
it — "Caroline is a trans woman" against gold "Transgender woman" scores F1 0.33
and judge ✅. Report judge accuracy as primary and F1 as secondary rather than
tightening answer-format prompts, which would be fitting to the test set.

Runs are parallel across conversations and checkpointed per arm, so an
interrupted run resumes instead of restarting:

```bash
# resumes automatically from results/<run>/checkpoint_<arm>.json
python -u -m scripts.benchmarks.run_locomo --arms raw,rag,neuromorphic,neuromorphic_tuned \
  --output scripts/benchmarks/results/full/
```

Because the runner only prints when a whole conversation completes — and memory-arm
ingestion is one LLM call per exchange — the log can look frozen for long stretches.
`scripts/benchmarks/live_progress.py` appends a progress line every N seconds:

```bash
python -u scripts/benchmarks/live_progress.py \
  --log scripts/benchmarks/results/full/../full.log \
  --out scripts/benchmarks/results/live.log
```

### Quick Start

```bash
# Install benchmark dependencies
uv pip install -e ".[bench,llm,aws]"

# Tests use pytest-asyncio; without it the async tests error out instead of running
uv pip install pytest-asyncio

# Embeddings: Azure text-embedding-3-small (1536-dim) via .env, or locally:
#   ollama serve && ollama pull nomic-embed-text
# The local path is CPU-only on this machine and ~7x slower per embedding.

# Run LoCoMo benchmark (1 conversation, single-hop, F1 only)
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_locomo \
  --provider "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0" \
  --embedding "ollama/nomic-embed-text" \
  --conversations 1 --categories "1" --skip-judge

# Run all four arms in parallel, checkpointed, with a bounded question count
python -u -m scripts.benchmarks.run_locomo \
  --arms raw,rag,neuromorphic,neuromorphic_tuned \
  --conversations 2 --max-questions 15 \
  --concurrency 8 --judge-concurrency 12 \
  --output scripts/benchmarks/results/pilot/

# Run LongMemEval (10 questions, oracle variant)
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_longmemeval \
  --provider "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0" \
  --embedding "ollama/nomic-embed-text" \
  --variant oracle --limit 10
```

New runner flags: `--arms` (subset to run), `--concurrency` (conversations in
flight), `--judge-concurrency`, `--max-questions` (cap per conversation),
`--output` (checkpoint + results directory). All default from environment
variables — see `.env.example`.

See [scripts/benchmarks/README.md](scripts/benchmarks/README.md) for full documentation of all flags, examples, and output format.

## Mathematical Specification

### Memory State Tuple

Each record in Hot RAM: `S_i = <v_i, w_i(t), tau_i, k_i>`

| Symbol | Type | Description |
|--------|------|-------------|
| `v_i` | R^d | Embedding vector |
| `w_i(t)` | [0, 1.0] | Synaptic weight at time t |
| `tau_i` | Enum | Memory type (CoreAnchor, ActiveContext, EphemeralState) |
| `k_i` | N >= 0 | Consolidation index (retrieval count) |

### Decay Formula

```
w_i(t) = w_i(t_0) * exp(-lambda_eff * delta_t)

where:
  delta_t = current_turn - last_reinforced_turn
  lambda_eff = lambda_base(tau_i) * alpha(k_i)
  alpha(k) = exp(-eta * k)
  eta = 0.15
```

### Decay Comparison Table

| Tier | lambda_base | After 10 turns (k=0) | After 10 turns (k=5) | After 10 turns (k=10) |
|------|-------------|----------------------|----------------------|-----------------------|
| CoreAnchor | 0.0 | 1.000 | 1.000 | 1.000 |
| ActiveContext | 0.05 | 0.607 | 0.858 | 0.951 |
| EphemeralState | 0.69 | 0.001 | 0.056 | 0.314 |

### LTP Reinforcement (on retrieval)

```
w_i = 1.0                           # Reset to full strength
k_i = k_i + 1                       # Increment consolidation index
last_reinforced_turn = current_turn  # Reset decay clock
```

### Override Suppression

```
w_old = w_old * gamma    (gamma = 0.1, default)
```

### Pruning Condition

```
if w_i <= w_prune (default 0.1): delete from Hot RAM
```

### Consolidation Elevation

```
if k_i >= 10 AND tau_i == ActiveContext:
    tau_i = CoreAnchor    # Promoted — never decays again
    w_i = 1.0            # Reset to full weight
```

## Project Structure

```
nmafc/
├── src/nmafc/
│   ├── wrapper.py                 # Top-level NeuromorphicMemory class
│   ├── schemas/
│   │   └── memory.py              # Pydantic models (MemoryRecord, DecayConfig, etc.)
│   ├── engine/
│   │   ├── decay.py               # Ebbinghaus exponential decay
│   │   ├── reinforcement.py       # LTP (weight reset + k increment)
│   │   ├── pruning.py             # Override detection + eviction
│   │   ├── consolidation.py       # REM sleep (elevation + cleanup)
│   │   └── rollback.py            # State reconstruction from Cold ROM
│   ├── integration/
│   │   ├── factory.py             # Provider factory (provider/model strings)
│   │   ├── base.py                # Abstract LLMProvider + EmbeddingProvider
│   │   ├── extractor.py           # StateExtractor (tool-use based extraction)
│   │   ├── query_router.py        # Retrieval + Spreading Activation
│   │   ├── openai_provider.py     # OpenAI / OpenAI-compatible
│   │   ├── anthropic_provider.py  # Anthropic native
│   │   ├── bedrock_provider.py    # AWS Bedrock (boto3 + Anthropic SDK)
│   │   ├── azure_provider.py      # Azure OpenAI
│   │   └── fastembed_provider.py  # ONNX CPU embeddings
│   └── storage/
│       ├── config.py              # NMafcConfig + TOML parsing
│       ├── hot.py                 # HotStorage (LanceDB, supports S3)
│       ├── cold_base.py           # Abstract ColdStorageBase interface
│       ├── cold.py                # ColdStorage (SQLite + FTS5)
│       └── cold_pg.py             # PostgresColdStorage (PostgreSQL + tsvector)
├── configs/
│   └── default.toml               # Default configuration
├── scripts/benchmarks/             # Academic benchmark suite
│   ├── datasets/                   # LoCoMo + LongMemEval loaders
│   ├── arms/                       # 4 benchmark conditions
│   │   ├── base.py                # Shared answer-format rules + build_exchanges()
│   │   ├── raw_llm.py             # Baseline: full transcript in context
│   │   ├── rag.py                 # Chunked retrieval baseline
│   │   ├── stateful_nodecay.py    # Retired control: decay/pruning off
│   │   ├── neuromorphic.py        # Full NMAFC, published defaults
│   │   └── neuromorphic_tuned.py  # Same, lambda_active_context = 0.005
│   ├── evaluation/                 # F1 + LLM-as-judge metrics
│   ├── resilience.py              # Rate limiter + retry/backoff wrappers
│   ├── live_progress.py           # Appends progress lines during long runs
│   ├── run_locomo.py              # LoCoMo CLI runner (parallel + checkpointed)
│   ├── run_longmemeval.py         # LongMemEval CLI runner
│   └── visualize.py               # Publication-ready Plotly charts
├── tests/                          # pytest suite (unit + integration)
├── .env.example                    # All provider credential variables
└── pyproject.toml                  # Package metadata + dependencies
```

## Testing

```bash
# pytest-asyncio is REQUIRED -- without it the async tests do not fail loudly,
# they error out as "async def functions are not natively supported" and the
# coverage they were meant to provide is silently absent.
pip install pytest-asyncio

# Run all tests
pytest

# Run specific test categories
pytest tests/test_decay.py -v
pytest tests/test_wrapper_e2e.py -v

# Run with coverage
pytest --cov=nmafc
```

## Performance Notes

### Batched decay writes

The decay pass runs every turn and recomputes the weight of every mutable
record. Applying those weights one record at a time is quadratic: each
`update_weight()` costs a scan, a delete and an add, so a turn costs `1 + 3N`
storage operations against a table of `N` memories that grows every turn.

`HotStorage.apply_weight_updates()` collapses the whole pass into one query, one
delete and one add — 3 operations per turn regardless of `N`. `prune_cycle()`
likewise uses `delete_many()` instead of deleting record by record.

Measured on live LoCoMo ingestion: **~40s per exchange → ~5-6s per exchange**,
with byte-identical resulting memory state. The remaining per-turn full scans
(`get_all_mutable`, prune's `get_all`, consolidation) are still `O(n)`, so
per-turn cost still grows slowly as memory fills.

If you extend storage, prefer batch predicates (`id IN (...)`) over per-record
calls. LanceDB writes a new data file per operation, so per-record loops produce
thousands of tiny files and dominate ingestion time.

## Biological Analogies

| NMAFC Mechanism | Biological Analogue | Function |
|-----------------|---------------------|----------|
| Exponential decay | Ebbinghaus forgetting curve | Unused memories fade naturally |
| LTP reinforcement | Long-Term Potentiation | Repeated retrieval strengthens synapses |
| CoreAnchor promotion | Memory consolidation | Important facts move to permanent storage |
| Override suppression | Synaptic depression | Contradicted pathways are weakened |
| Pruning cycle | Synaptic homeostasis | Weak connections are physically removed |
| REM consolidation | Sleep-dependent memory processing | Periodic restructuring and promotion |
| Spreading Activation | Associative priming | Related concepts activate each other |
| Hot RAM / Cold ROM | Working memory / Long-term memory | Fast-access vs. archival storage |

## References

- Maharana et al. (2024). "LoCoMo: Long-term Conversational Memory Dataset" — F1 evaluation protocol
- Rasmussen et al. (2025). "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" — LLM-as-judge evaluation, LongMemEval benchmark
- Packer et al. (2024). "MemGPT: Towards LLMs as Operating Systems" — Stateful agent memory design
- Zhong et al. (2023). "MemoryBank: Enhancing Large Language Models with Long-Term Memory" — Ebbinghaus-inspired decay
- Ebbinghaus (1885). "Memory: A Contribution to Experimental Psychology" — Forgetting curve

## License

MIT
