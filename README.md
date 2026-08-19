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

The links come from the extractor, which is told to emit them for facts about
the same person, event or object. Measured on real LoCoMo data, 95% of records
carry at least one link and none dangle.

**This mechanism was inert until the OpenAI tool schema was fixed, and it is
still not validated.** On the full run multi-hop reaches 0.427, ahead of both
full-context stuffing (0.406) and RAG (0.396) — but n=96, and the comparison
that would isolate the graph is confounded: the earlier `max_hops=0` run also
used an extractor prompt containing no graph-linking instructions, so the graph
being walked was not built the same way.

The measurement is blocked from the other end too. **51% of multi-hop questions
end in the model refusing to answer**, with the gold fact present in the
retrieved context in most of those cases, so whatever the traversal surfaces is
being discarded before it reaches an answer. Until the refusal rate comes down,
`max_hops` cannot be evaluated. Do not treat the description above as a
validated benefit — see [Current results](#current-results),
[Spreading Activation had no edges to walk](#spreading-activation-had-no-edges-to-walk)
and the paired A/B that follows it.

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

# With Web UI (FastAPI + Next.js frontend)
pip install nmafc[web]

# With CLI tools (nmafc start/init/chat)
pip install nmafc[cli]

# With benchmark suite
pip install nmafc[bench]

# Everything (LLM + AWS + Postgres + Web + CLI + Benchmarks)
pip install nmafc[all]

# Development (from source)
git clone https://github.com/blok-hamster/nmafc.git
cd nmafc
uv pip install -e ".[all,cli]"
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

## Web UI

NMAFC ships with a full-stack visual memory explorer. The backend is a FastAPI server exposing 27 REST endpoints + 1 WebSocket; the frontend is a Next.js dashboard with live updates.

### Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Overview stats, hot RAM weight distribution, tier breakdown, recent events |
| **Memory Explorer** | Search records, filter by tier, view decay info, drill into entities |
| **Entity Graph** | D3 force-directed graph of entities and `related_entities` links |
| **Decay Curves** | Recharts projection of every record's weight over 200 turns |
| **Event Timeline** | Stacked bar chart of cognitive events (overrides, prunes, LTP, etc.) |
| **Documentation** | Full in-app docs (no external README needed) |

### Running the Web UI

```bash
# Start Ollama (for local embeddings)
ollama serve &
ollama pull nomic-embed-text

# Start backend + frontend together
nmafc start

# Custom port
nmafc start --port 9000

# Production mode (builds frontend, single port)
nmafc start --production

# Open http://localhost:3000
```

The Next.js frontend proxies `/api/*` and `/ws/*` requests to the backend — zero frontend config needed. Just start both and open port 3000.

### WebSocket Live Updates

The frontend connects to `/ws/live` and receives real-time broadcasts:

```json
{
  "type": "turn_processed",
  "turn": 42,
  "extracted_count": 3,
  "hot_count": 187,
  "cold_count": 843,
  "pruned_count": 2,
  "ts": "2026-08-18T12:00:00Z"
}
```

## CLI Reference

NMAFC provides a unified CLI for setup, development, and interactive chat.

### nmafc init

Interactive setup wizard — asks for LLM provider, embedding model, API keys, and storage paths. Generates `.env` and `configs/custom.toml`.

```bash
nmafc init
```

### nmafc start

Starts the FastAPI backend + Next.js frontend as subprocesses. Handles SIGINT for graceful shutdown.

```bash
# Dev mode: backend :8000, frontend :3000
nmafc start

# Custom port + config
nmafc start --port 9000 --config configs/custom.toml

# Production: build frontend, serve from one port
nmafc start --production
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8000 | Backend port |
| `--host` | 0.0.0.0 | Bind address |
| `--config` | configs/default.toml | Config TOML path |
| `--production` | off | Build frontend, serve from one port |

### nmafc chat

Terminal REPL — processes messages through the full neuromorphic pipeline. No web UI required.

```bash
nmafc chat
nmafc chat --llm "groq/llama-3.1-70b-versatile"
nmafc chat --config configs/custom.toml
```

| Command | Description |
|---------|-------------|
| `/stats` | Show system stats (records, weights, events) |
| `/memory` | List all Hot RAM records |
| `/events` | Show recent cognitive events |
| `/rollback N` | Restore memory to turn N |
| `/quit` | Exit the chat |

## Library Integration

### Context Manager (async)

`NeuromorphicMemory` supports `async with` for automatic resource cleanup:

```python
import asyncio
from nmafc.wrapper import NeuromorphicMemory

async def main():
    async with await NeuromorphicMemory.from_config() as mem:
        response = await mem.process_turn("My name is Alice")
        print(response)
        print(mem.get_hot_stats())

asyncio.run(main())
```

### Sync Wrapper (no async needed)

`SyncNeuromorphicMemory` wraps the async API with `asyncio.run()` and supports `with`:

```python
from nmafc.wrapper import SyncNeuromorphicMemory

with SyncNeuromorphicMemory.from_config() as mem:
    response = mem.process_turn_sync("Hello world")
    print(response)
    print(mem.get_hot_stats())
```

### Available Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `process_turn(msg)` | `str` | Full pipeline: retrieve + respond + extract + decay |
| `ingest_updates(updates)` | `None` | Inject facts without LLM call |
| `consolidate()` | `int` | Manual REM sleep pass |
| `rollback(to_turn)` | `int` | Rebuild state from Cold ROM |
| `get_hot_stats()` | `dict` | Record count, avg weight, type breakdown |
| `get_cold_stats()` | `dict` | Archive event counts |
| `get_event_stats()` | `dict` | Cognitive event counts by type |
| `get_events(**kwargs)` | `list` | Query events with filters |
| `get_event_timeline(limit)` | `list` | Aggregated counts per turn |
| `get_entity_events(name)` | `list` | Events for one entity |
| `current_turn` | `int` | Current turn counter (property) |
| `close()` | `None` | Close storage connections |

## Examples

```bash
# Minimal hello-world (10 lines)
python examples/minimal.py

# Manual memory injection (no LLM needed)
python examples/manual_ingestion.py

# Custom LLM + embedding providers
python examples/custom_provider.py

# Non-async usage with SyncNeuromorphicMemory
python examples/sync_usage.py
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
theta = 0.45                   # Cosine similarity below which Cold ROM fallback fires
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

### Current results

**Status: full run.** All 10 LoCoMo conversations, all 1,986 QA pairs, no
sampling. Answering `azure_v1/DeepSeek-V4-Pro`, embeddings
`azure_v1/text-embedding-3-small`, judge `azure_v1/Kimi-K2.6`.

Every figure below is regenerable from the committed per-question JSON:

```bash
python -m scripts.benchmarks._summarise_locomo \
  --run scripts/benchmarks/results/full_v2 \
  --baseline scripts/benchmarks/results/locomo_full
```

#### Which questions are scored

LoCoMo ships 1,986 QA pairs in five categories. The fifth — `adversarial`, 446
questions — is **excluded here, as it is in every published comparison** (Mem0,
Zep, MemMachine and others all report on the remaining 1,540). The category was
meant to hold unanswerable questions, but the released gold answers are ordinary
facts: 444 of the 446 carry a real answer rather than "not mentioned", so a
system that correctly declines is marked wrong. Scoring it measures the grader's
defect rather than the system's memory.

Both denominators are printed throughout, so the exclusion is visible rather
than quietly applied. Reporting on 1,986 is not more honest — it is a different
and incomparable number.

#### Headline

| arm | 4-cat accuracy | 5-cat | F1 | context tokens |
|---|---|---|---|---|
| Raw LLM (full context) | **0.7045** | 0.5670 | 0.4679 | 19,998 |
| RAG | 0.6026 | 0.4955 | 0.4815 | 1,454 |
| Neuromorphic (λ=0.05) | 0.5675 | 0.4471 | 0.4348 | 1,236 |
| **Neuromorphic Tuned (λ=0.005)** | **0.5955** | 0.4698 | 0.4437 | 1,304 |

Against the previous full run, paired over the scored categories:

| arm | before | after | McNemar exact |
|---|---|---|---|
| `neuromorphic` | 0.5351 | 0.5675 | +210/−159, **p = 0.0092** |
| `neuromorphic_tuned` | 0.5435 | 0.5955 | +211/−132, **p = 2.3e-05** |

**Harness validation.** The full-context arm scores 0.7045 where the published
LoCoMo baselines report ~0.73 for the same condition, and our category counts
(282 + 321 + 96 + 841) sum to exactly the 1,540 used in those papers. The ruler
gives close to the same reading as everyone else's, which is the prerequisite
for any of the comparisons below meaning anything.

#### Per category — where the architecture earns its place, and where it does not

| category | n | Tuned | Raw LLM | RAG | vs full context |
|---|---|---|---|---|---|
| **temporal** | 321 | **0.583** | 0.442 | 0.346 | **+14.1** |
| **multi-hop** | 96 | **0.427** | 0.406 | 0.396 | **+2.1** |
| single-hop | 282 | 0.486 | 0.599 | 0.418 | −11.3 |
| open-domain | 841 | 0.656 | 0.874 | 0.786 | −21.8 |

The temporal result is the one worth taking seriously: **on "when did X happen"
questions the memory system beats full-context stuffing by 14 points while using
15× less context**, and beats RAG by 24. That is the category a structured
memory ought to win, and it is the only place any arm beats the full-context
ceiling. Multi-hop edges ahead of both baselines too, though n=96 is small and
the margin is not significant on its own.

The deficit is concentrated almost entirely in **open-domain, which is 55% of
the scored set**. Closing that one category to RAG's level would put the overall
figure at ~0.667, above RAG. The headline is not lost across the board; it is
lost in one place.

#### Why answers are wrong: refusal, not just retrieval

`_diagnose_retrieval.py` replays retrieval against the persisted stores and
checks whether the gold answer was in the context the model actually received —
no regeneration, no judging, one embedding call per question. Of the 623 wrong
answers on the scored categories:

| | count | share |
|---|---|---|
| gold answer **present** in context, still answered wrong | 356 | 57.1% |
| gold answer absent from context | 267 | 42.9% |
| model abstained ("I don't know") | 337 | 54.1% |
| **abstained while holding the answer** | **190** | **30.5%** |

**More than half of all wrong answers are the system declining to answer**, and
in 190 cases it declined with the correct fact in front of it. Abstention rate
by category: multi-hop 51.0%, temporal 25.5%, open-domain 19.9%, single-hop
15.2%. On the excluded adversarial set it reaches 95.8%.

Two consequences:

- The largest single lever is the **answer prompt**, not retrieval. It is also
  the cheapest to test, because the persisted stores make retrieval-phase
  experiments nearly free.
- **Spreading Activation cannot currently be evaluated at all.** Half of every
  multi-hop question ends in a refusal, so whatever the graph retrieves is being
  discarded before it reaches an answer. The flat multi-hop result reported
  below is evidence about the answering step, not about the graph.

Read with these caveats, all of which are load-bearing:

- `max_hops=2` was developed against conv-26 and conv-30. Those two are included
  in this run, so some of the gain may still be fitting to them.
- The previous run had `max_hops=0` **and** an extractor prompt with no
  graph-linking instructions at all (see `integration/extractor.py`). The
  before/after comparison therefore moves two things at once and cannot be used
  to attribute the gain to graph traversal specifically.
- Do not quote a latency improvement against the previous release. Both runs
  were rate-limited; most of any gap is queueing, not code. The number that does
  stand is the comparison against our own RAG arm: 21.3 s per answer against
  3.6 s, roughly 6× slower for 1 point less accuracy.
- An earlier revision of this section claimed the system "confabulates rather
  than declining to answer" on adversarial. That was **backwards** — it abstains
  on 95.8% of them. The correction is what produced the refusal analysis above.

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

#### Committed benchmark data

Per-question results for the two runs the README cites are in the repository, so
the tables can be recomputed rather than taken on trust:

| path | contents |
|---|---|
| `results/full_v2/results.json` | current run — both memory arms, 1,986 rows each |
| `results/full_v2/checkpoint_*.json` | same rows grouped by conversation, as written during the run |
| `results/full_v2.log` | the run's console log, including the four network blips it recovered from |
| `results/locomo_full/` | the earlier baseline run — all four arms |

Each row carries the question, category, gold answer, prediction, judge verdict,
F1, context tokens and latency.

What is **not** committed is the persisted memory stores each run leaves behind:
4.1 GB across 222,485 files for ten conversations. They stay local (see
`.gitignore`), and they are what makes retrieval-phase experiments cheap —
`_diagnose_retrieval.py` replays against them for one embedding call per
question instead of a full re-ingest. They are written under `<output>/stores/`
as a side effect of ingestion checkpointing, so any run with
`--ingest-checkpoint-every` non-zero (the default is 25) leaves them behind.

Three read-only analysis scripts operate on this data and make no API calls
except where noted:

| script | question it answers | cost |
|---|---|---|
| `_summarise_locomo.py` | what are the numbers, on both denominators | none |
| `_diagnose_retrieval.py` | is a wrong answer a ranking failure or a refusal | 1 embedding/question |
| `_analyse_beta_survivors.py` | which facts did clustering protection actually save | none |

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

#### Spreading Activation had no edges to walk

The graph traversal ran on every query and reached nothing. `related_entities`
was declared on `MemoryStateUpdate`, written by `HotStorage` and traversed by
`QueryRouter`, but it was **missing from the OpenAI provider's tool schema** and
never mentioned in the extraction prompt. The model was therefore unable to emit
a single link. Measured across two populated benchmark stores: **0 of 675
records** carried one. Hop traversal timed at 0.0 ms — not because it was fast,
but because the frontier was always empty.

Nothing failed loudly. The Anthropic and Bedrock providers had the field all
along, so the schema looked complete on inspection; only the OpenAI path — the
one every benchmark run uses — was missing it. It is corroborated by multi-hop
being the worst-scoring category in the pilot (F1 0.089 / 0.141), which is the
category the mechanism exists to serve.

The fix adds the field to `MEMORY_TOOL_SCHEMA` and a `## Graph Links` section to
`EXTRACTION_SYSTEM_PROMPT` — declaring the field alone is not enough, since a
field the instructions never mention tends to stay empty. Measured on 20 real
LoCoMo exchanges through the full pipeline:

| | Before | After |
|---|---|---|
| Records with ≥1 link | 0% (0 of 675) | **95%** |
| Dangling pointers | — | **0%** |

#### …and turning it on is the single largest measured gain. Superseded finding.

> **Superseded 2026-08-18.** The A/B below is retained because its mechanics are
> sound and its fan-out numbers still hold, but its conclusion was wrong, and
> wrong in an instructive way. It measured 13 multi-hop questions on one
> conversation and concluded traversal does not help. Across all 303 questions of
> conv-26 and conv-30 it is the **only** change since the last release that
> improved anything: `max_hops` 0 → 2 moves `neuromorphic_tuned` from 0.472 to
> 0.525 judge accuracy (+32 questions, −16, McNemar exact **p = 0.029**), and the
> gain is broad rather than concentrated in multi-hop:
>
> | category | n | `max_hops=0` | `max_hops=2` | Δ |
> |---|---|---|---|---|
> | single-hop | 43 | 0.465 | 0.628 | **+0.163** |
> | multi-hop | 13 | 0.308 | 0.385 | +0.077 |
> | open-domain | 113 | 0.602 | 0.655 | +0.053 |
> | temporal | 63 | 0.698 | 0.746 | +0.048 |
> | adversarial | 71 | 0.099 | 0.085 | −0.014 |
>
> The lesson is about the measurement, not the mechanism. Multi-hop is 13 of 303
> questions here; scoping the A/B to the category the feature was *designed* for
> sampled the smallest stratum in the set and missed that traversal's real
> benefit lands on **single-hop** — where the neighbour of a matched fact turns
> out to be the specific detail the question wanted. A feature evaluated only on
> the questions its designer expected it to serve is not evaluated.
>
> The context cost below is real and unchanged: 461 → 1,571 tokens. The
> efficiency claim survives it — at 1,571 tokens the arm still uses **9× less
> context than the raw-LLM baseline** (14,326) while scoring within 2 points of
> it, and beats RAG (0.472 at 1,521 tokens) at parity of budget.

Enabling the graph was expected to move multi-hop, the category it exists to
serve. It does not. A paired A/B on conv-26 — ingested once, store copied, the
same 13 multi-hop questions answered against each copy with only `max_hops`
differing — gives:

| | graph off (`max_hops=0`) | graph on (`max_hops=2`) |
|---|---|---|
| Mean F1 | 0.242 | **0.203** (−0.039) |
| Judge accuracy | 7/13 | **6/13** |
| Context tokens | 457 | **1,585** (3.5×) |
| Per-question | — | helped 0, hurt 2, **unchanged 11** |

Eleven of thirteen answers were byte-identical despite ~1,100 extra tokens of
retrieved context. The traversal is not surfacing facts the answer needed; it is
surfacing facts the model then ignores. The two that changed both got worse.

Fan-out is why, and it scales far worse than a small store suggests. On a
48-record store 2-hop traversal returned 18.1 records; on a realistic 403-record
store it returns 34.8:

| `max_hops` | 0 | 1 | 2 |
|---|---|---|---|
| Records retrieved | 10.0 | 22.2 | **34.8** |
| Context tokens | 457 | 1,022 | **1,585** |

That last figure is the important one: **at `max_hops=2` the memory arm consumes
more context than the RAG baseline it is supposed to beat** (1,497 tokens). The
efficiency result and the graph cannot both stand as currently implemented.

Two cautions on reading this. n=13 on one conversation — conv-30 contains no
multi-hop questions at all — so it is directional, not conclusive; the full run
has 96 multi-hop questions across ten conversations. And the apparent latency
difference (4,700 ms off vs 1,939 ms on) is an artifact of run order: the
graph-off condition ran first and paid embedding cold-start and LanceDB warmup.
It is not evidence that traversal is free.

The mechanism is not necessarily wrong — unbounded BFS over a store where 95% of
records carry links is. A score-ranked expansion with a hard cap on added records
would test the idea without the fan-out, and has not been tried.

#### The Cold ROM fallback was unreachable, and discarded

Two independent bugs on the same code path, which is why neither showed up.

**The threshold could not be met.** `HotStorage.search` computed
`score = 1 - distance` while LanceDB was using its default L2 metric, so `score`
was not a similarity at all. Real measured `_distance` values ran 1.41–1.61
(squared L2), clamping `score` to **0.000** on every hit against `theta = 0.75`.
Search now sets `distance_type("cosine")`, for which `1 - distance` *is* the
cosine similarity. Regression tests pin the exact analytic value: a query at 45°
to a stored vector scores 0.707 under cosine and 0.0 under the L2 default.

**The result was thrown away.** The fallback then ran `keyword_search` and used
the output only `if cold_results and not vector_hits` — and on a populated store
`vector_hits` is never empty. So the archive was searched on every weak query and
consulted on none of them. Cold results now merge into the returned set, deduped
by entity against the Hot RAM hits, and are deliberately *not* reinforced:
reading the archive must not resurrect a memory into the working set.

**`theta` was then retuned, because the fix made its value matter.** At 0.75 it
was unreachable and inert; against real cosine scores it fires on ~60% of
questions, which makes the "fallback" the default path and injects up to
`fallback_keyword_limit` BM25 rows into contexts Hot RAM had already answered.
The new default is derived from the separation between answerable and
unanswerable queries — each store scored against its own questions and against
another conversation's, using **no answer keys**:

| Store | On-topic top-1 | Off-topic top-1 |
|---|---|---|
| conv-26 (410 records) | 0.493 – 0.867 | 0.201 – 0.472 |
| conv-30 (330 records) | 0.403 – 0.873 | 0.177 – 0.481 |

The populations barely overlap. `theta = 0.45` sits in the gap: it wrongly falls
back on 0–2% of answerable queries while catching 97–98% of unanswerable ones,
biased toward the on-topic side because a spurious fallback pollutes a context
that was already correct. Verified end to end — at 0.45 the fallback fires on 0
of 15 real questions, identical to disabling it outright.

Reproduce either measurement with `scripts/benchmarks/_measure_theta.py` and
`scripts/benchmarks/_verify_graph_links.py`.

#### The Cold ROM fallback is reachable and still never fires

Fixing the threshold made the fallback *possible*. Measuring it showed it is
still, in practice, dead code. Across all 304 LoCoMo questions on real ingested
stores at `theta = 0.45`, the archive was consulted **5 times (1.6%)**. Hot RAM's
best-hit cosine has a floor of 0.376 and a median near 0.70, so it clears 0.45 on
299 of 304 questions.

| `theta` | 0.45 | 0.55 | 0.65 | 0.70 |
|---|---|---|---|---|
| Fallback firing rate | **1.6%** | 10% | 32% | 50% |

This matters more than a tuning note, because it invalidates attribution rather
than accuracy: **every Cold ROM feature is untested by every benchmark run in
this repository.** Dense archive search, archive graph expansion and the
threshold fix itself have never executed on more than five questions. None can
be credited or blamed for any score reported here.

The archive is not, however, redundant. Measured directly across eight fully
ingested stores, it holds facts that Hot RAM has already pruned:

| | archive entities | Hot RAM entities | archive-only |
|---|---|---|---|
| typical store | 622–802 | 523–675 | **78–127 (11–16%)** |

Every archive-only record sampled was `ActiveContext` — the tier that decays —
and they are exactly the dated specifics LoCoMo asks about
(`james_current_game_witcher_3`, `james_cooking_class_cost`). Roughly one fact in
eight lives only in the archive, and the door stays shut on it.

The gate is the defect. `theta` compares a **topic** similarity against a
threshold and infers answer presence from it. Working memory answers "do we hold
anything about gaming?" with a confident 0.75 via `james_current_gaming_momentum`
while the fact the question needs sits unread in the archive. A gate that tested
answer presence rather than topic overlap would open the door on the questions
that need it; that change is not implemented, and it costs one extra model call
per question.

#### Three ablations, three negative results

Every mechanism added since the last release ships with an off-switch, which is
the only reason the section above could be written at all. All three switchable
additions were measured against `neuromorphic_tuned` on the same 303 questions,
paired, with McNemar exact tests:

| change | switch | accuracy | vs control | p |
|---|---|---|---|---|
| control | — | 0.525 | — | — |
| Tiered extraction prompt | `NMAFC_EXTRACTOR_VARIANT=tiered` | 0.451 | −8 pts (+9/−33) | **0.0003** |
| Dense + graph archive search | `cold_semantic_fallback=true` | 0.472 | ±0 (+3/−3) | 1.0000 |
| Clustering decay protection | `beta=0.5` | 0.477 | −4.8 pts (+18/−33) | **0.049** |

**Tiered extraction** tightened the classification rules so fewer facts land in
`CoreAnchor`. It works as designed — `CoreAnchor` share falls from ~82% to ~33% —
and costs eight accuracy points. Decay that actually runs is decay that actually
deletes answers.

**Dense archive search** was measured at `theta = 0.65` so the branch executed on
32% of questions rather than 1.6%. It is a dead heat against keyword-only search
and 17% slower. Note the scope: both conditions had the archive *open*, so this
compares two ways of searching it, not archive-vs-no-archive. The latter is still
untested.

**Clustering decay** scales `lambda` by `(1 - beta * C)`. The damage is not
uniform: single-hop falls 0.628 → 0.372 while multi-hop rises 0.385 → 0.462, and
retrieved context *falls* 1,571 → 1,351. The reading is that protection accrues
to densely-linked generic facts (`james_likes_gaming`) while the sparsely-linked
specifics (`witcher_3_march_2022`) lose it and are pruned — trading exactly the
records single-hop questions need for connective tissue. The falling context
count is the evidence: pure protection would raise it. Mechanism not confirmed by
inspecting survivors; treat as hypothesis.

None of these are arguments that the mechanisms are wrong in principle. They are
arguments that on retrospective-QA benchmarks, where every question is asked
after the conversation ends and nothing is ever re-retrieved mid-conversation,
forgetting has no upside to trade against its cost.

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

#### Ingestion checkpoints

The checkpoint above is per *conversation*: a conversation counts as done only
once every question against it has been answered. Ingestion is the expensive
half — one LLM extraction call per exchange, 20–30 minutes for a long
conversation — and none of it was recorded. An interruption at exchange 168 of
211 therefore discarded 168 extraction calls and restarted that conversation from
zero. A connection drop at 00:21 on 2026-08-18 destroyed a complete overnight run
that way, with both conversations inside four minutes of finishing.

`--ingest-checkpoint-every N` (default 25, `0` disables) records progress mid
conversation. Stores move from `tempfile.mkdtemp` to `<output>/stores/<arm>__<conv>/`,
which is what makes them findable by a later process at all — and incidentally
reusable by the `_ab_*` harnesses, which skip ingestion entirely and cost ten
minutes rather than thirty.

Two guards matter more than the feature:

**The fingerprint.** A store half-built under `beta=0.5` must never be finished
under `beta=0`. That failure is silent — one arm ingested under two decay
configurations, reporting a plausible number that answers no question. Decay
overrides, `NMAFC_EXTRACTOR_VARIANT` and the arm name are hashed into the state
file, and any mismatch discards the checkpoint and re-ingests.

**The turn clock.** `NeuromorphicMemory._current_turn` is in-memory and starts at
0. Reopening a store without restoring it would leave records stamped
`created_at_turn=168` in a system that believes no turns have elapsed, and every
subsequent decay and reinforcement calculation would run against a clock 168
turns behind its data. It is persisted alongside the exchange count.

Everything else fails toward re-ingesting: a missing, truncated, stale-version or
wrong-conversation state file reads as "no checkpoint". Losing twenty minutes is
the cheap outcome; accepting a bad checkpoint is the expensive one.

#### What a resume must not do

Three failures found on a full 1,986-question run, all invisible until a
restart, all now covered by `tests/unit/test_resume_integrity.py` and
`tests/unit/test_network_resilience.py`:

| Failure | Symptom | Fix |
|---|---|---|
| A conversation that died mid-ingestion was checkpointed as `rows=[]` | Marked *complete with zero questions*: its results vanished and every later resume skipped it. Four conversations — 864 of 1,986 questions — silently disappeared from an arm. | Failures are no longer checkpointed at all, and `_load_checkpoints` discards empty entries so checkpoints written by the old code repair themselves. |
| The judge phase filtered only on `not row["error"]` | A resume with nothing left to answer still re-graded every restored answer: ~40 min and a full arm's quota per completed arm, recomputing verdicts already on disk. | Skip rows that already hold a verdict. `judge_correct is None`, not a falsy check — `False` is a real verdict, and re-grading every wrong answer means re-grading over half the set. |
| Cost samples lived only on the worker objects | A resumed arm's workers answer nothing, so it reported 0 ms latency and 0 tokens — losing the compression claim (431 context tokens against 20,006) that the run exists to measure. | Checkpoints now persist the per-call samples, not their averages, since p50/p95 cannot be recovered from a mean. Older checkpoints rebuild latency and context exactly from the per-question rows; prompt and completion counts were never written there and stay empty rather than being inferred from context size. |

The pattern behind all three: a checkpoint is a claim about what already
happened, so anything it fails to record is silently assumed not to have
happened. Recording *less* than the run measured is not a conservative default.

#### Live logs

A full run is 1,986 questions across four arms and takes hours. `--log-file`
mirrors every console line to a file, timestamped and flushed per line, so the
file can be left open in an editor and watched as it grows:

```bash
python -u -m scripts.benchmarks.run_locomo   --arms raw,rag,neuromorphic,neuromorphic_tuned   --max-hops 0 --checkpoint resume   --log-file logs/locomo_full.log   --output scripts/benchmarks/results/locomo_full/
```

The log is append-mode, so a resumed run extends it rather than truncating the
record of what came before. stderr is mirrored too: a traceback that only
reached the console would be lost as soon as the terminal scrolled, which is
precisely the failure a log file exists to catch.

Three progress signals keep it moving rather than silent:

| Signal | Interval | Why it exists |
|---|---|---|
| ingestion heartbeat | 60 s | Memory-arm ingestion is one LLM extraction per exchange and prints nothing for 20+ min on a 345-exchange conversation. Reports exchanges processed, facts extracted and an ETA, polled from `current_turn` and the Hot RAM count rather than threaded through every arm. |
| answering progress | every 25 questions | Questions run sequentially within a conversation (retrieval reinforces memory, so they cannot be parallelised against one arm), which on the raw arm is an 8-15 min silence. |
| judge progress | ~10 updates per batch | Judging ~2,000 answers is a single `asyncio.gather` that runs 20+ min; without a callback it is indistinguishable from a hang. |

Each line carries elapsed time and a projected remainder, so "slow" is
distinguishable from "stuck" without attaching a debugger. The older
`scripts/benchmarks/live_progress.py` polls a run from outside (per-worker
`memory_event_log` counts in the temp storage dirs) and remains useful for a run
already launched without `--log-file`:

```bash
python -u scripts/benchmarks/live_progress.py   --log logs/locomo_full.log   --out scripts/benchmarks/results/live.log
```

#### Recording the retrieval settings

`--max-hops` sets Spreading Activation depth for both memory arms (`0` disables
graph traversal entirely). It is written into `results.json` metadata alongside
`theta`, and rendered into the generated `SUMMARY.md`, because two results files
with identical arms can differ by this alone: the paired A/B above measured
2-hop traversal costing **3.5x the context for no accuracy gain**. A reader
comparing runs needs to see which setting produced which number rather than
assume the default.

#### Charts

`visualize.py` renders a results file into publication-ready Plotly figures --
accuracy by question type, token cost, context-injection size, latency, a
combined dashboard, and a `SUMMARY.md` carrying the run metadata:

```bash
python -m scripts.benchmarks.visualize   --input scripts/benchmarks/results/locomo_full/results.json
# --format svg|png|pdf for static output; html is the default
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
`--max-hops` (Spreading Activation depth; `0` disables it), `--log-file` (live
timestamped log), `--output` (checkpoint + results directory). All default from environment
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
│   ├── wrapper.py                 # Top-level NeuromorphicMemory class (async + context manager)
│   ├── cli.py                     # Unified CLI (nmafc start/init/chat)
│   ├── schemas/
│   │   ├── memory.py              # Pydantic models (MemoryRecord, DecayConfig, etc.)
│   │   └── events.py              # EventType enum + MemoryEvent model
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
│   ├── storage/
│   │   ├── config.py              # NMafcConfig + TOML parsing
│   │   ├── hot.py                 # HotStorage (LanceDB, supports S3)
│   │   ├── cold_base.py           # Abstract ColdStorageBase interface
│   │   ├── cold.py                # ColdStorage (SQLite + FTS5)
│   │   ├── cold_pg.py             # PostgresColdStorage (PostgreSQL + tsvector)
│   │   └── event_log.py           # SQLite-backed cognitive event log
│   └── web/
│       ├── app.py                 # FastAPI app factory + lifespan + dotenv
│       ├── deps.py                # Dependency injection
│       ├── ws.py                  # WebSocket ConnectionManager
│       └── routes/
│           ├── memory.py          # Memory explorer endpoints
│           ├── graph.py           # Entity graph endpoints
│           ├── events.py          # Event log endpoints
│           ├── decay.py           # Decay curve endpoints
│           ├── config.py          # Config endpoints
│           └── process.py         # Process/ingest/rollback endpoints
├── web-ui/                        # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx         # Root layout with Sidebar + WebSocketProvider
│   │   │   ├── page.tsx           # Dashboard
│   │   │   ├── memory/page.tsx    # Memory Explorer
│   │   │   ├── graph/page.tsx     # Entity Graph (D3 force-directed)
│   │   │   ├── decay/page.tsx     # Decay Curves (Recharts)
│   │   │   ├── events/page.tsx    # Event Timeline
│   │   │   └── docs/page.tsx      # In-app documentation
│   │   ├── components/layout/
│   │   │   ├── Sidebar.tsx        # Navigation sidebar
│   │   │   └── WebSocketProvider.tsx  # WS context provider
│   │   ├── hooks/useWebSocket.ts  # WebSocket hook
│   │   └── lib/
│   │       ├── types.ts           # TypeScript types mirroring Pydantic models
│   │       └── api.ts             # API client (relative paths, proxy-friendly)
│   └── next.config.ts             # API proxy rewrites (/api/*, /ws/*)
├── examples/
│   ├── minimal.py                 # 10-line hello world
│   ├── custom_provider.py         # Implement LLMProvider + EmbeddingProvider
│   ├── manual_ingestion.py        # Bootstrap memory without LLM
│   └── sync_usage.py              # Non-async usage with SyncNeuromorphicMemory
├── configs/
│   └── default.toml               # Default configuration
├── scripts/benchmarks/            # Academic benchmark suite
│   ├── datasets/                  # LoCoMo + LongMemEval loaders
│   ├── arms/                      # 4 benchmark conditions
│   │   ├── base.py                # Shared answer-format rules + build_exchanges()
│   │   ├── raw_llm.py             # Baseline: full transcript in context
│   │   ├── rag.py                 # Chunked retrieval baseline
│   │   ├── stateful_nodecay.py    # Retired control: decay/pruning off
│   │   ├── neuromorphic.py        # Full NMAFC, published defaults
│   │   └── neuromorphic_tuned.py  # Same, lambda_active_context = 0.005
│   ├── evaluation/                # F1 + LLM-as-judge metrics
│   ├── resilience.py              # Rate limiter + retry/backoff wrappers
│   ├── live_progress.py           # Appends progress lines during long runs
│   ├── run_locomo.py              # LoCoMo CLI runner (parallel + checkpointed)
│   ├── run_longmemeval.py         # LongMemEval CLI runner
│   └── visualize.py               # Publication-ready Plotly charts
├── tests/                         # pytest suite (unit + integration)
├── .env.example                   # All provider credential variables
└── pyproject.toml                 # Package metadata + dependencies
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
