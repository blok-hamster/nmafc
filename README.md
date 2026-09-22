# NMAFC — Neuromorphic Memory Architecture for Conversational AI

A biologically-inspired stateful memory system for LLM agents. NMAFC gives conversational AI the ability to remember, forget, and prioritize information the way biological memory does — using exponential decay, spaced repetition, override suppression, and active pruning.

Unlike context-window stuffing or naive vector stores that grow without bound, NMAFC maintains a bounded, high-signal memory that improves with use. Frequently accessed facts become permanent. Contradicted facts are immediately suppressed. Stale information naturally decays away.

## Where it stands

Full LoCoMo, all ten conversations, **paired against a RAG baseline in the same
window and against the same stores** — one row per question holding both arms'
answers, so every comparison is a McNemar exact test rather than two run totals
subtracted:

| | ours | RAG | |
|---|---|---|---|
| accuracy, four scored categories (n=1,539) | **71.4%** | 64.6% | **+6.8**, p = 5.5e-08 |
| **temporal** (n=321) | **71.7%** | 46.1% | **+25.5**, p = 1.1e-14 |
| **multi-hop** (n=96) | **57.3%** | 44.8% | **+12.5**, p = 0.0075 |
| rendered context per question | **963 t** | 1,461 t | **−34%** |

Beating a retrieval baseline on accuracy *while spending a third less context* is
the claim this project exists to make, and temporal and multi-hop are where it
lands. Single-hop and open-domain are honest ties. Adversarial is a clear loss
(−14.6) and the section on it explains why it is not fixable from the prompt.
Full table, every category, and the negative results are in
[Benchmark Suite](#benchmark-suite).

## Why NMAFC

| Problem | Current Approaches | NMAFC Solution |
|---------|-------------------|----------------|
| Context windows overflow | Truncate oldest messages | Hot RAM with bounded record count via cognitive decay |
| Contradictions persist | Old facts coexist with new ones | Override detection + temporal invalidation (validity windows, history preserved) |
| Everything treated equally | Flat vector stores | Three-tier typing: CoreAnchor (permanent), ActiveContext (moderate decay), EphemeralState (aggressive decay) |
| No concept of importance | Retrieval count ignored | Spaced repetition — each retrieval strengthens retention (LTP) |
| Retrieval misses related facts | Single-hop vector search | Spreading Activation graph traversal (multi-hop entity linking) |
| Unbounded retrieval context | Dump all matches into prompt | RRF reranking across Hot + Cold + graph → bounded `rerank_top_k` output |
| No recoverability | Mutable state only | Dual-track: Hot RAM (fast, mutable) + Cold ROM (append-only event log, full rollback) |
| Expensive per-turn overhead | Separate extraction + response calls | Single LLM call with tool-use for simultaneous response + extraction |

## Core Mechanisms

### 1. Three-Tier Memory Classification

Every extracted fact is classified by the LLM into one of three tiers, each with distinct decay behavior:

| Tier | Decay Rate (lambda) | Half-life | Examples |
|------|---------------------|-----------|----------|
| **CoreAnchor** | 0.0 (never decays) | Infinite | Name, allergies, identity, relationships |
| **ActiveContext** | 0.005 per turn | ~139 turns | Current goals, schedules, projects |
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

A fact retrieved 10 times retains 80% weight after 200 turns vs. 37% for a never-retrieved fact. This naturally surfaces important information.

### 4. Override Detection & Temporal Invalidation

When the LLM detects a contradicting fact (e.g., "I moved to Berlin" contradicts "I live in Paris"):

1. New fact specifies `overrides_entity` pointing to the old record
2. Old record is marked `invalid_at = current_turn` (temporal invalidation)
3. Old record's weight is frozen — decay skips invalidated records
4. Invalidated records are excluded from search by default (`exclude_invalidated=True`)

Zero hallucination for contradictions — invalidated facts cannot be retrieved. Unlike the previous weight-suppression approach, the temporal history is preserved: both "lived in Paris (turn 1-5)" and "lives in Berlin (turn 5-present)" remain in storage for rollback and temporal queries.

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

### 8. Temporal Validity Windows

Every memory record carries `valid_at` (when the fact became true) and `invalid_at` (when it was superseded). These fields enable:

- **Temporal queries:** "What was true at turn N?" can be answered without rollback
- **History preservation:** "lived in Paris (turn 1-5)" and "moved to Berlin (turn 5-present)" coexist
- **Efficient decay:** invalidated records are skipped by `decay_all()` — their weight is frozen
- **Graceful pruning:** ActiveContext records that decay below `w_prune` are marked `invalid_at` rather than deleted, preserving archive access

### 9. Unified Parallel Search & RRF Reranking

Retrieval searches both Hot RAM and Cold ROM in parallel on every query (controlled by `always_search_cold`), removing the old theta-gated fallback that fired on only 1.6% of questions. Results from all sources — Hot vector hits, Cold semantic/keyword matches, and BFS graph expansion — are fused via Reciprocal Rank Fusion:

```
RRF_score(d) = Σ 1/(k + rank_in_list_i)   for each list containing d
```

This controls fan-out naturally: BFS can discover 35 records, but only `rerank_top_k` (default 20) survive fusion. Entity-name deduplication ensures Hot-sourced records take priority over archived versions. Optional additive signals (recency boost, weight signal) can fine-tune ranking.

### 10. Zep-Style Context Presentation

Retrieved facts are presented to the answering LLM using structured `<FACTS>` tags with temporal validity ranges, inspired by Zep's knowledge graph format:

```
<FACTS>
User's name is Joshua (Valid: turn 1 - present)
Lives in Berlin (Valid: turn 5 - present)
Started new job at Google (Valid: turn 12 - present)
</FACTS>
```

This gives the LLM scannable, atomic facts rather than a narrative list. Each fact carries its validity window, helping the model reason about temporal relationships.

### 11. Graded Hydration (source turns beside the facts)

The largest single addition in this branch. A compacted fact is a summary, and a
summary loses the detail the question wanted often enough to matter — the failure
analysis found the gold answer *reachable* far more often than it was *answered*.
Hydration puts the original conversation turns back alongside the facts they were
extracted from, so the model reads the evidence and not only the gist.

It is graded because turns are expensive. Measured at roughly 55 tokens per
hydrated turn against 30 per rendered fact, restoring everything blows the
context budget immediately. So:

| setting | what it controls |
|---|---|
| `hydrate_top_k` | how many ranked facts get their source turn pulled back |
| `hydrate_full_turns` | how many of the best turns are returned **whole** |
| `hydrate_lines` | for the rest, keep only the N speaker lines that match best |
| `hydrate_pool` / `hydrate_scan` | choose turns by question match rather than fact rank |

Depth where it pays, a pointer where it does not. `hydrate_pool` exists because
fact rank is the wrong chooser in one specific case: the turn that answers the
question can be one whose facts all ranked below the cut, and no increase in
`hydrate_top_k` reaches it. Selecting from a wider pool by BM25 against the
question hydrates the same *number* of turns and changes *which* ones.

### 12. The Answer-Type Gate

A question often names the category of its own answer — "in which **state**",
"what **console**", "how **old**" — and when it does, the retrieved evidence
usually sits one rung below it. Asked which state a shelter is in, the store held
"Stamford" and the answer given was "Stamford". That is evidence for the answer,
not the answer, and it was scored wrong.

This is not a retrieval failure and no amount of extra context fixes it: the
right fact was already in the window. It is caused by the answering rules, which
say to reuse the facts' exact wording because a synonym scores as a miss. That
rule is correct nearly everywhere and exactly wrong here, because "Connecticut"
is not a synonym for "Stamford" — it is the level the question asked for.

`integration/answer_type.gate(question)` returns a prompt rule plus a tag when,
and only when, the question names a type. Measured over 1,535 paired questions
before it existed: on the 238 that name a type we scored 56.7% against RAG's
61.8%, while on the other 1,297 we scored 66.7% against 64.9%. The entire deficit
lived in the typed questions.

Deliberately narrow on two counts. **Only when the question names the type** — a
directive derived from a question that named nothing is a guess, and a guess in
the prompt is indistinguishable from a hallucination in the answer. **Only a
level shift, never an invention** — it licenses naming the state a retrieved town
sits in, not naming a state when nothing retrieved points at one.

It is a regex over a string already in the prompt: no embedding, no model call,
no store access, so it costs nothing per query and fires on about 15% of them.

The companion `integration/list_shape.wants_list()` has the same form and is
**deliberately not wired in**; see the negative results in the benchmark section
for why.

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
                    └───────────┘  │ Unified │  │  │  Reinforce│  │
                                   │ Search  │  │  │   Prune   │  │
                                   │ + Graph │  │  │  Rerank   │  │
                                   │ + RRF   │  │  │  Consol.  │  │
                                   │ Rerank  │  │  │  Rollback │  │
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
2. Unified parallel retrieval:
   a. Hot RAM vector search (top_k=10, cosine similarity)
   b. Cold ROM hybrid search (semantic + keyword), always in parallel
   c. BFS graph expansion (Spreading Activation, max_hops=2) from Hot seeds
   d. One-hop graph expansion from Cold-only seeds into the archive
   e. Reciprocal Rank Fusion (RRF) across all candidate lists → rerank_top_k winners
3. Format retrieved memories as structured <FACTS> context with validity ranges
4. LLM call with tool-use → simultaneous response + state extraction
5. For each extracted update:
   a. Log to Cold ROM (append-only)
   b. Detect overrides → set invalid_at on contradicted records (temporal invalidation)
   c. Embed new fact → upsert to Hot RAM (weight=1.0, valid_at=current_turn)
6. LTP reinforcement on Hot-sourced records that survived reranking
7. Decay all mutable records: w(t) = w(t0) * e^(-lambda * dt)
8. Prune: delete records where weight <= w_prune
9. Every N turns: REM consolidation (elevation + dead pointer cleanup)
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
# Hot RAM: [ActiveContext] user_location: "Moved to Berlin" (weight=1.0, valid_at=5)
# The old "Paris" record is marked invalid_at=5 (excluded from search, preserved for history)

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
[storage]
hot_uri = "./data/lancedb"     # LanceDB path (supports s3://)
cold_uri = "./data/cold.db"    # SQLite path

[decay]
lambda_core_anchor = 0.0       # CoreAnchor never decays
lambda_active_context = 0.005  # Slow decay (~460 turn horizon, outlasts LoCoMo conversations)
lambda_ephemeral = 0.69        # Aggressive decay (~1 turn half-life)
eta = 0.15                     # Consolidation constant (higher = faster LTP effect)
gamma = 0.1                    # Override suppression multiplier
w_prune = 0.1                  # Eviction threshold (records at or below are deleted)

[retrieval]
theta = 0.45                   # Cosine similarity threshold for Cold ROM fallback
top_k = 10                     # Vector search result count
fallback_keyword_limit = 20    # Cold ROM FTS5 result limit
always_search_cold = true      # Search Cold ROM in parallel (bypasses theta gate)
rrf_k = 60                     # RRF fusion constant (higher = less weight to top ranks)
rerank_top_k = 20              # Max records surviving reranking into prompt

[time]
unit = "turns"                 # Decay time unit

[embedding]
provider_model = "openai/text-embedding-3-small"
dim = 1536                     # Auto-detected on init

[llm]
provider_model = "openai/gpt-4o-mini"
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

### Answer Prompt Engineering

LoCoMo gold answers are 1-4 word noun phrases. A correct but verbose reply ("Both Jon and Gina like to destress through dancing together on weekends") scores F1≈0.042 against gold "by dancing". This is a formatting artifact, not a memory result.

**Fill-in-the-blank framing:** The answering prompt tells the model it is completing a quiz scored by exact token overlap with a 1-5 word reference. Few-shot examples demonstrate the expected format for dates, lists, yes/no, and names.

**`strip_answer()` post-processing:** A regex-based pipeline in `arms/base.py` removes:
- Preamble patterns ("Based on the facts...", "According to my memories...")
- Meta/negation sentences ("I don't have information about...", "However...")
- Trailing validity annotations and explanations
- Markdown formatting and numbered lists

**Anti-refusal instruction:** "NEVER say 'No information available' — always attempt an answer from the facts, even if uncertain." Refusal was the dominant failure mode: 54% of wrong answers were the model declining to answer, and in 30% of those cases the correct fact was in the retrieved context.

Applied identically to all arms, so it changes absolute numbers without advantaging any arm over another.

### Current results

**Status: paired full run, 10 September.** All 10 LoCoMo conversations, 1,985 of
the 1,986 QA pairs, no sampling. Both arms answered **in the same window against
the same persisted stores**, one row per question holding both predictions, so
every comparison below is paired and McNemar exact applies directly to it rather
than to two runs subtracted from each other. Answering
`azure_v1/DeepSeek-V4-Pro`, embeddings `azure_v1/text-embedding-3-small`, judge
`azure_v1/Kimi-K2.6`.

#### Which questions are scored

LoCoMo ships 1,986 QA pairs in five categories. Published comparisons (Mem0,
Zep, MemMachine and others) report on 1,540, dropping `adversarial`. That
category was meant to hold unanswerable questions, but the released gold answers
are ordinary facts: **444 of the 446 carry a real answer** rather than "not
mentioned", so a system that correctly declines is marked wrong.

We report **both denominators, and we report adversarial paired against RAG**,
because dropping a category we lose on is not a measurement decision. The 1,539
figure is the one comparable to the literature. The 1,985 figure is the one that
says what the system would do if you pointed it at the whole file.

#### Headline

| set | n | ours | RAG | lead | McNemar exact |
|---|---|---|---|---|---|
| **scored four categories** | 1,539 | **71.4%** | 64.6% | **+6.8** | +238/−133, **p = 5.5e-08** |
| adversarial | 446 | 35.9% | 50.4% | −14.6 | +38/−103, p = 4.1e-08 |
| all five | 1,985 | 63.4% | 61.4% | +2.0 | +276/−236, p = 0.085 |

| | ours | RAG | |
|---|---|---|---|
| rendered context, mean | **963 t** | 1,461 t | **−34%** |

The context figure is the one that constrains everything else. The design target
was to beat RAG **at or under ~1,000 rendered tokens**, and the shipped
configuration sits at 963. Several changes below buy accuracy and are not
shipped for exactly this reason.

#### Per category

| category | n | ours | RAG | lead | McNemar exact | ours tok | RAG tok |
|---|---|---|---|---|---|---|---|
| **temporal** | 321 | **71.7%** | 46.1% | **+25.5** | +101/−19, **p = 1.1e-14** | 943 | 1,422 |
| **multi-hop** | 96 | **57.3%** | 44.8% | **+12.5** | +15/−3, **p = 0.0075** | 961 | 1,410 |
| single-hop | 281 | 50.9% | 45.6% | +5.3 | +51/−36, p = 0.133 | 980 | 1,486 |
| open-domain | 841 | 79.8% | 80.3% | −0.5 | +71/−75, p = 0.804 | 967 | 1,461 |
| adversarial | 446 | 35.9% | 50.4% | −14.6 | +38/−103, p = 4.1e-08 | 960 | 1,486 |

**What is defensible from this table:** temporal, multi-hop, and the context
saving. Temporal is the largest effect in the run and the one a structured
memory ought to win — "when did X happen" needs dated events, and dated events
are what the extractor produces. Multi-hop is significant despite n=96.

**What is not:** single-hop and open-domain are ties. The +5.3 on single-hop
looks like a win and is not one at p=0.133; the −0.5 on open-domain looks like a
loss and is not one either. Neither should be quoted as a result. **The overall
+2.0 across all five categories is p=0.085 and is not significant** — the honest
headline is the scored-set +6.8 plus the 34% context saving, not the all-five
number.

Detectable effect sizes, so the ties above can be read for what they are: at 40
discordant pairs this design can see 1.7 points, at 150 it can see 3.1. The
single-hop and open-domain gaps are inside that band.

#### Where the loss is: adversarial, and it is not recoverable from the prompt

Adversarial is the one clear loss, and it is large enough to erase most of the
scored-set lead when the categories are pooled. The obvious theory is that we
refuse too much. Measured, that theory does not hold: **RAG refuses on
adversarial at a rate between comparable to ours and higher than ours** —
different refusal detectors disagree on the exact rate, which is itself a
warning about quoting one — **and still scores 14.5 points above us**. Roughly a
third of the gap is recoverable refusal; the rest is wrong answers on questions
we committed to, which no prompt instruction reaches.

Two prompt clauses were built and measured against this over the full 1,984, and
both are dead:

| clause | adversarial | temporal | all five |
|---|---|---|---|
| `commit_short` | **+5.4**, p = 0.0022 | **−4.0**, p = 0.0024 | +0.0, +72/−72 |
| `commit_exact` | +0.2 | **−3.4**, p = 0.0074 | worse than both |

`commit_short` buys adversarial and pays for it exactly, one for one, out of
temporal and multi-hop. `commit_exact` was an attempt to keep the gain without
the damage by removing the offending wording; it removed the gain instead and
kept the damage. Neither ships. The behaviour that answers an adversarial
question and the behaviour that dates a temporal one are welded together in this
model, and the join is not in the prompt.

#### Latency, and why the mean is the wrong statistic

Earlier revisions of this file compared latency across two runs on two different
nights against a shared provider quota, and got a nonsense answer out of it —
two runs of the *identical* RAG arm once came out 91% apart. The paired run
measures both arms interleaved in one window:

| | p50 | p75 | p90 | p99 | mean |
|---|---|---|---|---|---|
| ours | 1,741 ms | 2,407 ms | 3,559 ms | 26,100 ms | 2,721 ms |
| RAG | 1,769 ms | 2,398 ms | 3,346 ms | 12,019 ms | 2,295 ms |

**The two arms are indistinguishable through p90.** We are marginally faster at
the median and marginally slower at p90; neither gap is a result. The 19% gap in
the mean lives entirely in the tail, and both arms top out at ~93.8 s, which is a
provider retry ladder rather than anything either architecture computes.

An earlier revision of this section claimed the memory arm was "roughly 6× slower
for 1 point less accuracy", derived from exactly this mean on two unpaired runs.
That was wrong and has been removed. **Quote the median, and do not read a
latency delta out of a run whose arms were not interleaved.**

#### Negative results, kept because they cost real money to learn

Everything in this section was built, wired end to end, measured on a large
paired sample, and then not shipped. They are recorded here so nobody pays for
them twice.

**Width on open-domain: real, and still not shippable.** Open-domain is the
category where we merely tie, so it got the most attention. Three arms, one
window, 830 paired questions:

| arm | accuracy | tokens | vs shipped | vs RAG |
|---|---|---|---|---|
| wide (hydrate 10, grounding 0.003) | 81.3% | 1,300 t | +1.6, +35/−22, p = 0.111 | +1.2, p = 0.426 |
| grounding only (0.003) | 80.6% | 1,047 t | +0.8, +31/−24, p = 0.419 | +0.5, p = 0.797 |
| shipped | 79.8% | 968 t | baseline | −0.4, p = 0.867 |
| RAG | 80.1% | 1,464 t | | |

Two findings. Splitting the knobs shows **facts are about twice as
token-efficient as hydrated turns** — grounding alone gets half the gain for a
sixth of the token cost. And neither arm is significant. But the reason neither
ships is simpler than significance: **applying width to open-domain only
requires knowing the category, and the category is not knowable at inference
time.** It is a label in the benchmark file, not a property of the question. A
policy built on it is a benchmark artefact, not a system.

**The list gate: a real signal, and width is not the answer to it.**
`integration/list_shape.py` detects, from the question string alone, whether it
asks to enumerate ("What activities does Melanie partake in") rather than to
name one thing. Unlike a category, this **is** knowable at inference. It fires
on 327 of 1,985 questions, and the shape it finds is real: we score **50.2% on
the questions it fires on against 66.0% on the ones it does not**.

Handing those questions more context does not close it. Every firing question
answered both ways, 322 of them: wide 50.9% against the shipped 49.4%, +17/−12,
**p = 0.458**, for 324 extra tokens each. Single-hop, the category the gate was
built from, goes *backwards* at −1.4. The module is kept, tested and documented
because the detection is sound and the gap is worth attacking; it is wired into
no shipping path.

**Sample size is the recurring lesson.** An earlier read of that same gate, on
108 questions, said +5.6 and looked like the best result of the cycle. It was
nine coin flips landing the same way. Four separate changes in this project have
been mispriced by a small sample — a 6-question token estimate that came in 292 t
high, an 8-question clause read, the +5.6 above, and a 281-question precision
screen that could not see the miss it caused. **Nothing under roughly 800
questions is believed here**, and the harness is built to run paired arms at full
scale for that reason.

#### Why answers are wrong: refusal, not just retrieval

`_diagnose_retrieval.py` replays retrieval against the persisted stores and
checks whether the gold answer was in the context the model actually received —
no regeneration, no judging, one embedding call per question. It was run against
an earlier configuration, and the diagnosis is what produced the answer prompt
described above. Of the wrong answers on the scored categories at that time,
**more than half were the system declining to answer**, and in 30% of all wrong
answers it declined with the correct fact in front of it.

The absolute rates there are superseded by the run above and are not repeated.
What survives is the method and two consequences that still hold:

- The **answer prompt is a first-class lever**, not a detail, and it is the
  cheapest one to test because the persisted stores make retrieval-phase
  experiments nearly free.
- Refusal analysis has to precede retrieval analysis. A category can look like a
  retrieval failure while being an answering failure, and the fix for one does
  nothing for the other. Multi-hop was exactly this: it looked flat because half
  of it ended in a refusal before the graph's output ever reached an answer.

An earlier revision of this section claimed the system "confabulates rather than
declining to answer" on adversarial. That was **backwards**, and correcting it is
what produced the refusal analysis in the first place.

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
- **Arms are run paired, in one window, against the same stores.** The harness in
  `_ab_budget.py` writes one row per question holding *both* arms' predictions,
  rather than running two arms separately and subtracting the totals. This is
  what makes McNemar exact applicable, and it removes drift between runs from the
  comparison entirely. Our own arm re-run against its saved output moves 0.1
  points at n=841 (+21/−22), so anything the paired design reports at that scale
  is signal rather than run-to-run wobble.
- **Reading the store must not write to it.** Retrieval reinforces what it
  returns, so an analysis pass that simply reads back a store changes it. Any
  read-only script calls `close_readonly()`, which drops pending reinforcements
  before closing. Without this, the persisted `k` and `last_reinforced_turn`
  values describe how many times the benchmark ran, not how the framework
  behaves.
- **Nothing mutates a store in place.** Every experiment that changes state
  copies the store first. The ten indexed LoCoMo stores are expensive enough to
  rebuild (~5 hours of paid ingestion) that they are treated as immutable inputs.

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

Per-question results for the runs the README cites are in the repository, so the
tables can be recomputed rather than taken on trust:

| path | contents |
|---|---|
| `results/paired_2026_09_10/` | **the current run.** Both arms on the same row, 1,985 questions, plus the seven A/B arms |
| `results/paired_2026_09_10/summarise.py` | regenerates every table in the results section above. Standard library only, no API calls |
| `results/full_v2/results.json` | the earlier unpaired run — both memory arms, 1,986 rows each |
| `results/full_v2/checkpoint_*.json` | same rows grouped by conversation, as written during the run |
| `results/full_v2.log` | that run's console log, including the four network blips it recovered from |
| `results/locomo_full/` | the original baseline run — all four arms |

Each row carries the question, category, gold answer, prediction, judge verdict,
context characters and latency. The paired files additionally carry the *other*
arm's prediction on the same row, which is the entire point of them — see
`results/paired_2026_09_10/README.md` for the column layout and the one join
trap the column names invite.

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
| `_screen_*.py`, `_sweep_*.py` | does a retrieval setting change whether the gold answer is *reachable* | 1 embedding/question |
| `_probe_stored_vs_reached.py` | is the fact missing from the store, or present and unranked | none |

The `_screen_` and `_sweep_` family is the reason this project could test as many
configurations as it did. Retrieval is deterministic given a store, so a setting
can be scored on reachability without generating a single answer: about four
minutes per configuration against roughly an hour and a half of paid generation.
Screening is not a substitute for the real thing — reachable is not answered —
but it eliminates the settings that cannot possibly help before any money is
spent.

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

#### …and traversal is worth having, after two wrong readings of it

This feature was measured three times and the first two readings were both
wrong. The sequence is worth keeping because the errors are the ordinary ones.

**Reading one (wrong): "traversal does not help."** A paired A/B on conv-26,
store copied, only `max_hops` differing, gave 6/13 with the graph on against
7/13 with it off, and 11 of the 13 answers byte-identical despite ~1,100 extra
tokens. n=13, on the single conversation the feature was developed against.
Scoping the test to the category the feature was *designed* for sampled the
smallest stratum in the whole benchmark.

**Reading two (wrong in the other direction): "largest gain in the project."**
Widened to all 303 questions of conv-26 and conv-30, `max_hops` 0 → 2 moved
accuracy 0.472 → 0.525, +32/−16, p = 0.029, with the gain landing mostly on
**single-hop** rather than multi-hop. True of those two conversations. They are
also the two the retrieval settings were tuned on, so it was partly reading its
own tuning back.

**Reading three (the one that stands).** In the full paired run, multi-hop
scores **57.3% against RAG's 44.8%, +15/−3, p = 0.0075**, at 961 rendered
tokens. Traversal earns its place, and it does so without the fan-out blowup
that made the earlier readings alarming.

The fan-out concern was real when it was raised and is now bounded. Unrestricted
2-hop BFS over a store where 95% of records carry links returned 34.8 records
and 1,585 tokens — **more context than the RAG baseline it is supposed to
beat**. The shipped path does not do that: expansion is score-ranked and capped,
and the whole retrieval budget is reranked down before rendering, which is why
the measured multi-hop context is 961 tokens rather than 1,585.

Two general lessons, both of which cost money to learn:

- **A feature evaluated only on the questions its designer expected it to serve
  is not evaluated.** Traversal's largest benefit was on single-hop, where the
  neighbour of a matched fact turns out to be the detail the question wanted.
- **Do not read a latency delta out of a run whose arms were not interleaved.**
  The apparent 4,700 ms vs 1,939 ms gap here was run order: the first condition
  paid embedding cold-start and LanceDB warmup.

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

#### …then the gate was removed entirely, which is what shipped

**Superseded.** The two sections above describe a fallback *gated* on `theta`.
The shipped system has no such gate. `always_search_cold` defaults to `true` and
Cold ROM is searched **in parallel with Hot on every query**, with the two result
sets merged and reranked together. `theta` still exists in config and no longer
decides anything on the default path.

The reasoning is in the numbers those sections produced. Gated on `theta = 0.45`
the archive was consulted on 1.6% of questions, which made every Cold ROM
feature — dense archive search, archive graph expansion, the distance-metric fix
itself — untested by every benchmark in the repository. And the archive is not
redundant: measured across eight fully ingested stores it holds **78–127
entities (11–16%) that Hot RAM has already pruned**, and every archive-only
record sampled was `ActiveContext`, the tier that decays. They are exactly the
dated specifics LoCoMo asks about (`james_current_game_witcher_3`,
`james_cooking_class_cost`).

The gate itself was the defect, and not because it was mistuned. `theta`
compares a **topic** similarity and infers answer presence from it. Working
memory answers "do we hold anything about gaming?" with a confident 0.75 while
the fact the question actually needs sits unread in the archive. No threshold
value fixes that, because the quantity being thresholded is the wrong quantity.
Opening the door unconditionally and letting the reranker decide is cheaper than
a gate that tests answer presence properly, which would cost a model call per
question.

One counter-intuitive consequence, measured: **widening the Cold budget hurts.**
Once the archive is searched on every query, giving it more slots displaces Hot
RAM hits that were already right. The lever that matters on the merged path is
`rerank_top_k`; `rrf_k`, `max_hops` and the decay weight all screened as no
better than the default.

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

All three were measured on 303 questions of two conversations. Read them as
directional. The full paired run is the only measurement in this file large
enough to settle a small effect.

#### Screening retrieval settings without paying for generation

Most configuration questions do not need answers generated at all. Retrieval is
deterministic given a store, so a config can be scored on **whether the gold
answer reached the context** rather than on whether the model then said it. That
turns a config sweep from roughly an hour and a half of paid generation into
about four minutes of embedding calls, which is why the settings below could be
swept at full scale instead of sampled.

What the sweep found is mostly that there is nothing to find. `rerank_top_k` is
the only setting that moves reachability. `rrf_k`, `max_hops` beyond the default,
and the decay-weighted retrieval score all come back flat or negative.

**Weighting retrieval by decay state is the clearest negative.** The idea is
natural — a memory the system has reinforced should rank above one it has nearly
forgotten — and the switch had in fact never been on in any reported run. Turning
it on **costs about 5 points of reachability in every category**, with no
category spared. Recency of reinforcement is not evidence of relevance to the
question being asked, and on a retrospective benchmark it is close to
anti-correlated with it.

A caution that applies to every number in this section: **reading the store
mutates it.** Retrieval reinforces what it returns, so a single benchmark
question rewrites on the order of a dozen records. Stored reinforcement counts
and `last_reinforced_turn` values therefore measure the benchmark, not the
framework, and any read-only analysis has to disable the writeback first — see
`close_readonly` in `scripts/benchmarks/_ab_budget.py`.

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
| ActiveContext | 0.005 | 0.951 | 0.983 | 0.996 |
| EphemeralState | 0.69 | 0.001 | 0.056 | 0.314 |

### LTP Reinforcement (on retrieval)

```
w_i = 1.0                           # Reset to full strength
k_i = k_i + 1                       # Increment consolidation index
last_reinforced_turn = current_turn  # Reset decay clock
```

### Override (Temporal Invalidation)

```
old_record.invalid_at = current_turn    # Marks end of validity window
# Weight is frozen — decay_all() skips records with invalid_at set
# Record excluded from search when exclude_invalidated=True (default)
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
│   │   ├── events.py              # EventType enum + MemoryEvent model
│   │   └── code.py                # Symbol/file records for the code-memory path
│   ├── py.typed                   # PEP 561 marker for IDE type resolution
│   ├── engine/
│   │   ├── decay.py               # Ebbinghaus exponential decay
│   │   ├── reinforcement.py       # LTP (weight reset + k increment)
│   │   ├── pruning.py             # Override detection + temporal invalidation
│   │   ├── reranking.py           # Reciprocal Rank Fusion (RRF) reranker
│   │   ├── linking.py             # Resolve extracted links onto entities that exist
│   │   ├── consolidation.py       # REM sleep (elevation + cleanup)
│   │   └── rollback.py            # State reconstruction from Cold ROM
│   ├── integration/
│   │   ├── factory.py             # Provider factory (provider/model strings)
│   │   ├── base.py                # Abstract LLMProvider + EmbeddingProvider
│   │   ├── extractor.py           # StateExtractor (tool-use based extraction)
│   │   ├── query_router.py        # Unified parallel search, RRF rerank, hydration
│   │   ├── answer_type.py         # Question -> answer-shape prompt rule (SHIPPED)
│   │   ├── grounding.py           # Rank facts by evidence, not by the summary
│   │   ├── quantities.py          # Numbers as content, not as short stopwords
│   │   ├── list_shape.py          # "Does this ask for a list?" (measured, NOT wired)
│   │   ├── openai_provider.py     # OpenAI / OpenAI-compatible
│   │   ├── anthropic_provider.py  # Anthropic native
│   │   ├── bedrock_provider.py    # AWS Bedrock (boto3 + Anthropic SDK)
│   │   ├── azure_provider.py      # Azure OpenAI
│   │   └── fastembed_provider.py  # ONNX CPU embeddings
│   ├── code/                      # Code memory: exact symbols, no embeddings
│   │   ├── symbols.py             # Python `ast` -> definitions + references
│   │   ├── index.py               # Symbol graph: defined, referred to, changed
│   │   └── render.py              # Graded hydration under a hard token budget
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
│   ├── ingest_checkpoint.py       # Resume ingestion without re-paying for it
│   ├── run_locomo.py              # LoCoMo CLI runner (parallel + checkpointed)
│   ├── run_longmemeval.py         # LongMemEval CLI runner
│   ├── visualize.py               # Publication-ready Plotly charts
│   ├── _ab_budget.py              # Paired-arm harness: same window, same stores
│   ├── _run_open_domain_full.py   # Full-scale paired runner with gate filters
│   ├── _ab_*.py                   # Paired A/Bs, one per hypothesis
│   ├── _screen_*.py / _sweep_*.py # Retrieval-only screens (no generation, no cost)
│   └── _diagnose_*.py / _probe_*.py  # Failure analysis against persisted stores
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
| Temporal invalidation | Synaptic depression | Contradicted pathways are inhibited (not destroyed) |
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
