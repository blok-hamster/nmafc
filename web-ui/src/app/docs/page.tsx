"use client";

import { useState } from "react";
import {
  IconOverview,
  IconPlay,
  IconTerminal,
  IconApi,
  IconConfig,
  IconLibrary,
  IconGraph,
  IconProviders,
  IconDecay,
} from "@/components/icons";

type Section =
  | "overview"
  | "quickstart"
  | "cli"
  | "api"
  | "config"
  | "library"
  | "architecture"
  | "providers"
  | "decay";

const SECTIONS: { id: Section; label: string; Icon: React.ComponentType<{ className?: string }> }[] = [
  { id: "overview", label: "Overview", Icon: IconOverview },
  { id: "quickstart", label: "Quick Start", Icon: IconPlay },
  { id: "cli", label: "CLI Reference", Icon: IconTerminal },
  { id: "api", label: "API Endpoints", Icon: IconApi },
  { id: "config", label: "Configuration", Icon: IconConfig },
  { id: "library", label: "Library Usage", Icon: IconLibrary },
  { id: "architecture", label: "Architecture", Icon: IconGraph },
  { id: "providers", label: "Providers", Icon: IconProviders },
  { id: "decay", label: "Decay System", Icon: IconDecay },
];

export default function DocsPage() {
  const [active, setActive] = useState<Section>("overview");

  return (
    <div className="flex h-full">
      <nav className="w-48 shrink-0 border-r border-zinc-800 bg-zinc-950 py-4 overflow-auto">
        <p className="px-4 text-[10px] text-zinc-500 uppercase tracking-widest mb-3">
          Documentation
        </p>
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            onClick={() => setActive(s.id)}
            className={`w-full text-left px-4 py-2 text-sm flex items-center gap-2 transition-colors ${
              active === s.id
                ? "bg-zinc-800 text-zinc-100"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900"
            }`}
          >
            <s.Icon className="w-4 h-4 shrink-0" />
            {s.label}
          </button>
        ))}
      </nav>
      <div className="flex-1 overflow-auto p-8 max-w-4xl">
        {active === "overview" && <Overview />}
        {active === "quickstart" && <QuickStart />}
        {active === "cli" && <CLIRef />}
        {active === "api" && <APIRef />}
        {active === "config" && <ConfigRef />}
        {active === "library" && <LibraryUsage />}
        {active === "architecture" && <Architecture />}
        {active === "providers" && <Providers />}
        {active === "decay" && <DecaySystem />}
      </div>
    </div>
  );
}

function H({ children }: { children: React.ReactNode }) {
  return (
    <h1 className="text-2xl font-bold text-zinc-100 mb-4">{children}</h1>
  );
}

function H2({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-lg font-semibold text-zinc-200 mt-8 mb-3">
      {children}
    </h2>
  );
}

function H3({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-sm font-semibold text-zinc-300 mt-6 mb-2">
      {children}
    </h3>
  );
}

function P({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-zinc-400 mb-3 leading-relaxed">{children}</p>;
}

function Code({ children }: { children: string; lang?: string }) {
  return (
    <pre className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 mb-4 overflow-x-auto text-xs">
      <code className="text-zinc-300">{children}</code>
    </pre>
  );
}

function Table({
  headers,
  rows,
}: {
  headers: string[];
  rows: string[][];
}) {
  return (
    <div className="mb-4 overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-zinc-800">
            {headers.map((h) => (
              <th
                key={h}
                className="px-3 py-2 text-left text-zinc-400 font-medium"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-zinc-800/50">
              {row.map((cell, j) => (
                <td key={j} className="px-3 py-2 text-zinc-300 font-mono">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Callout({
  children,
  type = "info",
}: {
  children: React.ReactNode;
  type?: "info" | "warning" | "tip";
}) {
  const colors = {
    info: "border-blue-800 bg-blue-900/20 text-blue-300",
    warning: "border-amber-800 bg-amber-900/20 text-amber-300",
    tip: "border-emerald-800 bg-emerald-900/20 text-emerald-300",
  };
  return (
    <div className={`border rounded-lg p-3 mb-4 text-xs ${colors[type]}`}>
      {children}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
//  Section Components
// ═══════════════════════════════════════════════════════════════════

function Overview() {
  return (
    <>
      <H>NMAFC Documentation</H>
      <P>
        Neuromorphic Memory Architecture for Conversational AI — a
        biologically-inspired stateful memory system for LLM agents.
      </P>
      <P>
        NMAFC gives conversational AI the ability to remember, forget, and
        prioritize information the way biological memory does — using exponential
        decay, spaced repetition, override suppression, and active pruning.
      </P>

      <H2>Key Features</H2>
      <Table
        headers={["Feature", "Description"]}
        rows={[
          ["Three-tier memory", "CoreAnchor (permanent), ActiveContext (moderate decay), EphemeralState (fast decay)"],
          ["Cognitive decay", "Ebbinghaus forgetting curve — unused memories fade naturally"],
          ["Spaced repetition", "Each retrieval strengthens retention (LTP)"],
          ["Override detection", "Contradicted facts are immediately suppressed"],
          ["Spreading activation", "Multi-hop graph traversal retrieves related facts"],
          ["Dual-track storage", "Hot RAM (fast vectors) + Cold ROM (append-only archive)"],
          ["Web UI", "Visual memory explorer with live updates"],
          ["CLI tools", "Terminal chat, setup wizard, unified server startup"],
        ]}
      />

      <H2>System Components</H2>
      <Table
        headers={["Component", "Technology", "Purpose"]}
        rows={[
          ["Python library", "nmafc package", "Core memory engine for app integration"],
          ["FastAPI backend", "web module", "REST API + WebSocket live updates"],
          ["Next.js frontend", "web-ui/", "Visual explorer (Dashboard, Memory, Graph, Decay, Events)"],
          ["CLI", "nmafc command", "Unified start, init, chat commands"],
          ["Ollama / OpenAI", "Provider plugins", "LLM and embedding providers"],
        ]}
      />
    </>
  );
}

function QuickStart() {
  return (
    <>
      <H>Quick Start</H>

      <H2>1. Install</H2>
      <Code lang="bash">
{`# Core library
pip install nmafc[llm]

# With web UI + CLI
pip install nmafc[web,cli,llm]

# Everything
pip install nmafc[all]

# From source
git clone https://github.com/blok-hamster/nmafc.git
cd nmafc
uv pip install -e ".[all,cli]"
uv pip install --group dev`}
      </Code>

      <H2>2. Configure</H2>
      <Code lang="bash">
{`# Copy the example config
cp .env.example .env

# Edit .env — add your API key
# At minimum: OPENAI_API_KEY or GROQ_API_KEY or ANTHROPIC_API_KEY`}
      </Code>

      <H2>3. Run</H2>

      <H3>Option A: Web UI (recommended for exploration)</H3>
      <Code lang="bash">
{`# Start Ollama (for local embeddings)
ollama serve &
ollama pull nomic-embed-text

# Start everything
nmafc start

# Opens at http://localhost:3000`}
      </Code>

      <H3>Option B: Terminal chat (no web UI)</H3>
      <Code lang="bash">
{`nmafc chat
nmafc chat --llm "groq/llama-3.1-70b-versatile"`}
      </Code>

      <H3>Option C: Python library</H3>
      <Code lang="python">
{`import asyncio
from nmafc.wrapper import NeuromorphicMemory

async def main():
    async with await NeuromorphicMemory.from_config() as mem:
        response = await mem.process_turn("My name is Alice")
        print(response)

asyncio.run(main())`}
      </Code>

      <H3>Option D: Sync (no async)</H3>
      <Code lang="python">
{`from nmafc.wrapper import SyncNeuromorphicMemory

with SyncNeuromorphicMemory.from_config() as mem:
    response = mem.process_turn_sync("My name is Alice")
    print(response)`}
      </Code>

      <Callout type="tip">
        Ollama must be running if your config uses ollama/* models.
        Run `ollama serve` in a separate terminal before starting NMAFC.
      </Callout>
    </>
  );
}

function CLIRef() {
  return (
    <>
      <H2>CLI Reference</H2>

      <P>
        The <code>nmafc</code> command provides three subcommands for managing
        NMAFC.
      </P>

      <H2>nmafc init</H2>
      <P>
        Interactive setup wizard. Asks for LLM provider, embedding model, API
        keys, and storage paths. Generates <code>.env</code> and{" "}
        <code>configs/custom.toml</code>.
      </P>
      <Code lang="bash">{`nmafc init`}</Code>

      <H2>nmafc start</H2>
      <P>
        Starts the backend (FastAPI) and frontend (Next.js) servers. In dev
        mode, runs both on separate ports. In production mode, builds the
        frontend and serves everything from one port.
      </P>
      <Code lang="bash">
{`# Dev mode: backend :8000, frontend :3000
nmafc start

# Custom port
nmafc start --port 9000

# Production: single port
nmafc start --production

# Custom config
nmafc start --config configs/custom.toml`}
      </Code>
      <Table
        headers={["Flag", "Default", "Description"]}
        rows={[
          ["--port", "8000", "Backend port"],
          ["--host", "0.0.0.0", "Bind address"],
          ["--config", "configs/default.toml", "Config TOML path"],
          ["--production", "off", "Build frontend, serve from one port"],
        ]}
      />

      <H2>nmafc chat</H2>
      <P>
        Interactive terminal chat REPL. Processes messages through the full
        neuromorphic pipeline. Slash commands for introspection.
      </P>
      <Code lang="bash">
{`# Default config
nmafc chat

# Override model
nmafc chat --llm "groq/llama-3.1-70b-versatile"

# Custom config
nmafc chat --config configs/custom.toml`}
      </Code>
      <H3>Chat Commands</H3>
      <Table
        headers={["Command", "Description"]}
        rows={[
          ["/stats", "Show system stats (records, weights, events)"],
          ["/memory", "List all Hot RAM records"],
          ["/events", "Show recent cognitive events"],
          ["/rollback N", "Restore memory to turn N"],
          ["/quit", "Exit the chat"],
        ]}
      />
    </>
  );
}

function APIRef() {
  return (
    <>
      <H2>API Endpoints</H2>
      <P>
        The FastAPI backend exposes 27 REST endpoints + 1 WebSocket. All
        endpoints are prefixed with <code>/api</code>.
      </P>

      <H2>Health & Config</H2>
      <Table
        headers={["Method", "Endpoint", "Description"]}
        rows={[
          ["GET", "/api/health", "Health check"],
          ["GET", "/api/config", "Full NMafcConfig as JSON"],
          ["GET", "/api/stats", "Combined stats (Hot, Cold, Events)"],
        ]}
      />

      <H2>Memory Explorer</H2>
      <Table
        headers={["Method", "Endpoint", "Description"]}
        rows={[
          ["GET", "/api/memory", "Hot RAM stats (count, avg weight, types)"],
          ["GET", "/api/memory/all", "All records"],
          ["GET", "/api/memory/mutable", "Non-CoreAnchor records only"],
          ["GET", "/api/memory/search?q=...", "Vector similarity search"],
          ["GET", "/api/memory/entity/{name}", "Records for an entity"],
          ["GET", "/api/memory/{id}", "Single record by ID"],
        ]}
      />

      <H2>Entity Graph</H2>
      <Table
        headers={["Method", "Endpoint", "Description"]}
        rows={[
          ["GET", "/api/graph", "Full graph (nodes + edges)"],
          ["GET", "/api/graph/entity/{name}", "Entity detail with neighbors"],
        ]}
      />

      <H2>Event Log</H2>
      <Table
        headers={["Method", "Endpoint", "Description"]}
        rows={[
          ["GET", "/api/events", "Query events (filters: turn_from, turn_to, event_type, entity_name)"],
          ["GET", "/api/events/timeline", "Aggregated counts per turn"],
          ["GET", "/api/events/stats", "Event count by type"],
          ["GET", "/api/events/overrides", "Suppression events"],
          ["GET", "/api/events/consolidations", "Consolidation events"],
          ["GET", "/api/events/prunes", "Prune events"],
          ["GET", "/api/events/ltp", "LTP reinforcement events"],
          ["GET", "/api/events/entity/{name}", "Events for an entity"],
        ]}
      />

      <H2>Decay Curves</H2>
      <Table
        headers={["Method", "Endpoint", "Description"]}
        rows={[
          ["GET", "/api/decay/curves", "Projected curves for all mutable records"],
          ["GET", "/api/decay/curves/{id}", "Single record curve"],
          ["GET", "/api/decay/params", "Current decay hyperparameters"],
          ["GET", "/api/decay/compare?record_ids=...", "Compare specific records"],
        ]}
      />

      <H2>Process & Ingest</H2>
      <Table
        headers={["Method", "Endpoint", "Description"]}
        rows={[
          ["POST", "/api/process", "Process a user message (full pipeline)"],
          ["POST", "/api/ingest", "Ingest memory updates (no LLM call)"],
          ["POST", "/api/consolidate", "Trigger REM sleep consolidation"],
          ["POST", "/api/rollback", "Roll back to a specific turn"],
        ]}
      />

      <H2>WebSocket</H2>
      <Table
        headers={["Protocol", "Endpoint", "Description"]}
        rows={[
          ["WS", "/ws/live", "Real-time state change broadcasts"],
        ]}
      />
      <P>
        Messages broadcast: <code>turn_processed</code>,{" "}
        <code>memory_update</code>, <code>consolidation</code>,{" "}
        <code>rollback</code>.
      </P>
    </>
  );
}

function ConfigRef() {
  return (
    <>
      <H2>Configuration</H2>
      <P>
        NMAFC uses a 3-layer config system. Each layer overrides the previous.
      </P>

      <H2>Layer 1: TOML Config File</H2>
      <Code lang="toml">
{`# configs/default.toml
[storage]
hot_uri = "./data/lancedb"
cold_uri = "./data/cold.db"
event_log_uri = "./data/events.db"

[decay]
lambda_core_anchor = 0.0
lambda_active_context = 0.05
lambda_ephemeral = 0.69
eta = 0.15
gamma = 0.1
w_prune = 0.1

[retrieval]
theta = 0.45
top_k = 10
max_hops = 2

[llm]
provider_model = "openai/gpt-4o-mini"

[embedding]
provider_model = "openai/text-embedding-3-small"
dim = 1536`}
      </Code>

      <H2>Layer 2: Environment Variables</H2>
      <P>
        API keys and overrides. Auto-loaded from <code>.env</code> via{" "}
        <code>python-dotenv</code>.
      </P>
      <Table
        headers={["Variable", "Purpose", "Required"]}
        rows={[
          ["OPENAI_API_KEY", "OpenAI API auth", "If using OpenAI"],
          ["GROQ_API_KEY", "Groq API auth", "If using Groq"],
          ["ANTHROPIC_API_KEY", "Anthropic API auth", "If using Anthropic"],
          ["NMAFC_LLM_PROVIDER_MODEL", "Override LLM model", "No"],
          ["NMAFC_EMBEDDING_PROVIDER_MODEL", "Override embedding model", "No"],
          ["NMAFC_EMBEDDING_DIM", "Vector dimension", "If not 1536"],
          ["NMAFC_AGENT_ID", "Tenant isolation", "No (default: default)"],
          ["NMAFC_CONVERSATION_ID", "Conversation isolation", "No (default: default)"],
          ["NMAFC_HOT_URI", "Hot RAM path", "No"],
          ["NMAFC_COLD_URI", "Cold ROM path", "No"],
        ]}
      />

      <H2>Layer 3: CLI Flags</H2>
      <P>Highest priority — overrides both TOML and env vars.</P>
      <Code lang="bash">
{`nmafc start --port 9000 --config configs/custom.toml
nmafc chat --llm "groq/llama-3.1-70b-versatile"`}
      </Code>

      <Callout type="tip">
        Set NMAFC_EMBEDDING_DIM if using a non-1536-dim model (e.g. 768 for
        nomic-embed-text). Without it, the system probes the provider at
        startup, which can block if the provider connection is already in use.
      </Callout>
    </>
  );
}

function LibraryUsage() {
  return (
    <>
      <H2>Library Usage</H2>
      <P>
        Use NMAFC as a Python library in your own application. Two construction
        paths: async (NeuromorphicMemory) and sync (SyncNeuromorphicMemory).
      </P>

      <H2>Async with Context Manager</H2>
      <Code lang="python">
{`import asyncio
from nmafc.wrapper import NeuromorphicMemory

async def main():
    async with await NeuromorphicMemory.from_config() as mem:
        # Full pipeline: retrieve + LLM + extract + decay + prune
        response = await mem.process_turn("I'm allergic to peanuts")

        # Manual injection (no LLM call)
        from nmafc.schemas.memory import MemoryStateUpdate
        await mem.ingest_updates([
            MemoryStateUpdate(
                entity_name="user_name",
                fact_content="User is Alice",
                memory_type="CoreAnchor",
            )
        ])

        # Introspect
        print(mem.get_hot_stats())
        print(mem.get_event_stats())

asyncio.run(main())`}
      </Code>

      <H2>Sync (No Async)</H2>
      <Code lang="python">
{`from nmafc.wrapper import SyncNeuromorphicMemory

with SyncNeuromorphicMemory.from_config() as mem:
    response = mem.process_turn_sync("Hello world")
    print(response)
    print(mem.get_hot_stats())`}
      </Code>

      <H2>Manual Provider Construction</H2>
      <Code lang="python">
{`from nmafc.wrapper import NeuromorphicMemory
from nmafc.integration.factory import create_llm_provider, create_embedding_provider

llm = create_llm_provider("groq/llama-3.1-70b-versatile")
embedder = create_embedding_provider("ollama/nomic-embed-text")

mem = NeuromorphicMemory(llm_provider=llm, embedding_provider=embedder)`}
      </Code>

      <H2>Custom Config</H2>
      <Code lang="python">
{`from nmafc.storage.config import NMafcConfig, StorageConfig

config = NMafcConfig(
    storage=StorageConfig(
        hot_uri="./my-data/vectors",
        cold_uri="./my-data/events.db",
        embedding_dim=768,
    ),
    llm_provider_model="groq/llama-3.1-70b-versatile",
    embedding_provider_model="ollama/nomic-embed-text",
)

mem = NeuromorphicMemory.from_config(config=config)`}
      </Code>

      <H2>Custom Providers</H2>
      <P>
        Implement <code>LLMProvider</code> and{" "}
        <code>EmbeddingProvider</code> to plug in any backend.
      </P>
      <Code lang="python">
{`from nmafc.integration.base import LLMProvider, EmbeddingProvider
from nmafc.schemas.memory import MemoryStateUpdate

class MyLLM(LLMProvider):
    async def chat_with_extraction(self, messages, system_prompt):
        # Call your LLM here
        response = "..."
        return response, []  # (text, list of MemoryStateUpdate)

class MyEmbedder(EmbeddingProvider):
    async def embed(self, texts):
        # Return list of vectors
        return [[0.1] * 768 for _ in texts]

mem = NeuromorphicMemory(llm_provider=MyLLM(), embedding_provider=MyEmbedder())`}
      </Code>

      <H2>Available Methods</H2>
      <Table
        headers={["Method", "Returns", "Description"]}
        rows={[
          ["process_turn(msg)", "str", "Full pipeline: retrieve + respond + extract + decay"],
          ["ingest_updates(updates)", "None", "Inject facts without LLM call"],
          ["consolidate()", "int", "Manual REM sleep pass"],
          ["rollback(to_turn)", "int", "Rebuild state from Cold ROM"],
          ["get_hot_stats()", "dict", "Record count, avg weight, type breakdown"],
          ["get_cold_stats()", "dict", "Archive event counts"],
          ["get_event_stats()", "dict", "Cognitive event counts by type"],
          ["get_events(**kwargs)", "list", "Query events with filters"],
          ["get_event_timeline(limit)", "list", "Aggregated counts per turn"],
          ["get_entity_events(name)", "list", "Events for one entity"],
          ["current_turn", "int", "Current turn counter (property)"],
          ["close()", "None", "Close storage connections"],
        ]}
      />
    </>
  );
}

function Architecture() {
  return (
    <>
      <H2>Architecture</H2>

      <H2>System Overview</H2>
      <Code lang="text">
{`┌─────────────────────────────────────────────────────────┐
│                  Web UI (Next.js :3000)                  │
│  Dashboard │ Memory Explorer │ Graph │ Decay │ Events   │
└────────────────────────┬────────────────────────────────┘
                         │ /api/* proxy
┌────────────────────────▼────────────────────────────────┐
│              FastAPI Backend (:8000)                      │
│  REST API (27 endpoints) + WebSocket (/ws/live)          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│             NeuromorphicMemory (wrapper.py)              │
│                                                          │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────────┐ │
│  │  Query   │  │  Extractor │  │     Engine            │ │
│  │  Router  │  │  (LLM +    │  │  Decay │ Reinforce    │ │
│  │  Vector  │  │   Tool)    │  │  Prune │ Consolidate  │ │
│  │  + Graph │  │            │  │  Rollback             │ │
│  │  + Cold  │  │            │  │                       │ │
│  │  Fallback│  │            │  │  Event Log (SQLite)   │ │
│  └────┬─────┘  └────────────┘  └──────────────────────┘ │
└───────┼─────────────────────────────────────────────────┘
        │
┌───────▼─────────────┐  ┌──────────────┐  ┌────────────┐
│   Hot RAM (LanceDB) │  │ Cold ROM     │  │ Embedder   │
│   Vectors + Weights │  │ (SQLite/PG)  │  │            │
│   Mutable           │  │ Append-only  │  │            │
└─────────────────────┘  └──────────────┘  └────────────┘`}
      </Code>

      <H2>Processing Pipeline</H2>
      <P>Each call to process_turn() executes this sequence:</P>
      <Code lang="text">
{`1. Increment turn counter
2. Retrieve context (vector search + spreading activation + cold fallback)
3. Format retrieved memories as context string
4. LLM call with tool-use → simultaneous response + state extraction
5. For each extracted update:
   a. Log to Cold ROM (append-only)
   b. Detect overrides → suppress contradicted records (w *= gamma)
   c. Embed new fact → upsert to Hot RAM (weight=1.0)
6. Decay all mutable records: w(t) = w(t0) * e^(-lambda * dt)
7. Prune: delete records where weight <= w_prune
8. Every N turns: REM consolidation (elevation + dead pointer cleanup)`}
      </Code>

      <H2>Event Logging</H2>
      <P>
        Every cognitive event is logged to the Event Log (SQLite). Events
        include weight updates, overrides, suppressions, prunes, consolidations,
        LTP reinforcements, and retrievals.
      </P>
      <Table
        headers={["Event Type", "When Emitted"]}
        rows={[
          ["weight_update", "Every decay pass where weight changes"],
          ["override", "When a contradicting fact is detected"],
          ["suppression", "When old record weight is multiplied by gamma"],
          ["prune", "When a record is evicted (weight <= w_prune)"],
          ["consolidation", "When ActiveContext is promoted to CoreAnchor"],
          ["ltp", "When a retrieved record is reinforced"],
          ["retrieval", "When context is retrieved for a query"],
        ]}
      />
    </>
  );
}

function Providers() {
  return (
    <>
      <H2>Supported Providers</H2>

      <H2>LLM Providers</H2>
      <Table
        headers={["Provider", "Format", "API Key Env"]}
        rows={[
          ["OpenAI", "openai/gpt-4o", "OPENAI_API_KEY"],
          ["Anthropic", "anthropic/claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"],
          ["AWS Bedrock", "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", "ANTHROPIC_API_KEY_BEDROCK"],
          ["Azure OpenAI", "azure/DeepSeek-V4-Pro", "AZURE_OPENAI_API_KEY"],
          ["Groq", "groq/llama-3.1-70b-versatile", "GROQ_API_KEY"],
          ["OpenRouter", "openrouter/anthropic/claude-sonnet-4-20250514", "OPENROUTER_API_KEY"],
          ["Together", "together/meta-llama/Llama-3-70b-chat-hf", "TOGETHER_API_KEY"],
          ["Ollama", "ollama/llama3.2", "None (local)"],
          ["LM Studio", "lmstudio/local-model", "None (local)"],
          ["vLLM", "vllm/meta-llama/Llama-3-8b", "None (local)"],
        ]}
      />

      <H2>Embedding Providers</H2>
      <Table
        headers={["Provider", "Format", "Dimensions"]}
        rows={[
          ["OpenAI", "openai/text-embedding-3-small", "1536"],
          ["OpenAI", "openai/text-embedding-3-large", "3072"],
          ["Azure OpenAI", "azure/text-embedding-3-small", "1536"],
          ["Bedrock Titan", "bedrock/amazon.titan-embed-text-v2:0", "1024"],
          ["Ollama", "ollama/nomic-embed-text", "768"],
          ["Ollama", "ollama/mxbai-embed-large", "1024"],
          ["Together", "together/togethercomputer/m2-bert-80M-8k-retrieval", "768"],
          ["FastEmbed", "fastembed/BAAI/bge-small-en-v1.5", "384"],
        ]}
      />

      <Callout type="info">
        Embedding dimension is auto-detected on initialization. Set
        NMAFC_EMBEDDING_DIM explicitly for non-1536-dim models to skip the
        startup probe.
      </Callout>
    </>
  );
}

function DecaySystem() {
  return (
    <>
      <H2>Decay System</H2>
      <P>
        The decay engine implements the Ebbinghaus forgetting curve with three
        memory tiers and spaced repetition reinforcement.
      </P>

      <H2>Decay Formula</H2>
      <Code lang="text">
{`w_i(t) = w_i(t_0) * exp(-lambda_eff * delta_t)

where:
  delta_t = current_turn - last_reinforced_turn
  lambda_eff = lambda_base(tau_i) * alpha(k_i)
  alpha(k) = exp(-eta * k)
  eta = 0.15`}
      </Code>

      <H2>Memory Tiers</H2>
      <Table
        headers={["Tier", "Lambda", "Half-life", "Examples"]}
        rows={[
          ["CoreAnchor", "0.0", "Infinite", "Name, allergies, identity"],
          ["ActiveContext", "0.05", "~14 turns", "Current goals, projects"],
          ["EphemeralState", "0.69", "~1 turn", "Mood, passing comments"],
        ]}
      />

      <H2>Spaced Repetition (LTP)</H2>
      <P>
        When a memory is retrieved, its weight resets to 1.0 and its
        consolidation index k increments. Higher k means slower future decay.
      </P>
      <Table
        headers={["Tier", "k=0 (10 turns)", "k=5 (10 turns)", "k=10 (10 turns)"]}
        rows={[
          ["CoreAnchor", "1.000", "1.000", "1.000"],
          ["ActiveContext", "0.607", "0.858", "0.951"],
          ["EphemeralState", "0.001", "0.056", "0.314"],
        ]}
      />

      <H2>Override & Pruning</H2>
      <P>
        When a contradicting fact is detected, the old record{"'"}s weight is
        multiplied by gamma (0.1). Records below w_prune (0.1) are evicted.
      </P>
      <Code lang="text">
{`Override:  w_old = w_old * gamma    (gamma = 0.1)
Pruning:   if w_i <= w_prune (0.1): delete
Consolidation: if k >= 10 AND ActiveContext → promote to CoreAnchor`}
      </Code>

      <H2>Clustering Protection (beta)</H2>
      <P>
        When beta {" > "}0, the decay rate is scaled by the entity{"'"}s clustering
        coefficient: lambda_eff *= (1 - beta * C). Densely interconnected
        entities decay slower. Inspired by hippocampal synapse clustering.
      </P>
    </>
  );
}
