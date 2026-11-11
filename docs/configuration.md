# Configuration

NMAFC is configured through one dataclass (`NMafcConfig`), environment
variables, or a TOML file — in that order of precedence:

**explicit argument > environment variable > TOML file > library default**

All of it is plain Pydantic: invalid values fail fast with a message naming
the field.

## The config object

```python
from nmafc import NMafcConfig

config = NMafcConfig(
    llm_provider_model="openai/gpt-4o-mini",      # "provider/model" string
    embedding_provider_model="openai/text-embedding-3-small",
    agent_id="default",        # tenant namespace
    conversation_id="default", # session namespace
    # storage=StorageConfig(...),   # see storage.md
    # decay=DecayConfig(...),       # engine hyperparameters (leave alone)
)
```

`llm_provider_model` is `"provider/model"`: the provider prefix selects the
route (see [providers](providers.md) for the full matrix), the rest is passed
to that provider as the model identifier. Nested names are fine —
`"openrouter/anthropic/claude-sonnet-4-20250514"`.

## Environment variables

### Model selection

| Variable | Default | Meaning |
|---|---|---|
| `NMAFC_LLM_PROVIDER_MODEL` | `openai/gpt-4o-mini` | Provider/model string for extraction & chat |
| `NMAFC_EMBEDDING_PROVIDER_MODEL` | `openai/text-embedding-3-small` | Provider/model string for embeddings |
| `NMAFC_EMBEDDING_DIM` | `1536` | Vector width — must match the embedding model |
| `NMAFC_LLM_TEMPERATURE` | provider default | Sampling temperature |
| `NMAFC_LLM_MAX_TOKENS` | provider default | Max output tokens |
| `NMAFC_EXTRACTOR_VARIANT` | library default | Extraction prompt variant |

### Provider credentials

| Provider prefix | Env var | Also uses |
|---|---|---|
| `openai/` | `OPENAI_API_KEY` | — |
| `anthropic/` | `ANTHROPIC_API_KEY` | — |
| `groq/` | `GROQ_API_KEY` | — |
| `openrouter/` | `OPENROUTER_API_KEY` | — |
| `together/` | `TOGETHER_API_KEY` | — |
| `azure/` | `AZURE_OPENAI_API_KEY` | `AZURE_OPENAI_ENDPOINT` (required) |
| `bedrock/` | AWS credential chain | `AWS_REGION` |
| `ollama/` | none | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) |
| `lmstudio/` | none | `LMSTUDIO_BASE_URL` (default `http://localhost:1234/v1`) |
| `vllm/` | none | `VLLM_BASE_URL` (default `http://localhost:8000/v1`) |

A missing key is not a crash: you get `MissingAPIKeyError` whose message names
the env var above. An unconfigured Azure route gets
`ProviderNotConfiguredError` naming `AZURE_OPENAI_ENDPOINT`. Details in
[errors](errors.md).

### Tenancy

| Variable | Default | Meaning |
|---|---|---|
| `NMAFC_AGENT_ID` | `default` | Tenant/agent namespace for all records |
| `NMAFC_CONVERSATION_ID` | `default` | Conversation session namespace |

Both can also be passed per call (`mem.recall(..., agent_id=..., conversation_id=...)`),
which wins over the environment.

### Storage location

| Variable | Default | Meaning |
|---|---|---|
| `NMAFC_DATA_DIR` | platform data dir | Root for all three stores (see below) |
| `NMAFC_HOT_URI` | `$NMAFC_DATA_DIR/lancedb` | Hot-tier URI; `s3://…` goes remote |
| `NMAFC_COLD_URI` | `$NMAFC_DATA_DIR/cold.db` | Cold-tier URI; `postgresql://…` goes remote |
| `NMAFC_EVENT_LOG_URI` | `$NMAFC_DATA_DIR/events.db` | Cognitive event log (SQLite) |

`NMAFC_DATA_DIR` defaults to the per-user platform directory — macOS
`~/Library/Application Support/nmafc`, Linux `~/.local/share/nmafc`, Windows
`%APPDATA%\nmafc`. Set it to keep everything under one folder (e.g. a
volume mount in Docker). Full detail in [storage](storage.md).

## TOML file

Optional. `from_config()` and `from_env(config_path=...)` load a TOML file;
without one, the library default `configs/default.toml` is attempted and
skipped silently when absent (so a wheel install needs no repo files).

```toml
[llm]
provider_model = "groq/llama-3.1-70b-versatile"

[embedding]
provider_model = "openai/text-embedding-3-small"
```

Pass an absolute path (or `config_path=None` for env-only) — the library
default is relative to the current directory and only makes sense inside a
checkout.

## Engine hyperparameters

`DecayConfig` holds the decay/consolidation constants (`lambda_*`, `eta`,
`gamma`, `w_prune`, …). They are documented in the
[Mathematical Specification](../README.md#mathematical-specification) and
**frozen** — the values shipped are the ones the benchmark ran with, and
changing them changes retrieval behaviour. Configure *where data lives* and
*which models run*, not the cognitive constants, unless you are doing research.

## What NOT to configure

- Do not point two NMAFC processes at one hot store with different
  `embedding_dim` values; the vector column width is fixed at connect time.
- Do not set `NMAFC_HOT_URI` and `NMAFC_COLD_URI` to mismatched backends for
  the same data (an S3 hot tier with a local cold archive works, but you
  forfeit single-directory portability) — see [storage](storage.md).
