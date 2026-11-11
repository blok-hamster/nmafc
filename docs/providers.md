# Providers

NMAFC needs two model routes: an **LLM** (extraction + replies) and an
**embedding model** (vector similarity). Both are selected with a
`"provider/model"` string — the prefix picks the route:

```python
config = NMafcConfig(llm_provider_model="groq/llama-3.1-70b-versatile",
                     embedding_provider_model="fastembed/BAAI/bge-small-en-v1.5")
```

## LLM routes

All routes except `anthropic/` and `bedrock/` are OpenAI-compatible; the SDK
required is shown so you know which extra to install.

| Prefix | Example | Key env var | SDK / extra |
|---|---|---|---|
| `openai/` | `openai/gpt-4o-mini` | `OPENAI_API_KEY` | `nmafc[llm]` |
| `anthropic/` | `anthropic/claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` | `nmafc[anthropic]` |
| `azure/` | `azure/my-gpt4-deployment` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | `nmafc[llm]` |
| `bedrock/` | `bedrock/anthropic.claude-3-sonnet-20240229-v1:0` | AWS credential chain | `nmafc[aws]` |
| `groq/` | `groq/llama-3.1-70b-versatile` | `GROQ_API_KEY` | `nmafc[llm]` |
| `openrouter/` | `openrouter/anthropic/claude-sonnet-4-20250514` | `OPENROUTER_API_KEY` | `nmafc[llm]` |
| `together/` | `together/mixtral-8x7b` | `TOGETHER_API_KEY` | `nmafc[llm]` |
| `azure_v1/` | `azure_v1/my-deployment` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | `nmafc[llm]` |
| `ollama/` | `ollama/llama3` | none (local) | `nmafc[llm]` |
| `lmstudio/` | `lmstudio/local-model` | none (local) | `nmafc[llm]` |
| `vllm/` | `vllm/my-model` | none (local) | `nmafc[llm]` |

Notes:

- `azure/` uses deployment-based routing (`AsyncAzureOpenAI`); `azure_v1/`
  targets the raw OpenAI-v1 endpoint shape at your
  `AZURE_OPENAI_ENDPOINT`. Either way the endpoint env var is required and a
  missing one raises `ProviderNotConfiguredError` naming it.
- `bedrock/` requires the AWS credential chain to work (env, profile, or
  instance role) and `AWS_REGION`; it needs `nmafc[aws]`.
- Local routes (`ollama/`, `lmstudio/`, `vllm/`) need no key and no cloud —
  the base URL env vars default to the standard local ports and can be
  overridden.
- A bare `"model-name"` with no prefix is treated as `openai/<model>`.

## Embedding routes

| Prefix | Example | Key env var | SDK / extra |
|---|---|---|---|
| `openai/` | `openai/text-embedding-3-small` | `OPENAI_API_KEY` | `nmafc[llm]` |
| `azure/` | `azure/my-embedding-deployment` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | `nmafc[llm]` |
| `bedrock/` | `bedrock/amazon.titan-embed-text-v2:0` | AWS credential chain | `nmafc[aws]` |
| `ollama/` | `ollama/nomic-embed-text` | none (local) | `nmafc[llm]` |
| `fastembed/` | `fastembed/BAAI/bge-small-en-v1.5` | none (local, ONNX on your CPU) | `nmafc[fastembed]` |

`fastembed/` is the only fully key-free, offline-capable route — useful for
development, air-gapped deployments, and not paying per token. When you change
the embedding model, set `NMAFC_EMBEDDING_DIM` to its vector width (e.g. 384
for `bge-small-en-v1.5`, 768 for `nomic-embed-text`); the hot store's vector
column is created at that width and must match.

## Error behaviour (what a bad setup looks like)

| Situation | Exception raised | Message contains |
|---|---|---|
| No key for the chosen provider | `MissingAPIKeyError` | the env var to set (`GROQ_API_KEY`, …) |
| Azure route, no endpoint | `ProviderNotConfiguredError` | `AZURE_OPENAI_ENDPOINT` |
| SDK not installed | `MissingExtraError` | the exact `pip install nmafc[extra]` command |
| Unknown provider prefix | `ProviderNotConfiguredError` | the valid prefixes |
| Storage won't connect | `StorageError` | the credential chain (S3) or path (local) to check |

Full catalogue with copy-paste fixes: [errors.md](errors.md).

## Custom providers

Implement two methods on the `LLMProvider`/`EmbeddingProvider` ABCs and pass
the objects straight to the constructor (`from_providers` on the sync class) —
see `examples/custom_provider.py`. Registration-by-prefix requires editing the
factory dispatch; constructor injection does not.
