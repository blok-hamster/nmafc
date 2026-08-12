# NMAFC Benchmark Suite

Academic-grade evaluation of the Neuromorphic Memory Architecture for Conversational AI (NMAFC). Uses real datasets from published research papers and proper evaluation metrics (F1, LLM-as-judge) to honestly compare three memory approaches.

## Datasets

### LoCoMo (Maharana et al., 2024)
- **Source:** `KimmoZZZ/locomo` on HuggingFace
- **Size:** 10 conversations, ~1986 QA pairs, ~5882 turns
- **Categories:** Single-hop, Temporal, Multi-hop, Open-domain, Adversarial
- **Metrics:** Token-level F1 (normalized) + LLM-as-judge accuracy

### LongMemEval (Wu et al., 2024)
- **Source:** `xiaowu0162/longmemeval-cleaned` on HuggingFace
- **Size:** 500 questions across 6 types
- **Types:** Temporal-reasoning, Multi-session, Knowledge-update, Single-session-user, Single-session-assistant, Single-session-preference
- **Variants:** `oracle` (shortest context), `s` (medium), `m` (longest)
- **Metrics:** LLM-as-judge binary accuracy + F1

## Arms (Conditions)

| Arm | Name | Description |
|-----|------|-------------|
| 1 | `raw` | **Raw LLM** — Full conversation history stuffed into context window. No memory system. Equivalent to "Base"/"Long-context" baselines in the literature. |
| 2 | `stateful` | **Stateful No-Decay** — NMAFC with all decay/pruning disabled (lambda=0, gamma=1.0, w_prune=0). Simulates MemGPT/Zep "keep everything" philosophy. Isolates the exact contribution of cognitive decay. |
| 3 | `neuromorphic` | **Full Neuromorphic** — Production NMAFC with Ebbinghaus decay, spaced repetition (LTP), override suppression, and active pruning. The proposed system. |

## Setup

### 1. Install dependencies

```bash
uv pip install -e ".[bench,llm,aws]"
```

### 2. Configure credentials

Copy the example env file and fill in credentials for the providers you want to use:

```bash
cp .env.example .env
```

At minimum you need:
- **One LLM provider** (for answering questions and memory extraction)
- **One embedding provider** (for vector storage and retrieval)

### 3. Start local services (if using Ollama)

```bash
ollama serve
ollama pull nomic-embed-text   # 768-dim embedding model
```

## Usage

### LoCoMo Benchmark

```bash
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_locomo [OPTIONS]
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `$NMAFC_BENCH_PROVIDER` or `ollama/llama3.2` | LLM provider string for question answering and memory extraction. Format: `provider/model-name`. |
| `--embedding` | `$NMAFC_BENCH_EMBEDDING` or `ollama/nomic-embed-text` | Embedding provider string for vector storage. Format: `provider/model-name`. |
| `--judge` | Same as `--provider` | LLM provider string for the LLM-as-judge evaluator. Use a stronger model here for reliable scoring. |
| `--arms` | `raw,stateful,neuromorphic` | Comma-separated list of arms to evaluate. Options: `raw`, `stateful`, `neuromorphic`. |
| `--conversations` | All (10) | Number of conversations to evaluate. Use `1` or `2` for quick testing. |
| `--categories` | All (1-5) | Comma-separated QA category numbers to include. `1`=single-hop, `2`=temporal, `3`=multi-hop, `4`=open-domain, `5`=adversarial. |
| `--skip-judge` | Off | Skip LLM-as-judge evaluation (F1 only). Much faster, useful for development. |
| `--checkpoint` | None | Path to a checkpoint file to resume from. |
| `--output` | `scripts/benchmarks/results/locomo/` | Output directory for results JSON and checkpoints. |

#### Examples

```bash
# Quick smoke test — 1 conversation, single-hop only, no judge
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_locomo \
  --provider "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0" \
  --embedding "ollama/nomic-embed-text" \
  --conversations 1 \
  --categories "1" \
  --skip-judge

# Full run with Azure DeepSeek as LLM, Bedrock Haiku as judge
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_locomo \
  --provider "azure/DeepSeek-V4-Pro" \
  --embedding "ollama/nomic-embed-text" \
  --judge "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Only test neuromorphic arm on temporal + multi-hop questions
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_locomo \
  --arms "neuromorphic" \
  --categories "2,3" \
  --conversations 3

# Use OpenAI for everything
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_locomo \
  --provider "openai/gpt-4o-mini" \
  --embedding "openai/text-embedding-3-small" \
  --judge "openai/gpt-4o"
```

### LongMemEval Benchmark

```bash
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_longmemeval [OPTIONS]
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `$NMAFC_BENCH_PROVIDER` or `ollama/llama3.2` | LLM provider string for question answering and memory extraction. |
| `--embedding` | `$NMAFC_BENCH_EMBEDDING` or `ollama/nomic-embed-text` | Embedding provider string for vector storage. |
| `--judge` | Same as `--provider` | LLM provider for LLM-as-judge scoring. |
| `--arms` | `raw,stateful,neuromorphic` | Comma-separated list of arms to evaluate. |
| `--variant` | `oracle` | Dataset variant. `oracle` = shortest haystack (fastest), `s` = medium, `m` = longest context (hardest). |
| `--limit` | All (500) | Maximum number of questions to evaluate. Use `10` or `25` for quick testing. |
| `--types` | All | Comma-separated question types to include. Options: `temporal-reasoning`, `multi-session`, `knowledge-update`, `single-session-user`, `single-session-assistant`, `single-session-preference`. |
| `--concurrency` | `3` | Maximum concurrent questions per arm. Higher = faster but more API load. |
| `--output` | `scripts/benchmarks/results/longmemeval/` | Output directory for results JSON and checkpoints. |

#### Examples

```bash
# Quick test — 10 questions, oracle variant
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_longmemeval \
  --provider "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0" \
  --embedding "ollama/nomic-embed-text" \
  --variant oracle \
  --limit 10

# Full oracle run with all arms
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_longmemeval \
  --provider "azure/DeepSeek-V4-Pro" \
  --embedding "ollama/nomic-embed-text" \
  --variant oracle

# Only temporal-reasoning and multi-session (the hardest types)
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_longmemeval \
  --arms "neuromorphic,stateful" \
  --types "temporal-reasoning,multi-session" \
  --limit 50

# Medium variant (longer haystack, more challenging)
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_longmemeval \
  --variant m \
  --limit 100
```

### Visualization

Generate publication-ready charts from benchmark results:

```bash
python -m scripts.benchmarks.visualize --input results/locomo/results.json
python -m scripts.benchmarks.visualize --input results/longmemeval/results.json --format svg
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | Required | Path to the `results.json` file from a benchmark run. |
| `--format` | `html` | Output format for individual charts. Options: `html`, `svg`, `png`, `pdf`. |
| `--output` | Same dir as input | Output directory for generated charts. |

#### Generated Charts

1. **Accuracy by Question Type** — Grouped bar chart across all arms
2. **Token Cost** — Stacked bar (prompt vs completion tokens per arm)
3. **Context Injection Size** — Average tokens injected per question (proves compression claim)
4. **Latency** — Avg/P50/P95 response latency comparison
5. **Dashboard** — Combined HTML dashboard with all charts
6. **SUMMARY.md** — Markdown table with all metrics

## Provider Strings

The `--provider` and `--embedding` flags use a `provider/model` format:

| Provider | LLM Example | Embedding Example |
|----------|-------------|-------------------|
| AWS Bedrock | `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0` | `bedrock/amazon.titan-embed-text-v2:0` |
| Azure OpenAI | `azure/DeepSeek-V4-Pro` | `azure/text-embedding-3-small` |
| OpenAI | `openai/gpt-4o-mini` | `openai/text-embedding-3-small` |
| Anthropic | `anthropic/claude-haiku-4-5-20251001` | — |
| Groq | `groq/llama-3.1-70b-versatile` | — |
| OpenRouter | `openrouter/anthropic/claude-sonnet-4-20250514` | — |
| Together | `together/meta-llama/Llama-3-70b-chat-hf` | `together/togethercomputer/m2-bert-80M-8k-retrieval` |
| Ollama | `ollama/llama3.2` | `ollama/nomic-embed-text` |
| LM Studio | `lmstudio/local-model` | `lmstudio/local-embed-model` |
| vLLM | `vllm/meta-llama/Llama-3-8b` | `vllm/embed-model` |

## Output Format

Results are saved as JSON with this structure:

```json
{
  "metadata": {
    "dataset": "locomo",
    "provider": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "embedding": "ollama/nomic-embed-text",
    "timestamp": "2026-08-12T07:14:00",
    "conversations_evaluated": 10
  },
  "results": {
    "raw_llm": {
      "accuracy": { "overall": 0.72, "overall_f1": 0.45, "by_category": {...} },
      "operational": {
        "latency_ms": { "avg": 3200, "p50": 2800, "p95": 6100 },
        "tokens": { "total": 850000, "avg_context_per_question": 12000 },
        "storage": { "hot_records": 0, "cold_events": 0 }
      }
    },
    "stateful_nodecay": {...},
    "neuromorphic": {...}
  }
}
```

## Checkpointing

Both runners save checkpoints incrementally:
- **LoCoMo:** After each conversation (`checkpoint_<arm>.json`)
- **LongMemEval:** Every 25 questions (`checkpoint_<arm>.json`)

If a run is interrupted, the checkpoint files in the output directory contain all completed results. Currently manual resume is required (copy results from checkpoint).

## Evaluation Metrics

### F1 Score (LoCoMo protocol)
Token-level F1 with answer normalization — lowercased, articles/punctuation removed, whitespace collapsed. Measures partial match quality. A verbose but correct answer scores lower than an exact match.

### LLM-as-Judge (Zep/LongMemEval protocol)
Binary accuracy using an LLM to evaluate correctness. The judge considers semantic equivalence, not just string match. A verbose answer containing the correct information scores as "correct." Has high correlation with human evaluators (Rasmussen et al., 2025).

## Time Estimates

Per conversation (419 turns, ~32 single-hop questions):

| Arm | Ingestion | Questions | Total |
|-----|-----------|-----------|-------|
| `raw` | ~0s | ~2 min | ~2 min |
| `stateful` | ~20-40 min | ~2 min | ~25 min |
| `neuromorphic` | ~20-40 min | ~2 min | ~25 min |

Ingestion time scales linearly with turn count and depends on LLM speed. Bedrock Haiku and Azure DeepSeek-V4-Pro are comparable (~1-2s per extraction call).

## Troubleshooting

### No output / logs not streaming
Always prefix with `PYTHONUNBUFFERED=1`:
```bash
PYTHONUNBUFFERED=1 python -m scripts.benchmarks.run_locomo ...
```

### HuggingFace rate limit
Set a token to avoid throttling:
```bash
export HF_TOKEN=hf_your_token_here
```

### Ollama not connected
```bash
ollama serve          # Start the server
ollama list           # Verify models are available
ollama pull nomic-embed-text  # Pull if missing
```

### Low F1 but correct answers
F1 penalizes verbosity. If the gold answer is "Paris" and the model says "The answer is Paris, the capital of France", the F1 will be low despite being correct. Use `--skip-judge` only for development; always use LLM-as-judge for final results.

### Context tokens = 0
If stateful/neuromorphic arms show `ctx_tokens=0`, the retrieval pipeline isn't finding relevant memories. Ensure:
1. Enough turns are ingested (needs 50+ turns to build meaningful memory)
2. Ollama embedding model is running
3. The question relates to content that was actually ingested
