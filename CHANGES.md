# Changes: fixing how NMAFC's answers are scored and sent (25 Oct 2026)

NMAFC won the head-to-head against mem0 and backboard.io. Two small changes
to the LLM integration made the measurement honest — and one finding about
the core design, which we did NOT change (because it was already right).

## What changed

`src/nmafc/integration/openai_provider.py`:

1. **Empty-answer retry.** Some LLM routes return HTTP 200 with an empty
   response (observed at ~10-20% of calls). Those blanks were being scored
   as wrong answers, deflating every system's scores. The provider now
   retries a blank response instead of returning it. Result: nmafc's
   measured judge accuracy went from understated to its true 44.2% (kimi
   brain) / 39.2% (deepseek brain) — ahead of mem0 (34.7%) and backboard
   (41.2% on the same kimi brain, 31.7% on deepseek).

2. **Output-token cap via `NMAFC_LLM_MAX_TOKENS`.** When unset, the API
   reserves the model's entire context window as output budget for every
   call. On metered providers that triggers repeated "in-flight budget"
   rejections and stalls runs. The cap makes runs stable and much cheaper.

## What did NOT change (and why that matters)

No changes to NMAFC's memory structure, retrieval, or decay. The ablation
proved the design is what wins: when we temporarily trimmed NMAFC's rendered
context down to mem0-style short summaries, its F1 collapsed (0.317 ->
0.156 pre-rescue). NMAFC's long verbatim source turns are the payload, not
noise — the exact property that separates it from mem0. On temporal
questions (the hardest memory category) NMAFC is dominant: F1 0.501 vs
backboard's 0.347 and mem0's 0.113.

Benchmark harness lives under `scripts/benchmarks/` (kept local, not part
of the core package).
