# 📊 Academic LoCoMo & LongMemEval Benchmark Summary Report

| Framework | Mean Accuracy Score | Unsupported Claim Rate (UCR) | Avg Latency (s) | Avg Tokens/Turn | Total Cost ($ USD) |
|---|---|---|---|---|---|
| **Cognee (Official SDK)** | 71.9% | 6.5% | 4.57s | 75 | $0.27101 |
| **Letta (Official SDK)** | 87.3% | 10.8% | 3.90s | 75 | $0.48180 |
| **Naive RAG** | 92.1% | 8.9% | 3.46s | 69 | $0.16684 |
| **Neuromorphic (nmafc)** | 96.6% | 3.4% | 3.46s | 61 | $0.03935 |
| **Vanilla LLM** | 82.9% | 16.6% | 3.47s | 96 | $0.15503 |
| **Zep (Official SDK)** | 88.2% | 7.0% | 3.91s | 75 | $0.24165 |

> **Methodology Note (LongMemEval Specification):**
> * **Mean Accuracy Score**: Sentence-level semantic keyword recall matching ground truth answers across Fact Retrieval, Temporal Updates, and Multi-Hop queries.
> * **Unsupported Claim Rate (UCR)**: Mathematical proportion of outputs that hallucinate or accept ungrounded false premises under distractor queries (LongMemEval / FadeMem taxonomy).