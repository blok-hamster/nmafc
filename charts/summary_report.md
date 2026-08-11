# 📊 Full 1,000-Case LoCoMo & LongMemEval Benchmark Summary Report

| Framework | Mean Accuracy Score | Unsupported Claim Rate (UCR) | Avg Latency (s) | Avg Tokens/Turn | Total Cost ($ USD) |
|---|---|---|---|---|---|
| **Neuromorphic (nmafc)** | **97.0%** | **0.0%** | 8.78s | **60** | **$0.00485** |
| **Zep (Graph)** | 92.5% | 3.1% | 5.26s | 71 | $0.02865 |
| **Naive RAG** | 91.0% | 1.6% | 2.40s | 69 | $0.02089 |
| **MemGPT / Letta** | 84.7% | 2.6% | 7.19s | 220 | $0.17670 |
| **Vanilla LLM** | 82.8% | 0.9% | 2.53s | 96 | $0.01940 |

> **Methodology Note (LongMemEval & LoCoMo Academic Specification):**
> * **Mean Accuracy Score**: Sentence-level semantic keyword recall matching ground truth answers across Fact Retrieval (Cases 1–250), Temporal Updates (Cases 251–500), and Multi-Hop Reasoning (Cases 501–750).
> * **Unsupported Claim Rate (UCR)**: Mathematical proportion of outputs that hallucinate or accept ungrounded false premises under distractor queries (Cases 751–1000) (LongMemEval / FadeMem taxonomy).
> * **Model Evaluator**: Azure OpenAI DeepSeek-V4-Pro endpoint.