# NMAFC Research Report

Real-world applications, market positioning, distribution strategy, and the findings from `WHAT-CHANGED.md`.

Compiled: 22 Sep 2026. Research-only — no code was modified. Sources footnoted in §6.

---

## Contents

1. [What the system is and what it does well](#1-what-the-system-is-and-what-it-does-well)
2. [Findings from WHAT-CHANGED.md](#2-findings-from-whatchangedmd)
3. [Industry applications](#3-industry-applications)
4. [Competitive landscape](#4-competitive-landscape)
5. [Market evidence](#5-market-evidence)
6. [Distribution & easy-usage strategy](#6-distribution--easy-usage-strategy)
7. [Sources](#7-sources)

---

## 1. What the system is and what it does well

**NMAFC (Neuromorphic Memory Architecture for Conversational AI)** is a biologically-inspired stateful memory wrapper for LLM agents. It sits between a user and an LLM: every turn it extracts structured facts, stores them in dual-tier storage, retrieves relevant memories as prompt context, and reinforces/decays/prunes them biologically.

### Capability reference (used throughout)

| Key | Capability | What it does |
|---|---|---|
| **OD** | Override detection / temporal supersession | A new fact contradicts an old one → old fact invalidated (not overwritten), supersession recorded |
| **DK** | Ebbinghaus decay + spaced-repetition reinforcement | 3-tier classification (CoreAnchor permanent / ActiveContext decaying / EphemeralState fast-fading); unretrieved facts decay exponentially and prune; retrieved facts reinforced |
| **EM** | REM-style consolidation | Frequently-retrieved active facts promoted to permanent (CoreAnchor) |
| **MH** | Multi-hop entity-graph retrieval | Entity graph with spreading activation (BFS) across facts |
| **RB** | Rollback / point-in-time rebuild | Full decision-state reconstruction to any past turn from the append-only event log |
| **AU** | Cognitive audit event log | Every weight update, override, prune, consolidation logged and replayable |
| **HY** | Graded hydration | Bounded context budget with verbatim source-text restore at retrieval |
| **MT** | Multi-tenancy + isolation | Hard isolation at `agent_id` + `conversation_id` scope in the storage layer |

### Benchmark position (LoCoMo paired run, 10 Sep 2026)

- **71.4% vs 64.6%** RAG on the four scored categories (+6.8, McNemar exact **p=5.5e-08**, n=1,539)
- **Temporal +25.5** (71.7% vs 46.1%, p=1.1e-14), **multi-hop +12.5** (57.3% vs 44.8%, p=0.0075)
- **−34% context tokens** per question (964 vs 1,461)
- Adversarial: **−14.6** (reported separately, honest loss; concluded not fixable from the prompt)
- Latency indistinguishable through p90; framework's own work ≈56ms/question (83% of latency is provider waiting)

---

## 2. Findings from WHAT-CHANGED.md

### 2.1 The performance arc

| Date | Run | NMAFC | RAG | Context |
|---|---|---|---|---|
| 14 Aug | first full run | 54.4% | 60.3% | 440 tok |
| 19 Aug | `full_v2` | 59.6% | – | 1,304 tok |
| 21 Aug | `full_v3` (graph fix) | 63.3% | 64.2% | 656 tok |
| 5 Sept | + hydration + turn dates | 67.0% | 66.2% | 1,137 tok |
| **10 Sept** | **paired: answer-type gate + grounding** | **71.4%** | **64.6%** | **964 tok** |

+17.0 points on its own arm since 14 Aug; +6.8 over RAG at a third less context.

### 2.2 What paid off (ranked)

1. **Graph search across both tiers + link resolution — +9.0 points.** Only change that was better *and* cheaper at once (context fell 1,304→656 while accuracy rose). Fixed: graph traversal had walked only Hot RAM (chains through pruned facts stopped); link targets pointed at nothing (`engine/linking.py`, Jaccard threshold 0.5). Category effect: multi-hop 38.5→52.1%, temporal 51.4→61.4%.
2. **Hydration — +1.75, p=0.021.** Fuzzy-trace theory (gist vs verbatim): `turn_text` stores verbatim turns *outside* FTS with no embedding — reachable only via a fact that won a slot. Facts only 64.4% → +hydrate top-5 66.1%.
3. **Wider retrieval budget — +1.5.** `rerank_top_k` 20→40. Honest: "gained 75, lost to distraction 52."
4. **Real dates instead of turn numbers — +1.3**, placebo-controlled (scrambled dates −0.3%).
5. **Answer-type gate — regex, zero cost.** "In which *state*…" was answered with a town. Quiet on ~85% of questions; a *level shift, never an invention*.

### 2.3 What did NOT pay (negative results, kept on record)

| Experiment | Effect |
|---|---|
| Decay as a ranking signal (`weight_signal`) | **−5.3** reachability, negative in *every* category |
| Contradiction/supersession pruning | **−5.9** |
| Rewriting the answer prompt | −1.04, retried null |
| Context compaction | +0.5, inside noise |
| Width on open-domain | +1.6 p=0.111 — unshippable; also a benchmark-artifact risk |
| The list-shape gate | +1.6 p=0.458; module kept, wired into nothing |
| `commit_short`/`commit_exact` clauses | buys adversarial, pays 1:1 in temporal |

Mechanistic stories worth keeping:

- **`weight_signal` stays 0.0 forever.** 97.8% of live weights sit at exactly 1.0. Injected graded weights made retrieval monotonically worse at every setting (churn 43–47% confirmed the ranking genuinely moved — a real negative, not a wiring failure).
- **`detect_override` fired 0 times across 8,276 facts.** The extractor names entities per event (`jon_job_loss_jan_2023` vs `jon_new_shop_july_2023`), so exact-match override never triggers. LLM-adjudicated retry: invalidated stale answers 19.6% vs correct answers 17.9% — a coin flip. Lesson: *vector similarity + LLM cannot separate "overtaken by events" from "restated differently"; measure gold-invalidation rate next to stale-invalidation rate.*
- **Adversarial and temporal are welded together** — two prompt clauses both traded them 1:1; the join is not in the prompt.

### 2.4 Methodology rules (stated repeatedly)

1. Pair everything in one session; report gross +N/−M, never nets; McNemar exact.
2. Report adversarial separately; don't drop the category you lose.
3. **Nothing under ~800 questions survives re-measurement** (four documented mispricings: 6, 8, 108, 281-question reads all regressed).
4. Screen retrieval by *reachability* (~4 min) before full scoring (~87 min) — but reach screens don't price changes that alter generation behaviour.
5. Never write to a store in place; reading a store writes ~14 records unless `close_readonly()`.
6. Always read examples before believing a rate.

### 2.5 Correctness fixes (score-neutral, real bugs)

1. **The clock guard** — 3,477/3,807 records had `last_reinforced_turn` earlier than `created_at_turn`.
2. **Reading memory was writing to it** — one `retrieve()` changed 14 records on disk. `defer_reinforcement_writes` now defaults True.
3. **Cold ROM ignored `invalid_at`** — invalidated facts stayed servable from the archive (64/101 stale prompts, *worse than doing nothing*). Fixed: `invalidate_facts()` + `AND invalid_at IS NULL` filters. Pruning deliberately does **not** propagate (superseded is wrong/goes; pruned is weak/true/stays).

### 2.6 The core insight for what to build next

> **"It is not a retrieval problem."** On LoCoMo, gold facts are stored 82.2% and reached 80.2% — retrieval is 2 points from its ceiling. **90% of misses were never extracted at all** ("not amnesic, inattentive at encoding"). The largest live lever is **extraction recall at write time** (dropped qualifiers, quantities, destinations).

Also: 68.4% of surviving Hot facts are classified `CoreAnchor` (λ=0) and sampling says ~half are wrongly classified one-off episodes — a fix that will likely make scores *drop* honestly.

### 2.7 Roadmap items (ranked by expected value)

1. **List-shaped questions** — 50.2% where `wants_list` fires vs 66.0% elsewhere; width doesn't close it; untried: change *answering* behaviour (enumerate what it holds).
2. **Fix `CoreAnchor` over-classification** — batch with other write-time changes (needs full re-ingest, ~5.3h; expect LoCoMo to go down).
3. **LongMemEval** — belief-update design untested.
4. **Adversarial from somewhere other than the prompt.**
5. **NOT** more supersession-detector work; **NOT** re-ingestion for its own sake.
6. Standing: destinations-dropped extraction recall (unmeasured).

---

## 3. Industry applications

### 3.1 Tier 1 — strongest fits (unique capabilities match hard requirements)

#### Healthcare — highest-value compliance wedge

| Need | Capability | Product idea |
|---|---|---|
| Med/dosage/allergy changes must invalidate old state | OD | Medication-supersession: conflicting past med facts invalidated with reason + source; "current med list" derived from non-superseded facts only |
| HIPAA §164.312(b) immutable audit (~6-yr); EU AI Act Art. 12 lifetime logging | AU + RB | HIPAA-ready audit replay: "what did the agent know at T when it recommended the dose?" |
| Point-in-time decision reconstruction | RB | Decision-evidence capture: memory + source text + confidence + ancestor turns |
| Care-team continuity; no cross-patient leakage | MT | Patient chart in separate tenant namespace; storage-layer (not app-layer) isolation |
| Agent action auditing in hospital environments | AU | Append-only action log feeding existing SIEM |

**Product:** *Clinical Memory Backbone* — HIPAA-ready memory layer for care copilots: medication supersession with provenance, escalation-summary hydration, audit replay. Supporting pattern: DoctorAgent, OpenHealthAgents, Cascade Protocol already assemble this piecemeal — NMAFC is the missing single layer. **Gap to build:** GDPR/HIPAA erasure needs crypto-shred-with-audit-certificate (audit survives erasure).

#### Finance — most documented regulatory opening

| Need | Capability | Product idea |
|---|---|---|
| KYC/AML profiles evolve; periodic reviews misaligned with pace of change | OD | Continuous-KYC memory: PEP-status supersession, automated next-review dates |
| SEC 17a-4 / FINRA 4511 / MiFID II point-in-time records | RB + AU | Regulatory-exportable agent decision logs, hash-chained integrity |
| Backtest lookahead-bias prevention | RB | `backtest_check`: rebuild memory to simulation date; agent cannot cite post-simulation facts |
| Information barriers; related-party detection | MH + MT | "Is counterparty X connected to a restricted entity within N hops?" + desk-level silos |
| Know-Your-Agent (IMF Notes 2026/004; MAS MetaComp KYA) | AU | Agent-identity ledger: every decision with agent identity, intent, confidence, source chain |

**Product:** *Compliance-grade memory for trading/risk agents* — examiner-reconstructable decisions, backtest-proof rollback, graph-based information-barrier enforcement.

#### Legal

| Need | Capability | Product idea |
|---|---|---|
| Contract amendments invalidate old terms; reconstruct clause-as-of date | OD + RB | Clause-versioning: prior term superseded and queryable as-of; rollback to claim date |
| Conflict-of-interest reachability (ABA 1.7/1.9) | MH | "Is any attorney connected to adverse party X within N hops?" |
| Privilege cutoffs; matter walls | RB + MT | `recall_at(privilege_date)`; per-matter tenant silos |
| M&A diligence: every assertion traceable to source | HY | Verbatim-source hydration with citation + document checksum |
| Chain-of-custody for discovery | AU + RB | "What did we know on date X" for privilege logs |

**Product:** *Deal-room diligence copilot* — per-matter silos, clause supersession, reconstructable audit exports.

#### Cybersecurity

| Need | Capability | Product idea |
|---|---|---|
| Threat-actor TTPs change; old notes must be superseded not duplicated | OD | Actor-profile supersession: "APT28 dropped tool X" invalidates prior default with source+date |
| Correlate alerts across weeks into one campaign | MH | Campaign BFS linkage: credential-reuse → months-ago recon → shared infra |
| Organizational knowledge when analysts leave | EM | Consolidation promotes validated triage findings to tenure-independent memory |
| Reduce alert fatigue via entity baselines | DK | Baselines reinforce on confirmed-benign, decay on stale |
| FedRAMP AU / OCSF audit | AU | OCSF-format audit events for every memory mutation |

**Product:** *CTI memory plane* — alias resolution, TTP supersession, campaign linkage, OCSF audit. **Incumbent alert:** ZettelForge (OSS, feature-complete) is closest here; lacks decay curves and point-in-time rollback — those two are the wedge.

### 3.2 Tier 2 — strong fits (decay/consolidation are the differentiator)

#### Customer support

- **OD** → preference/entitlement supersession ("that plan no longer exists; that price was grandfather-only")
- **DK** → a published audit found ~97.8% of entries in a leading memory platform "garbage" after a month; retention policy + reinforcement + pruning fixes exactly this
- **HY + AU** → handoff brief: verbatim tickets + structured customer-state + traceable sources (ibl.ai claims 85% handle-time reduction with memory-layer context-at-handoff)
- **MH** → symptom→product→root-cause→resolution graph across tickets

**Product:** *360° customer-story memory* pluggable beneath existing helpdesks.

#### Education

- **DK** → per-concept mastery with Ebbinghaus decay; review scheduling at the forgetting threshold (TASA arXiv 2511.15163 decays mastery via forgetting curves; Studeia runs SM-2 spaced repetition)
- **OD** → misconception supersession (active → resolving → resolved, detect regression) preserving evidence trail
- **EM** → mastered knowledge promoted to permanent across semesters
- **RB + AU** → reconstruct learner's exact knowledge state on any date (progress reports)

**Product:** *Tutor memory plane* — decay curves *are* the pedagogy.

#### Gaming / NPCs

- **EM + CoreAnchor** → stable personality/relationship across 100 sessions and patches
- **DK** → relationship decay as a *gameplay feature* (friendliness decays after 144h offline; grudges fade, revive on reinforcement)
- **MH** → gossip propagation along relationship edges (`neighbor → guard` leak with probability)
- **RB** → point-in-time scene gating: "agents can only access events prior to the current timestamp" (DREAM arXiv 2608.05170) — prevents future-knowledge leakage

**Product:** *NPC memory SDK* — relationship decay, social-graph gossip, per-scene point-in-time retrieval.

### 3.3 Tier 3 — real fits

| Industry | Mapping | Product |
|---|---|---|
| **HR/recruiting** | DK → nurture timed to relationship decay (contact before it goes cold); MH → re-open silver-medialist pools on skill match; AU → auditable screening (EEO defense); OD → invalidated superseded JD criteria | Nurture-scheduler memory |
| **Autonomous/coding agents** | 65% of enterprise agent failures trace to context drift; HY's −34% tokens addresses budgets; RB → "why did the agent do that at step N?" replay; EM + MT → per-repo project memory | Session-continuity + replay memory |
| **Personal assistants** | OD → preference supersession/expiry; MT → work/personal/financial tenants; DK+EM → rules sticky only when used | *Mem0's home turf — the default, not the differentiator* |
| **E-commerce** | OD → "user stopped wanting X after returning it 3×"; HY → multi-session journey-state | Belief-supersession store |
| **Real estate** | DK+OD → criteria evolve over months without re-briefing; RB → deal event timeline | Client-relationship memory |
| **Automotive** | EM+MH → ownership-lifetime consolidation across OEM/dealer/insurer; OD → evolved preferences; edge/on-prem residency | Vehicle-relationship memory |
| **Government** | AU+RB → 3,611 documented federal AI use cases; OMB/NIST AI RMF; EU Annex III scope | Decision-trace + point-in-time state |
| **Research/life sciences** | EM+OD → failed-experiment markers consolidated, superseded protocols invalidated; AU+RB → GxP/IRB traceability | Lab-notebook memory |

### 3.4 Regulatory drivers (hard requirements)

| Instrument | Mapped capability | Consequence |
|---|---|---|
| **EU AI Act Art. 12** (Annex III standalone from **2 Dec 2027**) | AU + RB — automatic lifetime event logging | Fines €15M / 3% turnover; ≥6-month retention floor |
| **HIPAA** §164.312(b), §164.502(b) | AU; Minimum Necessary extends to memory stores | Memory layer = BAA-covered; auditable + 6-yr retention |
| **SEC 17a-4 / FINRA 4511 / MiFID II** | RB + AU — tamper-evident, append-only, point-in-time | Hash-chained records; reconstructable at T |
| **GDPR Art. 17 / CCPA** | Provable forgetting — audit must survive erasure | **Build:** crypto-shred-with-audit-certificate (the Lians pattern) |
| **Singapore IMDA Agentic Framework (Jan 2026) → MetaComp KYA; IMF Notes 2026/004** | AU | "Knowing your agent" becoming licensable |

**Time-to-market trigger:** EU AI Act Art. 12 enforcement Dec 2027 → ~1-year procurement runway for audit-ready buyers. Ship audit-export + point-in-time rebuild as headline features now.

---

## 4. Competitive landscape

### 4.1 The commercial set

| Platform | Funding | Pricing (2026) | OD | DK | RB | AU |
|---|---|---|---|---|---|---|
| **Mem0** | $24.5M ($3.9M seed Kindred; $20M Series A Basis Set/YC/Peak XV/GitHub Fund) | Free 10K → $19/mo → $249/mo Pro (graph gated at Pro) | update only, no history | ✗ | ✗ | ✗ |
| **Zep / Graphiti** | ~$2.3M pre-seed; ~$1M ARR bootstrapped | Free 10K credits → $25–125/mo Flex | ✅ `invalid_at` edges | ✗ | point-in-time *queries* only | ✗ |
| **Letta / MemGPT** | $10M seed (Felicis), ~$70M post | OSS free; managed ~$20–200/mo | agent rewrites, no record | ✗ | ✗ | ✗ |
| **Cognee** | $7.5M seed | Free OSS; Cloud $35–750; Ent ~$1,970/mo | timestamped, not enforced | ✗ | ✗ | ✗ |
| **LangMem** | free (LangChain, MIT) | Free | ✗ | ✗ | ✗ | ✗ |
| **MemoryBank** (research) | — | — | ✅ threshold-decay | ✅ | ✗ | ✗ |

### 4.2 Feature-competitive OSS (not company-competitive yet)

| System | Ships | Missing vs NMAFC |
|---|---|---|
| **Cortex** | Self-organizing graph, decay, contradiction detection | No rollback-to-turn; no full audit log; no managed offering |
| **Hakuya** | Confidence decay, contradiction detection, tamper-evident audit ledger | No consolidation; no graded hydration |
| **CraniMem** | Gated transfer, consolidation stages | Experimental; no product surface |
| **SF-AMS** (arXiv 2607.05787) | Strategic forgetting improves multi-hop F1 +9.65, temporal +6.91 | Research-scale only |
| **Membread** | Bi-temporal graph, `as_of` queries, hybrid retrieval, MCP | Not an audit log; no consolidation/decay |
| **Lians** | Append-only, SHA-256 hash chain, bitemporal, crypto-shred erasure | Ledger, not a cognitive engine — no decay/graph/consolidation |
| **ZettelForge** | CTI supersession, OCSF audit | No decay; no rollback |

### 4.3 NMAFC gap-analysis conclusion

**What no incumbent ships (the moat combinations):**

1. **Decay + spaced-repetition + reinforcement** — commercially unclaimed by all four major platforms; only research/early OSS have decay forms, and SF-AMS/CraniMem show decay *improves* the exact temporal/multi-hop categories NMAFC wins. Vendors accumulate stale junk (~97.8%-garbage audit).
2. **Rollback-to-full-state** from an append-only cognitive event log — none of the commercial four; Zep only has point-in-time *queries*.
3. **Full cognitive audit event log** — 2026 surveys explicitly call audit trails + retention "gaps across all five" (Mem0, Zep, Letta, LangMem, Cognee).
4. **The combination** — no single product ships decay + override + rollback + audit + consolidation + graded hydration together.

**Careful reads (what NOT to lead with):**

- **Zep already owns supersession/temporal** — don't lead with "temporal invalidation" alone.
- **LoCoMo deltas are commoditizing** — Mem0 markets near-collinear numbers (+29.6 temporal / +23.1 multi-hop). Benchmark-only pitches are table stakes. Counter with *memory behaviour*: decay-correctness, supersession correctness, reconstructability.
- **Letta owns consolidation-like behaviour** (sleep-time compute) but it's LLM-driven; NMAFC's EM is deterministic.

**Positioning:** lead wedge = **regulated memory accountability** (RB + AU: "what did the agent know *when* it acted?"); corroborating wedge = **decay-correctness** ("persistence without decay is confident misinformation").

---

## 5. Market evidence

### 5.1 Funding & strategic signals — memory is now a venture category

- **Mem0 $24M** (Oct 2025); API calls 35M → **186M/quarter** (~30% MoM); **AWS-exclusive memory provider for the Agent SDK**; 80k+ developer signups.
- **Letta $10M @ ~$70M post** (Sep 2024, Felicis; angels incl. Jeff Dean).
- **Cognee $7.5M seed.** Zep bootstrapped ~$1M ARR.
- **Meta acquired Manus** (agent-memory startup) Dec 2025 — memory as M&A target.
- **Every major platform now owns a memory story:** AWS (Mem0 exclusive), Twilio Conversation Memory, Redis Agent Memory Server, Google Vertex AI Memory Bank, Oracle governed agent-memory, Anthropic+Mem0 connector.

### 5.2 Market sizing (analysts disagree — treat as range)

| Measure | 2025 | Forecast | CAGR |
|---|---|---|---|
| Agentic AI orchestration + memory | $6.27–6.49B | $28.45–33.54B (2030) | 35–39% |
| AI-agent memory infrastructure | $1.2B | $18.9B (2034) | **62%** |
| AI-agent memory platform (conservative) | $1.15B | $2.45B (2034) | 7.8% |

### 5.3 Demand signals

- **Memory persistence = #2 most-requested feature in enterprise AI-agent RFPs (Q1 2026)**, second only to security/compliance (120+ buyer interviews).
- **~65% of enterprise AI-agent failures trace to context drift**, not model capability.
- ~35% of Fortune 500 run production multi-agent deployments; Gartner: 40% of enterprise apps integrate task-specific agents by end-2026.
- Memory-specific evals (LoCoMo, LongMemEval) are now standard — "did the agent remember the right fact at the right time" is a measured capability.

---

## 6. Distribution & easy-usage strategy

### 6.1 The proven ramp

Every successful competitor converged on the same architecture:

> **OSS library (`pip`) → self-hosted server (Docker) → managed cloud API**, pushed up the ramp by **OpenAI-compatible middleware, MCP servers, and framework adapters**.

Mem0 presents it explicitly: "Library · Self-Hosted Server · Cloud Platform."

NMAFC already has (a) a real editable-install package + CLI + extras, and most of (b) a FastAPI server + dashboard + `nmafc start --production`. Missing: **PyPI publication**, **all three distribution channels**, and the hosted tier (future).

### 6.2 The one blocking dependency

**Add two thin stateless methods to `NeuromorphicMemory`:**

- `recall(query, ...) → bounded context string` — over `router.retrieve` + `format_context`
- `remember(messages, ...) → extract-only mode` — StateExtractor without response generation → `process_updates` → decay → consolidate

**Do NOT** route integrations through `process_turn()` — it generates a response and would double-generate on the user's own LLM call. These two methods unblock paths 1–4 simultaneously.

### 6.3 Ranked integration paths

| # | Path | Effort | What users change | Why |
|---|---|---|---|---|
| **1** | **OpenAI-compatible middleware** — client wrapper (`MemoryOpenAI(OpenAI(), memory=mem)`) + `base_url` HTTP proxy (`OpenAI(base_url="http://localhost:8000")`) | **S** | 1–2 lines | Mem0's most-cited feature; language-agnostic via proxy; `stream=True` passes through, extraction runs post-stream, abort-safe |
| **2** | **MCP server** (`uvx nmafc-mcp`) — `recall`/`remember`/`forget` **plus `SymbolIndex` code tools** (`index_repo`, `find_callers`, `repo_status`) | **S–M** | Zero code — Claude Desktop/Cursor/Codex config | 22,000+ MCP servers indexed; plugs the unintegrated AST code-memory subsystem — exact "what calls this?" beats embedding-RAG for coding agents |
| **3** | **REST `/v1/memories/*`** — polish existing FastAPI to Mem0 semantics (`add`/`search`/`history`) + API-key auth | **S** | Standard REST | ~90% exists; trivial migration path for Mem0 users |
| **4** | **LangGraph `BaseStore` / LlamaIndex `MemoryBlock` / CrewAI adapters** | **M** | One import | Mem0 attributes much of its 30% MoM growth to framework bundling |
| **5** | **Docker one-liner** — static-export dashboard into one image + pgvector compose | **S** (ops) | `docker compose up` | Today's blocker = "npm install + two dev servers." Killer line: **LanceDB is embedded — no standalone vector DB, just `./data`** |
| **6** | **CLI wizard depth** — `nmafc init` key-discovery + live smoke turn + integrate snippet; `nmafc eval locomo`; `nmafc code index .` | **S** | `nmafc init` | The "my first memory" moment; first-run retention |
| **7** | **Managed cloud API** | **L** | `base_url` swap | Design the seam now (SDK + MCP accept `base_url`), build later |

### 6.4 MCP server sketch (the zero-code channel)

```python
@mcp.tool() async def recall(query, agent_id="default", top_k=10): ...
@mcp.tool() async def remember(turn, agent_id="default"): ...
@mcp.tool() async def forget(memory_id, agent_id="default"): ...
@mcp.tool() async def index_repo(path): ...        # SymbolIndex — differentiator
@mcp.tool() async def find_callers(symbol, path): ...
@mcp.tool() async def repo_status(path): ...
```

Distribution (mirror mem0-mcp playbook): PyPI `nmafc-mcp` → Docker Streamable-HTTP `/mcp` → official MCP Registry (`mcp-publisher`) → Smithery → plugin manifests for Claude Code/Cursor/Codex. Hosted `https://mcp.<domain>/mcp` = natural **first paid cloud feature** (~80% of MCP users prefer remote).

### 6.5 Go-to-market formula

1. **Publish the LoCoMo result as a paper + open reproducibility harness.** The McNemar-paired, gross +N/−M rigor is a credibility moat; Mem0, Zep, and Letta each converted "beats RAG with less context" into stars and funding.
2. **Publish to PyPI this week** (Trusted Publishing; `py.typed` exists; entry points defined).
3. **Keep the memory engine fully open-source — including decay/temporal/consolidation.** Mem0 gated decay to cloud and it's cited as a cautionary tale. Gate **operations**: hosting, dashboard, audit-log export, SSO, SOC 2/HIPAA, managed storage.
4. **Usage-based freemium ladder** — proven: Free → $19 → $249 → enterprise custom.
5. **Target default-memory status in one distribution channel** (CrewAI/Flowise/Langflow, or a hosted MCP endpoint — lowest-cash path to first revenue for a solo project).

### 6.6 Fastest DX wins (independent of strategy)

1. **Zero-config construction** — `NeuromorphicMemory.from_openai()`: keys from env, `gpt-4o-mini` default, existing embedding-dim probe. Today's on-ramp requires install + keys + `.env` + TOML + wrapper code; this one change *is* Mem0's adoption curve.
2. **Streaming** — extraction runs on the reassembled transcript post-stream, abort-safe; never sit between user and tokens.
3. **One-env-var observability** — OTel spans on the 5 pipeline stages (`retrieve`, `extract`, `decay`, `prune`, `consolidate`) → Langfuse/LangSmith via `NMAFC_OTEL_EXPORTER=`.
4. **Tenant scoping as body/query params**, not just `X-NMAFC-*` headers (MCP and wrappers can't always set headers).
5. **Packaging hygiene** — stale `uv.lock`, no CI, 300 ruff errors, the staged `SyntaxError` in `_ab_vs_rag.py` — all matter more for adoption than any new feature.
6. **Docs for AI clients** — `llms.txt` + docs-MCP mirror (Zep and Mem0 both publish these).

### 6.7 Combined recommendation (the plan this research feeds)

**Positioning:** *"Auditable memory for LLM agents"* — lead with rollback + cognitive audit log (EU AI Act Art. 12, Dec 2027), corroborate with decay-correctness against the stale-memory problem, benchmark quietly.

**Build order:** `recall()`/`remember()` → OpenAI middleware + MCP server (with SymbolIndex tools) → PyPI publish + paper → Docker/dashboard polish → framework adapters → hosted API.

**Industry beachheads:** healthcare (HIPAA audit replay), finance (point-in-time examiner logs), developer tools (MCP + code memory) — in that order of willingness-to-pay.

**Do not:** lead with LoCoMo deltas alone; lead with supersession alone (Zep owns it); open-core-gate the decay/temporal algorithms; spend more effort on the supersession detector (the signal doesn't exist in the embeddings).

---

## 7. Sources

**Funding / market**
- TechCrunch: Mem0 $24M (2025-10-28); Letta $10M @ $70M (2024-09-23)
- mem0.ai/series-a; sacra.com/c/mem0
- Mordor Intelligence (agentic AI orchestration + memory); TBRC/ResearchAndMarkets; Market Intelo; Intel Market Research
- Gartner (40% enterprise apps by end-2026); McKinsey State of AI 2025; Value Add VC (context-drift 65%)
- Meta acquired Manus (GII Research / deal tracking, Dec 2025)

**Regulatory**
- EUR-Lex EU AI Act Art. 12/19/99 + Digital Omnibus deferral; HIPAA §164.312(b)/§164.502(b); SEC 17a-4; FINRA 4511; MiFID II; GDPR Art. 17
- Singapore IMDA Model AI Governance Framework for Agentic AI (Jan 2026); MetaComp StableX KYA (Apr 2026); IMF Notes 2026/004
- GAO/OMB federal AI use-case counts (3,611, 2025)

**Competitors**
- github.com/mem0ai/mem0 · docs.mem0.ai · mem0.ai/pricing · mem0-mcp · mem0-plugin
- github.com/getzep/graphiti · getzep.com/pricing · arxiv.org/abs/2501.13956
- github.com/letta-ai/letta · pypi.org/project/letta-mcp-server
- github.com/chroma-core/chroma · trychroma.com (funding: $18M seed @ ~$75M)
- pinecone.io · techtarget.com 2024-08-27
- LangGraph persistence docs; LlamaIndex memory-block docs

**Adjacent / feature-competitive OSS**
- Cortex (OpenAgents-Labs); Hakuya; CraniMem; SF-AMS arXiv:2607.05787; Membread (G-SaiVishwas); Lians-ai/Lians; ZettelForge; MemoryBank
- DoctorAgent; OpenHealthAgents; Cascade Protocol; ZettelForge; MAAT-SOC; AuraXP
- TASA arXiv:2511.15163; DREAM arXiv:2608.05170; Studeia docs; MemoryLake; Serif/OpenClaw HR

**Evals / methodology**
- LoCoMo; LongMemEval arXiv:2410.10813; LongMemEval-V2; MemDelta arXiv:2606.29914
- datapace.ai survey (Jul 2026); CallSphere engine-level view (Apr 2026)

**MCP ecosystem**
- pulsemcp.com (≥22,000 servers); mcpmanager.ai adoption stats; modelcontextprotocol.io registry + roadmap

**Observability / DX**
- langfuse.com OpenTelemetry integration; langchain.com OpenTelemetry-LangSmith post

**Internal**
- `WHAT-CHANGED.md` (canonical change history), `RECENT_OPS.md` (session logs), `README.md`, `scripts/benchmarks/results/paired_2026_09_10/`
