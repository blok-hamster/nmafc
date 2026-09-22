# NMAFC Implementation Plan

Execution plan for the combined recommendation in [RESEARCH.md §6.7](RESEARCH.md#67-combined-recommendation-the-plan-this-research-feeds):

> **Positioning:** "Auditable memory for LLM agents" — lead with rollback + cognitive audit log (EU AI Act Art. 12, Dec 2027), corroborate with decay-correctness, benchmark quietly.
>
> **Build order:** `recall()`/`remember()` → OpenAI middleware + MCP server → PyPI publish + paper → Docker/dashboard polish → framework adapters → hosted API.
>
> **Beachheads:** healthcare, finance, developer tools.

Compiled: 22 Sep 2026.

---

## Principles

1. **Hygiene before features.** Nothing ships on a broken lockfile, a staged `SyntaxError`, or no CI. Phase 0 gates everything.
2. **`recall()`/`remember()` unblock everything.** They are the single blocking dependency for middleware, MCP, REST, and framework adapters — do them first among features.
3. **Never route integrations through `process_turn()`.** It generates a response and would double-generate on the user's LLM call.
4. **Keep the engine OSS — including decay/temporal/consolidation.** Gate operations (hosting, dashboard, audit export, SSO), never the algorithm.
5. **Do not** lead with LoCoMo deltas alone, lead with supersession alone, spend more effort on the supersession detector, or re-ingest stores without a write-time reason.
6. **Measure properly** per `WHAT-CHANGED.md` methodology: paired runs, gross +N/−M, ≥800 questions for claims.

---

## Phase 0 — Hygiene (gates everything)

> Found in the repo audit. No feature work merges until these pass.

| # | Task | Detail | Done when |
|---|---|---|---|
| 0.1 | Finish or abort the in-progress merge | ~90 staged files, unstaged edits layered on top, 5 unpushed commits | `git status` clean; merge committed or `git merge --abort` |
| 0.2 | Fix `_ab_vs_rag.py:245` | Python 3.12 multi-line f-string syntax; project targets ≥3.11 | `python -m py_compile scripts/benchmarks/_ab_vs_rag.py` passes on 3.11 |
| 0.3 | Delete stray `file:/` tree | LanceDB manifests from a URI-as-path bug at repo root | Directory gone; added to `.gitignore` if root cause not yet fixed |
| 0.4 | Resync lockfile | `uv lock` — currently missing httptools, psycopg2-binary, python-dotenv, starlette, uvicorn, websockets, etc. | `uv lock --check` passes |
| 0.5 | Mark the live Azure test | `@pytest.mark.live` on `tests/integration/test_azure_deepseek.py`; register `markers` in `pyproject.toml` | Bare `pytest` makes zero network calls |
| 0.6 | Minimal CI | GitHub Actions: `ruff check`, `pytest -m "not live"`, `uv lock --check`, `python -m py_compile` on all files | Green check on PR |
| 0.7 | Auto-fix ruff + fix `F821` | 130 auto-fixables; `pruning.py:56` `MemoryEvent` import | `ruff check .` clean or ≤ agreed budget with zero F/E-syntax classes |
| 0.8 | Fix README testing section | Nonexistent paths (`tests/test_decay.py` → `tests/unit/…`); broken `--cov` (add `pytest-cov` or remove the claim) | Documented commands run verbatim |

**Exit criteria:** CI green on a clean checkout; `uv sync && pytest -m "not live"` passes for a new contributor with zero manual steps.

---

## Phase 1 — Core primitives (the blocking dependency)

### 1A. `recall()` and `remember()` on `NeuromorphicMemory`

| # | Task | Detail | Done when |
|---|---|---|---|
| 1.1 | `async def recall(query, *, agent_id=None, conversation_id=None, top_k=None, turn=None) -> RecallResult` | Thin over `query_router.retrieve` + `format_context`. Returns bounded context string + raw hits (ids, scores, sources) + token estimate. **Read-only** (uses `close_readonly` / deferred reinforcement semantics — never mutates on read) | Unit tests: returns bounded string; no disk writes; honors tenant scope |
| 1.2 | `async def remember(messages, *, agent_id=None, conversation_id=None) -> RememberResult` | **Extract-only mode**: StateExtractor prompt minus response generation → `_process_updates` → decay → prune → scheduled consolidation. New `NMAFC_EXTRACTOR_MODE=extract_only` or a `generate_response: bool` flag on the extractor | Unit tests: facts land in Hot+Cold; no assistant response produced; overrides/links still processed |
| 1.3 | Sync variants | `SyncNeuromorphicMemory.recall/remember` | Mirror tests pass |
| 1.4 | Streaming-safe contract | Document + test that `remember()` accepts a full reassembled transcript and runs *after* the stream, abort-safe | Test: partial/aborted transcript → graceful no-op or partial extract, never exception to caller |
| 1.5 | Tenant params, not just headers | `agent_id`/`conversation_id` accepted as method args and later as body/query params | Tests pass with args and with default-env behavior |

**Exit criteria:** `recall()` + `remember()` usable standalone in a 10-line example (`examples/recall_remember.py`) with zero config beyond keys.

### 1B. Zero-config construction

| # | Task | Detail | Done when |
|---|---|---|---|
| 1.6 | `NeuromorphicMemory.from_env()` / `from_openai()` | Reads `OPENAI_API_KEY` (or chosen provider), defaults `gpt-4o-mini` + `text-embedding-3-small`, runs existing embedding-dim probe, `config_path=None` ⇒ pure defaults | `NeuromorphicMemory.from_env()` works with only `OPENAI_API_KEY` set |
| 1.7 | `nmafc init` depth | Key detection → write `.env` + config → one live smoke turn ("its first memory") → print the 3-line integrate snippet | Wizard completes on a fresh machine; smoke turn visible in dashboard |

**Exit criteria:** Time-to-first-memory < 2 minutes from `pip install` with one API key.

---

## Phase 2 — Distribution channels

### 2A. OpenAI-compatible middleware (Path 1, effort S)

| # | Task | Detail | Done when |
|---|---|---|---|
| 2.1 | Client wrapper `MemoryOpenAI` | `nmafc/proxy.py`: wraps `OpenAI`/`AsyncOpenAI`; `__getattr__` delegates; interposes only `chat.completions.create`: recall → bounded preamble → forward → post-response `remember()`. `stream=True` forwarded as SSE; extraction on reassembled transcript after | Integration test against a mock server: call sites unchanged; memory populated; streaming works |
| 2.2 | HTTP proxy `/v1/chat/completions` | Reuse FastAPI scaffold; same recall → inject → forward → remember behavior; tenant via body params | `OpenAI(base_url="http://localhost:8000")` passes the same test |
| 2.3 | Docs snippet | README "3 lines with memory" section | Copy-paste works verbatim |

**Exit criteria:** A user with an existing OpenAI app adds memory by changing 1–2 lines; streaming unbroken.

### 2B. MCP server (Path 2, effort S–M)

| # | Task | Detail | Done when |
|---|---|---|---|
| 2.4 | `nmafc-mcp` package + entry point | FastMCP tools: `recall`, `remember`, `forget`, `index_repo`, `find_callers`, `repo_status` (plugs `src/nmafc/code/` SymbolIndex — the differentiator vs Mem0/Zep MCP) | `uvx nmafc-mcp` runs with `NMAFC_DATA_DIR` config only |
| 2.5 | Distribution | PyPI package; Docker Streamable-HTTP `/mcp`; publish `server.json` to official MCP Registry; list on Smithery; plugin manifests for Claude Code/Cursor/Codex (`.claude-plugin/`, `.cursor-mcp.json`, `.codex-mcp.json`) | Installable in Claude Desktop with one command |
| 2.6 | Code-memory integration test | `index_repo` → `find_callers` → stale detection on this repo | Returns exact callers for a known symbol; staleness flagged after edit |

**Exit criteria:** A Claude/Cursor user gets persistent memory + code navigation with zero application code.

### 2C. REST `/v1/memories/*` polish (Path 3, effort S)

| # | Task | Detail | Done when |
|---|---|---|---|
| 2.7 | Mem0-style endpoints | `POST /v1/memories/add`, `/search`, `GET/PATCH/DELETE /v1/memories/{id}`, `GET /v1/memories/{id}/history`; `POST /v1/conversations/{id}/process` wrapping `process_turn` | Contract tests against documented schema |
| 2.8 | API-key auth | `Authorization: Bearer nmafc_…` alongside `X-NMAFC-*` headers; tenant scoping in body/query **and** headers | Both auth styles pass |
| 2.9 | `llms.txt` + OpenAPI publish | Serve `/docs`, `llms.txt` | Linked from README |

**Exit criteria:** A Mem0 user can migrate by changing base URL + key.

---

## Phase 3 — Publishing & credibility

| # | Task | Detail | Done when |
|---|---|---|---|
| 3.1 | Publish to PyPI | Trusted Publishing; verify `pip` (non-uv) install; keep `lancedb<=0.25.3` pin verified for wheels; `py.typed` (exists) | `pip install nmafc[all]` works on a clean machine |
| 3.2 | Reproducibility harness | `nmafc eval locomo` wrapping `scripts/benchmarks/run_locomo.py` + `summarise.py`; document paired-run methodology | Third party regenerates the README table from committed JSON with no API calls |
| 3.3 | ~~Paper / write-up~~ **DONE** | The research paper already exists (see repo). Align the paper with the committed results; backfill any omitted negative results. | Paper references the committed paired_2026_09_10 data |
| 3.4 | Docker one-liner (Path 5) | Static-export dashboard into one image; `docker-compose.yml` with optional pgvector for Cold ROM; document "LanceDB is embedded — just `./data`" | `docker compose up` serves API + dashboard on one port |

**Exit criteria:** `pip install nmafc` + paper + Docker all live simultaneously (the ramp exists end-to-end: library → server → [cloud later]).

---

## Phase 4 — DX & observability

| # | Task | Detail | Done when |
|---|---|---|---|
| 4.1 | OTel spans | 5 pipeline stages (`retrieve`, `extract`, `decay`, `prune`, `consolidate`) with `gen_ai.*` conventions; `NMAFC_OTEL_EXPORTER=` fans out to Langfuse/LangSmith/console; `nmafc[obs]` extra | One env var shows traces in Langfuse |
| 4.2 | Eval integrations | RAGAS (`context_recall`/`precision`/`faithfulness`) + DeepEval on the same retrievals; LongMemEval-V2-style Insert/Query wrapper around `recall`/`remember` | `nmafc eval bench` reproduces README table with McNemar pairs |
| 4.3 | Framework adapters (Path 4) | LangGraph `BaseStore` (`namespace=(agent_id, conversation_id)`), LlamaIndex `NmafcMemoryBlock`, CrewAI provider — separate packages per `langgraph-checkpoint-*` convention | Each adapter passes its framework's store/memory tests |
| 4.4 | Extraction recall work (from roadmap) | The #1 insight from WHAT-CHANGED: 90% of misses are write-time. Target: qualifiers, quantities, destinations (destinations currently unmeasured — measure first). Batch with any `CoreAnchor` reclassification; **expect LoCoMo to move honestly** | Measurement first (`_audit`-style), then paired A/B ≥800 q |

**Exit criteria:** Observability in one env var; adapters installable; extraction improvements measured to methodology rules (not 108-question reads).

---

## Phase 5 — Compliance wedge (the positioning made real)

Aligns to EU AI Act Art. 12 enforcement (2 Dec 2027) — the ~1-year procurement runway.

| # | Task | Detail | Done when |
|---|---|---|---|
| 5.1 | Audit export | `nmafc audit export --from --to` → hash-chained, SIEM-friendly (OCSF-shaped) bundle of cognitive events | Export verifies (recomputed chain matches); importable by a mock examiner tool |
| 5.2 | Point-in-time API | `recall_at(turn_or_timestamp)` — public wrapper over existing `rollback`/rebuild path **without mutating live state** (read-only reconstruction) | Test: query at T returns only facts valid at T; live store untouched |
| 5.3 | Erasure path | GDPR Art. 17: crypto-shred-with-audit-certificate (Lians pattern) — delete payload, keep audit skeleton | Erasure removes content; audit trail still proves the event happened |
| 5.4 | Vertical landing pages | Healthcare (HIPAA replay), finance (examiner logs), developer tools (MCP + code memory) — in that willingness-to-pay order | Pages cite the specific capabilities + regulation clauses |

**Exit criteria:** An enterprise buyer can answer "what did the agent know when it acted?" and "prove you deleted user X's data" with shipped features, not slides.

---

## Sequencing & dependencies

```
Phase 0 (hygiene)          ──► gates all merges
    │
Phase 1A recall/remember   ──► blocks 2A, 2B, 2C, 4.3
Phase 1B zero-config       ──► blocks 3.1 time-to-value
    │
    ├─► 2A OpenAI middleware   ─┐
    ├─► 2B MCP server          ─┼─► 3.1 PyPI ─► 3.2 harness ─► 3.3 paper
    └─► 2C REST polish         ─┘        └─► 3.4 Docker
    │
Phase 4 DX/obs/adapters    (parallel after Phase 1)
Phase 5 compliance wedge   (parallel after Phase 1; audit export reads EventLog)
```

**Suggested milestones**

| Milestone | Contents | Rough effort* |
|---|---|---|
| **M0 — Clean repo** | Phase 0 fully exit-criteried | days |
| **M1 — Two-line integration** | Phase 1 + 2A + minimal 2C auth | ~1–2 weeks |
| **M2 — Zero-code agents** | 2B MCP + code tools + registry listing | ~1 week after M1 |
| **M3 — Public ramp** | Phase 3 (PyPI + harness + Docker; paper can trail) | ~1 week after M1 |
| **M4 — Enterprise wedge** | Phase 5 + 4.1 observability | ~2–3 weeks |
| **M5 — Ecosystem** | Framework adapters + extraction-recall campaign | ongoing |

\* Single-developer rough order-of-magnitude; Phase 0 and measurement-gated items (4.4) have hard external dependencies (runs, ≥800-q A/Bs).

---

## What we are explicitly NOT doing

- **No supersession-detector work** — signal doesn't exist in the embeddings (0 exact matches / 8,276 facts; LLM retry was a coin flip).
- **No re-ingestion** unless batched with a write-time change (stores last written 10 Sept; ~6,000 questions answered since).
- **No `weight_signal` / decay-as-ranking** — measured −5.3, stays 0.0.
- **No list-gate / width / prompt-clause shipping** — measured null or trade-off.
- **No open-core gating of decay/temporal/consolidation** — the engine stays OSS; monetize operations.
- **No benchmark-only marketing** — adversarial −14.6 stays in the paper; LoCoMo deltas are commoditizing.
- **No hosted cloud build yet** — only the `base_url` seam (design it in 2A/2B; build after ramp proves demand).

---

## Success metrics

| Stage | Metric |
|---|---|
| M0 | CI green; `uv lock --check` passes; bare pytest = 0 network calls |
| M1 | Integration in ≤2 lines; time-to-first-memory < 2 min |
| M2 | `uvx nmafc-mcp` installs into Claude Desktop; code tools return exact callers |
| M3 | PyPI installs succeed via plain `pip`; third party regenerates benchmark table; `docker compose up` one-command demo |
| M4 | Audit export chain verifies; `recall_at` read-only proof; erasure keeps audit |
| Adoption | PyPI downloads, MCP installs, API-call growth (the Mem0 underwriting metric), framework-adapter usage |

---

## Immediate next actions (this session, if approved)

1. Phase 0: resolve git merge state (0.1), fix `_ab_vs_rag.py` (0.2), delete `file:/` (0.3), `uv lock` (0.4).
2. Phase 0: mark live test + register markers (0.5), add CI workflow (0.6).
3. Phase 1: design `RecallResult`/`RememberResult` schemas against existing `SearchResult`/`UnifiedMemoryPayload` — then implement 1.1/1.2.

See [RESEARCH.md](RESEARCH.md) for the full research backing this plan.
