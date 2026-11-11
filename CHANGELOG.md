# Changelog

All notable changes to NMAFC are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Note: versions below `0.2.0` were not published to PyPI. They are recorded here
from the git history so the release trail is continuous.

## [Unreleased]

### Added

- `LICENSE` (PolyForm Noncommercial 1.0.0), `NOTICE`, `CONTRIBUTING.md`,
  `CONTRIBUTOR_LICENSE_AGREEMENT.md`, `SECURITY.md`.
- Project metadata: licence, classifiers, keywords, `[project.urls]`.
- `NOTICE` records LoCoMo benchmark attribution under CC BY-NC 4.0.

### Changed

- Packaging declares both authors.
- Licence switched from Apache-2.0 to PolyForm Noncommercial 1.0.0: free for
  any non-commercial use (personal, research, education, non-profit,
  government); commercial use now requires a separate licence. Packaging
  metadata, `REUSE.toml`, `NOTICE`, `CITATION.cff`, `CONTRIBUTING.md` and the
  CLA updated to match. The LoCoMo dataset remains CC BY-NC 4.0.

### Known gaps

See `SECURITY.md` for the current security posture and the plan phase that
addresses each item. The most significant: the web server defaults open, and the
`record_id` path parameter is interpolated unescaped into LanceDB predicates.

---

## [0.1.0] — unreleased

Initial internal version. Never published to PyPI.

### Added

- Two-tier memory substrate: LanceDB hot tier (ANN cosine), SQLite cold tier
  (FTS5 + archive vectors).
- Exponential decay with long-term potentiation across three memory types
  (`CoreAnchor`, `ActiveContext`, `EphemeralState`).
- Five-channel reciprocal rank fusion with an additive grounding bonus.
- Graded hydration: top-20 atomic facts plus hydrated source turns.
- `NeuromorphicMemory` (async) and `SyncNeuromorphicMemory` in-process API:
  `process_turn`, `recall`, `remember`, `forget_entity`, `forget_record`,
  `ingest_updates`, `consolidate`, `rollback`.
- Providers: OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, plus
  OpenAI-compatible endpoints (Groq, OpenRouter, Together, Ollama, LM Studio,
  vLLM) and local `fastembed` embeddings.
- Postgres cold-tier backend (`psycopg2`).
- FastAPI web server, REST API, and in-app memory explorer.
- MCP server exposing `memory_recall`, `memory_remember`, `memory_forget`, and
  the offline code-memory tools `index_repo`, `find_callers`, `repo_status`.
- CLI: `nmafc start`, `init`, `chat`, `eval locomo`.
- Paired LoCoMo benchmark harness with McNemar exact testing.
- Code-memory subsystem: AST symbol index with forward and reverse references.
- 625 tests.

### Known issues

- No licence file (fixed post-0.1.0, above).
- `uv.lock` stale; `mcp` absent from the resolved set.
- Web server defaults open; no key-to-tenant binding; no rate limiting.
- `record_id` interpolated unescaped into LanceDB predicates.
- Postgres backend has no tests and discards embeddings.
- Container image installs `.[web]` only, so it cannot call an LLM, serve MCP,
  or use Postgres.
- `fastembed` is advertised in the README but is in no dependency group.
- `examples/custom_provider.py` calls `from_providers`, which exists only on the
  sync class.

## [Unreleased]: https://github.com/blok-hamster/nmafc/compare/v0.1.0...HEAD