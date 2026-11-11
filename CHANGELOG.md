# Changelog

All notable changes to NMAFC are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Note: versions below `0.2.0` were not published to PyPI. They are recorded here
from the git history so the release trail is continuous.

## [Unreleased]

_Nothing yet._

## [0.2.0] — unreleased

Licence, packaging and library-usability release. First version intended for
PyPI.

### Added

- `LICENSE` (PolyForm Noncommercial 1.0.0), `NOTICE`, `CONTRIBUTING.md`,
  `CONTRIBUTOR_LICENSE_AGREEMENT.md`, `SECURITY.md`.
- Project metadata: licence, classifiers, keywords, `[project.urls]`.
- `NOTICE` records LoCoMo benchmark attribution under CC BY-NC 4.0.
- Public exception hierarchy `nmafc.exceptions` (`NmafcError`,
  `MissingAPIKeyError`, `MissingExtraError`, `ProviderNotConfiguredError`,
  `StorageError`), exported from the top-level package together with the full
  public surface (`__all__`).
- Provider-aware errors at the public boundary: a missing key names the env var
  for the selected provider (e.g. `GROQ_API_KEY`); an unconfigured Azure route
  names `AZURE_OPENAI_ENDPOINT`; a missing optional SDK names the exact
  `nmafc[extra]` install command.
- `SyncNeuromorphicMemory.from_env` / `from_openai` constructors, for parity
  with the async class.
- `StorageConfig.storage_options`, forwarded to `lancedb.connect` — region,
  endpoint and session credentials for S3-compatible hot stores.
- `fastembed` dependency group: `pip install nmafc[fastembed]` for key-free
  local embeddings.
- PostgreSQL cold-store integration suite, run in CI against an ephemeral
  Postgres service container (previously the backend had no tests at all).
- `CONTRIBUTING.md` release runbook; tag-driven release workflow publishing via
  PyPI Trusted Publishing (no API token in CI).

### Changed

- Licence switched from Apache-2.0 to PolyForm Noncommercial 1.0.0: free for
  any non-commercial use (personal, research, education, non-profit,
  government); commercial use now requires a separate licence. Packaging
  metadata, `REUSE.toml`, `NOTICE`, `CITATION.cff`, `CONTRIBUTING.md` and the
  CLA updated to match. The LoCoMo dataset remains CC BY-NC 4.0.
- Packaging declares both authors.
- Default data location is the per-user platform directory (macOS Application
  Support / XDG data home / `%APPDATA%`), overridable with `NMAFC_DATA_DIR`.
  The old cwd-relative `./data` default stranded data wherever a script ran.
  Nothing moves automatically: a one-time notice names the
  `NMAFC_DATA_DIR=./data` escape hatch when a legacy tree is found.
- Hot-store connect failures raise `StorageError` — S3 paths name the AWS
  credential chain, local paths name the file/permission check.
- `nmafc init` prompts with the resolved data directory, shows it in the final
  panel, and generates snippets that import from the top-level package.
- Container image pins `NMAFC_DATA_DIR=/app/data` so the compose volume keeps
  binding the stores.
- CI: Python 3.11/3.12/3.13 matrix; a wheel job that builds the distributions,
  audits their contents, and smoke-tests a clean-venv install; the Postgres
  cold-store job.

### Fixed

- Record IDs were interpolated into LanceDB SQL filters inside bare single
  quotes; every id, entity name and scope value now goes through SQL string
  escaping first (208 new tests covering quote-bearing and malicious IDs).

### Known gaps

See `SECURITY.md` for the current security posture and the plan phase that
addresses each item. The most significant remaining: the web server defaults
open.

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