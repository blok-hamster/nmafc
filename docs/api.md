# API reference

The stable public surface of NMAFC is exactly what `nmafc.__all__` lists.
Import from the package root:

```python
from nmafc import (
    NeuromorphicMemory, SyncNeuromorphicMemory, NMafcConfig,
    RecallResult, RememberResult,
    NmafcError, MissingAPIKeyError, MissingExtraError,
    ProviderNotConfiguredError, StorageError,
)
```

Everything not listed there — deep modules (`nmafc.engine.*`,
`nmafc.storage.*`, `nmafc.integration.*`), the web server, the CLI — is
internal and may change between any two releases.

---

## `NeuromorphicMemory`

The async, in-process memory object. Construct via classmethods; there is no
reason to call `__init__` directly.

### Constructors

| Classmethod | Behaviour |
|---|---|
| `from_openai()` | OpenAI route (`gpt-4o-mini` + `text-embedding-3-small`); reads `OPENAI_API_KEY`. Raises `MissingAPIKeyError` naming the var when unset. |
| `from_env(config_path=None)` | Env vars decide everything (optionally pre-loaded from a TOML file whose values env vars may override). |
| `from_config(config=None, config_path="configs/default.toml")` | Build from an `NMafcConfig` you pass in; when `config` is None, load from `config_path`. |

`await NeuromorphicMemory.from_openai()` — the classmethods are async and
return an awaiting-able instance usable with
`async with await NeuromorphicMemory.from_openai() as mem:`.

### Core operations

| Method | Returns | Behaviour |
|---|---|---|
| `async process_turn(user_msg, conversation_history)` | reply + extraction result | Full pipeline in one LLM call: recall relevant memory, generate the reply grounded in it, extract this turn's facts, write cold archive, suppress overrides, upsert hot records, decay, prune, consolidate on schedule. |
| `async remember(messages, *, agent_id=None, conversation_id=None, occurred_at=None)` | `RememberResult` | Store what a transcript said without generating a reply. Same ingest path as `process_turn`; an unusable transcript yields an empty result rather than raising. |
| `async recall(query, *, agent_id=None, conversation_id=None, top_k=None, turn=None)` | `RecallResult` | Read-only retrieval: bounded context string + scored hits. Never mutates the store, never emits events. |
| `async forget_entity(entity_name)` | — | Remove an entity and its records from hot storage, log the removal. |
| `async forget_record(record_id)` | — | Remove one record by id. |
| `async ingest_updates(updates)` | — | Inject facts without an LLM call (e.g. importing from another system). |
| `async consolidate()` | `int` | Run the consolidation ("REM sleep") pass now instead of on schedule. Returns records consolidated. |
| `async rollback(to_turn)` | `int` | Rebuild hot state as of an earlier turn from the append-only cold archive. Returns events replayed. |

All core methods accept `agent_id` / `conversation_id` overrides for
multi-tenant use (default: the instance config's values).

### Introspection & lifecycle

| Member | Returns |
|---|---|
| `current_turn` (property) | Current turn counter |
| `get_hot_stats()` | dict: record count, average weight, per-type breakdown |
| `get_cold_stats()` | dict: archive event counts |
| `get_event_stats()` | dict: cognitive event counts by type |
| `get_events(**filters)` | list: query the event log |
| `get_event_timeline(limit=100)` | list: aggregated counts per turn |
| `get_entity_events(entity_name, limit=50)` | list: events for one entity |
| `maintain()` | bool: run pending decay/prune housekeeping |
| `close()` | Close storage connections (called by `async with` on exit) |

## `SyncNeuromorphicMemory`

Blocking wrapper over the same engine — identical constructor classmethods
plus `from_providers(llm_provider, embedding_provider, config=None)` for
injected provider objects. Methods mirror the async class without `await`,
named `process_turn_sync`, `ingest_updates_sync`, `consolidate_sync`,
`rollback_sync`; `recall`, `remember`, `forget_entity`, `forget_record` are
sync directly. Supports `with` blocks (`__enter__`/`__exit__` call `close()`).

## `NMafcConfig`

Pydantic model, the single top-level configuration object.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `llm_provider_model` | `str` | `"openai/gpt-4o-mini"` | `"provider/model"` route selector |
| `embedding_provider_model` | `str` | `"openai/text-embedding-3-small"` | Same, for embeddings |
| `agent_id` | `str` | `"default"` | Tenant namespace |
| `conversation_id` | `str` | `"default"` | Session namespace |
| `storage` | `StorageConfig` | resolved defaults | Where data lives — see [storage](storage.md) |
| `decay` | `DecayConfig` | frozen defaults | Cognitive constants — do not tune (see README) |

Also accepted: `embedding_dim`, `embedding_model`, `time_unit`, plus any
`StorageConfig`/`DecayConfig` field flattened through the sub-objects.
Environment overrides apply per field (see
[configuration](configuration.md#environment-variables)).

## `RecallResult`

Returned by `recall`.

| Field | Type | Meaning |
|---|---|---|
| `query` | `str` | The query as given |
| `context` | `str` | Formatted memory context, bounded by `top_k` — paste into a prompt |
| `hits` | `list[MemoryHit]` | Raw scored records behind the context |
| `token_estimate` | `int` | Approximate token count of `context` (chars / 4) |
| `turn` | `int` | Turn the recall was executed as of |
| `agent_id`, `conversation_id` | `str` | Tenant scope of the recall |

## `RememberResult`

Returned by `remember`.

| Field | Type | Meaning |
|---|---|---|
| `turn` | `int` | Turn the transcript was stored as of |
| `updates_ingested` | `int` | Facts written to hot storage |
| `overrides_suppressed` | `int` | Older records displaced by this ingest |
| `consolidated` | `bool` | Whether this call also triggered the scheduled consolidation pass |
| `agent_id`, `conversation_id` | `str` | Tenant scope |

## Exceptions

All inherit from `NmafcError` (which inherits `Exception`). Full catalogue
with triggers and fixes: [errors.md](errors.md).

| Exception | Raised when |
|---|---|
| `NmafcError` | Base — catch this to handle any NMAFC failure |
| `MissingAPIKeyError` | Selected provider has no API key; message names the env var |
| `MissingExtraError` | Selected provider's SDK is not installed; message names the `pip install nmafc[extra]` command |
| `ProviderNotConfiguredError` | Provider needs configuration beyond a key (Azure endpoint, unknown prefix); message names what to set |
| `StorageError` | Storage backend failed to open or operate; message names the credential chain or path to check |

## `__version__`

`str`, mirrors the installed release (and `pyproject.toml`; a test keeps them
in lockstep).
