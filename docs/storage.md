# Storage: where data lives, and remote backends

NMAFC keeps three stores. Everything defaults to one per-user directory; every
piece can be redirected.

| Store | Holds | Default location | Remote option |
|---|---|---|---|
| **Hot tier** | current mutable records, vectors (LanceDB) | `<data>/lancedb` | any S3-compatible object store |
| **Cold tier** | append-only event archive (SQLite or PostgreSQL) | `<data>/cold.db` | PostgreSQL / CockroachDB |
| **Event log** | cognitive event history for introspection (SQLite) | `<data>/events.db` | none — local by design (see limitations) |

## The data directory

`NMAFC_DATA_DIR` overrides everything at once; the default is the per-user
platform directory:

| OS | Default |
|---|---|
| macOS | `~/Library/Application Support/nmafc` |
| Linux | `~/.local/share/nmafc` (XDG data home) |
| Windows | `%APPDATA%\nmafc` |

Per-store env vars override individual stores: `NMAFC_HOT_URI`,
`NMAFC_COLD_URI`, `NMAFC_EVENT_LOG_URI` — each may be a local path or a
remote URI (`s3://…` for hot, `postgresql://…` for cold). Explicit
`StorageConfig(...)` arguments win over everything.

**Legacy data:** before 0.2.0 the defaults were the cwd-relative `./data`,
which stranded data wherever a script happened to run. If a default store now
opens while a `./data` tree exists, you get a one-time notice suggesting
`NMAFC_DATA_DIR=./data` to keep using the old location. Nothing is moved
automatically.

## Remote backends at a glance

| Component | Local default | Remote backend | Configure with | Extra |
|---|---|---|---|---|
| Hot tier | LanceDB directory | S3, R2, MinIO, Wasabi, Ceph | `NMAFC_HOT_URI=s3://bucket/prefix` | — (LanceDB handles it) |
| Cold tier | SQLite file | PostgreSQL / CockroachDB | `NMAFC_COLD_URI=postgresql://user:pass@host:5432/db` | `nmafc[postgres]` |
| Event log | SQLite file | — | — | — |

## Hot tier → S3

```bash
export NMAFC_HOT_URI=s3://your-bucket/nmafc/hot_lancedb
```

Credentials come from the standard AWS chain (`AWS_ACCESS_KEY_ID` +
`AWS_SECRET_ACCESS_KEY`, a profile, or an instance role). To pass explicit
settings instead, use `storage_options` — forwarded to `lancedb.connect`:

```python
from nmafc import NMafcConfig
from nmafc.storage.config import StorageConfig

config = NMafcConfig(storage=StorageConfig(
    hot_uri="s3://bucket/prefix",
    storage_options={"region": "eu-central-1"},   # or endpoint_url + keys
))
```

Self-hosted MinIO (or R2/Wasabi/Ceph):

```python
StorageConfig(
    hot_uri="s3://nmafc-bucket/hot_lancedb",
    storage_options={
        "endpoint_url": "http://localhost:9000",
        "allow_http": "true",
        "access_key_id": "minioadmin",
        "secret_access_key": "minioadmin",
    },
)
```

When `storage_options` is unset, the ambient AWS credential chain applies —
handy for IAM roles in production. A failed connect raises `StorageError`
naming the credential chain to check; see [errors](errors.md#storageerror).

## Cold tier → PostgreSQL

```bash
pip install "nmafc[postgres]"
export NMAFC_COLD_URI=postgresql://user:pass@host:5432/nmafc
```

The archive table (`memory_event_log`) is created on first connect. Choose
PostgreSQL when multiple processes or machines must share the same archive —
rollbacks, overrides and invalidations are visible to every tenant of the
server. Embeddings are currently not stored in the Postgres archive (the
column exists in the SQLite backend), so cross-archive semantic fallback
search is keyword-only there — a documented gap, not a bug.

## Event log — local only

`NMAFC_EVENT_LOG_URI` accepts a local SQLite path only. The event log powers
introspection (`get_event_stats`, `get_events`, the web UI timeline), not
retrieval, so it stays local even when both tiers are remote. Documented
limitation; no code path pretends otherwise.

## Multi-tenancy

Every record is scoped by `(agent_id, conversation_id)` — the default for
both is `"default"`. Queries, archives, rollbacks and searches only ever see
their own scope:

```python
mem.remember(messages, agent_id="support", conversation_id="ticket-42")
mem.recall("refund policy?", agent_id="support",
           conversation_id="ticket-999")   # sees nothing from ticket-42
```

Set `NMAFC_AGENT_ID` / `NMAFC_CONVERSATION_ID` for process-wide defaults; the
per-call arguments win. Two agents sharing a server (or an S3 bucket) never
collide as long as their IDs differ — the IDs are part of every storage key
and every archive row.

## Choosing a layout

- **Laptop/dev:** defaults (one directory, zero credentials). Point
  `NMAFC_DATA_DIR` at a project folder if you want the data visible.
- **Single server, multiple agents:** defaults + distinct `agent_id`s.
- **Multiple app instances, shared memory:** S3 hot + PostgreSQL cold; keep
  the event log local per instance.
- **Air-gapped:** local stores + `fastembed/` embeddings + `ollama/` LLM —
  no network calls at all.
