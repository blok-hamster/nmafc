# Security Policy

## Reporting a vulnerability

**Do not open a public GitHub issue for a security vulnerability.** A public
issue gives an attacker a reproduction and a head start.

Report privately via **GitHub Security Advisories**:

> Repository → `Security` tab → `Report a vulnerability`

If that route is unavailable, contact a maintainer directly through the repository
owner listing on GitHub and ask for a private channel.

### What to include

- Affected version or commit SHA
- Reproduction steps or a proof of concept
- Impact: what an attacker gains, and what data they reach
- Whether you believe it is already public

### What to expect

| Stage | Target |
|---|---|
| Acknowledgement | 3 business days |
| Triage and severity assessment | 10 business days |
| Fix or mitigation plan | 10 business days after triage |
| Public advisory | On fix release, or 90 days after acknowledgement, whichever is first |

We will credit you in the advisory unless you prefer otherwise. We are happy to
coordinate a disclosure date around your deployment schedule.

Please do not run automated scanners against infrastructure you do not own, and
please do not access data belonging to other tenants.

---

## Scope

In scope: the `nmafc` package — `src/nmafc/**`, the web/MCP servers, the CLI, the
benchmark harness, and the container images.

Out of scope:

- Third-party agent skills and plugins, which are customer-supplied runtime
  content bind-mounted into agent containers and not distributed as part of
  NMAFC.
- Vulnerabilities in model output that the memory layer faithfully records. If an
  extractor produces a malformed or adversarial fact and the system stores and
  returns it, that is upstream; where the framework could have rejected or
  contained it, that is in scope and we want to know.
- Weaknesses in a caller's own deployment (binding to `0.0.0.0` with no
  authentication configured, for example).

---

## Security posture as published

This section is maintained deliberately and is intended to be read by evaluators
doing procurement review. We would rather state a gap than have you find it.

### Known gaps

These are **known and open**. They are tracked in
`pixel_bots/loft_mem/docs/ENTERPRISE_TIER_IMPLEMENTATION_PLAN.md`.

| Area | Status |
|---|---|
| **Authorisation on the web server** | By default the FastAPI application opens most routes. Only the `/v1/memories` router is gated. There is no key-to-tenant binding; the tenant is a caller-supplied header. **Do not expose this service to an untrusted network.** Default-open is being changed to default-closed (plan G1.2). |
| **Authentication | principal binding** | API keys are flat and unscoped. Any valid key can address any tenant (plan G1.3, G3.1). |
| **Rate limiting** | None in the serving path. Endpoints that call a paid LLM provider are unauthenticated and unbounded by default (plan G1.5). |
| **Tenant instance cache** | `_memory_cache` is unbounded and non-evicting, so a caller can exhaust memory and file descriptors by supplying novel tenant ids (plan G1.4). |
| **Encryption at rest** | None. LanceDB, SQLite, and Postgres contents are stored in the clear. |
| **Audit trail** | The event log is a *cognitive* log, not an audit trail: no actor identity, no tamper evidence, no retention policy, no export. Cold ROM is mutated in place rather than being append-only (plan G4). |
| **Erasure** | Deletion is temporal invalidation, not destruction. Cold ROM has no purge path, so GDPR Art. 17 erasure is architecturally unsupported today (plan G6). |
| **Multi-tenancy** | Storage-layer tenant scoping is implemented and consistent across all three backends. There is no authenticated principal bound to it (plan G3). |

### What is done

- **Record-id escaping in the LanceDB layer.** `HotStorage` escapes `agent_id`,
  `conversation_id`, `entity_name`, and every record-id path — including the
  batch `id IN (...)` builders — through `_sql_str()`. DataFusion exposes no
  parameter binding on this API, so escaping is the only defence. Regression
  suite: `tests/unit/test_hot_storage_injection.py`, which exercises the
  production topology (one shared table, many tenant handles) against 20
  injection payloads across read, write, delete, and batch paths.
- **Tenant scoping at the storage layer** is enforced on every read and write in
  the LanceDB, SQLite, and Postgres backends.
- **Read-only recall.** `recall()` performs no writes and applies no
  reinforcement. A retrieval cannot change what is remembered.
- **Tenant binding in the library.** A `NeuromorphicMemory` instance refuses to
  serve a tenant other than the one it was constructed for — it raises rather
  than silently widening scope.
- **Embedding/model misalignment is loud.** If a provider returns fewer vectors
  than texts, ingest fails rather than recording a partial write.
- **No third-party code is vendored.** Dependencies resolve through `uv.lock`.
- **Path containment** on the filesystem sync path validates every path segment
  and asserts prefix containment.
- **Secrets** are read from the environment only, never logged, and `.env` is
  gitignored.

---

## Deployment guidance

NMAFC is currently a research-grade library with a local-first operational
surface. If you must deploy it before the gaps above are closed:

1. **Bind to `127.0.0.1`** or place it behind an authenticating reverse proxy.
   Do not expose it directly.
2. **Set `NMAFC_API_KEYS`** to enable the only existing gate, and understand that
   it authenticates but does not authorise per tenant.
3. **Terminate TLS in front.** There is no in-process TLS.
4. **Treat the in-app memory explorer as sensitive.** It can read every memory
  record for the connected tenant, including full source turns.
5. **Do not put the agent-content API key on an internet-reachable host.** It is
  the server's own key and there is no spend cap.
6. **Prefer a local embedding provider** (`fastembed`) in isolated environments
   so that memory content does not leave the boundary.

Sovereign and air-gapped deployment artefacts are tracked as phase G8 in the
enterprise plan.