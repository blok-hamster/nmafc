# Quickstart walkthrough

The [README](../README.md) gets you running in two minutes. This page is the
first hour: the three constructors, the read/write API, and how to keep it
healthy.

## Three ways to construct

All constructors are classmethods on `NeuromorphicMemory` (async) and
`SyncNeuromorphicMemory` (sync wrapper — same methods, no `await`).

**1. `from_openai()` — zero config, needs only a key.**

```python
from nmafc import SyncNeuromorphicMemory

mem = SyncNeuromorphicMemory.from_openai()
# gpt-4o-mini for extraction, text-embedding-3-small for embeddings.
# No OPENAI_API_KEY in the environment? You get MissingAPIKeyError,
# and the message names that env var.
```

**2. `from_env()` — environment variables decide everything.**

```python
mem = SyncNeuromorphicMemory.from_env()
# Reads NMAFC_LLM_PROVIDER_MODEL / NMAFC_EMBEDDING_PROVIDER_MODEL,
# falls back to library defaults for anything unset. Optionally pass
# config_path="my.toml" to load TOML values *under* the env overrides.
```

**3. `from_config()` — explicit configuration in code or TOML.**

```python
from nmafc import NMafcConfig, SyncNeuromorphicMemory

config = NMafcConfig(llm_provider_model="groq/llama-3.1-70b-versatile",
                     embedding_provider_model="openai/nomic-embed-text")
mem = SyncNeuromorphicMemory.from_config(config=config)
```

Precedence everywhere: **explicit argument > environment variable > TOML file
> library default**. See [configuration](configuration.md).

If a provider SDK is missing you get `MissingExtraError` with the exact
`pip install nmafc[extra]` command. If a key is missing you get
`MissingAPIKeyError` naming the right env var for the provider you picked
(`GROQ_API_KEY`, `ANTHROPIC_API_KEY`, ...). Full list:
[errors](errors.md).

## The two verbs

NMAFC separates **remember** (write) from **recall** (read). In a chat
application, `process_turn` does both plus the reply in one LLM call; if you
own the conversation loop, use the verbs directly:

```python
stored = mem.remember(
    [{"role": "user", "content": "I'm allergic to shellfish."}]
)
print(stored.updates_ingested)   # facts written to hot storage
print(stored.turn)               # turn number they were written as of

hits = mem.recall("what can't I eat?")
print(hits.context)              # bounded, formatted context — paste into any prompt
print(hits.token_estimate)       # size of that context, approximately
for hit in hits.hits:            # raw scored records if you want them
    print(hit.entity_name, hit.score)
```

- `remember` runs extraction, logs to the cold archive, suppresses overridden
  facts, writes hot records, applies decay and prunes — but generates no reply
  text.
- `recall` is strictly read-only: it never mutates the store, emits no events,
  and changes nothing downstream.

## The full turn (reply + memory in one call)

```python
reply = await mem.process_turn(          # sync: mem.process_turn_sync(...)
    user_msg="What should I avoid eating?",
    conversation_history=[
        {"role": "user", "content": "What should I avoid eating?"}
    ],
)
# reply contains the answer, grounded in recalled memory; the same LLM call
# extracted and stored the turn's facts.
```

## Context managers

```python
from nmafc import SyncNeuromorphicMemory

with SyncNeuromorphicMemory.from_openai() as mem:
    mem.remember([{"role": "user", "content": "hi"}])
# close() called for you

# async equivalent:
# async with NeuromorphicMemory.from_openai() as mem: ...
```

## Multi-tenant: agent & conversation

Every record is scoped by `agent_id` and `conversation_id` (defaults:
`"default"`). Pass them per call, or set `NMAFC_AGENT_ID` /
`NMAFC_CONVERSATION_ID` in the environment:

```python
mem.remember(messages, agent_id="support", conversation_id="ticket-42")
mem.recall("refund policy?", agent_id="support", conversation_id="ticket-42")
```

A conversation never sees another's memories — same store, same keys. Details
in [storage](storage.md#multi-tenancy).

## Maintenance

```python
mem.consolidate_sync()        # manual "REM sleep" pass (runs on schedule anyway)
mem.rollback_sync(to_turn=5)  # rebuild state as of an earlier turn from the archive
mem.forget_entity("user_allergy")   # drop one entity everywhere
mem.forget_record(record_id)        # drop one record
```

`consolidate` promotes frequently-reinforced facts and prunes decayed ones;
`rollback` is possible because the cold archive is append-only — nothing is
ever overwritten in it.

## Next

- [providers](providers.md) — every supported LLM/embedding route
- [configuration](configuration.md) — every knob and env var
- [storage](storage.md) — where data lives, S3/Postgres remote backends
- [API reference](api.md) — every public symbol
- [errors](errors.md) — what each exception means and how to fix it
