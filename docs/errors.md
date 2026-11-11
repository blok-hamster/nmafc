# Error catalogue

Every error NMAFC raises on purpose derives from `NmafcError`, and every
message names the fix: the env var to set, the `pip install` command to run,
or the setting that is wrong. Catch `NmafcError` to handle all of them; catch
a specific one to handle that failure mode. Anything else that raises is an
internal bug — please [report it](https://github.com/blok-hamster/nmafc/issues).

Quick index:

| Exception | One-line trigger | Fix in the message |
|---|---|---|
| `MissingAPIKeyError` | no API key for the selected provider | the env var to export |
| `MissingExtraError` | SDK/driver not installed | the `pip install nmafc[...]` command |
| `ProviderNotConfiguredError` | provider needs a setting beyond a key | the setting to configure |
| `StorageError` | storage backend failed to open | the credential chain or path to check |

---

## `MissingAPIKeyError`

**Trigger:** constructing a provider whose API key is not in the environment
and was not passed as `api_key=`. Raised at the public boundary — `from_openai()`,
`from_env()`, `from_config()`, `from_providers()` — never silently later.

**Example:**

```
MissingAPIKeyError: provider 'groq' requires an API key. Set GROQ_API_KEY in
your environment, or pass api_key= to the provider constructor.
```

**Fix:** export the env var the message names. The var depends on the provider
you selected — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`,
`OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, or `AZURE_OPENAI_API_KEY`. Local
routes (`ollama/`, `lmstudio/`, `vllm/`) need no key.

**Note:** NMAFC checks for the key *before* importing any SDK, so a missing
key is reported even when the provider's SDK is also not yet installed. Fix
the key first, then the extra (below) if that error appears next.

## `MissingExtraError`

**Trigger:** the provider you selected needs an optional SDK or driver that is
not installed in the current environment.

**Example:**

```
MissingExtraError: the 'openai' package is required for the OpenAI-compatible
provider routes. Install it with: pip install 'nmafc[llm]'
```

**Fix:** run exactly the command in the message:

| What you were doing | Install |
|---|---|
| OpenAI, Groq, OpenRouter, Together, Azure, Ollama, LM Studio, vLLM | `pip install 'nmafc[llm]'` |
| Anthropic | `pip install 'nmafc[anthropic]'` |
| Bedrock | `pip install 'nmafc[aws]'` |
| `fastembed/` embeddings | `pip install 'nmafc[fastembed]'` |
| PostgreSQL cold tier | `pip install 'nmafc[postgres]'` |

The message adapts: guards on individual SDK imports name the matching group,
and the PostgreSQL driver guard names `nmafc[postgres]`.

## `ProviderNotConfiguredError`

**Trigger:** the provider route is selected but a required non-key setting is
absent, or the provider prefix is unknown.

**Two common cases:**

```
ProviderNotConfiguredError: provider 'azure' requires AZURE_OPENAI_ENDPOINT
(the https://... resource URL) to be set.
```

```
ProviderNotConfiguredError: unknown LLM provider 'groq-mistyped'. Valid
prefixes: openai, anthropic, groq, openrouter, together, ollama, lmstudio,
vllm, azure, azure_v1, bedrock.
```

**Fix:** for Azure and Azure-Foundry routes, set
`AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/`. For an
unknown prefix, check the spelling against the
[provider table](providers.md#llm-routes). Base-URL-driven routes
(`ollama/`, `lmstudio/`, `vllm/`) fall back to standard local ports and only
raise this if a custom URL is required and cannot be resolved.

## `StorageError`

**Trigger:** a storage backend could not be opened — local path unwritable,
S3 bucket unreachable or unauthorized, PostgreSQL server unreachable.

**S3 case:**

```
StorageError: could not open S3 hot store s3://bucket/prefix. Check the AWS
credential chain (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_PROFILE,
or an instance role) and that the bucket/region is reachable.
```

**Local case:**

```
StorageError: could not open local hot store /path/to/lancedb: check that the
directory exists or can be created, and that the process has write permission.
```

**Fix:** the message names the chain to check. For S3, either export AWS
credentials or pass `StorageConfig(storage_options={...})` with explicit
session credentials/`region`/`endpoint_url` (forwarded to `lancedb.connect`).
For PostgreSQL, verify the DSN in `NMAFC_COLD_URI` and that
`nmafc[postgres]` is installed. Full remote-backend setup:
[storage.md](storage.md).

---

## Not-an-error notes

**The legacy-data-dir notice.** If a default (not explicitly configured) store
opens while a legacy `./data` tree exists in the current directory, NMAFC
prints a one-time notice pointing at `NMAFC_DATA_DIR=./data` so your existing
data is found again. It is informational, not an exception, and nothing is
moved automatically.

**Empty results are not errors.** `remember()` with an unusable transcript
returns an empty `RememberResult`; `recall()` with no matches returns an
empty `RecallResult`. Only infrastructure failures raise.

**Env-var typos are not detected.** NMAFC only reads the variable names
documented in [configuration.md](configuration.md); `NMAFC_HOTURI` or
`OPENAI_APIKEY` are simply ignored — the first sign is the corresponding
`MissingAPIKeyError`/default-location behaviour.
