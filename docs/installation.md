# Installation

NMAFC is a Python library. Python **3.11, 3.12 or 3.13**, any OS.

```bash
pip install nmafc            # core: hot/cold storage, extraction via your LLM
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add nmafc
```

## Extras

The core install has zero required API keys and talks to no cloud service —
you choose what to connect:

| Extra | Install | What it adds |
|---|---|---|
| `llm` | `pip install "nmafc[llm]"` | The `openai` SDK — covers OpenAI, Groq, OpenRouter, Together, Ollama, LM Studio, vLLM, Azure OpenAI (all OpenAI-compatible routes) |
| `aws` | `pip install "nmafc[aws]"` | `boto3` for AWS Bedrock models |
| `anthropic` | `pip install "nmafc[anthropic]"` | The `anthropic` SDK |
| `postgres` | `pip install "nmafc[postgres]"` | `psycopg2` for the PostgreSQL cold tier |
| `fastembed` | `pip install "nmafc[fastembed]"` | Key-free local embeddings (ONNX, runs on your machine) |
| `web` | `pip install "nmafc[web]"` | FastAPI web server + browser UI (`nmafc-web`) |
| `mcp` | `pip install "nmafc[mcp]"` | MCP server (`nmafc-mcp`) for Claude/other MCP clients |
| `cli` | `pip install "nmafc[cli]"` | Rich terminal output for `nmafc init`/`chat` |
| `all` | `pip install "nmafc[all]"` | Everything above except the benchmark tooling |

Extras compose: `pip install "nmafc[llm,postgres]"`. If you call a provider
whose SDK is not installed, you get a `MissingExtraError` naming the exact
command — nothing crashes with a bare import error.

## Verify

```python
import nmafc
print(nmafc.__version__)
from nmafc import SyncNeuromorphicMemory, NeuromorphicMemory, NmafcError
```

The top-level import is cheap: importing `nmafc` does not load any provider
SDK, only the code needed to raise a good error if one is missing.

## Where your data goes

Storage defaults to a per-user directory (macOS **Application Support/nmafc**,
Linux **~/.local/share/nmafc**, Windows **%APPDATA%\\nmafc**), not your project
folder. Point `NMAFC_DATA_DIR` anywhere to override — see
[storage](storage.md).

## Upgrading

Releases follow [semantic versioning](https://semver.org/). Everything not
exported from the `nmafc` package root is internal and may change in any
release; the exported surface only changes with a deprecation notice in the
[changelog](../CHANGELOG.md).
