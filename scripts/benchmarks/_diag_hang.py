"""Throwaway diagnostic: reproduce the memory-arm construction hang.

Hypothesis: NeuromorphicMemory.__init__ auto-detects the embedding dimension by
running `embedding_provider.embed_single("test")` via `asyncio.run` inside a
ThreadPoolExecutor when a loop is already running. If that provider's underlying
async HTTP client has ALREADY been used on the main loop, its connection pool is
bound to that loop and the call from the temp loop never completes. `.result()`
has no timeout, so it hangs forever and the `except Exception` cannot see it.

Run one case per process so a hang in one does not mask the other:
    python -u -m scripts.benchmarks._diag_hang fresh   -> expected to succeed
    python -u -m scripts.benchmarks._diag_hang warmed  -> expected to hang

faulthandler dumps every thread's stack at the timeout, which is the trace we
would otherwise need py-spy for.
"""

from __future__ import annotations

import asyncio
import faulthandler
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dotenv import load_dotenv

load_dotenv()

from nmafc.integration.factory import create_embedding_provider, create_llm_provider
from nmafc.storage.config import NMafcConfig
from nmafc.wrapper import NeuromorphicMemory

PROVIDER = os.environ["NMAFC_BENCH_PROVIDER"]
EMBEDDING = os.environ["NMAFC_BENCH_EMBEDDING"]
TIMEOUT = float(os.environ.get("NMAFC_DIAG_TIMEOUT", "45"))


def make_config() -> NMafcConfig:
    tmp = tempfile.mkdtemp(prefix="nmafc_diag_")
    config = NMafcConfig.from_env_or_toml("configs/default.toml")
    config.storage.hot_uri = os.path.join(tmp, "hot_lancedb")
    config.storage.cold_uri = os.path.join(tmp, "cold.db")
    return config


async def main(case: str) -> None:
    llm = create_llm_provider(PROVIDER)
    embedder = create_embedding_provider(EMBEDDING)

    if case == "warmed":
        vec = await embedder.embed_single("warm this client on the main loop")
        print(f"pre-warmed on main loop, dim={len(vec)}", flush=True)

    print(f"[{case}] constructing NeuromorphicMemory on a running loop...", flush=True)
    faulthandler.dump_traceback_later(TIMEOUT, exit=True)
    memory = NeuromorphicMemory(
        llm_provider=llm, embedding_provider=embedder, config=make_config()
    )
    faulthandler.cancel_dump_traceback_later()
    print(f"[{case}] constructed OK, embedding_dim={memory._config.storage.embedding_dim}", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "fresh"))
