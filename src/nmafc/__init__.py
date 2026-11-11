"""Neuromorphic Memory Architecture for Conversational AI (NMAFC).

A biologically-inspired state-based memory wrapper for LLM agents: dual-tier
storage, Ebbinghaus decay with reinforcement, entity-graph retrieval, and a
replayable cognitive event log.

Public API
----------
Everything in `__all__` is the stable public surface. Import it from the
package root:

    from nmafc import NeuromorphicMemory

    mem = NeuromorphicMemory.from_openai()   # needs only OPENAI_API_KEY
    result = await mem.remember([{"role": "user", "content": "..."}])
    hits = await mem.recall("what is alice's job?")

Other provider routes (Anthropic, Groq, Azure, Bedrock, Ollama, ...) are
selected via `NMafcConfig` / `from_env()` — see the factory in
`nmafc.integration.factory` for the full matrix.

Everything not listed in `__all__` is internal and may change between any
two releases without notice: deep modules (`nmafc.engine.*`,
`nmafc.storage.*`, `nmafc.integration.*`), the web server, and the CLI.
"""

from nmafc.exceptions import (
    MissingAPIKeyError,
    MissingExtraError,
    NmafcError,
    ProviderNotConfiguredError,
    StorageError,
)
from nmafc.schemas.memory import RecallResult, RememberResult
from nmafc.storage.config import NMafcConfig
from nmafc.wrapper import NeuromorphicMemory, SyncNeuromorphicMemory

__version__ = "0.2.0"

__all__ = [
    # Constructors / entry points
    "NeuromorphicMemory",
    "SyncNeuromorphicMemory",
    # Configuration
    "NMafcConfig",
    # Result types
    "RecallResult",
    "RememberResult",
    # Exceptions
    "NmafcError",
    "MissingAPIKeyError",
    "MissingExtraError",
    "ProviderNotConfiguredError",
    "StorageError",
    # Metadata
    "__version__",
]
