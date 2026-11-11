from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

from pydantic import BaseModel, Field

if sys.version_info >= (3, 12):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

from nmafc.schemas.memory import DecayConfig

_LEGACY_CWD_DATA_DIR = "data"


def default_data_dir() -> Path:
    """Resolve the default location for NMAFC's local data stores.

    Precedence: the ``NMAFC_DATA_DIR`` environment variable, else a stable
    per-user platform directory (macOS Application Support, Windows
    ``%APPDATA%``, XDG data dir elsewhere).

    The historical default was cwd-relative ``./data``, which stranded user
    memory in whatever directory a script happened to run from. Set
    ``NMAFC_DATA_DIR=./data`` to keep that layout deliberately.
    """
    override = os.environ.get("NMAFC_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / "nmafc"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "nmafc"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "nmafc"
    return Path.home() / ".local" / "share" / "nmafc"


_legacy_data_dir_notice_shown = False


def notice_legacy_data_dir(hot_uri: str, cold_uri: str) -> None:
    """Notice when a fresh default store coexists with an old cwd ./data tree.

    Called by the storage layer when a store is about to be opened. Fires at
    most once per process, and only when the URIs in play are the *library
    defaults* — an explicit ``NMAFC_DATA_DIR``, ``NMAFC_HOT_URI`` /
    ``NMAFC_COLD_URI``, TOML setting, or programmatic override means the
    caller chose the location and gets no warning.
    """
    global _legacy_data_dir_notice_shown
    if _legacy_data_dir_notice_shown:
        return
    if os.environ.get("NMAFC_DATA_DIR"):
        return  # deliberate override: the caller picked the location
    legacy = Path(_LEGACY_CWD_DATA_DIR)
    if not (legacy / "lancedb").exists() and not (legacy / "cold.db").exists():
        return  # nothing legacy to lose
    if hot_uri != str(default_data_dir() / "lancedb") or cold_uri != str(
        default_data_dir() / "cold.db"
    ):
        return  # URIs came from env/TOML/explicit config, not the defaults
    warnings.warn(
        "NMAFC found memory stores in ./data (the pre-0.2 default location), "
        f"but the default data directory is now {default_data_dir()}. "
        "Nothing was moved. To keep reading the existing stores, set "
        "NMAFC_DATA_DIR=./data (or point NMAFC_HOT_URI / NMAFC_COLD_URI at "
        "the files directly).",
        UserWarning,
        stacklevel=2,
    )
    _legacy_data_dir_notice_shown = True


class StorageConfig(BaseModel):
    """Configuration for the dual-track storage layer."""

    hot_uri: str = Field(
        default_factory=lambda: str(default_data_dir() / "lancedb"),
        description=(
            "LanceDB storage URI. Local path or s3:// for cloud. "
            "Defaults under the NMAFC_DATA_DIR-resolved data directory."
        ),
    )
    cold_uri: str = Field(
        default_factory=lambda: str(default_data_dir() / "cold.db"),
        description=(
            "Cold ROM URI. Local .db path for SQLite, or "
            "postgresql:// DSN for remote PostgreSQL."
        ),
    )
    event_log_uri: str = Field(
        default_factory=lambda: str(default_data_dir() / "events.db"),
        description=(
            "Path to the cognitive event log SQLite database for the Web UI. "
            "Local SQLite only - there is no remote event log backend."
        ),
    )
    storage_options: dict[str, str] | None = Field(
        default=None,
        description=(
            "Extra options forwarded to lancedb.connect() for remote object "
            "stores (s3:// hot URIs), e.g. region, endpoint_url, or session "
            "credentials. Ignored for local paths. When unset, LanceDB uses "
            "the ambient AWS credential chain."
        ),
    )
    embedding_dim: int = Field(
        default_factory=lambda: int(os.environ.get("NMAFC_EMBEDDING_DIM", "1536")),
        gt=0,
        description=(
            "Vector width of the hot-storage column. Defaults from "
            "NMAFC_EMBEDDING_DIM so callers that build StorageConfig directly "
            "still match the configured embedding model."
        ),
    )
    embedding_model: str = Field(default="text-embedding-3-small")
    agent_id: str = Field(
        default="default",
        description=(
            "Namespace for agent/tenant isolation. All storage queries are "
            "scoped to this ID."
        ),
    )
    conversation_id: str = Field(
        default="default",
        description=(
            "Conversation session ID. Isolates memories between separate "
            "conversation threads."
        ),
    )

    @property
    def is_cloud(self) -> bool:
        return self.hot_uri.startswith("s3://")

    @property
    def cold_is_postgres(self) -> bool:
        return self.cold_uri.startswith("postgresql://") or self.cold_uri.startswith("postgres://")


class NMafcConfig(BaseModel):
    """Top-level configuration combining storage and decay parameters."""

    storage: StorageConfig = Field(default_factory=StorageConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    time_unit: str = Field(default="turns")
    llm_provider_model: str = Field(default="openai/gpt-4o-mini")
    embedding_provider_model: str = Field(default="openai/text-embedding-3-small")

    @classmethod
    def from_toml(cls, path: str | Path) -> NMafcConfig:
        path = Path(path)
        with path.open("rb") as f:
            raw = tomllib.load(f)

        storage_raw = raw.get("storage", {})
        decay_raw = raw.get("decay", {})
        retrieval_raw = raw.get("retrieval", {})
        time_raw = raw.get("time", {})
        embedding_raw = raw.get("embedding", {})
        llm_raw = raw.get("llm", {})

        storage_raw.setdefault("embedding_dim", embedding_raw.get("dim", 1536))
        storage_raw.setdefault(
            "embedding_model", embedding_raw.get("model", "text-embedding-3-small")
        )

        decay_kwargs = {**decay_raw, **retrieval_raw}

        storage = StorageConfig(**storage_raw)
        decay = DecayConfig(**decay_kwargs)

        return cls(
            storage=storage,
            decay=decay,
            time_unit=time_raw.get("unit", "turns"),
            llm_provider_model=llm_raw.get("provider_model", "openai/gpt-4o-mini"),
            embedding_provider_model=embedding_raw.get(
                "provider_model", "openai/text-embedding-3-small"
            ),
        )

    @classmethod
    def from_env_or_toml(cls, toml_path: str | Path | None = "configs/default.toml") -> NMafcConfig:
        """Build config from a TOML file (if present) then overlay the environment.

        `toml_path=None` skips the file entirely: environment over library
        defaults only, which is what `from_env`/`from_openai` want when no
        config directory has been set up.
        """
        config = (
            cls.from_toml(toml_path)
            if toml_path is not None and Path(toml_path).exists()
            else cls()
        )

        if hot_uri := os.environ.get("NMAFC_HOT_URI"):
            config.storage.hot_uri = hot_uri
        if cold_uri := os.environ.get("NMAFC_COLD_URI"):
            config.storage.cold_uri = cold_uri
        if event_log_uri := os.environ.get("NMAFC_EVENT_LOG_URI"):
            config.storage.event_log_uri = event_log_uri
        if agent_id := os.environ.get("NMAFC_AGENT_ID"):
            config.storage.agent_id = agent_id
        if conversation_id := os.environ.get("NMAFC_CONVERSATION_ID"):
            config.storage.conversation_id = conversation_id
        if llm := os.environ.get("NMAFC_LLM_PROVIDER_MODEL"):
            config.llm_provider_model = llm
        if emb := os.environ.get("NMAFC_EMBEDDING_PROVIDER_MODEL"):
            config.embedding_provider_model = emb
        if dim := os.environ.get("NMAFC_EMBEDDING_DIM"):
            config.storage.embedding_dim = int(dim)

        return config
