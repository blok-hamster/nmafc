from __future__ import annotations

import os
import sys
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


class StorageConfig(BaseModel):
    """Configuration for the dual-track storage layer."""

    hot_uri: str = Field(
        default="./data/lancedb",
        description="LanceDB storage URI. Local path or s3:// for cloud.",
    )
    cold_uri: str = Field(
        default="./data/cold.db",
        description="SQLite database path for the Cold ROM event log.",
    )
    embedding_dim: int = Field(default=1536, gt=0)
    embedding_model: str = Field(default="text-embedding-3-small")

    @property
    def is_cloud(self) -> bool:
        return self.hot_uri.startswith("s3://")


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
        storage_raw.setdefault("embedding_model", embedding_raw.get("model", "text-embedding-3-small"))

        decay_kwargs = {**decay_raw, **retrieval_raw}

        storage = StorageConfig(**storage_raw)
        decay = DecayConfig(**decay_kwargs)

        return cls(
            storage=storage,
            decay=decay,
            time_unit=time_raw.get("unit", "turns"),
            llm_provider_model=llm_raw.get("provider_model", "openai/gpt-4o-mini"),
            embedding_provider_model=embedding_raw.get("provider_model", "openai/text-embedding-3-small"),
        )

    @classmethod
    def from_env_or_toml(cls, toml_path: str | Path = "configs/default.toml") -> NMafcConfig:
        config = cls.from_toml(toml_path) if Path(toml_path).exists() else cls()

        if hot_uri := os.environ.get("NMAFC_HOT_URI"):
            config.storage.hot_uri = hot_uri
        if cold_uri := os.environ.get("NMAFC_COLD_URI"):
            config.storage.cold_uri = cold_uri
        if llm := os.environ.get("NMAFC_LLM_PROVIDER_MODEL"):
            config.llm_provider_model = llm
        if emb := os.environ.get("NMAFC_EMBEDDING_PROVIDER_MODEL"):
            config.embedding_provider_model = emb

        return config
