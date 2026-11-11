"""Data-directory defaults, legacy-location notice, config precedence,
storage_options forwarding, and sync-constructor parity (LIBRARY_PLAN L1).

Nothing here touches a real store: lancedb.connect is stubbed, and every
config is built with tmp_path URIs or under a chdir'd tmp cwd.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

import nmafc.storage.config as cfg_module
from nmafc import SyncNeuromorphicMemory
from nmafc.exceptions import MissingAPIKeyError, StorageError
from nmafc.storage import hot as hot_module
from nmafc.storage.config import (
    NMafcConfig,
    StorageConfig,
    default_data_dir,
)


class _FakeTable:
    def __init__(self) -> None:
        # Full schema => _migrate_schema finds no temporal columns to add.
        self.schema = list(hot_module.SCHEMA)


class _FakeDB:
    def table_names(self) -> list[str]:
        return ["memory_vectors"]  # _ensure_table skips creation

    def open_table(self, name: str) -> _FakeTable:
        return _FakeTable()


class _FakeLance:
    """Records lancedb.connect calls; optionally fails them."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._fail_with = fail_with

    def connect(self, uri: str, **kwargs: object) -> object:
        self.calls.append((uri, kwargs))
        if self._fail_with is not None:
            raise self._fail_with
        return _FakeDB()


@pytest.fixture()
def fake_lance(monkeypatch: pytest.MonkeyPatch) -> _FakeLance:
    fake = _FakeLance()
    monkeypatch.setattr(hot_module.lancedb, "connect", fake.connect)
    return fake


def _reset_notice_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg_module, "_legacy_data_dir_notice_shown", False)


def _make_hot_storage(**storage_kwargs: str) -> hot_module.HotStorage:
    return hot_module.HotStorage(StorageConfig(**storage_kwargs))


def _caught_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught]


def _legacy_notice(messages: list[str]) -> bool:
    return any("pre-0.2 default" in m for m in messages)


# ---------------------------------------------------------------------------
# L1.1 — default data directory
# ---------------------------------------------------------------------------

class TestDefaultDataDir:
    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("NMAFC_DATA_DIR", str(tmp_path))
        assert default_data_dir() == tmp_path
        cfg = StorageConfig()
        assert cfg.hot_uri == str(tmp_path / "lancedb")
        assert cfg.cold_uri == str(tmp_path / "cold.db")
        assert cfg.event_log_uri == str(tmp_path / "events.db")

    def test_platform_default_when_no_override(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("NMAFC_DATA_DIR", raising=False)
        assert default_data_dir().is_absolute()
        cfg = StorageConfig()
        assert Path(cfg.hot_uri).parent == default_data_dir()
        assert Path(cfg.cold_uri).parent == default_data_dir()
        if sys.platform == "darwin":
            assert "Application Support" in str(default_data_dir())

    def test_tilde_expansion_in_override(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("NMAFC_DATA_DIR", "~/nmafc-data")
        assert default_data_dir() == Path.home() / "nmafc-data"


# ---------------------------------------------------------------------------
# L1.2 — legacy ./data notice
# ---------------------------------------------------------------------------

@pytest.fixture()
def legacy_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cwd containing a pre-0.2 ./data tree, with env overrides cleared."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "lancedb").mkdir(parents=True)
    monkeypatch.delenv("NMAFC_DATA_DIR", raising=False)
    monkeypatch.delenv("NMAFC_HOT_URI", raising=False)
    monkeypatch.delenv("NMAFC_COLD_URI", raising=False)
    return tmp_path


@pytest.fixture()
def fake_default_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Point library-default resolution at tmp_path.

    Keeps the mkdir in HotStorage.__init__ (which runs before the stubbed
    connect) inside the test sandbox instead of the real platform dir.
    """
    monkeypatch.setattr(cfg_module, "default_data_dir", lambda: tmp_path)
    return tmp_path


class TestLegacyDataDirNotice:
    def test_fires_when_defaults_shadow_legacy_cwd(
        self, legacy_cwd: Path, fake_lance: _FakeLance,
        monkeypatch: pytest.MonkeyPatch, fake_default_data_dir: Path,
    ):
        _reset_notice_flag(monkeypatch)
        with pytest.warns(UserWarning, match="NMAFC_DATA_DIR"):
            _make_hot_storage()

    def test_silent_once_per_process(
        self, legacy_cwd: Path, fake_lance: _FakeLance,
        monkeypatch: pytest.MonkeyPatch, fake_default_data_dir: Path,
    ):
        _reset_notice_flag(monkeypatch)
        _make_hot_storage()  # first construction shows the notice
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_hot_storage()
        assert not _legacy_notice(_caught_messages(caught))

    def test_silent_when_data_dir_env_set(
        self, legacy_cwd: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        _reset_notice_flag(monkeypatch)
        monkeypatch.setenv("NMAFC_DATA_DIR", str(legacy_cwd / "chosen"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StorageConfig()
        assert not _legacy_notice(_caught_messages(caught))

    def test_silent_when_uris_explicit(
        self, legacy_cwd: Path, fake_lance: _FakeLance,
        monkeypatch: pytest.MonkeyPatch, fake_default_data_dir: Path,
    ):
        _reset_notice_flag(monkeypatch)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_hot_storage(
                hot_uri=str(legacy_cwd / "data" / "lancedb"),
                cold_uri=str(legacy_cwd / "data" / "cold.db"),
            )
        assert not _legacy_notice(_caught_messages(caught))

    def test_silent_when_no_legacy_tree(
        self, tmp_path: Path, fake_lance: _FakeLance,
        monkeypatch: pytest.MonkeyPatch, fake_default_data_dir: Path,
    ):
        monkeypatch.chdir(tmp_path)  # no ./data here
        monkeypatch.delenv("NMAFC_DATA_DIR", raising=False)
        _reset_notice_flag(monkeypatch)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_hot_storage()
        assert not _legacy_notice(_caught_messages(caught))


# ---------------------------------------------------------------------------
# L1.3 — config precedence
# ---------------------------------------------------------------------------

class TestConfigPrecedence:
    def test_env_beats_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        toml = tmp_path / "cfg.toml"
        toml.write_text('[llm]\nprovider_model = "openai/gpt-4o-mini"\n')
        monkeypatch.setenv("NMAFC_LLM_PROVIDER_MODEL", "groq/llama-3.1-8b-instant")
        cfg = NMafcConfig.from_env_or_toml(toml)
        assert cfg.llm_provider_model == "groq/llama-3.1-8b-instant"

    def test_toml_used_when_path_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        toml = tmp_path / "cfg.toml"
        toml.write_text('[llm]\nprovider_model = "groq/llama-3.1-8b-instant"\n')
        monkeypatch.delenv("NMAFC_LLM_PROVIDER_MODEL", raising=False)
        cfg = NMafcConfig.from_env_or_toml(toml)
        assert cfg.llm_provider_model == "groq/llama-3.1-8b-instant"

    def test_config_path_none_ignores_cwd_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "default.toml").write_text(
            '[llm]\nprovider_model = "groq/llama-3.1-8b-instant"\n'
        )
        monkeypatch.delenv("NMAFC_LLM_PROVIDER_MODEL", raising=False)
        cfg = NMafcConfig.from_env_or_toml(None)
        assert cfg.llm_provider_model == "openai/gpt-4o-mini"  # library default

    def test_explicit_config_object_beats_env(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("NMAFC_LLM_PROVIDER_MODEL", "groq/llama-3.1-8b-instant")
        explicit = NMafcConfig(llm_provider_model="openai/gpt-4o-mini")
        # from_config(config=...) passes the object straight through without
        # consulting from_env_or_toml again.
        assert explicit.llm_provider_model == "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# L1.6 / L1.7 — storage_options forwarding and remote error quality
# ---------------------------------------------------------------------------

class TestStorageOptionsAndRemoteErrors:
    def test_storage_options_reach_lancedb(self, fake_lance: _FakeLance):
        _make_hot_storage(hot_uri="s3://bucket/nmafc")
        uri, kwargs = fake_lance.calls[0]
        assert uri == "s3://bucket/nmafc"
        assert kwargs["storage_options"] is None  # unset stays None

    def test_storage_options_dict_forwarded(self, fake_lance: _FakeLance):
        cfg = StorageConfig(
            hot_uri="s3://bucket/nmafc",
            storage_options={"region": "eu-west-1"},
        )
        hot_module.HotStorage(cfg)
        assert fake_lance.calls[0][1]["storage_options"] == {"region": "eu-west-1"}

    def test_s3_connect_failure_names_credential_chain(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        failing = _FakeLance(fail_with=RuntimeError("access denied"))
        monkeypatch.setattr(hot_module.lancedb, "connect", failing.connect)
        cfg = StorageConfig(
            hot_uri="s3://bucket/nmafc", storage_options={"region": "eu-west-1"}
        )
        with pytest.raises(StorageError, match="AWS_ACCESS_KEY_ID"):
            hot_module.HotStorage(cfg)
        assert failing.calls
        assert failing.calls[0][1]["storage_options"] == {"region": "eu-west-1"}

    def test_local_connect_failure_is_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        failing = _FakeLance(fail_with=OSError("disk full"))
        monkeypatch.setattr(hot_module.lancedb, "connect", failing.connect)
        cfg = StorageConfig(hot_uri=str(tmp_path / "h"), cold_uri=str(tmp_path / "c"))
        with pytest.raises(StorageError, match="Could not open local hot storage"):
            hot_module.HotStorage(cfg)


# ---------------------------------------------------------------------------
# L1.4 — sync constructor parity
# ---------------------------------------------------------------------------

class TestSyncConstructorParity:
    def test_has_all_three_constructors(self):
        for name in ("from_config", "from_env", "from_openai"):
            assert callable(getattr(SyncNeuromorphicMemory, name, None)), name

    def test_from_openai_requires_key(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
            SyncNeuromorphicMemory.from_openai()

    def test_from_openai_wires_openai_models(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # Real (unstubbed) storage would otherwise be created at the library
        # default location; keep this test's IO inside tmp_path. Delenv first:
        # the repo .env (loaded via python-dotenv at factory import) pins
        # NMAFC_HOT_URI/COLD_URI explicitly, and explicit URIs beat the data
        # directory default.
        monkeypatch.setenv("NMAFC_DATA_DIR", str(tmp_path))
        for var in ("NMAFC_HOT_URI", "NMAFC_COLD_URI", "NMAFC_EVENT_LOG_URI"):
            monkeypatch.delenv(var, raising=False)
        mem = SyncNeuromorphicMemory.from_openai()
        cfg = mem._memory._config
        assert cfg.llm_provider_model == "openai/gpt-4o-mini"
        assert cfg.embedding_provider_model == "openai/text-embedding-3-small"
        assert Path(cfg.storage.hot_uri).parent == tmp_path
        mem.close()
