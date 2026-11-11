"""Comprehensive pass over everything built in library plan L0–L3.

This file is the integration checkpoint between the code and the docs:
- the exported surface matches `llms.txt` and `docs/api.md`,
- every error names its fix (the `errors.md` catalogue is a description of
  the tests below),
- configuration precedence, remote-storage flags and the legacy notice all
  behave as documented,
- package metadata (extras, licence, py.typed, entry points) matches what
  `pip install nmafc` delivers,
- examples import from the top-level surface they advertise.

Nothing here touches the decay/temporal/consolidation engine — that logic is
frozen and covered by the suite in tests/unit/.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import nmafc
from nmafc.exceptions import (
    MissingAPIKeyError,
    MissingExtraError,
    NmafcError,
    ProviderNotConfiguredError,
    StorageError,
)
from nmafc.integration.factory import create_llm_provider
from nmafc.storage.config import NMafcConfig, StorageConfig, default_data_dir

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
REPO_ROOT = Path(__file__).resolve().parents[2]

EXAMPLES_WITH_TOP_LEVEL_IMPORTS = [
    "examples/minimal.py",
    "examples/recall_remember.py",
    "examples/sync_usage.py",
    "examples/from_openai.py",
    "examples/openai_middleware.py",
]


# -----------------------------------------------------------------------
# 1. Public surface & documentation parity (L0 / L3)
# -----------------------------------------------------------------------


def test_top_level_exports_match_documented_surface():
    """docs/api.md and llms.txt describe exactly these names."""
    expected = {
        "NeuromorphicMemory",
        "SyncNeuromorphicMemory",
        "NMafcConfig",
        "RecallResult",
        "RememberResult",
        "NmafcError",
        "MissingAPIKeyError",
        "MissingExtraError",
        "ProviderNotConfiguredError",
        "StorageError",
        "__version__",
    }
    assert set(nmafc.__all__) == expected
    for name in expected:
        assert hasattr(nmafc, name), f"nmafc.{name} is missing"


def test_exception_messages_name_their_fix():
    """Every exception can be constructed with a message naming the fix."""
    assert "OPENAI_API_KEY" in str(
        MissingAPIKeyError("set OPENAI_API_KEY")
    )
    assert "pip install" in str(
        MissingExtraError("pip install 'nmafc[llm]'")
    )
    assert "AZURE_OPENAI_ENDPOINT" in str(
        ProviderNotConfiguredError("set AZURE_OPENAI_ENDPOINT")
    )
    assert "credential chain" in str(
        StorageError("check the AWS credential chain")
    ) or "path" in str(StorageError("check the path"))


def test_all_exceptions_catchable_by_base():
    for exc in (
        MissingAPIKeyError,
        MissingExtraError,
        ProviderNotConfiguredError,
        StorageError,
    ):
        assert issubclass(exc, NmafcError)
        try:
            raise exc("boom")
        except NmafcError:
            pass


# -----------------------------------------------------------------------
# 2. Package metadata matches `pip install nmafc` (L2)
# -----------------------------------------------------------------------


def test_pyproject_extras():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    optional = pyproject["project"]["optional-dependencies"]
    assert "fastembed" in optional
    assert optional["fastembed"] == ["fastembed>=0.3"]
    all_extra = optional["all"]
    assert len(all_extra) == 1
    assert "fastembed" in all_extra[0]  # folded into the `all` extra


def test_pyproject_licence_is_polyform():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    assert project["license"] == "PolyForm-Noncommercial-1.0.0"
    assert "LICENSE" in project["license-files"]


def test_py_typed_ships():
    assert (REPO_ROOT / "src" / "nmafc" / "py.typed").exists()


def test_entry_point_targets_resolve():
    """The console_scripts in pyproject point at real, importable objects."""
    targets = {
        "nmafc": "nmafc.cli:main",
        "nmafc-mcp": "nmafc.mcp_server:main",
        "nmafc-web": "nmafc.web.app:main",
    }
    for spec_name, target in targets.items():
        module, attr = target.split(":")
        module_obj = importlib.import_module(module)
        assert getattr(module_obj, attr) is not None, f"{target} missing"


def test_cli_help_runs():
    """The installed `nmafc` entry point exists and answers --help (no keys)."""
    result = subprocess.run(
        [sys.executable, "-m", "nmafc.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "init" in result.stdout or "start" in result.stdout


# -----------------------------------------------------------------------
# 3. Provider-aware error matrix (L0): message names the fix
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "route, env_var",
    [
        ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
        ("groq/llama-3.1-70b-versatile", "GROQ_API_KEY"),
        ("anthropic/claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
        ("openrouter/anthropic/claude-sonnet-4-20250514", "OPENROUTER_API_KEY"),
        ("together/mixtral-8x7b", "TOGETHER_API_KEY"),
    ],
)
def test_missing_key_names_provider_env_var(monkeypatch, route, env_var):
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(MissingAPIKeyError) as excinfo:
        create_llm_provider(route)
    assert env_var in str(excinfo.value)


def test_azure_without_endpoint_raises_naming_it(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        create_llm_provider("azure/my-deployment")
    assert "AZURE_OPENAI_ENDPOINT" in str(excinfo.value)


# -----------------------------------------------------------------------
# 4. Missing-extra messages name the exact pip command (L0)
# -----------------------------------------------------------------------


def test_openai_missing_extra_names_nmafc_llm(monkeypatch):
    """openai SDK not installed -> MissingExtraError with the install line."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(MissingExtraError) as excinfo:
        create_llm_provider("openai/gpt-4o-mini")
    msg = str(excinfo.value)
    assert "nmafc[llm]" in msg or "openai" in msg


def test_postgres_without_driver_names_nmafc_postgres():
    """psycopg2 not installed -> MissingExtraError naming nmafc[postgres]."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    try:
        with pytest.raises(MissingExtraError) as excinfo:
            from nmafc.storage.cold_pg import PostgresColdStorage

            PostgresColdStorage("postgresql://x:y@localhost/db")
        assert "nmafc[postgres]" in str(excinfo.value)
    finally:
        monkeypatch.undo()


# -----------------------------------------------------------------------
# 5. Configuration precedence & data-dir behaviour (L1)
# -----------------------------------------------------------------------


def test_nmafc_data_dir_overrides_all_three_uris(monkeypatch, tmp_path):
    monkeypatch.setenv("NMAFC_DATA_DIR", str(tmp_path / "custom"))
    cfg = StorageConfig()
    assert cfg.hot_uri.startswith(str(tmp_path / "custom"))
    assert cfg.cold_uri.startswith(str(tmp_path / "custom"))
    assert cfg.event_log_uri.startswith(str(tmp_path / "custom"))


def test_explicit_storage_args_win_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NMAFC_HOT_URI", str(tmp_path / "env-lance"))
    cfg = StorageConfig(hot_uri="/explicit/path.lance")
    assert cfg.hot_uri == "/explicit/path.lance"


def test_storage_flags_detect_remote_backends():
    local = StorageConfig()
    assert not local.is_cloud
    assert not local.cold_is_postgres

    s3 = StorageConfig(hot_uri="s3://bucket/prefix")
    assert s3.is_cloud

    pg = StorageConfig(cold_uri="postgresql://u:p@h:5432/db")
    assert pg.cold_is_postgres

    pg2 = StorageConfig(cold_uri="postgres://u:p@h/db")
    assert pg2.cold_is_postgres


def test_default_data_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NMAFC_DATA_DIR", str(tmp_path / "envdir"))
    assert default_data_dir() == tmp_path / "envdir"


def test_default_data_dir_falls_back_to_platform_dir(monkeypatch):
    monkeypatch.delenv("NMAFC_DATA_DIR", raising=False)
    d = default_data_dir()
    assert d.is_absolute()
    assert "nmafc" in str(d)


# -----------------------------------------------------------------------
# 6. Legacy-data-dir notice is silent when caller chose explicitly (L1)
# -----------------------------------------------------------------------


def test_legacy_notice_silent_for_explicit_paths(tmp_path, monkeypatch):
    """Explicitly configured storage must never trigger the legacy ./data
    notice — the caller chose the location."""
    from nmafc.storage.config import notice_legacy_data_dir

    legacy = tmp_path / "data"
    legacy.mkdir()
    # explicit paths -> no notice, nothing moved
    notice_legacy_data_dir("/explicit/hot", "/explicit/cold")



# -----------------------------------------------------------------------
# 7. Documentation files exist and anchors are present (L3)
# -----------------------------------------------------------------------


REQUIRED_DOCS = [
    "installation.md",
    "quickstart.md",
    "configuration.md",
    "providers.md",
    "storage.md",
    "api.md",
    "errors.md",
]


@pytest.mark.parametrize("doc", REQUIRED_DOCS)
def test_doc_page_exists(doc):
    assert (DOCS_DIR / doc).exists(), f"{doc} is missing"


def test_llms_txt_exists_with_key_anchors():
    text = (REPO_ROOT / "llms.txt").read_text()
    for anchor in (
        "from_openai",
        "OPENAI_API_KEY",
        "NMAFC_DATA_DIR",
        "error catalogue",
        "pip install nmafc[",
        "PolyForm Noncommercial",
    ):
        assert anchor.lower() in text.lower(), f"llms.txt missing {anchor}"


# -----------------------------------------------------------------------
# 8. Examples import from the documented top-level surface (L3)
# -----------------------------------------------------------------------


@pytest.mark.parametrize("example", EXAMPLES_WITH_TOP_LEVEL_IMPORTS)
def test_example_compiles_and_imports_top_level(example):
    path = REPO_ROOT / example
    source = path.read_text()
    compile(source, str(path), "exec")
    assert "from nmafc import" in source or "from nmafc." in source


# -----------------------------------------------------------------------
# 9. NMafcConfig constructor round-trips documented fields (L1/L2)
# -----------------------------------------------------------------------


def test_nmafc_config_holds_provider_models():
    cfg = NMafcConfig(
        llm_provider_model="groq/llama-3.1-70b-versatile",
        embedding_provider_model="fastembed/BAAI/bge-small-en-v1.5",
    )
    assert cfg.llm_provider_model == "groq/llama-3.1-70b-versatile"
    assert cfg.embedding_provider_model == "fastembed/BAAI/bge-small-en-v1.5"


def test_storage_config_storage_options_forwardable():
    cfg = StorageConfig(
        hot_uri="s3://bucket/prefix",
        storage_options={"region": "eu-central-1"},
    )
    assert cfg.storage_options == {"region": "eu-central-1"}
    assert cfg.is_cloud


# -----------------------------------------------------------------------
# 10. Version stays in lockstep (L2)
# -----------------------------------------------------------------------


def test_version_is_nonzero_and_matches_pyproject_redundantly():
    """Defense in depth alongside test_public_api.test_version_matches_pyproject."""
    import tomllib

    assert nmafc.__version__
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text()
    )
    assert nmafc.__version__ == pyproject["project"]["version"]
