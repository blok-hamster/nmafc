"""Tests for the package-level public API surface (LIBRARY_PLAN L0).

Pins what `import nmafc` promises: the exported names, the exception
hierarchy, and the import-cost contract (no optional SDKs loaded eagerly).
"""

from __future__ import annotations

import subprocess
import sys
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

EXPECTED_EXPORTS = {
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


def test_all_matches_expected_exports():
    assert set(nmafc.__all__) == EXPECTED_EXPORTS


def test_every_export_resolves():
    for name in nmafc.__all__:
        assert getattr(nmafc, name, None) is not None, f"nmafc.{name} is missing"


def test_exceptions_derive_from_nmafc_error():
    for exc in (MissingAPIKeyError, MissingExtraError,
                ProviderNotConfiguredError, StorageError):
        assert issubclass(exc, NmafcError)


def test_version_matches_pyproject():
    """Single source of truth: pyproject.toml version. A mismatch means a
    release forgot to bump `nmafc.__version__` (or vice versa)."""
    import tomllib

    pyproject = Path(nmafc.__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():  # installed from a wheel, no source tree
        pytest.skip("pyproject.toml not next to the package")
    with open(pyproject, "rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert nmafc.__version__ == declared


def test_import_does_not_load_optional_sdks():
    """`import nmafc` must stay cheap: no provider SDKs, no web stack.

    Runs in a subprocess so the check is independent of whatever the test
    session itself has already imported.
    """
    code = (
        "import sys, nmafc; heavy = [m for m in "
        "('openai', 'anthropic', 'fastapi', 'fastembed', 'boto3', 'psycopg2') "
        "if m in sys.modules]; assert not heavy, heavy"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_missing_key_error_is_importable_and_catchable(
    monkeypatch: pytest.MonkeyPatch,
):
    """A user following the docs can catch NMAFC errors by base class."""
    from nmafc.integration.factory import create_llm_provider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(NmafcError):
        create_llm_provider("openai/gpt-4o-mini")
