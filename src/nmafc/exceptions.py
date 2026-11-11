"""Public exception hierarchy for NMAFC.

Everything in `__all__` is stable public API. Every error NMAFC raises on
purpose derives from `NmafcError`, so a caller can write `except NmafcError`
and be sure the message names the fix (the env var to set, the pip command to
run, or the config key that is missing). Anything outside this module that
raises is an internal bug and will surface as a standard Python exception.
"""

from __future__ import annotations

__all__ = [
    "NmafcError",
    "MissingAPIKeyError",
    "MissingExtraError",
    "ProviderNotConfiguredError",
    "StorageError",
]


class NmafcError(Exception):
    """Base class for every error NMAFC raises on purpose."""


class MissingAPIKeyError(NmafcError):
    """The selected provider requires an API key and none was found.

    The message names the provider and the exact environment variable to set
    (for example ``OPENAI_API_KEY`` or ``GROQ_API_KEY``), or the ``api_key=``
    argument that supplies it directly.
    """


class MissingExtraError(NmafcError):
    """An optional provider SDK or backend driver is not installed.

    The message contains the exact ``pip install`` command for the needed
    optional-dependency group, e.g. ``pip install 'nmafc[llm]'``.
    """


class ProviderNotConfiguredError(NmafcError):
    """The selected provider is missing required settings other than a key.

    Examples: an Azure deployment without ``AZURE_OPENAI_ENDPOINT``, or an
    OpenAI-compatible route whose base URL cannot be determined.
    """


class StorageError(NmafcError):
    """A storage backend could not be opened or written to.

    Covers local paths, S3 object stores (hot tier), and PostgreSQL (cold
    tier). The message names the backend and the credential or URI setting
    that is likely wrong.
    """
