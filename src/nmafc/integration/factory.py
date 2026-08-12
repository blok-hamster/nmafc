"""Provider factory for creating LLM and embedding providers from config strings.

Supports any OpenAI-compatible API (Groq, OpenRouter, Together, Ollama, LM Studio, vLLM),
Azure OpenAI, AWS Bedrock, and Anthropic natively.

Usage:
    llm = create_llm_provider("groq/llama-3.1-70b-versatile")
    llm = create_llm_provider("openrouter/anthropic/claude-sonnet-4-20250514")
    llm = create_llm_provider("ollama/llama3")
    llm = create_llm_provider("openai/gpt-4o-mini")
    llm = create_llm_provider("anthropic/claude-sonnet-4-20250514")
    llm = create_llm_provider("azure/my-gpt4-deployment")
    llm = create_llm_provider("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")

    embedder = create_embedding_provider("openai/text-embedding-3-small")
    embedder = create_embedding_provider("ollama/nomic-embed-text")
    embedder = create_embedding_provider("azure/my-embedding-deployment")
    embedder = create_embedding_provider("bedrock/amazon.titan-embed-text-v2:0")
"""

from __future__ import annotations

import os
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nmafc.integration.base import EmbeddingProvider, LLMProvider

PROVIDER_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "lmstudio": os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
    "vllm": os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
}

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "bedrock": "",
    "ollama": "",
    "lmstudio": "",
    "vllm": "",
}


def _parse_provider_model(provider_model: str) -> tuple[str, str]:
    """Parse 'provider/model' string into (provider, model).

    Handles nested model names like 'openrouter/anthropic/claude-sonnet-4-20250514'.
    """
    parts = provider_model.split("/", 1)
    if len(parts) == 1:
        return "openai", parts[0]
    return parts[0].lower(), parts[1]


def _get_api_key(provider: str, api_key: str | None = None) -> str | None:
    if api_key:
        return api_key
    env_var = PROVIDER_API_KEY_ENV.get(provider, "")
    if not env_var:
        return "not-needed"
    return os.environ.get(env_var)


def create_llm_provider(
    provider_model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Create an LLM provider from a 'provider/model' string.

    Args:
        provider_model: Format "provider/model" e.g. "groq/llama-3.1-70b-versatile"
        api_key: Override API key (otherwise reads from environment)
        base_url: Override base URL (otherwise uses known provider URLs)
        **kwargs: Additional provider-specific arguments

    Returns:
        An LLMProvider instance ready for use.

    Examples:
        create_llm_provider("openai/gpt-4o-mini")
        create_llm_provider("groq/llama-3.1-70b-versatile")
        create_llm_provider("openrouter/anthropic/claude-sonnet-4-20250514")
        create_llm_provider("ollama/llama3")
        create_llm_provider("anthropic/claude-sonnet-4-20250514")
        create_llm_provider("together/meta-llama/Llama-3-70b-chat-hf")
        create_llm_provider("lmstudio/local-model")
        create_llm_provider("vllm/meta-llama/Llama-3-8b")
        create_llm_provider("azure/my-gpt4-deployment")
        create_llm_provider("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
    """
    provider, model = _parse_provider_model(provider_model)
    resolved_key = _get_api_key(provider, api_key)

    if provider == "anthropic":
        from nmafc.integration.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model, api_key=resolved_key, **kwargs)

    if provider == "azure":
        from nmafc.integration.azure_provider import AzureOpenAIProvider

        resolved_url = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_version = kwargs.pop("api_version", None) or os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-10-21"
        )
        return AzureOpenAIProvider(
            deployment=model,
            api_key=resolved_key,
            endpoint=resolved_url,
            api_version=api_version,
            **kwargs,
        )

    if provider == "bedrock":
        region = kwargs.pop("region", None) or os.environ.get(
            "AWS_REGION", "us-east-1"
        )
        bedrock_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY_BEDROCK")
        if "anthropic" in model or bedrock_api_key:
            from nmafc.integration.bedrock_provider import BedrockAnthropicProvider

            return BedrockAnthropicProvider(
                model_id=model, region=region, api_key=bedrock_api_key, **kwargs
            )

        from nmafc.integration.bedrock_provider import BedrockProvider

        return BedrockProvider(model_id=model, region=region, **kwargs)

    from nmafc.integration.openai_provider import OpenAIProvider

    resolved_url = base_url or PROVIDER_BASE_URLS.get(provider)
    return OpenAIProvider(
        model=model,
        api_key=resolved_key,
        base_url=resolved_url,
        **kwargs,
    )


def create_embedding_provider(
    provider_model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Create an embedding provider from a 'provider/model' string.

    Args:
        provider_model: Format "provider/model" e.g. "openai/text-embedding-3-small"
        api_key: Override API key
        base_url: Override base URL

    Returns:
        An EmbeddingProvider instance.

    Examples:
        create_embedding_provider("openai/text-embedding-3-small")
        create_embedding_provider("ollama/nomic-embed-text")
        create_embedding_provider("together/togethercomputer/m2-bert-80M-8k-retrieval")
        create_embedding_provider("azure/my-embedding-deployment")
        create_embedding_provider("bedrock/amazon.titan-embed-text-v2:0")
    """
    provider, model = _parse_provider_model(provider_model)
    resolved_key = _get_api_key(provider, api_key)

    if provider == "azure":
        from nmafc.integration.azure_provider import AzureOpenAIEmbedding

        resolved_url = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_version = kwargs.pop("api_version", None) or os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-10-21"
        )
        return AzureOpenAIEmbedding(
            deployment=model,
            api_key=resolved_key,
            endpoint=resolved_url,
            api_version=api_version,
            **kwargs,
        )

    if provider == "bedrock":
        from nmafc.integration.bedrock_provider import BedrockEmbedding

        region = kwargs.pop("region", None) or os.environ.get(
            "AWS_REGION", "us-east-1"
        )
        return BedrockEmbedding(model_id=model, region=region, **kwargs)

    resolved_url = base_url or PROVIDER_BASE_URLS.get(provider)

    from nmafc.integration.openai_provider import OpenAIEmbedding

    return OpenAIEmbedding(
        model=model,
        api_key=resolved_key,
        base_url=resolved_url,
        **kwargs,
    )
