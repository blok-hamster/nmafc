"""Tests for the multi-provider factory and config wiring."""

from __future__ import annotations

import pytest

from nmafc.integration.factory import (
    _get_api_key,
    _parse_provider_model,
    create_embedding_provider,
    create_llm_provider,
)
from nmafc.storage.config import NMafcConfig


class TestParseProviderModel:
    def test_simple_provider_model(self):
        assert _parse_provider_model("openai/gpt-4o-mini") == ("openai", "gpt-4o-mini")

    def test_nested_model_name(self):
        assert _parse_provider_model("openrouter/anthropic/claude-sonnet-4-20250514") == (
            "openrouter",
            "anthropic/claude-sonnet-4-20250514",
        )

    def test_bare_model_defaults_to_openai(self):
        assert _parse_provider_model("gpt-4o") == ("openai", "gpt-4o")

    def test_provider_lowercased(self):
        assert _parse_provider_model("Groq/llama3") == ("groq", "llama3")

    def test_ollama_model(self):
        assert _parse_provider_model("ollama/llama3") == ("ollama", "llama3")

    def test_together_nested_model(self):
        assert _parse_provider_model("together/meta-llama/Llama-3-70b-chat-hf") == (
            "together",
            "meta-llama/Llama-3-70b-chat-hf",
        )

    def test_azure_deployment(self):
        assert _parse_provider_model("azure/my-gpt4-deployment") == (
            "azure",
            "my-gpt4-deployment",
        )

    def test_bedrock_model_id(self):
        assert _parse_provider_model("bedrock/anthropic.claude-3-sonnet-20240229-v1:0") == (
            "bedrock",
            "anthropic.claude-3-sonnet-20240229-v1:0",
        )


class TestGetApiKey:
    def test_explicit_key_takes_priority(self):
        assert _get_api_key("openai", "my-key") == "my-key"

    def test_local_providers_return_not_needed(self):
        assert _get_api_key("ollama") == "not-needed"
        assert _get_api_key("lmstudio") == "not-needed"
        assert _get_api_key("vllm") == "not-needed"
        assert _get_api_key("bedrock") == "not-needed"

    def test_env_key_lookup(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test123")
        assert _get_api_key("groq") == "gsk_test123"

    def test_missing_env_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _get_api_key("openai") is None


from importlib.util import find_spec

HAS_OPENAI = find_spec("openai") is not None
HAS_ANTHROPIC = find_spec("anthropic") is not None
HAS_BOTO3 = find_spec("boto3") is not None


@pytest.mark.skipif(not HAS_OPENAI, reason="openai package not installed")
class TestCreateLlmProvider:
    def test_openai_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        provider = create_llm_provider("openai/gpt-4o-mini")
        from nmafc.integration.openai_provider import OpenAIProvider

        assert isinstance(provider, OpenAIProvider)

    def test_groq_provider_uses_correct_base_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        provider = create_llm_provider("groq/llama-3.1-70b-versatile")
        from nmafc.integration.openai_provider import OpenAIProvider

        assert isinstance(provider, OpenAIProvider)
        assert provider._client.base_url.host == "api.groq.com"

    def test_ollama_provider_local(self):
        provider = create_llm_provider("ollama/llama3")
        from nmafc.integration.openai_provider import OpenAIProvider

        assert isinstance(provider, OpenAIProvider)
        assert "localhost" in str(provider._client.base_url)
        assert "11434" in str(provider._client.base_url)

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic package not installed")
    def test_anthropic_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        provider = create_llm_provider("anthropic/claude-sonnet-4-20250514")
        from nmafc.integration.anthropic_provider import AnthropicProvider

        assert isinstance(provider, AnthropicProvider)

    def test_custom_base_url_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        provider = create_llm_provider(
            "openai/gpt-4o", base_url="http://my-proxy:8080/v1"
        )
        from nmafc.integration.openai_provider import OpenAIProvider

        assert isinstance(provider, OpenAIProvider)
        assert "my-proxy" in str(provider._client.base_url)


@pytest.mark.skipif(not HAS_OPENAI, reason="openai package not installed")
class TestCreateEmbeddingProvider:
    def test_openai_embedding(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        embedder = create_embedding_provider("openai/text-embedding-3-small")
        from nmafc.integration.openai_provider import OpenAIEmbedding

        assert isinstance(embedder, OpenAIEmbedding)

    def test_ollama_embedding(self):
        embedder = create_embedding_provider("ollama/nomic-embed-text")
        from nmafc.integration.openai_provider import OpenAIEmbedding

        assert isinstance(embedder, OpenAIEmbedding)
        assert "localhost" in str(embedder._client.base_url)


@pytest.mark.skipif(not HAS_OPENAI, reason="openai package not installed")
class TestCreateAzureProvider:
    def test_azure_llm_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key-123")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my-resource.openai.azure.com")
        provider = create_llm_provider("azure/gpt4-deployment")
        from nmafc.integration.azure_provider import AzureOpenAIProvider

        assert isinstance(provider, AzureOpenAIProvider)
        assert provider._deployment == "gpt4-deployment"

    def test_azure_embedding_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key-123")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my-resource.openai.azure.com")
        embedder = create_embedding_provider("azure/embedding-deployment")
        from nmafc.integration.azure_provider import AzureOpenAIEmbedding

        assert isinstance(embedder, AzureOpenAIEmbedding)
        assert embedder._deployment == "embedding-deployment"

    def test_azure_requires_endpoint(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key-123")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="Azure endpoint required"):
            create_llm_provider("azure/gpt4-deployment")


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 package not installed")
class TestCreateBedrockProvider:
    def test_bedrock_anthropic_provider(self):
        provider = create_llm_provider("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
        from nmafc.integration.bedrock_provider import BedrockAnthropicProvider

        assert isinstance(provider, BedrockAnthropicProvider)
        assert provider._model_id == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_bedrock_embedding_provider(self):
        embedder = create_embedding_provider("bedrock/amazon.titan-embed-text-v2:0")
        from nmafc.integration.bedrock_provider import BedrockEmbedding

        assert isinstance(embedder, BedrockEmbedding)
        assert embedder._model_id == "amazon.titan-embed-text-v2:0"

    def test_bedrock_custom_region(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        provider = create_llm_provider("bedrock/anthropic.claude-3-haiku-20240307-v1:0")
        from nmafc.integration.bedrock_provider import BedrockAnthropicProvider

        assert isinstance(provider, BedrockAnthropicProvider)


class TestNMafcConfigProviderFields:
    def test_defaults(self):
        config = NMafcConfig()
        assert config.llm_provider_model == "openai/gpt-4o-mini"
        assert config.embedding_provider_model == "openai/text-embedding-3-small"

    def test_from_toml_parses_provider_model(self, tmp_path):
        toml_content = """\
[storage]
hot_uri = "./data/lancedb"
cold_uri = "./data/cold.db"

[decay]
lambda_core_anchor = 0.0
lambda_active_context = 0.05
lambda_ephemeral = 0.69
eta = 0.15
gamma = 0.1
w_prune = 0.1

[retrieval]
theta = 0.75
top_k = 10
fallback_keyword_limit = 20

[time]
unit = "turns"

[embedding]
provider_model = "ollama/nomic-embed-text"
dim = 768

[llm]
provider_model = "groq/llama-3.1-70b-versatile"
"""
        config_file = tmp_path / "test.toml"
        config_file.write_text(toml_content)
        config = NMafcConfig.from_toml(config_file)
        assert config.llm_provider_model == "groq/llama-3.1-70b-versatile"
        assert config.embedding_provider_model == "ollama/nomic-embed-text"
        assert config.storage.embedding_dim == 768

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        toml_content = """\
[llm]
provider_model = "openai/gpt-4o-mini"

[embedding]
provider_model = "openai/text-embedding-3-small"
dim = 1536
"""
        config_file = tmp_path / "test.toml"
        config_file.write_text(toml_content)
        monkeypatch.setenv("NMAFC_LLM_PROVIDER_MODEL", "groq/mixtral-8x7b")
        monkeypatch.setenv("NMAFC_EMBEDDING_PROVIDER_MODEL", "ollama/nomic-embed-text")

        config = NMafcConfig.from_env_or_toml(config_file)
        assert config.llm_provider_model == "groq/mixtral-8x7b"
        assert config.embedding_provider_model == "ollama/nomic-embed-text"
