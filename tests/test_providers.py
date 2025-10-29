"""Tests for LLM providers."""

from unittest.mock import Mock, patch

import pytest

from mozzarellm.providers import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    create_provider,
)


class TestProviderFactory:
    """Tests for create_provider factory function."""

    def test_create_openai_provider_gpt(self):
        """Test creating OpenAI provider for GPT models."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = create_provider(model="gpt-4o")
            assert isinstance(provider, OpenAIProvider)
            assert provider.model == "gpt-4o"

    def test_create_openai_provider_o4(self):
        """Test creating OpenAI provider for o4 models."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = create_provider(model="o4-mini")
            assert isinstance(provider, OpenAIProvider)

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = create_provider(model="claude-3-7-sonnet-20250219")
            assert isinstance(provider, AnthropicProvider)

    def test_create_gemini_provider(self):
        """Test creating Gemini provider."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = create_provider(model="gemini-2.5-pro-preview-03-25")
            assert isinstance(provider, GeminiProvider)

    def test_unknown_model_prefix(self):
        """Test that unknown model prefix raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model prefix"):
            create_provider(model="unknown-model-xyz")

    def test_custom_parameters(self):
        """Test creating provider with custom parameters."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = create_provider(model="gpt-4o", temperature=0.7, max_tokens=4000)
            assert provider.temperature == 0.7
            assert provider.max_tokens == 4000


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_initialization_with_env_key(self):
        """Test provider initialization with environment variable."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider(model="gpt-4o")
            assert provider.api_key == "test-key"

    def test_initialization_with_explicit_key(self):
        """Test provider initialization with explicit API key."""
        provider = OpenAIProvider(model="gpt-4o", api_key="explicit-key")
        assert provider.api_key == "explicit-key"

    def test_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="OPENAI_API_KEY"),
        ):
            OpenAIProvider(model="gpt-4o")

    @patch("openai.OpenAI")
    def test_query_success(self, mock_openai_class):
        """Test successful query."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock the response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(total_tokens=100)
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
        response, error = provider.query("System prompt", "User prompt")

        assert response == "Test response"
        assert error is None
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_query_with_retry(self, mock_openai_class):
        """Test query with retry logic."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Success"))]
        mock_response.usage = Mock(total_tokens=100)

        mock_client.chat.completions.create.side_effect = [Exception("API error"), mock_response]

        provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
        response, error = provider.query("System", "User", max_retries=2)

        assert response == "Success"
        assert error is None
        assert mock_client.chat.completions.create.call_count == 2


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_initialization_with_env_key(self):
        """Test provider initialization with environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider(model="claude-3-7-sonnet-20250219")
            assert provider.api_key == "test-key"

    def test_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="ANTHROPIC_API_KEY"),
        ):
            AnthropicProvider(model="claude-3-7-sonnet-20250219")

    @patch("anthropic.Anthropic")
    def test_query_success(self, mock_anthropic_class):
        """Test successful query."""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock the response
        mock_response = Mock()
        mock_content = Mock(text="Test response")
        mock_response.content = [mock_content]
        mock_response.usage = Mock(input_tokens=50, output_tokens=50)
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(model="claude-3-7-sonnet-20250219", api_key="test-key")
        response, error = provider.query("System prompt", "User prompt")

        assert response == "Test response"
        assert error is None
        mock_client.messages.create.assert_called_once()


class TestGeminiProvider:
    """Tests for GeminiProvider."""

    def test_initialization_with_env_key(self):
        """Test provider initialization with environment variable."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = GeminiProvider(model="gemini-2.5-pro-preview-03-25")
            assert provider.api_key == "test-key"

    def test_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="GOOGLE_API_KEY"),
        ):
            GeminiProvider(model="gemini-2.5-pro-preview-03-25")

    @patch("google.genai.Client")
    def test_query_success(self, mock_genai_client):
        """Test successful query."""
        # Mock the Gemini client
        mock_client = Mock()
        mock_genai_client.return_value = mock_client

        # Mock the response
        mock_response = Mock(text="Test response")
        mock_client.models.generate_content.return_value = mock_response

        provider = GeminiProvider(model="gemini-2.5-pro-preview-03-25", api_key="test-key")
        response, error = provider.query("System prompt", "User prompt")

        assert response == "Test response"
        assert error is None
        mock_client.models.generate_content.assert_called_once()


class TestProviderErrorHandling:
    """Tests for error handling across all providers."""

    @patch("openai.OpenAI")
    def test_max_retries_exceeded(self, mock_openai_class):
        """Test that max retries returns error message."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Persistent error")

        provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
        response, error = provider.query("System", "User", max_retries=2)

        assert response is None
        assert "API error after 2 attempts" in error
        assert mock_client.chat.completions.create.call_count == 2
