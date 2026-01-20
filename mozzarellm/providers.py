"""
Unified LLM provider system for mozzarellm.

This module provides a consistent interface for querying different LLM providers
(OpenAI, Anthropic, Google Gemini) with automatic retry logic and error handling.
"""

import logging
import os
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8000,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize LLM provider.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0, optional)
            top_k: Top-K sampling parameter (optional, Claude/Gemini only)
            stop_sequences: List of stop sequences (optional)
            api_key: API key (if None, reads from environment)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = stop_sequences
        self.api_key = api_key or self._get_api_key_from_env()
        self._validate_api_key()

    @abstractmethod
    def _get_api_key_from_env(self) -> str | None:
        """Get API key from environment variable."""
        pass

    @abstractmethod
    def _get_env_var_name(self) -> str:
        """Get name of environment variable for API key."""
        pass

    def _validate_api_key(self):
        """Validate that API key is available."""
        if not self.api_key:
            raise ValueError(f"{self._get_env_var_name()} not found in environment or constructor")

    @abstractmethod
    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make the actual API call to the provider."""
        pass

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> tuple[str | None, str | None]:
        """
        Query the LLM with retry logic.

        Args:
            system_prompt: System/context prompt
            user_prompt: User query prompt
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (response_text, error_message)
            If successful: (text, None)
            If failed: (None, error_message)
        """
        for attempt in range(max_retries):
            try:
                response = self._make_api_call(system_prompt, user_prompt)
                logger.info(
                    f"{self.__class__.__name__} call successful "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                return response, None

            except Exception as e:
                error_str = str(e)[:100]  # Truncate long errors
                error_msg = (
                    f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {error_str}"
                )
                logger.warning(f"{self.__class__.__name__}: {error_msg}")

                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return None, f"API error after {max_retries} attempts: {str(e)}"

        return None, "Maximum retries exceeded"


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, o4-mini, o3-mini, etc.)"""

    def _get_env_var_name(self) -> str:
        return "OPENAI_API_KEY"

    def _get_api_key_from_env(self) -> str | None:
        return os.environ.get(self._get_env_var_name())

    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Build kwargs with optional parameters
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "seed": 42,  # For reproducibility
        }

        # Add optional sampling parameters
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.stop_sequences:
            kwargs["stop"] = self.stop_sequences

        response = client.chat.completions.create(**kwargs)

        # Log token usage
        if hasattr(response, "usage"):
            tokens = response.usage.total_tokens
            logger.info(f"OpenAI tokens used: {tokens}")

        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic API provider (Claude models)"""

    def _get_env_var_name(self) -> str:
        return "ANTHROPIC_API_KEY"

    def _get_api_key_from_env(self) -> str | None:
        return os.environ.get(self._get_env_var_name())

    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        # Build kwargs with optional parameters
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
        }

        # Add optional sampling parameters
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences

        response = client.messages.create(**kwargs)

        # Log token usage
        if hasattr(response, "usage"):
            tokens = response.usage.input_tokens + response.usage.output_tokens
            logger.info(f"Anthropic tokens used: {tokens}")

        return response.content[0].text


class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""

    def _get_env_var_name(self) -> str:
        return "GOOGLE_API_KEY"

    def _get_api_key_from_env(self) -> str | None:
        return os.environ.get(self._get_env_var_name())

    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        # Build config with optional parameters (no hardcoded values!)
        config_kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "system_instruction": system_prompt,
        }

        # Add optional sampling parameters
        if self.top_p is not None:
            config_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            config_kwargs["top_k"] = self.top_k
        if self.stop_sequences:
            config_kwargs["stop_sequences"] = self.stop_sequences

        config = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=self.model, contents=user_prompt, config=config
        )

        return response.text


def create_provider(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 8000,
    top_p: float | None = None,
    top_k: int | None = None,
    stop_sequences: list[str] | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """
    Factory function to create the appropriate provider based on model name.

    Args:
        model: Model identifier
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter (0.0-1.0, optional)
        top_k: Top-K sampling parameter (optional, Claude/Gemini only)
        stop_sequences: List of stop sequences (optional)
        api_key: Optional API key (reads from env if None)

    Returns:
        Appropriate LLMProvider instance

    Raises:
        ValueError: If model prefix is not recognized
    """
    model_lower = model.lower()

    # OpenAI models
    if any(model_lower.startswith(prefix) for prefix in ["gpt", "o4", "o3", "o1"]):
        return OpenAIProvider(model, temperature, max_tokens, top_p, top_k, stop_sequences, api_key)

    # Anthropic models
    elif model_lower.startswith("claude"):
        return AnthropicProvider(
            model, temperature, max_tokens, top_p, top_k, stop_sequences, api_key
        )

    # Google models
    elif model_lower.startswith("gemini"):
        return GeminiProvider(model, temperature, max_tokens, top_p, top_k, stop_sequences, api_key)

    else:
        raise ValueError(
            f"Unknown model prefix: {model}. "
            "Supported prefixes: gpt*, o4*, o3*, o1*, claude*, gemini*"
        )
