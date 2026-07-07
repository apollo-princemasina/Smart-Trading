"""
MarketIntelligenceProvider — abstract base class for all LLM providers.

All providers must implement this interface. The rest of the MIA stack
depends only on this interface, never on a concrete provider implementation.
This ensures zero coupling to Groq — any LLM can be plugged in by implementing
this interface.

Supported providers (current and future):
  GroqProvider        — Llama 3.x via Groq (current)
  OpenAIProvider      — GPT-4o / o1 (future)
  ClaudeProvider      — Claude Opus / Sonnet (future)
  GeminiProvider      — Gemini 1.5 Pro (future)
  LocalLLMProvider    — Ollama / vLLM / LM Studio (future)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProviderResponse:
    """Raw response from any LLM provider."""
    content:     str       # Raw text (expected to be valid JSON)
    model:       str       # Model name used
    tokens_in:   int       # Prompt tokens consumed
    tokens_out:  int       # Completion tokens consumed
    latency_ms:  float     # End-to-end latency in milliseconds
    request_id:  Optional[str] = None


@dataclass(frozen=True)
class ProviderHealth:
    """Health snapshot of a provider."""
    provider_name:    str
    model:            str
    status:           str      # "ok" | "degraded" | "down" | "unconfigured"
    avg_latency_ms:   Optional[float]
    error_rate:       float
    is_configured:    bool     # API key present and non-empty
    last_error:       Optional[str] = None


class MarketIntelligenceProvider(ABC):
    """
    Abstract interface for all MIA LLM providers.

    Every method must be async. Implementations must handle all provider-specific
    errors internally and raise only `ProviderError` for the gateway to catch.
    """

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        model:         str,
        temperature:   float,
        max_tokens:    int,
    ) -> ProviderResponse:
        """
        Execute a chat completion and return a structured response.

        Raises:
            ProviderError: on any provider-level failure (network, auth, rate limit)
        """

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Return a health snapshot of this provider."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'groq', 'openai')."""

    @property
    @abstractmethod
    def provider_version(self) -> str:
        """Provider version string (e.g. 'groq_v1')."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model name for this provider."""

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True if the provider has all required credentials."""


class ProviderError(Exception):
    """Raised by any provider implementation on a non-retryable failure."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderRateLimitError(ProviderError):
    """Rate limit hit — caller should back off."""

    def __init__(self, message: str, retry_after_s: Optional[float] = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after_s = retry_after_s


class ProviderAuthError(ProviderError):
    """Authentication failed — misconfigured API key."""
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=401)
