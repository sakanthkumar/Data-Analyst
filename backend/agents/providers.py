"""
LLM Provider Abstraction Layer.

This module defines the base interface and concrete provider implementations
for a provider-agnostic LLM integration. The application interacts ONLY with
LLMService, which delegates to the active provider (Groq or Ollama) based
on the LLM_BACKEND environment variable.

Architecture:
    LLMService
        |
        +---- GroqProvider   (Cloud - production cloud deployments)
        |
        +---- OllamaProvider (Local - privacy-first on-premise deployments)
"""

import os
import logging
import requests
from abc import ABC, abstractmethod

logger = logging.getLogger("DataAnalystAgent.Providers")


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM provider implementations."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, model: str, temperature: float) -> str:
        """Generate a completion from the LLM.

        Args:
            prompt: The user prompt text.
            system_prompt: The system instruction text.
            model: The model identifier string.
            temperature: Sampling temperature.

        Returns:
            The model's response text, stripped of whitespace.
        """
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the human-readable provider name."""
        ...


class GroqProvider(BaseLLMProvider):
    """Groq cloud LLM provider using the OpenAI-compatible REST API."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
        if not self.api_key:
            logger.warning("GROQ_API_KEY is not set. Groq provider will fail on requests.")

    def generate(self, prompt: str, system_prompt: str, model: str, temperature: float) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            response = requests.post(self.API_URL, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            logger.error(f"Groq request timed out after {self.timeout}s for model={model}")
            raise TimeoutError(f"LLM request timed out after {self.timeout} seconds.")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            body = e.response.text[:300] if e.response is not None else ""
            if status == 401:
                logger.error("Groq API authentication failed. Check GROQ_API_KEY.")
                raise ValueError("Invalid GROQ_API_KEY. Authentication failed.")
            elif status == 429:
                logger.warning("Groq rate limit exceeded.")
                raise RuntimeError("Groq rate limit exceeded. Please retry later.")
            elif status == 404:
                logger.error(f"Groq model '{model}' not found.")
                raise ValueError(f"Model '{model}' is not available on Groq.")
            else:
                logger.error(f"Groq HTTP error {status}: {body}")
                raise RuntimeError(f"Groq API error (HTTP {status}).")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Groq API.")
            raise ConnectionError("Cannot reach Groq API. Check network connectivity.")

    def get_provider_name(self) -> str:
        return "groq"


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider using the Ollama REST API."""

    def __init__(self):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.api_url = f"{base_url}/api/generate"
        self.timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
        self.num_gpu = int(os.getenv("OLLAMA_NUM_GPU", "-1"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))

        if self.num_gpu == 0:
            logger.warning("OLLAMA_NUM_GPU=0 forces CPU mode. Running on CPU fallback.")

    def generate(self, prompt: str, system_prompt: str, model: str, temperature: float) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_gpu": self.num_gpu,
                "num_predict": self.num_predict,
                "stop": ["Question:", "Question", "\nQuestion"],
            },
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s for model={model}")
            raise TimeoutError(f"LLM request timed out after {self.timeout} seconds.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to Ollama at {self.api_url}.")
            raise ConnectionError(f"Cannot reach Ollama at {self.api_url}. Is 'ollama serve' running?")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            body = e.response.text[:300] if e.response is not None else ""
            logger.error(f"Ollama HTTP error {status}: {body}")
            raise RuntimeError(f"Ollama API error (HTTP {status}).")

    def get_provider_name(self) -> str:
        return "ollama"


def create_provider(backend: str = None) -> BaseLLMProvider:
    """Factory function to instantiate the correct provider based on LLM_BACKEND.

    Args:
        backend: Override for LLM_BACKEND env var. If None, reads from environment.

    Returns:
        A concrete BaseLLMProvider instance.

    Raises:
        ValueError: If the backend value is not recognized.
    """
    backend = (backend or os.getenv("LLM_BACKEND", "groq")).lower()
    if backend == "groq":
        return GroqProvider()
    elif backend == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unsupported LLM_BACKEND: '{backend}'. Supported: 'groq', 'ollama'.")
