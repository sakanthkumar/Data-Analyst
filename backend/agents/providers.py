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
import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger("DataAnalystAgent.Providers")


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM provider implementations."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str], model: str, temperature: float) -> str:
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
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
        if not self.api_key:
            logger.warning("GROQ_API_KEY is not set. Groq provider will fail on requests.")

    def _validate_payload_inputs(self, prompt: str, system_prompt: Optional[str], model: str, temperature: float) -> tuple:
        """Validates request inputs before building payload."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing or empty.")

        if not model or not isinstance(model, str) or not model.strip():
            raise ValueError("Invalid Groq model: model must be a non-empty string.")

        if prompt is None or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Invalid prompt: prompt must be a non-empty string.")

        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("Invalid system_prompt: system_prompt must be a string or None.")

        # Normalize temperature between 0.0 and 2.0
        try:
            temp_val = float(temperature)
            if temp_val < 0.0 or temp_val > 2.0:
                logger.warning(f"Temperature {temp_val} out of supported range [0.0, 2.0]. Normalizing.")
                temp_val = max(0.0, min(2.0, temp_val))
        except (ValueError, TypeError):
            logger.warning(f"Invalid temperature '{temperature}'. Defaulting to 0.1.")
            temp_val = 0.1

        return prompt.strip(), system_prompt.strip() if system_prompt else None, model.strip(), temp_val

    def generate(self, prompt: str, system_prompt: Optional[str], model: str, temperature: float) -> str:
        # Pre-validate inputs
        clean_prompt, clean_system_prompt, clean_model, clean_temp = self._validate_payload_inputs(
            prompt, system_prompt, model, temperature
        )

        messages: List[Dict[str, str]] = []
        if clean_system_prompt:
            messages.append({"role": "system", "content": clean_system_prompt})
        messages.append({"role": "user", "content": clean_prompt})

        # Validate message roles and content integrity
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Invalid message role '{role}' at index {idx}.")
            if not content or not isinstance(content, str):
                raise ValueError(f"Invalid message content at index {idx}: content must be a non-empty string.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Compatible OpenAI Chat Completions Payload
        payload = {
            "model": clean_model,
            "messages": messages,
            "temperature": clean_temp,
        }

        # Sanitized payload metadata for logging (excludes API key)
        sanitized_payload = {
            "model": clean_model,
            "messages_count": len(messages),
            "messages": messages,
            "temperature": clean_temp,
        }

        # Log outgoing request before sending
        sys_len = len(clean_system_prompt) if clean_system_prompt else 0
        logger.info(
            f"[GroqRequest] Provider=groq, Model={clean_model}, Temp={clean_temp}, "
            f"MessageCount={len(messages)}, PromptLen={len(clean_prompt)}, "
            f"SystemPromptLen={sys_len}, Timeout={self.timeout}s"
        )

        try:
            response = requests.post(self.API_URL, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices:
                raise ValueError("Groq API returned an empty choices list.")

            content = choices[0].get("message", {}).get("content", "")
            if content is None:
                content = ""
            return content.strip()

        except requests.exceptions.Timeout:
            logger.error(f"Groq request timed out after {self.timeout}s for model={clean_model}")
            raise TimeoutError(f"Groq request timed out after {self.timeout} seconds for model '{clean_model}'.")

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Groq API at {self.API_URL}: {e}")
            raise ConnectionError("Cannot reach Groq API endpoint. Check network connectivity.") from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            raw_body = e.response.text if e.response is not None else ""

            error_reason = raw_body
            try:
                if e.response is not None:
                    error_json = e.response.json()
                    error_reason = error_json.get("error", {}).get("message", raw_body)
            except Exception:
                pass

            # Detailed logging of the failure
            logger.error(
                f"Groq Request Failed [HTTP {status_code}]\n"
                f"Model: {clean_model}\n"
                f"Reason: {error_reason}\n"
                f"Response Body: {raw_body}\n"
                f"Payload: {json.dumps(sanitized_payload)}"
            )

            # Raise detailed exception containing status, model, error reason, full body, and payload
            raise RuntimeError(
                f"\nGroq Request Failed\n"
                f"HTTP Status: {status_code}\n"
                f"Model: {clean_model}\n"
                f"Reason: {error_reason}\n"
                f"Response Body: {raw_body}\n"
                f"Payload:\n{json.dumps(sanitized_payload, indent=2)}"
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error during Groq API call: {e}", exc_info=True)
            raise RuntimeError(f"Groq API call failed unexpectedly: {str(e)}") from e

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

    def generate(self, prompt: str, system_prompt: Optional[str], model: str, temperature: float) -> str:
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Invalid prompt for Ollama: prompt must be a non-empty string.")

        if not model or not isinstance(model, str) or not model.strip():
            raise ValueError("Invalid model for Ollama: model must be a non-empty string.")

        sys_len = len(system_prompt) if system_prompt else 0
        logger.info(
            f"[OllamaRequest] Provider=ollama, Model={model}, Temp={temperature}, "
            f"PromptLen={len(prompt)}, SystemPromptLen={sys_len}, Timeout={self.timeout}s"
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": float(temperature),
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
            raise TimeoutError(f"Ollama request timed out after {self.timeout} seconds for model '{model}'.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Ollama at {self.api_url}: {e}")
            raise ConnectionError(f"Cannot reach Ollama at {self.api_url}. Is 'ollama serve' running?") from e
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            body = e.response.text if e.response is not None else ""
            logger.error(f"Ollama HTTP error {status}: {body}")
            raise RuntimeError(f"Ollama API error (HTTP {status}): {body}") from e

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
