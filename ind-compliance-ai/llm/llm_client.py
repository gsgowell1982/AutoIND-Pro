from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib import error, request


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class LLMClientError(RuntimeError):
    """Raised when an LLM provider cannot fulfill the request."""


@dataclass(slots=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    request_timeout_s: float = 120.0
    temperature: float = 0.1


class LLMClient:
    """HTTP client for Ollama-compatible and OpenAI-compatible LLM backends."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def resolved_base_url(self) -> str:
        base_url = str(self._config.base_url or "").strip()
        if base_url:
            return base_url.rstrip("/")
        if self._provider_family == "ollama":
            return DEFAULT_OLLAMA_BASE_URL
        raise LLMClientError(
            f"Provider '{self._config.provider}' requires an explicit base_url."
        )

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        normalized_prompt = str(prompt or "").strip()
        if not normalized_prompt:
            raise LLMClientError("Prompt must not be empty.")
        if self._provider_family == "ollama":
            return self._generate_via_ollama(prompt=normalized_prompt, system_prompt=system_prompt)
        if self._provider_family == "openai_compatible":
            return self._generate_via_openai_compatible(prompt=normalized_prompt, system_prompt=system_prompt)
        raise LLMClientError(f"Unsupported provider '{self._config.provider}'.")

    @property
    def _provider_family(self) -> str:
        provider = str(self._config.provider or "").strip().lower()
        if provider in {"ollama", "ollama_cloud", "ollama_self_hosted"}:
            return "ollama"
        if provider in {"custom_cloud_llm", "openai_compatible"}:
            return "openai_compatible"
        return "unknown"

    def _generate_via_ollama(self, *, prompt: str, system_prompt: str | None) -> str:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(self._config.temperature),
            },
        }
        if system_prompt:
            payload["system"] = str(system_prompt).strip()
        response_payload = self._post_json("/api/generate", payload)
        response_text = str(response_payload.get("response") or "").strip()
        if not response_text:
            raise LLMClientError("Ollama response did not contain a non-empty 'response' field.")
        return response_text

    def _generate_via_openai_compatible(self, *, prompt: str, system_prompt: str | None) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt).strip()})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self._config.model,
            "messages": messages,
            "temperature": float(self._config.temperature),
        }
        response_payload = self._post_json("/v1/chat/completions", payload)
        choices = list(response_payload.get("choices", []) or [])
        if not choices:
            raise LLMClientError("OpenAI-compatible response did not contain choices.")
        message = dict(choices[0].get("message", {}) or {})
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict)
            )
        response_text = str(content or "").strip()
        if not response_text:
            raise LLMClientError("OpenAI-compatible response did not contain non-empty message content.")
        return response_text

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        api_key = str(self._config.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = request.Request(
            url=f"{self.resolved_base_url}{path}",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=float(self._config.request_timeout_s)) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise LLMClientError(
                f"LLM provider HTTP error {exc.code}: {detail or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise LLMClientError(f"LLM provider connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMClientError("LLM provider request timed out.") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM provider returned non-JSON response.") from exc
