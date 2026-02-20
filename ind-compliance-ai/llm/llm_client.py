from dataclasses import dataclass


@dataclass(slots=True)
class LLMConfig:
    provider: str
    base_url: str
    api_key: str


class LLMClient:
    """Thin wrapper for private/local LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def generate(self, prompt: str) -> str:
        # Placeholder deterministic response for Phase 1.
        return f"[provider={self._config.provider}] {prompt[:200]}"
