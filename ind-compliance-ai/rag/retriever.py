from typing import Any


class Retriever:
    """Restricted retriever that only queries approved regulation corpora."""

    def __init__(self, allowed_sources: set[str]) -> None:
        self._allowed_sources = allowed_sources

    def retrieve(self, query: str, source: str) -> list[dict[str, Any]]:
        if source not in self._allowed_sources:
            raise PermissionError(f"Source not allowed: {source}")
        return [{"source": source, "query": query, "score": 0.0, "snippet": ""}]
