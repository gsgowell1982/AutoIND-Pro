from typing import Any


def rerank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort retrieval results by score descending."""
    return sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)
