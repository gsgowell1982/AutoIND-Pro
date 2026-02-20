from typing import Any


def tag_source(payload: dict[str, Any], source_label: str) -> dict[str, Any]:
    """Attach source classification tags to parsed payload."""
    merged = dict(payload)
    merged["source_label"] = source_label
    return merged
