from typing import Any


def extract_layout(raw_content: str) -> dict[str, Any]:
    """Extract generic layout blocks for downstream parsers."""
    return {
        "headings": [],
        "blocks": [raw_content] if raw_content else [],
        "tables": [],
    }
