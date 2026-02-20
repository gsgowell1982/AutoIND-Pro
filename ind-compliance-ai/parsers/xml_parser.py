from pathlib import Path
from typing import Any


def parse_xml(path: Path) -> dict[str, Any]:
    """Parse XML/eCTD metadata into normalized nodes."""
    return {
        "source_path": str(path),
        "source_type": "xml",
        "nodes": [],
        "metadata": {},
    }
