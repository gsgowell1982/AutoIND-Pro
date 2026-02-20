from pathlib import Path
from typing import Any


def parse_pdf(path: Path) -> dict[str, Any]:
    """Parse PDF content into normalized material representation."""
    return {
        "source_path": str(path),
        "source_type": "pdf",
        "sections": [],
        "tables": [],
        "metadata": {},
    }
