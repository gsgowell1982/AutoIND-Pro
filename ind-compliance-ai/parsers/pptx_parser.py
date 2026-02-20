from pathlib import Path
from typing import Any


def parse_pptx(path: Path) -> dict[str, Any]:
    """Parse PPTX slides and speaker notes for supporting materials."""
    return {
        "source_path": str(path),
        "source_type": "pptx",
        "slides": [],
        "notes": [],
        "metadata": {},
    }
