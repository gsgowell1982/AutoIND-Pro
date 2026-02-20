from pathlib import Path
from typing import Any


def parse_docx(path: Path) -> dict[str, Any]:
    """Parse DOCX structure and body content."""
    return {
        "source_path": str(path),
        "source_type": "docx",
        "headings": [],
        "paragraphs": [],
        "metadata": {},
    }
