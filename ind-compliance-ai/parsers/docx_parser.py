from pathlib import Path
from typing import Any

try:
    from docx import Document
except ImportError:  # pragma: no cover - optional runtime dependency
    Document = None  # type: ignore[assignment]

from parsers.common.atomic_fact_extractor import extract_atomic_facts


def parse_docx(path: Path) -> dict[str, Any]:
    """Parse DOCX or fallback Word content into normalized structure."""
    paragraphs: list[dict[str, Any]] = []
    extracted_text: list[str] = []

    if path.suffix.lower() == ".docx":
        if Document is None:
            raise RuntimeError("python-docx is required for .docx parsing. Install with: pip install python-docx")
        document = Document(path)
        for index, paragraph in enumerate(document.paragraphs):
            text = paragraph.text.strip()
            if not text:
                continue
            paragraphs.append({"index": index, "text": text})
            extracted_text.append(text)
    else:
        decoded_text = path.read_bytes().decode("utf-8", errors="ignore")
        normalized_lines = [line.strip() for line in decoded_text.splitlines() if line.strip()]
        for index, line in enumerate(normalized_lines):
            paragraphs.append({"index": index, "text": line})
        extracted_text.extend(normalized_lines)

    merged_text = "\n".join(extracted_text)

    return {
        "source_path": str(path),
        "source_type": "word",
        "headings": [],
        "paragraphs": paragraphs,
        "text": merged_text,
        "atomic_facts": extract_atomic_facts(merged_text),
        "metadata": {"paragraph_count": len(paragraphs)},
    }
