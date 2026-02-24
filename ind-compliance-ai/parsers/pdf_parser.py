from __future__ import annotations

from pathlib import Path
from typing import Any

from parsers.pdf.pipeline import run_pdf_extraction_pipeline
from parsers.pdf.postprocess import build_pdf_parse_result


def parse_pdf(path: Path) -> dict[str, Any]:
    """Parse PDF into text/image/table AST while preserving BBox anchors."""
    pipeline_state = run_pdf_extraction_pipeline(path)
    return build_pdf_parse_result(path, pipeline_state)

