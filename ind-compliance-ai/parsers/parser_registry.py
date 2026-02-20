from pathlib import Path
from typing import Any, Callable

from parsers.docx_parser import parse_docx
from parsers.pdf_parser import parse_pdf
from parsers.pptx_parser import parse_pptx
from parsers.xml_parser import parse_xml

ParserFn = Callable[[Path], dict[str, Any]]

PARSER_REGISTRY: dict[str, ParserFn] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
    ".xml": parse_xml,
}


def parse_file(path: Path) -> dict[str, Any]:
    """Dispatch parser by file extension."""
    extension = path.suffix.lower()
    parser = PARSER_REGISTRY.get(extension)
    if parser is None:
        raise ValueError(f"Unsupported file type: {extension}")
    return parser(path)
