from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

try:
    from docx import Document
except ImportError:  # pragma: no cover - optional runtime dependency
    Document = None  # type: ignore[assignment]

from parsers.common.atomic_fact_extractor import extract_atomic_facts

logger = logging.getLogger("ind_compliance.parsers.word")


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _to_paragraphs(text: str) -> list[dict[str, Any]]:
    paragraphs: list[dict[str, Any]] = []
    for index, raw_line in enumerate(text.splitlines()):
        line = " ".join(raw_line.replace("\x00", "").split())
        if line:
            paragraphs.append({"index": index, "text": line})
    return paragraphs


def _read_docx_page_count(path: Path) -> int | None:
    try:
        with ZipFile(path) as archive:
            xml_content = archive.read("docProps/app.xml").decode("utf-8", errors="ignore")
    except Exception:
        return None
    match = re.search(r"<Pages>(\d+)</Pages>", xml_content)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_docx(path: Path) -> tuple[list[dict[str, Any]], str, int | None]:
    if Document is None:
        raise RuntimeError("python-docx is required for .docx parsing. Install with: pip install python-docx")
    document = Document(path)
    paragraphs: list[dict[str, Any]] = []
    extracted_text: list[str] = []
    for index, paragraph in enumerate(document.paragraphs):
        text = " ".join(paragraph.text.split())
        if not text:
            continue
        paragraphs.append({"index": index, "text": text})
        extracted_text.append(text)
    return paragraphs, "\n".join(extracted_text), _read_docx_page_count(path)


def _score_text_quality(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    replacement = text.count("\ufffd")
    readable = sum(1 for ch in text if ch.isprintable())
    informative = sum(
        1
        for ch in text
        if ch.isalnum() or _is_cjk(ch) or ch in "，。；：、（）《》“”‘’！？,.:;!?()[]{}+-/_"
    )
    return (readable / total) * 0.45 + (informative / total) * 0.55 - (replacement / total) * 1.5


def _extract_plain_text_candidate(path: Path) -> str | None:
    raw_bytes = path.read_bytes()
    best_text = ""
    best_score = 0.0
    for encoding in ("utf-8", "gb18030", "utf-16le", "utf-16be"):
        decoded = raw_bytes.decode(encoding, errors="ignore")
        normalized_lines = [line for line in (" ".join(item.split()) for item in decoded.splitlines()) if line]
        candidate = "\n".join(normalized_lines)
        score = _score_text_quality(candidate)
        if score > best_score:
            best_score = score
            best_text = candidate
    if best_score < 0.55:
        return None
    return best_text


def _extract_doc_with_soffice(path: Path) -> tuple[list[dict[str, Any]], str, int | None] | None:
    soffice_binary = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice_binary is None or Document is None:
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        command = [
            soffice_binary,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            str(output_dir),
            str(path),
        ]
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except Exception:
            return None
        converted_path = output_dir / f"{path.stem}.docx"
        if process.returncode != 0 or not converted_path.exists():
            return None
        return _extract_docx(converted_path)


def _extract_doc_with_antiword(path: Path) -> str | None:
    antiword_binary = shutil.which("antiword") or shutil.which("catdoc")
    if antiword_binary is None:
        return None
    command = [antiword_binary, str(path)]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=120,
            check=False,
        )
    except Exception:
        return None
    if process.returncode != 0:
        return None
    text = "\n".join(line for line in (" ".join(item.split()) for item in process.stdout.splitlines()) if line)
    if _score_text_quality(text) < 0.55:
        return None
    return text


def _extract_doc_with_windows_word(path: Path) -> str | None:
    if os.name != "nt":
        return None
    powershell_binary = shutil.which("powershell") or shutil.which("pwsh")
    if powershell_binary is None:
        return None
    script = (
        "$ErrorActionPreference = 'Stop';"
        "$inputPath = $args[0];"
        "$word = New-Object -ComObject Word.Application;"
        "$word.Visible = $false;"
        "$doc = $word.Documents.Open($inputPath, $false, $true);"
        "try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $doc.Content.Text }"
        "finally { $doc.Close(); $word.Quit(); }"
    )
    try:
        process = subprocess.run(
            [powershell_binary, "-NoProfile", "-Command", script, str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=120,
            check=False,
        )
    except Exception:
        return None
    if process.returncode != 0:
        return None
    text = "\n".join(line for line in (" ".join(item.split()) for item in process.stdout.splitlines()) if line)
    if _score_text_quality(text) < 0.55:
        return None
    return text


def _extract_doc(path: Path) -> tuple[list[dict[str, Any]], str, int | None, str]:
    text_from_word = _extract_doc_with_windows_word(path)
    if text_from_word:
        return _to_paragraphs(text_from_word), text_from_word, None, "doc-via-windows-word"

    converted_result = _extract_doc_with_soffice(path)
    if converted_result:
        paragraphs, merged_text, page_count = converted_result
        return paragraphs, merged_text, page_count, "doc-via-soffice"

    text_from_antiword = _extract_doc_with_antiword(path)
    if text_from_antiword:
        return _to_paragraphs(text_from_antiword), text_from_antiword, None, "doc-via-antiword"

    plain_text = _extract_plain_text_candidate(path)
    if plain_text:
        return _to_paragraphs(plain_text), plain_text, None, "doc-plain-text"

    raise RuntimeError(
        "Unable to reliably parse legacy .doc file. "
        "Please convert it to .docx, or install Microsoft Word/LibreOffice/antiword."
    )


def parse_docx(path: Path) -> dict[str, Any]:
    """Parse DOCX and legacy DOC into normalized Word payload."""
    suffix = path.suffix.lower()
    parser_hint = "unknown"
    if suffix == ".docx":
        paragraphs, merged_text, page_count = _extract_docx(path)
        parser_hint = "docx-native"
    elif suffix == ".doc":
        paragraphs, merged_text, page_count, parser_hint = _extract_doc(path)
    else:
        raise RuntimeError(f"Unsupported Word file extension: {suffix}")

    if not merged_text.strip():
        raise RuntimeError("No readable text extracted from Word file")

    logger.info("Parsed word file %s with parser=%s", path.name, parser_hint)
    return {
        "source_path": str(path),
        "source_type": "word",
        "headings": [],
        "paragraphs": paragraphs,
        "text": merged_text,
        "atomic_facts": extract_atomic_facts(merged_text),
        "metadata": {
            "paragraph_count": len(paragraphs),
            "page_count": page_count,
            "parser_hint": parser_hint,
        },
    }
