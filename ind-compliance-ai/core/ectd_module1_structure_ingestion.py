from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import fitz


ECTD_MODULE1_STRUCTURE_BUNDLE_VERSION = "ectd-module1-structure-bundle-v1"
ECTD_MODULE1_STRUCTURE_BUNDLE_ID = "cn_ectd_attachment_1_4"
_HEADER_PATTERNS = (
    re.compile(r"^CTD 模块一文件组织结构V1\.0$"),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^CTD 模块一文件组织结构$"),
)
_RECORD_START_RE = re.compile(r"^(?P<ordinal>\d+)\.\s*(?:编号)?\s*$")


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_section_no(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def _merge_wrapped_text(parts: list[str]) -> str:
    return "".join(_normalize_text(part) for part in parts if _normalize_text(part)).strip()


def _iter_pdf_lines(path: Path) -> list[dict[str, Any]]:
    doc = fitz.open(path)
    lines: list[dict[str, Any]] = []
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            for raw_line in page.get_text("text").splitlines():
                text = _normalize_text(raw_line)
                if not text:
                    continue
                if any(pattern.match(text) for pattern in _HEADER_PATTERNS):
                    continue
                lines.append({"page_no": page_index + 1, "text": text})
    finally:
        doc.close()
    return lines


def _finalize_record(record: dict[str, Any] | None, records: list[dict[str, Any]]) -> None:
    if not record:
        return
    description = _merge_wrapped_text(record.pop("description_lines", []))
    record["description"] = description
    record["section_no"] = _normalize_section_no(record.get("raw_section_no") or "")
    records.append(record)


def build_ectd_module1_structure_bundle(source_path: Path) -> dict[str, Any]:
    source_path = Path(source_path)
    lines = _iter_pdf_lines(source_path)
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    state = "idle"

    for item in lines:
        page_no = int(item["page_no"])
        text = str(item["text"])
        start_match = _RECORD_START_RE.match(text)
        if start_match:
            _finalize_record(current, records)
            current = {
                "ordinal": int(start_match.group("ordinal")),
                "raw_section_no": "",
                "title": "",
                "element": "",
                "entry_kind": "",
                "path": "",
                "description_lines": [],
                "page_span": [page_no, page_no],
            }
            state = "await_field_or_section"
            continue

        if current is None:
            continue

        current["page_span"][1] = page_no

        if text == "编号":
            state = "await_section_or_title"
            continue
        if text == "标题":
            state = "title"
            continue
        if text == "元素":
            state = "element"
            continue
        if text in {"文件", "目录"}:
            current["entry_kind"] = text
            state = "path"
            continue
        if text == "说明":
            state = "description"
            continue

        if state in {"await_field_or_section", "await_section_or_title"}:
            if re.match(r"^\d+(?:\s*\.\s*\d+)*$", text):
                current["raw_section_no"] = text
                state = "await_next_label"
                continue
            if current.get("raw_section_no"):
                state = "await_next_label"
            else:
                state = "title"

        if state == "title":
            current["title"] = _merge_wrapped_text([current["title"], text]) if current["title"] else text
            continue
        if state == "element":
            current["element"] = text
            state = "await_next_label"
            continue
        if state == "path":
            current["path"] = text
            state = "await_next_label"
            continue
        if state == "description":
            current["description_lines"].append(text)
            continue

    _finalize_record(current, records)

    entry_kind_counts: dict[str, int] = {}
    cross_page_record_count = 0
    for record in records:
        kind = str(record.get("entry_kind") or "").strip()
        if kind:
            entry_kind_counts[kind] = entry_kind_counts.get(kind, 0) + 1
        page_span = list(record.get("page_span") or [None, None])
        if len(page_span) == 2 and page_span[0] != page_span[1]:
            cross_page_record_count += 1

    return {
        "schema_version": ECTD_MODULE1_STRUCTURE_BUNDLE_VERSION,
        "bundle_id": ECTD_MODULE1_STRUCTURE_BUNDLE_ID,
        "source_path": str(source_path),
        "page_count": max((item["page_no"] for item in lines), default=0),
        "record_count": len(records),
        "entry_kind_counts": entry_kind_counts,
        "cross_page_record_count": cross_page_record_count,
        "records": records,
    }


def write_ectd_module1_structure_bundle(
    source_path: Path,
    *,
    output_root: Path,
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_ectd_module1_structure_bundle(source_path)
    output_path = output_root / f"{ECTD_MODULE1_STRUCTURE_BUNDLE_ID}.module1_structure_bundle.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
