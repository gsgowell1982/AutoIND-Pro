from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_NORMALIZED_ROOT = _PROJECT_ROOT / "data" / "regulations" / "normalized"
_REGULATION_DISPLAY_NAMES = {
    "cn_ectd_technical_specification": "eCTD技术规范.pdf",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=1)
def _requirement_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in sorted(_NORMALIZED_ROOT.glob("*.requirement_matrix.json")):
        for item in (_read_json(path).get("requirements", []) or []):
            requirement_id = str(item.get("requirement_id") or "").strip()
            if requirement_id:
                lookup[requirement_id] = dict(item)
    return lookup


@lru_cache(maxsize=1)
def _clause_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in sorted(_NORMALIZED_ROOT.glob("*.clauses.json")):
        for item in (_read_json(path).get("clauses", []) or []):
            locator = dict(item.get("source_locator", {}) or {})
            anchor = str(locator.get("citation_anchor") or "").strip()
            clause_id = str(item.get("clause_id") or "").strip()
            if anchor:
                lookup[anchor] = dict(item)
            if clause_id:
                lookup[clause_id] = dict(item)
    return lookup


def _section_from_clause(clause: dict[str, Any], citation_anchor: str) -> str:
    for value in (
        clause.get("article_no_raw"),
        clause.get("article_no"),
        clause.get("source_heading"),
        clause.get("heading"),
    ):
        match = re.search(r"\b(\d+(?:\.\d+)+)\b", str(value or ""))
        if match:
            return match.group(1)
    match = re.search(r"sec_(\d+(?:_\d+)+)", str(clause.get("clause_id") or citation_anchor))
    return match.group(1).replace("_", ".") if match else ""


def _resolve_pdf_path(source_path: str, regulation_id: str) -> Path | None:
    candidate = _PROJECT_ROOT / str(source_path or "").replace("/", "\\")
    if candidate.exists():
        return candidate
    if regulation_id == "cn_ectd_technical_specification":
        candidates = sorted((_PROJECT_ROOT / "data" / "regulations").glob("*.pdf"), key=lambda p: p.stat().st_size)
        for item in candidates:
            if item.stat().st_size > 1_500_000:
                return item
    return None


@lru_cache(maxsize=128)
def _find_pdf_page(source_path: str, regulation_id: str, section: str) -> int | None:
    if not section:
        return None
    path = _resolve_pdf_path(source_path, regulation_id)
    if path is None:
        return None
    try:
        import fitz  # type: ignore

        document = fitz.open(path)
        pages = [index + 1 for index, page in enumerate(document) if re.search(rf"(?<!\d){re.escape(section)}(?:\s|[\u3000:：])", page.get_text())]
        return max(pages) if pages else None
    except Exception:
        return None


def build_requirement_provenance(
    requirement: dict[str, Any] | None = None,
    *,
    fallback_requirement_id: str = "",
    fallback_citation_anchor: str = "",
) -> dict[str, Any]:
    supplied = dict(requirement or {})
    requirement_id = str(supplied.get("requirement_id") or fallback_requirement_id).strip()
    record = dict(_requirement_lookup().get(requirement_id) or {})
    if record:
        merged = dict(record)
        merged.update({key: value for key, value in supplied.items() if value not in (None, "")})
    else:
        merged = supplied
    citation_anchor = str(merged.get("citation_anchor") or fallback_citation_anchor).strip()
    clause_id = str(merged.get("source_clause_id") or "").strip()
    clause = dict(_clause_lookup().get(citation_anchor) or _clause_lookup().get(clause_id) or {})
    if not clause_id:
        clause_id = str(clause.get("clause_id") or "").strip()
    regulation_id = str(merged.get("regulation_id") or clause.get("regulation_id") or "").strip()
    section = _section_from_clause(clause, citation_anchor)
    rule_description = str(merged.get("requirement_text") or "").strip()
    source_excerpt = str(clause.get("original_text") or clause.get("normalized_text") or rule_description).strip()
    if len(source_excerpt) > 1600:
        source_excerpt = source_excerpt[:1597].rstrip() + "..."
    source_path = str(merged.get("source_path") or clause.get("source_path") or "").strip()
    source_filename = _REGULATION_DISPLAY_NAMES.get(
        regulation_id,
        str(merged.get("source_filename") or clause.get("source_filename") or "").strip(),
    )
    source_locator = dict(clause.get("source_locator", {}) or {})
    page = _find_pdf_page(source_path, regulation_id, section)
    if page is not None:
        source_locator["page"] = page
    exact = bool(record and clause and section and rule_description)
    return {
        "requirement_id": requirement_id,
        "regulation_id": regulation_id,
        "source_filename": source_filename,
        "source_path": source_path,
        "version": str(clause.get("version_label") or "").strip(),
        "clause_id": clause_id,
        "chapter": section.split(".", 1)[0] if section else "",
        "section": section,
        "chapter_title": str(clause.get("chapter_title") or "").strip(),
        "section_heading": str(clause.get("heading") or merged.get("source_heading") or "").strip(),
        "rule_description": rule_description,
        "review_focus": str(merged.get("review_focus") or "").strip(),
        "citation_anchor": citation_anchor,
        "source_locator": source_locator,
        "source_page": page,
        "source_excerpt": source_excerpt,
        "traceability_status": "exact_clause" if exact else "fallback_requirement_id_only",
    }


RULE_REQUIREMENT_IDS = {
    "HR-ECTD-001": "cn_ectd_technical_specification:req_sequence_number_progression",
    "HR-ECTD-002": "cn_ectd_technical_specification:req_application_number_format",
    "HR-ECTD-005": "cn_ectd_technical_specification:req_file_name_character_constraints",
    "HR-ECTD-015": "cn_ectd_technical_specification:req_module1_package_backbone_composition",
    "HR-ECTD-017": "cn_ectd_technical_specification:req_module1_package_backbone_composition",
    "HR-ECTD-019": "cn_ectd_technical_specification:req_content_file_format_allowed",
    "HR-ECTD-020": "cn_ectd_technical_specification:req_package_declared_file_coverage",
    "HR-ECTD-021": "cn_ectd_technical_specification:req_no_empty_directories",
    "HR-ECTD-022": "cn_ectd_technical_specification:req_no_placeholder_documents",
}


def provenance_for_rule(rule_id: str) -> dict[str, Any]:
    requirement_id = RULE_REQUIREMENT_IDS.get(str(rule_id or "").strip(), "")
    return build_requirement_provenance(
        {"requirement_id": requirement_id} if requirement_id else {},
        fallback_requirement_id=str(rule_id or "").strip(),
    )
