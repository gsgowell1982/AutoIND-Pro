from __future__ import annotations

from copy import deepcopy
from typing import Any

from .shared import _clean_text, _compact_text


def _annotate_content_evidence_order_from_document_ast(
    content_evidence: list[dict[str, Any]],
    document_ast_pages: list[dict[str, Any]],
) -> None:
    order_by_key: dict[tuple[str, str], int] = {}
    fallback_by_source_id: dict[str, int] = {}
    order_index = 0
    for page in document_ast_pages:
        for block in page.get("blocks", []) or []:
            if not isinstance(block, dict):
                continue
            source_type = _content_evidence_source_type_for_ast_block(block)
            source_id = _content_evidence_source_id_for_ast_block(block)
            if not source_id:
                continue
            order_by_key.setdefault((source_type, source_id), order_index)
            fallback_by_source_id.setdefault(source_id, order_index)
            block_id = str(block.get("block_id") or "").strip()
            if block_id:
                order_by_key.setdefault((source_type, block_id), order_index)
                fallback_by_source_id.setdefault(block_id, order_index)
            if str(block.get("block_type") or "").strip().lower() == "equation":
                order_by_key.setdefault(("text", source_id), order_index)
            order_index += 1

    for evidence in content_evidence:
        source_type = str(evidence.get("source_type") or "").strip()
        source_id = str(evidence.get("source_id") or "").strip()
        if not source_id:
            continue
        order = order_by_key.get((source_type, source_id))
        if order is None:
            order = fallback_by_source_id.get(source_id)
        if order is not None:
            evidence["content_order_index"] = order


def _content_evidence_source_type_for_ast_block(block: dict[str, Any]) -> str:
    block_type = str(block.get("block_type") or "").strip().lower()
    if block_type == "table":
        return "table"
    if block_type == "image":
        return "image"
    if block_type == "toc":
        return "toc"
    if block_type == "structure_template":
        return "structure_template"
    if block_type == "algorithm":
        return "algorithm"
    return "text"


def _content_evidence_source_id_for_ast_block(block: dict[str, Any]) -> str:
    block_type = str(block.get("block_type") or "").strip().lower()
    candidate_keys = [
        "table_id" if block_type == "table" else "",
        "image_id" if block_type == "image" else "",
        "toc_id" if block_type == "toc" else "",
        "structure_template_id" if block_type == "structure_template" else "",
        "algorithm_id" if block_type == "algorithm" else "",
        "equation_id" if block_type == "equation" else "",
        "block_id",
    ]
    for key in candidate_keys:
        if not key:
            continue
        value = str(block.get(key) or "").strip()
        if value:
            return value
    return ""


def _build_content_units(content_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []

    for evidence_order_index, evidence in enumerate(content_evidence):
        evidence_id = str(evidence.get("evidence_id", "")).strip()
        source_type = str(evidence.get("source_type", "unknown") or "unknown")
        source_id = str(evidence.get("source_id", "")).strip()
        page = int(evidence.get("page", 0) or 0)
        bbox = list(evidence.get("bbox", []))
        semantic_role = str(evidence.get("semantic_role", "") or "")
        segments = list(evidence.get("segments", []) or [])

        for unit_index, segment in enumerate(segments, start=1):
            unit_role = str(segment.get("role", "content") or "content")
            text = _clean_text(str(segment.get("text", "")))
            if not text:
                continue
            attributes = {
                key: value
                for key, value in segment.items()
                if key not in {"role", "text"}
            }
            units.append(
                {
                    "unit_id": f"cu_{evidence_id}_{unit_index:03d}",
                    "evidence_id": evidence_id,
                    "source_type": source_type,
                    "source_id": source_id,
                    "page": page,
                    "bbox": bbox,
                    "semantic_role": semantic_role,
                    "unit_role": unit_role,
                    "unit_index": unit_index,
                    "evidence_order_index": evidence_order_index,
                    **(
                        {"content_order_index": int(evidence.get("content_order_index", 0) or 0)}
                        if "content_order_index" in evidence
                        else {}
                    ),
                    "text": text,
                    "attributes": attributes,
                    "fact_extraction_eligible": _is_fact_extraction_unit(source_type, unit_role),
                    **(
                        {"heading_profile": str(evidence.get("heading_profile") or "").strip()}
                        if str(evidence.get("heading_profile") or "").strip()
                        else {}
                    ),
                    **(
                        {"heading_level": int(evidence.get("heading_level", 0) or 0)}
                        if int(evidence.get("heading_level", 0) or 0) > 0
                        else {}
                    ),
                    **(
                        {"toc_sequence_id": str(evidence.get("toc_sequence_id") or "").strip()}
                        if str(evidence.get("toc_sequence_id") or "").strip()
                        else {}
                    ),
                    **(
                        {"toc_entry_index": int(evidence.get("toc_entry_index", 0) or 0)}
                        if "toc_entry_index" in evidence
                        else {}
                    ),
                    **(
                        {"toc_text": str(evidence.get("toc_text") or "").strip()}
                        if str(evidence.get("toc_text") or "").strip()
                        else {}
                    ),
                    **(
                        {"toc_page_locator_value": int(evidence.get("toc_page_locator_value", 0) or 0)}
                        if "toc_page_locator_value" in evidence
                        else {}
                    ),
                    **(
                        {"section_context": deepcopy(evidence.get("section_context", {}) or {})}
                        if dict(evidence.get("section_context", {}) or {})
                        else {}
                    ),
                }
            )

    units.sort(key=_content_unit_sort_key)
    return units


def _content_unit_sort_key(unit: dict[str, Any]) -> tuple[Any, ...]:
    if "content_order_index" in unit:
        return (
            int(unit.get("page", 0) or 0),
            int(unit.get("content_order_index", 0) or 0),
            int(unit.get("unit_index", 0) or 0),
            str(unit.get("unit_id", "")),
        )
    if "evidence_order_index" in unit:
        return (
            int(unit.get("page", 0) or 0),
            int(unit.get("evidence_order_index", 0) or 0),
            int(unit.get("unit_index", 0) or 0),
            str(unit.get("unit_id", "")),
        )
    bbox = list(unit.get("bbox", []))
    y0 = float(bbox[1]) if len(bbox) >= 2 else float("inf")
    x0 = float(bbox[0]) if len(bbox) >= 1 else float("inf")
    source_type = str(unit.get("source_type", "unknown") or "unknown")
    source_priority = {
        "text": 0,
        "table": 1,
        "image": 2,
        "toc": 3,
    }.get(source_type, 9)
    return (
        int(unit.get("page", 0) or 0),
        y0,
        x0,
        source_priority,
        int(unit.get("unit_index", 0) or 0),
        str(unit.get("unit_id", "")),
    )


def _is_fact_extraction_unit(source_type: str, unit_role: str) -> bool:
    if source_type == "toc":
        return False
    if source_type == "text":
        return unit_role == "body"
    if source_type == "table":
        return unit_role in {"title", "header", "row"}
    if source_type == "image":
        return unit_role in {"caption", "embedded_text", "embedded_code", "nearby_context"}
    return False


def _count_content_evidence_types(content_evidence: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in content_evidence:
        source_type = str(item.get("source_type") or "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def _count_content_unit_types(content_units: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in content_units:
        source_type = str(item.get("source_type") or "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def _build_fact_extraction_corpus(content_units: list[dict[str, Any]]) -> tuple[str, int]:
    parts: list[str] = []
    seen_norms: set[str] = set()
    for unit in content_units:
        if not unit.get("fact_extraction_eligible"):
            continue
        text = _clean_text(str(unit.get("text") or ""))
        normalized = _compact_text(text)
        if not text or not normalized or normalized in seen_norms:
            continue
        seen_norms.add(normalized)
        parts.append(text)
    return "\n".join(parts).strip(), len(parts)
