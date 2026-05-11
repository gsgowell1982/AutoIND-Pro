from __future__ import annotations

from pathlib import Path
import re
from typing import Any

try:
    import pymupdf
except ImportError:  # pragma: no cover - optional runtime dependency
    pymupdf = None  # type: ignore[assignment]

from parsers.pdf.pipeline import run_pdf_extraction_pipeline
from parsers.pdf.postprocess import build_pdf_parse_result

_PDF_FONT_FILE_MARKER_RE = re.compile(r"/(FontFile3|FontFile2|FontFile)\b")
_PDF_OBJECT_REFERENCE_RE = re.compile(r"(\d+)\s+\d+\s+R")
_PDF_ACTION_NEXT_MARKER_RE = re.compile(r"/Next\b")
_PDF_DESTINATION_ARRAY_RE = re.compile(r"/D\s*(\[[^\]]*\])")
_PDF_DIRECT_DESTINATION_ARRAY_RE = re.compile(r"/Dest\s*(\[[^\]]*\])")
_PDF_DESTINATION_PAGE_REF_RE = re.compile(r"^\[\s*(\d+)\s+(\d+)\s+R(?:\s|/|\])")
_PDF_DESTINATION_KIND_RE = re.compile(
    r"^\[\s*(?:\d+\s+\d+\s+R|\d+|/[A-Za-z0-9_.-]+)\s*(/[A-Za-z]+)(?:\s|/|\])"
)


def _extract_pdf_linearization_metadata(path: Path) -> dict[str, Any]:
    try:
        header = path.read_bytes()[:4096]
    except Exception:
        header = b""
    return {
        "pdf_is_linearized": b"/Linearized" in header,
    }


def _extract_pdf_initial_view_metadata(document: Any) -> dict[str, Any]:
    metadata = {
        "pdf_initial_view_page_mode": "",
        "pdf_initial_view_page_layout": "",
        "pdf_initial_view_open_action_present": False,
        "pdf_initial_view_open_action_kind": "",
    }
    try:
        catalog_xref = int(document.pdf_catalog() or 0)
    except Exception:
        return metadata
    if catalog_xref <= 0:
        return metadata

    try:
        page_mode_type, page_mode_value = document.xref_get_key(catalog_xref, "PageMode")
    except Exception:
        page_mode_type, page_mode_value = ("null", "null")
    try:
        page_layout_type, page_layout_value = document.xref_get_key(catalog_xref, "PageLayout")
    except Exception:
        page_layout_type, page_layout_value = ("null", "null")
    try:
        open_action_type, open_action_value = document.xref_get_key(catalog_xref, "OpenAction")
    except Exception:
        open_action_type, open_action_value = ("null", "null")

    if str(page_mode_type).strip().lower() == "name":
        metadata["pdf_initial_view_page_mode"] = str(page_mode_value or "").strip()
    if str(page_layout_type).strip().lower() == "name":
        metadata["pdf_initial_view_page_layout"] = str(page_layout_value or "").strip()
    open_action_kind = str(open_action_type or "").strip().lower()
    metadata["pdf_initial_view_open_action_present"] = open_action_kind not in {"", "null"}
    metadata["pdf_initial_view_open_action_kind"] = open_action_kind if open_action_kind not in {"", "null"} else ""
    return metadata


def _extract_pdf_font_file_kind_from_object(
    document: Any,
    xref: int,
    *,
    visited: set[int] | None = None,
    depth: int = 0,
) -> str:
    if xref <= 0 or depth > 4:
        return ""
    visited = visited or set()
    if xref in visited:
        return ""
    visited.add(xref)

    try:
        obj = str(document.xref_object(xref, compressed=False) or "")
    except Exception:
        obj = ""
    if obj:
        marker = _PDF_FONT_FILE_MARKER_RE.search(obj)
        if marker:
            return str(marker.group(1) or "").strip()

    for ref_match in _PDF_OBJECT_REFERENCE_RE.finditer(obj):
        try:
            ref_xref = int(ref_match.group(1) or 0)
        except ValueError:
            continue
        font_file_kind = _extract_pdf_font_file_kind_from_object(
            document,
            ref_xref,
            visited=visited,
            depth=depth + 1,
        )
        if font_file_kind:
            return font_file_kind

    return ""


def _extract_pdf_font_embedding_metadata(document: Any) -> dict[str, Any]:
    records_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    try:
        page_count = int(document.page_count or 0)
    except Exception:
        page_count = 0

    try:
        page_range = range(page_count)
        for page_index in page_range:
            page = document.load_page(page_index)
            for font in page.get_fonts(full=True):
                if not font:
                    continue
                xref = int(font[0] or 0)
                file_extension = str(font[1] or "").strip()
                font_type = str(font[2] or "").strip()
                base_font = str(font[3] or "").strip()
                resource_name = str(font[4] or "").strip()
                encoding = str(font[5] or "").strip()
                font_file_kind = _extract_pdf_font_file_kind_from_object(document, xref)
                is_embedded = bool(font_file_kind) or file_extension.lower() not in {"", "n/a"}
                key = (xref, base_font, resource_name)
                existing = records_by_key.get(key)
                if existing is None:
                    records_by_key[key] = {
                        "xref": xref,
                        "font_type": font_type,
                        "base_font": base_font,
                        "resource_name": resource_name,
                        "encoding": encoding,
                        "file_extension": file_extension,
                        "is_embedded": is_embedded,
                        "font_file_kind": font_file_kind,
                        "page_numbers": [page_index + 1],
                    }
                    continue
                page_numbers = list(existing.get("page_numbers", []) or [])
                if page_index + 1 not in page_numbers:
                    page_numbers.append(page_index + 1)
                existing["page_numbers"] = page_numbers
                existing["is_embedded"] = bool(existing.get("is_embedded")) or is_embedded
                if font_file_kind and not existing.get("font_file_kind"):
                    existing["font_file_kind"] = font_file_kind
    except Exception:
        return {
            "pdf_font_records": [],
            "pdf_font_count": 0,
            "pdf_font_embedding_evidence_available": False,
        }

    font_records = sorted(
        records_by_key.values(),
        key=lambda record: (
            str(record.get("base_font") or ""),
            str(record.get("resource_name") or ""),
            int(record.get("xref", 0) or 0),
        ),
    )
    return {
        "pdf_font_records": font_records,
        "pdf_font_count": len(font_records),
        "pdf_font_embedding_evidence_available": True,
    }


def _extract_pdf_disallowed_content_metadata(document: Any) -> dict[str, Any]:
    markers: set[str] = set()
    token_map = {
        "/JavaScript": "javascript",
        "/3D": "3d",
        "/ThreeD": "3d",
        "/RichMedia": "richmedia",
        "/Movie": "movie",
        "/Sound": "sound",
        "/Screen": "screen",
    }
    try:
        xref_length = int(document.xref_length() or 0)
    except Exception:
        xref_length = 0

    for xref in range(1, max(xref_length, 1)):
        try:
            obj = str(document.xref_object(xref, compressed=False) or "")
        except Exception:
            continue
        if not obj:
            continue
        for token, marker in token_map.items():
            if token in obj:
                markers.add(marker)

    return {
        "pdf_disallowed_content_markers": sorted(markers),
    }


def _extract_pdf_link_action_metadata(document: Any) -> dict[str, Any]:
    action_kinds: set[str] = set()
    broken_link_annotation_count = 0
    multiple_action_xrefs: list[int] = []
    link_destination_count = 0
    non_inherit_zoom_details: list[dict[str, Any]] = []
    try:
        xref_length = int(document.xref_length() or 0)
    except Exception:
        xref_length = 0

    for xref in range(1, max(xref_length, 1)):
        try:
            obj = str(document.xref_object(xref, compressed=False) or "")
        except Exception:
            continue
        if not obj or "/Subtype /Link" not in obj:
            continue
        if _PDF_ACTION_NEXT_MARKER_RE.search(obj):
            multiple_action_xrefs.append(xref)
        match = re.search(r"/A\s*<<[\s\S]*?/S\s*(/[A-Za-z]+)", obj)
        if match:
            action_kinds.add(str(match.group(1) or "").strip())
        action_destination_match = _PDF_DESTINATION_ARRAY_RE.search(obj)
        direct_destination_match = _PDF_DIRECT_DESTINATION_ARRAY_RE.search(obj)
        for destination_source, destination_match in (
            ("A.D", action_destination_match),
            ("Dest", direct_destination_match),
        ):
            if destination_match is None:
                continue
            destination = str(destination_match.group(1) or "").strip()
            detail = _pdf_destination_non_inherit_zoom_detail(
                xref=xref,
                destination_value=destination,
                destination_source=destination_source,
            )
            if detail is None and _pdf_destination_kind(destination):
                link_destination_count += 1
            elif detail is not None:
                link_destination_count += 1
                non_inherit_zoom_details.append(detail)
        if not match and ("/A null" in obj or ("/A" not in obj and "/Dest" not in obj)):
            broken_link_annotation_count += 1

    return {
        "pdf_link_action_kinds": sorted(kind for kind in action_kinds if kind),
        "pdf_broken_link_annotation_count": broken_link_annotation_count,
        "pdf_link_multiple_action_count": len(multiple_action_xrefs),
        "pdf_link_multiple_action_xrefs": multiple_action_xrefs,
        "pdf_link_multiple_action_evidence_available": True,
        "pdf_link_inherit_zoom_evidence_available": True,
        "pdf_link_destination_count": link_destination_count,
        "pdf_link_non_inherit_zoom_count": len(non_inherit_zoom_details),
        "pdf_link_non_inherit_zoom_xrefs": [
            int(item["xref"]) for item in non_inherit_zoom_details if int(item.get("xref", 0) or 0) > 0
        ],
        "pdf_link_non_inherit_zoom_details": non_inherit_zoom_details,
    }


def _extract_pdf_link_action_page_metadata(document: Any) -> dict[str, Any]:
    page_records: list[dict[str, Any]] = []
    uri_link_records: list[dict[str, Any]] = []
    try:
        page_count = int(document.page_count or 0)
    except Exception:
        page_count = 0

    for page_index in range(page_count):
        try:
            page = document.load_page(page_index)
            links = list(page.get_links() or [])
        except Exception:
            continue
        page_action_kinds: list[str] = []
        page_action_xrefs: list[int] = []
        for link in links:
            try:
                xref = int(link.get("xref", 0) or 0)
            except Exception:
                xref = 0
            if xref <= 0:
                continue
            kind = int(link.get("kind", 0) or 0)
            page_action_xrefs.append(xref)
            if kind == pymupdf.LINK_GOTO:
                page_action_kinds.append("/GoTo")
            elif kind == pymupdf.LINK_GOTOR:
                page_action_kinds.append("/GoToR")
            elif kind == pymupdf.LINK_LAUNCH:
                page_action_kinds.append("/Launch")
            elif kind == pymupdf.LINK_URI:
                page_action_kinds.append("/URI")
                uri = str(link.get("uri") or "").strip()
                rect = link.get("from")
                try:
                    bbox = [
                        round(float(rect.x0), 2),
                        round(float(rect.y0), 2),
                        round(float(rect.x1), 2),
                        round(float(rect.y1), 2),
                    ]
                except Exception:
                    bbox = []
                if uri and len(bbox) == 4 and any(float(value or 0.0) != 0.0 for value in bbox):
                    uri_link_records.append(
                        {
                            "page": page_index + 1,
                            "xref": xref,
                            "uri": uri,
                            "bbox": bbox,
                        }
                    )
            else:
                page_action_kinds.append(str(kind))
        if not page_action_xrefs and not page_action_kinds:
            continue
        page_records.append(
            {
                "page": page_index + 1,
                "link_annotation_count": len(page_action_xrefs),
                "link_annotation_xrefs": page_action_xrefs,
                "link_action_kinds": sorted({kind for kind in page_action_kinds if kind}),
            }
        )

    return {
        "pdf_link_action_pages": [int(record["page"]) for record in page_records if int(record.get("page", 0) or 0) > 0],
        "pdf_link_action_page_records": page_records,
        "pdf_uri_link_annotation_records": uri_link_records,
    }


def _extract_pdf_bookmark_action_metadata(document: Any) -> dict[str, Any]:
    action_kinds: set[str] = set()
    external_file_targets: list[str] = []
    uri_targets: list[str] = []
    invalid_bookmark_xrefs: list[int] = []
    damaged_bookmark_target_details: list[dict[str, Any]] = []
    bookmark_destination_count = 0
    non_inherit_zoom_details: list[dict[str, Any]] = []
    multiple_action_xrefs: list[int] = []
    try:
        toc_rows = list(document.get_toc(simple=False) or [])
        page_xrefs = {int(document.page_xref(index)) for index in range(int(document.page_count or 0))}
    except Exception:
        return {
            "pdf_bookmark_action_kinds": [],
            "pdf_bookmark_action_evidence_available": False,
            "pdf_bookmark_validity_evidence_available": False,
            "pdf_invalid_bookmark_count": 0,
            "pdf_invalid_bookmark_xrefs": [],
            "pdf_bookmark_target_integrity_evidence_available": False,
            "pdf_damaged_bookmark_target_count": 0,
            "pdf_damaged_bookmark_target_xrefs": [],
            "pdf_damaged_bookmark_target_details": [],
            "pdf_bookmark_inherit_zoom_evidence_available": False,
            "pdf_bookmark_destination_count": 0,
            "pdf_bookmark_non_inherit_zoom_count": 0,
            "pdf_bookmark_non_inherit_zoom_xrefs": [],
            "pdf_bookmark_non_inherit_zoom_details": [],
            "pdf_bookmark_external_file_targets": [],
            "pdf_bookmark_external_file_target_count": 0,
            "pdf_bookmark_external_file_target_evidence_available": False,
            "pdf_bookmark_uri_targets": [],
            "pdf_bookmark_uri_target_count": 0,
            "pdf_bookmark_uri_target_evidence_available": False,
            "pdf_bookmark_multiple_action_count": 0,
            "pdf_bookmark_multiple_action_xrefs": [],
            "pdf_bookmark_multiple_action_evidence_available": False,
        }

    for row in toc_rows:
        if len(row) < 4 or not isinstance(row[3], dict):
            continue
        try:
            outline_xref = int(row[3].get("xref") or 0)
        except (TypeError, ValueError):
            outline_xref = 0
        if outline_xref <= 0:
            continue
        try:
            action_type, action_value = document.xref_get_key(outline_xref, "A")
        except Exception:
            action_type, action_value = ("null", "null")
        try:
            dest_type, _dest_value = document.xref_get_key(outline_xref, "Dest")
        except Exception:
            dest_type, _dest_value = ("null", "null")
        if str(action_type or "").strip().lower() == "null":
            if str(dest_type or "").strip().lower() == "null":
                invalid_bookmark_xrefs.append(outline_xref)
            else:
                damage = _pdf_bookmark_destination_damage_detail(
                    outline_xref=outline_xref,
                    destination_value=str(_dest_value or ""),
                    page_xrefs=page_xrefs,
                    destination_source="Dest",
                )
                if damage is not None:
                    damaged_bookmark_target_details.append(damage)
                zoom_detail = _pdf_destination_non_inherit_zoom_detail(
                    xref=outline_xref,
                    destination_value=str(_dest_value or ""),
                    destination_source="Dest",
                )
                if zoom_detail is None and _pdf_destination_kind(str(_dest_value or "")):
                    bookmark_destination_count += 1
                elif zoom_detail is not None:
                    bookmark_destination_count += 1
                    non_inherit_zoom_details.append(zoom_detail)
            continue
        action_value_text = str(action_value or "")
        match = re.search(r"/S\s*(/[A-Za-z]+)", action_value_text)
        if match:
            action_kinds.add(str(match.group(1) or "").strip())
        elif str(dest_type or "").strip().lower() == "null":
            invalid_bookmark_xrefs.append(outline_xref)
        action_kind = str(match.group(1) or "").strip() if match else ""
        destination_match = _PDF_DESTINATION_ARRAY_RE.search(action_value_text)
        if action_kind == "/GoTo":
            damage = _pdf_bookmark_destination_damage_detail(
                outline_xref=outline_xref,
                destination_value=str(destination_match.group(1) if destination_match else ""),
                page_xrefs=page_xrefs,
                destination_source="A.D",
            )
            if damage is not None:
                damaged_bookmark_target_details.append(damage)
        if destination_match is not None:
            zoom_detail = _pdf_destination_non_inherit_zoom_detail(
                xref=outline_xref,
                destination_value=str(destination_match.group(1) or ""),
                destination_source="A.D",
            )
            if zoom_detail is None and _pdf_destination_kind(str(destination_match.group(1) or "")):
                bookmark_destination_count += 1
            elif zoom_detail is not None:
                bookmark_destination_count += 1
                non_inherit_zoom_details.append(zoom_detail)
        nested_file_targets = [
            str(match.group(1) or "").strip()
            for match in re.finditer(r"/F\s*<<[\s\S]*?/F\s*\(([^)]*)\)", action_value_text)
        ]
        action_value_without_nested_file_dicts = re.sub(r"/F\s*<<[\s\S]*?>>", "", action_value_text)
        direct_file_targets = [
            str(match.group(1) or "").strip()
            for match in re.finditer(r"/F\s*\(([^)]*)\)", action_value_without_nested_file_dicts)
        ]
        for target in [*nested_file_targets, *direct_file_targets]:
            if target:
                external_file_targets.append(target)
        for match in re.finditer(r"/URI\s*\(([^)]*)\)", action_value_text):
            target = str(match.group(1) or "").strip()
            if target:
                uri_targets.append(target)
        if _PDF_ACTION_NEXT_MARKER_RE.search(action_value_text):
            multiple_action_xrefs.append(outline_xref)

    return {
        "pdf_bookmark_action_kinds": sorted(kind for kind in action_kinds if kind),
        "pdf_bookmark_action_evidence_available": True,
        "pdf_bookmark_validity_evidence_available": True,
        "pdf_invalid_bookmark_count": len(invalid_bookmark_xrefs),
        "pdf_invalid_bookmark_xrefs": invalid_bookmark_xrefs,
        "pdf_bookmark_target_integrity_evidence_available": True,
        "pdf_damaged_bookmark_target_count": len(damaged_bookmark_target_details),
        "pdf_damaged_bookmark_target_xrefs": [
            int(item["xref"]) for item in damaged_bookmark_target_details if int(item.get("xref", 0) or 0) > 0
        ],
        "pdf_damaged_bookmark_target_details": damaged_bookmark_target_details,
        "pdf_bookmark_inherit_zoom_evidence_available": True,
        "pdf_bookmark_destination_count": bookmark_destination_count,
        "pdf_bookmark_non_inherit_zoom_count": len(non_inherit_zoom_details),
        "pdf_bookmark_non_inherit_zoom_xrefs": [
            int(item["xref"]) for item in non_inherit_zoom_details if int(item.get("xref", 0) or 0) > 0
        ],
        "pdf_bookmark_non_inherit_zoom_details": non_inherit_zoom_details,
        "pdf_bookmark_external_file_targets": external_file_targets,
        "pdf_bookmark_external_file_target_count": len(external_file_targets),
        "pdf_bookmark_external_file_target_evidence_available": True,
        "pdf_bookmark_uri_targets": uri_targets,
        "pdf_bookmark_uri_target_count": len(uri_targets),
        "pdf_bookmark_uri_target_evidence_available": True,
        "pdf_bookmark_multiple_action_count": len(multiple_action_xrefs),
        "pdf_bookmark_multiple_action_xrefs": multiple_action_xrefs,
        "pdf_bookmark_multiple_action_evidence_available": True,
    }


def _pdf_bookmark_destination_damage_detail(
    *,
    outline_xref: int,
    destination_value: str,
    page_xrefs: set[int],
    destination_source: str,
) -> dict[str, Any] | None:
    destination_text = str(destination_value or "").strip()
    if not destination_text:
        return {
            "xref": outline_xref,
            "reason": "destination_missing",
            "destination_source": destination_source,
            "destination": destination_text,
        }
    if not destination_text.startswith("["):
        return None
    if re.fullmatch(r"\[\s*\]", destination_text):
        return {
            "xref": outline_xref,
            "reason": "destination_array_empty",
            "destination_source": destination_source,
            "destination": destination_text,
        }
    page_ref_match = _PDF_DESTINATION_PAGE_REF_RE.match(destination_text)
    if not page_ref_match:
        return {
            "xref": outline_xref,
            "reason": "destination_page_reference_missing",
            "destination_source": destination_source,
            "destination": destination_text,
        }
    page_xref = int(page_ref_match.group(1))
    if page_xref not in page_xrefs:
        return {
            "xref": outline_xref,
            "reason": "destination_page_object_missing",
            "destination_source": destination_source,
            "destination": destination_text,
            "page_xref": page_xref,
        }
    return None


def _pdf_destination_kind(destination_value: str) -> str:
    match = _PDF_DESTINATION_KIND_RE.match(str(destination_value or "").strip())
    return str(match.group(1) or "").strip() if match else ""


def _pdf_destination_non_inherit_zoom_detail(
    *,
    xref: int,
    destination_value: str,
    destination_source: str,
) -> dict[str, Any] | None:
    destination_text = str(destination_value or "").strip()
    destination_kind = _pdf_destination_kind(destination_text)
    if not destination_kind:
        return None
    if destination_kind != "/XYZ":
        return {
            "xref": xref,
            "reason": "destination_not_xyz",
            "destination_source": destination_source,
            "destination": destination_text,
            "destination_kind": destination_kind,
        }
    after_kind = destination_text[destination_text.find(destination_kind) + len(destination_kind) :].strip()
    if after_kind.endswith("]"):
        after_kind = after_kind[:-1].strip()
    operands = [item for item in re.split(r"\s+", after_kind) if item]
    zoom = operands[2] if len(operands) >= 3 else ""
    if not zoom:
        return {
            "xref": xref,
            "reason": "xyz_zoom_missing",
            "destination_source": destination_source,
            "destination": destination_text,
            "destination_kind": destination_kind,
        }
    if zoom.lower() == "null":
        return None
    try:
        if float(zoom) == 0.0:
            return None
    except ValueError:
        pass
    return {
        "xref": xref,
        "reason": "xyz_zoom_not_inherit",
        "destination_source": destination_source,
        "destination": destination_text,
        "destination_kind": destination_kind,
        "zoom": zoom,
    }


def _extract_pdf_security_metadata(path: Path) -> dict[str, Any]:
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is required for .pdf parsing. Install with: pip install pymupdf")

    document = pymupdf.open(path)
    try:
        metadata = dict(document.metadata or {})
        format_label = str(metadata.get("format") or "").strip()
        encryption_scheme = str(metadata.get("encryption") or "").strip()
        needs_password = bool(document.needs_pass)
        permissions = int(document.permissions or 0)
        return {
            "page_count": int(document.page_count or 0),
            "pdf_is_readable": True,
            "pdf_readability_issue": "",
            "pdf_format_version": format_label,
            "pdf_is_encrypted": bool(document.is_encrypted),
            "pdf_needs_password": needs_password,
            "pdf_openable_without_password": not needs_password,
            "pdf_security_permissions": permissions,
            "pdf_encryption_scheme": encryption_scheme,
            "pdf_has_security_settings": bool(needs_password or encryption_scheme or permissions != -4),
            **_extract_pdf_linearization_metadata(path),
            **_extract_pdf_initial_view_metadata(document),
            **_extract_pdf_font_embedding_metadata(document),
            **_extract_pdf_disallowed_content_metadata(document),
            **_extract_pdf_link_action_metadata(document),
            **_extract_pdf_link_action_page_metadata(document),
            **_extract_pdf_bookmark_action_metadata(document),
        }
    finally:
        document.close()


def _build_unreadable_pdf_result(path: Path, readability_issue: str, error_message: str = "") -> dict[str, Any]:
    return {
        "filename": path.name,
        "source_type": "pdf",
        "source_path": str(path),
        "text": "",
        "atomic_facts": {},
        "content_units": [],
        "content_evidence": [],
        "toc_sequences": [],
        "toc_blocks": [],
        "table_asts": [],
        "image_blocks": [],
        "document_ast": {
            "pages": [],
            "table_refs": [],
            "toc_refs": [],
            "toc_sequence_refs": [],
            "image_refs": [],
            "content_evidence_refs": [],
            "content_unit_refs": [],
        },
        "metadata": {
            "page_count": 0,
            "table_count": 0,
            "image_count": 0,
            "figure_count": 0,
            "toc_count": 0,
            "toc_sequence_count": 0,
            "content_evidence_count": 0,
            "content_unit_count": 0,
            "fact_extraction_unit_count": 0,
            "review_required_table_count": 0,
            "review_required_toc_count": 0,
            "toc_review_item_count": 0,
            "aggregated_toc_review_item_count": 0,
            "continuation_table_count": 0,
            "cross_page_table_links": 0,
            "cross_page_boundary_row_merge_count": 0,
            "low_confidence_table_count": 0,
            "diagnostic_table_count": 0,
            "pdf_is_readable": False,
            "pdf_readability_issue": str(readability_issue or "").strip() or "open_failed",
            "pdf_format_version": "",
            "pdf_is_encrypted": False,
            "pdf_needs_password": False,
            "pdf_openable_without_password": False,
            "pdf_security_permissions": 0,
            "pdf_encryption_scheme": "",
            "pdf_has_security_settings": False,
            "pdf_is_linearized": False,
            "pdf_font_records": [],
            "pdf_font_count": 0,
            "pdf_font_embedding_evidence_available": False,
            "pdf_initial_view_page_mode": "",
            "pdf_initial_view_page_layout": "",
            "pdf_initial_view_open_action_present": False,
            "pdf_initial_view_open_action_kind": "",
            "pdf_disallowed_content_markers": [],
            "pdf_link_action_kinds": [],
            "pdf_link_action_pages": [],
            "pdf_link_action_page_records": [],
            "pdf_broken_link_annotation_count": 0,
            "pdf_link_multiple_action_count": 0,
            "pdf_link_multiple_action_xrefs": [],
            "pdf_link_multiple_action_evidence_available": False,
            "pdf_bookmark_action_kinds": [],
            "pdf_bookmark_action_evidence_available": False,
            "pdf_bookmark_validity_evidence_available": False,
            "pdf_invalid_bookmark_count": 0,
            "pdf_invalid_bookmark_xrefs": [],
            "pdf_bookmark_target_integrity_evidence_available": False,
            "pdf_damaged_bookmark_target_count": 0,
            "pdf_damaged_bookmark_target_xrefs": [],
            "pdf_damaged_bookmark_target_details": [],
            "pdf_bookmark_external_file_targets": [],
            "pdf_bookmark_external_file_target_count": 0,
            "pdf_bookmark_external_file_target_evidence_available": False,
            "pdf_bookmark_uri_targets": [],
            "pdf_bookmark_uri_target_count": 0,
            "pdf_bookmark_uri_target_evidence_available": False,
            "pdf_bookmark_multiple_action_count": 0,
            "pdf_bookmark_multiple_action_xrefs": [],
            "pdf_bookmark_multiple_action_evidence_available": False,
            "parser_hint": "pdf-unreadable",
            "parse_error": str(error_message or "").strip(),
        },
    }


def _build_password_protected_pdf_result(path: Path, security_metadata: dict[str, Any]) -> dict[str, Any]:
    page_count = int(security_metadata.get("page_count", 0) or 0)
    return {
        "filename": path.name,
        "source_type": "pdf",
        "source_path": str(path),
        "text": "",
        "atomic_facts": {},
        "content_units": [],
        "content_evidence": [],
        "toc_sequences": [],
        "toc_blocks": [],
        "table_asts": [],
        "image_blocks": [],
        "document_ast": {
            "pages": [{"page": page_number} for page_number in range(1, page_count + 1)],
            "table_refs": [],
            "toc_refs": [],
            "toc_sequence_refs": [],
            "image_refs": [],
            "content_evidence_refs": [],
            "content_unit_refs": [],
        },
        "metadata": {
            "page_count": page_count,
            "table_count": 0,
            "image_count": 0,
            "figure_count": 0,
            "toc_count": 0,
            "toc_sequence_count": 0,
            "content_evidence_count": 0,
            "content_unit_count": 0,
            "fact_extraction_unit_count": 0,
            "review_required_table_count": 0,
            "review_required_toc_count": 0,
            "toc_review_item_count": 0,
            "aggregated_toc_review_item_count": 0,
            "continuation_table_count": 0,
            "cross_page_table_links": 0,
            "cross_page_boundary_row_merge_count": 0,
            "low_confidence_table_count": 0,
            "diagnostic_table_count": 0,
            "parser_hint": "pdf-password-protected",
            **security_metadata,
        },
    }


def parse_pdf(path: Path) -> dict[str, Any]:
    """Parse PDF into text/image/table AST while preserving BBox anchors."""
    try:
        security_metadata = _extract_pdf_security_metadata(path)
    except Exception as exc:
        return _build_unreadable_pdf_result(path, "open_failed", str(exc))

    if int(security_metadata.get("page_count", 0) or 0) <= 0:
        return _build_unreadable_pdf_result(path, "zero_pages")

    if bool(security_metadata.get("pdf_needs_password")):
        return _build_password_protected_pdf_result(path, security_metadata)

    pipeline_state = run_pdf_extraction_pipeline(path)
    parsed = build_pdf_parse_result(path, pipeline_state)
    parsed_metadata = dict(parsed.get("metadata", {}) or {})
    parsed_metadata.update(security_metadata)
    parsed["metadata"] = parsed_metadata
    return parsed

