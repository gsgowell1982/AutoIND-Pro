from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from parsers.common.atomic_fact_extractor import extract_atomic_fact_matches


MATERIAL_REVIEW_CONTRACT_VERSION = "material-review-contract-v1"

_MODULE_SIGNAL_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("M1", "module_1", re.compile(r"(?<![a-z0-9])(?:module[\s._/-]*1|m[\s._/-]*1)(?![a-z0-9])", re.IGNORECASE)),
    ("M2", "module_2", re.compile(r"(?<![a-z0-9])(?:module[\s._/-]*2|m[\s._/-]*2)(?![a-z0-9])", re.IGNORECASE)),
    (
        "M3",
        "module_3",
        re.compile(
            r"(?<![a-z0-9])(?:module[\s._/-]*3|m[\s._/-]*3)(?![a-z0-9])|(?<!\d)3\.2(?:\.[sp])?(?!\d)",
            re.IGNORECASE,
        ),
    ),
    ("M4", "module_4", re.compile(r"(?<![a-z0-9])(?:module[\s._/-]*4|m[\s._/-]*4)(?![a-z0-9])", re.IGNORECASE)),
    ("M5", "module_5", re.compile(r"(?<![a-z0-9])(?:module[\s._/-]*5|m[\s._/-]*5)(?![a-z0-9])", re.IGNORECASE)),
)
_QUALITY_OVERVIEW_SIGNAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("quality_overall_summary", re.compile(r"quality[\s._/-]*overall[\s._/-]*summary", re.IGNORECASE)),
    ("quality_overview", re.compile(r"quality[\s._/-]*overview", re.IGNORECASE)),
    ("qos", re.compile(r"(?<![a-z0-9])qos(?![a-z0-9])", re.IGNORECASE)),
    ("ctd_2_3", re.compile(r"(?<!\d)2\.3(?!\d)", re.IGNORECASE)),
    ("ctd_2_3_flat", re.compile(r"(?<!\d)23(?=[\s._/-]*quality)", re.IGNORECASE)),
)
_QUALITY_MODULE_SIGNAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ctd_3_2", re.compile(r"(?<!\d)3\.2(?:\.[sp])?(?!\d)", re.IGNORECASE)),
    ("cmc", re.compile(r"(?<![a-z0-9])cmc(?![a-z0-9])", re.IGNORECASE)),
)
_SECTION_OUTLINE_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)*$")
_SECTION_OUTLINE_APPENDIX_RE = re.compile(r"^APPENDIX\s+([A-Z]+)$", re.IGNORECASE)
_SECTION_OUTLINE_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_SECTION_OUTLINE_ALPHA_RE = re.compile(r"^[A-Z]+$", re.IGNORECASE)
_ECTD_SUBMISSION_METADATA_KEYS = (
    "ectd_application_number",
    "ectd_sequence_number",
    "ectd_related_sequence_number",
    "ectd_previous_sequence_number",
    "ectd_sequence_directory_number",
    "ectd_schema_version",
    "ectd_sequence_description",
    "ectd_sequence_contact",
    "ectd_sequence_contact_name",
    "ectd_sequence_contact_phone",
    "ectd_sequence_contact_email",
    "ectd_envelope_attributes",
    "ectd_envelope_count",
    "ectd_element_records",
    "ectd_attribute_records",
    "ectd_checksum_types",
    "ectd_leaf_hrefs",
    "ectd_leaf_records",
    "ectd_leaf_title_records",
    "ectd_leaf_lifecycle_records",
    "ectd_leaf_count",
    "ectd_node_extension_records",
    "ectd_32r_regional_information_present",
    "ectd_32r_regional_information_count",
    "ectd_32r_extension_titles",
    "ectd_32r_extension_count",
    "ectd_32r_extension_leaf_hrefs",
    "ectd_controlled_vocabulary_name",
    "ectd_controlled_vocabulary_values",
    "ectd_dependency_matrix_name",
    "ectd_dependency_matrix_rows",
    "xml_is_well_formed",
    "xml_parse_error",
    "xml_parse_error_line",
    "xml_parse_error_column",
    "xml_doctype_present",
    "xml_doctype_name",
    "xml_doctype_public_id",
    "xml_doctype_system_id",
    "xml_doctype_system_id_normalized",
    "xml_dtd_system_id_is_local_path",
    "xml_dtd_resolved_path",
    "xml_dtd_resolved_path_exists",
    "xml_dtd_resolved_filename",
    "xml_dtd_validation_attempted",
    "xml_dtd_validation_prerequisite_missing",
    "xml_dtd_is_valid",
    "xml_dtd_validation_error_count",
    "xml_dtd_validation_errors",
    "xml_dtd_validation_error",
    "xml_schema_location_raw",
    "xml_schema_location_count",
    "xml_schema_location_records",
    "xml_schema_locations_all_local",
    "xml_schema_locations_all_resolve",
    "xml_schema_locations_all_point_to_util",
    "xml_schema_validation_attempted",
    "xml_schema_validation_schema_path",
    "xml_schema_is_valid",
    "xml_schema_validation_error_count",
    "xml_schema_validation_errors",
    "xml_schema_validation_prerequisite_missing",
    "xml_schema_validation_error",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_document_classification(
    document: dict[str, Any],
    document_order: int,
) -> dict[str, Any]:
    source_texts = _build_classification_sources(document)
    module_label = f"DOC-{document_order}"
    module_signal_source = "fallback"
    module_signal_hit = "fallback"

    for source_name, source_text in source_texts:
        matched_label = _match_module_label(source_text)
        if matched_label is None:
            continue
        module_label, module_signal_hit = matched_label
        module_signal_source = source_name
        break

    quality_overview_hits = _match_signal_hits(source_texts, _QUALITY_OVERVIEW_SIGNAL_PATTERNS)
    quality_module_hits = _match_signal_hits(source_texts, _QUALITY_MODULE_SIGNAL_PATTERNS)

    if module_label == "M3" and "module_3" not in quality_module_hits:
        quality_module_hits.insert(0, "module_3")

    document_tags: list[str] = []
    if quality_overview_hits:
        document_tags.append("quality_overview")
    if module_label == "M3" or quality_module_hits:
        document_tags.append("quality_module")

    signal_hits = _dedupe_preserve_order(
        [
            *([module_signal_hit] if module_signal_source != "fallback" else []),
            *quality_overview_hits,
            *quality_module_hits,
        ]
    )

    return {
        "module_label": module_label,
        "module_signal_source": module_signal_source,
        "module_signal_hit": module_signal_hit,
        "quality_overview_candidate": bool(quality_overview_hits),
        "quality_module_candidate": module_label == "M3" or bool(quality_module_hits),
        "document_tags": document_tags,
        "signal_hits": signal_hits,
    }


def _build_classification_sources(document: dict[str, Any]) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    for source_name in ("filename", "source_path"):
        raw_value = str(document.get(source_name) or "").strip()
        if not raw_value:
            continue
        normalized = raw_value.lower().replace("\\", "/")
        sources.append((source_name, normalized))
    return sources


def _match_module_label(source_text: str) -> tuple[str, str] | None:
    for module_label, signal_hit, pattern in _MODULE_SIGNAL_PATTERNS:
        if pattern.search(source_text):
            return module_label, signal_hit
    return None


def _match_signal_hits(
    source_texts: list[tuple[str, str]],
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> list[str]:
    hits: list[str] = []
    for signal_name, pattern in patterns:
        if any(pattern.search(source_text) for _, source_text in source_texts):
            hits.append(signal_name)
    return hits


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def build_material_review_contract(
    parsed_documents: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    evidence_index: list[dict[str, Any]] = []
    unit_index: list[dict[str, Any]] = []
    navigation_index: list[dict[str, Any]] = []
    section_index: list[dict[str, Any]] = []
    paragraph_index: list[dict[str, Any]] = []
    fact_index: list[dict[str, Any]] = []
    fact_signal_index: list[dict[str, Any]] = []
    diagnostic_index: list[dict[str, Any]] = []

    for document_order, document in enumerate(parsed_documents, start=1):
        document_id = _build_document_id(document, document_order)
        fact_signals = _build_fact_signal_index(document, document_id, document_order)
        document_section_index = _build_section_index(document, document_id, document_order)
        document_paragraph_index = _build_paragraph_index(document, document_id, document_order)
        _enrich_structure_outline_hierarchy(document_section_index, document_paragraph_index)
        documents.append(_build_document_summary(document, document_id, document_order, fact_signals))
        evidence_index.extend(_build_evidence_index(document, document_id, document_order))
        unit_index.extend(_build_unit_index(document, document_id, document_order))
        navigation_index.extend(_build_navigation_index(document, document_id, document_order))
        section_index.extend(document_section_index)
        paragraph_index.extend(document_paragraph_index)
        fact_index.extend(_build_fact_index(document, document_id, document_order, fact_signals))
        fact_signal_index.extend(fact_signals)
        diagnostic_index.extend(_build_diagnostic_index(document, document_id, document_order))

    section_tree = _build_section_tree(section_index, paragraph_index)
    submission_scope = _build_submission_scope(documents)

    return {
        "schema_version": MATERIAL_REVIEW_CONTRACT_VERSION,
        "generated_at": generated_at or _utc_now(),
        "document_count": len(documents),
        "documents": documents,
        "evidence_index": evidence_index,
        "unit_index": unit_index,
        "navigation_index": navigation_index,
        "section_index": section_index,
        "paragraph_index": paragraph_index,
        "section_tree": section_tree,
        "fact_index": fact_index,
        "fact_signal_index": fact_signal_index,
        "diagnostic_index": diagnostic_index,
        "submission_scope": submission_scope,
        "index_summary": {
            "document_count": len(documents),
            "evidence_count": len(evidence_index),
            "unit_count": len(unit_index),
            "navigation_sequence_count": len(navigation_index),
            "section_anchor_count": len(section_index),
            "paragraph_count": len(paragraph_index),
            "section_tree_node_count": _count_section_tree_nodes(section_tree),
            "fact_count": len(fact_index),
            "fact_signal_count": len(fact_signal_index),
            "diagnostic_count": len(diagnostic_index),
            "upload_mode": str(submission_scope.get("upload_mode") or "").strip(),
        },
    }


def _resolve_ectd_sequence_root_from_source_path(source_path: str) -> Path | None:
    normalized_path = str(source_path or "").strip()
    if not normalized_path:
        return None
    try:
        path = Path(normalized_path).resolve()
    except OSError:
        return None

    filename = path.name.lower()
    if filename in {"index.xml", "index-md5.txt"}:
        return path.parent
    if (
        filename == "cn-regional.xml"
        and path.parent.name.lower() == "cn"
        and path.parent.parent.name.lower() == "m1"
    ):
        return path.parents[2]
    return None


def _find_ectd_regional_original_number(metadata: dict[str, Any]) -> str:
    envelope_attributes = dict(metadata.get("ectd_envelope_attributes") or {})
    for field_name in ("product-number", "original-number"):
        field_value = str(envelope_attributes.get(field_name) or "").strip()
        if field_value:
            return field_value

    for element in list(metadata.get("ectd_element_records") or []):
        element_record = dict(element or {})
        normalized_name = str(element_record.get("element_normalized_name") or "").strip()
        if normalized_name not in {"productnumber", "originalnumber"}:
            continue
        ancestor_names = {
            str(item or "").strip()
            for item in list(element_record.get("ancestor_normalized_names") or [])
            if str(item or "").strip()
        }
        if "cnenvelope" not in ancestor_names and "envelope" not in ancestor_names:
            continue
        return str(element_record.get("element_text") or "").strip()

    return ""


def _build_submission_scope(documents: list[dict[str, Any]]) -> dict[str, Any]:
    total_document_count = len(documents)
    if total_document_count <= 1:
        upload_mode = "single_document"
        available_scopes = ["document"]
    else:
        upload_mode = "document_batch"
        available_scopes = ["document"]

    sequence_packages: dict[str, dict[str, Any]] = {}
    has_ectd_submission_metadata = False
    for document in documents:
        if dict(document.get("ectd_submission_metadata") or {}):
            has_ectd_submission_metadata = True
        source_path = str(document.get("source_path") or "").strip()
        sequence_root = _resolve_ectd_sequence_root_from_source_path(source_path)
        if sequence_root is None:
            continue
        metadata = dict(document.get("ectd_submission_metadata") or {})
        envelope_attributes = dict(metadata.get("ectd_envelope_attributes") or {})
        package = sequence_packages.setdefault(
            str(sequence_root),
            {
                "sequence_root": str(sequence_root),
                "sequence_name": sequence_root.name,
                "application_root": str(sequence_root.parent),
                "application_root_name": sequence_root.parent.name,
                "application_number": "",
                "application_type": "",
                "product_type": "",
                "original_number": "",
                "sequence_number": "",
                "sequence_type": "",
                "schema_version": "",
                "sequence_description": "",
                "sequence_contact_name": "",
                "sequence_contact_phone": "",
                "sequence_contact_email": "",
                "related_sequence_number": "",
                "regulatory_activity_type": "",
                "document_ids": [],
            },
        )
        package["document_ids"].append(str(document.get("document_id") or "").strip())
        if not package["application_number"]:
            package["application_number"] = str(metadata.get("ectd_application_number") or "").strip()
        if not package["application_type"]:
            package["application_type"] = str(envelope_attributes.get("application-type") or "").strip()
        if not package["product_type"]:
            package["product_type"] = str(envelope_attributes.get("product-type") or "").strip()
        if (
            not package["original_number"]
            and str(document.get("filename") or "").strip().lower() == "cn-regional.xml"
        ):
            package["original_number"] = _find_ectd_regional_original_number(metadata)
        if not package["sequence_number"]:
            package["sequence_number"] = str(metadata.get("ectd_sequence_number") or "").strip()
        if not package["sequence_type"]:
            package["sequence_type"] = str(envelope_attributes.get("sequence-type") or "").strip()
        if (
            not package["schema_version"]
            and str(document.get("filename") or "").strip().lower() == "cn-regional.xml"
        ):
            package["schema_version"] = str(metadata.get("ectd_schema_version") or "").strip()
        if not package["sequence_description"]:
            package["sequence_description"] = str(metadata.get("ectd_sequence_description") or "").strip()
        if not package["sequence_contact_name"]:
            package["sequence_contact_name"] = str(
                metadata.get("ectd_sequence_contact_name")
                or (metadata.get("ectd_sequence_contact") or {}).get("name")
                or ""
            ).strip()
        if not package["sequence_contact_phone"]:
            package["sequence_contact_phone"] = str(
                metadata.get("ectd_sequence_contact_phone")
                or (metadata.get("ectd_sequence_contact") or {}).get("phone")
                or ""
            ).strip()
        if not package["sequence_contact_email"]:
            package["sequence_contact_email"] = str(
                metadata.get("ectd_sequence_contact_email")
                or (metadata.get("ectd_sequence_contact") or {}).get("email")
                or ""
            ).strip()
        if not package["related_sequence_number"]:
            package["related_sequence_number"] = str(
                metadata.get("ectd_related_sequence_number")
                or metadata.get("ectd_previous_sequence_number")
                or ""
            ).strip()
        if not package["regulatory_activity_type"]:
            package["regulatory_activity_type"] = str(envelope_attributes.get("regulatory-activity-type") or "").strip()

    normalized_sequence_packages: list[dict[str, Any]] = []
    for package in sequence_packages.values():
        application_key = str(package.get("application_number") or package.get("application_root_name") or "").strip()
        sequence_number = str(package.get("sequence_number") or "").strip()
        sequence_root = str(package.get("sequence_root") or "").strip()
        sequence_package_id = (
            f"seqpkg:{application_key}:{sequence_number or Path(sequence_root).name}"
            if application_key
            else f"seqpkg:{Path(sequence_root).name}"
        )
        filenames = sorted(
            {
                str(document.get("filename") or "").strip()
                for document in documents
                if str(document.get("document_id") or "").strip() in set(package.get("document_ids", []) or [])
                and str(document.get("filename") or "").strip()
            }
        )
        normalized_sequence_packages.append(
            {
                **package,
                "sequence_package_id": sequence_package_id,
                "application_key": application_key,
                "filenames": filenames,
            }
        )

    regulatory_activities: dict[str, dict[str, Any]] = {}
    for package in normalized_sequence_packages:
        application_key = str(package.get("application_key") or "").strip()
        if not application_key:
            continue
        related_sequence_number = str(
            package.get("related_sequence_number") or package.get("sequence_number") or ""
        ).strip()
        regulatory_activity_type = str(package.get("regulatory_activity_type") or "").strip()
        regulatory_activity_id = (
            f"activity:{application_key}:{related_sequence_number or 'unknown'}:{regulatory_activity_type or 'unknown'}"
        )
        activity = regulatory_activities.setdefault(
            regulatory_activity_id,
            {
                "regulatory_activity_id": regulatory_activity_id,
                "application_key": application_key,
                "related_sequence_number": related_sequence_number,
                "regulatory_activity_type": regulatory_activity_type,
                "sequence_package_ids": [],
                "sequence_numbers": [],
                "document_ids": [],
            },
        )
        sequence_package_id = str(package.get("sequence_package_id") or "").strip()
        if sequence_package_id and sequence_package_id not in activity["sequence_package_ids"]:
            activity["sequence_package_ids"].append(sequence_package_id)
        sequence_number = str(package.get("sequence_number") or "").strip()
        if sequence_number and sequence_number not in activity["sequence_numbers"]:
            activity["sequence_numbers"].append(sequence_number)
        for document_id in package.get("document_ids", []) or []:
            if document_id and document_id not in activity["document_ids"]:
                activity["document_ids"].append(document_id)

    application_projects: dict[str, dict[str, Any]] = {}
    for package in normalized_sequence_packages:
        application_key = str(package.get("application_key") or "").strip()
        if not application_key:
            continue
        project = application_projects.setdefault(
            application_key,
            {
                "application_key": application_key,
                "application_root_name": str(package.get("application_root_name") or "").strip(),
                "sequence_roots": [],
                "sequence_package_ids": [],
                "sequence_numbers": [],
                "regulatory_activity_ids": [],
            },
        )
        sequence_root = str(package.get("sequence_root") or "").strip()
        if sequence_root and sequence_root not in project["sequence_roots"]:
            project["sequence_roots"].append(sequence_root)
        sequence_package_id = str(package.get("sequence_package_id") or "").strip()
        if sequence_package_id and sequence_package_id not in project["sequence_package_ids"]:
            project["sequence_package_ids"].append(sequence_package_id)
        sequence_number = str(package.get("sequence_number") or "").strip()
        if sequence_number and sequence_number not in project["sequence_numbers"]:
            project["sequence_numbers"].append(sequence_number)
        related_sequence_number = str(
            package.get("related_sequence_number") or package.get("sequence_number") or ""
        ).strip()
        regulatory_activity_type = str(package.get("regulatory_activity_type") or "").strip()
        regulatory_activity_id = (
            f"activity:{application_key}:{related_sequence_number or 'unknown'}:{regulatory_activity_type or 'unknown'}"
        )
        if regulatory_activity_id not in project["regulatory_activity_ids"]:
            project["regulatory_activity_ids"].append(regulatory_activity_id)

    for project in application_projects.values():
        project["sequence_count"] = len(project.get("sequence_package_ids") or [])
        project["regulatory_activity_count"] = len(project.get("regulatory_activity_ids") or [])

    if total_document_count > 1 and has_ectd_submission_metadata:
        available_scopes = ["document", "sequence"]
        upload_mode = "ectd_sequence_candidate"
        if normalized_sequence_packages:
            upload_mode = "ectd_sequence_package" if len(normalized_sequence_packages) == 1 else "ectd_sequence_batch"
        if regulatory_activities:
            available_scopes.append("activity")
        if application_projects:
            if "activity" not in available_scopes and regulatory_activities:
                available_scopes.append("activity")
            available_scopes.append("application")
        if any(len(project.get("sequence_roots") or []) >= 2 for project in application_projects.values()):
            upload_mode = "ectd_application_project"

    return {
        "upload_mode": upload_mode,
        "available_scopes": available_scopes,
        "scope_status": {
            "document": True,
            "sequence": "sequence" in available_scopes,
            "activity": "activity" in available_scopes,
            "application": "application" in available_scopes,
        },
        "ectd_project_context": {
            "sequence_package_count": len(normalized_sequence_packages),
            "regulatory_activity_count": len(regulatory_activities),
            "application_project_count": len(application_projects),
            "sequence_packages": normalized_sequence_packages,
            "regulatory_activities": list(regulatory_activities.values()),
            "application_projects": list(application_projects.values()),
        },
    }


def _build_document_id(document: dict[str, Any], document_order: int) -> str:
    explicit = str(document.get("document_id") or "").strip()
    if explicit:
        return explicit
    return f"doc_{document_order:03d}"


def _build_document_summary(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
    fact_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(document.get("metadata", {}) or {})
    document_ast = dict(document.get("document_ast", {}) or {})
    toc_sequences = list(document.get("toc_sequences", []) or [])
    atomic_facts = dict(document.get("atomic_facts", {}) or {})
    classification = infer_document_classification(document, document_order)
    fact_support_lookup = _build_fact_support_lookup(fact_signals)
    ectd_submission_metadata = {
        key: metadata.get(key)
        for key in _ECTD_SUBMISSION_METADATA_KEYS
        if metadata.get(key) not in (None, "")
    }
    content_evidence = list(document.get("content_evidence", []) or [])
    content_units = list(document.get("content_units", []) or [])
    section_anchored_evidence_count = sum(
        1 for evidence in content_evidence if dict(evidence.get("section_context", {}) or {})
    )
    section_anchored_unit_count = sum(
        1 for unit in content_units if dict(unit.get("section_context", {}) or {})
    )
    outline_indices = sorted(
        {
            str((item.get("section_context", {}) or {}).get("outline_index") or "").strip()
            for item in [*content_evidence, *content_units]
            if str((item.get("section_context", {}) or {}).get("outline_index") or "").strip()
        }
    )
    fact_with_unit_provenance_count = sum(
        1
        for fact_key, fact_value in atomic_facts.items()
        if _resolve_fact_supporting_units(fact_support_lookup, fact_key, fact_value)
    )

    return {
        "document_id": document_id,
        "document_order": document_order,
        "file_id": document.get("file_id"),
        "filename": document.get("filename"),
        "source_type": document.get("source_type"),
        "source_path": document.get("source_path"),
        "classification": classification,
        "submission_scope_kind": document.get("submission_scope_kind"),
        "document_scope_kind": document.get("document_scope_kind"),
        "ectd_submission_metadata": ectd_submission_metadata,
        "summary": {
            "page_count": int(metadata.get("page_count", 0) or 0),
            "table_count": int(metadata.get("table_count", 0) or 0),
            "image_count": int(metadata.get("image_count", 0) or 0),
            "figure_count": int(metadata.get("figure_count", 0) or 0),
            "toc_count": int(metadata.get("toc_count", 0) or 0),
            "toc_sequence_count": int(metadata.get("toc_sequence_count", len(toc_sequences)) or 0),
            "embedded_outline_count": int(metadata.get("embedded_outline_count", 0) or 0),
            "embedded_outline_depth": int(metadata.get("embedded_outline_depth", 0) or 0),
            "embedded_file_count": int(metadata.get("embedded_file_count", 0) or 0),
            "embedded_file_names": [
                str(item).strip()
                for item in list(metadata.get("embedded_file_names", []) or [])
                if str(item).strip()
            ],
            "link_annotation_count": int(metadata.get("link_annotation_count", 0) or 0),
            "navigational_link_count": int(metadata.get("navigational_link_count", 0) or 0),
            "internal_link_count": int(metadata.get("internal_link_count", 0) or 0),
            "external_file_link_count": int(metadata.get("external_file_link_count", 0) or 0),
            "external_uri_link_count": int(metadata.get("external_uri_link_count", 0) or 0),
            "external_file_link_targets": [
                str(item).strip()
                for item in list(metadata.get("external_file_link_targets", []) or [])
                if str(item).strip()
            ],
            "pdf_link_action_kinds": [
                str(item).strip()
                for item in list(metadata.get("pdf_link_action_kinds", []) or [])
                if str(item).strip()
            ],
            "pdf_link_action_pages": [
                int(item)
                for item in list(metadata.get("pdf_link_action_pages", []) or [])
                if str(item).strip().isdigit()
            ],
            "pdf_link_action_page_records": [
                {
                    "page": int(item.get("page", 0) or 0),
                    "link_annotation_count": int(item.get("link_annotation_count", 0) or 0),
                    "link_annotation_xrefs": [
                        int(xref)
                        for xref in list(item.get("link_annotation_xrefs", []) or [])
                        if str(xref).strip().isdigit()
                    ],
                    "link_action_kinds": [
                        str(kind).strip()
                        for kind in list(item.get("link_action_kinds", []) or [])
                        if str(kind).strip()
                    ],
                }
                for item in list(metadata.get("pdf_link_action_page_records", []) or [])
                if isinstance(item, dict)
            ],
            "pdf_broken_link_annotation_count": int(metadata.get("pdf_broken_link_annotation_count", 0) or 0),
            "pdf_link_multiple_action_evidence_available": bool(
                metadata.get("pdf_link_multiple_action_evidence_available", "pdf_link_multiple_action_count" in metadata)
            ),
            "pdf_link_multiple_action_count": int(metadata.get("pdf_link_multiple_action_count", 0) or 0),
            "pdf_link_multiple_action_xrefs": [
                int(item)
                for item in list(metadata.get("pdf_link_multiple_action_xrefs", []) or [])
                if str(item).strip().isdigit()
            ],
            "pdf_link_inherit_zoom_evidence_available": bool(
                metadata.get(
                    "pdf_link_inherit_zoom_evidence_available",
                    "pdf_link_non_inherit_zoom_count" in metadata,
                )
            ),
            "pdf_link_destination_count": int(metadata.get("pdf_link_destination_count", 0) or 0),
            "pdf_link_non_inherit_zoom_count": int(metadata.get("pdf_link_non_inherit_zoom_count", 0) or 0),
            "pdf_link_non_inherit_zoom_xrefs": [
                int(item)
                for item in list(metadata.get("pdf_link_non_inherit_zoom_xrefs", []) or [])
                if str(item).strip().isdigit()
            ],
            "pdf_link_non_inherit_zoom_details": [
                {
                    "xref": int(item.get("xref", 0) or 0),
                    "reason": str(item.get("reason") or "").strip(),
                    "destination_source": str(item.get("destination_source") or "").strip(),
                    "destination": str(item.get("destination") or "").strip(),
                    "destination_kind": str(item.get("destination_kind") or "").strip(),
                    **(
                        {"zoom": str(item.get("zoom") or "").strip()}
                        if str(item.get("zoom") or "").strip()
                        else {}
                    ),
                }
                for item in list(metadata.get("pdf_link_non_inherit_zoom_details", []) or [])
                if isinstance(item, dict)
            ],
            "pdf_bookmark_action_kinds": [
                str(item).strip()
                for item in list(metadata.get("pdf_bookmark_action_kinds", []) or [])
                if str(item).strip()
            ],
            "pdf_bookmark_action_evidence_available": bool(
                metadata.get("pdf_bookmark_action_evidence_available", "pdf_bookmark_action_kinds" in metadata)
            ),
            "pdf_bookmark_validity_evidence_available": bool(
                metadata.get(
                    "pdf_bookmark_validity_evidence_available",
                    "pdf_invalid_bookmark_count" in metadata,
                )
            ),
            "pdf_invalid_bookmark_count": int(metadata.get("pdf_invalid_bookmark_count", 0) or 0),
            "pdf_invalid_bookmark_xrefs": [
                int(item)
                for item in list(metadata.get("pdf_invalid_bookmark_xrefs", []) or [])
                if str(item).strip().isdigit()
            ],
            "pdf_bookmark_target_integrity_evidence_available": bool(
                metadata.get(
                    "pdf_bookmark_target_integrity_evidence_available",
                    "pdf_damaged_bookmark_target_count" in metadata,
                )
            ),
            "pdf_damaged_bookmark_target_count": int(metadata.get("pdf_damaged_bookmark_target_count", 0) or 0),
            "pdf_damaged_bookmark_target_xrefs": [
                int(item)
                for item in list(metadata.get("pdf_damaged_bookmark_target_xrefs", []) or [])
                if str(item).strip().isdigit()
            ],
            "pdf_damaged_bookmark_target_details": [
                {
                    "xref": int(item.get("xref", 0) or 0),
                    "reason": str(item.get("reason") or "").strip(),
                    "destination_source": str(item.get("destination_source") or "").strip(),
                    "destination": str(item.get("destination") or "").strip(),
                    **(
                        {"page_xref": int(item.get("page_xref", 0) or 0)}
                        if str(item.get("page_xref", "")).strip().isdigit()
                        else {}
                    ),
                }
                for item in list(metadata.get("pdf_damaged_bookmark_target_details", []) or [])
                if isinstance(item, dict)
            ],
            "pdf_bookmark_inherit_zoom_evidence_available": bool(
                metadata.get(
                    "pdf_bookmark_inherit_zoom_evidence_available",
                    "pdf_bookmark_non_inherit_zoom_count" in metadata,
                )
            ),
            "pdf_bookmark_destination_count": int(metadata.get("pdf_bookmark_destination_count", 0) or 0),
            "pdf_bookmark_non_inherit_zoom_count": int(
                metadata.get("pdf_bookmark_non_inherit_zoom_count", 0) or 0
            ),
            "pdf_bookmark_non_inherit_zoom_xrefs": [
                int(item)
                for item in list(metadata.get("pdf_bookmark_non_inherit_zoom_xrefs", []) or [])
                if str(item).strip().isdigit()
            ],
            "pdf_bookmark_non_inherit_zoom_details": [
                {
                    "xref": int(item.get("xref", 0) or 0),
                    "reason": str(item.get("reason") or "").strip(),
                    "destination_source": str(item.get("destination_source") or "").strip(),
                    "destination": str(item.get("destination") or "").strip(),
                    "destination_kind": str(item.get("destination_kind") or "").strip(),
                    **(
                        {"zoom": str(item.get("zoom") or "").strip()}
                        if str(item.get("zoom") or "").strip()
                        else {}
                    ),
                }
                for item in list(metadata.get("pdf_bookmark_non_inherit_zoom_details", []) or [])
                if isinstance(item, dict)
            ],
            "pdf_bookmark_external_file_targets": [
                str(item).strip()
                for item in list(metadata.get("pdf_bookmark_external_file_targets", []) or [])
                if str(item).strip()
            ],
            "pdf_bookmark_external_file_target_count": int(
                metadata.get(
                    "pdf_bookmark_external_file_target_count",
                    len(list(metadata.get("pdf_bookmark_external_file_targets", []) or [])),
                )
                or 0
            ),
            "pdf_bookmark_external_file_target_evidence_available": bool(
                metadata.get(
                    "pdf_bookmark_external_file_target_evidence_available",
                    "pdf_bookmark_external_file_targets" in metadata,
                )
            ),
            "pdf_bookmark_uri_targets": [
                str(item).strip()
                for item in list(metadata.get("pdf_bookmark_uri_targets", []) or [])
                if str(item).strip()
            ],
            "pdf_bookmark_uri_target_count": int(
                metadata.get(
                    "pdf_bookmark_uri_target_count",
                    len(list(metadata.get("pdf_bookmark_uri_targets", []) or [])),
                )
                or 0
            ),
            "pdf_bookmark_uri_target_evidence_available": bool(
                metadata.get(
                    "pdf_bookmark_uri_target_evidence_available",
                    "pdf_bookmark_uri_targets" in metadata,
                )
            ),
            "pdf_bookmark_multiple_action_evidence_available": bool(
                metadata.get(
                    "pdf_bookmark_multiple_action_evidence_available",
                    "pdf_bookmark_multiple_action_count" in metadata,
                )
            ),
            "pdf_bookmark_multiple_action_count": int(metadata.get("pdf_bookmark_multiple_action_count", 0) or 0),
            "pdf_bookmark_multiple_action_xrefs": [
                int(item)
                for item in list(metadata.get("pdf_bookmark_multiple_action_xrefs", []) or [])
                if str(item).strip().isdigit()
            ],
            "non_link_annotation_count": int(metadata.get("non_link_annotation_count", 0) or 0),
            "non_link_annotation_types": [
                str(item).strip()
                for item in list(metadata.get("non_link_annotation_types", []) or [])
                if str(item).strip()
            ],
            "pdf_is_encrypted": bool(metadata.get("pdf_is_encrypted")),
            "pdf_is_readable": bool(metadata.get("pdf_is_readable", True)),
            "pdf_readability_issue": str(metadata.get("pdf_readability_issue") or "").strip(),
            "pdf_needs_password": bool(metadata.get("pdf_needs_password")),
            "pdf_openable_without_password": bool(
                metadata.get("pdf_openable_without_password", True)
            ),
            "pdf_format_version": str(metadata.get("pdf_format_version") or "").strip(),
            "pdf_security_permissions": int(metadata.get("pdf_security_permissions", 0) or 0),
            "pdf_encryption_scheme": str(metadata.get("pdf_encryption_scheme") or "").strip(),
            "pdf_has_security_settings": bool(metadata.get("pdf_has_security_settings")),
            "pdf_linearization_evidence_available": "pdf_is_linearized" in metadata,
            "pdf_is_linearized": bool(metadata.get("pdf_is_linearized")),
            "pdf_font_embedding_evidence_available": bool(
                metadata.get("pdf_font_embedding_evidence_available", "pdf_font_records" in metadata)
            ),
            "pdf_font_count": int(metadata.get("pdf_font_count", len(metadata.get("pdf_font_records", []) or [])) or 0),
            "pdf_font_records": [
                {
                    "xref": int((record or {}).get("xref", 0) or 0),
                    "font_type": str((record or {}).get("font_type") or "").strip(),
                    "base_font": str((record or {}).get("base_font") or "").strip(),
                    "resource_name": str((record or {}).get("resource_name") or "").strip(),
                    "encoding": str((record or {}).get("encoding") or "").strip(),
                    "file_extension": str((record or {}).get("file_extension") or "").strip(),
                    "is_embedded": bool((record or {}).get("is_embedded")),
                    "font_file_kind": str((record or {}).get("font_file_kind") or "").strip(),
                    "page_numbers": [
                        int(page_number)
                        for page_number in list((record or {}).get("page_numbers", []) or [])
                        if str(page_number).strip().isdigit()
                    ],
                }
                for record in list(metadata.get("pdf_font_records", []) or [])
                if isinstance(record, dict)
            ],
            "pdf_initial_view_page_mode": str(metadata.get("pdf_initial_view_page_mode") or "").strip(),
            "pdf_initial_view_page_layout": str(metadata.get("pdf_initial_view_page_layout") or "").strip(),
            "pdf_initial_view_open_action_present": bool(metadata.get("pdf_initial_view_open_action_present")),
            "pdf_initial_view_open_action_kind": str(metadata.get("pdf_initial_view_open_action_kind") or "").strip(),
            "pdf_disallowed_content_markers": [
                str(item).strip()
                for item in list(metadata.get("pdf_disallowed_content_markers", []) or [])
                if str(item).strip()
            ],
            "content_evidence_count": int(metadata.get("content_evidence_count", 0) or 0),
            "content_unit_count": int(metadata.get("content_unit_count", 0) or 0),
            "fact_extraction_unit_count": int(metadata.get("fact_extraction_unit_count", 0) or 0),
            "section_anchored_evidence_count": section_anchored_evidence_count,
            "section_anchored_unit_count": section_anchored_unit_count,
            "outline_index_count": len(outline_indices),
            "fact_signal_count": len(fact_signals),
            "fact_with_unit_provenance_count": fact_with_unit_provenance_count,
            "fact_without_unit_provenance_count": max(len(atomic_facts) - fact_with_unit_provenance_count, 0),
        },
        "parser_diagnostics_summary": {
            "review_required_table_count": int(metadata.get("review_required_table_count", 0) or 0),
            "review_required_toc_count": int(metadata.get("review_required_toc_count", 0) or 0),
            "toc_review_item_count": int(metadata.get("toc_review_item_count", 0) or 0),
            "aggregated_toc_review_item_count": int(metadata.get("aggregated_toc_review_item_count", 0) or 0),
            "continuation_table_count": int(metadata.get("continuation_table_count", 0) or 0),
            "cross_page_table_links": int(metadata.get("cross_page_table_links", 0) or 0),
            "cross_page_boundary_row_merge_count": int(metadata.get("cross_page_boundary_row_merge_count", 0) or 0),
            "low_confidence_table_count": int(metadata.get("low_confidence_table_count", 0) or 0),
            "diagnostic_table_count": int(metadata.get("diagnostic_table_count", 0) or 0),
        },
        "count_breakdown": {
            "content_evidence_counts": deepcopy(metadata.get("content_evidence_counts", {}) or {}),
            "content_unit_counts": deepcopy(metadata.get("content_unit_counts", {}) or {}),
        },
        "structure_refs": {
            "page_refs": [
                int(page.get("page", 0) or 0)
                for page in document_ast.get("pages", []) or []
            ],
            "table_refs": list(document_ast.get("table_refs", []) or []),
            "toc_refs": list(document_ast.get("toc_refs", []) or []),
            "toc_sequence_refs": list(document_ast.get("toc_sequence_refs", []) or []),
            "image_refs": list(document_ast.get("image_refs", []) or []),
            "algorithm_refs": list(document_ast.get("algorithm_refs", []) or []),
            "content_evidence_refs": list(document_ast.get("content_evidence_refs", []) or []),
            "content_unit_refs": list(document_ast.get("content_unit_refs", []) or []),
        },
        "navigation_sequence_ids": [
            str(sequence.get("toc_sequence_id") or "").strip()
            for sequence in toc_sequences
            if str(sequence.get("toc_sequence_id") or "").strip()
        ],
        "fact_keys": sorted(str(key) for key in atomic_facts.keys()),
    }


def _build_unit_index(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for unit in document.get("content_units", []) or []:
        record = dict(unit)
        record["document_id"] = document_id
        record["document_order"] = document_order
        record["filename"] = document.get("filename")
        records.append(record)
    return records


def _build_evidence_index(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for evidence in document.get("content_evidence", []) or []:
        record = dict(evidence)
        record["document_id"] = document_id
        record["document_order"] = document_order
        record["filename"] = document.get("filename")
        record["segment_count"] = len(list(evidence.get("segments", []) or []))
        record["segment_roles"] = [
            str(segment.get("role") or "").strip()
            for segment in evidence.get("segments", []) or []
            if str(segment.get("role") or "").strip()
        ]
        records.append(record)
    return records


def _build_navigation_index(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sequence_order, sequence in enumerate(document.get("toc_sequences", []) or [], start=1):
        diagnostics = dict(sequence.get("toc_sequence_diagnostics", {}) or {})
        sequence_entries = []
        for entry in sequence.get("entries", []) or []:
            sequence_entries.append(
                {
                    "sequence_entry_index": int(entry.get("sequence_entry_index", 0) or 0),
                    "toc_id": entry.get("toc_id"),
                    "page": int(entry.get("page", 0) or 0),
                    "outline_index": entry.get("outline_index"),
                    "text": entry.get("text"),
                    "page_locator": entry.get("page_locator"),
                    "page_locator_kind": entry.get("page_locator_kind"),
                    "page_locator_value": entry.get("page_locator_value"),
                    "level": int(entry.get("level", 0) or 0),
                    "parent_sequence_entry_index": entry.get("parent_sequence_entry_index"),
                    "parent_toc_id": entry.get("parent_toc_id"),
                    "section_anchor_sequence_entry_index": entry.get("section_anchor_sequence_entry_index"),
                    "review_required": bool(entry.get("review_required", False)),
                    "audit_flags": list(entry.get("audit_flags", []) or []),
                }
            )

        records.append(
            {
                "document_id": document_id,
                "document_order": document_order,
                "filename": document.get("filename"),
                "sequence_order": sequence_order,
                "toc_sequence_id": sequence.get("toc_sequence_id"),
                "title": sequence.get("title"),
                "semantic_role": sequence.get("semantic_role"),
                "toc_ids": list(sequence.get("toc_ids", []) or []),
                "pages": list(sequence.get("pages", []) or []),
                "page_count": int(sequence.get("page_count", 0) or 0),
                "page_span": list(sequence.get("page_span", []) or []),
                "bbox": list(sequence.get("bbox", []) or []),
                "entry_count": int(sequence.get("entry_count", 0) or 0),
                "root_entry_count": int(sequence.get("root_entry_count", 0) or 0),
                "leaf_entry_count": int(sequence.get("leaf_entry_count", 0) or 0),
                "max_branching_factor": int(sequence.get("max_branching_factor", 0) or 0),
                "max_entry_level": int(sequence.get("max_entry_level", 0) or 0),
                "max_outline_depth": int(sequence.get("max_outline_depth", 0) or 0),
                "missing_page_locator_count": int(sequence.get("missing_page_locator_count", 0) or 0),
                "page_locator_kinds": deepcopy(sequence.get("page_locator_kinds", {}) or {}),
                "review_required": bool(sequence.get("review_required", False)),
                "review_item_count": int(diagnostics.get("review_item_count", 0) or 0),
                "aggregated_review_item_count": int(diagnostics.get("aggregated_review_item_count", 0) or 0),
                "mixed_page_numbering": bool(diagnostics.get("mixed_page_numbering", False)),
                "entries": sequence_entries,
                "root_nodes": deepcopy(sequence.get("root_nodes", []) or []),
                "navigation_summary": deepcopy(sequence.get("navigation_summary", {}) or {}),
            }
        )
    return records


def _build_section_index(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
) -> list[dict[str, Any]]:
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    classification = dict(document.get("classification", {}) or infer_document_classification(document, document_order))
    default_module_label = str(classification.get("module_label") or "").strip()

    def _ingest_item(
        item: dict[str, Any],
        *,
        ref_kind: str,
        ref_id: str,
    ) -> None:
        section_context = dict(item.get("section_context", {}) or {})
        if not section_context:
            return
        outline_index = str(section_context.get("outline_index") or "").strip()
        section_title = str(section_context.get("section_title") or "").strip()
        module_label = str(section_context.get("module_label") or default_module_label or "").strip()
        anchor_source = str(section_context.get("anchor_source") or "").strip()
        if not any((outline_index, section_title, module_label)):
            return
        key = (module_label, outline_index, section_title, anchor_source)
        record = records.get(key)
        if record is None:
            record = {
                "document_id": document_id,
                "document_order": document_order,
                "filename": document.get("filename"),
                "module_label": module_label or None,
                "outline_index": outline_index or None,
                "outline_path": str(section_context.get("outline_path") or "").strip() or None,
                "section_title": section_title or None,
                "section_level": section_context.get("section_level"),
                "anchor_source": anchor_source or None,
                "anchor_confidence": section_context.get("anchor_confidence"),
                "anchor_page": section_context.get("anchor_page"),
                "anchor_bbox": deepcopy(section_context.get("anchor_bbox", []) or []),
                "evidence_refs": [],
                "unit_refs": [],
                "pages": [],
            }
            records[key] = record
        page = int(item.get("page", 0) or 0)
        if page > 0 and page not in record["pages"]:
            record["pages"].append(page)
        target_list_key = "evidence_refs" if ref_kind == "evidence" else "unit_refs"
        if ref_id and ref_id not in record[target_list_key]:
            record[target_list_key].append(ref_id)

    for evidence in document.get("content_evidence", []) or []:
        _ingest_item(
            dict(evidence),
            ref_kind="evidence",
            ref_id=str(evidence.get("evidence_id") or "").strip(),
        )

    for unit in document.get("content_units", []) or []:
        _ingest_item(
            dict(unit),
            ref_kind="unit",
            ref_id=str(unit.get("unit_id") or "").strip(),
        )

    output: list[dict[str, Any]] = []
    for record in records.values():
        record["pages"] = sorted(record["pages"])
        if record["pages"]:
            record["page_span"] = [record["pages"][0], record["pages"][-1]]
        else:
            record["page_span"] = [None, None]
        output.append(record)
    output.sort(
        key=lambda item: (
            str(item.get("module_label") or ""),
            int((item.get("page_span") or [0])[0] or 0),
            _section_outline_sort_key(item.get("outline_index")),
            str(item.get("section_title") or ""),
        )
    )
    return output


_WORD_HEADING_RE = re.compile(r"^(?P<outline>\d+(?:\.\d+)*)(?:\.)?\s+(?P<title>\S.*)$")


def _roman_to_int(value: str) -> int:
    numeral_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(str(value or "").strip().upper()):
        current = numeral_values.get(character, 0)
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def _alpha_to_int(value: str) -> int:
    total = 0
    for character in str(value or "").strip().upper():
        if not ("A" <= character <= "Z"):
            return 0
        total = total * 26 + (ord(character) - ord("A") + 1)
    return total


def _section_outline_sort_key(outline_index: str | None) -> tuple[int, Any, str]:
    candidate = str(outline_index or "").strip()
    if not candidate:
        return (99, "", "")
    normalized = candidate.upper()
    if _SECTION_OUTLINE_NUMERIC_RE.fullmatch(candidate):
        return (0, tuple(int(segment) for segment in candidate.split(".")), normalized)
    appendix_match = _SECTION_OUTLINE_APPENDIX_RE.fullmatch(normalized)
    if appendix_match:
        return (1, _alpha_to_int(appendix_match.group(1)), normalized)
    if _SECTION_OUTLINE_ROMAN_RE.fullmatch(normalized):
        return (2, _roman_to_int(normalized), normalized)
    if _SECTION_OUTLINE_ALPHA_RE.fullmatch(normalized):
        return (3, _alpha_to_int(normalized), normalized)
    return (4, normalized, normalized)


def _resolve_existing_parent_outline_index_from_set(
    outline_index: str | None,
    available_outlines: set[str],
) -> str | None:
    for candidate in _candidate_parent_outline_indices(outline_index):
        if candidate in available_outlines:
            return candidate
    return None


def _build_outline_hierarchy_metadata(outline_indices: list[str]) -> dict[str, dict[str, Any]]:
    available_outlines = {
        str(outline_index or "").strip()
        for outline_index in outline_indices
        if str(outline_index or "").strip()
    }
    metadata: dict[str, dict[str, Any]] = {}

    def _resolve(outline_index: str) -> dict[str, Any]:
        normalized = str(outline_index or "").strip()
        if not normalized:
            return {
                "parent_outline_index": None,
                "outline_path": None,
                "tree_depth": 0,
                "ancestor_outline_indices": [],
            }
        if normalized in metadata:
            return metadata[normalized]
        parent_outline_index = _resolve_existing_parent_outline_index_from_set(normalized, available_outlines)
        if parent_outline_index:
            parent_metadata = _resolve(parent_outline_index)
            ancestor_outline_indices = [*list(parent_metadata.get("ancestor_outline_indices", []) or []), parent_outline_index]
            outline_path = " > ".join([*ancestor_outline_indices, normalized])
            tree_depth = int(parent_metadata.get("tree_depth", 0) or 0) + 1
        else:
            ancestor_outline_indices = []
            outline_path = normalized
            tree_depth = 1
        metadata[normalized] = {
            "parent_outline_index": parent_outline_index,
            "outline_path": outline_path,
            "tree_depth": tree_depth,
            "ancestor_outline_indices": ancestor_outline_indices,
        }
        return metadata[normalized]

    for outline_index in sorted(available_outlines, key=_section_outline_sort_key):
        _resolve(outline_index)
    return metadata


def _enrich_structure_outline_hierarchy(
    section_rows: list[dict[str, Any]],
    paragraph_rows: list[dict[str, Any]],
) -> None:
    hierarchy_metadata = _build_outline_hierarchy_metadata(
        [
            *[str(row.get("outline_index") or "").strip() for row in section_rows],
            *[str(row.get("outline_index") or "").strip() for row in paragraph_rows],
        ]
    )

    for row in [*section_rows, *paragraph_rows]:
        outline_index = str(row.get("outline_index") or "").strip()
        if not outline_index:
            continue
        metadata = hierarchy_metadata.get(outline_index)
        if not metadata:
            continue
        row["parent_outline_index"] = metadata.get("parent_outline_index")
        row["outline_path"] = metadata.get("outline_path") or str(row.get("outline_path") or "").strip() or None
        row["tree_depth"] = int(metadata.get("tree_depth", 0) or 0)
        row["ancestor_outline_indices"] = list(metadata.get("ancestor_outline_indices", []) or [])
        if row.get("section_level") in (None, ""):
            row["section_level"] = row["tree_depth"]


def _build_paragraph_index(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
) -> list[dict[str, Any]]:
    source_type = str(document.get("source_type") or "").strip().lower()
    if source_type == "word":
        return _build_word_paragraph_index(document, document_id, document_order)
    return _build_unit_backed_paragraph_index(document, document_id, document_order)


def _build_unit_backed_paragraph_index(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
) -> list[dict[str, Any]]:
    classification = infer_document_classification(document, document_order)
    default_module_label = str(classification.get("module_label") or "").strip() or None
    rows: list[dict[str, Any]] = []
    counter = 0
    for unit in document.get("content_units", []) or []:
        unit_role = str(unit.get("unit_role") or "").strip().lower()
        if unit_role not in {"body", "section_heading"}:
            continue
        text = str(unit.get("text") or "").strip()
        if not text:
            continue
        counter += 1
        section_context = dict(unit.get("section_context", {}) or {})
        rows.append(
            {
                "paragraph_id": f"{document_id}:para_{counter:04d}",
                "document_id": document_id,
                "document_order": document_order,
                "filename": document.get("filename"),
                "source_type": document.get("source_type"),
                "source_ref": str(unit.get("unit_id") or "").strip() or None,
                "module_label": str(section_context.get("module_label") or default_module_label or "").strip() or None,
                "outline_index": str(section_context.get("outline_index") or "").strip() or None,
                "section_title": str(section_context.get("section_title") or "").strip() or None,
                "page": int(unit.get("page", 0) or 0) or None,
                "page_span": (
                    [int(unit.get("page", 0) or 0), int(unit.get("page", 0) or 0)]
                    if int(unit.get("page", 0) or 0) > 0
                    else [None, None]
                ),
                "text": text,
                "paragraph_kind": unit_role,
            }
        )
    return rows


def _build_word_paragraph_index(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
) -> list[dict[str, Any]]:
    classification = infer_document_classification(document, document_order)
    module_label = str(classification.get("module_label") or "").strip() or None
    current_outline_index: str | None = None
    current_section_title: str | None = None
    rows: list[dict[str, Any]] = []

    for paragraph in document.get("paragraphs", []) or []:
        text = str(paragraph.get("text") or "").strip()
        if not text:
            continue
        heading_match = _WORD_HEADING_RE.match(text)
        paragraph_kind = "body"
        if heading_match:
            current_outline_index = str(heading_match.group("outline") or "").strip() or current_outline_index
            current_section_title = str(heading_match.group("title") or "").strip() or current_section_title
            paragraph_kind = "heading"
        rows.append(
            {
                "paragraph_id": f"{document_id}:para_{int(paragraph.get('index', len(rows))) + 1:04d}",
                "document_id": document_id,
                "document_order": document_order,
                "filename": document.get("filename"),
                "source_type": document.get("source_type"),
                "source_ref": None,
                "module_label": module_label,
                "outline_index": current_outline_index,
                "section_title": current_section_title,
                "page": None,
                "page_span": [None, None],
                "text": text,
                "paragraph_kind": paragraph_kind,
            }
        )
    return rows


def _build_section_tree(
    section_index: list[dict[str, Any]],
    paragraph_index: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for paragraph in paragraph_index:
        document_id = str(paragraph.get("document_id") or "").strip()
        outline_index = str(paragraph.get("outline_index") or "").strip()
        section_title = str(paragraph.get("section_title") or "").strip()
        if not document_id or not outline_index:
            continue
        key = (document_id, outline_index)
        node = grouped.get(key)
        if node is None:
            node = {
                "section_node_id": f"{document_id}:sec_{outline_index.replace('.', '_')}",
                "document_id": document_id,
                "filename": paragraph.get("filename"),
                "module_label": paragraph.get("module_label"),
                "outline_index": outline_index,
                "parent_outline_index": paragraph.get("parent_outline_index"),
                "parent_section_node_id": None,
                "outline_path": paragraph.get("outline_path") or outline_index,
                "tree_depth": paragraph.get("tree_depth"),
                "ancestor_outline_indices": list(paragraph.get("ancestor_outline_indices", []) or []),
                "section_title": section_title or None,
                "section_level": paragraph.get("section_level"),
                "page_span": list(paragraph.get("page_span", [None, None])),
                "paragraph_count": 0,
                "paragraph_refs": [],
                "child_outline_indices": [],
                "children": [],
            }
            grouped[key] = node
        node["paragraph_count"] += 1
        paragraph_id = str(paragraph.get("paragraph_id") or "").strip()
        if paragraph_id and paragraph_id not in node["paragraph_refs"]:
            node["paragraph_refs"].append(paragraph_id)
        paragraph_page_span = list(paragraph.get("page_span", [None, None]))
        _merge_page_span(node["page_span"], paragraph_page_span)
        if not node.get("section_title") and section_title:
            node["section_title"] = section_title

    for section_row in section_index:
        document_id = str(section_row.get("document_id") or "").strip()
        outline_index = str(section_row.get("outline_index") or "").strip()
        if not document_id or not outline_index:
            continue
        key = (document_id, outline_index)
        node = grouped.get(key)
        if node is None:
            node = {
                "section_node_id": f"{document_id}:sec_{outline_index.replace('.', '_')}",
                "document_id": document_id,
                "filename": section_row.get("filename"),
                "module_label": section_row.get("module_label"),
                "outline_index": outline_index,
                "parent_outline_index": section_row.get("parent_outline_index"),
                "parent_section_node_id": None,
                "outline_path": section_row.get("outline_path") or outline_index,
                "tree_depth": section_row.get("tree_depth"),
                "ancestor_outline_indices": list(section_row.get("ancestor_outline_indices", []) or []),
                "section_title": section_row.get("section_title"),
                "section_level": section_row.get("section_level"),
                "page_span": list(section_row.get("page_span", [None, None])),
                "paragraph_count": 0,
                "paragraph_refs": [],
                "child_outline_indices": [],
                "children": [],
            }
            grouped[key] = node
        _merge_page_span(node["page_span"], list(section_row.get("page_span", [None, None])))
        if not node.get("section_title"):
            node["section_title"] = section_row.get("section_title")
        if node.get("parent_outline_index") in (None, ""):
            node["parent_outline_index"] = section_row.get("parent_outline_index")
        if not node.get("outline_path"):
            node["outline_path"] = section_row.get("outline_path") or outline_index
        if node.get("tree_depth") in (None, 0):
            node["tree_depth"] = section_row.get("tree_depth")
        if not list(node.get("ancestor_outline_indices", []) or []):
            node["ancestor_outline_indices"] = list(section_row.get("ancestor_outline_indices", []) or [])
        if node.get("section_level") in (None, ""):
            node["section_level"] = section_row.get("section_level")

    children_by_parent: dict[tuple[str, str], list[dict[str, Any]]] = {}
    roots: list[dict[str, Any]] = []
    available_keys = set(grouped.keys())
    for (document_id, outline_index), node in grouped.items():
        parent_outline = _resolve_existing_parent_outline_index(
            document_id,
            outline_index,
            available_keys,
        )
        if parent_outline:
            children_by_parent.setdefault((document_id, parent_outline), []).append(node)
        else:
            roots.append(node)

    for (document_id, outline_index), node in grouped.items():
        child_nodes = sorted(
            children_by_parent.get((document_id, outline_index), []),
            key=lambda item: (
                int((item.get("page_span") or [0])[0] or 0),
                _section_outline_sort_key(item.get("outline_index")),
                str(item.get("section_title") or ""),
            ),
        )
        node["children"] = child_nodes
        node["child_outline_indices"] = [
            str(child.get("outline_index") or "").strip()
            for child in child_nodes
            if str(child.get("outline_index") or "").strip()
        ]
        for child in child_nodes:
            child["parent_section_node_id"] = node["section_node_id"]
            if child.get("parent_outline_index") in (None, ""):
                child["parent_outline_index"] = node.get("outline_index")

    for root in roots:
        _expand_section_tree_page_span_from_descendants(root)

    roots.sort(
        key=lambda item: (
            str(item.get("document_id") or ""),
            int((item.get("page_span") or [0])[0] or 0),
            _section_outline_sort_key(item.get("outline_index")),
            str(item.get("section_title") or ""),
        )
    )
    return roots


def _expand_section_tree_page_span_from_descendants(node: dict[str, Any]) -> list[Any]:
    node_page_span = list(node.get("page_span", [None, None]) or [None, None])
    children = list(node.get("children", []) or [])
    for child in children:
        child_page_span = _expand_section_tree_page_span_from_descendants(child)
        _merge_page_span(node_page_span, child_page_span)
    node["page_span"] = node_page_span
    return node_page_span


def _resolve_existing_parent_outline_index(
    document_id: str,
    outline_index: str | None,
    available_keys: set[tuple[str, str]],
) -> str | None:
    for candidate in _candidate_parent_outline_indices(outline_index):
        if (document_id, candidate) in available_keys:
            return candidate
    return None


def _candidate_parent_outline_indices(outline_index: str | None) -> list[str]:
    candidate = str(outline_index or "").strip()
    if not candidate:
        return []
    segments = [segment for segment in candidate.split(".") if segment]
    if len(segments) <= 1:
        return []

    parents: list[str] = []
    seen: set[str] = set()

    def _append(value: str | None) -> None:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen or normalized == candidate:
            return
        parents.append(normalized)
        seen.add(normalized)

    for depth in range(len(segments) - 1, 0, -1):
        prefix_segments = segments[:depth]
        prefix = ".".join(prefix_segments)
        _append(prefix)
        if len(prefix_segments) == 2 and prefix_segments[1] != "0":
            _append(f"{prefix_segments[0]}.0")
            _append(prefix_segments[0])

    return parents


def _merge_page_span(target: list[Any], incoming: list[Any]) -> None:
    if not target or len(target) < 2 or not incoming or len(incoming) < 2:
        return
    start_candidates = [value for value in [target[0], incoming[0]] if isinstance(value, int)]
    end_candidates = [value for value in [target[1], incoming[1]] if isinstance(value, int)]
    target[0] = min(start_candidates) if start_candidates else None
    target[1] = max(end_candidates) if end_candidates else None


def _count_section_tree_nodes(nodes: list[dict[str, Any]]) -> int:
    return sum(1 + _count_section_tree_nodes(list(node.get("children", []) or [])) for node in nodes)


def _build_fact_index(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
    fact_signals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    atomic_facts = dict(document.get("atomic_facts", {}) or {})
    fact_support_lookup = _build_fact_support_lookup(fact_signals)
    for fact_key in sorted(atomic_facts.keys()):
        fact_value = atomic_facts.get(fact_key)
        supporting_units = _resolve_fact_supporting_units(fact_support_lookup, fact_key, fact_value)
        records.append(
            {
                "document_id": document_id,
                "document_order": document_order,
                "filename": document.get("filename"),
                "fact_key": fact_key,
                "fact_value": fact_value,
                "provenance": {
                    "source": "atomic_facts",
                    "granularity": "unit_level" if supporting_units else "document_level",
                    "supporting_unit_count": len(supporting_units),
                    "supporting_units": supporting_units,
                },
            }
        )
    return records


def _build_fact_signal_index(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for unit in document.get("content_units", []) or []:
        if not bool(unit.get("fact_extraction_eligible", False)):
            continue
        unit_id = str(unit.get("unit_id") or "").strip()
        text = str(unit.get("text") or "").strip()
        if not unit_id or not text:
            continue
        fact_matches = extract_atomic_fact_matches(text)
        for fact_key, match in fact_matches.items():
            records.append(
                {
                    "document_id": document_id,
                    "document_order": document_order,
                    "filename": document.get("filename"),
                    "fact_key": fact_key,
                    "fact_value": match.get("fact_value"),
                    "unit_id": unit_id,
                    "evidence_id": unit.get("evidence_id"),
                    "source_type": unit.get("source_type"),
                    "source_id": unit.get("source_id"),
                    "page": unit.get("page"),
                    "unit_role": unit.get("unit_role"),
                    "matched_text": match.get("matched_text"),
                    "value_span": match.get("value_span"),
                    "text_excerpt": text[:240],
                }
            )
    return records


def _build_fact_support_lookup(
    fact_signals: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for signal in fact_signals:
        fact_key = str(signal.get("fact_key") or "").strip()
        if not fact_key:
            continue
        lookup.setdefault(fact_key, []).append(
            {
                "unit_id": signal.get("unit_id"),
                "evidence_id": signal.get("evidence_id"),
                "source_type": signal.get("source_type"),
                "source_id": signal.get("source_id"),
                "page": signal.get("page"),
                "unit_role": signal.get("unit_role"),
                "matched_value": signal.get("fact_value"),
                "matched_text": signal.get("matched_text"),
                "value_span": signal.get("value_span"),
                "text_excerpt": signal.get("text_excerpt"),
            }
        )
    return lookup


def _resolve_fact_supporting_units(
    fact_support_lookup: dict[str, list[dict[str, Any]]],
    fact_key: Any,
    fact_value: Any,
) -> list[dict[str, Any]]:
    normalized_fact_value = _normalize_fact_value(fact_value)
    if not normalized_fact_value:
        return []
    supporting_units: list[dict[str, Any]] = []
    for unit in fact_support_lookup.get(str(fact_key), []) or []:
        if _normalize_fact_value(unit.get("matched_value")) != normalized_fact_value:
            continue
        supporting_units.append(deepcopy(unit))
    return supporting_units


def _normalize_fact_value(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _build_diagnostic_index(
    document: dict[str, Any],
    document_id: str,
    document_order: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    filename = document.get("filename")

    for table in document.get("table_asts", []) or []:
        table_id = str(table.get("table_id") or "").strip()
        table_page = int(table.get("page", 0) or 0)
        table_bbox = list(table.get("bbox", []) or [])
        table_diagnostics = dict(table.get("diagnostics", {}) or {})

        if table_diagnostics.get("possible_missing_table_content"):
            records.append(
                {
                    "document_id": document_id,
                    "document_order": document_order,
                    "filename": filename,
                    "source_type": "table",
                    "source_id": table_id,
                    "page": table_page,
                    "bbox": table_bbox,
                    "diagnostic_scope": "parser",
                    "code": "possible_missing_table_content",
                    "severity": "medium",
                    "message": "Possible missing table content detected during parsing.",
                    "review_required": bool(table.get("review_required", False)),
                    "details": deepcopy(table_diagnostics),
                }
            )

        for reason_index, reason in enumerate(table.get("review_reasons", []) or [], start=1):
            records.append(
                {
                    "document_id": document_id,
                    "document_order": document_order,
                    "filename": filename,
                    "source_type": "table",
                    "source_id": table_id,
                    "page": table_page,
                    "bbox": table_bbox,
                    "diagnostic_scope": "parser",
                    "code": "table_review_reason",
                    "severity": "medium",
                    "message": str(reason),
                    "review_required": bool(table.get("review_required", False)),
                    "details": {
                        "reason_index": reason_index,
                    },
                }
            )

    for toc_block in document.get("toc_blocks", []) or []:
        toc_id = str(toc_block.get("toc_id") or "").strip()
        toc_page = int(toc_block.get("page", 0) or 0)
        toc_bbox = list(toc_block.get("bbox", []) or [])
        toc_diagnostics = dict(toc_block.get("toc_diagnostics", {}) or {})

        for review_item in toc_diagnostics.get("review_items", []) or []:
            records.append(
                {
                    "document_id": document_id,
                    "document_order": document_order,
                    "filename": filename,
                    "source_type": "toc",
                    "source_id": toc_id,
                    "page": toc_page,
                    "bbox": toc_bbox,
                    "diagnostic_scope": "parser",
                    "code": review_item.get("code"),
                    "severity": review_item.get("severity"),
                    "message": review_item.get("message"),
                    "review_required": bool(toc_diagnostics.get("review_required", False)),
                    "details": deepcopy(review_item),
                }
            )

        for aggregated_item in toc_diagnostics.get("aggregated_review_items", []) or []:
            records.append(
                {
                    "document_id": document_id,
                    "document_order": document_order,
                    "filename": filename,
                    "source_type": "toc",
                    "source_id": toc_id,
                    "page": toc_page,
                    "bbox": toc_bbox,
                    "diagnostic_scope": "parser_aggregated",
                    "code": aggregated_item.get("code"),
                    "severity": aggregated_item.get("severity"),
                    "message": aggregated_item.get("message"),
                    "review_required": bool(toc_diagnostics.get("review_required", False)),
                    "details": deepcopy(aggregated_item),
                }
            )

    return records
