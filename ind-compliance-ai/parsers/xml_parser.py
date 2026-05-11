from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from parsers.common.atomic_fact_extractor import extract_atomic_facts

_REGULATION_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "data" / "regulations"
_DOCTYPE_RE = re.compile(
    r"<!DOCTYPE\s+(?P<name>[^\s>]+)\s+(?:(?:SYSTEM\s+[\"'](?P<system_id>[^\"']+)[\"'])|(?:PUBLIC\s+[\"'](?P<public_id>[^\"']*)[\"']\s+[\"'](?P<public_system_id>[^\"']+)[\"']))",
    re.IGNORECASE,
)


def _strip_xml_namespace(name: str) -> str:
    candidate = str(name or "").strip()
    if "}" in candidate:
        candidate = candidate.split("}", 1)[1]
    if ":" in candidate:
        candidate = candidate.split(":", 1)[1]
    return candidate


def _normalize_xml_name(name: str) -> str:
    stripped = _strip_xml_namespace(name)
    return "".join(ch for ch in stripped.lower() if ch.isalnum())


def _normalize_xml_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _is_local_xml_reference(reference: str) -> bool:
    normalized_reference = str(reference or "").strip().replace("\\", "/")
    if not normalized_reference:
        return False
    if re.match(r"^[a-z][a-z0-9+.-]*:", normalized_reference, re.IGNORECASE):
        return False
    return True


def _xml_reference_points_to_util(reference: str) -> bool:
    normalized_reference = str(reference or "").strip().replace("\\", "/")
    parts = [part for part in normalized_reference.split("/") if part and part != "."]
    return bool(parts) and parts[0].lower() == "util"


def _extract_xml_doctype_metadata(path: Path, xml_text: str) -> dict[str, Any]:
    match = _DOCTYPE_RE.search(xml_text or "")
    if not match:
        return {"xml_doctype_present": False}
    system_id = str(match.group("system_id") or match.group("public_system_id") or "").strip()
    metadata: dict[str, Any] = {
        "xml_doctype_present": True,
        "xml_doctype_name": str(match.group("name") or "").strip(),
        "xml_doctype_system_id": system_id,
    }
    public_id = str(match.group("public_id") or "").strip()
    if public_id:
        metadata["xml_doctype_public_id"] = public_id
    if not system_id:
        return metadata
    normalized_system_id = system_id.replace("\\", "/")
    metadata["xml_doctype_system_id_normalized"] = normalized_system_id
    if re.match(r"^[a-z][a-z0-9+.-]*:", normalized_system_id, re.IGNORECASE):
        metadata["xml_dtd_system_id_is_local_path"] = False
        return metadata
    resolved_path = (path.parent / normalized_system_id).resolve(strict=False)
    metadata.update(
        {
            "xml_dtd_system_id_is_local_path": True,
            "xml_dtd_resolved_path": str(resolved_path),
            "xml_dtd_resolved_path_exists": resolved_path.exists(),
            "xml_dtd_resolved_filename": resolved_path.name,
        }
    )
    return metadata


def _extract_xml_schema_location_metadata(path: Path, root: ET.Element | None) -> dict[str, Any]:
    if root is None:
        return {}

    records: list[dict[str, Any]] = []
    raw_values: list[str] = []
    for attr_name, attr_value in root.attrib.items():
        stripped_name = _strip_xml_namespace(attr_name)
        normalized_name = _normalize_xml_name(stripped_name)
        if normalized_name not in {"schemalocation", "nonamespaceschemalocation"}:
            continue
        raw_value = _normalize_xml_text(attr_value)
        if not raw_value:
            continue
        raw_values.append(raw_value)
        attr_namespace = ""
        if str(attr_name).startswith("{") and "}" in str(attr_name):
            attr_namespace = str(attr_name)[1:].split("}", 1)[0]
        tokens = raw_value.split()
        if normalized_name == "nonamespaceschemalocation":
            location_pairs = [("", token) for token in tokens]
        else:
            location_pairs = [
                (tokens[index], tokens[index + 1])
                for index in range(0, len(tokens) - 1, 2)
            ]
            if len(tokens) % 2 == 1:
                location_pairs.append(("", tokens[-1]))

        for namespace, location in location_pairs:
            normalized_location = location.replace("\\", "/")
            is_local_path = _is_local_xml_reference(normalized_location)
            resolved_path = (path.parent / normalized_location).resolve(strict=False) if is_local_path else None
            record: dict[str, Any] = {
                "attribute_name": stripped_name,
                "attribute_namespace": attr_namespace,
                "namespace": namespace,
                "schema_location": location,
                "schema_location_normalized": normalized_location,
                "schema_resolved_path": str(resolved_path) if resolved_path is not None else "",
                "schema_resolved_path_exists": bool(resolved_path and resolved_path.exists()),
                "schema_resolved_filename": resolved_path.name if resolved_path is not None else "",
                "schema_location_is_local_path": is_local_path,
                "schema_location_points_to_util": _xml_reference_points_to_util(normalized_location),
            }
            records.append(record)

    if not records:
        return {}
    return {
        "xml_schema_location_raw": " ".join(raw_values),
        "xml_schema_location_count": len(records),
        "xml_schema_location_records": records,
        "xml_schema_locations_all_local": all(bool(record["schema_location_is_local_path"]) for record in records),
        "xml_schema_locations_all_resolve": all(bool(record["schema_resolved_path_exists"]) for record in records),
        "xml_schema_locations_all_point_to_util": all(bool(record["schema_location_points_to_util"]) for record in records),
    }


def _build_xml_well_formedness_metadata(
    *,
    is_well_formed: bool,
    parse_error: ET.ParseError | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"xml_is_well_formed": is_well_formed}
    if parse_error is not None:
        metadata["xml_parse_error"] = str(parse_error)
        if getattr(parse_error, "position", None):
            line, column = parse_error.position
            metadata["xml_parse_error_line"] = int(line)
            metadata["xml_parse_error_column"] = int(column)
    return metadata


def _build_xml_dtd_validation_metadata(path: Path, xml_text: str, xml_metadata: dict[str, Any]) -> dict[str, Any]:
    if not bool(xml_metadata.get("xml_is_well_formed")):
        return {
            "xml_dtd_validation_attempted": False,
            "xml_dtd_validation_prerequisite_missing": "xml_not_well_formed",
        }
    dtd_path_value = str(xml_metadata.get("xml_dtd_resolved_path") or "").strip()
    if not dtd_path_value:
        return {
            "xml_dtd_validation_attempted": False,
            "xml_dtd_validation_prerequisite_missing": "doctype_system_id_missing_or_nonlocal",
        }
    dtd_path = Path(dtd_path_value)
    if not dtd_path.exists():
        return {
            "xml_dtd_validation_attempted": False,
            "xml_dtd_validation_prerequisite_missing": "dtd_file_missing",
        }
    try:
        from lxml import etree as lxml_etree
    except ImportError:
        return {
            "xml_dtd_validation_attempted": False,
            "xml_dtd_validation_prerequisite_missing": "lxml_unavailable",
        }

    try:
        parser = lxml_etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False)
        document = lxml_etree.fromstring(xml_text.encode("utf-8"), parser=parser)
        dtd = lxml_etree.DTD(str(dtd_path))
        is_valid = bool(dtd.validate(document))
        validation_errors = [
            {
                "line": int(error.line or 0),
                "column": int(error.column or 0),
                "message": str(error.message or "").strip(),
                "domain": str(error.domain_name or "").strip(),
                "type": str(error.type_name or "").strip(),
                "level": str(error.level_name or "").strip(),
            }
            for error in dtd.error_log.filter_from_errors()
        ]
        return {
            "xml_dtd_validation_attempted": True,
            "xml_dtd_is_valid": is_valid,
            "xml_dtd_validation_error_count": len(validation_errors),
            "xml_dtd_validation_errors": validation_errors,
        }
    except Exception as exc:
        return {
            "xml_dtd_validation_attempted": False,
            "xml_dtd_validation_prerequisite_missing": "dtd_validation_exception",
            "xml_dtd_validation_error": str(exc),
        }


def _build_xml_schema_validation_metadata(
    path: Path,
    xml_text: str,
    xml_metadata: dict[str, Any],
) -> dict[str, Any]:
    if not bool(xml_metadata.get("xml_is_well_formed")):
        return {
            "xml_schema_validation_attempted": False,
            "xml_schema_validation_prerequisite_missing": "xml_not_well_formed",
        }

    records = [dict(record or {}) for record in list(xml_metadata.get("xml_schema_location_records") or [])]
    if not records:
        return {
            "xml_schema_validation_attempted": False,
            "xml_schema_validation_prerequisite_missing": "schema_location_missing",
        }

    schema_record = next(
        (
            record
            for record in records
            if bool(record.get("schema_location_is_local_path"))
            and bool(record.get("schema_resolved_path_exists"))
            and str(record.get("schema_resolved_path") or "").strip()
        ),
        {},
    )
    schema_path_value = str(schema_record.get("schema_resolved_path") or "").strip()
    if not schema_path_value:
        return {
            "xml_schema_validation_attempted": False,
            "xml_schema_validation_prerequisite_missing": "schema_file_missing_or_nonlocal",
        }

    schema_path = Path(schema_path_value)
    try:
        from lxml import etree as lxml_etree
    except ImportError:
        return {
            "xml_schema_validation_attempted": False,
            "xml_schema_validation_prerequisite_missing": "lxml_unavailable",
            "xml_schema_validation_schema_path": str(schema_path),
        }

    try:
        schema_document = lxml_etree.parse(str(schema_path))
        schema = lxml_etree.XMLSchema(schema_document)
    except Exception as exc:
        return {
            "xml_schema_validation_attempted": False,
            "xml_schema_validation_prerequisite_missing": "schema_compile_exception",
            "xml_schema_validation_schema_path": str(schema_path),
            "xml_schema_validation_error": str(exc),
        }

    try:
        parser = lxml_etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False)
        document = lxml_etree.fromstring(xml_text.encode("utf-8"), parser=parser)
        is_valid = bool(schema.validate(document))
        validation_errors = [
            {
                "line": int(error.line or 0),
                "column": int(error.column or 0),
                "message": str(error.message or "").strip(),
                "domain": str(error.domain_name or "").strip(),
                "type": str(error.type_name or "").strip(),
                "level": str(error.level_name or "").strip(),
            }
            for error in schema.error_log.filter_from_errors()
        ]
        return {
            "xml_schema_validation_attempted": True,
            "xml_schema_validation_schema_path": str(schema_path),
            "xml_schema_is_valid": is_valid,
            "xml_schema_validation_error_count": len(validation_errors),
            "xml_schema_validation_errors": validation_errors,
            "xml_schema_validation_prerequisite_missing": "",
            "xml_schema_validation_error": "",
        }
    except Exception as exc:
        return {
            "xml_schema_validation_attempted": False,
            "xml_schema_validation_prerequisite_missing": "schema_validation_exception",
            "xml_schema_validation_schema_path": str(schema_path),
            "xml_schema_validation_error": str(exc),
        }


def _iter_child_elements(node: ET.Element) -> list[ET.Element]:
    return [child for child in list(node) if isinstance(child.tag, str)]


def _get_xml_lang(attributes: dict[str, Any]) -> str:
    for attr_name, attr_value in attributes.items():
        normalized_name = _normalize_xml_name(attr_name)
        if normalized_name != "lang":
            continue
        normalized_value = _normalize_xml_text(attr_value)
        if normalized_value:
            return normalized_value
    return ""


def _walk_xml(node: ET.Element, pointer: str, nodes: list[dict[str, Any]]) -> None:
    raw_text = node.text or ""
    text = " ".join(raw_text.split())
    local_tag = _strip_xml_namespace(node.tag)
    nodes.append(
        {
            "pointer": pointer,
            "tag": node.tag,
            "local_tag": local_tag,
            "attributes": dict(node.attrib),
            "text": text,
            "raw_text": raw_text,
        }
    )
    for child_index, child in enumerate(list(node)):
        child_pointer = f"{pointer}/{child.tag}[{child_index}]"
        _walk_xml(child, child_pointer, nodes)


def _first_matching_attribute(
    nodes: list[dict[str, Any]],
    *,
    tag_names: set[str],
    attribute_aliases: set[str],
) -> str | None:
    for node in nodes:
        if _normalize_xml_name(node.get("local_tag", "")) not in tag_names:
            continue
        attributes = dict(node.get("attributes", {}) or {})
        for attr_name, attr_value in attributes.items():
            if _normalize_xml_name(attr_name) not in attribute_aliases:
                continue
            normalized_value = " ".join(str(attr_value or "").split()).strip()
            if normalized_value:
                return normalized_value
    return None


def _collect_matching_attributes(
    nodes: list[dict[str, Any]],
    *,
    tag_names: set[str],
    attribute_aliases: set[str],
) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        if _normalize_xml_name(node.get("local_tag", "")) not in tag_names:
            continue
        attributes = dict(node.get("attributes", {}) or {})
        for attr_name, attr_value in attributes.items():
            if _normalize_xml_name(attr_name) not in attribute_aliases:
                continue
            normalized_value = " ".join(str(attr_value or "").split()).strip()
            if not normalized_value or normalized_value in seen:
                continue
            values.append(normalized_value)
            seen.add(normalized_value)
    return values


def _extract_controlled_vocabulary_metadata(
    path: Path,
    root: ET.Element | None,
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    filename = str(path.name or "").strip().lower()
    if not filename.startswith("cv-") or not filename.endswith(".xml"):
        return {}

    if root is not None and _normalize_xml_name(root.tag) == "controlledvocabulary":
        vocabulary_name = _normalize_xml_text(root.attrib.get("name"))
        versions_payload: list[dict[str, Any]] = []
        values: list[str] = []
        entries: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        version_nodes = [
            child for child in _iter_child_elements(root) if _normalize_xml_name(child.tag) == "version"
        ]
        for version_node in version_nodes:
            version_number = _normalize_xml_text(version_node.attrib.get("number"))
            valid_from = _normalize_xml_text(version_node.attrib.get("valid-from"))
            valid_to = _normalize_xml_text(version_node.attrib.get("valid-to"))
            code_entries: list[dict[str, Any]] = []
            for code_node in _iter_child_elements(version_node):
                if _normalize_xml_name(code_node.tag) != "code":
                    continue
                code_name = _normalize_xml_text(code_node.attrib.get("name"))
                if not code_name:
                    continue
                descriptions: dict[str, str] = {}
                for desc_node in _iter_child_elements(code_node):
                    if _normalize_xml_name(desc_node.tag) != "description":
                        continue
                    lang = _get_xml_lang(desc_node.attrib)
                    text_value = _normalize_xml_text(desc_node.text)
                    if lang and text_value:
                        descriptions[lang] = text_value
                code_entry = {
                    "code": code_name,
                    "descriptions": descriptions,
                }
                code_entries.append(code_entry)
                if code_name not in seen_codes:
                    values.append(code_name)
                    seen_codes.add(code_name)
                    entries.append(code_entry)
            versions_payload.append(
                {
                    "number": version_number,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "code_count": len(code_entries),
                    "entries": code_entries,
                }
            )

        if values:
            metadata = {
                "ectd_controlled_vocabulary_name": path.stem,
                "ectd_controlled_vocabulary_key": vocabulary_name,
                "ectd_controlled_vocabulary_values": values,
                "ectd_controlled_vocabulary_code_count": len(values),
                "ectd_controlled_vocabulary_entries": entries,
                "ectd_controlled_vocabulary_versions": versions_payload,
            }
            if versions_payload:
                metadata["ectd_controlled_vocabulary_version"] = versions_payload[0]["number"]
                metadata["ectd_controlled_vocabulary_valid_from"] = versions_payload[0]["valid_from"]
                if versions_payload[0]["valid_to"]:
                    metadata["ectd_controlled_vocabulary_valid_to"] = versions_payload[0]["valid_to"]
            return metadata

    values: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        local_tag = _normalize_xml_name(node.get("local_tag", ""))
        attributes = dict(node.get("attributes", {}) or {})
        for attr_name, attr_value in attributes.items():
            if _normalize_xml_name(attr_name) not in {"code", "value", "name", "id"}:
                continue
            if local_tag in {"controlledvocabulary", "dependency"}:
                continue
            normalized_value = " ".join(str(attr_value or "").split()).strip()
            if not normalized_value or normalized_value in seen:
                continue
            values.append(normalized_value)
            seen.add(normalized_value)
        text_value = " ".join(str(node.get("text") or "").split()).strip()
        if (
            local_tag in {"code", "value", "name", "id"}
            and text_value
            and text_value not in seen
        ):
            values.append(text_value)
            seen.add(text_value)
    if not values:
        return {}
    return {
        "ectd_controlled_vocabulary_name": path.stem,
        "ectd_controlled_vocabulary_values": values,
    }


def _extract_dependency_matrix_metadata(
    path: Path,
    root: ET.Element | None,
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    filename = str(path.name or "").strip().lower()
    if filename != "depend-apt-rat-sqt.xml":
        return {}

    if root is not None and _normalize_xml_name(root.tag) == "dependency":
        version_nodes = [
            child for child in _iter_child_elements(root) if _normalize_xml_name(child.tag) == "version"
        ]
        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        versions_payload: list[dict[str, Any]] = []
        for version_node in version_nodes:
            version_number = _normalize_xml_text(version_node.attrib.get("number"))
            valid_from = _normalize_xml_text(version_node.attrib.get("valid-from"))
            valid_to = _normalize_xml_text(version_node.attrib.get("valid-to"))
            version_row_count = 0
            for application_node in _iter_child_elements(version_node):
                if _normalize_xml_name(application_node.tag) != "code":
                    continue
                application_type = _normalize_xml_text(application_node.attrib.get("name"))
                if not application_type:
                    continue
                for regulatory_node in _iter_child_elements(application_node):
                    if _normalize_xml_name(regulatory_node.tag) != "code":
                        continue
                    regulatory_activity_type = _normalize_xml_text(regulatory_node.attrib.get("name"))
                    if not regulatory_activity_type:
                        continue
                    for sequence_node in _iter_child_elements(regulatory_node):
                        if _normalize_xml_name(sequence_node.tag) != "code":
                            continue
                        sequence_type = _normalize_xml_text(sequence_node.attrib.get("name"))
                        if not sequence_type:
                            continue
                        row = {
                            "application_type": application_type,
                            "regulatory_activity_type": regulatory_activity_type,
                            "sequence_type": sequence_type,
                        }
                        row_key = (
                            application_type,
                            regulatory_activity_type,
                            sequence_type,
                        )
                        if row_key in seen:
                            continue
                        seen.add(row_key)
                        rows.append(row)
                        version_row_count += 1
            versions_payload.append(
                {
                    "number": version_number,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "row_count": version_row_count,
                }
            )

        if rows:
            metadata = {
                "ectd_dependency_matrix_name": path.stem,
                "ectd_dependency_matrix_rows": rows,
                "ectd_dependency_matrix_row_count": len(rows),
                "ectd_dependency_matrix_versions": versions_payload,
            }
            if versions_payload:
                metadata["ectd_dependency_matrix_version"] = versions_payload[0]["number"]
                metadata["ectd_dependency_matrix_valid_from"] = versions_payload[0]["valid_from"]
                if versions_payload[0]["valid_to"]:
                    metadata["ectd_dependency_matrix_valid_to"] = versions_payload[0]["valid_to"]
            return metadata

    field_aliases = {
        "application_type": {"applicationtype", "application", "apt"},
        "regulatory_activity_type": {"regulatoryactivitytype", "regulatoryactivity", "rat"},
        "sequence_type": {"sequencetype", "sequence", "sqt"},
    }

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for scope_node in nodes:
        scope_pointer = str(scope_node.get("pointer") or "")
        related_nodes = [
            scope_node,
            *[
                node
                for node in nodes
                if scope_pointer and str(node.get("pointer") or "").startswith(f"{scope_pointer}/")
            ],
        ]
        candidate: dict[str, str] = {}
        for node in related_nodes:
            local_tag = _normalize_xml_name(node.get("local_tag", ""))
            text_value = " ".join(str(node.get("text") or "").split()).strip()
            attributes = dict(node.get("attributes", {}) or {})
            for field_name, aliases in field_aliases.items():
                if field_name in candidate:
                    continue
                if local_tag in aliases and text_value:
                    candidate[field_name] = text_value
                    continue
                for attr_name, attr_value in attributes.items():
                    if _normalize_xml_name(attr_name) not in aliases:
                        continue
                    normalized_value = " ".join(str(attr_value or "").split()).strip()
                    if normalized_value:
                        candidate[field_name] = normalized_value
                        break

        if len(candidate) != 3:
            continue
        row_key = (
            candidate["application_type"],
            candidate["regulatory_activity_type"],
            candidate["sequence_type"],
        )
        if row_key in seen:
            continue
        seen.add(row_key)
        rows.append(candidate)

    if not rows:
        return {}
    return {
        "ectd_dependency_matrix_name": path.stem,
        "ectd_dependency_matrix_rows": rows,
    }


def _extract_xml_schema_metadata(path: Path, root: ET.Element | None) -> dict[str, Any]:
    filename = str(path.name or "").strip().lower()
    if filename.endswith(".xsd"):
        schema_name = path.stem
    elif root is not None and _normalize_xml_name(root.tag) == "schema":
        schema_name = path.stem
    else:
        return {}

    if root is None or _normalize_xml_name(root.tag) != "schema":
        return {"xml_schema_name": schema_name}

    imports: list[dict[str, str]] = []
    named_elements: list[str] = []
    complex_type_count = 0
    element_count = 0
    seen_named_elements: set[str] = set()
    for node in root.iter():
        local_tag = _normalize_xml_name(node.tag)
        if local_tag == "import":
            imports.append(
                {
                    "namespace": _normalize_xml_text(node.attrib.get("namespace")),
                    "schema_location": _normalize_xml_text(node.attrib.get("schemaLocation")),
                }
            )
        elif local_tag == "element":
            element_count += 1
            name = _normalize_xml_text(node.attrib.get("name"))
            if name and name not in seen_named_elements:
                named_elements.append(name)
                seen_named_elements.add(name)
        elif local_tag == "complextype":
            complex_type_count += 1

    return {
        "xml_schema_name": schema_name,
        "xml_schema_target_namespace": _normalize_xml_text(root.attrib.get("targetNamespace")),
        "xml_schema_imports": imports,
        "xml_schema_import_count": len(imports),
        "xml_schema_element_count": element_count,
        "xml_schema_complex_type_count": complex_type_count,
        "xml_schema_named_elements": named_elements,
    }


def _resolve_schema_dependency_path(schema_path: Path, location: str) -> Path | None:
    normalized_location = _normalize_xml_text(location)
    if not normalized_location or "://" in normalized_location:
        return None

    direct_candidate = schema_path.parent / normalized_location
    if direct_candidate.exists():
        return direct_candidate

    location_path = Path(normalized_location)
    fallback_name = location_path.name
    if not fallback_name or not _REGULATION_SOURCE_ROOT.exists():
        return None

    matches = sorted(_REGULATION_SOURCE_ROOT.rglob(fallback_name))
    if not matches:
        return None
    return matches[0]


def _audit_xml_schema_dependencies(path: Path, root: ET.Element | None) -> dict[str, Any]:
    if root is None:
        return {}

    filename = str(path.name or "").strip().lower()
    primary_schema_name = ""
    for attr_name, attr_value in root.attrib.items():
        if _normalize_xml_name(attr_name) == "nonamespaceschemalocation":
            primary_schema_name = _normalize_xml_text(attr_value).split()[-1]
            break

    if not primary_schema_name:
        if filename.startswith("cv-") and filename.endswith(".xml"):
            primary_schema_name = "cn-cv.xsd"
        elif filename == "depend-apt-rat-sqt.xml":
            primary_schema_name = "cn-dependency.xsd"
        elif filename.endswith(".xsd"):
            primary_schema_name = path.name

    if not primary_schema_name:
        return {}

    primary_schema_path = path.parent / primary_schema_name
    dependency_files: list[str] = []
    missing_dependencies: list[str] = []
    visited: set[Path] = set()

    def _walk_schema(schema_path: Path) -> None:
        resolved_path = schema_path.resolve()
        if resolved_path in visited:
            return
        visited.add(resolved_path)
        if not schema_path.exists():
            name = schema_path.name
            if name not in missing_dependencies:
                missing_dependencies.append(name)
            return
        dependency_files.append(str(schema_path))
        try:
            schema_root = ET.fromstring(schema_path.read_text(encoding="utf-8", errors="ignore"))
        except ET.ParseError:
            name = schema_path.name
            if name not in missing_dependencies:
                missing_dependencies.append(name)
            return
        for node in schema_root.iter():
            if _normalize_xml_name(node.tag) not in {"import", "include"}:
                continue
            location = _normalize_xml_text(node.attrib.get("schemaLocation"))
            if not location or "://" in location:
                continue
            dependency_path = _resolve_schema_dependency_path(schema_path, location)
            if dependency_path is None:
                if Path(location).name not in missing_dependencies:
                    missing_dependencies.append(Path(location).name)
                continue
            _walk_schema(dependency_path)

    _walk_schema(primary_schema_path)
    return {
        "xml_schema_primary_path": str(primary_schema_path),
        "xml_schema_dependency_files": dependency_files,
        "xml_schema_missing_dependencies": missing_dependencies,
        "xml_schema_dependency_status": "ready" if not missing_dependencies else "missing_dependencies",
    }


def _extract_sequence_contact_metadata(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    scope_nodes = [
        node
        for node in nodes
        if _normalize_xml_name(node.get("local_tag", "")) == "sequencecontact"
    ]
    if not scope_nodes:
        return {}

    field_aliases = {
        "name": {"name", "contactname", "contactperson", "personname"},
        "phone": {"telephone", "phone", "phonenumber", "tel"},
        "email": {"email", "emailaddress", "mail"},
    }

    contact_values: dict[str, str] = {}
    for scope_node in scope_nodes:
        scope_pointer = str(scope_node.get("pointer") or "")
        related_nodes = [
            scope_node,
            *[
                node
                for node in nodes
                if str(node.get("pointer") or "").startswith(f"{scope_pointer}/")
            ],
        ]
        for node in related_nodes:
            local_tag = _normalize_xml_name(node.get("local_tag", ""))
            text_value = " ".join(str(node.get("text") or "").split()).strip()
            attributes = dict(node.get("attributes", {}) or {})
            for field_name, aliases in field_aliases.items():
                if field_name in contact_values:
                    continue
                if local_tag in aliases and text_value:
                    contact_values[field_name] = text_value
                    continue
                for attr_name, attr_value in attributes.items():
                    if _normalize_xml_name(attr_name) not in aliases:
                        continue
                    normalized_value = " ".join(str(attr_value or "").split()).strip()
                    if normalized_value:
                        contact_values[field_name] = normalized_value
                        break

    if not contact_values:
        return {}

    metadata = {
        "ectd_sequence_contact": contact_values,
    }
    if contact_values.get("name"):
        metadata["ectd_sequence_contact_name"] = contact_values["name"]
    if contact_values.get("phone"):
        metadata["ectd_sequence_contact_phone"] = contact_values["phone"]
    if contact_values.get("email"):
        metadata["ectd_sequence_contact_email"] = contact_values["email"]
    return metadata


def _extract_sequence_directory_number(path: Path) -> str | None:
    parts = [str(part or "").strip() for part in path.parts]
    lowered_parts = [part.lower() for part in parts]
    for index, part in enumerate(parts[:-1]):
        if len(part) == 4 and part.isdigit():
            next_part = lowered_parts[index + 1] if index + 1 < len(lowered_parts) else ""
            if next_part in {"m1", "m2", "m3", "m4", "m5"}:
                return part
    return None


def _extract_node_extension_metadata(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    node_lookup = {
        str(node.get("pointer") or "").strip(): dict(node)
        for node in nodes
        if str(node.get("pointer") or "").strip()
    }
    regional_32r_nodes = [
        dict(node)
        for node in nodes
        if _normalize_xml_name(node.get("local_tag", "")) == "m32rregionalinformation"
    ]

    def _parent_pointer(pointer: str) -> str:
        candidate = str(pointer or "").strip()
        if "/" not in candidate:
            return ""
        return candidate.rsplit("/", 1)[0]

    def _is_32r_ancestor(pointer: str) -> tuple[bool, str]:
        candidate = str(pointer or "").strip()
        matching_pointers = [
            ancestor_pointer
            for ancestor_pointer, ancestor_node in node_lookup.items()
            if _normalize_xml_name(ancestor_node.get("local_tag", "")) == "m32rregionalinformation"
            and candidate.startswith(f"{ancestor_pointer}/")
        ]
        if not matching_pointers:
            return False, ""
        return True, max(matching_pointers, key=len)

    def _extract_href(attributes: dict[str, Any]) -> str:
        for attr_name, attr_value in attributes.items():
            if _normalize_xml_name(attr_name) != "href":
                continue
            normalized_value = " ".join(str(attr_value or "").split()).strip()
            if normalized_value:
                return normalized_value
        return ""

    extension_records: list[dict[str, Any]] = []
    extension_titles_32r: list[str] = []
    extension_leaf_hrefs_32r: list[str] = []

    for node in nodes:
        if _normalize_xml_name(node.get("local_tag", "")) != "nodeextension":
            continue
        pointer = str(node.get("pointer") or "").strip()
        parent_pointer = _parent_pointer(pointer)
        parent_node = node_lookup.get(parent_pointer, {})
        subtree_nodes = [
            dict(candidate)
            for candidate in nodes
            if str(candidate.get("pointer") or "").startswith(f"{pointer}/")
        ]

        extension_title = ""
        leaf_hrefs: list[str] = []
        leaf_titles: list[str] = []
        for subtree_node in subtree_nodes:
            subtree_pointer = str(subtree_node.get("pointer") or "").strip()
            subtree_parent_pointer = _parent_pointer(subtree_pointer)
            if (
                _normalize_xml_name(subtree_node.get("local_tag", "")) == "title"
                and subtree_parent_pointer == pointer
                and not extension_title
            ):
                extension_title = " ".join(str(subtree_node.get("text") or "").split()).strip()
                continue
            if _normalize_xml_name(subtree_node.get("local_tag", "")) != "leaf":
                continue
            href = _extract_href(dict(subtree_node.get("attributes", {}) or {}))
            if href:
                leaf_hrefs.append(href)
            leaf_pointer = subtree_pointer
            for leaf_child in subtree_nodes:
                if _parent_pointer(str(leaf_child.get("pointer") or "").strip()) != leaf_pointer:
                    continue
                if _normalize_xml_name(leaf_child.get("local_tag", "")) != "title":
                    continue
                title_text = " ".join(str(leaf_child.get("text") or "").split()).strip()
                if title_text:
                    leaf_titles.append(title_text)

        is_32r, _ = _is_32r_ancestor(pointer)
        record = {
            "extension_title": extension_title,
            "parent_local_tag": str(parent_node.get("local_tag") or "").strip(),
            "parent_pointer": parent_pointer,
            "leaf_hrefs": leaf_hrefs,
            "leaf_titles": leaf_titles,
            "is_within_32r_scope": is_32r,
        }
        extension_records.append(record)

        if is_32r:
            if extension_title:
                extension_titles_32r.append(extension_title)
            extension_leaf_hrefs_32r.extend(href for href in leaf_hrefs if href)

    metadata: dict[str, Any] = {}
    if regional_32r_nodes:
        metadata["ectd_32r_regional_information_present"] = True
        metadata["ectd_32r_regional_information_count"] = len(regional_32r_nodes)
    if extension_records:
        metadata["ectd_node_extension_records"] = extension_records
    if extension_titles_32r:
        metadata["ectd_32r_extension_titles"] = extension_titles_32r
        metadata["ectd_32r_extension_count"] = len(extension_titles_32r)
    if extension_leaf_hrefs_32r:
        metadata["ectd_32r_extension_leaf_hrefs"] = extension_leaf_hrefs_32r
    return metadata


def _extract_ectd_metadata(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    envelope_tags = {"envelope", "cnenvelope"}
    leaf_tags = {"leaf"}
    metadata: dict[str, Any] = {}
    envelope_attributes: dict[str, str] = {}
    envelope_count = 0
    leaf_records: list[dict[str, str]] = []
    leaf_title_records: list[dict[str, Any]] = []
    leaf_lifecycle_records: list[dict[str, Any]] = []
    node_lookup = {
        str(node.get("pointer") or "").strip(): dict(node)
        for node in nodes
        if str(node.get("pointer") or "").strip()
    }

    def _parent_pointer(pointer: str) -> str:
        candidate = str(pointer or "").strip()
        if "/" not in candidate:
            return ""
        return candidate.rsplit("/", 1)[0]

    def _ancestor_nodes(pointer: str) -> list[dict[str, Any]]:
        ancestors: list[dict[str, Any]] = []
        candidate = str(pointer or "").strip()
        while candidate:
            node = node_lookup.get(candidate)
            if node:
                ancestors.append(node)
            parent = _parent_pointer(candidate)
            if not parent or parent == candidate:
                break
            candidate = parent
        return list(reversed(ancestors))

    envelope_field_aliases = {
        "application-number": {"applicationnumber", "applicationid", "applicationno"},
        "application-type": {"applicationtype", "application"},
        "product-type": {"producttype", "product"},
        "related-sequence": {"relatedsequence", "relatedsequencenumber", "previoussequencenumber", "previoussequence"},
        "regulatory-activity-type": {"regulatoryactivitytype", "regulatoryactivity"},
        "sequence-number": {"sequencenumber", "seqnumber"},
        "sequence-type": {"sequencetype", "sequence", "sqt"},
        "sequence-description": {"sequencedescription", "submissiondescription", "description"},
    }

    def _set_envelope_attribute_if_missing(canonical_name: str, raw_value: Any) -> None:
        normalized_value = " ".join(str(raw_value or "").split()).strip()
        if canonical_name and normalized_value and canonical_name not in envelope_attributes:
            envelope_attributes[canonical_name] = normalized_value

    for node in nodes:
        if _normalize_xml_name(node.get("local_tag", "")) not in envelope_tags:
            pass
        else:
            envelope_count += 1
            attributes = dict(node.get("attributes", {}) or {})
            for attr_name, attr_value in attributes.items():
                normalized_name = _strip_xml_namespace(attr_name)
                _set_envelope_attribute_if_missing(normalized_name, attr_value)

            scope_pointer = str(node.get("pointer") or "").strip()
            related_nodes = [
                node,
                *[
                    child_node
                    for child_node in nodes
                    if scope_pointer and str(child_node.get("pointer") or "").startswith(f"{scope_pointer}/")
                ],
            ]
            for canonical_name, aliases in envelope_field_aliases.items():
                if canonical_name in envelope_attributes:
                    continue
                for candidate_node in related_nodes:
                    local_tag = _normalize_xml_name(candidate_node.get("local_tag", ""))
                    if local_tag not in aliases:
                        continue
                    candidate_attributes = dict(candidate_node.get("attributes", {}) or {})
                    if canonical_name in {
                        "application-type",
                        "product-type",
                        "regulatory-activity-type",
                        "sequence-type",
                    }:
                        if "code" in candidate_attributes:
                            _set_envelope_attribute_if_missing(canonical_name, candidate_attributes.get("code"))
                    text_value = " ".join(str(candidate_node.get("text") or "").split()).strip()
                    if text_value:
                        _set_envelope_attribute_if_missing(canonical_name, text_value)
                    if canonical_name in envelope_attributes:
                        break

        if _normalize_xml_name(node.get("local_tag", "")) in leaf_tags:
            attributes = dict(node.get("attributes", {}) or {})
            pointer = str(node.get("pointer") or "").strip()
            parent_pointer = _parent_pointer(pointer)
            parent_node = node_lookup.get(parent_pointer, {})
            href = ""
            checksum_type = ""
            checksum = ""
            operation = ""
            operation_present = False
            href_present = False
            xml_lang = ""
            xml_lang_present = False
            leaf_title = ""
            raw_leaf_title = ""
            leaf_title_present = False
            modified_file_href = ""
            modified_file_present = False
            modified_file_pointer = ""
            for attr_name, attr_value in attributes.items():
                normalized_key = _normalize_xml_name(attr_name)
                normalized_value = " ".join(str(attr_value or "").split()).strip()
                if normalized_key == "href":
                    href_present = True
                    href = normalized_value
                elif normalized_key == "checksumtype":
                    checksum_type = normalized_value
                elif normalized_key == "checksum":
                    checksum = normalized_value
                elif normalized_key == "operation":
                    operation_present = True
                    operation = normalized_value
                elif normalized_key == "lang":
                    xml_lang_present = True
                    xml_lang = normalized_value
                elif normalized_key == "modifiedfile":
                    modified_file_present = True
                    modified_file_href = normalized_value
            for child in nodes:
                child_pointer = str(child.get("pointer") or "").strip()
                if _parent_pointer(child_pointer) != pointer:
                    continue
                if _normalize_xml_name(child.get("local_tag", "")) != "title":
                    continue
                leaf_title_present = True
                raw_leaf_title = str(child.get("raw_text") if "raw_text" in child else child.get("text") or "")
                leaf_title = " ".join(raw_leaf_title.split()).strip()
                break
            for child in nodes:
                child_pointer = str(child.get("pointer") or "").strip()
                if _parent_pointer(child_pointer) != pointer:
                    continue
                if _normalize_xml_name(child.get("local_tag", "")) != "modifiedfile":
                    continue
                modified_file_present = True
                modified_file_pointer = child_pointer
                child_attributes = dict(child.get("attributes", {}) or {})
                for child_attr_name, child_attr_value in child_attributes.items():
                    if _normalize_xml_name(child_attr_name) != "href":
                        continue
                    modified_file_href = " ".join(str(child_attr_value or "").split()).strip()
                    break
                if not modified_file_href:
                    modified_file_href = " ".join(str(child.get("text") or "").split()).strip()
                break
            leaf_title_records.append(
                {
                    "href": href,
                    "title_present": leaf_title_present,
                    "raw_title_text": raw_leaf_title,
                    "normalized_title_text": leaf_title,
                    "leaf_pointer": pointer,
                    "parent_pointer": parent_pointer,
                    "parent_local_tag": str(parent_node.get("local_tag") or "").strip(),
                }
            )
            leaf_lifecycle_records.append(
                {
                    "href": href,
                    "href_present": href_present,
                    "operation": operation,
                    "operation_present": operation_present,
                    "modified_file_href": modified_file_href,
                    "modified_file_present": modified_file_present,
                    "modified_file_pointer": modified_file_pointer,
                    "leaf_title": leaf_title,
                    "leaf_pointer": pointer,
                    "parent_pointer": parent_pointer,
                    "parent_local_tag": str(parent_node.get("local_tag") or "").strip(),
                }
            )
            if href:
                leaf_records.append(
                    {
                        "href": href,
                        "checksum_type": checksum_type,
                        "checksum": checksum,
                        "operation": operation,
                        "operation_present": operation_present,
                        "xml_lang": xml_lang,
                        "xml_lang_present": xml_lang_present,
                        "leaf_title": leaf_title,
                        "leaf_pointer": pointer,
                        "parent_pointer": parent_pointer,
                        "parent_local_tag": str(parent_node.get("local_tag") or "").strip(),
                    }
                )

    application_number = str(envelope_attributes.get("application-number") or "").strip() or _first_matching_attribute(
        nodes,
        tag_names=envelope_tags,
        attribute_aliases={"applicationnumber", "applicationid", "applicationno"},
    )
    sequence_number = str(envelope_attributes.get("sequence-number") or "").strip() or _first_matching_attribute(
        nodes,
        tag_names=envelope_tags,
        attribute_aliases={"sequencenumber", "seqnumber"},
    )
    related_sequence_number = str(envelope_attributes.get("related-sequence") or "").strip() or _first_matching_attribute(
        nodes,
        tag_names=envelope_tags,
        attribute_aliases={"relatedsequence", "relatedsequencenumber", "previoussequencenumber", "previoussequence"},
    )
    sequence_description = str(envelope_attributes.get("sequence-description") or "").strip() or _first_matching_attribute(
        nodes,
        tag_names=envelope_tags,
        attribute_aliases={"sequencedescription", "submissiondescription", "description"},
    )
    leaf_hrefs = _collect_matching_attributes(
        nodes,
        tag_names=leaf_tags,
        attribute_aliases={"href"},
    )
    checksum_types = _collect_matching_attributes(
        nodes,
        tag_names=leaf_tags,
        attribute_aliases={"checksumtype"},
    )
    leaf_count = sum(
        1
        for node in nodes
        if _normalize_xml_name(node.get("local_tag", "")) in leaf_tags
    )
    element_records: list[dict[str, Any]] = []
    attribute_records: list[dict[str, Any]] = []
    for node in nodes:
        pointer = str(node.get("pointer") or "").strip()
        local_tag = str(node.get("local_tag") or "").strip()
        attributes = dict(node.get("attributes", {}) or {})
        if not pointer:
            continue
        ancestors = _ancestor_nodes(pointer)
        ancestor_local_tags = [
            str(ancestor.get("local_tag") or "").strip()
            for ancestor in ancestors
            if str(ancestor.get("local_tag") or "").strip()
        ]
        ancestor_normalized_names = [
            _normalize_xml_name(tag)
            for tag in ancestor_local_tags
            if _normalize_xml_name(tag)
        ]
        element_records.append(
            {
                "element_local_tag": local_tag,
                "element_normalized_name": _normalize_xml_name(local_tag),
                "element_pointer": pointer,
                "parent_pointer": _parent_pointer(pointer),
                "element_text": _normalize_xml_text(node.get("text")),
                "element_raw_text": str(node.get("raw_text") or ""),
                "attribute_names": [
                    _strip_xml_namespace(str(attribute_name or "").strip())
                    for attribute_name in attributes.keys()
                    if str(attribute_name or "").strip()
                ],
                "ancestor_local_tags": ancestor_local_tags,
                "ancestor_normalized_names": ancestor_normalized_names,
            }
        )
        if not attributes:
            continue
        for attr_name, attr_value in attributes.items():
            attribute_name = _strip_xml_namespace(str(attr_name or "").strip())
            raw_value = str(attr_value if attr_value is not None else "")
            normalized_value = " ".join(raw_value.split()).strip()
            attribute_records.append(
                {
                    "attribute_name": attribute_name,
                    "normalized_attribute_name": _normalize_xml_name(attribute_name),
                    "raw_value": raw_value,
                    "normalized_value": normalized_value,
                    "has_edge_whitespace": raw_value != raw_value.strip(),
                    "element_local_tag": local_tag,
                    "element_normalized_name": _normalize_xml_name(local_tag),
                    "element_pointer": pointer,
                    "parent_pointer": _parent_pointer(pointer),
                    "ancestor_local_tags": ancestor_local_tags,
                    "ancestor_normalized_names": ancestor_normalized_names,
                }
            )

    if application_number:
        metadata["ectd_application_number"] = application_number
    if sequence_number:
        metadata["ectd_sequence_number"] = sequence_number
    if related_sequence_number:
        metadata["ectd_related_sequence_number"] = related_sequence_number
        metadata["ectd_previous_sequence_number"] = related_sequence_number
    if sequence_description:
        metadata["ectd_sequence_description"] = sequence_description
    for attribute_record in attribute_records:
        if str(attribute_record.get("element_pointer") or "").strip().count("/") != 1:
            continue
        if str(attribute_record.get("normalized_attribute_name") or "").strip().lower() != "schemaversion":
            continue
        schema_version = str(attribute_record.get("normalized_value") or "").strip()
        if schema_version:
            metadata["ectd_schema_version"] = schema_version
        break
    if envelope_attributes:
        metadata["ectd_envelope_attributes"] = envelope_attributes
    if envelope_count > 0:
        metadata["ectd_envelope_count"] = envelope_count
    if checksum_types:
        metadata["ectd_checksum_types"] = checksum_types
    if leaf_hrefs:
        metadata["ectd_leaf_hrefs"] = leaf_hrefs
    if leaf_records:
        metadata["ectd_leaf_records"] = leaf_records
    if leaf_title_records:
        metadata["ectd_leaf_title_records"] = leaf_title_records
    if leaf_lifecycle_records:
        metadata["ectd_leaf_lifecycle_records"] = leaf_lifecycle_records
    if leaf_count > 0:
        metadata["ectd_leaf_count"] = leaf_count
    if element_records:
        metadata["ectd_element_records"] = element_records
    if attribute_records:
        metadata["ectd_attribute_records"] = attribute_records
    metadata.update(_extract_sequence_contact_metadata(nodes))
    metadata.update(_extract_node_extension_metadata(nodes))

    return metadata


def parse_xml(path: Path) -> dict[str, Any]:
    """Parse XML/eCTD metadata into normalized nodes."""
    xml_text = path.read_text(encoding="utf-8", errors="ignore")
    xml_metadata = _extract_xml_doctype_metadata(path, xml_text)
    nodes: list[dict[str, Any]] = []
    root: ET.Element | None = None
    try:
        root = ET.fromstring(xml_text)
        _walk_xml(root, f"/{root.tag}[0]", nodes)
        parser_hint = "xml-tree-v1"
        xml_metadata.update(_build_xml_well_formedness_metadata(is_well_formed=True))
    except ET.ParseError as exc:
        parser_hint = "xml-fallback-plain-text"
        xml_metadata.update(_build_xml_well_formedness_metadata(is_well_formed=False, parse_error=exc))
        for line_index, raw_line in enumerate(xml_text.splitlines()):
            cleaned = " ".join(raw_line.split())
            if cleaned:
                nodes.append({"pointer": f"/raw[{line_index}]", "tag": "raw", "attributes": {}, "text": cleaned})
    xml_metadata.update(_build_xml_dtd_validation_metadata(path, xml_text, xml_metadata))

    merged_text = "\n".join(node["text"] for node in nodes if node.get("text"))
    ectd_metadata = _extract_ectd_metadata(nodes)
    sequence_directory_number = _extract_sequence_directory_number(path)
    if sequence_directory_number:
        ectd_metadata["ectd_sequence_directory_number"] = sequence_directory_number
    controlled_vocabulary_metadata = _extract_controlled_vocabulary_metadata(path, root, nodes)
    dependency_matrix_metadata = _extract_dependency_matrix_metadata(path, root, nodes)
    schema_location_metadata = _extract_xml_schema_location_metadata(path, root)
    xml_metadata.update(schema_location_metadata)
    schema_validation_metadata = _build_xml_schema_validation_metadata(path, xml_text, xml_metadata)
    xml_schema_metadata = _extract_xml_schema_metadata(path, root)
    schema_audit_metadata = _audit_xml_schema_dependencies(path, root)
    return {
        "filename": path.name,
        "source_path": str(path),
        "source_type": "xml",
        "nodes": nodes,
        "document_ast": {
            "source_type": "xml",
            "nodes": nodes,
            "node_count": len(nodes),
        },
        "text": merged_text,
        "atomic_facts": extract_atomic_facts(merged_text),
        "metadata": {
            "node_count": len(nodes),
            "parser_hint": parser_hint,
            **xml_metadata,
            **ectd_metadata,
            **controlled_vocabulary_metadata,
            **dependency_matrix_metadata,
            **schema_validation_metadata,
            **xml_schema_metadata,
            **schema_audit_metadata,
        },
    }
