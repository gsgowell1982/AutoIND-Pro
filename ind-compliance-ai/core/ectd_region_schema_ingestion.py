from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from parsers.parser_registry import parse_file


ECTD_REGION_SCHEMA_BUNDLE_VERSION = "ectd-region-schema-bundle-v1"
ECTD_REGION_SCHEMA_BUNDLE_ID = "cn_ectd_attachment_1_1"
_REGION_SCHEMA_FILENAME = "cn-regional-1-0.xsd"
_XSD_NS = {"xs": "http://www.w3.org/2001/XMLSchema"}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _extract_sequence_fields(sequence_node: ET.Element | None) -> list[dict[str, str]]:
    if sequence_node is None:
        return []
    rows: list[dict[str, str]] = []
    for child in sequence_node.findall("xs:element", _XSD_NS):
        rows.append(
            {
                "name": _normalize_text(child.attrib.get("name")),
                "type": _normalize_text(child.attrib.get("type")),
                "min_occurs": _normalize_text(child.attrib.get("minOccurs")) or "1",
                "max_occurs": _normalize_text(child.attrib.get("maxOccurs")) or "1",
            }
        )
    return rows


def _extract_attributes(node: ET.Element | None) -> list[dict[str, Any]]:
    if node is None:
        return []
    rows: list[dict[str, Any]] = []
    for child in node.findall("xs:attribute", _XSD_NS):
        row: dict[str, Any] = {
            "name": _normalize_text(child.attrib.get("name")),
            "type": _normalize_text(child.attrib.get("type")) or "inline-restriction",
            "use": _normalize_text(child.attrib.get("use")) or "optional",
        }
        restriction = child.find("xs:simpleType/xs:restriction", _XSD_NS)
        if restriction is not None:
            allowed_values = [
                _normalize_text(item.attrib.get("value"))
                for item in restriction.findall("xs:enumeration", _XSD_NS)
                if _normalize_text(item.attrib.get("value"))
            ]
            if allowed_values:
                row["allowed_values"] = allowed_values
        rows.append(row)
    return rows


def _load_schema_tree(path: Path) -> ET.Element:
    return ET.fromstring(path.read_text(encoding="utf-8", errors="ignore"))


def _extract_region_schema_structure(path: Path) -> dict[str, Any]:
    root = _load_schema_tree(path)

    root_element = root.find("xs:element[@name='cn_ectd']", _XSD_NS)
    root_sequence = (
        root_element.find("xs:complexType/xs:sequence", _XSD_NS) if root_element is not None else None
    )
    envelope_type = root.find("xs:complexType[@name='cn-envelope']", _XSD_NS)
    envelope_sequence = envelope_type.find("xs:sequence", _XSD_NS) if envelope_type is not None else None
    contact_type = root.find("xs:complexType[@name='cn-contact']", _XSD_NS)
    contact_sequence = contact_type.find("xs:sequence", _XSD_NS) if contact_type is not None else None
    content_type = root.find("xs:complexType[@name='cn-content']", _XSD_NS)
    content_sequence = content_type.find("xs:sequence", _XSD_NS) if content_type is not None else None

    return {
        "root_element": {
            "name": "cn_ectd",
            "children": _extract_sequence_fields(root_sequence),
        },
        "root_attributes": _extract_attributes(root_element.find("xs:complexType", _XSD_NS) if root_element is not None else None),
        "envelope_fields": _extract_sequence_fields(envelope_sequence),
        "sequence_contact_fields": _extract_sequence_fields(contact_sequence),
        "content_top_level_elements": _extract_sequence_fields(content_sequence),
    }


def build_ectd_region_schema_bundle(source_dir: Path) -> dict[str, Any]:
    source_dir = Path(source_dir)
    schema_path = source_dir / _REGION_SCHEMA_FILENAME
    parsed = parse_file(schema_path)
    metadata = dict(parsed.get("metadata") or {})
    structure = _extract_region_schema_structure(schema_path)

    region_schema = {
        "filename": str(parsed.get("filename") or schema_path.name).strip(),
        "source_path": str(parsed.get("source_path") or schema_path).strip(),
        "schema_name": str(metadata.get("xml_schema_name") or "").strip(),
        "target_namespace": str(metadata.get("xml_schema_target_namespace") or "").strip(),
        "import_count": int(metadata.get("xml_schema_import_count", 0) or 0),
        "imports": list(metadata.get("xml_schema_imports") or []),
        "element_count": int(metadata.get("xml_schema_element_count", 0) or 0),
        "complex_type_count": int(metadata.get("xml_schema_complex_type_count", 0) or 0),
        "named_elements": list(metadata.get("xml_schema_named_elements") or []),
        "schema_primary_path": str(metadata.get("xml_schema_primary_path") or "").strip(),
        "schema_dependency_files": list(metadata.get("xml_schema_dependency_files") or []),
        "schema_missing_dependencies": list(metadata.get("xml_schema_missing_dependencies") or []),
        "schema_dependency_status": str(metadata.get("xml_schema_dependency_status") or "").strip(),
        **structure,
    }

    missing_dependencies = sorted(set(region_schema["schema_missing_dependencies"]))
    return {
        "schema_version": ECTD_REGION_SCHEMA_BUNDLE_VERSION,
        "bundle_id": ECTD_REGION_SCHEMA_BUNDLE_ID,
        "source_directory": str(source_dir),
        "region_schema": region_schema,
        "schema_status_summary": {
            "status": "ready" if not missing_dependencies else "partial_dependency_closure",
            "missing_dependencies": missing_dependencies,
        },
    }


def write_ectd_region_schema_bundle(
    source_dir: Path,
    *,
    output_root: Path,
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_ectd_region_schema_bundle(source_dir)
    output_path = output_root / f"{ECTD_REGION_SCHEMA_BUNDLE_ID}.region_schema_bundle.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
