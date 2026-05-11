from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from parsers.parser_registry import parse_file


ECTD_STF_VALID_VALUES_BUNDLE_VERSION = "ectd-stf-valid-values-bundle-v1"
ECTD_STF_VALID_VALUES_BUNDLE_ID = "cn_ectd_attachment_2_6"
_VALID_VALUES_FILENAME = "valid-values.xml"
def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _load_xml_tree(path: Path) -> ET.Element:
    return ET.fromstring(path.read_text(encoding="utf-8", errors="ignore"))


def _extract_valid_value_groups(root: ET.Element) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for child in root:
        local_name = child.tag.split("}", 1)[-1]
        group_name = _normalize_text(child.attrib.get("name"))
        values: list[dict[str, str]] = []
        for item in child.findall("valid-value"):
            value = _normalize_text(item.attrib.get("value"))
            realm = _normalize_text(item.attrib.get("realm"))
            if not value:
                continue
            values.append({"realm": realm, "value": value})
        groups.append(
            {
                "element": local_name,
                "name": group_name,
                "value_count": len(values),
                "realms": sorted({row["realm"] for row in values if row["realm"]}),
                "values": values,
            }
        )
    return groups


def build_ectd_stf_valid_values_bundle(source_dir: Path) -> dict[str, Any]:
    source_dir = Path(source_dir)
    valid_values_path = source_dir / _VALID_VALUES_FILENAME
    parsed = parse_file(valid_values_path)
    metadata = dict(parsed.get("metadata") or {})
    root = _load_xml_tree(valid_values_path)
    groups = _extract_valid_value_groups(root)

    return {
        "schema_version": ECTD_STF_VALID_VALUES_BUNDLE_VERSION,
        "bundle_id": ECTD_STF_VALID_VALUES_BUNDLE_ID,
        "source_directory": str(source_dir),
        "valid_values_file": {
            "filename": str(parsed.get("filename") or valid_values_path.name).strip(),
            "source_path": str(parsed.get("source_path") or valid_values_path).strip(),
            "root_tag": _normalize_text(root.tag.split("}", 1)[-1]),
            "namespace": _normalize_text(root.tag.split("}", 1)[0].lstrip("{")),
            "dtd_version": _normalize_text(root.attrib.get("dtd-version")),
            "xml_root_tag": str(metadata.get("xml_root_tag") or "").strip(),
            "xml_namespace_map": dict(metadata.get("xml_namespace_map") or {}),
        },
        "group_count": len(groups),
        "total_value_count": sum(int(group.get("value_count", 0) or 0) for group in groups),
        "groups": groups,
        "valid_values_status_summary": {
            "status": "ready",
            "group_elements": [str(group.get("element") or "").strip() for group in groups],
        },
    }


def write_ectd_stf_valid_values_bundle(
    source_dir: Path,
    *,
    output_root: Path,
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_ectd_stf_valid_values_bundle(source_dir)
    output_path = output_root / f"{ECTD_STF_VALID_VALUES_BUNDLE_ID}.stf_valid_values_bundle.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
