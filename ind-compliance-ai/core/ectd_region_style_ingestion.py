from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


ECTD_REGION_STYLE_BUNDLE_VERSION = "ectd-region-style-bundle-v1"
ECTD_REGION_STYLE_BUNDLE_ID = "cn_ectd_attachment_1_3"
_REGION_STYLE_FILENAME = "cn-regional-1-0.xsl"
_XSL_NS = {
    "xsl": "http://www.w3.org/1999/XSL/Transform",
}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _load_xsl_tree(path: Path) -> ET.Element:
    return ET.fromstring(path.read_text(encoding="utf-8", errors="ignore"))


def _extract_namespace_map(path: Path) -> dict[str, str]:
    namespace_map: dict[str, str] = {}
    for _, item in ET.iterparse(path, events=["start-ns"]):
        prefix, uri = item
        namespace_map[prefix or "default"] = uri
    return namespace_map


def _extract_template_records(root: ET.Element) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template in root.findall("xsl:template", _XSL_NS):
        rows.append(
            {
                "match": _normalize_text(template.attrib.get("match")),
                "mode": _normalize_text(template.attrib.get("mode")),
            }
        )
    return rows


def _extract_code_display_mapping(root: ET.Element, mode: str) -> dict[str, str]:
    template = root.find(f"xsl:template[@mode='{mode}']", _XSL_NS)
    if template is None:
        return {}

    mapping: dict[str, str] = {}
    for when in template.findall(".//xsl:when", _XSL_NS):
        test_expr = _normalize_text(when.attrib.get("test"))
        code = ""
        if test_expr.startswith("@code='") and test_expr.endswith("'"):
            code = test_expr[len("@code='") : -1]
        if not code:
            continue
        label = _normalize_text("".join(when.itertext()))
        if label:
            mapping[code] = label
    return mapping


def _extract_static_texts(root: ET.Element) -> dict[str, str]:
    title_node = root.find(".//title")
    h1_node = root.find(".//h1")
    small_nodes = root.findall(".//small")
    return {
        "html_title_prefix": _normalize_text("".join(title_node.itertext())) if title_node is not None else "",
        "main_heading": _normalize_text("".join(h1_node.itertext())) if h1_node is not None else "",
        "schema_version_label": _normalize_text("".join(small_nodes[0].itertext())) if len(small_nodes) >= 1 else "",
        "style_version_label": _normalize_text("".join(small_nodes[1].itertext())) if len(small_nodes) >= 2 else "",
    }


def build_ectd_region_style_bundle(source_dir: Path) -> dict[str, Any]:
    source_dir = Path(source_dir)
    style_path = source_dir / _REGION_STYLE_FILENAME
    root = _load_xsl_tree(style_path)
    output_node = root.find("xsl:output", _XSL_NS)
    template_records = _extract_template_records(root)

    stylesheet = {
        "filename": style_path.name,
        "source_path": str(style_path),
        "stylesheet_version": _normalize_text(root.attrib.get("version")),
        "output_method": _normalize_text(output_node.attrib.get("method")) if output_node is not None else "",
        "output_encoding": _normalize_text(output_node.attrib.get("encoding")) if output_node is not None else "",
        "output_indent": _normalize_text(output_node.attrib.get("indent")) if output_node is not None else "",
        "template_count": len(template_records),
        "templates": template_records,
        "static_texts": _extract_static_texts(root),
        "controlled_display_mappings": {
            "application-type": _extract_code_display_mapping(root, "application-type"),
            "product-type": _extract_code_display_mapping(root, "product-type"),
            "regulatory-activity-type": _extract_code_display_mapping(root, "regulatory-activity-type"),
            "sequence-type": _extract_code_display_mapping(root, "sequence-type"),
        },
        "xml_root_tag": _normalize_text(root.tag.split("}", 1)[-1]),
        "xml_namespace_map": _extract_namespace_map(style_path),
    }

    return {
        "schema_version": ECTD_REGION_STYLE_BUNDLE_VERSION,
        "bundle_id": ECTD_REGION_STYLE_BUNDLE_ID,
        "source_directory": str(source_dir),
        "stylesheet_count": 1,
        "stylesheets": [stylesheet],
        "style_status_summary": {
            "status": "ready",
            "filenames": [stylesheet["filename"]],
        },
    }


def write_ectd_region_style_bundle(
    source_dir: Path,
    *,
    output_root: Path,
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_ectd_region_style_bundle(source_dir)
    output_path = output_root / f"{ECTD_REGION_STYLE_BUNDLE_ID}.region_style_bundle.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
