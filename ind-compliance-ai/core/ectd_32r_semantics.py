from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIGURE2_PATH = _REPO_ROOT / "data" / "regulations" / "normalized" / "cn_ectd_technical_specification.32r_figure2_skeleton.xml"
_XLINK_NS = "http://www.w3.org/1999/xlink"
_LEAF_HREF_PATTERN = re.compile(r"^m3/32-body-data/32r-reg-info/[^\s]+$", re.IGNORECASE)

_EXTENSION_NODES = (
    {"structure_number": "3.2.R.1", "title": "3.2.R.1工艺验证", "folder": "cn32r1"},
    {"structure_number": "3.2.R.2", "title": "3.2.R.2批记录", "folder": "cn32r2"},
    {"structure_number": "3.2.R.3", "title": "3.2.R.3分析方法验证报告", "folder": "cn32r3"},
    {"structure_number": "3.2.R.4", "title": "3.2.R.4稳定性图谱", "folder": "cn32r4"},
    {"structure_number": "3.2.R.5", "title": "3.2.R.5可比性方案", "folder": "cn32r5"},
    {"structure_number": "3.2.R.6", "title": "3.2.R.6其他", "folder": "cn32r6"},
)


def build_ectd_32r_semantic_contract() -> dict[str, Any]:
    """Return the executable contract for regional pharmaceutical information (3.2.R)."""

    return {
        "schema_version": "ectd-32r-semantic-contract-v1",
        "section": "3.2.R",
        "applicability": {
            "regional_information_must_be_in_section": True,
            "biologic_requires_node_extensions_when_section_present": True,
            "non_biologic_must_not_be_forced_into_six_extensions": True,
            "content_category_adequacy_requires_document_review": True,
        },
        "parent": {
            "local_tag": "m3-2-r-regional-information",
            "ancestry": ["m3-quality", "m3-2-body-of-data", "m3-2-r-regional-information"],
            "directory": "m3/32-body-data/32r-reg-info",
        },
        "leaf": {
            "href_pattern": "^m3/32-body-data/32r-reg-info/[^\\s]+$",
            "required_attributes": ["ID", "operation", "xlink:type", "xlink:href", "checksum", "checksum-type"],
            "title_required": True,
        },
        "extension_nodes": [dict(item) for item in _EXTENSION_NODES],
        "sources": {
            "table4": "data/regulations/附件1-2：受控词汇文件包/node-extension-property_CN.xml",
            "cn_technical_specification": "data/regulations/eCTD技术规范.pdf#3.2",
            "ich_technical_specification": "data/regulations/eCTD_Specification_v3_2_2_0.pdf#3.2.R",
            "validation_standard": "data/regulations/eCTD验证标准.pdf#3.16-3.17",
            "figure2": "data/regulations/eCTD技术规范.pdf#图2",
        },
        "figure2_skeleton_path": "data/regulations/normalized/cn_ectd_technical_specification.32r_figure2_skeleton.xml",
    }


def parse_figure2_skeleton(xml_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(xml_path) if xml_path is not None else _FIGURE2_PATH
    root = ElementTree.parse(path).getroot()
    parent = root.find("./m3-2-body-of-data/m3-2-r-regional-information")
    if parent is None:
        raise ValueError("Figure 2 skeleton is missing the 3.2.R parent path")
    extensions = list(parent.findall("./node-extension"))
    titles = [str(node.findtext("./title") or "").strip() for node in extensions]
    leaf_attributes: list[str] = []
    if extensions:
        leaf = extensions[0].find("./leaf")
        if leaf is not None:
            for key in leaf.attrib:
                if key == f"{{{_XLINK_NS}}}type":
                    leaf_attributes.append("xlink:type")
                elif key == f"{{{_XLINK_NS}}}href":
                    leaf_attributes.append("xlink:href")
                else:
                    leaf_attributes.append(key)
    return {
        "parent_path": ["m3-quality", "m3-2-body-of-data", "m3-2-r-regional-information"],
        "extension_count": len(extensions),
        "extension_titles": titles,
        "leaf_attributes": leaf_attributes,
        "leaf_hrefs": [
            str(leaf.attrib.get(f"{{{_XLINK_NS}}}href") or "").strip()
            for node in extensions
            for leaf in node.findall("./leaf")
        ],
    }


def validate_ectd_32r_records(
    records: list[dict[str, Any]],
    *,
    product_type: str | None,
    regional_information_present: bool,
) -> dict[str, Any]:
    """Validate deterministic 3.2.R structure and return a reviewer-friendly result."""

    normalized_product_type = str(product_type or "").strip().lower()
    if not regional_information_present and not records:
        return {"status": "pass", "review_required": False, "issue_codes": [], "matched_records": []}
    if normalized_product_type == "biologic" and regional_information_present and not records:
        return {
            "status": "review",
            "review_required": True,
            "issue_codes": ["biologic_32r_requires_node_extension"],
            "matched_records": [],
        }

    issue_codes: list[str] = []
    matched_records: list[dict[str, Any]] = []
    allowed_titles = {item["title"] for item in _EXTENSION_NODES}
    for record in records:
        title = str(record.get("extension_title") or "").strip()
        parent = str(record.get("parent_local_tag") or "").strip()
        hrefs = [str(value or "").strip().replace("\\", "/") for value in record.get("leaf_hrefs", []) or []]
        record_issues: list[str] = []
        if title not in allowed_titles:
            record_issues.append("invalid_extension_title")
        if parent != "m3-2-r-regional-information":
            record_issues.append("invalid_parent")
        if not hrefs or any(not _LEAF_HREF_PATTERN.match(href) for href in hrefs):
            record_issues.append("invalid_leaf_href")
        issue_codes.extend(record_issues)
        matched_records.append({**record, "issue_codes": record_issues})

    unique_issue_codes = list(dict.fromkeys(issue_codes))
    return {
        "status": "fail" if unique_issue_codes else "pass",
        "review_required": False,
        "issue_codes": unique_issue_codes,
        "matched_records": matched_records,
    }


__all__ = [
    "build_ectd_32r_semantic_contract",
    "parse_figure2_skeleton",
    "validate_ectd_32r_records",
]
