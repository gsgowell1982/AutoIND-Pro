from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from core.regulation_provenance import provenance_for_rule


_APPLICATION_PREFIX_CATEGORIES = {
    "x": "new_drug_application",
    "y": "generic_drug_application",
    "l": "clinical_trial_application",
}
_ECTD_ALLOWED_NAME_PATH_PATTERN = re.compile(r"^[a-z0-9/_\-.]+$")
_ECTD_MAX_RELATIVE_PATH_LENGTH = 180
_ECTD_MAX_NAME_SEGMENT_LENGTH = 64


def build_ectd_path_naming_contract() -> dict[str, Any]:
    return {
        "schema_version": "ectd-path-naming-contract-v1",
        "source": {
            "regulation_id": "cn_ectd_technical_specification",
            "clause": "3.3.2",
            "table": "5",
            "source_path": "data/regulations/eCTD技术规范.pdf",
        },
        "allowed_characters": "a-z0-9-_",
        "path_character_pattern": "^[a-z0-9/_\\-.]+$",
        "max_sequence_relative_path_length": 180,
        "max_name_segment_length": 64,
        "xml_skeleton_reference_required": True,
        "unknown_name_action": "fail",
    }


def parse_application_number(value: str) -> dict[str, Any]:
    raw = str(value or "").strip()
    normalized = raw.lower()
    prefix = normalized[:1] if normalized else ""
    year_text = normalized[1:5] if len(normalized) >= 5 else ""
    serial = normalized[5:] if len(normalized) > 5 else ""
    year = int(year_text) if len(year_text) == 4 and year_text.isdigit() else None
    current_year = datetime.now().year
    year_relation = "unknown"
    if year is not None:
        year_relation = "current" if year == current_year else "future" if year > current_year else "historical"
    return {
        "raw": raw,
        "prefix": prefix,
        "category": _APPLICATION_PREFIX_CATEGORIES.get(prefix),
        "year": year,
        "year_text": year_text,
        "year_relation": year_relation,
        "year_is_current": year == current_year if year is not None else False,
        "serial": serial,
        "format_valid": bool(re.fullmatch(r"[xyl]\d{9}", normalized, re.IGNORECASE)),
        "length": len(raw),
        "current_year": current_year,
    }


def _finding(
    rule_id: str,
    path: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    status: str = "fail",
    severity: str = "error",
    blocking: bool = True,
) -> dict[str, Any]:
    finding_details = dict(details or {})
    finding_details.setdefault("regulatory_provenance", provenance_for_rule(rule_id))
    return {
        "rule_id": rule_id,
        "status": status,
        "severity": severity,
        "scope": "directory",
        "relative_path": path,
        "message": message,
        "blocking": blocking,
        "details": finding_details,
    }


def validate_package_naming(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for app in inventory.get("application_roots", []):
        application_number = str(app.get("name", "")).strip()
        parsed = parse_application_number(application_number)
        path = app.get("relative_path", "")
        base_details = {
            "issue_family": "application_number_format",
            "application_number": application_number,
            "application_category": parsed["category"],
            "application_year": parsed["year"],
            "application_serial": parsed["serial"],
            "year_relation": parsed["year_relation"],
        }
        if parsed["length"] != 10:
            findings.append(
                _finding(
                    "HR-ECTD-002",
                    path,
                    "Application root name must be exactly 10 characters: 1 type letter, 4 year digits, and 5 serial digits.",
                    details={**base_details, "issue_code": "application_length", "observed_length": parsed["length"], "expected_length": 10},
                )
            )
        if parsed["prefix"] not in _APPLICATION_PREFIX_CATEGORIES:
            findings.append(
                _finding(
                    "HR-ECTD-002",
                    path,
                    "Application root type prefix must be x (new drug), y (generic drug), or l (clinical trial).",
                    details={**base_details, "issue_code": "application_prefix", "allowed_prefixes": sorted(_APPLICATION_PREFIX_CATEGORIES)},
                )
            )
        if len(parsed["year_text"]) != 4 or not parsed["year_text"].isdigit():
            findings.append(
                _finding(
                    "HR-ECTD-002",
                    path,
                    "Application root year segment must contain exactly 4 digits.",
                    details={**base_details, "issue_code": "application_year", "observed_year_segment": parsed["year_text"], "expected_year_digits": 4},
                )
            )
        if len(parsed["serial"]) != 5 or not parsed["serial"].isdigit():
            findings.append(
                _finding(
                    "HR-ECTD-002",
                    path,
                    "Application root serial segment must contain exactly 5 digits.",
                    details={**base_details, "issue_code": "application_serial", "observed_serial": parsed["serial"], "expected_serial_digits": 5},
                )
            )
        if parsed["year_relation"] == "future" and len(parsed["year_text"]) == 4 and parsed["year_text"].isdigit():
            findings.append(
                _finding(
                    "HR-ECTD-002",
                    path,
                    "Application root year is later than the current calendar year and requires manual confirmation of the regulator-assigned identifier.",
                    details={**base_details, "issue_code": "application_year_future", "current_year": parsed["current_year"]},
                    status="human_review",
                    severity="warning",
                    blocking=False,
                )
            )
        for sequence in app.get("sequences", []):
            if not re.match(r"^\d{4}$", str(sequence.get("name", ""))):
                findings.append(_finding("HR-ECTD-001", sequence.get("relative_path", ""), "Sequence directory must be exactly four digits."))
    for path in inventory.get("file_paths", []):
        if "\\" in path or path.startswith("/") or ".." in path.split("/"):
            findings.append(_finding("HR-ECTD-005", path, "Package paths must be relative, normalized, and use forward slashes."))
    findings.extend(_validate_inventory_name_constraints(inventory))
    return findings


def _validate_inventory_name_constraints(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    paths = set(inventory.get("directory_paths", [])) | set(inventory.get("file_paths", []))
    for raw_path in sorted(paths):
        original_path = str(raw_path or "").strip()
        normalized_path = original_path.replace("\\", "/").lstrip("/")
        parts = PurePosixPath(normalized_path).parts
        if len(parts) < 2 or not re.fullmatch(r"\d{4}", parts[1]):
            continue
        sequence_relative_path = "/".join(parts[2:])
        if not sequence_relative_path:
            continue
        issue_codes: list[str] = []
        if len(sequence_relative_path) > _ECTD_MAX_RELATIVE_PATH_LENGTH:
            issue_codes.append("path_too_long")
        if not _ECTD_ALLOWED_NAME_PATH_PATTERN.fullmatch(sequence_relative_path):
            issue_codes.append("invalid_character")
        oversized_segment = next(
            (segment for segment in parts[2:] if len(segment) > _ECTD_MAX_NAME_SEGMENT_LENGTH),
            None,
        )
        if oversized_segment is not None:
            issue_codes.append("segment_too_long")
        if not issue_codes:
            continue
        findings.append(
            _finding(
                "HR-ECTD-005",
                original_path,
                "eCTD folder/file names must use lowercase a-z, digits, hyphen, underscore, and remain within path/name length limits.",
                details={
                    "issue_family": "package_name_constraints",
                    "issue_codes": issue_codes,
                    "sequence_relative_path": sequence_relative_path,
                    "observed_path_length": len(sequence_relative_path),
                    "max_path_length": _ECTD_MAX_RELATIVE_PATH_LENGTH,
                    "max_segment_length": _ECTD_MAX_NAME_SEGMENT_LENGTH,
                    "oversized_segment": oversized_segment,
                    "allowed_character_pattern": "^[a-z0-9/_\\-.]+$",
                },
            )
        )
    return findings
