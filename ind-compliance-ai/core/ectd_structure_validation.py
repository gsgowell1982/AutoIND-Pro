from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from core.regulation_provenance import provenance_for_rule


_ALLOWED_CONTENT_EXTENSIONS = {".pdf", ".xml", ".xpt", ".txt", ".xsl"}
_ALLOWED_UTIL_SUPPORT_EXTENSIONS = {".dtd", ".xsd", ".xml", ".xsl", ".txt"}
_SCAFFOLD_EMPTY_DIRECTORY_RELATIVE_PATHS = {
    "m2",
    "m3",
    "m4",
    "m5",
    "util",
    "util/dtd",
    "util/style",
    *{f"m1/cn/{index:02d}" for index in range(13)},
}


def build_ectd_content_file_format_contract() -> dict[str, Any]:
    return {
        "schema_version": "ectd-content-file-format-contract-v1",
        "source": {
            "regulation_id": "cn_ectd_technical_specification",
            "clause": "3.3.1",
            "source_path": "data/regulations/eCTD技术规范.pdf",
        },
        "allowed_content_extensions": [".pdf", ".xml", ".xpt", ".txt", ".xsl"],
        "util_support_extensions": [".dtd", ".xsd", ".xml", ".xsl", ".txt"],
        "metadata_files": ["index.xml", "index-md5.txt"],
        "unknown_extension_action": "fail",
        "scope": "project_inventory_and_declared_leaf_href",
    }


def _finding(rule_id: str, path: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    finding_details = dict(details or {})
    finding_details.setdefault("regulatory_provenance", provenance_for_rule(rule_id))
    return {
        "rule_id": rule_id,
        "status": "fail",
        "severity": "error",
        "scope": "directory",
        "relative_path": path,
        "message": message,
        "blocking": True,
        "details": finding_details,
    }


def _validate_inventory_file_formats(file_paths: set[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for raw_path in sorted(file_paths):
        path = str(raw_path or "").replace("\\", "/").strip().lstrip("/")
        if not path:
            continue
        name = PurePosixPath(path).name.lower()
        if name in {"index.xml", "index-md5.txt"}:
            continue
        parts = PurePosixPath(path).parts
        in_util = "util" in {part.lower() for part in parts}
        extension = PurePosixPath(path).suffix.lower()
        allowed = _ALLOWED_UTIL_SUPPORT_EXTENSIONS if in_util else _ALLOWED_CONTENT_EXTENSIONS
        if extension in allowed:
            continue
        findings.append(
            _finding(
                "HR-ECTD-019",
                path,
                "eCTD project file format is outside the allowed content/support extension set.",
                {
                    "detected_extension": extension or "(missing extension)",
                    "allowed_extensions": sorted(allowed),
                    "file_role": "util_support" if in_util else "content_file",
                },
            )
        )
    return findings


def _sequence_relative_path(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if len(parts) < 2 or not parts[0] or not parts[1].isdigit() or len(parts[1]) != 4:
        return None
    return "/".join(parts[2:])


def _validate_inventory_empty_directories(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    directories = {
        str(path or "").replace("\\", "/").strip().strip("/")
        for path in inventory.get("directory_paths", [])
        if str(path or "").strip()
    }
    files = {
        str(path or "").replace("\\", "/").strip().strip("/")
        for path in inventory.get("file_paths", [])
        if str(path or "").strip()
    }
    findings: list[dict[str, Any]] = []
    for directory in sorted(directories):
        relative = _sequence_relative_path(directory)
        if not relative or relative in _SCAFFOLD_EMPTY_DIRECTORY_RELATIVE_PATHS:
            continue
        prefix = directory + "/"
        has_child = any(path.startswith(prefix) for path in directories | files if path != directory)
        if has_child:
            continue
        findings.append(
            _finding(
                "HR-ECTD-021",
                directory,
                "Submitted eCTD sequence contains a non-scaffold empty directory.",
                {"relative_to_sequence": relative, "reason": "empty_directory"},
            )
        )
    return findings


def _validate_inventory_zero_byte_files(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for entry in inventory.get("files", []) or []:
        if not isinstance(entry, dict) or entry.get("size") != 0:
            continue
        path = str(entry.get("relative_path") or "").replace("\\", "/").strip().strip("/")
        if not path or PurePosixPath(path).name.lower() in {"index.xml", "index-md5.txt"}:
            continue
        findings.append(
            _finding(
                "HR-ECTD-022",
                path,
                "Submitted eCTD document has zero bytes and is a placeholder/content-missing candidate.",
                {"reason": "zero_byte_file", "size": 0},
            )
        )
    return findings


def validate_package_structure(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    apps = list(inventory.get("application_roots") or [])
    directories = set(inventory.get("directory_paths") or [])
    files = set(inventory.get("file_paths") or [])
    if not apps:
        return [_finding("HR-ECTD-015", "", "No application/sequence package root was detected.")]
    findings.extend(_validate_inventory_file_formats(files))
    findings.extend(_validate_inventory_empty_directories(inventory))
    findings.extend(_validate_inventory_zero_byte_files(inventory))
    for app in apps:
        for sequence in app.get("sequences", []):
            root = sequence["relative_path"]
            required = [f"{root}/m{module}" for module in range(1, 6)] + [f"{root}/util", f"{root}/index.xml", f"{root}/index-md5.txt"]
            missing = [path for path in required if path not in directories and path not in files]
            if missing:
                findings.append(_finding("HR-ECTD-015", root, f"Missing required eCTD sequence entries: {', '.join(missing)}"))
            if f"{root}/m1/cn/cn-regional.xml" not in files:
                findings.append(_finding("HR-ECTD-017", f"{root}/m1/cn", "Missing cn-regional.xml in the China regional directory."))
    return findings
