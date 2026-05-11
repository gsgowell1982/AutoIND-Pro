from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parsers.parser_registry import parse_file


ECTD_CONTROLLED_VOCABULARY_BUNDLE_VERSION = "ectd-controlled-vocabulary-bundle-v1"
ECTD_CONTROLLED_VOCABULARY_BUNDLE_ID = "cn_ectd_attachment_1_2"
_CONTROLLED_VOCABULARY_FILENAMES = (
    "cv-application-type.xml",
    "cv-product-type.xml",
    "cv-regulatory-activity-type.xml",
    "cv-sequence-type.xml",
)
_DEPENDENCY_MATRIX_FILENAME = "depend-apt-rat-sqt.xml"


def _load_parsed_xml(path: Path) -> dict[str, Any]:
    parsed = parse_file(path)
    metadata = dict(parsed.get("metadata") or {})
    return {
        "filename": str(parsed.get("filename") or path.name).strip(),
        "source_path": str(parsed.get("source_path") or path).strip(),
        "metadata": metadata,
    }


def _build_controlled_vocabulary_record(parsed: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(parsed.get("metadata") or {})
    return {
        "filename": str(parsed.get("filename") or "").strip(),
        "source_path": str(parsed.get("source_path") or "").strip(),
        "controlled_vocabulary_name": str(metadata.get("ectd_controlled_vocabulary_name") or "").strip(),
        "controlled_vocabulary_key": str(metadata.get("ectd_controlled_vocabulary_key") or "").strip(),
        "version": str(metadata.get("ectd_controlled_vocabulary_version") or "").strip(),
        "valid_from": str(metadata.get("ectd_controlled_vocabulary_valid_from") or "").strip(),
        "valid_to": str(metadata.get("ectd_controlled_vocabulary_valid_to") or "").strip(),
        "code_count": int(metadata.get("ectd_controlled_vocabulary_code_count", 0) or 0),
        "values": list(metadata.get("ectd_controlled_vocabulary_values") or []),
        "entries": list(metadata.get("ectd_controlled_vocabulary_entries") or []),
        "versions": list(metadata.get("ectd_controlled_vocabulary_versions") or []),
        "schema_primary_path": str(metadata.get("xml_schema_primary_path") or "").strip(),
        "schema_dependency_files": list(metadata.get("xml_schema_dependency_files") or []),
        "schema_missing_dependencies": list(metadata.get("xml_schema_missing_dependencies") or []),
        "schema_dependency_status": str(metadata.get("xml_schema_dependency_status") or "").strip(),
    }


def _build_dependency_matrix_record(parsed: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(parsed.get("metadata") or {})
    return {
        "filename": str(parsed.get("filename") or "").strip(),
        "source_path": str(parsed.get("source_path") or "").strip(),
        "matrix_name": str(metadata.get("ectd_dependency_matrix_name") or "").strip(),
        "version": str(metadata.get("ectd_dependency_matrix_version") or "").strip(),
        "valid_from": str(metadata.get("ectd_dependency_matrix_valid_from") or "").strip(),
        "valid_to": str(metadata.get("ectd_dependency_matrix_valid_to") or "").strip(),
        "row_count": int(metadata.get("ectd_dependency_matrix_row_count", 0) or 0),
        "rows": list(metadata.get("ectd_dependency_matrix_rows") or []),
        "versions": list(metadata.get("ectd_dependency_matrix_versions") or []),
        "schema_primary_path": str(metadata.get("xml_schema_primary_path") or "").strip(),
        "schema_dependency_files": list(metadata.get("xml_schema_dependency_files") or []),
        "schema_missing_dependencies": list(metadata.get("xml_schema_missing_dependencies") or []),
        "schema_dependency_status": str(metadata.get("xml_schema_dependency_status") or "").strip(),
    }


def build_ectd_controlled_vocabulary_bundle(source_dir: Path) -> dict[str, Any]:
    source_dir = Path(source_dir)
    controlled_vocabularies: list[dict[str, Any]] = []
    missing_dependencies: set[str] = set()
    missing_source_files: list[str] = []

    for filename in _CONTROLLED_VOCABULARY_FILENAMES:
        path = source_dir / filename
        if not path.exists():
            missing_source_files.append(filename)
            continue
        record = _build_controlled_vocabulary_record(_load_parsed_xml(path))
        controlled_vocabularies.append(record)
        missing_dependencies.update(record.get("schema_missing_dependencies") or [])

    dependency_matrix_path = source_dir / _DEPENDENCY_MATRIX_FILENAME
    dependency_matrix: dict[str, Any]
    if dependency_matrix_path.exists():
        dependency_matrix = _build_dependency_matrix_record(_load_parsed_xml(dependency_matrix_path))
        missing_dependencies.update(dependency_matrix.get("schema_missing_dependencies") or [])
    else:
        missing_source_files.append(_DEPENDENCY_MATRIX_FILENAME)
        dependency_matrix = {
            "filename": _DEPENDENCY_MATRIX_FILENAME,
            "source_path": str(dependency_matrix_path),
            "matrix_name": "",
            "version": "",
            "valid_from": "",
            "valid_to": "",
            "row_count": 0,
            "rows": [],
            "versions": [],
            "schema_primary_path": "",
            "schema_dependency_files": [],
            "schema_missing_dependencies": [],
            "schema_dependency_status": "",
        }

    schema_status = "ready"
    if missing_source_files:
        schema_status = "missing_source_files"
    elif missing_dependencies:
        schema_status = "partial_dependency_closure"

    return {
        "schema_version": ECTD_CONTROLLED_VOCABULARY_BUNDLE_VERSION,
        "bundle_id": ECTD_CONTROLLED_VOCABULARY_BUNDLE_ID,
        "source_directory": str(source_dir),
        "missing_source_files": missing_source_files,
        "controlled_vocabulary_count": len(controlled_vocabularies),
        "controlled_vocabularies": controlled_vocabularies,
        "dependency_matrix": dependency_matrix,
        "schema_status_summary": {
            "status": schema_status,
            "missing_dependencies": sorted(missing_dependencies),
            "missing_source_files": missing_source_files,
        },
    }


def write_ectd_controlled_vocabulary_bundle(
    source_dir: Path,
    *,
    output_root: Path,
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_ectd_controlled_vocabulary_bundle(source_dir)
    output_path = output_root / f"{ECTD_CONTROLLED_VOCABULARY_BUNDLE_ID}.controlled_vocabulary_bundle.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
