from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_BUNDLE_FILENAME = "cn_ectd_attachment_1_2.controlled_vocabulary_bundle.json"
_FIELD_REQUIREMENTS = {
    "application-type": {
        "vocabulary_name": "cv-application-type",
        "requirement_id": "cn_ectd_technical_specification:req_application_type_controlled_vocabulary_validity",
        "source_clause_id": "cn_ectd_technical_specification:sec_2_1_2",
        "xml_locations": ["cn-regional.xml/@application-type", "index.xml//application-type"],
    },
    "product-type": {
        "vocabulary_name": "cv-product-type",
        "requirement_id": "cn_ectd_technical_specification:req_product_type_controlled_vocabulary_validity",
        "source_clause_id": "cn_ectd_technical_specification:sec_2_1_3",
        "xml_locations": ["cn-regional.xml/@product-type", "index.xml//product-type"],
    },
    "regulatory-activity-type": {
        "vocabulary_name": "cv-regulatory-activity-type",
        "requirement_id": "cn_ectd_technical_specification:req_regulatory_activity_type_controlled_vocabulary_validity",
        "source_clause_id": "cn_ectd_technical_specification:sec_2_2_1",
        "xml_locations": ["cn-regional.xml/@regulatory-activity-type", "index.xml//regulatory-activity-type"],
    },
    "sequence-type": {
        "vocabulary_name": "cv-sequence-type",
        "requirement_id": "cn_ectd_technical_specification:req_sequence_type_controlled_vocabulary_validity",
        "source_clause_id": "cn_ectd_technical_specification:sec_2_3_2",
        "xml_locations": ["cn-regional.xml/@sequence-type", "index.xml//sequence-type"],
    },
}


def _default_bundle_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "regulations" / "normalized" / _BUNDLE_FILENAME


def _load_bundle(bundle_path: Path | None = None) -> dict[str, Any]:
    path = Path(bundle_path) if bundle_path else _default_bundle_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _vocabulary_records(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in bundle.get("controlled_vocabularies", []) or []:
        payload = dict(record or {})
        key = str(payload.get("controlled_vocabulary_key") or "").strip()
        if not key:
            name = str(payload.get("controlled_vocabulary_name") or "").strip()
            key = name.removeprefix("cv-")
        if key:
            records[key] = payload
    return records


def build_ectd_vocabulary_rule_contract(*, bundle_path: Path | None = None) -> dict[str, Any]:
    bundle = _load_bundle(bundle_path)
    records = _vocabulary_records(bundle)
    vocabularies: dict[str, dict[str, Any]] = {}
    for field_name, binding in _FIELD_REQUIREMENTS.items():
        record = dict(records.get(field_name) or {})
        entries = list(record.get("entries") or [])
        vocabularies[field_name] = {
            "vocabulary_name": str(binding["vocabulary_name"]),
            "version": str(record.get("version") or ""),
            "valid_from": str(record.get("valid_from") or ""),
            "valid_to": str(record.get("valid_to") or ""),
            "code_count": int(record.get("code_count", len(entries)) or 0),
            "codes": [str(item).strip() for item in record.get("values", []) or [] if str(item).strip()],
            "entries": entries,
            "source_path": str(record.get("source_path") or ""),
        }

    dependency_matrix = dict(bundle.get("dependency_matrix") or {})
    rows = [
        {
            "application_type": str(row.get("application_type") or "").strip(),
            "regulatory_activity_type": str(row.get("regulatory_activity_type") or "").strip(),
            "sequence_type": str(row.get("sequence_type") or "").strip(),
        }
        for row in dependency_matrix.get("rows", []) or []
        if isinstance(row, dict)
    ]
    return {
        "schema_version": "ectd-semantic-rule-contract-v1",
        "bundle_id": str(bundle.get("bundle_id") or "cn_ectd_attachment_1_2"),
        "bundle_version": str(bundle.get("schema_version") or ""),
        "vocabularies": vocabularies,
        "field_bindings": {
            field_name: {
                **dict(binding),
                "required": True,
                "evidence_sources": ["parsed cn-regional.xml envelope", "parsed index.xml envelope", "uploaded supporting CV file"],
            }
            for field_name, binding in _FIELD_REQUIREMENTS.items()
        },
        "application_prefix_mapping": {
            "x": {"application_type_code": "cnapt2", "category": "new_drug_application"},
            "y": {"application_type_code": "cnapt3", "category": "generic_drug_application"},
            "l": {"application_type_code": "cnapt1", "category": "clinical_trial_application"},
        },
        "type_compatibility": {
            "matrix_name": str(dependency_matrix.get("matrix_name") or "depend-apt-rat-sqt"),
            "version": str(dependency_matrix.get("version") or ""),
            "row_count": len(rows),
            "rows": rows,
            "source_path": str(dependency_matrix.get("source_path") or ""),
        },
    }


def validate_ectd_envelope_vocabulary(
    envelope_attributes: dict[str, Any],
    *,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_contract = contract or build_ectd_vocabulary_rule_contract()
    normalized_attributes = {
        str(key).strip(): str(value or "").strip().lower()
        for key, value in dict(envelope_attributes or {}).items()
    }
    invalid_fields: list[str] = []
    missing_fields: list[str] = []
    field_results: dict[str, dict[str, Any]] = {}
    for field_name, vocabulary in dict(active_contract.get("vocabularies", {}) or {}).items():
        value = normalized_attributes.get(field_name, "")
        codes = {str(code).strip().lower() for code in vocabulary.get("codes", []) or []}
        if not value:
            missing_fields.append(field_name)
            field_results[field_name] = {"status": "missing", "value": "", "allowed_codes": sorted(codes)}
        elif value not in codes:
            invalid_fields.append(field_name)
            field_results[field_name] = {"status": "invalid", "value": value, "allowed_codes": sorted(codes)}
        else:
            field_results[field_name] = {"status": "pass", "value": value, "allowed_codes": sorted(codes)}

    compatibility = dict(active_contract.get("type_compatibility", {}) or {})
    triplet_values = tuple(
        normalized_attributes.get(field, "")
        for field in ("application-type", "regulatory-activity-type", "sequence-type")
    )
    rows = {
        (
            str(row.get("application_type") or "").strip().lower(),
            str(row.get("regulatory_activity_type") or "").strip().lower(),
            str(row.get("sequence_type") or "").strip().lower(),
        )
        for row in compatibility.get("rows", []) or []
    }
    complete_triplet = all(triplet_values)
    compatibility_result = {
        "valid": triplet_values in rows if complete_triplet else None,
        "values": {
            "application_type": triplet_values[0],
            "regulatory_activity_type": triplet_values[1],
            "sequence_type": triplet_values[2],
        },
        "issue_code": "" if not complete_triplet or triplet_values in rows else "incompatible_type_triplet",
        "matrix_name": compatibility.get("matrix_name", ""),
    }
    if compatibility_result["issue_code"]:
        invalid_fields.append("type-compatibility")

    status = "fail" if invalid_fields else "warn" if missing_fields else "pass"
    return {
        "status": status,
        "field_results": field_results,
        "invalid_fields": sorted(set(invalid_fields)),
        "missing_fields": sorted(set(missing_fields)),
        "type_compatibility": compatibility_result,
        "contract_version": str(active_contract.get("schema_version") or ""),
    }


__all__ = ["build_ectd_vocabulary_rule_contract", "validate_ectd_envelope_vocabulary"]
