from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REGULATION_ID = "cn_drug_registration_classification_and_dossier_requirements"
SCHEMA_VERSION = "dossier-checklist-v1"

_PREREQUISITE_FACTS: tuple[dict[str, str], ...] = (
    {
        "fact_key": "application_type",
        "label": "Application type",
        "description": "Needed to distinguish clinical trial, marketing registration, and API-supporting submissions.",
    },
    {
        "fact_key": "registration_class",
        "label": "Registration class",
        "description": "Needed before class-specific requirements can be treated as applicable.",
    },
    {
        "fact_key": "product_type",
        "label": "Product type",
        "description": "Needed to confirm this dossier follows the chemical-drug registration classification source.",
    },
    {
        "fact_key": "submission_stage",
        "label": "Submission stage",
        "description": "Needed to separate clinical-trial, marketing-registration, and post-clinical database obligations.",
    },
    {
        "fact_key": "package_scope",
        "label": "Package scope",
        "description": "Needed to know whether the upload represents a document, sequence package, activity, or application project.",
    },
    {
        "fact_key": "clinical_trial_completion_status",
        "label": "Clinical trial completion status",
        "description": "Needed before checking whether electronic clinical trial databases should be present.",
    },
)


def _load_requirement_matrix(project_root: Path) -> dict[str, Any]:
    path = (
        project_root
        / "data"
        / "regulations"
        / "normalized"
        / f"{REGULATION_ID}.requirement_matrix.json"
    )
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _trimmed(value: Any) -> str:
    return str(value or "").strip()


def _preview(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _field_value(value: Any, source: str) -> dict[str, Any]:
    return {"value": value, "source": source}


def _has_fact_value(fact_value: dict[str, Any] | None) -> bool:
    if not fact_value:
        return False
    value = fact_value.get("value")
    return value is not None and value != "" and value != []


def _first_available_fact(*fact_values: dict[str, Any] | None) -> dict[str, Any]:
    for fact_value in fact_values:
        if _has_fact_value(fact_value):
            return dict(fact_value)
    return dict(fact_values[0] or {})


def _source_label(source: Any) -> str:
    source_key = _trimmed(source)
    return {
        "ectd_sequence_package": "eCTD sequence package",
        "submission_context": "submission context",
        "ectd_envelope": "eCTD envelope",
    }.get(source_key, "not available")


def _fact_evidence_status(has_value: bool) -> str:
    return "local_evidence_present" if has_value else "missing_required_prerequisite"


def _fact_confidence(source: Any, has_value: bool) -> str:
    if not has_value:
        return "none"
    source_key = _trimmed(source)
    if source_key in {"ectd_sequence_package", "ectd_envelope", "submission_context"}:
        return "high"
    return "medium"


def _fact_review_action(has_value: bool) -> str:
    if has_value:
        return "confirm_if_business_context_disagrees"
    return "provide_before_applicability_review"


def _non_empty_text_values(items: list[dict[str, Any]], field_name: str) -> set[str]:
    values: set[str] = set()
    for item in items:
        value = _trimmed(item.get(field_name))
        if value:
            values.add(value)
    return values


def _single_sequence_package_value(
    sequence_packages: list[dict[str, Any]],
    field_name: str,
) -> dict[str, Any] | None:
    values = _non_empty_text_values(sequence_packages, field_name)
    if len(values) != 1:
        return None
    return _field_value(next(iter(values)), "ectd_sequence_package")


def _available_context_values(submission_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    context = dict(submission_context or {})
    project_context = dict(context.get("ectd_project_context", {}) or {})
    envelope = dict(context.get("envelope", {}) or {})
    sequence_packages = [
        dict(item)
        for item in list(project_context.get("sequence_packages", []) or [])
        if isinstance(item, dict)
    ]

    sequence_application_type = _single_sequence_package_value(sequence_packages, "application_type")
    sequence_product_type = _single_sequence_package_value(sequence_packages, "product_type")
    sequence_type = _single_sequence_package_value(sequence_packages, "sequence_type")

    return {
        "application_type": _first_available_fact(
            sequence_application_type,
            _field_value(context.get("application_type"), "submission_context"),
            _field_value(envelope.get("application-type"), "ectd_envelope"),
        ),
        "registration_class": _field_value(context.get("registration_class"), "submission_context"),
        "product_type": _first_available_fact(
            sequence_product_type,
            _field_value(context.get("product_type"), "submission_context"),
            _field_value(envelope.get("product-type"), "ectd_envelope"),
        ),
        "submission_stage": _first_available_fact(
            sequence_type,
            _field_value(context.get("submission_stage"), "submission_context"),
            _field_value(context.get("sequence_type"), "submission_context"),
        ),
        "package_scope": _field_value(
            context.get("upload_mode") or context.get("package_scope"),
            "submission_context",
        ),
        "clinical_trial_completion_status": _field_value(
            context.get("clinical_trial_completion_status"),
            "submission_context",
        ),
        "sequence_package_count": _field_value(
            project_context.get("sequence_package_count"),
            "ectd_project_context",
        ),
        "regulatory_activity_count": _field_value(
            project_context.get("regulatory_activity_count"),
            "ectd_project_context",
        ),
        "application_project_count": _field_value(
            project_context.get("application_project_count"),
            "ectd_project_context",
        ),
    }


def _prerequisite_fact_payloads(submission_context: dict[str, Any]) -> list[dict[str, Any]]:
    values = _available_context_values(submission_context)
    payloads: list[dict[str, Any]] = []
    for definition in _PREREQUISITE_FACTS:
        fact_key = definition["fact_key"]
        fact_value = values.get(fact_key, {})
        value = fact_value.get("value")
        has_value = value is not None and value != "" and value != []
        source = fact_value.get("source") if has_value else None
        payloads.append(
            {
                "fact_key": fact_key,
                "label": definition["label"],
                "description": definition["description"],
                "status": "present" if has_value else "missing",
                "value": value if has_value else None,
                "source": source,
                "source_label": _source_label(source),
                "evidence_status": _fact_evidence_status(has_value),
                "confidence": _fact_confidence(source, has_value),
                "review_action": _fact_review_action(has_value),
            }
        )
    return payloads


def _blocking_fact_keys(requirement: dict[str, Any]) -> list[str]:
    keys = ["application_type", "registration_class", "product_type", "submission_stage", "package_scope"]
    requirement_type = _trimmed(requirement.get("requirement_type"))
    applicable_stage = _trimmed(requirement.get("applicable_stage"))
    if requirement_type == "clinical_data_submission" or applicable_stage == "marketing_registration":
        keys.append("clinical_trial_completion_status")
    return keys


def _requirement_payload(requirement: dict[str, Any], missing_fact_keys: set[str]) -> dict[str, Any]:
    blocking_keys = _blocking_fact_keys(requirement)
    missing_blocking_keys = [key for key in blocking_keys if key in missing_fact_keys]
    return {
        "requirement_id": _trimmed(requirement.get("requirement_id")),
        "source_clause_id": _trimmed(requirement.get("source_clause_id")),
        "source_article_no": int(requirement.get("source_article_no") or 0),
        "section_no": int(requirement.get("section_no") or 0),
        "section_title": _trimmed(requirement.get("section_title")),
        "source_heading": _trimmed(requirement.get("source_heading")),
        "registration_classes": _string_list(requirement.get("registration_classes")),
        "applicable_stage": _trimmed(requirement.get("applicable_stage")),
        "requirement_type": _trimmed(requirement.get("requirement_type")),
        "requirement_level": _trimmed(requirement.get("requirement_level")),
        "citation_anchor": _trimmed(requirement.get("citation_anchor")),
        "expected_material_evidence": _string_list(requirement.get("expected_material_evidence")),
        "review_focus": _trimmed(requirement.get("review_focus")),
        "requirement_text_preview": _preview(_trimmed(requirement.get("requirement_text"))),
        "applicability_status": "prerequisite_required" if missing_blocking_keys else "human_review",
        "default_triage": "prerequisite_required" if missing_blocking_keys else "human_review",
        "blocking_prerequisite_fact_keys": missing_blocking_keys,
        "automation_boundary": (
            "Do not hard judge applicability until prerequisite facts and local dossier scope evidence are present. "
            "Scientific adequacy and class-eligibility content remain human-review guidance."
        ),
    }


def build_dossier_checklist_projection(
    project_root: Path,
    submission_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(submission_context or {})
    matrix = _load_requirement_matrix(project_root)
    requirements = list(matrix.get("requirements", []) or [])
    prerequisite_facts = _prerequisite_fact_payloads(context)
    missing_fact_keys = {
        str(item["fact_key"])
        for item in prerequisite_facts
        if item.get("status") == "missing"
    }
    items = [_requirement_payload(dict(requirement), missing_fact_keys) for requirement in requirements]
    prerequisite_required_count = sum(
        1 for item in items if item["applicability_status"] == "prerequisite_required"
    )
    present_prerequisite_fact_count = sum(
        1 for item in prerequisite_facts if item.get("status") == "present"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "phase_a_demo_workbench",
        "regulation_id": REGULATION_ID,
        "regulation_title": _trimmed(matrix.get("regulation_title")),
        "source_file": "药品注册分类及申报资料要求.doc",
        "applicability_mode": "prerequisite_required" if missing_fact_keys else "human_review",
        "requirement_count": len(items),
        "prerequisite_facts": prerequisite_facts,
        "missing_prerequisite_fact_keys": sorted(missing_fact_keys),
        "items": items,
        "summary": {
            "requirement_count": len(items),
            "deterministic_decision_count": 0,
            "prerequisite_required_count": prerequisite_required_count,
            "human_review_count": len(items) - prerequisite_required_count,
            "present_prerequisite_fact_count": present_prerequisite_fact_count,
            "missing_prerequisite_fact_count": len(missing_fact_keys),
            "missing_prerequisite_fact_keys": sorted(missing_fact_keys),
            "evidence_boundary": (
                "This projection is a dossier checklist and prerequisite prompt layer. It is not a hard "
                "classification or scientific-adequacy verdict."
            ),
        },
    }
