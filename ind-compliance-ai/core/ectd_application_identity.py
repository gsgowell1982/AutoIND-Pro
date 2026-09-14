from __future__ import annotations

from typing import Any

from core.ectd_naming_validation import parse_application_number


_APPLICATION_TYPE_BY_CATEGORY = {
    "clinical_trial_application": "cnapt1",
    "new_drug_application": "cnapt2",
    "generic_drug_application": "cnapt3",
}

_APPLICATION_TYPE_ALIASES = {
    "cnapt1": "clinical_trial_application",
    "clinical-trial-application": "clinical_trial_application",
    "clinical_trial_application": "clinical_trial_application",
    "clinical trial application": "clinical_trial_application",
    "clinical trial": "clinical_trial_application",
    "临床试验申请": "clinical_trial_application",
    "cnapt2": "new_drug_application",
    "new-drug-application": "new_drug_application",
    "new_drug_application": "new_drug_application",
    "new drug application": "new_drug_application",
    "new drug": "new_drug_application",
    "新药申请": "new_drug_application",
    "cnapt3": "generic_drug_application",
    "generic-drug-application": "generic_drug_application",
    "generic_drug_application": "generic_drug_application",
    "generic drug application": "generic_drug_application",
    "generic drug": "generic_drug_application",
    "仿制药申请": "generic_drug_application",
}

_STRONG_EVIDENCE_TYPES = {"envelope_application_type", "application_form_field"}
_WEAK_EVIDENCE_TYPES = {"module1_title_signal", "content_keyword_signal"}


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _resolve_category(value: Any) -> str:
    normalized = _normalize(value)
    if normalized in _APPLICATION_TYPE_ALIASES:
        return _APPLICATION_TYPE_ALIASES[normalized]
    for alias in sorted(_APPLICATION_TYPE_ALIASES, key=len, reverse=True):
        if alias in normalized:
            return _APPLICATION_TYPE_ALIASES[alias]
    return normalized if normalized in _APPLICATION_TYPE_BY_CATEGORY else ""


def _evidence_strength(evidence_type: str, explicit_strength: Any = None) -> str:
    supplied = _normalize(explicit_strength)
    if supplied in {"strong", "medium", "weak"}:
        return supplied
    if evidence_type in _STRONG_EVIDENCE_TYPES:
        return "strong"
    if evidence_type in _WEAK_EVIDENCE_TYPES:
        return "weak"
    return "medium"


def _signal_category(signal: dict[str, Any]) -> str:
    for key in ("observed_application_type", "application_type", "observed_category", "observed_value"):
        category = _resolve_category(signal.get(key))
        if category:
            return category
    return ""


def assess_application_identity(
    application_root_name: str,
    *,
    sequence_packages: list[dict[str, Any]] | None = None,
    content_signals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assess whether an application-root prefix is supported by submission evidence.

    Directory naming remains a deterministic check. This adapter only adds semantic
    evidence, so weak title/keyword signals never turn into an automatic pass.
    """

    parsed = parse_application_number(application_root_name)
    expected_category = str(parsed.get("category") or "").strip()
    evidence: list[dict[str, Any]] = []
    issue_codes: list[str] = []

    for package in sequence_packages or []:
        package_payload = dict(package or {})
        application_type = str(package_payload.get("application_type") or "").strip()
        category = _resolve_category(application_type)
        if category:
            evidence.append(
                {
                    "evidence_type": "envelope_application_type",
                    "strength": "strong",
                    "observed_application_type": application_type,
                    "observed_category": category,
                    "source_path": str(
                        package_payload.get("source_path")
                        or package_payload.get("sequence_root")
                        or ""
                    ).strip(),
                    "matched_documents": list(package_payload.get("matched_documents") or []),
                }
            )
        package_application_number = str(
            package_payload.get("application_number") or ""
        ).strip()
        if (
            package_application_number
            and str(application_root_name or "").strip()
            and package_application_number.lower() != str(application_root_name).strip().lower()
        ):
            issue_codes.append("application_number_envelope_mismatch")
            evidence.append(
                {
                    "evidence_type": "envelope_application_number",
                    "strength": "strong",
                    "observed_value": package_application_number,
                    "expected_value": str(application_root_name).strip(),
                    "source_path": str(package_payload.get("sequence_root") or "").strip(),
                }
            )

    for raw_signal in content_signals or []:
        signal = dict(raw_signal or {})
        evidence_type = str(signal.get("evidence_type") or "").strip().lower() or "content_signal"
        category = _signal_category(signal)
        if not category:
            continue
        strength = _evidence_strength(evidence_type, signal.get("strength"))
        evidence.append(
            {
                "evidence_type": evidence_type,
                "strength": strength,
                "observed_application_type": str(
                    signal.get("observed_application_type") or signal.get("application_type") or ""
                ).strip(),
                "observed_category": category,
                "observed_value": str(signal.get("observed_value") or "").strip(),
                "source_path": str(signal.get("source_path") or signal.get("filename") or "").strip(),
                "evidence_id": str(signal.get("evidence_id") or "").strip(),
            }
        )

    strong_evidence = [item for item in evidence if item.get("strength") == "strong"]
    weak_evidence = [item for item in evidence if item.get("strength") == "weak"]
    medium_evidence = [item for item in evidence if item.get("strength") == "medium"]
    strong_categories = {
        str(item.get("observed_category") or "").strip()
        for item in strong_evidence
        if str(item.get("observed_category") or "").strip()
    }
    explicit_conflicts = bool(expected_category and any(category != expected_category for category in strong_categories))
    if explicit_conflicts:
        issue_codes.append("application_type_conflict")
    if "application_number_envelope_mismatch" in issue_codes:
        explicit_conflicts = True

    if explicit_conflicts:
        status = "conflict"
        review_required = True
    elif expected_category and strong_categories and strong_categories == {expected_category}:
        status = "supported"
        review_required = False
    elif not expected_category:
        status = "insufficient_evidence"
        review_required = True
        issue_codes.append("application_root_type_unresolved")
    elif strong_evidence or medium_evidence or weak_evidence:
        status = "insufficient_evidence"
        review_required = False
    else:
        status = "insufficient_evidence"
        review_required = True
        issue_codes.append("application_type_evidence_missing")

    return {
        "application_root_name": str(application_root_name or "").strip(),
        "application_category": expected_category or None,
        "application_number_format_valid": bool(parsed.get("format_valid")),
        "status": status,
        "review_required": review_required,
        "issue_codes": sorted(set(issue_codes)),
        "evidence": evidence,
        "evidence_summary": {
            "strong_count": len(strong_evidence),
            "medium_count": len(medium_evidence),
            "weak_count": len(weak_evidence),
            "strong_categories": sorted(strong_categories),
        },
        "policy": {
            "strong_evidence_types": sorted(_STRONG_EVIDENCE_TYPES),
            "weak_evidence_types": sorted(_WEAK_EVIDENCE_TYPES),
            "weak_signals_can_auto_pass": False,
        },
    }


__all__ = ["assess_application_identity"]
