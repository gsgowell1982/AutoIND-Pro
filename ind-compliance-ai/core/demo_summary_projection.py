from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "demo-summary-v1"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _summary_dict(payload: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(payload or {}).get("summary")
    if isinstance(value, dict):
        return dict(value)
    return {}


def build_demo_summary_projection(
    regulatory_readiness: dict[str, Any],
    dossier_checklist: dict[str, Any],
    rule_checks: dict[str, Any],
    content_consistency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    readiness_summary = _summary_dict(regulatory_readiness)
    dossier_summary = _summary_dict(dossier_checklist)
    rule_summary = dict(rule_checks.get("summary", {}) or {})
    consistency_summary = _summary_dict(content_consistency)
    content_check_count = int(consistency_summary.get("check_count") or 0)
    comparable_content_check_count = int(consistency_summary.get("comparable_check_count") or 0)
    content_issue_count = int(consistency_summary.get("issue_count") or 0)
    missing_fact_keys = _string_list(dossier_checklist.get("missing_prerequisite_fact_keys"))
    source_count = int(readiness_summary.get("source_count") or 0)
    closed_source_count = int(readiness_summary.get("closed_source_count") or 0)
    dossier_requirement_count = int(
        dossier_checklist.get("requirement_count")
        or dossier_summary.get("requirement_count")
        or 0
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "phase_a_demo_workbench",
        "phase_status": "demo_ready_in_progress",
        "verdict_policy": "no_new_verdicts_projection_only",
        "demo_capability_codes": [
            "deterministic_ectd_validation",
            "source_readiness_matrix",
            "technical_spec_evidence_boundaries",
            "dossier_checklist_prerequisite_prompting",
            *(["content_consistency_checks"] if comparable_content_check_count else []),
            "human_review_guidance",
        ],
        "remaining_work_codes": [
            "demo_report_export",
            *([] if comparable_content_check_count else ["content_consistency_checks"]),
            "prerequisite_fact_source_ui",
        ],
        "recommended_next_action_code": "build_demo_report_summary_then_high_signal_consistency_checks",
        "recommended_next_action_label": (
            "Build a customer-demo report summary, then add only high-signal consistency checks "
            "supported by local evidence"
        ),
        "summary": {
            "source_count": source_count,
            "closed_source_count": closed_source_count,
            "source_readiness_ratio": closed_source_count / source_count if source_count else 0,
            "dossier_requirement_count": dossier_requirement_count,
            "deterministic_dossier_decision_count": int(
                dossier_summary.get("deterministic_decision_count") or 0
            ),
            "prerequisite_required_count": int(
                dossier_summary.get("prerequisite_required_count") or 0
            ),
            "human_review_count": int(dossier_summary.get("human_review_count") or 0),
            "missing_prerequisite_fact_keys": missing_fact_keys,
            "present_prerequisite_fact_count": sum(
                1
                for fact in list(dossier_checklist.get("prerequisite_facts", []) or [])
                if isinstance(fact, dict) and fact.get("status") == "present"
            ),
            "rule_check_count": int(rule_summary.get("rule_count") or len(rule_checks.get("items", []) or [])),
            "pass_rule_count": int(rule_summary.get("pass_rules") or 0),
            "warn_rule_count": int(rule_summary.get("warn_rules") or 0),
            "na_rule_count": int(rule_summary.get("na_rules") or 0),
            "risk_count": int(rule_checks.get("risk_count") or 0),
            "content_consistency_check_count": content_check_count,
            "content_consistency_comparable_check_count": comparable_content_check_count,
            "content_consistency_issue_count": content_issue_count,
        },
        "customer_demo_message": (
            "Phase A can demonstrate source readiness, deterministic eCTD checks, dossier checklist "
            "prerequisite prompting, and explicit human-review boundaries without overclaiming "
            "classification or scientific adequacy."
        ),
        "evidence_boundary": (
            "This is a demo/readiness projection, not a hard applicability, legal, classification, or "
            "scientific-adequacy verdict. Missing or ambiguous prerequisites must remain prerequisite "
            "prompts or human-review guidance."
        ),
    }
