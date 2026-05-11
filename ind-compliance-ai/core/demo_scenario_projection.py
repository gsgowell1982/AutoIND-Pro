from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "demo-scenario-v1"


def _summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(payload or {}).get("summary")
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def build_demo_scenario_projection(
    *,
    demo_summary: dict[str, Any],
    dossier_checklist: dict[str, Any],
    content_consistency: dict[str, Any],
    demo_flow: dict[str, Any],
    demo_report_markdown_download_url: str | None,
) -> dict[str, Any]:
    """Build a stable customer-demo scenario from existing projections only."""
    demo_flow_summary = _summary(demo_flow)
    consistency_summary = _summary(content_consistency)
    missing_facts = _string_list(dossier_checklist.get("missing_prerequisite_fact_keys"))
    walkthrough_steps = [
        str(item.get("step_id") or "").strip()
        for item in list(demo_flow.get("walkthrough_steps", []) or [])
        if isinstance(item, dict) and str(item.get("step_id") or "").strip()
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "phase_a_demo_workbench",
        "scenario_id": "phase_a_ectd_sequence_batch_readiness_demo",
        "scenario_status": "ready_for_controlled_demo",
        "verdict_policy": "no_new_verdicts_scenario_only",
        "recommended_upload_mode": "ectd_sequence_batch",
        "demo_goal": (
            "Demonstrate source readiness, prerequisite prompts, local eCTD identity consistency review, "
            "demo report download, and evidence-boundary discipline in one controlled walkthrough."
        ),
        "recommended_sample_profile": {
            "sample_kind": "two_sequence_ectd_batch_with_identity_conflict",
            "must_have_local_evidence": [
                "ectd_project_context.sequence_packages",
                "application_root_name",
                "application_number",
                "sequence_name",
                "sequence_number",
                "application_type",
                "product_type",
                "sequence_type",
            ],
            "intentionally_missing_prerequisites": missing_facts,
            "expected_review_focus": [
                "dossier_prerequisite_prompting",
                "content_consistency_review",
                "demo_report_export",
                "evidence_boundary_closeout",
            ],
        },
        "summary": {
            "walkthrough_step_count": int(demo_flow_summary.get("walkthrough_step_count") or len(walkthrough_steps)),
            "expected_issue_count": int(consistency_summary.get("issue_count") or 0),
            "expected_missing_prerequisite_count": len(missing_facts),
            "deterministic_rule_verdict_count": int(
                consistency_summary.get("deterministic_rule_verdict_count")
                or _summary(demo_summary).get("deterministic_dossier_decision_count")
                or 0
            ),
        },
        "demo_success_criteria": [
            "source_readiness",
            "dossier_prerequisites",
            "content_consistency_review",
            "report_download",
            "evidence_boundary_closeout",
        ],
        "primary_asset_urls": {
            "demo_report_markdown": demo_report_markdown_download_url,
        },
        "do_not_claim": [
            "No hard regulatory pass/fail",
            "No registration classification decision",
            "No legal adequacy decision",
            "No scientific adequacy decision",
        ],
        "evidence_boundary": (
            "This scenario is a controlled demo recipe. It does not add verdicts; it selects a stable "
            "sample shape and talk-track for existing projections, prerequisite prompts, and human-review boundaries."
        ),
    }
