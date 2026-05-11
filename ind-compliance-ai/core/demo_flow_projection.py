from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "demo-flow-v1"


def _summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(payload or {}).get("summary")
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _content_issue_focus_items(content_consistency: dict[str, Any]) -> list[str]:
    focus_items: list[str] = []
    for check in list(content_consistency.get("checks", []) or []):
        if not isinstance(check, dict):
            continue
        check_id = str(check.get("check_id") or "").strip()
        if check_id and int(check.get("issue_count") or 0) > 0:
            focus_items.append(check_id)
        for issue in list(check.get("issues", []) or []):
            if not isinstance(issue, dict):
                continue
            sequence_package_id = str(issue.get("sequence_package_id") or "").strip()
            if sequence_package_id:
                focus_items.append(sequence_package_id)
    return list(dict.fromkeys(focus_items))


def build_demo_flow_projection(
    *,
    demo_summary: dict[str, Any],
    regulatory_readiness: dict[str, Any],
    dossier_checklist: dict[str, Any],
    content_consistency: dict[str, Any],
    demo_report_markdown_download_url: str | None,
) -> dict[str, Any]:
    """Build a customer walkthrough guide from existing projections only."""
    demo_summary_summary = _summary(demo_summary)
    readiness_summary = _summary(regulatory_readiness)
    consistency_summary = _summary(content_consistency)
    missing_facts = _string_list(dossier_checklist.get("missing_prerequisite_fact_keys"))
    issue_focus_items = _content_issue_focus_items(content_consistency)

    walkthrough_steps = [
        {
            "step_id": "source_readiness",
            "title": "Source readiness",
            "status": "ready_for_demo",
            "talk_track": (
                "Start with regulatory source readiness and explain which sources are closed, bounded, "
                "or intentionally review-oriented."
            ),
            "focus_items": [
                f"closed_sources:{int(readiness_summary.get('closed_source_count') or 0)}",
                f"total_sources:{int(readiness_summary.get('source_count') or 0)}",
            ],
        },
        {
            "step_id": "dossier_prerequisites",
            "title": "Dossier prerequisites",
            "status": "prerequisite_prompt",
            "talk_track": (
                "Show that missing application facts are surfaced as prerequisites instead of forced "
                "classification or applicability decisions."
            ),
            "focus_items": missing_facts,
        },
        {
            "step_id": "content_consistency_review",
            "title": "Content consistency review",
            "status": "review_required" if issue_focus_items else "ready_for_demo",
            "talk_track": (
                "Review local eCTD identity consistency issues and keep conflicts as manual-review "
                "items until the user confirms the correct source of truth."
            ),
            "focus_items": issue_focus_items,
        },
        {
            "step_id": "report_download",
            "title": "Demo report download",
            "status": "ready_for_demo" if demo_report_markdown_download_url else "not_available",
            "talk_track": (
                "Export the same evidence-bound summary as a markdown report for customer discussion."
            ),
            "focus_items": ["demo_report_markdown"],
            "asset_url": demo_report_markdown_download_url,
        },
        {
            "step_id": "evidence_boundary_closeout",
            "title": "Evidence boundary closeout",
            "status": "ready_for_demo",
            "talk_track": (
                "Close by explaining that ambiguous, prerequisite-dependent, legal, and scientific "
                "adequacy questions stay as prerequisite prompts or human review guidance."
            ),
            "focus_items": [
                "No hard regulatory pass/fail",
                "prerequisite_required",
                "human_review",
            ],
        },
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "phase_a_demo_workbench",
        "verdict_policy": "no_new_verdicts_walkthrough_only",
        "summary": {
            "recommended_demo_mode": "customer_readiness_walkthrough",
            "walkthrough_step_count": len(walkthrough_steps),
            "source_count": int(readiness_summary.get("source_count") or 0),
            "closed_source_count": int(readiness_summary.get("closed_source_count") or 0),
            "missing_prerequisite_count": len(missing_facts),
            "issue_count": int(consistency_summary.get("issue_count") or 0),
            "deterministic_rule_verdict_count": int(
                consistency_summary.get("deterministic_rule_verdict_count")
                or demo_summary_summary.get("deterministic_dossier_decision_count")
                or 0
            ),
        },
        "walkthrough_steps": walkthrough_steps,
        "evidence_boundary": (
            "No hard regulatory pass/fail is created by this walkthrough. It is a customer-demo guide "
            "for existing projections, prerequisite prompts, and human review boundaries."
        ),
    }
