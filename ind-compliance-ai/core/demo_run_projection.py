from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "demo-run-v1"


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


def _status_counts(run_steps: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "ready_step_count": sum(1 for item in run_steps if item.get("status") == "ready_for_demo"),
        "review_required_step_count": sum(1 for item in run_steps if item.get("status") == "review_required"),
        "prerequisite_prompt_step_count": sum(1 for item in run_steps if item.get("status") == "prerequisite_prompt"),
        "not_available_step_count": sum(1 for item in run_steps if item.get("status") == "not_available"),
    }


def build_demo_run_projection(
    *,
    demo_summary: dict[str, Any],
    regulatory_readiness: dict[str, Any],
    dossier_checklist: dict[str, Any],
    content_consistency: dict[str, Any],
    demo_scenario: dict[str, Any],
    demo_flow: dict[str, Any],
    demo_report_markdown_download_url: str | None,
    demo_script_markdown_download_url: str | None,
) -> dict[str, Any]:
    """Build a controlled demo run checklist from existing projections only."""
    readiness_summary = _summary(regulatory_readiness)
    dossier_summary = _summary(dossier_checklist)
    consistency_summary = _summary(content_consistency)
    demo_summary_summary = _summary(demo_summary)
    missing_facts = _string_list(dossier_checklist.get("missing_prerequisite_fact_keys"))
    issue_focus_items = _content_issue_focus_items(content_consistency)
    scenario_id = str(demo_scenario.get("scenario_id") or "").strip()
    sample_profile = dict(demo_scenario.get("recommended_sample_profile") or {})
    sample_kind = str(sample_profile.get("sample_kind") or "").strip()
    report_ready = bool(demo_report_markdown_download_url)
    script_ready = bool(demo_script_markdown_download_url)

    run_steps = [
        {
            "step_id": "load_controlled_sample",
            "title": "Load controlled sample",
            "status": "ready_for_demo",
            "talk_track": "Use the stable Phase A eCTD sequence-batch sample shape for a predictable walkthrough.",
            "focus_items": [sample_kind or "two_sequence_ectd_batch_with_identity_conflict"],
        },
        {
            "step_id": "review_source_readiness",
            "title": "Review source readiness",
            "status": "ready_for_demo",
            "talk_track": "Start with the regulatory source readiness matrix and explain closed versus bounded sources.",
            "focus_items": [
                f"closed_sources:{int(readiness_summary.get('closed_source_count') or 0)}",
                f"total_sources:{int(readiness_summary.get('source_count') or 0)}",
            ],
        },
        {
            "step_id": "explain_prerequisite_facts",
            "title": "Explain prerequisite facts",
            "status": "prerequisite_prompt" if missing_facts else "ready_for_demo",
            "talk_track": "Show present local facts, missing prerequisite facts, confidence, and the required user action.",
            "focus_items": missing_facts,
        },
        {
            "step_id": "review_content_consistency",
            "title": "Review content consistency",
            "status": "review_required" if issue_focus_items else "ready_for_demo",
            "talk_track": "Review identity conflicts as manual-review items rather than hard regulatory failures.",
            "focus_items": issue_focus_items,
        },
        {
            "step_id": "open_demo_assets",
            "title": "Open demo assets",
            "status": "ready_for_demo" if report_ready and script_ready else "not_available",
            "talk_track": "Open the markdown report and customer demo script generated from existing projections.",
            "focus_items": ["demo_report_markdown", "demo_script_markdown"],
            "asset_urls": {
                "demo_report_markdown": demo_report_markdown_download_url,
                "demo_script_markdown": demo_script_markdown_download_url,
            },
        },
        {
            "step_id": "close_evidence_boundary",
            "title": "Close evidence boundary",
            "status": "ready_for_demo",
            "talk_track": "Close by explaining prerequisite, review-required, and human-review boundaries.",
            "focus_items": [
                "No hard regulatory pass/fail",
                "No registration classification decision",
                "No legal adequacy decision",
                "No scientific adequacy decision",
            ],
        },
    ]
    counts = _status_counts(run_steps)
    run_status = "ready"
    if counts["not_available_step_count"]:
        run_status = "asset_missing"
    elif counts["review_required_step_count"] or counts["prerequisite_prompt_step_count"]:
        run_status = "ready_with_review_items"

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "phase_a_demo_workbench",
        "run_mode": "controlled_customer_demo",
        "run_status": run_status,
        "verdict_policy": "no_new_verdicts_run_checklist_only",
        "scenario_id": scenario_id,
        "summary": {
            "run_step_count": len(run_steps),
            **counts,
            "closed_source_count": int(readiness_summary.get("closed_source_count") or 0),
            "source_count": int(readiness_summary.get("source_count") or 0),
            "missing_prerequisite_fact_count": int(
                dossier_summary.get("missing_prerequisite_fact_count") or len(missing_facts)
            ),
            "content_consistency_issue_count": int(consistency_summary.get("issue_count") or 0),
            "deterministic_rule_verdict_count": int(
                consistency_summary.get("deterministic_rule_verdict_count")
                or demo_summary_summary.get("deterministic_dossier_decision_count")
                or 0
            ),
        },
        "asset_urls": {
            "demo_report_markdown": demo_report_markdown_download_url,
            "demo_script_markdown": demo_script_markdown_download_url,
        },
        "run_steps": run_steps,
        "evidence_boundary": (
            "No hard regulatory pass/fail is created by this demo run checklist. It only orders existing "
            "projections, assets, prerequisite prompts, review-required items, and human-review boundaries."
        ),
    }
