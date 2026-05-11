from __future__ import annotations

from typing import Any


def _summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(payload or {}).get("summary")
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _line_items(items: list[str]) -> str:
    return ", ".join(items) if items else "None"


def build_demo_report_markdown(
    *,
    demo_summary: dict[str, Any],
    regulatory_readiness: dict[str, Any],
    dossier_checklist: dict[str, Any],
    content_consistency: dict[str, Any],
) -> str:
    """Build a concise demo report from existing projections only."""
    demo_summary_summary = _summary(demo_summary)
    readiness_summary = _summary(regulatory_readiness)
    dossier_summary = _summary(dossier_checklist)
    consistency_summary = _summary(content_consistency)
    missing_facts = _string_list(dossier_checklist.get("missing_prerequisite_fact_keys"))

    lines = [
        "# AutoIND-Pro Phase A Demo Report",
        "",
        "## Demo Summary",
        f"- Schema: {demo_summary.get('schema_version', '')}",
        f"- Phase: {demo_summary.get('phase', '')}",
        f"- Status: {demo_summary.get('phase_status', '')}",
        f"- Verdict policy: {demo_summary.get('verdict_policy', '')}",
        (
            "- Closed regulatory sources: "
            f"{int(readiness_summary.get('closed_source_count') or demo_summary_summary.get('closed_source_count') or 0)}/"
            f"{int(readiness_summary.get('source_count') or demo_summary_summary.get('source_count') or 0)}"
        ),
        (
            "- Dossier requirements: "
            f"{int(dossier_checklist.get('requirement_count') or dossier_summary.get('requirement_count') or 0)}"
        ),
        f"- Missing prerequisite facts: {_line_items(missing_facts)}",
        (
            "- Hard dossier verdicts: "
            f"{int(dossier_summary.get('deterministic_decision_count') or demo_summary_summary.get('deterministic_dossier_decision_count') or 0)}"
        ),
        "",
        "## Content Consistency",
        f"- Schema: {content_consistency.get('schema_version', '')}",
        f"- Verdict policy: {content_consistency.get('verdict_policy', '')}",
        f"- Checks: {int(consistency_summary.get('check_count') or 0)}",
        f"- Comparable checks: {int(consistency_summary.get('comparable_check_count') or 0)}",
        f"- Content consistency issues: {int(consistency_summary.get('issue_count') or 0)}",
        f"- Hard rule verdicts: {int(consistency_summary.get('deterministic_rule_verdict_count') or 0)}",
        "",
    ]

    for check in list(content_consistency.get("checks", []) or []):
        if not isinstance(check, dict):
            continue
        lines.extend(
            [
                f"### {check.get('check_id', '')}",
                f"- Status: {check.get('status', '')}",
                f"- Comparable packages: {int(check.get('comparable_package_count') or 0)}",
                f"- Issues: {int(check.get('issue_count') or 0)}",
            ]
        )
        for issue in list(check.get("issues", []) or []):
            if not isinstance(issue, dict):
                continue
            lines.extend(
                [
                    (
                        "- Issue: "
                        f"{issue.get('sequence_package_id', '')} "
                        f"{issue.get('field_name', '')} "
                        f"expected={issue.get('expected_value', '')} "
                        f"observed={issue.get('observed_value', '')}"
                    ),
                    f"  - Code: {issue.get('issue_code', '')}",
                    f"  - Recommendation: {issue.get('review_recommendation', '')}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Evidence Boundary",
            "- No hard regulatory pass/fail is created by this report.",
            "- Ambiguous, incomplete, or prerequisite-dependent items remain prerequisite prompts or manual-review guidance.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
