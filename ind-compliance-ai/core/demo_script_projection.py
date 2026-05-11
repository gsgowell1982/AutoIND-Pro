from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "demo-script-v1"


def _summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(payload or {}).get("summary")
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _line_items(items: list[str]) -> list[str]:
    if not items:
        return ["- None"]
    return [f"- {item}" for item in items]


def build_demo_script_markdown(
    *,
    demo_scenario: dict[str, Any],
    demo_flow: dict[str, Any],
    demo_report_markdown_download_url: str | None,
) -> str:
    """Build a customer-facing demo script from existing demo projections only."""
    scenario_id = str(demo_scenario.get("scenario_id") or "").strip()
    scenario_policy = str(demo_scenario.get("verdict_policy") or "").strip()
    demo_mode = str(_summary(demo_flow).get("recommended_demo_mode") or "").strip()
    demo_goal = str(demo_scenario.get("demo_goal") or "").strip()
    evidence_boundary = str(demo_scenario.get("evidence_boundary") or "").strip()
    sample_profile = dict(demo_scenario.get("recommended_sample_profile") or {})
    must_have_evidence = _string_list(sample_profile.get("must_have_local_evidence"))
    missing_prerequisites = _string_list(sample_profile.get("intentionally_missing_prerequisites"))
    do_not_claim = _string_list(demo_scenario.get("do_not_claim"))

    lines = [
        "# AutoIND-Pro Customer Demo Script",
        "",
        f"- Schema: {SCHEMA_VERSION}",
        f"- Scenario: {scenario_id}",
        f"- Demo mode: {demo_mode}",
        "- Verdict policy: no_new_verdicts_script_only",
        f"- Source scenario policy: {scenario_policy}",
        f"- Report asset: {demo_report_markdown_download_url or 'not_available'}",
        "",
        "## Demo Goal",
        demo_goal or "Demonstrate the current Phase A workbench flow without adding new verdicts.",
        "",
        "## Walkthrough",
    ]

    for index, step in enumerate(list(demo_flow.get("walkthrough_steps", []) or []), start=1):
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("step_id") or "").strip()
        title = str(step.get("title") or step_id or f"Step {index}").strip()
        status = str(step.get("status") or "").strip()
        talk_track = str(step.get("talk_track") or "").strip()
        focus_items = _string_list(step.get("focus_items"))
        asset_url = str(step.get("asset_url") or "").strip()
        lines.extend(
            [
                "",
                f"### {index}. {step_id or title}",
                f"- Title: {title}",
                f"- Status: {status}",
                f"- Talk track: {talk_track}",
                "- Focus items:",
                *_line_items(focus_items),
            ]
        )
        if asset_url:
            lines.append(f"- Asset: {asset_url}")

    lines.extend(
        [
            "",
            "## Sample Profile",
            f"- Sample kind: {str(sample_profile.get('sample_kind') or '').strip()}",
            "- Required local evidence:",
            *_line_items(must_have_evidence),
            "- Intentionally missing prerequisites:",
            *_line_items(missing_prerequisites),
            "",
            "## Evidence Boundary",
            evidence_boundary,
            "",
            "## Do Not Claim",
            *_line_items(do_not_claim),
            "",
            "No hard regulatory pass/fail is created by this script.",
        ]
    )

    return "\n".join(lines).strip() + "\n"
