from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "content-consistency-v1"


def _trimmed(value: Any) -> str:
    return str(value or "").strip()


def _sequence_packages(submission_scope: dict[str, Any]) -> list[dict[str, Any]]:
    project_context = dict(submission_scope.get("ectd_project_context", {}) or {})
    return [
        dict(item)
        for item in list(project_context.get("sequence_packages", []) or [])
        if isinstance(item, dict)
    ]


def _identity_issue(
    package: dict[str, Any],
    *,
    field_name: str,
    expected_field_name: str,
    observed_field_name: str,
    issue_code: str,
    message: str,
) -> dict[str, Any] | None:
    expected_value = _trimmed(package.get(expected_field_name))
    observed_value = _trimmed(package.get(observed_field_name))
    if not expected_value or not observed_value or expected_value == observed_value:
        return None
    return {
        "issue_code": issue_code,
        "field_name": field_name,
        "sequence_package_id": _trimmed(package.get("sequence_package_id")),
        "expected_value": expected_value,
        "observed_value": observed_value,
        "expected_source": expected_field_name,
        "observed_source": observed_field_name,
        "message": message,
        "review_recommendation": (
            "Manual review recommended before treating this package as identity-consistent."
        ),
    }


def _build_identity_check(
    *,
    check_id: str,
    title: str,
    description: str,
    packages: list[dict[str, Any]],
    field_name: str,
    expected_field_name: str,
    observed_field_name: str,
    issue_code: str,
    issue_message: str,
) -> dict[str, Any]:
    comparable_package_count = sum(
        1
        for package in packages
        if _trimmed(package.get(expected_field_name)) and _trimmed(package.get(observed_field_name))
    )
    issues = [
        issue
        for package in packages
        if (
            issue := _identity_issue(
                package,
                field_name=field_name,
                expected_field_name=expected_field_name,
                observed_field_name=observed_field_name,
                issue_code=issue_code,
                message=issue_message,
            )
        )
        is not None
    ]
    if issues:
        status = "review_required"
    elif comparable_package_count:
        status = "consistent"
    else:
        status = "prerequisite_required"
    return {
        "check_id": check_id,
        "title": title,
        "description": description,
        "status": status,
        "field_name": field_name,
        "comparable_package_count": comparable_package_count,
        "issue_count": len(issues),
        "issues": issues,
        "automation_boundary": (
            "This compares local package identity evidence only. It is not a regulatory pass/fail "
            "verdict; conflicts require manual review before downstream assumptions are made."
        ),
    }


def build_content_consistency_projection(
    submission_scope: dict[str, Any] | None,
) -> dict[str, Any]:
    packages = _sequence_packages(dict(submission_scope or {}))
    checks = [
        _build_identity_check(
            check_id="ectd_application_identity",
            title="eCTD application identity consistency",
            description="Compare application folder identity with envelope application-number.",
            packages=packages,
            field_name="application_number",
            expected_field_name="application_root_name",
            observed_field_name="application_number",
            issue_code="application_identity_mismatch",
            issue_message="Envelope application-number differs from the application folder name.",
        ),
        _build_identity_check(
            check_id="ectd_sequence_identity",
            title="eCTD sequence identity consistency",
            description="Compare sequence folder identity with envelope sequence-number.",
            packages=packages,
            field_name="sequence_number",
            expected_field_name="sequence_name",
            observed_field_name="sequence_number",
            issue_code="sequence_identity_mismatch",
            issue_message="Envelope sequence-number differs from the sequence folder name.",
        ),
    ]
    issue_count = sum(int(check.get("issue_count") or 0) for check in checks)
    comparable_check_count = sum(
        1
        for check in checks
        if int(check.get("comparable_package_count") or 0) > 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "phase_a_demo_workbench",
        "verdict_policy": "review_projection_only",
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "comparable_check_count": comparable_check_count,
            "issue_count": issue_count,
            "review_required_count": sum(1 for check in checks if check.get("status") == "review_required"),
            "prerequisite_required_count": sum(
                1 for check in checks if check.get("status") == "prerequisite_required"
            ),
            "deterministic_rule_verdict_count": 0,
        },
        "evidence_boundary": (
            "Content consistency checks are local evidence projections for manual review. They do not "
            "create hard regulatory pass/fail, classification, legal, or scientific-adequacy verdicts."
        ),
    }
