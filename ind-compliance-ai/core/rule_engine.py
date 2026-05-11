from dataclasses import dataclass
from typing import Any, Callable


RuleEvaluationOutput = tuple[str, str] | tuple[str, str, dict[str, Any]]


@dataclass(slots=True)
class Rule:
    rule_id: str
    category: str
    evaluator: Callable[[dict[str, Any]], RuleEvaluationOutput]
    profiles: tuple[str, ...] | None = None
    citation: str | None = None
    scope: str | None = None


@dataclass(slots=True)
class RuleResult:
    rule_id: str
    category: str
    status: str
    message: str
    citation: str | None = None
    details: dict[str, Any] | None = None
    scope: str | None = None
    scope_status: str | None = None


class RuleEngine:
    """Execute hard and soft rules with profile-based filtering."""

    def __init__(self, rules: list[Rule]) -> None:
        self._rules = rules

    def run(self, material: dict[str, Any], submission_profile: str | None = None) -> list[RuleResult]:
        results: list[RuleResult] = []
        normalized_profile = _normalize_submission_profile(submission_profile)
        submission_scope = _normalize_submission_scope(material)
        for rule in self._rules:
            if rule.profiles:
                normalized_profiles = {
                    _normalize_submission_profile(profile)
                    for profile in rule.profiles
                }
                if normalized_profile not in normalized_profiles:
                    results.append(
                        RuleResult(
                            rule_id=rule.rule_id,
                            category=rule.category,
                            status="na",
                            message=(
                                "Rule is not applicable to submission profile "
                                f"'{submission_profile or 'unknown'}'."
                            ),
                            citation=rule.citation,
                            scope=rule.scope,
                            scope_status="profile_filtered",
                        )
                    )
                    continue
            if rule.scope and rule.scope not in set(submission_scope.get("available_scopes", []) or []):
                required_scope = str(rule.scope or "").strip() or "unknown"
                upload_mode = str(submission_scope.get("upload_mode") or "unknown").strip() or "unknown"
                available_scopes = list(submission_scope.get("available_scopes", []) or [])
                details = {
                    "scope": required_scope,
                    "scope_status": "insufficient",
                    "upload_mode": upload_mode,
                    "available_scopes": available_scopes,
                    "match_strength": "scope_insufficient",
                }
                results.append(
                    RuleResult(
                        rule_id=rule.rule_id,
                        category=rule.category,
                        status="na",
                        message=(
                            f"Rule requires {required_scope}-scope context, but current upload mode "
                            f"'{upload_mode}' only provides {available_scopes}."
                        ),
                        citation=rule.citation,
                        details=details,
                        scope=rule.scope,
                        scope_status="insufficient",
                    )
                )
                continue
            evaluation_output = rule.evaluator(material)
            status, message, details = _normalize_rule_evaluation_output(evaluation_output)
            results.append(
                RuleResult(
                    rule_id=rule.rule_id,
                    category=rule.category,
                    status=status,
                    message=message,
                    citation=rule.citation,
                    details=details,
                    scope=rule.scope,
                    scope_status="satisfied" if rule.scope else None,
                )
            )
        return results


def _normalize_submission_profile(submission_profile: str | None) -> str:
    return str(submission_profile or "unknown").strip().lower()


def _normalize_submission_scope(material: dict[str, Any]) -> dict[str, Any]:
    scope = dict(material.get("submission_scope", {}) or {})
    available_scopes = list(scope.get("available_scopes", []) or [])
    if "document" not in available_scopes:
        available_scopes.insert(0, "document")
    scope["available_scopes"] = available_scopes
    scope["upload_mode"] = str(scope.get("upload_mode") or "single_document").strip() or "single_document"
    return scope


def _normalize_rule_evaluation_output(
    evaluation_output: RuleEvaluationOutput,
) -> tuple[str, str, dict[str, Any] | None]:
    if len(evaluation_output) == 2:
        status, message = evaluation_output
        return status, message, None
    status, message, details = evaluation_output
    normalized_details = dict(details or {}) if isinstance(details, dict) else None
    return status, message, normalized_details
