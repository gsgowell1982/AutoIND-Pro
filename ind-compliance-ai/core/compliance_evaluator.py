from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.rule_engine import RuleEngine, RuleResult
from parsers.parser_registry import parse_file


@dataclass(slots=True)
class EvaluationResult:
    submission_profile: str
    rule_results: list[RuleResult]
    risks: list[dict[str, Any]]


class ComplianceEvaluator:
    """Main workflow entrypoint for compliance assessment."""

    def __init__(self, rule_engine: RuleEngine) -> None:
        self._rule_engine = rule_engine

    def evaluate(self, file_path: Path, submission_profile: str) -> EvaluationResult:
        parsed_material = parse_file(file_path)
        rule_results = self._rule_engine.run(parsed_material)
        risks = [
            {
                "risk_id": f"RISK-{index + 1:03d}",
                "severity": "medium",
                "reason": item.message,
            }
            for index, item in enumerate(rule_results)
            if item.status in {"fail", "warn"}
        ]
        return EvaluationResult(
            submission_profile=submission_profile,
            rule_results=rule_results,
            risks=risks,
        )
