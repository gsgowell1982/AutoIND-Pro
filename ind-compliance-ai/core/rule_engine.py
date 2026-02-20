from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class Rule:
    rule_id: str
    category: str
    evaluator: Callable[[dict[str, Any]], tuple[str, str]]


@dataclass(slots=True)
class RuleResult:
    rule_id: str
    category: str
    status: str
    message: str


class RuleEngine:
    """Execute hard and soft rules with profile-based filtering."""

    def __init__(self, rules: list[Rule]) -> None:
        self._rules = rules

    def run(self, material: dict[str, Any]) -> list[RuleResult]:
        results: list[RuleResult] = []
        for rule in self._rules:
            status, message = rule.evaluator(material)
            results.append(
                RuleResult(
                    rule_id=rule.rule_id,
                    category=rule.category,
                    status=status,
                    message=message,
                )
            )
        return results
