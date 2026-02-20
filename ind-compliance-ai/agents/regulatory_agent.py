from typing import Any


class RegulatoryAgent:
    """Supports rule maintenance with explicit source linkage."""

    def suggest_rule_updates(self, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "proposal_id": f"PROP-{index + 1:03d}",
                "change": "Review rule wording",
                "evidence": finding,
            }
            for index, finding in enumerate(findings)
        ]
