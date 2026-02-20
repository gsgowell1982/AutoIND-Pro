from typing import Any

from agents.cmc_agent import CMCAgent
from agents.regulatory_agent import RegulatoryAgent


class AgentOrchestrator:
    """Coordinate bounded agent tasks under rule-engine control."""

    def __init__(self) -> None:
        self._cmc_agent = CMCAgent()
        self._regulatory_agent = RegulatoryAgent()

    def run(self, module_payload: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
        cmc_result = self._cmc_agent.analyze(module_payload)
        proposals = self._regulatory_agent.suggest_rule_updates(findings)
        return {
            "cmc_result": cmc_result,
            "rule_proposals": proposals,
        }
