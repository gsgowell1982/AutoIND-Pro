from llm.llm_client import LLMClient
from llm.prompt_guard import sanitize_prompt


class ExplanationAgent:
    """Generate risk reasons without making approval decisions."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def explain_risk(self, risk_statement: str) -> str:
        prompt = sanitize_prompt(
            "Explain the regulatory risk in neutral language with citations required: "
            f"{risk_statement}"
        )
        return self._llm_client.generate(prompt)
