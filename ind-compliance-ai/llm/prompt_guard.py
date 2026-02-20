def sanitize_prompt(prompt: str) -> str:
    """Apply minimal prompt guardrails for Phase 1."""
    disallowed_tokens = ["ignore previous instructions", "exfiltrate", "leak"]
    lowered = prompt.lower()
    if any(token in lowered for token in disallowed_tokens):
        raise ValueError("Prompt blocked by guard policy")
    return prompt.strip()
