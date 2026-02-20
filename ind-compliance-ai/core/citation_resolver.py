from typing import Any


def resolve_citation(rule_id: str, rule_metadata: dict[str, Any]) -> dict[str, str]:
    """Resolve regulation anchor and snippet pointer for a rule."""
    rule_info = next((item for item in rule_metadata.get("rules", []) if item.get("rule_id") == rule_id), None)
    if not rule_info:
        return {"anchor": "unknown", "snippet": "No citation found"}
    return {
        "anchor": f"{rule_info.get('source_regulation')}@{rule_info.get('source_version')}",
        "snippet": rule_info.get("title", ""),
    }
