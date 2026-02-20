from typing import Any


def format_response(result: dict[str, Any]) -> dict[str, Any]:
    """Ensure response payload has stable top-level keys."""
    return {
        "submission_profile": result.get("submission_profile", "unknown"),
        "summary": result.get("summary", {}),
        "rules": result.get("rules", []),
        "risks": result.get("risks", []),
    }
