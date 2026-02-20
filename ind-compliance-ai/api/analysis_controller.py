from pathlib import Path

from api.response_formatter import format_response


def run_analysis(file_path: str, submission_profile: str) -> dict[str, object]:
    """Phase 1 orchestration placeholder for analysis execution."""
    payload = {
        "submission_profile": submission_profile,
        "input_file": str(Path(file_path)),
        "summary": {
            "hard_failures": 0,
            "soft_risks": 0,
        },
        "rules": [],
        "risks": [],
    }
    return format_response(payload)
