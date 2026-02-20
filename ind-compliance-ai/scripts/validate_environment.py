import sys
from pathlib import Path


REQUIRED_PATHS = [
    "docs",
    "schemas",
    "parsers",
    "rules",
    "core",
    "config",
    "llm",
    "agents",
    "rag",
    "api",
    "ui/frontend",
    "tests",
]


def validate_python() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11+ is required")


def validate_structure(base_dir: Path) -> None:
    missing = [path for path in REQUIRED_PATHS if not (base_dir / path).exists()]
    if missing:
        raise RuntimeError(f"Missing required paths: {', '.join(missing)}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    validate_python()
    validate_structure(project_root)
    print("Environment validation passed")
