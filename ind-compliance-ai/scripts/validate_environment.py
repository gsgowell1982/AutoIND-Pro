import argparse
import importlib
import shutil
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

REQUIRED_IMPORTS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "python-multipart": "multipart",
    "pymupdf": "pymupdf",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "httpx": "httpx",
    "numpy": "numpy",
    "rapidocr_onnxruntime": "rapidocr_onnxruntime",
    "onnxruntime": "onnxruntime",
}


def validate_python() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
        raise RuntimeError("Python 3.11 or 3.12 is required")


def validate_structure(base_dir: Path) -> None:
    missing = [path for path in REQUIRED_PATHS if not (base_dir / path).exists()]
    if missing:
        raise RuntimeError(f"Missing required paths: {', '.join(missing)}")


def validate_python_dependencies() -> None:
    missing: list[str] = []
    for package_name, import_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            missing.append(f"{package_name} ({type(exc).__name__}: {exc})")
    if missing:
        raise RuntimeError(
            "Missing or broken required Python packages: "
            + ", ".join(missing)
            + ". Install them with `python -m pip install -r requirements.txt`. "
            + "This project's protected PDF regression gate requires vector-table OCR support for wide borderless tables."
        )


def validate_frontend_tooling() -> None:
    missing = [tool for tool in ("node", "npm") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "Missing frontend tooling: "
            + ", ".join(missing)
            + ". Install Node.js so the React frontend can start."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local ind-compliance-ai environment")
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip node/npm checks for API-only environments.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    validate_python()
    validate_structure(project_root)
    validate_python_dependencies()
    if not args.skip_frontend:
        validate_frontend_tooling()
    print("Environment validation passed")


if __name__ == "__main__":
    main()
