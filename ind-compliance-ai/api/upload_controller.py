from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xml"}


def validate_upload(path: Path) -> None:
    """Validate file extension for upload intake."""
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported upload type: {path.suffix}")
