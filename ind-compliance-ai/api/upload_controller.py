from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xml", ".zip"}


def is_directory_upload_path(relative_path: str, filename: str) -> bool:
    """Identify browser folder-upload records without weakening single-file validation."""
    normalized = str(relative_path or "").replace("\\", "/").strip("/")
    name = str(filename or "").strip()
    return bool(normalized and normalized != name and "/" in normalized)


def validate_upload(path: Path, *, relative_path: str | None = None) -> None:
    """Validate file extension for upload intake."""
    if path.suffix.lower() not in ALLOWED_EXTENSIONS and not is_directory_upload_path(relative_path or path.name, path.name):
        raise ValueError(f"Unsupported upload type: {path.suffix}")
