from __future__ import annotations

from hashlib import md5, sha256
from pathlib import Path, PurePosixPath
import re
from typing import Any

from core.ectd_naming_validation import parse_application_number


_SEQUENCE_RE = re.compile(r"^\d{4}$")


def _normalize_relative_path(value: str) -> str:
    path = str(value or "").replace("\\", "/").strip("/")
    if not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
        raise ValueError(f"Invalid relative path: {value}")
    return str(PurePosixPath(path))


def _file_entry(relative_path: str, content: bytes, source_path: Path | None = None) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "name": PurePosixPath(relative_path).name,
        "extension": Path(relative_path).suffix.lower(),
        "size": len(content),
        "sha256": sha256(content).hexdigest(),
        "md5": md5(content).hexdigest(),
        **({"source_path": str(source_path)} if source_path is not None else {}),
    }


def build_package_inventory(
    root: Path,
    *,
    file_records: list[dict[str, Any]] | None = None,
    explicit_directory_paths: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    files: list[dict[str, Any]] = []
    directory_paths: set[str] = set()
    seen_paths: set[str] = set()

    if file_records is not None:
        for record in file_records:
            relative_path = _normalize_relative_path(str(record.get("relative_path") or record.get("filename") or ""))
            if relative_path in seen_paths:
                raise ValueError(f"Duplicate normalized path: {relative_path}")
            seen_paths.add(relative_path)
            content = record.get("content")
            if content is None:
                source = Path(str(record.get("path") or ""))
                content = source.read_bytes()
            if isinstance(content, str):
                content = content.encode("utf-8")
            files.append(_file_entry(relative_path, bytes(content), Path(record["path"]) if record.get("path") else None))
            parts = PurePosixPath(relative_path).parts[:-1]
            directory_paths.update("/".join(parts[:i]) for i in range(1, len(parts) + 1))
    else:
        if not root.exists():
            raise ValueError(f"Package root does not exist: {root}")
        for directory in root.rglob("*"):
            relative = _normalize_relative_path(directory.relative_to(root).as_posix())
            if directory.is_dir():
                directory_paths.add(relative)
                continue
            if not directory.is_file():
                raise ValueError(f"Unsupported package entry: {relative}")
            if relative in seen_paths:
                raise ValueError(f"Duplicate normalized path: {relative}")
            seen_paths.add(relative)
            files.append(_file_entry(relative, directory.read_bytes(), directory))

    files.sort(key=lambda item: item["relative_path"])
    directory_paths.update(
        "/".join(PurePosixPath(item["relative_path"]).parts[:i])
        for item in files
        for i in range(1, len(PurePosixPath(item["relative_path"]).parts))
    )
    for directory_path in explicit_directory_paths or []:
        normalized = _normalize_relative_path(directory_path)
        directory_paths.add(normalized)
        parts = PurePosixPath(normalized).parts
        directory_paths.update("/".join(parts[:i]) for i in range(1, len(parts)))
    directories = [{"relative_path": path, "name": PurePosixPath(path).name} for path in sorted(directory_paths)]

    applications: dict[str, dict[str, Any]] = {}
    for path in sorted(directory_paths | {item["relative_path"] for item in files}):
        parts = PurePosixPath(path).parts
        if len(parts) < 2 or not _SEQUENCE_RE.match(parts[1]):
            continue
        application = applications.setdefault(parts[0], {"name": parts[0], "relative_path": parts[0], "sequences": []})
        sequence_path = f"{parts[0]}/{parts[1]}"
        if not any(item["relative_path"] == sequence_path for item in application["sequences"]):
            application["sequences"].append({"name": parts[1], "relative_path": sequence_path})

    for application in applications.values():
        parsed = parse_application_number(application["name"])
        application.update(
            {
                "application_category": parsed["category"],
                "application_year": parsed["year"],
                "application_serial": parsed["serial"],
                "application_number_format_valid": parsed["format_valid"],
            }
        )

    return {
        "source_root": str(root),
        "files": files,
        "file_paths": [item["relative_path"] for item in files],
        "directories": directories,
        "directory_paths": [item["relative_path"] for item in directories],
        "application_roots": list(applications.values()),
        "inventory_diagnostics": [],
    }
