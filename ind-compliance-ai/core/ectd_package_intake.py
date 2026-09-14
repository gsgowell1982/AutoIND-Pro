from __future__ import annotations

from hashlib import sha256
import io
from pathlib import Path
import zipfile
from typing import Any

from core.ectd_package_inventory import build_package_inventory


DEFAULT_MAX_FILES = 50_000
DEFAULT_MAX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024 * 1024


def inventory_from_zip_bytes(
    payload: bytes,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> dict[str, Any]:
    if not payload:
        raise ValueError("ZIP payload is empty")
    source_hash = sha256(payload).hexdigest()
    records: list[dict[str, Any]] = []
    explicit_directories: set[str] = set()
    total_size = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        if len(entries) > max_files:
            raise ValueError("ZIP contains too many entries")
        for entry in entries:
            name = entry.filename.replace("\\", "/")
            if entry.is_dir():
                normalized = name.strip("/")
                if normalized:
                    if normalized.startswith("/") or Path(normalized).drive or ".." in Path(normalized).parts:
                        raise ValueError(f"ZIP entry has unsafe path: {entry.filename}")
                    explicit_directories.add(normalized)
                continue
            if entry.create_system == 3 and ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f"ZIP symlink entries are not allowed: {entry.filename}")
            if name.startswith("/") or Path(name).drive or ".." in Path(name).parts:
                raise ValueError(f"ZIP entry has unsafe path: {entry.filename}")
            total_size += entry.file_size
            if total_size > max_uncompressed_bytes:
                raise ValueError("ZIP uncompressed size exceeds limit")
            records.append({"relative_path": name, "content": archive.read(entry)} )
    inventory = build_package_inventory(
        Path("<zip>"),
        file_records=records,
        explicit_directory_paths=sorted(explicit_directories),
    )
    inventory["source_kind"] = "zip"
    inventory["source_archive_sha256"] = source_hash
    return inventory


def records_from_zip_bytes(
    payload: bytes,
    *,
    output_root: Path,
    max_files: int = DEFAULT_MAX_FILES,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = inventory_from_zip_bytes(
        payload,
        max_files=max_files,
        max_uncompressed_bytes=max_uncompressed_bytes,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            relative_path = entry.filename.replace("\\", "/")
            target = output_root.joinpath(*relative_path.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
            records.append({"relative_path": relative_path, "path": str(target)})
    return records, inventory
