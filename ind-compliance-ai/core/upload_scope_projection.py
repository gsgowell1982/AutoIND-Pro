from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Any


_UPLOAD_MODE_LABELS = {
    "single_document": "单文件上传",
    "document_batch": "文档批次",
    "ectd_sequence_candidate": "eCTD 序列候选",
    "ectd_sequence_package": "eCTD 序列包",
    "ectd_sequence_batch": "eCTD 序列批次",
    "ectd_application_project": "eCTD 申请项目",
}
_SCOPE_LABELS = {
    "document": "文档",
    "sequence": "序列",
    "application": "申请项目",
}
_SEQUENCE_DIR_PATTERN = re.compile(r"^\d{4}$")


def _normalize_upload_path(file_record: dict[str, Any]) -> str:
    candidate = str(file_record.get("relative_path") or file_record.get("filename") or "").strip()
    if not candidate:
        return ""
    return candidate.replace("\\", "/").strip("/")


def build_upload_scope_overview(file_records: list[dict[str, Any]] | None) -> dict[str, Any]:
    records = list(file_records or [])
    normalized_paths = [_normalize_upload_path(record) for record in records]
    normalized_paths = [path for path in normalized_paths if path]
    file_count = len(normalized_paths)

    signal_filenames = sorted(
        {
            PurePosixPath(path).name.lower()
            for path in normalized_paths
            if PurePosixPath(path).name.lower() in {"index.xml", "cn-regional.xml", "index-md5.txt"}
        }
    )
    uses_relative_paths = any("/" in path for path in normalized_paths)
    detection_basis = "relative_path" if uses_relative_paths else "filename_only"

    sequence_roots: set[tuple[str, str]] = set()
    application_roots: set[str] = set()
    for path in normalized_paths:
        parts = PurePosixPath(path).parts
        for index, part in enumerate(parts):
            if not _SEQUENCE_DIR_PATTERN.match(part):
                continue
            if index <= 0:
                continue
            application_root = parts[index - 1]
            sequence_roots.add((application_root, part))
            application_roots.add(application_root)
            break

    if file_count <= 1:
        upload_mode = "single_document"
        likely_scopes = ["document"]
    elif len(sequence_roots) >= 2 and len(application_roots) == 1:
        upload_mode = "ectd_application_project"
        likely_scopes = ["document", "sequence", "application"]
    elif len(sequence_roots) == 1:
        upload_mode = "ectd_sequence_package"
        likely_scopes = ["document", "sequence"]
    elif len(signal_filenames) >= 2:
        upload_mode = "ectd_sequence_candidate"
        likely_scopes = ["document", "sequence"]
    else:
        upload_mode = "document_batch"
        likely_scopes = ["document"]

    upload_mode_label = _UPLOAD_MODE_LABELS.get(upload_mode, upload_mode)
    likely_scope_labels = [_SCOPE_LABELS.get(scope_name, scope_name) for scope_name in likely_scopes]

    if upload_mode == "ectd_application_project":
        scope_notice = (
            "上传批次已呈现多序列结构，可优先按 eCTD 申请项目路线继续解析与规则调度。"
        )
    elif upload_mode in {"ectd_sequence_package", "ectd_sequence_candidate"}:
        scope_notice = (
            "上传批次已呈现 eCTD 序列特征，可优先按 sequence 级规则与包结构检查继续处理。"
        )
    elif upload_mode == "single_document":
        scope_notice = "当前仅检测到单文件上传，后续默认先执行 document 级解析与规则。"
    else:
        scope_notice = "当前更像常规文档批次上传，后续默认先执行 document 级解析与规则。"

    return {
        "upload_mode": upload_mode,
        "upload_mode_label": upload_mode_label,
        "likely_scopes": likely_scopes,
        "likely_scope_labels": likely_scope_labels,
        "file_count": file_count,
        "sequence_root_count": len(sequence_roots),
        "application_root_count": len(application_roots),
        "signal_filenames": signal_filenames,
        "detection_basis": detection_basis,
        "scope_notice": scope_notice,
    }
