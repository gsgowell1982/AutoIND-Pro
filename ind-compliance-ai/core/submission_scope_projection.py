from __future__ import annotations

from typing import Any


_SCOPE_ORDER = ("document", "sequence", "activity", "application")
_SCOPE_LABELS = {
    "document": "文档",
    "sequence": "序列",
    "activity": "注册行为",
    "application": "申请项目",
}
_UPLOAD_MODE_LABELS = {
    "single_document": "单文件上传",
    "document_batch": "文档批次",
    "ectd_sequence_candidate": "eCTD 序列候选",
    "ectd_sequence_package": "eCTD 序列包",
    "ectd_sequence_batch": "eCTD 序列批次",
    "ectd_application_project": "eCTD 申请项目",
}


def build_submission_scope_overview(submission_scope: dict[str, Any] | None) -> dict[str, Any]:
    scope = dict(submission_scope or {})
    available_scope_set = {
        str(item or "").strip()
        for item in scope.get("available_scopes", []) or []
        if str(item or "").strip()
    }
    available_scopes = [scope_name for scope_name in _SCOPE_ORDER if scope_name in available_scope_set]
    if "document" not in available_scopes:
        available_scopes.insert(0, "document")
    blocked_scopes = [scope_name for scope_name in _SCOPE_ORDER if scope_name not in set(available_scopes)]

    upload_mode = str(scope.get("upload_mode") or "single_document").strip() or "single_document"
    upload_mode_label = _UPLOAD_MODE_LABELS.get(upload_mode, upload_mode)
    project_context = dict(scope.get("ectd_project_context", {}) or {})
    sequence_package_count = int(project_context.get("sequence_package_count", 0) or 0)
    regulatory_activity_count = int(project_context.get("regulatory_activity_count", 0) or 0)
    application_project_count = int(project_context.get("application_project_count", 0) or 0)

    available_scope_labels = [_SCOPE_LABELS.get(scope_name, scope_name) for scope_name in available_scopes]
    blocked_scope_labels = [_SCOPE_LABELS.get(scope_name, scope_name) for scope_name in blocked_scopes]

    if blocked_scopes:
        scope_notice = (
            f"当前上传判定为{upload_mode_label}，仅执行 {'/'.join(available_scopes)} 级规则；"
            f"{'/'.join(blocked_scopes)} 级规则将标记为不适用。"
        )
    else:
        scope_notice = (
            f"当前上传判定为{upload_mode_label}，已具备 {'/'.join(available_scopes)} 级规则执行条件。"
        )

    return {
        "upload_mode": upload_mode,
        "upload_mode_label": upload_mode_label,
        "available_scopes": available_scopes,
        "available_scope_labels": available_scope_labels,
        "blocked_scopes": blocked_scopes,
        "blocked_scope_labels": blocked_scope_labels,
        "sequence_package_count": sequence_package_count,
        "regulatory_activity_count": regulatory_activity_count,
        "application_project_count": application_project_count,
        "scope_notice": scope_notice,
    }
