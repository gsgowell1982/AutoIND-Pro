from __future__ import annotations

from typing import Any


_SCOPE_ORDER = ("document", "sequence", "activity", "application")
_SCOPE_LABELS = {
    "document": "文档",
    "sequence": "序列",
    "activity": "注册行为",
    "application": "申请项目",
}
_STATUS_LABELS = {
    "confirmed": "预判确认",
    "expanded": "作用域提升",
    "narrowed": "作用域收缩",
    "divergent": "预判偏差",
    "unavailable": "待确认",
}
_REASON_LABELS = {
    "stable_scope_match": "上传预判与最终作用域一致",
    "multi_sequence_confirmed": "解析后确认多序列/更高层级结构",
    "metadata_insufficient": "解析元数据不足以支撑预判作用域",
    "missing_path_context": "上传阶段路径上下文不足",
    "scope_divergence": "上传预判与最终判定存在偏差",
    "scope_unavailable": "作用域信息暂不可用",
}
_ACTION_LABELS = {
    "proceed_with_current_scope": "继续按当前最终作用域审阅规则结果",
    "review_activity_application_rules": "继续检查 activity/application 级规则结果与项目聚合结构",
    "verify_ectd_metadata_files": "优先核查 index.xml、cn-regional.xml 中的 envelope 元数据是否完整",
    "preserve_relative_paths": "建议保持目录结构上传，确保 relative path 一并传入系统",
    "review_final_scope_only": "优先以解析后最终作用域为准，并复核上传包结构",
    "wait_for_scope_confirmation": "当前先补充材料或完成解析后，再确认高层级规则是否适用",
}


def _normalize_scope_list(raw_values: list[Any] | None) -> list[str]:
    normalized = {
        str(value or "").strip()
        for value in list(raw_values or [])
        if str(value or "").strip()
    }
    return [scope_name for scope_name in _SCOPE_ORDER if scope_name in normalized]


def _build_guidance_payload(recommended_action_code: str) -> dict[str, Any]:
    if recommended_action_code == "verify_ectd_metadata_files":
        return {
            "guidance_title": "优先核查 eCTD 元数据文件",
            "guidance_summary": "当前上传材料未能支撑更高层级作用域判断，请先核对关键 eCTD 元数据文件是否完整一致。",
            "guidance_priority": "priority",
            "guidance_steps": [
                "检查 index.xml 是否存在，且与当前提交包内容一致。",
                "检查 cn-regional.xml 是否存在且能够被系统正常解析。",
                "核对 envelope 中的申请号、序列号、注册行为等关键字段是否完整一致。",
            ],
            "guidance_targets": ["index.xml", "cn-regional.xml"],
            "guidance_target_details": [
                {"target_type": "file", "label": "index.xml", "description": "确认主索引文件存在且与当前提交包一致。"},
                {"target_type": "file", "label": "cn-regional.xml", "description": "确认区域信封文件存在且能够被系统正常解析。"},
                {
                    "target_type": "metadata_field_group",
                    "label": "envelope metadata",
                    "description": "重点核对 application-number、sequence-number、regulatory-activity-type 等关键字段。",
                },
                {
                    "target_type": "metadata_field",
                    "label": "application-number",
                    "description": "核对申请号是否存在、格式正确且与当前提交包一致。",
                },
                {
                    "target_type": "metadata_field",
                    "label": "sequence-number",
                    "description": "核对序列号是否存在、格式正确且与目录序列一致。",
                },
                {
                    "target_type": "metadata_field",
                    "label": "regulatory-activity-type",
                    "description": "核对注册行为类型是否完整且与当前提交意图一致。",
                },
                {
                    "target_type": "metadata_field",
                    "label": "sequence-type",
                    "description": "核对序列类型是否完整且与注册行为/申请类型组合一致。",
                },
            ],
            "guidance_rule_groups": [],
            "guidance_rule_group_labels": [],
        }
    if recommended_action_code == "preserve_relative_paths":
        return {
            "guidance_title": "请保持 eCTD 目录结构上传",
            "guidance_summary": "系统检测到文件名具备 eCTD 特征，但当前上传方式未保留足够的目录层级，无法确认更高层级作用域。",
            "guidance_priority": "priority",
            "guidance_steps": [
                "重新上传时保留原始目录结构，不要只传平铺后的单个 XML 或散文件。",
                "确保系统能够看到 application root、sequence root 和相对路径关系。",
                "优先上传包含 index.xml、cn-regional.xml 及其父级目录结构的完整材料。",
            ],
            "guidance_targets": ["application root folder", "sequence root folder", "relative paths"],
            "guidance_target_details": [
                {
                    "target_type": "folder",
                    "label": "application root folder",
                    "description": "上传时保留申请根目录，避免系统丢失项目层级上下文。",
                },
                {
                    "target_type": "folder",
                    "label": "sequence root folder",
                    "description": "上传时保留序列根目录，避免 sequence-scope 判断退化为单文件推断。",
                },
                {
                    "target_type": "upload_requirement",
                    "label": "relative paths",
                    "description": "不要只上传平铺文件，需保留原始相对路径关系。",
                },
            ],
            "guidance_rule_groups": [],
            "guidance_rule_group_labels": [],
        }
    if recommended_action_code == "review_activity_application_rules":
        guidance_rule_groups = ["activity", "application"]
        return {
            "guidance_title": "请继续核查 activity / application 级规则",
            "guidance_summary": "解析结果已确认当前材料具备更高层级项目上下文，建议从 activity 和 application 级规则继续审阅。",
            "guidance_priority": "attention",
            "guidance_steps": [
                "优先查看 activity 级规则组，确认注册行为相关判断是否完整一致。",
                "继续查看 application 级规则组，确认申请项目层面的聚合判断是否存在问题。",
                "结合聚合后的项目结构，复核是否存在跨序列或跨活动的不一致风险。",
            ],
            "guidance_targets": [],
            "guidance_target_details": [
                {"target_type": "rule_scope", "label": "activity", "description": "优先查看 activity 级规则组。"},
                {"target_type": "rule_scope", "label": "application", "description": "继续查看 application 级规则组。"},
            ],
            "guidance_rule_groups": guidance_rule_groups,
            "guidance_rule_group_labels": [_SCOPE_LABELS.get(group, group) for group in guidance_rule_groups],
        }
    return {
        "guidance_title": "",
        "guidance_summary": "",
        "guidance_priority": "info",
        "guidance_steps": [],
        "guidance_targets": [],
        "guidance_target_details": [],
        "guidance_rule_groups": [],
        "guidance_rule_group_labels": [],
    }


def build_scope_transition_overview(
    upload_scope_overview: dict[str, Any] | None,
    submission_scope_overview: dict[str, Any] | None,
) -> dict[str, Any]:
    upload_overview = dict(upload_scope_overview or {})
    submission_overview = dict(submission_scope_overview or {})

    upload_scopes = _normalize_scope_list(upload_overview.get("likely_scopes"))
    final_scopes = _normalize_scope_list(submission_overview.get("available_scopes"))
    added_scopes = [scope_name for scope_name in final_scopes if scope_name not in set(upload_scopes)]
    removed_scopes = [scope_name for scope_name in upload_scopes if scope_name not in set(final_scopes)]
    upload_mode = str(upload_overview.get("upload_mode") or "").strip()
    final_mode = str(submission_overview.get("upload_mode") or "").strip()
    detection_basis = str(upload_overview.get("detection_basis") or "").strip()

    if not upload_scopes and not final_scopes:
        transition_status = "unavailable"
        primary_reason_code = "scope_unavailable"
        recommended_action_code = "wait_for_scope_confirmation"
        transition_notice = "上传预判和解析后作用域均不可用，暂无法确认规则适用范围。"
    elif upload_scopes == final_scopes:
        transition_status = "confirmed"
        primary_reason_code = "stable_scope_match"
        recommended_action_code = "proceed_with_current_scope"
        transition_notice = "上传预判与解析后作用域一致，可直接按当前作用域解释规则结果。"
    elif upload_scopes and final_scopes and all(scope_name in final_scopes for scope_name in upload_scopes):
        transition_status = "expanded"
        primary_reason_code = (
            "multi_sequence_confirmed"
            if final_mode == "ectd_application_project" or "application" in added_scopes
            else "stable_scope_match"
        )
        recommended_action_code = (
            "review_activity_application_rules"
            if primary_reason_code == "multi_sequence_confirmed"
            else "proceed_with_current_scope"
        )
        transition_notice = "解析后确认了更高层级作用域，说明仅凭上传路径无法完整判断，需要结合解析元数据继续提升。"
    elif final_scopes and upload_scopes and all(scope_name in upload_scopes for scope_name in final_scopes):
        transition_status = "narrowed"
        primary_reason_code = "missing_path_context" if detection_basis == "filename_only" else "metadata_insufficient"
        recommended_action_code = (
            "preserve_relative_paths"
            if primary_reason_code == "missing_path_context"
            else "verify_ectd_metadata_files"
        )
        transition_notice = "解析后未能确认部分预判作用域，说明上传路径存在候选特征，但最终材料不足以支撑更高层级规则。"
    else:
        transition_status = "divergent"
        primary_reason_code = "scope_divergence"
        recommended_action_code = "review_final_scope_only"
        transition_notice = "上传预判与解析后作用域存在偏差，建议优先以解析后最终判定为准审阅规则结果。"

    guidance = _build_guidance_payload(recommended_action_code)

    return {
        "transition_status": transition_status,
        "transition_status_label": _STATUS_LABELS.get(transition_status, transition_status),
        "primary_reason_code": primary_reason_code,
        "primary_reason_label": _REASON_LABELS.get(primary_reason_code, primary_reason_code),
        "recommended_action_code": recommended_action_code,
        "recommended_action_label": _ACTION_LABELS.get(recommended_action_code, recommended_action_code),
        "upload_mode": upload_mode,
        "upload_mode_label": str(upload_overview.get("upload_mode_label") or "").strip(),
        "final_mode": final_mode,
        "final_mode_label": str(submission_overview.get("upload_mode_label") or "").strip(),
        "upload_scopes": upload_scopes,
        "upload_scope_labels": [_SCOPE_LABELS.get(scope_name, scope_name) for scope_name in upload_scopes],
        "final_scopes": final_scopes,
        "final_scope_labels": [_SCOPE_LABELS.get(scope_name, scope_name) for scope_name in final_scopes],
        "added_scopes": added_scopes,
        "added_scope_labels": [_SCOPE_LABELS.get(scope_name, scope_name) for scope_name in added_scopes],
        "removed_scopes": removed_scopes,
        "removed_scope_labels": [_SCOPE_LABELS.get(scope_name, scope_name) for scope_name in removed_scopes],
        "transition_notice": transition_notice,
        **guidance,
    }
