from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _normalize_outline_indices(record: dict[str, Any]) -> list[str]:
    ordered_indices: list[str] = []
    seen: set[str] = set()
    for key in (
        "missing_body_root_outline_indices",
        "missing_body_direct_child_outline_indices",
        "missing_body_bounded_subtree_outline_indices",
    ):
        for raw_value in list(record.get(key, []) or []):
            value = str(raw_value or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered_indices.append(value)
    return ordered_indices


def _resolve_first_toc_page(
    toc_sequences: list[dict[str, Any]],
    preferred_sequence_ids: list[str] | None = None,
) -> tuple[int | None, str | None]:
    preferred = [
        str(sequence_id or "").strip()
        for sequence_id in list(preferred_sequence_ids or [])
        if str(sequence_id or "").strip()
    ]
    sequence_candidates = list(toc_sequences or [])
    if preferred:
        preferred_set = set(preferred)
        preferred_candidates = [
            sequence
            for sequence in sequence_candidates
            if str(sequence.get("toc_sequence_id") or "").strip() in preferred_set
        ]
        if preferred_candidates:
            preferred_candidates.sort(
                key=lambda sequence: preferred.index(str(sequence.get("toc_sequence_id") or "").strip())
            )
            sequence_candidates = preferred_candidates
    for sequence in sequence_candidates:
        pages = [int(page) for page in list(sequence.get("pages", []) or []) if int(page or 0) > 0]
        if pages:
            return min(pages), str(sequence.get("toc_sequence_id") or "").strip() or None
        page_span = list(sequence.get("page_span", []) or [])
        if page_span:
            page = int(page_span[0] or 0)
            if page > 0:
                return page, str(sequence.get("toc_sequence_id") or "").strip() or None
    return None, None


def build_structure_audit_navigation_targets(
    record: dict[str, Any] | None,
    toc_sequences: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    payload = dict(record or {})
    normalized_sequences = list(toc_sequences or [])
    outline_indices = _normalize_outline_indices(payload)
    targets: list[dict[str, Any]] = []

    toc_page, toc_sequence_id = _resolve_first_toc_page(
        normalized_sequences,
        list(payload.get("toc_sequence_ids", []) or []),
    )
    if toc_page is not None:
        targets.append(
            {
                "target_kind": "toc",
                "label": "目录起始页",
                "page": toc_page,
                "toc_sequence_id": toc_sequence_id,
                "outline_indices": outline_indices,
            }
        )

    projected_pages = [
        int(page)
        for page in list(payload.get("projected_root_page_values", []) or [])
        if int(page or 0) > 0
    ]
    if projected_pages:
        targets.append(
            {
                "target_kind": "body_root",
                "label": "正文根章节起始页",
                "page": min(projected_pages),
                "toc_sequence_id": None,
                "outline_indices": outline_indices,
            }
        )

    return targets


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_structure_audit_export_payload(
    records: list[dict[str, Any]] | None,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    normalized_records = [dict(record) for record in list(records or [])]
    warning_records = [record for record in normalized_records if not bool(record.get("alignment_ready", False))]
    passing_records = [record for record in normalized_records if bool(record.get("alignment_ready", False))]
    document_filenames = sorted(
        {
            str(record.get("filename") or "").strip()
            for record in normalized_records
            if str(record.get("filename") or "").strip()
        }
    )
    documents_with_path_gaps = sorted(
        {
            str(record.get("filename") or "").strip()
            for record in normalized_records
            if (
                list(record.get("missing_body_direct_child_path_rows", []) or [])
                or list(record.get("missing_body_bounded_subtree_path_rows", []) or [])
            )
            and str(record.get("filename") or "").strip()
        }
    )
    root_page_conflict_count = sum(
        1
        for record in normalized_records
        for row in list(record.get("root_page_alignment_rows", []) or [])
        if bool(row.get("order_conflict", False))
        or bool(row.get("offset_conflict", False))
        or bool(row.get("span_conflict", False))
    )

    return {
        "schema_version": "structure-audit-export-v1",
        "generated_at": generated_at or _utc_now(),
        "summary": {
            "record_count": len(normalized_records),
            "warning_record_count": len(warning_records),
            "passing_record_count": len(passing_records),
            "document_count": len(document_filenames),
            "documents_with_path_gaps": documents_with_path_gaps,
            "missing_direct_child_path_count": sum(
                len(list(record.get("missing_body_direct_child_path_rows", []) or []))
                for record in normalized_records
            ),
            "missing_bounded_subtree_path_count": sum(
                len(list(record.get("missing_body_bounded_subtree_path_rows", []) or []))
                for record in normalized_records
            ),
            "root_page_conflict_count": root_page_conflict_count,
        },
        "records": normalized_records,
    }


def _format_path_row_markdown(row: dict[str, Any]) -> str:
    primary_path = str(row.get("text_path") or row.get("outline_path") or row.get("outline_index") or "").strip()
    outline_path = str(row.get("outline_path") or "").strip()
    nearest_anchor = str(row.get("nearest_body_anchor_outline_index") or "").strip()
    nearest_anchor_page = row.get("nearest_body_anchor_page")

    details: list[str] = []
    if outline_path and outline_path != primary_path:
        details.append(f"编号路径：{outline_path}")
    if nearest_anchor:
        if isinstance(nearest_anchor_page, int):
            details.append(f"最近正文锚点：{nearest_anchor}（第 {nearest_anchor_page} 页）")
        else:
            details.append(f"最近正文锚点：{nearest_anchor}")

    return f"- {primary_path}" + (f"；{'；'.join(details)}" if details else "")


def _format_root_page_conflict_row_markdown(row: dict[str, Any]) -> str:
    conflict_types = [
        label
        for label, enabled in (
            ("页序冲突", bool(row.get("order_conflict", False))),
            ("偏移冲突", bool(row.get("offset_conflict", False))),
            ("区间冲突", bool(row.get("span_conflict", False))),
        )
        if enabled
    ]
    toc_navigation_page = row.get("toc_navigation_page")
    toc_page_locator_value = row.get("toc_page_locator_value")
    body_page_start = row.get("body_page_start")
    body_page_end = row.get("body_page_end")
    suffix = f" / 冲突类型 {'、'.join(conflict_types)}" if conflict_types else ""
    return (
        f"- {row.get('outline_index') or 'unknown'}："
        f"TOC页 {toc_navigation_page} / 标注页 {toc_page_locator_value} / 正文页区间 {body_page_start}-{body_page_end}"
        f"{suffix}"
    )


def _collect_record_severity(record: dict[str, Any]) -> str:
    if list(record.get("root_page_alignment_rows", []) or []):
        for row in list(record.get("root_page_alignment_rows", []) or []):
            if bool(row.get("order_conflict", False)) or bool(row.get("offset_conflict", False)) or bool(row.get("span_conflict", False)):
                return "高"
    if list(record.get("missing_body_bounded_subtree_path_rows", []) or []):
        return "中"
    if list(record.get("missing_body_direct_child_path_rows", []) or []):
        return "低"
    return "通过"


def _build_record_problem_sections(record: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    sections: list[tuple[str, str, list[str]]] = []

    direct_child_rows = list(record.get("missing_body_direct_child_path_rows", []) or [])
    if direct_child_rows:
        sections.append(
            (
                "低",
                "### 缺失直接子级路径",
                [_format_path_row_markdown(dict(row)) for row in direct_child_rows],
            )
        )

    subtree_rows = list(record.get("missing_body_bounded_subtree_path_rows", []) or [])
    if subtree_rows:
        sections.append(
            (
                "中",
                "### 缺失子树路径",
                [_format_path_row_markdown(dict(row)) for row in subtree_rows],
            )
        )

    conflict_rows = [
        dict(row)
        for row in list(record.get("root_page_alignment_rows", []) or [])
        if bool(row.get("order_conflict", False))
        or bool(row.get("offset_conflict", False))
        or bool(row.get("span_conflict", False))
    ]
    if conflict_rows:
        sections.append(
            (
                "高",
                "### 根章节页码冲突",
                [_format_root_page_conflict_row_markdown(row) for row in conflict_rows],
            )
        )

    severity_order = {"低": 0, "中": 1, "高": 2}
    sections.sort(key=lambda item: severity_order.get(item[0], 99))
    return sections


def build_structure_audit_markdown_report(
    records: list[dict[str, Any]] | None,
    *,
    generated_at: str | None = None,
) -> str:
    payload = build_structure_audit_export_payload(records, generated_at=generated_at)
    summary = dict(payload.get("summary", {}) or {})
    normalized_records = [dict(record) for record in list(payload.get("records", []) or [])]

    warning_records = [record for record in normalized_records if not bool(record.get("alignment_ready", False))]
    passing_records = [record for record in normalized_records if bool(record.get("alignment_ready", False))]
    warning_records.sort(key=lambda item: ({"高": 0, "中": 1, "低": 2}.get(_collect_record_severity(item), 99), str(item.get("filename") or "")))
    passing_records.sort(key=lambda item: str(item.get("filename") or ""))

    lines = [
        "# 结构审计报告",
        "",
        f"生成时间：{payload.get('generated_at')}",
        "",
        "## 摘要",
        "",
        f"- 文档数：{summary.get('document_count', 0)}",
        f"- 审计记录数：{summary.get('record_count', 0)}",
        f"- 预警记录数：{summary.get('warning_record_count', 0)}",
        f"- 通过记录数：{summary.get('passing_record_count', 0)}",
        f"- 缺失直接子级路径：{summary.get('missing_direct_child_path_count', 0)}",
        f"- 缺失子树路径：{summary.get('missing_bounded_subtree_path_count', 0)}",
        f"- 根章节页码冲突：{summary.get('root_page_conflict_count', 0)}",
        "",
    ]

    if warning_records:
        lines.extend(["## 预警文档", ""])
        for record in warning_records:
            filename = str(record.get("filename") or "unknown-document").strip()
            severity = _collect_record_severity(record)
            lines.extend(
                [
                    f"### {filename}",
                    "",
                    f"- 结构状态：预警",
                    f"- 严重级别：{severity}",
                    "",
                    "#### 低严重级别问题",
                    "",
                ]
            )
            problem_sections = _build_record_problem_sections(record)
            low_sections = [section for section in problem_sections if section[0] == "低"]
            medium_sections = [section for section in problem_sections if section[0] == "中"]
            high_sections = [section for section in problem_sections if section[0] == "高"]

            if not low_sections:
                lines.append("- 无")
                lines.append("")
            else:
                for _, title, body_lines in low_sections:
                    lines.extend([title, "", *body_lines, ""])

            lines.extend(["#### 中严重级别问题", ""])
            if not medium_sections:
                lines.append("- 无")
                lines.append("")
            else:
                for _, title, body_lines in medium_sections:
                    lines.extend([title, "", *body_lines, ""])

            lines.extend(["#### 高严重级别问题", ""])
            if not high_sections:
                lines.append("- 无")
                lines.append("")
            else:
                for _, title, body_lines in high_sections:
                    lines.extend([title, "", *body_lines, ""])

    if passing_records:
        lines.extend(["## 通过文档", ""])
        for record in passing_records:
            filename = str(record.get("filename") or "unknown-document").strip()
            lines.extend(
                [
                    f"### {filename}",
                    "",
                    "- 结构状态：通过",
                    "- 严重级别：通过",
                    "",
                ]
            )

    return "\n".join(lines).strip() + "\n"
