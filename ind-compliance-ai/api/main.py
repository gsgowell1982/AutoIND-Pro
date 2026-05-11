from __future__ import annotations

from datetime import datetime, timezone
import base64
from io import BytesIO
import html
import json
import logging
from pathlib import Path
import re
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.upload_controller import ALLOWED_EXTENSIONS
from core.material_assessment import (
    build_compliance_result_payload,
    build_fact_consistency_rows_from_documents,
)
from core.upload_scope_projection import build_upload_scope_overview
from core.submission_scope_projection import build_submission_scope_overview
from core.scope_transition_projection import build_scope_transition_overview
from core.structure_audit_projection import (
    build_structure_audit_export_payload,
    build_structure_audit_markdown_report,
    build_structure_audit_navigation_targets,
)
from core.content_consistency_projection import build_content_consistency_projection
from core.demo_flow_projection import build_demo_flow_projection
from core.demo_report_projection import build_demo_report_markdown
from core.demo_run_projection import build_demo_run_projection
from core.demo_scenario_projection import build_demo_scenario_projection
from core.demo_script_projection import build_demo_script_markdown
from core.demo_summary_projection import build_demo_summary_projection
from core.dossier_checklist_projection import build_dossier_checklist_projection
from core.regulatory_readiness_projection import build_regulatory_readiness_projection
from core.run_manager import (
    append_run_log,
    create_run_context,
    finalize_run_context,
    persist_document_artifacts,
    persist_normalized_artifacts,
    persist_run_outputs,
)
from parsers.pdf.table_modules.cell_text_projection import project_table_cell_display_text
from parsers.parser_registry import parse_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "samples" / "uploaded"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PARSED_MARKDOWN_DIR = PROJECT_ROOT / "output" / "parsed_markdown"
PARSED_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURE_AUDIT_DIR = PROJECT_ROOT / "output" / "structure_audits"
STRUCTURE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
DEMO_REPORT_DIR = PROJECT_ROOT / "output" / "demo_reports"
DEMO_REPORT_DIR.mkdir(parents=True, exist_ok=True)
DEMO_SCRIPT_DIR = PROJECT_ROOT / "output" / "demo_scripts"
DEMO_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

_SYSTEM_RULE_BASIS_LABELS: dict[str, tuple[str, str]] = {
    "material-review-contract-v1#documents.summary": (
        "材料审阅契约 v1 / 文档摘要",
        "检查解析结果是否已经形成可供规则判断使用的文档摘要、内容证据和内容单元基础。",
    ),
    "material-review-contract-v1#diagnostic_index": (
        "材料审阅契约 v1 / 解析诊断索引",
        "检查解析过程是否存在 review-required 级别的诊断阻断项，例如表格或目录结构不稳定。",
    ),
    "material-review-contract-v1#navigation_index": (
        "材料审阅契约 v1 / 导航索引",
        "检查目录、导航序列和页码映射是否内部一致，避免导航结构误导后续审阅。",
    ),
    "material-review-contract-v1#documents.classification": (
        "材料审阅契约 v1 / 文档分类",
        "检查文档是否被识别为模块 3 / CMC 或质量综述等关键分类，以支撑对应规则判断。",
    ),
    "material-review-contract-v1#fact_index": (
        "材料审阅契约 v1 / 原子事实索引",
        "检查已提取的原子事实是否完整、一致，并能被下游规则使用。",
    ),
    "material-review-contract-v1#evidence_index": (
        "材料审阅契约 v1 / 证据索引",
        "检查表格、图片与其他非文本结构是否已投影为可审阅证据。",
    ),
    "material-review-contract-v1#section_index": (
        "材料审阅契约 v1 / 章节锚点索引",
        "检查模块与章节锚点是否充分，以支持规则定位与证据追溯。",
    ),
    "material-review-contract-v1#fact_signal_index": (
        "材料审阅契约 v1 / 事实信号索引",
        "检查单元级事实信号与归一化事实之间的一致性和冲突情况。",
    ),
}

_RULE_REGULATION_BASIS_LABELS: dict[str, str] = {
    "cn_drug_registration_classification_and_dossier_requirements#art_015": "《药品注册分类及申报资料要求》第三部分（申报资料要求）第（一）项",
    "cn_drug_administration_law_implementation_regulation#art_006": "《中华人民共和国药品管理法实施条例》第六条",
}
_REGULATION_TITLE_OVERRIDES: dict[str, str] = {
    "cn_drug_registration_classification_and_dossier_requirements": "《药品注册分类及申报资料要求》",
    "cn_drug_administration_law_implementation_regulation": "《中华人民共和国药品管理法实施条例》",
    "cn_ectd_technical_specification": "《eCTD技术规范》",
}

_REGULATION_CLAUSE_LOOKUP: dict[str, dict[str, str]] | None = None
_REGULATION_REQUIREMENT_LOOKUP: dict[str, dict[str, str]] | None = None
_RULE_GROUP_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "group_id": "ectd_package_integrity",
        "title": "eCTD 包完整性",
        "description": "基于《eCTD技术规范》的申请编号、序列元数据、XML 主干、文件覆盖与受控词汇一致性校验。",
        "rule_prefixes": ("HR-ECTD-", "SR-ECTD-"),
    },
)

logger = logging.getLogger("ind_compliance.api")

JOB_STORE: dict[str, dict[str, Any]] = {}
FILE_STORE: dict[str, Path] = {}
STORE_LOCK = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)
    return sanitized or "uploaded_file"


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _normalize_text_preview(text: str, max_lines: int = 80) -> str:
    cleaned_lines = [" ".join(line.split()) for line in text.replace("\x00", "").splitlines()]
    filtered_lines = [line for line in cleaned_lines if line]
    return "\n".join(filtered_lines[:max_lines]).strip()


PREVIEW_NOISE_KEYWORDS = (
    "author",
    "lastauthor",
    "revision",
    "totaltime",
    "created",
    "lastsaved",
    "generator",
    "originator",
    "documentproperties",
    "progid",
    "file-list",
    "microsoft word",
    "word.document",
    "urn:schemas",
    "schema",
    "xmlns",
    "w3.org/tr/rec-html40",
    "colorschememapping",
    "latentstyles",
    "deflockedstate",
    "defunhidewhenused",
    "defsemihidden",
    "defqformat",
    "defpriority",
    "latentstylecount",
    "mso-",
)


PREVIEW_INLINE_NOISE_PATTERNS = (
    r"&lt;/?[a-zA-Z][^&]*&gt;",
    r"</?[a-zA-Z][^>]*>",
    r"</?[a-zA-Z][a-zA-Z0-9:-]*",
    r"\{[^{}]{0,260}\}",
    r"font-family\s*:[^;{}]+;?",
    r"\b(?:unhidewhenused|name|id|qformat|semihidden|priority|latentstylecount|deflockedstate|defunhidewhenused|defsemihidden|defqformat|defpriority)\s*=\s*\"[^\"]*\"",
    r"\b(?:en-us|zh-cn|x-none)\b",
    r"\b(?:true|false)\b",
    r"\S+\.files/\S+",
    r"\S+\.xml\b",
    r"标题\s*\d+\s*(?:字符)?",
    r"普通表格",
)


def _cjk_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _strip_inline_noise(text: str) -> str:
    cleaned = html.unescape(text.replace("\x00", " "))
    cleaned = html.unescape(cleaned)
    for pattern in PREVIEW_INLINE_NOISE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[<>]+", " ", cleaned)
    cleaned = re.sub(r"[\"'“”‘’]+\s*[;；,，]?", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _polish_plain_text(text: str) -> str:
    polished = re.sub(r"\s+", " ", text).strip()
    polished = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", polished)
    polished = re.sub(r"\s+([，。！？；：、])", r"\1", polished)
    polished = re.sub(r"([（《“【])\s+", r"\1", polished)
    polished = re.sub(r"\s+([）》”】])", r"\1", polished)
    polished = re.sub(r"[;；,，]\s*(?=[;；,，])", "", polished)
    polished = re.sub(r"^[;；,，.。:：\\s]+", "", polished)
    return polished


def _trim_to_body_anchor(text: str) -> str:
    anchors = ("中华人民共和国", "第一章", "第一条", "总则")
    anchor_positions = [text.find(anchor) for anchor in anchors if text.find(anchor) >= 0]
    if not anchor_positions:
        return text
    first_anchor = min(anchor_positions)
    if first_anchor <= 0:
        return text
    prefix = text[:first_anchor]
    if "标题" in prefix or "表格" in prefix or len(prefix) < 140:
        return text[first_anchor:].strip()
    return text


def _is_preview_noise_line(line: str) -> bool:
    lowered = line.lower()
    if any(keyword in lowered for keyword in PREVIEW_NOISE_KEYWORDS):
        return True

    iso_matches = re.findall(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}z", lowered)
    if len(iso_matches) >= 1 and sum(char.isdigit() for char in line) >= 10:
        return True

    if re.fullmatch(r"[\d\s:/\-tTzZ.]+", line):
        return True

    if len(re.findall(r"\b(?:true|false)\b", lowered)) >= 2:
        return True

    if re.search(r"\b[a-z]{2}-[a-z]{2}\b", lowered) and sum(char.isdigit() for char in line) >= 2:
        return True

    if lowered.count(".xml") >= 1 or ".files/" in lowered:
        return True

    if len(re.findall(r"\b\w+:\w+\b", lowered)) >= 2:
        return True

    if _cjk_count(line) == 0 and len(re.findall(r"[a-zA-Z]{2,}", line)) >= 8 and sum(char.isdigit() for char in line) >= 2:
        return True

    letters = sum(char.isalpha() for char in line)
    cjk_chars = _cjk_count(line)
    digit_chars = sum(char.isdigit() for char in line)
    if digit_chars >= 8 and (letters + cjk_chars) <= 22 and len(line) < 220:
        return True

    return False


def _sanitize_preview_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        normalized = _strip_inline_noise(raw_line)
        if not normalized:
            continue
        normalized = _trim_to_body_anchor(normalized)
        if not normalized:
            continue

        cjk_chars = _cjk_count(normalized)
        if cjk_chars >= 2:
            first_cjk_index = next(
                (index for index, char in enumerate(normalized) if "\u4e00" <= char <= "\u9fff"),
                -1,
            )
            if first_cjk_index > 0:
                leading = normalized[:first_cjk_index].strip()
                leading_letters = sum(char.isalpha() for char in leading)
                leading_digits = sum(char.isdigit() for char in leading)
                if leading_letters >= 4 or leading_digits >= 2:
                    normalized = normalized[first_cjk_index:].strip()

            normalized = re.sub(r"\b[a-zA-Z][a-zA-Z0-9._-]{4,}\b", " ", normalized)
            normalized = _polish_plain_text(normalized)

        if _is_preview_noise_line(normalized):
            continue
        lines.append(normalized)
    return lines


def _iter_preview_chunks(document: dict[str, Any]) -> list[str]:
    source_type = str(document.get("source_type", "")).lower()
    if source_type == "pdf":
        return [str(page.get("text", "")) for page in document.get("pages", [])]
    if source_type == "word":
        paragraphs = [str(item.get("text", "")) for item in document.get("paragraphs", [])]
        if paragraphs:
            return paragraphs
    if source_type == "presentation":
        slides = [str(item.get("text", "")) for item in document.get("slides", [])]
        if slides:
            return slides
    return [str(document.get("text", ""))]


def _build_plain_preview(document: dict[str, Any], max_chars: int = 500) -> str:
    clean_lines: list[str] = []
    for chunk in _iter_preview_chunks(document):
        clean_lines.extend(_sanitize_preview_lines(chunk))

    if not clean_lines:
        return ""

    # For Chinese-dominant docs, keep Chinese-bearing lines only.
    if any(_cjk_count(line) >= 4 for line in clean_lines):
        filtered_lines: list[str] = []
        for line in clean_lines:
            cjk_chars = _cjk_count(line)
            if cjk_chars >= 2:
                filtered_lines.append(line)
        if filtered_lines:
            clean_lines = filtered_lines

    deduped_lines: list[str] = []
    for line in clean_lines:
        if not deduped_lines or line != deduped_lines[-1]:
            deduped_lines.append(line)
    clean_lines = deduped_lines

    merged = ""
    for line in clean_lines:
        candidate = f"{merged} {line}".strip() if merged else line
        candidate = _polish_plain_text(candidate)
        if len(candidate) >= max_chars:
            return _polish_plain_text(candidate[:max_chars].rstrip()) + "..."
        merged = candidate
    return _polish_plain_text(merged)


def _estimate_first_page_text(document: dict[str, Any]) -> str:
    return _build_plain_preview(document, max_chars=500)


def _format_preview_for_display(text: str) -> str:
    polished = _polish_plain_text(text)
    if not polished:
        return ""
    # Split by Chinese punctuation for readable automatic line breaks.
    segments = [segment.strip() for segment in re.split(r"(?<=[。！？；])", polished) if segment.strip()]
    if segments:
        return "\n".join(segments)
    return polished


def _count_tokens(text: str) -> int:
    whitespace_tokens = len(re.findall(r"\S+", text))
    cjk_char_tokens = len(re.findall(r"[\u4e00-\u9fff]", text))
    return max(whitespace_tokens, 0) + cjk_char_tokens


def _build_enterprise_metrics(
    parsed_documents: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_files = len(file_records)
    parsed_count = len(parsed_documents)
    total_chars = sum(len(str(item.get("text", ""))) for item in parsed_documents)
    total_tokens = sum(_count_tokens(str(item.get("text", ""))) for item in parsed_documents)
    total_images = sum(len(item.get("image_blocks", [])) for item in parsed_documents)
    total_tables = sum(len(item.get("table_asts", [])) for item in parsed_documents)

    estimated_pages = 0
    for document in parsed_documents:
        metadata = document.get("metadata", {})
        source_type = str(document.get("source_type", "")).lower()
        page_count = metadata.get("page_count")
        if isinstance(page_count, int) and page_count > 0:
            estimated_pages += page_count
        elif source_type == "pdf":
            estimated_pages += len(document.get("pages", []))
        elif source_type == "presentation":
            estimated_pages += len(document.get("slides", []))
        elif source_type == "word":
            paragraph_count = len(document.get("paragraphs", []))
            estimated_pages += max(1, round(paragraph_count / 24)) if paragraph_count else 0

    expected_fact_keys = {"drug_name", "dosage_form", "batch_number", "manufacturing_site", "strength"}
    captured_facts = sum(
        len({key for key in document.get("atomic_facts", {}) if key in expected_fact_keys})
        for document in parsed_documents
    )
    expected_total_facts = len(expected_fact_keys) * max(parsed_count, 1)

    consistent_rows = sum(1 for row in consistency_rows if row.get("is_consistent"))
    consistency_total = len(consistency_rows)

    parser_strategy_counter: dict[str, int] = {}
    for document in parsed_documents:
        parser_hint = str(document.get("metadata", {}).get("parser_hint", "unknown"))
        parser_strategy_counter[parser_hint] = parser_strategy_counter.get(parser_hint, 0) + 1

    parse_success_rate = _safe_ratio(parsed_count, total_files) * 100
    extraction_quality_score = (
        _safe_ratio(total_chars, max(parsed_count, 1) * 5000) * 40
        + _safe_ratio(captured_facts, expected_total_facts) * 35
        + _safe_ratio(consistent_rows, max(consistency_total, 1)) * 25
    )
    extraction_quality_score = min(extraction_quality_score, 100.0)

    return {
        "total_files": total_files,
        "parsed_count": parsed_count,
        "parse_success_rate": parse_success_rate,
        "estimated_pages": estimated_pages,
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "total_images": total_images,
        "total_tables": total_tables,
        "captured_facts": captured_facts,
        "expected_total_facts": expected_total_facts,
        "fact_capture_rate": _safe_ratio(captured_facts, expected_total_facts) * 100,
        "consistency_passed": consistent_rows,
        "consistency_total": consistency_total,
        "consistency_rate": _safe_ratio(consistent_rows, max(consistency_total, 1)) * 100,
        "quality_score": extraction_quality_score,
        "parser_distribution": parser_strategy_counter,
    }


def _build_ui_markdown(
    parsed_documents: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
) -> str:
    metrics = _build_enterprise_metrics(parsed_documents, file_records, consistency_rows)

    lines = [
        "# Enterprise Parsing Executive Summary",
        "",
        f"- Generated at: {_utc_now()}",
        f"- Job scope: {metrics['total_files']} file(s), {metrics['parsed_count']} parsed successfully",
        "",
        "## Quantitative KPI Overview",
        "",
        "| Metric | Value | Interpretation |",
        "| --- | --- | --- |",
        (
            f"| Parse success rate | {metrics['parse_success_rate']:.1f}% "
            f"({metrics['parsed_count']}/{metrics['total_files']}) | Parser execution completeness |"
        ),
        f"| Estimated pages/slides | {metrics['estimated_pages']} | Approximate payload scale |",
        f"| Extracted images | {metrics['total_images']} | Image block coverage in source documents |",
        f"| Structured tables | {metrics['total_tables']} | Table AST extraction coverage |",
        f"| Extracted text volume | {metrics['total_characters']} chars / {metrics['total_tokens']} tokens | Text capture throughput |",
        (
            f"| Atomic fact capture rate | {metrics['fact_capture_rate']:.1f}% "
            f"({metrics['captured_facts']}/{metrics['expected_total_facts']}) | Cross-module key-field readiness |"
        ),
        (
            f"| Consistency pass rate | {metrics['consistency_rate']:.1f}% "
            f"({metrics['consistency_passed']}/{metrics['consistency_total']}) | Cross-document alignment quality |"
        ),
        f"| Enterprise quality score | {metrics['quality_score']:.1f}/100 | Composite extraction reliability indicator |",
        "",
        "## Parser Distribution",
        "",
    ]
    for parser_name, count in sorted(metrics["parser_distribution"].items()):
        lines.append(f"- {parser_name}: {count} file(s)")
    if not metrics["parser_distribution"]:
        lines.append("- No parser output")

    lines.extend(
        [
            "",
            "## Document Inventory",
            "",
            "| File | Type | Estimated pages/slides | Images | Tables | Extracted chars | Parser strategy |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for index, document in enumerate(parsed_documents):
        filename = str(document.get("filename", f"document-{index + 1}"))
        source_type = str(document.get("source_type", "unknown")).upper()
        metadata = document.get("metadata", {})
        parser_hint = str(metadata.get("parser_hint", "unknown"))
        page_count = metadata.get("page_count")
        if not isinstance(page_count, int) or page_count <= 0:
            if source_type == "PDF":
                page_count = len(document.get("pages", []))
            elif source_type == "PRESENTATION":
                page_count = len(document.get("slides", []))
            elif source_type == "WORD":
                paragraph_count = len(document.get("paragraphs", []))
                page_count = max(1, round(paragraph_count / 24)) if paragraph_count else 0
            else:
                page_count = 0
        extracted_chars = len(str(document.get("text", "")))
        image_count = len(document.get("image_blocks", []))
        table_count = len(document.get("table_asts", []))
        lines.append(
            f"| {filename} | {source_type} | {page_count} | {image_count} | {table_count} | "
            f"{extracted_chars} | {parser_hint} |"
        )

    toc_summary_lines = _build_toc_markdown_sections(parsed_documents, heading_level=2)
    if toc_summary_lines:
        lines.extend(["", *toc_summary_lines])

    return "\n".join(lines).strip()


def _format_toc_page_span(sequence: dict[str, Any]) -> str:
    page_span = sequence.get("page_span")
    if isinstance(page_span, list) and len(page_span) >= 2:
        start = page_span[0]
        end = page_span[1]
        if start and end:
            return str(start) if start == end else f"{start}-{end}"

    pages = [
        int(page)
        for page in sequence.get("pages", []) or []
        if isinstance(page, int) or str(page).strip().isdigit()
    ]
    if not pages:
        return ""
    return str(pages[0]) if len(set(pages)) == 1 else f"{min(pages)}-{max(pages)}"


def _format_toc_node_label(node: dict[str, Any]) -> str:
    outline_index = str(node.get("outline_index") or "").strip()
    title = str(node.get("text") or node.get("title") or "").strip()
    if outline_index and title:
        return f"{outline_index} {title}"
    return title or outline_index or "未命名目录项"


def _format_toc_node_meta(node: dict[str, Any]) -> str:
    parts: list[str] = []
    toc_page = node.get("page")
    if toc_page:
        parts.append(f"目录页: {toc_page}")

    locator = node.get("page_locator_value")
    if locator is None or locator == "":
        locator = node.get("page_locator")
    if locator is not None and str(locator).strip():
        parts.append(f"定位页码: {locator}")

    return f"（{'；'.join(parts)}）" if parts else ""


def _append_toc_node_markdown(
    lines: list[str],
    node: dict[str, Any],
    depth: int,
    remaining_budget: list[int] | None,
) -> None:
    if remaining_budget is not None:
        if remaining_budget[0] <= 0:
            return
        remaining_budget[0] -= 1

    indent = "  " * max(depth, 0)
    label = _format_toc_node_label(node)
    meta = _format_toc_node_meta(node)
    lines.append(f"{indent}- {label}{meta}")

    children = [child for child in node.get("children", []) or [] if isinstance(child, dict)]
    for child in children:
        _append_toc_node_markdown(lines, child, depth + 1, remaining_budget)


def _fallback_toc_entry_nodes(sequence: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [entry for entry in sequence.get("entries", []) or [] if isinstance(entry, dict)]
    ordered_entries = sorted(
        entries,
        key=lambda entry: int(entry.get("sequence_entry_index", entry.get("entry_index", 0)) or 0),
    )
    return [
        {
            "outline_index": entry.get("outline_index"),
            "text": entry.get("text"),
            "page": entry.get("page"),
            "page_locator": entry.get("page_locator"),
            "page_locator_value": entry.get("page_locator_value"),
            "children": [],
        }
        for entry in ordered_entries
    ]


def _build_toc_markdown_sections(
    parsed_documents: list[dict[str, Any]],
    *,
    heading_level: int,
    max_nodes_per_sequence: int | None = None,
) -> list[str]:
    lines: list[str] = []
    heading_prefix = "#" * max(1, heading_level)
    document_heading_prefix = "#" * max(1, heading_level + 1)

    documents_with_toc = [
        document
        for document in parsed_documents
        if document.get("toc_sequences")
    ]
    if not documents_with_toc:
        return lines

    lines.append(f"{heading_prefix} 解析目录结构")
    lines.append("")

    for document_index, document in enumerate(documents_with_toc):
        filename = str(document.get("filename") or f"document-{document_index + 1}")
        toc_sequences = [sequence for sequence in document.get("toc_sequences", []) or [] if isinstance(sequence, dict)]
        if not toc_sequences:
            continue

        lines.append(f"{document_heading_prefix} {filename}")
        lines.append("")

        for sequence_index, sequence in enumerate(toc_sequences, start=1):
            title = str(sequence.get("title") or f"目录序列 {sequence_index}").strip()
            page_span = _format_toc_page_span(sequence)
            entry_count = int(sequence.get("entry_count", 0) or 0)
            summary_parts = []
            if page_span:
                summary_parts.append(f"目录页: {page_span}")
            if entry_count:
                summary_parts.append(f"目录项 {entry_count} 条")
            summary = f"（{'；'.join(summary_parts)}）" if summary_parts else ""
            lines.append(f"- {title}{summary}")

            root_nodes = [node for node in sequence.get("root_nodes", []) or [] if isinstance(node, dict)]
            nodes = root_nodes or _fallback_toc_entry_nodes(sequence)
            if not nodes:
                lines.append("  - 未解析到可展示的目录项")
                continue

            remaining_budget = [max_nodes_per_sequence] if max_nodes_per_sequence is not None else None
            for node in nodes:
                before = remaining_budget[0] if remaining_budget is not None else None
                _append_toc_node_markdown(lines, node, 1, remaining_budget)
                if remaining_budget is not None and remaining_budget[0] <= 0:
                    omitted = max(0, entry_count - max_nodes_per_sequence)
                    if omitted:
                        lines.append(f"  - 其余 {omitted} 条目录项请下载完整解析 Markdown 查看")
                    break
                if remaining_budget is not None and before == remaining_budget[0]:
                    break
            lines.append("")

    return lines


def _markdown_escape_table_cell(value: Any) -> str:
    text = str(project_table_cell_display_text(value) or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return text.replace("|", "\\|")


def _format_markdown_table_cell_line_breaks(text: str) -> str:
    if "\n" not in text:
        return text
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    if not parts:
        return ""
    enum_parts: list[str] = []
    for part in parts:
        enum_parts.extend(_split_markdown_cell_enumeration(part))
    if _looks_like_short_cell_enumeration(enum_parts):
        return " / ".join(enum_parts)
    return _join_markdown_cell_continuation_parts(parts)


def _looks_like_short_cell_enumeration(parts: list[str]) -> bool:
    if len(parts) < 2 or len(parts) > 8:
        return False
    if not all(re.search(r"[\u4e00-\u9fff]", part) for part in parts):
        return False
    if any(len(part) > 18 for part in parts):
        return False
    if any(re.search(r"[。；;:：，,、.!?！？（）()]", part) for part in parts):
        return False
    if len(parts) == 2:
        if _looks_like_continuous_cjk_phrase(parts):
            return False
        if max(len(part) for part in parts) > 4:
            return False
        return _looks_like_paired_short_enum(parts)
    if not all(_looks_like_standalone_short_enum_token(part) for part in parts):
        return False
    avg_len = sum(len(part) for part in parts) / len(parts)
    return avg_len <= 8


def _join_markdown_cell_continuation_parts(parts: list[str]) -> str:
    merged = ""
    for part in parts:
        if not merged:
            merged = part
            continue
        if _should_join_markdown_cell_parts_without_space(merged, part):
            merged = f"{merged}{part}"
        else:
            merged = f"{merged} {part}"
    return merged


def _should_join_markdown_cell_parts_without_space(left: str, right: str) -> bool:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left or not right:
        return False
    if re.search(r"[\u4e00-\u9fff]$", left) and re.search(r"^[\u4e00-\u9fffA-Za-z0-9]", right):
        return True
    if re.search(r"[A-Za-z0-9]$", left) and re.search(r"^[\u4e00-\u9fff]", right):
        return True
    return False


def _looks_like_continuous_cjk_phrase(parts: list[str]) -> bool:
    if len(parts) != 2:
        return False
    left, right = parts
    if not all(re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9]+", part or "") for part in parts):
        return False
    if any(_looks_like_standalone_short_enum_token(part) for part in parts):
        return False
    return len(left) <= 6 and len(right) <= 6


def _looks_like_paired_short_enum(parts: list[str]) -> bool:
    if len(parts) != 2:
        return False
    return all(_looks_like_standalone_short_enum_token(part) for part in parts)


def _looks_like_standalone_short_enum_token(part: str) -> bool:
    token = str(part or "").strip()
    if not token or len(token) > 8:
        return False
    enum_tokens = {
        "回复",
        "撤回",
        "报告",
        "补充",
        "替换",
        "删除",
        "新增",
        "更新",
        "首次提交",
        "再次提交",
        "初始提交",
        "再注册",
        "新适应症",
        "联合用药",
        "批准",
        "不批准",
        "通过",
        "不通过",
        "适用",
        "不适用",
    }
    if token in enum_tokens:
        return True
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,8}", token):
        return True
    return False


def _markdown_linkify_visible_urls(text: str) -> str:
    url_pattern = re.compile(r"(?<!\]\()(https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)")

    def replace(match: re.Match[str]) -> str:
        url = match.group(1)
        trailing = ""
        while url and url[-1] in ".,;:!?)）]}，。；：！？":
            trailing = url[-1] + trailing
            url = url[:-1]
        if not url:
            return match.group(0)
        return f"[{url}]({url}){trailing}"

    return url_pattern.sub(replace, text)


def _bbox_area(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_intersection_area(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _as_bbox(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        return []
    try:
        return [float(item) for item in value]
    except Exception:
        return []


def _build_uri_link_records_by_page(document: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    records_by_page: dict[int, list[dict[str, Any]]] = {}
    metadata = document.get("metadata", {}) or {}
    for record in metadata.get("pdf_uri_link_annotation_records", []) or []:
        if not isinstance(record, dict):
            continue
        uri = str(record.get("uri") or "").strip()
        bbox = _as_bbox(record.get("bbox"))
        try:
            page = int(record.get("page", 0) or 0)
        except Exception:
            page = 0
        if page <= 0 or not uri or not bbox or _bbox_area(bbox) <= 0:
            continue
        records_by_page.setdefault(page, []).append({**record, "uri": uri, "bbox": bbox})
    return records_by_page


def _find_uri_for_text_block(
    block: dict[str, Any],
    uri_link_records_by_page: dict[int, list[dict[str, Any]]],
) -> str:
    text = str(block.get("text") or "").strip()
    if not text or "http://" in text or "https://" in text:
        return ""
    try:
        page = int(block.get("page", 0) or 0)
    except Exception:
        page = 0
    block_bbox = _as_bbox(block.get("bbox"))
    if page <= 0 or not block_bbox:
        return ""

    block_area = _bbox_area(block_bbox)
    if block_area <= 0:
        return ""

    best_uri = ""
    best_score = 0.0
    for record in uri_link_records_by_page.get(page, []):
        link_bbox = _as_bbox(record.get("bbox"))
        if not link_bbox:
            continue
        intersection = _bbox_intersection_area(block_bbox, link_bbox)
        if intersection <= 0:
            continue
        block_overlap = intersection / block_area
        link_overlap = intersection / max(_bbox_area(link_bbox), 1.0)
        if block_overlap < 0.45 and link_overlap < 0.08:
            continue
        score = block_overlap + (link_overlap * 0.25)
        if score > best_score:
            best_score = score
            best_uri = str(record.get("uri") or "").strip()
    return best_uri


def _markdown_escape_link_label(text: str) -> str:
    return str(text or "").replace("[", "\\[").replace("]", "\\]")


def _normalize_markdown_table_grid(block: dict[str, Any]) -> list[list[str]]:
    for key in ("display_grid", "data_grid", "grid"):
        grid = block.get(key)
        if not isinstance(grid, list) or not grid:
            continue
        normalized_rows = [
            [_markdown_escape_table_cell(cell) for cell in row]
            for row in grid
            if isinstance(row, list)
        ]
        normalized_rows = [row for row in normalized_rows if any(cell for cell in row)]
        if normalized_rows:
            return normalized_rows

    cells = [cell for cell in block.get("cells", []) or [] if isinstance(cell, dict)]
    if not cells:
        return []
    max_row = max(int(cell.get("row", 0) or 0) for cell in cells)
    max_col = max(int(cell.get("col", 0) or 0) for cell in cells)
    if max_row <= 0 or max_col <= 0:
        return []
    rows = [["" for _ in range(max_col)] for _ in range(max_row)]
    for cell in cells:
        row_index = int(cell.get("row", 0) or 0) - 1
        col_index = int(cell.get("col", 0) or 0) - 1
        if row_index < 0 or col_index < 0:
            continue
        rows[row_index][col_index] = _markdown_escape_table_cell(cell.get("text"))
    return rows


def _markdown_escape_inline_text(value: Any) -> str:
    text = str(project_table_cell_display_text(value) or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.replace("\\", "\\\\").replace("*", "\\*").replace("[", "\\[").replace("]", "\\]")


def _merged_rows_by_row(block: dict[str, Any]) -> dict[int, dict[str, Any]]:
    merged_rows: dict[int, dict[str, Any]] = {}
    for item in block.get("merged_rows", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "") not in {"section_group", "table_note_title"}:
            continue
        try:
            row_index = int(item.get("row", 0) or 0)
        except (TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if row_index <= 0 or not text:
            continue
        merged_rows[row_index] = item
    return merged_rows


def _append_markdown_pipe_table(lines: list[str], rows: list[list[str]], column_count: int) -> None:
    if not rows:
        return
    padded_rows = [row + [""] * (column_count - len(row)) for row in rows]
    header = padded_rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in range(column_count)) + " |")
    for row in padded_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def _append_markdown_table(lines: list[str], block: dict[str, Any]) -> None:
    grid = _normalize_markdown_table_grid(block)
    if not grid:
        return

    merged_rows = _merged_rows_by_row(block)
    while grid and 1 in merged_rows:
        merged_row = merged_rows[1]
        text = _markdown_escape_inline_text(merged_row.get("text") or (grid[0][0] if grid[0] else ""))
        if text:
            lines.append(f"**{text}**")
            lines.append("")
        grid = grid[1:]
        merged_rows = {
            row_index - 1: meta
            for row_index, meta in merged_rows.items()
            if row_index > 1
        }
    if not grid:
        return

    column_count = max(len(row) for row in grid)
    if not merged_rows:
        _append_markdown_pipe_table(lines, grid, column_count)
        return

    header = list(grid[0])
    segment: list[list[str]] = [header]
    for row_index, row in enumerate(grid[1:], start=2):
        merged_row = merged_rows.get(row_index)
        if merged_row:
            if len(segment) > 1:
                _append_markdown_pipe_table(lines, segment, column_count)
            text = _markdown_escape_inline_text(merged_row.get("text") or (row[0] if row else ""))
            if text:
                lines.append(f"**{text}**")
                lines.append("")
            segment = [header]
            continue
        segment.append(row)
    if len(segment) > 1:
        _append_markdown_pipe_table(lines, segment, column_count)


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if str(value or "").strip():
        return [str(value).strip()]
    return []


def _build_table_chain(table: dict[str, Any], table_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    chain = [table]
    seen = {str(table.get("table_id") or "").strip()}
    current = table
    while True:
        next_ids = _as_string_list(current.get("continued_to"))
        if not next_ids:
            break
        next_id = next_ids[0]
        if next_id in seen or next_id not in table_by_id:
            break
        next_table = table_by_id[next_id]
        chain.append(next_table)
        seen.add(next_id)
        current = next_table
    return chain


def _merge_continued_table_chain(chain: list[dict[str, Any]]) -> dict[str, Any]:
    if not chain:
        return {}
    merged = dict(chain[0])
    merged_grid: list[list[str]] = []
    merged_rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for index, table in enumerate(chain):
        grid = _normalize_markdown_table_grid(table)
        if not grid:
            continue
        local_merged_rows = _merged_rows_by_row(table)
        if index == 0:
            row_offset = len(merged_grid)
            merged_grid.extend(grid)
            for local_row, meta in local_merged_rows.items():
                meta_copy = dict(meta)
                meta_copy["row"] = row_offset + local_row
                merged_rows.append(meta_copy)
            header = grid[0] if grid else None
            continue
        continuation_rows = list(grid)
        skipped_leading_rows = 0
        if header is not None and continuation_rows and continuation_rows[0] == header:
            continuation_rows = continuation_rows[1:]
            skipped_leading_rows = 1
        before_fragment_merge_count = len(continuation_rows)
        continuation_rows = _merge_leading_continuation_fragments(merged_grid, continuation_rows)
        merged_fragment_count = before_fragment_merge_count - len(continuation_rows)
        skipped_leading_rows += merged_fragment_count
        row_offset = len(merged_grid)
        for local_row, meta in local_merged_rows.items():
            if local_row <= skipped_leading_rows:
                continue
            meta_copy = dict(meta)
            meta_copy["row"] = row_offset + local_row - skipped_leading_rows
            merged_rows.append(meta_copy)
        merged_grid.extend(continuation_rows)
    merged["display_grid"] = merged_grid
    if merged_rows:
        merged["merged_rows"] = merged_rows
    else:
        merged.pop("merged_rows", None)
    return merged


def _merge_leading_continuation_fragments(
    previous_rows: list[list[str]],
    continuation_rows: list[list[str]],
) -> list[list[str]]:
    if not previous_rows or not continuation_rows:
        return continuation_rows

    rows = [list(row) for row in continuation_rows]
    while rows and _merge_continuation_fragment_row(previous_rows[-1], rows[0]):
        rows.pop(0)
    return rows


def _merge_continuation_fragment_row(previous_row: list[str], candidate_row: list[str]) -> bool:
    previous_non_empty = _row_non_empty_indices(previous_row)
    candidate_non_empty = _row_non_empty_indices(candidate_row)
    if not previous_non_empty or len(candidate_non_empty) != 1:
        return False

    candidate_col = candidate_non_empty[0]
    if candidate_col < previous_non_empty[-1]:
        return False
    if candidate_col >= len(previous_row):
        return False

    fragment = str(candidate_row[candidate_col] or "").strip()
    if not fragment:
        return False
    previous_text = str(previous_row[candidate_col] or "").strip()
    if not previous_text:
        return False

    previous_row[candidate_col] = _join_table_cell_continuation(previous_text, fragment)
    return True


def _row_non_empty_indices(row: list[str]) -> list[int]:
    return [
        index
        for index, cell in enumerate(row)
        if str(cell or "").strip()
    ]


def _join_table_cell_continuation(base: str, fragment: str) -> str:
    if not base:
        return fragment
    if not fragment:
        return base
    projected = project_table_cell_display_text(f"{base}\n{fragment}")
    return str(projected or "")


def _split_markdown_cell_enumeration(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(" / ") if part.strip()]


def _enrich_document_ast_block(
    block: dict[str, Any],
    *,
    table_by_id: dict[str, dict[str, Any]],
    image_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    block_type = str(block.get("block_type") or "").strip().lower()
    if block_type == "table":
        table_id = str(block.get("table_id") or block.get("block_id") or "").strip()
        if table_id and table_id in table_by_id:
            return {**table_by_id[table_id], **block}
    if block_type == "image":
        image_id = str(block.get("image_id") or block.get("block_id") or "").strip()
        if image_id and image_id in image_by_id:
            return {**image_by_id[image_id], **block}
    return block


def _append_markdown_image(lines: list[str], block: dict[str, Any]) -> None:
    source_path = block.get("_source_path")
    image_id = str(block.get("image_id") or block.get("block_id") or "").strip()
    caption = str(
        block.get("caption_text")
        or block.get("title")
        or block.get("figure_ref")
        or image_id
        or "image"
    ).strip()
    alt_text = caption.replace("[", "(").replace("]", ")")
    image_markdown = None
    if isinstance(source_path, Path) and source_path.exists():
        page_number = int(block.get("page", 0) or 0)
        bbox = block.get("bbox")
        if page_number > 0 and isinstance(bbox, list) and len(bbox) == 4:
            try:
                import fitz

                with fitz.open(source_path) as pdf_document:
                    page = pdf_document.load_page(page_number - 1)
                    clip = fitz.Rect(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    )
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
                    image_bytes = pixmap.tobytes("png")
                if image_bytes:
                    image_markdown = f"![{alt_text}](data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')})"
            except Exception:
                image_markdown = None
    if image_markdown is None:
        target = f"#{image_id}" if image_id else ""
        image_markdown = f"![{alt_text}]({target})"
    lines.append(image_markdown)
    lines.append("")


def _append_markdown_text_block(lines: list[str], block: dict[str, Any]) -> None:
    text = str(block.get("text") or "").strip()
    if not text:
        return
    text = _markdown_linkify_visible_urls(text)
    lines.append(text)
    lines.append("")


_NUMBERED_BODY_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+(.+?)\s*$")


def _canonical_outline_index(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part for part in re.split(r"\.", text) if part != ""]
    normalized_parts: list[str] = []
    for part in parts:
        if not part.isdigit():
            return text
        normalized_parts.append(str(int(part)))
    while len(normalized_parts) > 1 and normalized_parts[-1] == "0":
        normalized_parts.pop()
    return ".".join(normalized_parts)


def _normalize_heading_match_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[.。．·:：;；,，、()（）\[\]【】《》<>]", "", text)
    return text.casefold()


def _strip_probable_footnote_suffix(text: str) -> str:
    stripped = str(text or "").strip()
    return re.sub(r"(?<=[\u4e00-\u9fff])\d{1,2}$", "", stripped).strip()


def _collect_toc_heading_candidates_from_node(
    node: dict[str, Any],
    *,
    depth: int,
    lookup: dict[str, list[dict[str, Any]]],
) -> None:
    outline_index = _canonical_outline_index(node.get("outline_index"))
    title = str(node.get("text") or node.get("title") or "").strip()
    if outline_index and title:
        lookup.setdefault(outline_index, []).append({"title": title, "depth": depth})
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            _collect_toc_heading_candidates_from_node(child, depth=depth + 1, lookup=lookup)


def _build_toc_heading_lookup(document: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for sequence in document.get("toc_sequences", []) or []:
        if not isinstance(sequence, dict):
            continue
        root_nodes = [node for node in sequence.get("root_nodes", []) or [] if isinstance(node, dict)]
        for node in root_nodes:
            _collect_toc_heading_candidates_from_node(node, depth=1, lookup=lookup)
        if root_nodes:
            continue
        for entry in sequence.get("entries", []) or []:
            if not isinstance(entry, dict):
                continue
            outline_index = _canonical_outline_index(entry.get("outline_index"))
            title = str(entry.get("text") or entry.get("title") or "").strip()
            if outline_index and title:
                depth = max(1, int(entry.get("outline_depth", entry.get("level", 1)) or 1))
                lookup.setdefault(outline_index, []).append({"title": title, "depth": depth})
    return lookup


def _matches_toc_heading_title(body_title: str, toc_title: str) -> bool:
    body = _normalize_heading_match_text(_strip_probable_footnote_suffix(body_title))
    toc = _normalize_heading_match_text(toc_title)
    if not body or not toc:
        return False
    if body == toc:
        return True
    if body.startswith(toc) and len(body) <= len(toc) + 2:
        return True
    return toc.startswith(body) and len(toc) <= len(body) + 2


def _markdown_heading_for_text_block(
    block: dict[str, Any],
    toc_heading_lookup: dict[str, list[dict[str, Any]]],
) -> tuple[str, int] | None:
    text = str(block.get("text") or "").strip()
    match = _NUMBERED_BODY_HEADING_RE.match(text)
    if not match:
        return None
    outline_index = _canonical_outline_index(match.group(1))
    body_title = match.group(2)
    candidates = toc_heading_lookup.get(outline_index, [])
    for candidate in candidates:
        if _matches_toc_heading_title(body_title, str(candidate.get("title") or "")):
            depth = max(1, int(candidate.get("depth", 1) or 1))
            return text, min(6, 3 + depth)
    return None


def _append_markdown_text_block_with_heading_context(
    lines: list[str],
    block: dict[str, Any],
    toc_heading_lookup: dict[str, list[dict[str, Any]]],
    uri_link_records_by_page: dict[int, list[dict[str, Any]]],
) -> None:
    heading = _markdown_heading_for_text_block(block, toc_heading_lookup)
    if heading is None:
        uri = _find_uri_for_text_block(block, uri_link_records_by_page)
        if uri:
            text = str(block.get("text") or "").strip()
            lines.append(f"[{_markdown_escape_link_label(text)}]({uri})")
            lines.append("")
        else:
            _append_markdown_text_block(lines, block)
        return
    text, level = heading
    lines.append(f"{'#' * level} {_markdown_linkify_visible_urls(text)}")
    lines.append("")


def _build_document_body_markdown_sections(document: dict[str, Any]) -> list[str]:
    document_ast = document.get("document_ast", {}) or {}
    ast_pages = [
        page
        for page in document_ast.get("pages", []) or []
        if isinstance(page, dict)
    ]
    if not ast_pages:
        return []

    lines = ["### 正文结构化内容", ""]
    table_by_id = {
        str(table.get("table_id") or "").strip(): table
        for table in document.get("table_asts", []) or []
        if isinstance(table, dict) and str(table.get("table_id") or "").strip()
    }
    image_by_id = {
        str(image.get("image_id") or "").strip(): image
        for image in document.get("image_blocks", []) or []
        if isinstance(image, dict) and str(image.get("image_id") or "").strip()
    }
    toc_heading_lookup = _build_toc_heading_lookup(document)
    uri_link_records_by_page = _build_uri_link_records_by_page(document)
    skipped_table_ids: set[str] = set()
    source_path_value = document.get("source_path")
    source_path = Path(str(source_path_value)) if str(source_path_value or "").strip() else None
    for page in ast_pages:
        page_number = page.get("page")
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        if not blocks:
            continue
        for block in blocks:
            if page_number is not None and "page" not in block:
                block = {**block, "page": page_number}
            block = _enrich_document_ast_block(
                block,
                table_by_id=table_by_id,
                image_by_id=image_by_id,
            )
            if source_path is not None:
                block = {**block, "_source_path": source_path}
            block_type = str(block.get("block_type") or "").strip().lower()
            if block_type == "table":
                table_id = str(block.get("table_id") or block.get("block_id") or "").strip()
                if table_id in skipped_table_ids:
                    continue
                if _as_string_list(block.get("continued_from")):
                    continue
                chain = _build_table_chain(block, table_by_id)
                for continuation in chain[1:]:
                    continuation_id = str(continuation.get("table_id") or "").strip()
                    if continuation_id:
                        skipped_table_ids.add(continuation_id)
                if len(chain) > 1:
                    block = _merge_continued_table_chain(chain)
                _append_markdown_table(lines, block)
            elif block_type == "image":
                _append_markdown_image(lines, block)
            elif block_type == "toc":
                continue
            else:
                _append_markdown_text_block_with_heading_context(
                    lines,
                    block,
                    toc_heading_lookup,
                    uri_link_records_by_page,
                )
    return lines


def _build_pdf_link_markdown_section(document: dict[str, Any]) -> list[str]:
    metadata = document.get("metadata", {}) or {}
    page_records = [
        record
        for record in metadata.get("pdf_link_action_page_records", []) or []
        if isinstance(record, dict)
    ]
    bookmark_uri_targets = [
        str(target).strip()
        for target in metadata.get("pdf_bookmark_uri_targets", []) or []
        if str(target).strip()
    ]
    if not page_records and not bookmark_uri_targets:
        return []

    lines = ["### PDF 超链接", ""]
    if page_records:
        lines.append("#### 页面链接注解")
        action_order = {"/GoTo": 0, "/GoToR": 1, "/Launch": 2, "/URI": 3}
        for record in page_records:
            page = record.get("page")
            count = int(record.get("link_annotation_count", 0) or 0)
            action_values = [str(item) for item in record.get("link_action_kinds", []) or [] if str(item).strip()]
            action_values = sorted(action_values, key=lambda item: (action_order.get(item, 99), item))
            actions = ", ".join(action_values) or "unknown"
            xrefs = ", ".join(str(item) for item in record.get("link_annotation_xrefs", []) or [])
            line = f"- Page {page}: {count} link annotation(s); actions: {actions}"
            if xrefs:
                line += f"; xrefs: {xrefs}"
            lines.append(line)
        lines.append("")

    if bookmark_uri_targets:
        lines.append("#### 书签 URI 目标")
        for target in bookmark_uri_targets:
            lines.append(f"- {target}")
        lines.append("")

    return lines


def _build_full_markdown(parsed_documents: list[dict[str, Any]]) -> str:
    lines = [
        "# IND Parse Snapshot (Full)",
        "",
        f"Generated at: {_utc_now()}",
        "",
    ]
    for index, document in enumerate(parsed_documents):
        filename = document.get("filename", f"document-{index + 1}")
        source_type = str(document.get("source_type", "unknown")).upper()
        lines.append(f"## {filename} ({source_type})")
        lines.append("")

        metadata = document.get("metadata", {})
        page_count = metadata.get("page_count")
        parser_hint = metadata.get("parser_hint")
        if page_count:
            lines.append(f"- Estimated pages: {page_count}")
        if parser_hint:
            lines.append(f"- Parser strategy: {parser_hint}")
        if page_count or parser_hint:
            lines.append("")

        facts = document.get("atomic_facts", {})
        if facts:
            lines.append("### Atomic Facts")
            for key, value in facts.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        toc_sections = _build_toc_markdown_sections([document], heading_level=3)
        if toc_sections:
            lines.extend(toc_sections)
            lines.append("")

        body_sections = _build_document_body_markdown_sections(document)
        if body_sections:
            lines.extend(body_sections)
            lines.append("")

        preview_text = str(document.get("text", "")).strip()
        if preview_text and not body_sections:
            lines.append("### Text Preview")
            lines.append("```text")
            lines.append(preview_text)
            lines.append("```")
            lines.append("")

    return "\n".join(lines).strip()


def _load_regulation_clause_lookup() -> dict[str, dict[str, str]]:
    global _REGULATION_CLAUSE_LOOKUP
    if _REGULATION_CLAUSE_LOOKUP is not None:
        return _REGULATION_CLAUSE_LOOKUP

    lookup: dict[str, dict[str, str]] = {}
    normalized_root = PROJECT_ROOT / "data" / "regulations" / "normalized"
    clause_files = sorted(normalized_root.glob("*.clauses.json"))
    for path in clause_files:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for clause in payload.get("clauses", []) or []:
            citation_anchor = str(((clause.get("source_locator") or {}).get("citation_anchor")) or "").strip()
            if not citation_anchor:
                continue
            regulation_id = str(clause.get("regulation_id") or "").strip()
            regulation_title = (
                _REGULATION_TITLE_OVERRIDES.get(regulation_id)
                or str(clause.get("regulation_title") or "").strip()
                or regulation_id
            )
            severity = str(clause.get("severity") or "").strip()
            if not severity:
                normalized_text = str(clause.get("normalized_text") or "").strip()
                for candidate in ("错误", "警告", "提示信息"):
                    if normalized_text.endswith(candidate):
                        severity = candidate
                        break
            lookup[citation_anchor] = {
                "label": _RULE_REGULATION_BASIS_LABELS.get(citation_anchor)
                or f"{regulation_title} / {clause.get('heading')}",
                "detail": str(clause.get("normalized_text") or "").strip(),
                "severity": severity,
                "source_clause_id": str(clause.get("clause_id") or "").strip(),
            }
    _REGULATION_CLAUSE_LOOKUP = lookup
    return lookup


def _load_regulation_requirement_lookup() -> dict[str, dict[str, str]]:
    global _REGULATION_REQUIREMENT_LOOKUP
    if _REGULATION_REQUIREMENT_LOOKUP is not None:
        return _REGULATION_REQUIREMENT_LOOKUP

    lookup: dict[str, dict[str, str]] = {}
    normalized_root = PROJECT_ROOT / "data" / "regulations" / "normalized"
    requirement_paths = sorted(normalized_root.glob("*.requirement_matrix.json"))
    for requirement_path in requirement_paths:
        if not requirement_path.exists():
            continue
        payload = json.loads(requirement_path.read_text(encoding="utf-8"))
        for requirement in payload.get("requirements", []) or []:
            requirement_id = str(requirement.get("requirement_id") or "").strip()
            if not requirement_id:
                continue
            citation_anchor = str(requirement.get("citation_anchor") or "").strip()
            regulation_id = str(requirement.get("regulation_id") or "").strip()
            regulation_title = (
                _REGULATION_TITLE_OVERRIDES.get(regulation_id)
                or str(requirement.get("regulation_title") or "").strip()
                or regulation_id
            )
            lookup[requirement_id] = {
                "label": _RULE_REGULATION_BASIS_LABELS.get(citation_anchor)
                or f"{regulation_title} / {requirement.get('source_heading')}",
                "detail": str(requirement.get("requirement_text") or "").strip(),
            }
    _REGULATION_REQUIREMENT_LOOKUP = lookup
    return lookup


def _resolve_rule_group_overall_status(group_items: list[dict[str, Any]]) -> str:
    statuses = {str(item.get("status") or "").strip() for item in group_items}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    if "pass" in statuses:
        return "pass"
    return "na"


def _build_rule_group_summaries(
    rule_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    group_summaries: list[dict[str, Any]] = []

    for group_definition in _RULE_GROUP_DEFINITIONS:
        rule_prefixes = tuple(group_definition.get("rule_prefixes") or ())
        matching_items = [
            item
            for item in rule_items
            if str(item.get("rule_id") or "").startswith(rule_prefixes)
        ]
        if not matching_items:
            continue

        fail_count = sum(1 for item in matching_items if item.get("status") == "fail")
        warn_count = sum(1 for item in matching_items if item.get("status") == "warn")
        pass_count = sum(1 for item in matching_items if item.get("status") == "pass")
        na_count = sum(1 for item in matching_items if item.get("status") == "na")
        applicable_count = sum(1 for item in matching_items if item.get("status") != "na")
        focus_rule_ids = [
            str(item.get("rule_id") or "").strip()
            for item in matching_items
            if item.get("status") in {"fail", "warn"}
        ][:3]

        basis_refs: list[dict[str, Any]] = []
        seen_basis_citations: set[str] = set()
        for item in matching_items:
            citation = str(item.get("citation") or "").strip()
            basis = dict(item.get("basis", {}) or {})
            if not citation or citation in seen_basis_citations or not basis:
                continue
            basis_refs.append(
                {
                    "citation": citation,
                    "basis_kind": str(basis.get("basis_kind") or "").strip() or "generic",
                    "basis_label": str(basis.get("basis_label") or citation).strip(),
                }
            )
            seen_basis_citations.add(citation)

        group_summaries.append(
            {
                "group_id": str(group_definition["group_id"]),
                "title": str(group_definition["title"]),
                "description": str(group_definition["description"]),
                "overall_status": _resolve_rule_group_overall_status(matching_items),
                "rule_ids": [
                    str(item.get("rule_id") or "").strip()
                    for item in matching_items
                    if str(item.get("rule_id") or "").strip()
                ],
                "focus_rule_ids": focus_rule_ids,
                "total_rules": len(matching_items),
                "applicable_rules": applicable_count,
                "fail_count": fail_count,
                "warn_count": warn_count,
                "pass_count": pass_count,
                "na_count": na_count,
                "basis_refs": basis_refs,
            }
        )

    return group_summaries


def _resolve_rule_basis(rule_item: dict[str, Any]) -> dict[str, Any] | None:
    details = dict(rule_item.get("details", {}) or {})
    citation = str(rule_item.get("citation") or details.get("citation_anchor") or "").strip()
    requirement_id = str(details.get("requirement_id") or "").strip()

    requirement_lookup = _load_regulation_requirement_lookup()
    if requirement_id and requirement_id in requirement_lookup:
        payload = requirement_lookup[requirement_id]
        return {
            "basis_kind": "regulation",
            "basis_label": payload["label"],
            "basis_detail": payload["detail"],
            "internal_anchor": citation or None,
        }

    clause_lookup = _load_regulation_clause_lookup()
    if citation and citation in clause_lookup:
        payload = clause_lookup[citation]
        return {
            "basis_kind": "regulation",
            "basis_label": payload["label"],
            "basis_detail": payload["detail"],
            "internal_anchor": citation,
        }

    if citation and citation in _SYSTEM_RULE_BASIS_LABELS:
        label, detail = _SYSTEM_RULE_BASIS_LABELS[citation]
        return {
            "basis_kind": "system",
            "basis_label": label,
            "basis_detail": detail,
            "internal_anchor": citation,
        }

    if citation:
        return {
            "basis_kind": "generic",
            "basis_label": citation,
            "basis_detail": "",
            "internal_anchor": citation,
        }
    return None


def _resolve_rule_navigation_target(
    document_lookup: dict[str, dict[str, Any]],
    *,
    filename: str,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    document = document_lookup.get(filename, {})
    evidence_by_id = {
        str(item.get("evidence_id") or "").strip(): dict(item)
        for item in document.get("content_evidence", []) or []
        if str(item.get("evidence_id") or "").strip()
    }
    for evidence_id in evidence_refs or []:
        evidence = evidence_by_id.get(str(evidence_id or "").strip())
        if not evidence:
            continue
        page = int(evidence.get("page", 0) or 0)
        source_type = str(evidence.get("source_type") or "").strip().lower()
        source_id = str(evidence.get("source_id") or "").strip()
        structural_id = source_id if source_type in {"table", "image", "algorithm", "equation", "toc"} and source_id else None
        return {
            "jump_page": page if page > 0 else None,
            "structural_id": structural_id,
            "source_type": source_type or None,
            "source_id": source_id or None,
            "navigation_status": "resolved_structural" if structural_id else "resolved_page",
            "navigation_reason": (
                "resolved_from_structural_evidence"
                if structural_id
                else "resolved_from_text_evidence"
            ),
        }
    return {
        "jump_page": None,
        "structural_id": None,
        "source_type": None,
        "source_id": None,
        "navigation_status": "unresolved",
        "navigation_reason": "no_navigation_target",
    }


def _enrich_rule_items_for_workbench(
    rule_items: list[dict[str, Any]],
    parsed_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    document_lookup = {
        str(document.get("filename") or "").strip(): dict(document)
        for document in parsed_documents
        if str(document.get("filename") or "").strip()
    }
    enriched_items: list[dict[str, Any]] = []

    for item in rule_items:
        enriched_item = dict(item)
        basis = _resolve_rule_basis(enriched_item)
        if basis:
            enriched_item["basis"] = basis
        details = dict(item.get("details", {}) or {})
        citation_anchor = str(details.get("citation_anchor") or enriched_item.get("citation") or "").strip()
        if citation_anchor:
            clause_lookup = _load_regulation_clause_lookup()
            clause_payload = dict(clause_lookup.get(citation_anchor) or {})
            if clause_payload:
                if not str(details.get("source_clause_id") or "").strip():
                    details["source_clause_id"] = (
                        str(clause_payload.get("source_clause_id") or "").strip()
                        or citation_anchor.replace("#", ":")
                    )
                if (
                    not str(details.get("regulation_severity") or "").strip()
                    and str(clause_payload.get("severity") or "").strip()
                ):
                    details["regulation_severity"] = str(clause_payload.get("severity") or "").strip()
        if not details:
            enriched_items.append(enriched_item)
            continue

        matched_documents = []
        for matched_document in details.get("matched_documents", []) or []:
            matched_document_payload = dict(matched_document)
            filename = str(matched_document_payload.get("filename") or "").strip()
            existing_jump_page = int(matched_document_payload.get("jump_page", 0) or 0)
            existing_structural_id = str(matched_document_payload.get("structural_id") or "").strip()
            if existing_jump_page > 0 and "navigation_status" not in matched_document_payload:
                matched_document_payload["navigation_status"] = (
                    "resolved_structural" if existing_structural_id else "resolved_page"
                )
                matched_document_payload["navigation_reason"] = "resolved_from_rule_evidence"
            navigation_target = _resolve_rule_navigation_target(
                document_lookup,
                filename=filename,
                evidence_refs=list(matched_document_payload.get("evidence_refs", []) or []),
            )
            if navigation_target.get("jump_page") is not None:
                matched_document_payload["jump_page"] = navigation_target["jump_page"]
            if navigation_target.get("structural_id"):
                matched_document_payload["structural_id"] = navigation_target["structural_id"]
            if existing_jump_page <= 0 or navigation_target.get("jump_page") is not None:
                matched_document_payload["navigation_status"] = navigation_target.get("navigation_status")
                matched_document_payload["navigation_reason"] = navigation_target.get("navigation_reason")

            section_refs = []
            for section_ref in matched_document_payload.get("section_refs", []) or []:
                section_ref_payload = dict(section_ref)
                page_span = list(section_ref_payload.get("page_span", []) or [])
                if page_span and page_span[0]:
                    section_ref_payload["jump_page"] = page_span[0]
                section_refs.append(section_ref_payload)
            matched_document_payload["section_refs"] = section_refs

            if matched_document_payload.get("jump_page") is None:
                for section_ref in section_refs:
                    jump_page = section_ref.get("jump_page")
                    if isinstance(jump_page, int) and jump_page > 0:
                        matched_document_payload["jump_page"] = jump_page
                        matched_document_payload["navigation_status"] = "resolved_page"
                        matched_document_payload["navigation_reason"] = "fallback_to_section_anchor"
                        break

            matched_documents.append(matched_document_payload)

        weak_signal_snippets = []
        for snippet in details.get("weak_signal_snippets", []) or []:
            snippet_payload = dict(snippet)
            filename = str(snippet_payload.get("filename") or "").strip()
            evidence_id = str(snippet_payload.get("evidence_id") or "").strip()
            navigation_target = _resolve_rule_navigation_target(
                document_lookup,
                filename=filename,
                evidence_refs=[evidence_id] if evidence_id else [],
            )
            if navigation_target.get("jump_page") is not None:
                snippet_payload["jump_page"] = navigation_target["jump_page"]
            elif int(snippet_payload.get("page", 0) or 0) > 0:
                snippet_payload["jump_page"] = int(snippet_payload.get("page", 0) or 0)
                snippet_payload["navigation_status"] = "resolved_page"
                snippet_payload["navigation_reason"] = "fallback_to_snippet_page"
            if navigation_target.get("structural_id"):
                snippet_payload["structural_id"] = navigation_target["structural_id"]
            if "navigation_status" not in snippet_payload:
                snippet_payload["navigation_status"] = navigation_target.get("navigation_status")
            if "navigation_reason" not in snippet_payload:
                snippet_payload["navigation_reason"] = navigation_target.get("navigation_reason")
            weak_signal_snippets.append(snippet_payload)

        details["matched_documents"] = matched_documents
        details["weak_signal_snippets"] = weak_signal_snippets
        enriched_item["details"] = details
        enriched_items.append(enriched_item)

    return enriched_items


def _build_rule_navigation_audit_records(
    rule_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audit_records: list[dict[str, Any]] = []

    for item in rule_items:
        rule_id = str(item.get("rule_id") or "").strip()
        citation = str(item.get("citation") or "").strip() or None
        details = dict(item.get("details", {}) or {})
        if not rule_id or not details:
            continue

        requirement_id = str(details.get("requirement_id") or "").strip() or None
        source_clause_id = str(details.get("source_clause_id") or "").strip() or None
        citation_anchor = str(details.get("citation_anchor") or "").strip() or citation

        for index, document in enumerate(details.get("matched_documents", []) or [], start=1):
            payload = dict(document)
            filename = str(payload.get("filename") or "").strip() or None
            audit_records.append(
                {
                    "audit_record_id": f"{rule_id}:matched_document:{index:02d}",
                    "rule_id": rule_id,
                    "requirement_id": requirement_id,
                    "source_clause_id": source_clause_id,
                    "citation_anchor": citation_anchor,
                    "target_kind": "matched_document",
                    "target_label": filename,
                    "filename": filename,
                    "target_page": int(payload.get("jump_page", 0) or 0) or None,
                    "target_structural_id": str(payload.get("structural_id") or "").strip() or None,
                    "backend_navigation_status": str(payload.get("navigation_status") or "").strip() or None,
                    "backend_navigation_reason": str(payload.get("navigation_reason") or "").strip() or None,
                    "evidence_refs": list(payload.get("evidence_refs", []) or []),
                    "section_outline_indices": [
                        str(section_ref.get("outline_index") or "").strip()
                        for section_ref in payload.get("section_refs", []) or []
                        if str(section_ref.get("outline_index") or "").strip()
                    ],
                }
            )

        for index, snippet in enumerate(details.get("weak_signal_snippets", []) or [], start=1):
            payload = dict(snippet)
            filename = str(payload.get("filename") or "").strip() or None
            audit_records.append(
                {
                    "audit_record_id": f"{rule_id}:weak_signal_snippet:{index:02d}",
                    "rule_id": rule_id,
                    "requirement_id": requirement_id,
                    "source_clause_id": source_clause_id,
                    "citation_anchor": citation_anchor,
                    "target_kind": "weak_signal_snippet",
                    "target_label": filename,
                    "filename": filename,
                    "evidence_id": str(payload.get("evidence_id") or "").strip() or None,
                    "target_page": int(payload.get("jump_page", 0) or 0) or None,
                    "target_structural_id": str(payload.get("structural_id") or "").strip() or None,
                    "backend_navigation_status": str(payload.get("navigation_status") or "").strip() or None,
                    "backend_navigation_reason": str(payload.get("navigation_reason") or "").strip() or None,
                }
            )

    return audit_records


def _build_rule_structure_audit_records(
    rule_items: list[dict[str, Any]],
    parsed_documents: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    audit_records: list[dict[str, Any]] = []
    toc_sequences_by_filename: dict[str, list[dict[str, Any]]] = {}
    for document in list(parsed_documents or []):
        filename = str(document.get("filename") or "").strip()
        if not filename:
            continue
        toc_sequences_by_filename[filename] = list(document.get("toc_sequences", []) or [])

    for item in rule_items:
        rule_id = str(item.get("rule_id") or "").strip()
        citation = str(item.get("citation") or "").strip() or None
        details = dict(item.get("details", {}) or {})
        if not rule_id or not details:
            continue

        for index, row in enumerate(details.get("structure_audit_rows", []) or [], start=1):
            payload = dict(row)
            filename = str(payload.get("filename") or "").strip() or None
            audit_records.append(
                {
                    "audit_record_id": f"{rule_id}:structure:{index:02d}",
                    "rule_id": rule_id,
                    "citation_anchor": citation,
                    "filename": filename,
                    "document_id": str(payload.get("document_id") or "").strip() or None,
                    "toc_outline_count": int(payload.get("toc_outline_count", 0) or 0),
                    "toc_root_outline_count": int(payload.get("toc_root_outline_count", 0) or 0),
                    "matched_outline_count": int(payload.get("matched_outline_count", 0) or 0),
                    "matched_root_outline_count": int(payload.get("matched_root_outline_count", 0) or 0),
                    "root_outline_coverage_ratio": payload.get("root_outline_coverage_ratio"),
                    "root_page_order_ready": bool(payload.get("root_page_order_ready", False)),
                    "root_page_offset_values": list(payload.get("root_page_offset_values", []) or []),
                    "root_page_offset_ready": bool(payload.get("root_page_offset_ready", False)),
                    "projected_root_page_values": list(payload.get("projected_root_page_values", []) or []),
                    "root_page_alignment_rows": list(payload.get("root_page_alignment_rows", []) or []),
                    "root_page_span_ready": bool(payload.get("root_page_span_ready", False)),
                    "direct_child_coverage_ratio": payload.get("direct_child_coverage_ratio"),
                    "direct_child_coverage_ready": bool(payload.get("direct_child_coverage_ready", False)),
                    "bounded_subtree_coverage_ratio": payload.get("bounded_subtree_coverage_ratio"),
                    "bounded_subtree_coverage_ready": bool(payload.get("bounded_subtree_coverage_ready", False)),
                    "missing_body_root_outline_indices": list(payload.get("missing_body_root_outline_indices", []) or []),
                    "missing_body_direct_child_outline_indices": list(
                        payload.get("missing_body_direct_child_outline_indices", []) or []
                    ),
                    "missing_body_direct_child_path_rows": list(
                        payload.get("missing_body_direct_child_path_rows", []) or []
                    ),
                    "missing_body_bounded_subtree_outline_indices": list(
                        payload.get("missing_body_bounded_subtree_outline_indices", []) or []
                    ),
                    "missing_body_bounded_subtree_path_rows": list(
                        payload.get("missing_body_bounded_subtree_path_rows", []) or []
                    ),
                    "toc_body_alignment_path_rows": list(payload.get("toc_body_alignment_path_rows", []) or []),
                    "navigation_targets": build_structure_audit_navigation_targets(
                        payload,
                        toc_sequences_by_filename.get(filename or "", []),
                    ),
                    "alignment_ready": bool(payload.get("alignment_ready", False)),
                }
            )

    return audit_records


def _build_workbench(
    parsed_documents: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
    markdown_download_url: str | None,
    structure_audit_download_url: str | None = None,
    structure_audit_markdown_download_url: str | None = None,
    demo_report_markdown_download_url: str | None = None,
    demo_script_markdown_download_url: str | None = None,
    compliance_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pdf_document: dict[str, Any] | None = None
    for document in parsed_documents:
        if document.get("source_type") == "pdf":
            pdf_document = {
                "file_id": document.get("file_id"),
                "filename": document.get("filename"),
                "file_url": f"/api/v1/files/{document.get('file_id')}",
                "pages": document.get("pages", []),
                "bounding_boxes": document.get("bounding_boxes", []),
                "image_blocks": document.get("image_blocks", []),
                "algorithm_blocks": document.get("algorithm_blocks", []),
                "equation_blocks": document.get("equation_blocks", []),
                "table_asts": document.get("table_asts", []),
                "figures": document.get("figures", []),
                "toc_blocks": document.get("toc_blocks", []),
                "toc_sequences": document.get("toc_sequences", []),
            }
            break

    compliance_payload = compliance_result or {}
    rule_items = _enrich_rule_items_for_workbench(
        list(compliance_payload.get("rules", []) or []),
        parsed_documents,
    )
    group_summaries = _build_rule_group_summaries(rule_items)
    navigation_audit_records = _build_rule_navigation_audit_records(rule_items)
    structure_audit_records = _build_rule_structure_audit_records(rule_items, parsed_documents)
    risk_items = list(compliance_payload.get("risks", []) or [])
    upload_scope_overview = build_upload_scope_overview(file_records)
    submission_scope_overview = build_submission_scope_overview(
        dict(compliance_payload.get("submission_scope", {}) or {})
    )
    regulatory_readiness = build_regulatory_readiness_projection(PROJECT_ROOT)
    dossier_checklist = build_dossier_checklist_projection(
        PROJECT_ROOT,
        dict(compliance_payload.get("submission_scope", {}) or {}),
    )
    content_consistency = build_content_consistency_projection(
        dict(compliance_payload.get("submission_scope", {}) or {})
    )
    rule_checks_for_demo_summary = {
        "enabled": bool(rule_items),
        "items": rule_items,
        "summary": {
            **dict(compliance_payload.get("summary", {}) or {}),
            "submission_scope": dict(compliance_payload.get("submission_scope", {}) or {}),
            "upload_scope_overview": upload_scope_overview,
            "submission_scope_overview": submission_scope_overview,
            "scope_transition_overview": build_scope_transition_overview(
                upload_scope_overview,
                submission_scope_overview,
            ),
        },
        "risk_count": len(risk_items),
    }
    demo_summary = build_demo_summary_projection(
        regulatory_readiness,
        dossier_checklist,
        rule_checks_for_demo_summary,
        content_consistency,
    )
    demo_report_markdown = build_demo_report_markdown(
        demo_summary=demo_summary,
        regulatory_readiness=regulatory_readiness,
        dossier_checklist=dossier_checklist,
        content_consistency=content_consistency,
    )
    demo_flow = build_demo_flow_projection(
        demo_summary=demo_summary,
        regulatory_readiness=regulatory_readiness,
        dossier_checklist=dossier_checklist,
        content_consistency=content_consistency,
        demo_report_markdown_download_url=demo_report_markdown_download_url,
    )
    demo_scenario = build_demo_scenario_projection(
        demo_summary=demo_summary,
        dossier_checklist=dossier_checklist,
        content_consistency=content_consistency,
        demo_flow=demo_flow,
        demo_report_markdown_download_url=demo_report_markdown_download_url,
    )
    demo_script_markdown = build_demo_script_markdown(
        demo_scenario=demo_scenario,
        demo_flow=demo_flow,
        demo_report_markdown_download_url=demo_report_markdown_download_url,
    )
    demo_run = build_demo_run_projection(
        demo_summary=demo_summary,
        regulatory_readiness=regulatory_readiness,
        dossier_checklist=dossier_checklist,
        content_consistency=content_consistency,
        demo_scenario=demo_scenario,
        demo_flow=demo_flow,
        demo_report_markdown_download_url=demo_report_markdown_download_url,
        demo_script_markdown_download_url=demo_script_markdown_download_url,
    )

    return {
        "pdf_document": pdf_document,
        "markdown": _build_ui_markdown(parsed_documents, file_records, consistency_rows),
        "full_markdown_download_url": markdown_download_url,
        "structure_audit_download_url": structure_audit_download_url,
        "structure_audit_markdown_download_url": structure_audit_markdown_download_url,
        "regulatory_readiness": regulatory_readiness,
        "dossier_checklist": dossier_checklist,
        "demo_summary": demo_summary,
        "demo_scenario": demo_scenario,
        "demo_flow": demo_flow,
        "demo_run": demo_run,
        "demo_report_markdown": demo_report_markdown,
        "demo_report_markdown_download_url": demo_report_markdown_download_url,
        "demo_script_markdown": demo_script_markdown,
        "demo_script_markdown_download_url": demo_script_markdown_download_url,
        "content_consistency": content_consistency,
        "rule_checks": {
            "enabled": bool(rule_items),
            "items": rule_items,
            "group_summaries": group_summaries,
            "navigation_audit_records": navigation_audit_records,
            "structure_audit_records": structure_audit_records,
            "summary": {
                **dict(compliance_payload.get("summary", {}) or {}),
                "submission_scope": dict(compliance_payload.get("submission_scope", {}) or {}),
                "upload_scope_overview": upload_scope_overview,
                "submission_scope_overview": submission_scope_overview,
                "scope_transition_overview": build_scope_transition_overview(
                    upload_scope_overview,
                    submission_scope_overview,
                ),
            },
            "risk_count": len(risk_items),
            "message": (
                "当前规则结果由材料审阅契约与确定性规则引擎生成。"
                if rule_items
                else "当前任务暂无可展示的确定性规则结果。"
            ),
        },
    }


def _build_compliance_result(
    submission_profile: str,
    parsed_documents: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
    final_status: str,
) -> dict[str, Any]:
    return build_compliance_result_payload(
        submission_profile=submission_profile,
        parsed_documents=parsed_documents,
        consistency_rows=consistency_rows,
        final_status=final_status,
    )


def _controlled_demo_submission_scope() -> dict[str, Any]:
    return {
        "upload_mode": "ectd_sequence_batch",
        "available_scopes": ["document", "sequence", "activity", "application"],
        "ectd_project_context": {
            "sequence_package_count": 2,
            "regulatory_activity_count": 1,
            "application_project_count": 1,
            "sequence_packages": [
                {
                    "sequence_package_id": "seqpkg:x202112345:0000",
                    "sequence_root": "synthetic://phase_a_controlled_demo/x202112345/0000",
                    "sequence_name": "0000",
                    "application_root": "synthetic://phase_a_controlled_demo/x202112345",
                    "application_root_name": "x202112345",
                    "application_number": "x202112345",
                    "sequence_number": "0000",
                    "application_type": "clinical-trial-application",
                    "product_type": "chemical",
                    "sequence_type": "initial-submission",
                },
                {
                    "sequence_package_id": "seqpkg:x202112999:0001",
                    "sequence_root": "synthetic://phase_a_controlled_demo/x202112345/0001",
                    "sequence_name": "0001",
                    "application_root": "synthetic://phase_a_controlled_demo/x202112345",
                    "application_root_name": "x202112345",
                    "application_number": "x202112999",
                    "sequence_number": "0002",
                    "application_type": "clinical-trial-application",
                    "product_type": "chemical",
                    "sequence_type": "initial-submission",
                },
            ],
        },
    }


def _controlled_demo_sample_metadata() -> dict[str, Any]:
    return {
        "sample_id": "phase_a_ectd_sequence_batch_readiness_demo",
        "synthetic": True,
        "sample_kind": "two_sequence_ectd_batch_with_identity_conflict",
        "evidence_boundary": (
            "Synthetic controlled demo sample. It is not an uploaded regulatory submission "
            "and does not create regulatory pass/fail decisions."
        ),
    }


def _controlled_demo_file_records(job_id: str) -> list[dict[str, Any]]:
    file_rows = [
        ("x202112345/0000/index.xml", "index.xml"),
        ("x202112345/0000/m1/cn/cn-regional.xml", "cn-regional.xml"),
        ("x202112345/0001/index.xml", "index.xml"),
        ("x202112345/0001/m1/cn/cn-regional.xml", "cn-regional.xml"),
    ]
    return [
        {
            "file_id": f"demo_{job_id}_{index}",
            "filename": filename,
            "relative_path": relative_path,
            "suffix": Path(filename).suffix.lower(),
            "path": "",
            "status": "completed",
            "message": "Synthetic demo placeholder; no uploaded file was parsed.",
            "progress": 100,
        }
        for index, (relative_path, filename) in enumerate(file_rows, start=1)
    ]


def _build_controlled_demo_sample_job(job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    created_at = _utc_now()
    file_records = _controlled_demo_file_records(job_id)
    upload_scope_overview = build_upload_scope_overview(file_records)
    demo_sample = _controlled_demo_sample_metadata()
    demo_report_markdown_path = DEMO_REPORT_DIR / f"{job_id}_demo_report.md"
    demo_report_markdown_download_url = f"/api/v1/jobs/{job_id}/demo-report/markdown/download"
    demo_script_markdown_path = DEMO_SCRIPT_DIR / f"{job_id}_demo_script.md"
    demo_script_markdown_download_url = f"/api/v1/jobs/{job_id}/demo-script/markdown/download"
    compliance_result = {
        "rules": [],
        "risks": [],
        "summary": {},
        "submission_scope": _controlled_demo_submission_scope(),
    }
    workbench = _build_workbench(
        parsed_documents=[],
        file_records=file_records,
        consistency_rows=[],
        markdown_download_url=None,
        structure_audit_download_url=None,
        structure_audit_markdown_download_url=None,
        demo_report_markdown_download_url=demo_report_markdown_download_url,
        demo_script_markdown_download_url=demo_script_markdown_download_url,
        compliance_result=compliance_result,
    )
    demo_report_markdown_path.write_text(
        str(workbench.get("demo_report_markdown") or ""),
        encoding="utf-8",
    )
    demo_script_markdown_path.write_text(
        str(workbench.get("demo_script_markdown") or ""),
        encoding="utf-8",
    )
    job_record = {
        "job_id": job_id,
        "run_id": None,
        "run_dir": None,
        "status": "completed",
        "progress": 100,
        "created_at": created_at,
        "updated_at": created_at,
        "files": file_records,
        "upload_scope_overview": upload_scope_overview,
        "parsed_documents": [],
        "workbench": workbench,
        "consistency_rows": [],
        "compliance_result": compliance_result,
        "demo_report_markdown_path": str(demo_report_markdown_path),
        "demo_script_markdown_path": str(demo_script_markdown_path),
        "demo_sample": demo_sample,
    }
    response_payload = {
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "created_at": created_at,
        "updated_at": created_at,
        "upload_scope_overview": upload_scope_overview,
        "demo_sample": demo_sample,
        "files": [
            {
                "file_id": file_item["file_id"],
                "filename": file_item["filename"],
                "relative_path": file_item.get("relative_path"),
                "status": file_item["status"],
                "message": file_item["message"],
                "progress": file_item["progress"],
            }
            for file_item in file_records
        ],
    }
    return job_record, response_payload


def _build_audit_log(
    run_id: str,
    job_id: str,
    final_status: str,
    parsed_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "job_id": job_id,
        "generated_at": _utc_now(),
        "status": final_status,
        "events": [
            {
                "step": "parse",
                "message": f"Parsed {len(parsed_documents)} document(s)",
            },
            {
                "step": "materialize_artifacts",
                "message": "AST/tables/images/normalized/atomic_facts persisted",
            },
            {
                "step": "emit_outputs",
                "message": "compliance_result.json and audit_log.json generated",
            },
        ],
    }


def _process_job(job_id: str) -> None:
    with STORE_LOCK:
        job = JOB_STORE.get(job_id)
        if job is None:
            return
        job["status"] = "processing"
        job["progress"] = 5
        job["updated_at"] = _utc_now()
        file_count = len(job["files"])
    logger.info("Job %s started processing (%s files)", job_id, file_count)

    with STORE_LOCK:
        file_records = list(JOB_STORE[job_id]["files"])

    run_context = create_run_context(PROJECT_ROOT, job_id, file_records)
    append_run_log(run_context, f"run context created for job={job_id}")
    with STORE_LOCK:
        JOB_STORE[job_id]["run_id"] = run_context.run_id
        JOB_STORE[job_id]["run_dir"] = str(run_context.run_dir)

    parsed_documents: list[dict[str, Any]] = []
    failed_files = 0

    total_files = len(file_records) or 1
    for index, file_record in enumerate(file_records):
        with STORE_LOCK:
            job = JOB_STORE[job_id]
            for item in job["files"]:
                if item["file_id"] == file_record["file_id"]:
                    item["status"] = "processing"
                    item["progress"] = 10
            job["updated_at"] = _utc_now()

        try:
            parsed = parse_file(Path(file_record["path"]))
            parsed["file_id"] = file_record["file_id"]
            parsed["filename"] = file_record["filename"]
            parsed_documents.append(parsed)
            persist_document_artifacts(run_context, index, parsed, file_record)
            append_run_log(
                run_context,
                (
                    f"parsed file={file_record['filename']} "
                    f"tables={len(parsed.get('table_asts', []))} "
                    f"images={len(parsed.get('image_blocks', []))}"
                ),
            )
            file_status = "completed"
            file_message = "Parsed successfully"
            file_progress = 100
            logger.info(
                "Job %s parsed file %s (%s)",
                job_id,
                file_record["filename"],
                file_record["suffix"],
            )
        except Exception as exc:  # pragma: no cover - defensive surface for parser errors
            failed_files += 1
            file_status = "failed"
            file_message = f"Parse failed: {exc}"
            file_progress = 100
            append_run_log(run_context, f"parse failed for file={file_record['filename']}: {exc}", level="ERROR")
            logger.exception("Job %s failed parsing file %s", job_id, file_record["filename"])

        processed_ratio = (index + 1) / total_files
        with STORE_LOCK:
            job = JOB_STORE[job_id]
            for item in job["files"]:
                if item["file_id"] == file_record["file_id"]:
                    item["status"] = file_status
                    item["message"] = file_message
                    item["progress"] = file_progress
            job["progress"] = 5 + int(processed_ratio * 80)
            job["updated_at"] = _utc_now()

    append_run_log(run_context, "building consistency rows")
    consistency_rows = build_fact_consistency_rows_from_documents(parsed_documents)
    persist_normalized_artifacts(run_context, parsed_documents)
    full_markdown_text = _build_full_markdown(parsed_documents)
    markdown_file_path = PARSED_MARKDOWN_DIR / f"{job_id}_parse_full.md"
    markdown_file_path.write_text(full_markdown_text, encoding="utf-8")
    markdown_download_url = f"/api/v1/jobs/{job_id}/markdown/download"

    if failed_files == len(file_records):
        final_status = "failed"
    elif failed_files > 0:
        final_status = "completed_with_warnings"
    else:
        final_status = "completed"

    compliance_result = _build_compliance_result("FIH", parsed_documents, consistency_rows, final_status)
    structure_audit_records = _build_rule_structure_audit_records(
        _enrich_rule_items_for_workbench(
            list(compliance_result.get("rules", []) or []),
            parsed_documents,
        )
    )
    structure_audit_export_payload = build_structure_audit_export_payload(structure_audit_records)
    structure_audit_path = STRUCTURE_AUDIT_DIR / f"{job_id}_structure_audit.json"
    structure_audit_path.write_text(
        json.dumps(structure_audit_export_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    structure_audit_markdown_path = STRUCTURE_AUDIT_DIR / f"{job_id}_structure_audit.md"
    structure_audit_markdown_path.write_text(
        build_structure_audit_markdown_report(
            structure_audit_records,
            generated_at=str(structure_audit_export_payload.get("generated_at") or _utc_now()),
        ),
        encoding="utf-8",
    )
    structure_audit_download_url = f"/api/v1/jobs/{job_id}/structure-audit/download"
    structure_audit_markdown_download_url = f"/api/v1/jobs/{job_id}/structure-audit/markdown/download"
    demo_report_markdown_path = DEMO_REPORT_DIR / f"{job_id}_demo_report.md"
    demo_report_markdown_download_url = f"/api/v1/jobs/{job_id}/demo-report/markdown/download"
    demo_script_markdown_path = DEMO_SCRIPT_DIR / f"{job_id}_demo_script.md"
    demo_script_markdown_download_url = f"/api/v1/jobs/{job_id}/demo-script/markdown/download"
    with STORE_LOCK:
        job = JOB_STORE[job_id]
        final_file_records = [dict(item) for item in job["files"]]
        workbench = _build_workbench(
            parsed_documents=parsed_documents,
            file_records=final_file_records,
            consistency_rows=consistency_rows,
            markdown_download_url=markdown_download_url,
            structure_audit_download_url=structure_audit_download_url,
            structure_audit_markdown_download_url=structure_audit_markdown_download_url,
            demo_report_markdown_download_url=demo_report_markdown_download_url,
            demo_script_markdown_download_url=demo_script_markdown_download_url,
            compliance_result=compliance_result,
        )
        demo_report_markdown_path.write_text(
            str(workbench.get("demo_report_markdown") or ""),
            encoding="utf-8",
        )
        demo_script_markdown_path.write_text(
            str(workbench.get("demo_script_markdown") or ""),
            encoding="utf-8",
        )
        job["parsed_documents"] = parsed_documents
        job["consistency_rows"] = consistency_rows
        job["compliance_result"] = compliance_result
        job["workbench"] = workbench
        job["full_markdown_path"] = str(markdown_file_path)
        job["structure_audit_path"] = str(structure_audit_path)
        job["structure_audit_markdown_path"] = str(structure_audit_markdown_path)
        job["demo_report_markdown_path"] = str(demo_report_markdown_path)
        job["demo_script_markdown_path"] = str(demo_script_markdown_path)
        job["progress"] = 100
        job["updated_at"] = _utc_now()
        job["status"] = final_status

    audit_log = _build_audit_log(run_context.run_id, job_id, final_status, parsed_documents)
    persist_run_outputs(run_context, compliance_result, audit_log)
    finalize_run_context(run_context, final_status)
    append_run_log(run_context, f"run finalized status={final_status}")
    logger.info("Job %s finished with status=%s", job_id, final_status)


def create_app() -> FastAPI:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    app = FastAPI(title="IND Compliance AI", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/demo/controlled-sample/job")
    def create_controlled_demo_sample_job() -> dict[str, Any]:
        job_id = f"demo_{uuid4().hex}"
        job_record, response_payload = _build_controlled_demo_sample_job(job_id)
        with STORE_LOCK:
            JOB_STORE[job_id] = job_record
        logger.info("Controlled demo sample job %s created", job_id)
        return response_payload

    @app.post("/api/v1/uploads")
    async def create_upload_job(
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...),
        relative_paths: list[str] = Form(default=[]),
    ) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        job_id = uuid4().hex
        file_records: list[dict[str, Any]] = []
        created_at = _utc_now()
        logger.info("Upload request received. job_id=%s, files=%s", job_id, len(files))

        for index, incoming_file in enumerate(files):
            filename = incoming_file.filename or "uploaded_file"
            relative_path = str(relative_paths[index] if index < len(relative_paths) else filename).strip() or filename
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
                )

            file_id = uuid4().hex
            target_path = UPLOAD_DIR / f"{file_id}_{_safe_filename(filename)}"
            target_path.write_bytes(await incoming_file.read())

            with STORE_LOCK:
                FILE_STORE[file_id] = target_path

            file_records.append(
                {
                    "file_id": file_id,
                    "filename": filename,
                    "relative_path": relative_path,
                    "suffix": suffix,
                    "path": str(target_path),
                    "status": "queued",
                    "message": "Queued",
                    "progress": 0,
                }
            )

        upload_scope_overview = build_upload_scope_overview(file_records)
        job_record = {
            "job_id": job_id,
            "run_id": None,
            "run_dir": None,
            "status": "queued",
            "progress": 0,
            "created_at": created_at,
            "updated_at": created_at,
            "files": file_records,
            "upload_scope_overview": upload_scope_overview,
            "parsed_documents": [],
            "workbench": None,
            "consistency_rows": [],
        }
        with STORE_LOCK:
            JOB_STORE[job_id] = job_record

        background_tasks.add_task(_process_job, job_id)
        logger.info("Job %s queued", job_id)
        return {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "created_at": created_at,
            "updated_at": created_at,
            "upload_scope_overview": upload_scope_overview,
            "files": [
                {
                    "file_id": file_item["file_id"],
                    "filename": file_item["filename"],
                    "relative_path": file_item.get("relative_path"),
                    "status": file_item["status"],
                    "progress": file_item["progress"],
                }
                for file_item in file_records
            ],
        }

    @app.get("/api/v1/jobs/{job_id}")
    def get_job_status(job_id: str) -> dict[str, Any]:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return {
                "job_id": job["job_id"],
                "run_id": job.get("run_id"),
                "run_dir": job.get("run_dir"),
                "status": job["status"],
                "progress": job["progress"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "upload_scope_overview": dict(job.get("upload_scope_overview", {}) or {}),
                "demo_sample": dict(job.get("demo_sample", {}) or {}) or None,
                "files": [
                    {
                        "file_id": file_item["file_id"],
                        "filename": file_item["filename"],
                        "relative_path": file_item.get("relative_path"),
                        "status": file_item["status"],
                        "message": file_item["message"],
                        "progress": file_item["progress"],
                    }
                    for file_item in job["files"]
                ],
            }

    @app.get("/api/v1/jobs/{job_id}/workbench")
    def get_workbench(job_id: str) -> dict[str, Any]:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job["status"] in {"queued", "processing"}:
                raise HTTPException(status_code=409, detail="Job is still processing")
            logger.info("Workbench requested for job %s", job_id)
            return job["workbench"] or {}

    @app.get("/api/v1/jobs/{job_id}/compliance-result")
    def get_compliance_result(job_id: str) -> dict[str, Any]:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job["status"] in {"queued", "processing"}:
                raise HTTPException(status_code=409, detail="Job is still processing")
            logger.info("Compliance result requested for job %s", job_id)
            return job.get("compliance_result") or {}

    @app.get("/api/v1/jobs/{job_id}/consistency")
    def get_consistency(job_id: str) -> dict[str, Any]:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job["status"] in {"queued", "processing"}:
                raise HTTPException(status_code=409, detail="Job is still processing")
            logger.info("Consistency board requested for job %s", job_id)
            return {"rows": job["consistency_rows"]}

    @app.get("/api/v1/jobs/{job_id}/markdown/download")
    def download_full_markdown(job_id: str) -> FileResponse:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            markdown_path_str = str(job.get("full_markdown_path", "")).strip()
        if not markdown_path_str:
            raise HTTPException(status_code=404, detail="Full markdown file not found")
        markdown_path = Path(markdown_path_str)
        if not markdown_path.exists():
            raise HTTPException(status_code=404, detail="Full markdown file not found")
        logger.info("Full markdown download requested for job %s", job_id)
        return FileResponse(
            markdown_path,
            media_type="text/markdown; charset=utf-8",
            filename=f"{job_id}_full_parse.md",
        )

    @app.get("/api/v1/jobs/{job_id}/structure-audit/download")
    def download_structure_audit(job_id: str) -> FileResponse:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            structure_audit_path_str = str(job.get("structure_audit_path", "")).strip()
        if not structure_audit_path_str:
            raise HTTPException(status_code=404, detail="Structure audit file not found")
        structure_audit_path = Path(structure_audit_path_str)
        if not structure_audit_path.exists():
            raise HTTPException(status_code=404, detail="Structure audit file not found")
        logger.info("Structure audit download requested for job %s", job_id)
        return FileResponse(
            structure_audit_path,
            media_type="application/json",
            filename=f"{job_id}_structure_audit.json",
        )

    @app.get("/api/v1/jobs/{job_id}/structure-audit/markdown/download")
    def download_structure_audit_markdown(job_id: str) -> FileResponse:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            structure_audit_markdown_path_str = str(job.get("structure_audit_markdown_path", "")).strip()
        if not structure_audit_markdown_path_str:
            raise HTTPException(status_code=404, detail="Structure audit markdown file not found")
        structure_audit_markdown_path = Path(structure_audit_markdown_path_str)
        if not structure_audit_markdown_path.exists():
            raise HTTPException(status_code=404, detail="Structure audit markdown file not found")
        logger.info("Structure audit markdown download requested for job %s", job_id)
        return FileResponse(
            structure_audit_markdown_path,
            media_type="text/markdown; charset=utf-8",
            filename=f"{job_id}_structure_audit.md",
        )

    @app.get("/api/v1/jobs/{job_id}/demo-report/markdown/download")
    def download_demo_report_markdown(job_id: str) -> FileResponse:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            demo_report_markdown_path_str = str(job.get("demo_report_markdown_path", "")).strip()
        if not demo_report_markdown_path_str:
            raise HTTPException(status_code=404, detail="Demo report markdown file not found")
        demo_report_markdown_path = Path(demo_report_markdown_path_str)
        if not demo_report_markdown_path.exists():
            raise HTTPException(status_code=404, detail="Demo report markdown file not found")
        logger.info("Demo report markdown download requested for job %s", job_id)
        return FileResponse(
            demo_report_markdown_path,
            media_type="text/markdown; charset=utf-8",
            filename=f"{job_id}_demo_report.md",
        )

    @app.get("/api/v1/jobs/{job_id}/demo-script/markdown/download")
    def download_demo_script_markdown(job_id: str) -> FileResponse:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            demo_script_markdown_path_str = str(job.get("demo_script_markdown_path", "")).strip()
        if not demo_script_markdown_path_str:
            raise HTTPException(status_code=404, detail="Demo script markdown file not found")
        demo_script_markdown_path = Path(demo_script_markdown_path_str)
        if not demo_script_markdown_path.exists():
            raise HTTPException(status_code=404, detail="Demo script markdown file not found")
        logger.info("Demo script markdown download requested for job %s", job_id)
        return FileResponse(
            demo_script_markdown_path,
            media_type="text/markdown; charset=utf-8",
            filename=f"{job_id}_demo_script.md",
        )

    @app.get("/api/v1/files/{file_id}")
    def get_file(file_id: str) -> FileResponse:
        with STORE_LOCK:
            file_path = FILE_STORE.get(file_id)
        if file_path is None or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)

    frontend_dist = PROJECT_ROOT / "ui" / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    else:
        @app.get("/")
        def frontend_placeholder() -> JSONResponse:
            return JSONResponse(
                content={
                    "message": (
                        "Frontend dist not found. Run `npm install && npm run dev` in ui/frontend "
                        "or `npm run build` to let FastAPI serve static files."
                    )
                }
            )

    return app


app = create_app()
