from __future__ import annotations

import copy
from datetime import datetime, timezone
import base64
from collections import Counter
from io import BytesIO
import html
import json
import logging
from pathlib import Path
import re
import zipfile
from threading import Lock
from typing import Any, Iterable
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.upload_controller import ALLOWED_EXTENSIONS, is_directory_upload_path
from core.material_assessment import (
    build_compliance_result_payload,
    build_fact_consistency_rows_from_documents,
)
from core.outline_markers import (
    classify_outline_heading_candidate,
    normalize_outline_marker,
    outline_titles_compatible,
    parse_outline_heading,
)
from core.upload_scope_projection import build_upload_scope_overview
from core.ectd_package_inventory import build_package_inventory
from core.ectd_structure_validation import validate_package_structure
from core.ectd_naming_validation import validate_package_naming
from core.regulation_provenance import provenance_for_rule
from core.ectd_application_identity import assess_application_identity
from core.ectd_controlled_vocabulary_rules import (
    build_ectd_vocabulary_rule_contract,
    validate_ectd_envelope_vocabulary,
)
from core.ectd_sequence_semantics import (
    build_ectd_sequence_semantic_contract,
    validate_ectd_sequence_semantics,
)
from core.ectd_package_intake import records_from_zip_bytes
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
from parsers.pdf.table_modules.cell_text_projection import (
    project_pdf_math_symbol_display_text,
    project_table_cell_display_text,
)
from parsers.parser_registry import parse_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "samples" / "uploaded"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PARSED_MARKDOWN_DIR = PROJECT_ROOT / "output" / "parsed_markdown"
PARSED_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
IND_REVIEW_MARKDOWN_DIR = PROJECT_ROOT / "output" / "ind_review_markdown"
IND_REVIEW_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
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
    text = str(project_pdf_math_symbol_display_text(block.get("display_text") or block.get("text") or "") or "").strip()
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


def _normalize_markdown_table_semantic_header_grid(block: dict[str, Any]) -> list[list[str]]:
    projection = block.get("semantic_projection_v2")
    semantic_grid = block.get("semantic_display_grid") or block.get("semantic_grid")
    if _should_export_table_semantic_projection_grid(block, projection, semantic_grid):
        projected_grid = _project_markdown_visible_semantic_grid(block, semantic_grid)
        projected_grid = _append_markdown_study_condition_trailing_metadata_rows(block, projected_grid)
        normalized_semantic_rows = [
            [_markdown_escape_table_cell(cell) for cell in row]
            for row in projected_grid
            if isinstance(row, list)
        ]
        normalized_semantic_rows = [row for row in normalized_semantic_rows if any(row)]
        if normalized_semantic_rows:
            display_preserving_rows = _markdown_display_preserving_rows_for_semantic_projection(
                block,
                normalized_semantic_rows,
            )
            if display_preserving_rows:
                return display_preserving_rows
            return normalized_semantic_rows

    header = [item for item in block.get("header", []) or [] if isinstance(item, dict)]
    data_grid = block.get("data_grid")
    if not header or not isinstance(data_grid, list) or not data_grid:
        return []

    header_texts = [
        _markdown_escape_table_cell(item.get("text"))
        for item in header
    ]
    if not header_texts or not any(header_texts):
        return []

    column_count = len(header_texts)
    normalized_data_rows = [
        [_markdown_escape_table_cell(cell) for cell in row]
        for row in data_grid
        if isinstance(row, list)
    ]
    normalized_data_rows = [row for row in normalized_data_rows if any(row)]
    if not normalized_data_rows:
        return []
    if any(len(row) > column_count for row in normalized_data_rows):
        return []

    display_grid = block.get("display_grid")
    if isinstance(display_grid, list) and display_grid:
        first_display_row = next((row for row in display_grid if isinstance(row, list) and any(row)), None)
        if first_display_row is not None:
            display_column_count = len(first_display_row)
            if display_column_count != column_count:
                return []
        display_data_rows = _markdown_display_rows_after_structural_prefix(block, column_count)
        if display_data_rows:
            return [header_texts] + display_data_rows

    return [header_texts] + normalized_data_rows


def _markdown_display_preserving_rows_for_semantic_projection(
    block: dict[str, Any],
    semantic_rows: list[list[str]],
) -> list[list[str]]:
    if (
        _markdown_table_has_word_logical_grid_projection(block)
        or _markdown_table_has_study_condition_result_matrix_projection(block)
        or _markdown_table_has_overview_inventory_schema_projection(block)
        or _markdown_table_has_study_metric_grouped_matrix_projection(block)
        or _markdown_table_has_ind_late_semantic_projection(block)
    ):
        return []

    display_grid = block.get("display_grid")
    if not isinstance(display_grid, list) or not display_grid:
        return []
    display_rows = [
        [_markdown_escape_table_cell(cell) for cell in row]
        for row in display_grid
        if isinstance(row, list)
    ]
    display_rows = [row for row in display_rows if any(row)]
    if len(display_rows) <= len(semantic_rows):
        return []
    semantic_signatures = {_markdown_row_compact_signature(row) for row in semantic_rows}
    missing_display_rows = [
        row
        for row in display_rows
        if _markdown_row_compact_signature(row)
        and _markdown_row_compact_signature(row) not in semantic_signatures
    ]
    if not missing_display_rows:
        return []
    missing_label_value_rows = [
        row
        for row in missing_display_rows
        if _markdown_row_looks_like_label_value_data_row(row)
    ]
    if not missing_label_value_rows:
        return []
    header = [item for item in block.get("header", []) or [] if isinstance(item, dict)]
    header_texts = [_markdown_escape_table_cell(item.get("text")) for item in header]
    if header_texts and len(header_texts) == max((len(row) for row in display_rows), default=0):
        return [header_texts] + display_rows
    return display_rows


def _markdown_row_looks_like_label_value_data_row(row: list[Any]) -> bool:
    texts = [_markdown_compact_table_text(cell) for cell in row]
    non_empty = [(idx, text) for idx, text in enumerate(texts) if text]
    if not non_empty:
        return False
    first_idx, first_text = non_empty[0]
    if first_idx != 0 or not re.search(r"[A-Za-z\u4e00-\u9fff]", first_text):
        return False
    if len(non_empty) == 1:
        return first_text.endswith((":","：")) and len(first_text) <= 42
    value_cells = [
        text
        for _, text in non_empty[1:]
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?|[-–—]|n\.?d\.?|微量", text, re.IGNORECASE)
        or re.fullmatch(r"\d+(?:\.\d+)?(?:\s*[A-Za-z%/]+)?", text)
    ]
    return len(value_cells) >= max(1, len(non_empty) - 2)


def _project_markdown_visible_semantic_grid(block: dict[str, Any], semantic_grid: Any) -> list[list[Any]]:
    rows = [row for row in semantic_grid if isinstance(row, list)] if isinstance(semantic_grid, list) else []
    if not rows:
        return []
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    if isinstance(block.get("cell_spans"), list):
        has_canonical_header = any(
            isinstance(span, dict) and str(span.get("role") or "") == "header"
            for span in block.get("cell_spans", []) or []
        )
        if has_canonical_header:
            canonical_header_grid = _project_markdown_canonical_header_grid(block, rows)
            if canonical_header_grid:
                return canonical_header_grid
        return rows
    genotoxicity_grid = _project_markdown_genotoxicity_multilevel_header_grid(block, rows)
    if genotoxicity_grid:
        return genotoxicity_grid
    dose_response_grid = _project_markdown_dose_response_multilevel_header_grid(block, rows)
    if dose_response_grid:
        return dose_response_grid
    study_metric_grid = _project_markdown_study_metric_grouped_header_grid(block, rows)
    if study_metric_grid:
        return study_metric_grid
    composite_study_grid = _project_markdown_study_condition_composite_grid(block, rows)
    if composite_study_grid:
        return composite_study_grid
    grouped_grid = _project_markdown_grouped_multilevel_borderless_grid(block, rows)
    if grouped_grid:
        return grouped_grid
    table_family = str(block.get("table_family") or (projection or {}).get("table_family") or "")
    display_grid = block.get("display_grid")
    display_rows = [row for row in display_grid if isinstance(row, list)] if isinstance(display_grid, list) else []
    if (
        table_family == "blank_form_comparison_matrix"
        and isinstance(projection, dict)
        and isinstance(projection.get("blank_stub_column_projection"), dict)
        and rows
        and len(rows[0]) >= 3
        and not _html_table_cell_text(rows[0][0])
        and display_rows
        and max((len(row) for row in display_rows), default=0) < len(rows[0])
        and not any(isinstance(item, dict) for item in block.get("header_row_groups", []) or [])
    ):
        return [[row[0], *row[2:]] if idx > 0 else row[1:] for idx, row in enumerate(rows)]
    return rows


def _project_markdown_canonical_header_grid(
    block: dict[str, Any],
    rows: list[list[Any]],
) -> list[list[Any]]:
    header_spans = [
        span
        for span in block.get("cell_spans", []) or []
        if isinstance(span, dict) and str(span.get("role") or "") == "header"
    ]
    if not header_spans:
        return []
    header_depth = _markdown_canonical_header_depth(block)
    if header_depth <= 1:
        return []
    if _markdown_canonical_header_grid_is_materialized(rows, header_spans, header_depth):
        return rows
    column_count = max((len(row) for row in rows), default=0)
    if column_count <= 0:
        return []
    header_rows: list[list[Any]] = [["" for _ in range(column_count)] for _ in range(header_depth)]
    covered: set[tuple[int, int]] = set()
    for span in header_spans:
        try:
            row = int(span.get("row", 0) or 0)
            col = int(span.get("col", 0) or 0)
            rowspan = max(1, int(span.get("rowspan", 1) or 1))
            colspan = max(1, int(span.get("colspan", 1) or 1))
        except (TypeError, ValueError):
            return []
        text = str(span.get("text") or "").strip()
        if not text or row < 0 or col < 0 or row + rowspan > header_depth or col + colspan > column_count:
            return []
        header_rows[row][col] = text
        for covered_row in range(row, row + rowspan):
            for covered_col in range(col, col + colspan):
                if (covered_row, covered_col) != (row, col):
                    covered.add((covered_row, covered_col))
    leaf_row = [
        _markdown_canonical_leaf_header_text(cell, header_spans, col)
        for col, cell in enumerate(rows[0])
    ] + [""] * (column_count - len(rows[0]))
    leaf_index = header_depth - 1
    for col, text in enumerate(leaf_row[:column_count]):
        if (leaf_index, col) in covered or header_rows[leaf_index][col]:
            continue
        header_rows[leaf_index][col] = text
    return [*header_rows, *rows[1:]]


def _markdown_canonical_leaf_header_text(
    value: Any,
    header_spans: list[dict[str, Any]],
    col: int,
) -> str:
    text = _markdown_normalize_grouped_multilevel_header_text(value)
    if not text:
        return ""
    for span in sorted(
        header_spans,
        key=lambda item: int(item.get("row", 0) or 0),
        reverse=True,
    ):
        try:
            start_col = int(span.get("col", -1))
            colspan = max(1, int(span.get("colspan", 1) or 1))
        except (TypeError, ValueError):
            continue
        if colspan <= 1 or col < start_col or col >= start_col + colspan:
            continue
        parent = _markdown_normalize_grouped_multilevel_header_text(span.get("text"))
        if not parent:
            continue
        match = re.fullmatch(rf"{re.escape(parent)}\s+(M|F)", text, re.IGNORECASE)
        if match:
            return str(match.group(1) or "").upper()
    return text


def _markdown_canonical_header_grid_is_materialized(
    rows: list[list[Any]],
    header_spans: list[dict[str, Any]],
    header_depth: int,
) -> bool:
    if len(rows) < header_depth:
        return False
    saw_deep_leaf = False
    for span in header_spans:
        try:
            row = int(span.get("row", 0) or 0)
            col = int(span.get("col", 0) or 0)
            rowspan = max(1, int(span.get("rowspan", 1) or 1))
            colspan = max(1, int(span.get("colspan", 1) or 1))
        except (TypeError, ValueError):
            return False
        if row >= len(rows) or col >= len(rows[row]):
            return False
        if _markdown_compact_table_text(rows[row][col]) != _markdown_compact_table_text(span.get("text")):
            return False
        for covered_row in range(row, row + rowspan):
            for covered_col in range(col, col + colspan):
                if (covered_row, covered_col) == (row, col):
                    continue
                if covered_row >= len(rows) or covered_col >= len(rows[covered_row]):
                    return False
                covered_text = _markdown_compact_table_text(rows[covered_row][covered_col])
                if covered_text and covered_text != _markdown_compact_table_text(span.get("text")):
                    return False
    if header_depth > 1:
        saw_deep_leaf = any(_markdown_compact_table_text(cell) for cell in rows[header_depth - 1])
    return header_depth == 1 or saw_deep_leaf


def _project_markdown_study_metric_grouped_header_grid(
    block: dict[str, Any],
    rows: list[list[Any]],
) -> list[list[Any]]:
    if len(rows) < 2:
        return []
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    metric_projection = projection.get("study_metric_grouped_matrix_projection")
    if not isinstance(metric_projection, dict):
        return []
    if str(metric_projection.get("semantic_profile") or "") != "study_metric_grouped_matrix":
        return []
    header_groups = [
        group
        for group in block.get("header_column_groups", [])
        or metric_projection.get("header_column_groups", [])
        or []
        if isinstance(group, dict)
    ]
    return _materialize_markdown_grouped_header_rows(rows, header_groups)


def _materialize_markdown_grouped_header_rows(
    rows: list[list[Any]],
    header_groups: list[dict[str, Any]],
) -> list[list[Any]]:
    if not rows or not header_groups:
        return []
    column_count = len(rows[0])
    if column_count <= 1:
        return []

    groups: list[tuple[int, int, int, str]] = []
    for group in header_groups:
        text = _markdown_normalize_grouped_multilevel_header_text(group.get("text"))
        try:
            row_index = int(group.get("row", 0) or 0)
            start_col = int(group.get("start_col", 0) or 0)
            end_col = int(group.get("end_col", start_col) or start_col)
        except (TypeError, ValueError):
            return []
        if (
            not text
            or row_index < 0
            or start_col < 0
            or end_col < start_col
            or end_col >= column_count
        ):
            return []
        groups.append((row_index, start_col, end_col, text))
    if not groups or not any(end_col > start_col for _, start_col, end_col, _ in groups):
        return []

    levels = sorted({row_index for row_index, _, _, _ in groups})
    level_positions = {row_index: position for position, row_index in enumerate(levels)}
    if len(rows) > len(levels) and all(
        all(
            _markdown_compact_table_text(rows[level_positions[row_index]][col])
            == _markdown_compact_table_text(text)
            for col in range(start_col, end_col + 1)
        )
        for row_index, start_col, end_col, text in groups
    ):
        return []
    group_rows: list[list[Any]] = [["" for _ in range(column_count)] for _ in levels]
    occupied: list[list[bool]] = [[False for _ in range(column_count)] for _ in levels]
    deepest_group_by_col: list[tuple[int, int, str] | None] = [None for _ in range(column_count)]
    for row_index, start_col, end_col, text in sorted(groups):
        level_position = level_positions[row_index]
        if any(occupied[level_position][col] for col in range(start_col, end_col + 1)):
            return []
        for col in range(start_col, end_col + 1):
            group_rows[level_position][col] = text
            occupied[level_position][col] = True
            previous = deepest_group_by_col[col]
            if previous is None or level_position > previous[0]:
                deepest_group_by_col[col] = (level_position, end_col - start_col + 1, text)

    leaf_row = [
        _markdown_normalize_grouped_multilevel_header_text(cell)
        for cell in rows[0]
    ]
    deepest_level = len(group_rows) - 1
    for col, leaf_text in enumerate(list(leaf_row)):
        deepest_group = deepest_group_by_col[col]
        if deepest_group is None:
            group_rows[deepest_level][col] = leaf_text
            leaf_row[col] = ""
            continue
        group_level, colspan, group_text = deepest_group
        if colspan == 1 and _markdown_compact_table_text(group_text) == _markdown_compact_table_text(leaf_text):
            leaf_row[col] = ""
            continue
        if group_level < deepest_level and not occupied[group_level + 1][col]:
            group_rows[group_level + 1][col] = leaf_text
            leaf_row[col] = ""

    if not any(leaf_row):
        return [*group_rows, *rows[1:]]
    return [*group_rows, leaf_row, *rows[1:]]


def _project_markdown_grouped_multilevel_borderless_grid(
    block: dict[str, Any],
    rows: list[list[Any]],
) -> list[list[Any]]:
    if len(rows) < 2:
        return []
    if _markdown_grid_rows_are_multilevel_header_pair(rows[0], rows[1]):
        return []
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    grouped_projection = projection.get("grouped_multilevel_borderless_projection")
    if not isinstance(grouped_projection, dict):
        return []
    if str(grouped_projection.get("semantic_profile") or "") != "grouped_multilevel_borderless_table":
        return []
    header_groups = [
        group
        for group in grouped_projection.get("header_column_groups", []) or []
        if isinstance(group, dict)
    ]
    if not header_groups:
        return []
    try:
        column_count = len(rows[0])
    except Exception:
        return []
    if column_count <= 1:
        return []

    normalized_header = [
        _markdown_normalize_grouped_multilevel_header_text(cell)
        for cell in rows[0]
    ]
    group_row: list[Any] = list(normalized_header)
    leaf_row: list[Any] = ["" for _ in range(column_count)]
    has_group_span = False
    for group in header_groups:
        text = _markdown_normalize_grouped_multilevel_header_text(group.get("text"))
        if not text:
            continue
        try:
            start_col = int(group.get("start_col", 0) or 0)
            end_col = int(group.get("end_col", start_col) or start_col)
        except (TypeError, ValueError):
            continue
        if start_col < 0 or start_col >= column_count or end_col < start_col:
            continue
        end_col = min(end_col, column_count - 1)
        if end_col <= start_col:
            continue
        has_group_span = True
        child_headers = [
            _markdown_normalize_grouped_multilevel_header_text(item)
            for item in group.get("child_headers", []) or []
        ]
        for col_index in range(start_col, end_col + 1):
            group_row[col_index] = text
            child_offset = col_index - start_col
            child_text = child_headers[child_offset] if child_offset < len(child_headers) else ""
            leaf_row[col_index] = child_text or normalized_header[col_index]

    if not has_group_span:
        return []
    return [group_row, leaf_row, *rows[1:]]


def _markdown_normalize_grouped_multilevel_header_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    compact = _markdown_compact_table_text(text)
    if compact == "采样时间或周期":
        return "采样时间或周期"
    if compact in {"占给药剂量", "占给药剂量的"}:
        return "占给药剂量的%"
    if compact in {"ctd中的位置", "ctd位置"}:
        return "CTD 中的位置"
    return text


def _project_markdown_genotoxicity_multilevel_header_grid(
    block: dict[str, Any],
    rows: list[list[Any]],
) -> list[list[Any]]:
    if len(rows) < 2:
        return []
    if _markdown_grid_rows_are_multilevel_header_pair(rows[0], rows[1]):
        return []
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    genotoxicity_projection = projection.get("genotoxicity_assay_matrix_projection")
    if not isinstance(genotoxicity_projection, dict):
        return []
    if str(genotoxicity_projection.get("semantic_profile") or "") != "genotoxicity_assay_matrix":
        return []
    span_cells = [
        dict(item)
        for item in block.get("span_header_cells", []) or []
        if isinstance(item, dict)
    ]
    if not span_cells:
        return []
    try:
        column_count = len(rows[0])
    except Exception:
        return []
    if column_count <= 1:
        return []
    group_row = ["" for _ in range(column_count)]
    leaf_row = list(rows[0])
    has_group = False
    for span in span_cells:
        try:
            row = int(span.get("row", 0) or 0)
            col = int(span.get("col", 0) or 0)
            colspan = max(1, int(span.get("colspan", 1) or 1))
        except (TypeError, ValueError):
            continue
        text = str(span.get("text") or "").strip()
        if not text or row != 0 or col < 0 or col >= column_count:
            continue
        if colspan <= 1:
            leaf_row[col] = text
            continue
        end_col = min(column_count, col + colspan)
        for target_col in range(col, end_col):
            group_row[target_col] = text
        has_group = True
    if not has_group:
        return []
    for col_index, value in enumerate(leaf_row):
        if group_row[col_index]:
            continue
        group_row[col_index] = value
    return [group_row, leaf_row, *rows[1:]]


def _project_markdown_dose_response_multilevel_header_grid(
    block: dict[str, Any],
    rows: list[list[Any]],
) -> list[list[Any]]:
    if len(rows) < 2:
        return []
    if _markdown_grid_rows_are_multilevel_header_pair(rows[0], rows[1]):
        return []
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    dose_projection = projection.get("dose_response_result_panel_projection")
    if not isinstance(dose_projection, dict):
        return []
    if str(dose_projection.get("semantic_profile") or "") != "dose_response_result_panel":
        return []
    if not dose_projection.get("has_sex_leaf_columns"):
        return []
    header = [str(cell or "").strip() for cell in rows[0]]
    if len(header) < 3 or (len(header) - 1) % 2 != 0:
        return []
    group_row: list[Any] = [header[0]]
    sex_row: list[Any] = ["性别"]
    for left, right in zip(header[1::2], header[2::2]):
        left_match = re.match(r"^(?P<dose>.+?)\s+(?P<sex>M|F)$", str(left or "").strip(), re.IGNORECASE)
        right_match = re.match(r"^(?P<dose>.+?)\s+(?P<sex>M|F)$", str(right or "").strip(), re.IGNORECASE)
        if left_match is None or right_match is None:
            return []
        left_dose = _markdown_normalize_dose_response_header_dose(left_match.group("dose"))
        right_dose = _markdown_normalize_dose_response_header_dose(right_match.group("dose"))
        left_sex = str(left_match.group("sex") or "").upper()
        right_sex = str(right_match.group("sex") or "").upper()
        if left_dose != right_dose or {left_sex, right_sex} != {"M", "F"}:
            return []
        group_row.extend([left_dose, right_dose])
        sex_row.extend([left_sex, right_sex])
    if dose_projection.get("source_has_explicit_sex_header_row"):
        return [group_row, sex_row, *rows[1:]]
    return [group_row, *rows[1:]]


def _markdown_normalize_dose_response_header_dose(value: str) -> str:
    text = str(value or "").strip()
    if text == "0":
        return "0"
    return text


def _project_markdown_study_condition_composite_grid(
    block: dict[str, Any],
    rows: list[list[Any]],
) -> list[list[Any]]:
    if not rows:
        return []
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    matrix_projection = projection.get("study_condition_grouped_result_matrix_projection")
    if not isinstance(matrix_projection, dict):
        return []
    if str(matrix_projection.get("presentation_boundary") or "") != "render_as_single_composite_table":
        return []
    header_groups = [
        group
        for group in matrix_projection.get("header_group_rows", []) or []
        if isinstance(group, dict)
    ]
    if not header_groups:
        return []
    header = list(rows[0])
    if len(header) < 2:
        return []
    descriptor_rows = _markdown_study_condition_descriptor_rows(header_groups, len(header))
    matrix_header_rows = _markdown_study_condition_matrix_header_rows(
        matrix_projection,
        len(header),
    )
    if descriptor_rows:
        if matrix_header_rows:
            return [*descriptor_rows, *matrix_header_rows, *rows[1:]]
        return [*descriptor_rows, header, *rows[1:]]

    flattened_header = [header[0]]
    for leaf_index, leaf_text in enumerate(header[1:], start=1):
        group = _study_condition_header_group_for_leaf(header_groups, leaf_index)
        group_label = _markdown_flatten_study_condition_group_label(group.get("label") if group else "")
        leaf_label = str(leaf_text or "").strip()
        if group_label and leaf_label:
            flattened_header.append(f"{group_label} {leaf_label}")
        else:
            flattened_header.append(leaf_label or group_label)
    return [flattened_header, *rows[1:]]


def _markdown_study_condition_matrix_header_rows(
    matrix_projection: dict[str, Any],
    column_count: int,
) -> list[list[Any]]:
    if column_count < 2:
        return []
    expected_roles = ("matrix_leaf_header", "matrix_row_axis")
    rows_by_role: dict[str, list[Any]] = {}
    for record in matrix_projection.get("matrix_header_rows", []) or []:
        if not isinstance(record, dict):
            continue
        role = str(record.get("role") or "").strip()
        stub = str(record.get("stub") or "").strip()
        cells = [str(cell or "").strip() for cell in record.get("cells", []) or []]
        if role not in expected_roles or not stub or len(cells) != column_count - 1:
            continue
        rows_by_role[role] = [stub, *cells]
    if any(role not in rows_by_role for role in expected_roles):
        return []
    return [rows_by_role[role] for role in expected_roles]


_STUDY_CONDITION_MARKDOWN_DESCRIPTOR_ORDER = [
    "种属",
    "性别(M/F)/动物数量",
    "进食情况",
    "溶媒/剂型",
    "给药方法",
    "剂量(mg/kg)",
    "分析物",
    "分析方法",
]


def _markdown_study_condition_descriptor_rows(
    header_groups: list[dict[str, Any]],
    column_count: int,
) -> list[list[Any]]:
    if column_count < 2:
        return []
    descriptor_labels = _markdown_study_condition_descriptor_labels(header_groups)
    if not descriptor_labels:
        return []
    rows: list[list[Any]] = []
    for label in descriptor_labels:
        row: list[Any] = [label]
        has_value = False
        for leaf_index in range(1, column_count):
            group = _study_condition_header_group_for_leaf(header_groups, leaf_index)
            descriptors = group.get("descriptors") if isinstance(group, dict) else {}
            descriptors = descriptors if isinstance(descriptors, dict) else {}
            value = str(descriptors.get(label) or "").strip()
            if value:
                has_value = True
            row.append(value)
        if has_value:
            rows.append(row)
    return rows


def _markdown_study_condition_descriptor_labels(header_groups: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    labels: list[str] = []
    for preferred_label in _STUDY_CONDITION_MARKDOWN_DESCRIPTOR_ORDER:
        for group in header_groups:
            descriptors = group.get("descriptors") if isinstance(group, dict) else {}
            if not isinstance(descriptors, dict):
                continue
            if preferred_label in descriptors and preferred_label not in seen:
                seen.add(preferred_label)
                labels.append(preferred_label)
                break
    return labels


def _append_markdown_study_condition_trailing_metadata_rows(
    block: dict[str, Any],
    rows: list[list[Any]],
) -> list[list[Any]]:
    if not rows or not _markdown_table_has_study_condition_result_matrix_projection(block):
        return rows
    metadata_rows = [
        row
        for row in block.get("_markdown_trailing_metadata_rows", []) or []
        if isinstance(row, list) and len(row) >= 2 and str(row[0] or "").strip()
    ]
    if not metadata_rows:
        return rows
    existing_labels = {_markdown_compact_table_text(row[0]) for row in rows if row and _markdown_compact_table_text(row[0])}
    column_count = max(len(row) for row in rows)
    appended_rows: list[list[Any]] = []
    for metadata_row in metadata_rows:
        label_norm = _markdown_compact_table_text(metadata_row[0])
        if label_norm and label_norm in existing_labels:
            continue
        row = list(metadata_row[:column_count])
        if len(row) < column_count:
            row.extend([""] * (column_count - len(row)))
        appended_rows.append(row)
        if label_norm:
            existing_labels.add(label_norm)
    if not appended_rows:
        return rows
    return [*rows, *appended_rows]


_MARKDOWN_STUDY_CONDITION_TRAILING_METADATA_LABELS = [
    "试验编号",
    "报告编号",
    "CTD 中的位置",
    "CTD中的位置",
    "CTD 位置",
    "CTD位置",
]


def _markdown_structure_template_trailing_metadata_rows(block: dict[str, Any]) -> list[list[Any]]:
    label_norms = {
        _markdown_compact_table_text(label)
        for label in _MARKDOWN_STUDY_CONDITION_TRAILING_METADATA_LABELS
    }
    rows: list[list[Any]] = []
    seen: set[str] = set()
    for raw_row in block.get("row_texts", []) or []:
        parsed = _markdown_parse_trailing_metadata_row(raw_row, label_norms)
        if parsed is None:
            continue
        label, value = parsed
        key = _markdown_compact_table_text(label)
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append([label, value])
    return rows


def _markdown_parse_trailing_metadata_row(
    value: Any,
    accepted_label_norms: set[str],
) -> tuple[str, str] | None:
    text = str(value or "").strip()
    if not text or "：" not in text and ":" not in text:
        return None
    parts = re.split(r"[:：]", text, maxsplit=1)
    if len(parts) != 2:
        return None
    label = re.sub(r"\s+", " ", parts[0]).strip()
    field_value = re.sub(r"\s+", " ", parts[1]).strip()
    if not label or not field_value:
        return None
    label_norm = _markdown_compact_table_text(label)
    if label_norm not in accepted_label_norms:
        return None
    canonical_label = _markdown_canonical_trailing_metadata_label(label)
    return canonical_label, field_value


def _markdown_canonical_trailing_metadata_label(label: str) -> str:
    normalized = re.sub(r"\s+", "", str(label or "")).strip()
    if normalized.upper().startswith("CTD") and "位置" in normalized:
        return "CTD 中的位置"
    return str(label or "").strip()


def _study_condition_header_group_for_leaf(
    header_groups: list[dict[str, Any]],
    leaf_index: int,
) -> dict[str, Any] | None:
    for group in header_groups:
        try:
            start_leaf_col = int(group.get("start_leaf_col", 0) or 0)
            end_leaf_col = int(group.get("end_leaf_col", start_leaf_col) or start_leaf_col)
        except (TypeError, ValueError):
            continue
        if start_leaf_col <= leaf_index <= end_leaf_col:
            return group
    return None


def _markdown_flatten_study_condition_group_label(value: Any) -> str:
    label = str(value or "").strip()
    if not label:
        return ""
    parts = [part.strip() for part in re.split(r"\s+/\s+", label) if part.strip()]
    normalized_parts: list[str] = []
    for part in parts:
        part = re.sub(r"(?<=\d)\s+mg/kg\b", "mg/kg", part, flags=re.IGNORECASE)
        part = re.sub(r"(?<=\d)mg/kg\b", "mgkg", part, flags=re.IGNORECASE)
        normalized_parts.append(part)
    return "-".join(normalized_parts)


def _markdown_table_has_word_logical_grid_projection(block: dict[str, Any]) -> bool:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    grouped_projection = projection.get("grouped_multilevel_borderless_projection")
    return (
        isinstance(grouped_projection, dict)
        and isinstance(grouped_projection.get("word_logical_grid_projection"), dict)
    )


def _markdown_table_has_study_condition_result_matrix_projection(block: dict[str, Any]) -> bool:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    matrix_projection = projection.get("study_condition_grouped_result_matrix_projection")
    return (
        isinstance(matrix_projection, dict)
        and str(matrix_projection.get("semantic_profile") or "") == "study_condition_grouped_result_matrix"
    )


def _markdown_table_suppress_note_norms_for_context(
    block: dict[str, Any],
    suppressed_by_context_id: dict[str, set[str]],
) -> set[str]:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    matrix_projection = projection.get("study_condition_grouped_result_matrix_projection")
    if not isinstance(matrix_projection, dict):
        return set()
    context_template_id = str(matrix_projection.get("context_template_id") or "").strip()
    if not context_template_id:
        return set()
    return set(suppressed_by_context_id.get(context_template_id) or set())


def _markdown_table_trailing_metadata_rows_for_context(
    block: dict[str, Any],
    rows_by_context_id: dict[str, list[list[Any]]],
) -> list[list[Any]]:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    matrix_projection = projection.get("study_condition_grouped_result_matrix_projection")
    if not isinstance(matrix_projection, dict):
        return []
    context_template_id = str(matrix_projection.get("context_template_id") or "").strip()
    if not context_template_id:
        return []
    return [list(row) for row in rows_by_context_id.get(context_template_id, []) if isinstance(row, list)]


def _markdown_structure_template_note_norms(block: dict[str, Any]) -> set[str]:
    return {
        norm
        for note in _structure_template_markdown_ordered_note_blocks(block)
        for norm in [_markdown_compact_table_text(note.get("text") or "")]
        if norm
    }


def _markdown_structure_template_note_norms_to_suppress_for_following_table(
    block: dict[str, Any],
    following_blocks: list[dict[str, Any]],
) -> set[str]:
    note_norms = _markdown_structure_template_note_norms(block)
    if not note_norms:
        return set()
    anchored_following_note_norms: set[str] = set()
    template_id = str(block.get("structure_template_id") or block.get("block_id") or "").strip()
    for candidate in following_blocks:
        if str(candidate.get("block_type") or "").strip().lower() == "structure_template":
            break
        if str(candidate.get("block_type") or "").strip().lower() != "table":
            continue
        projection = candidate.get("semantic_projection_v2")
        projection = projection if isinstance(projection, dict) else {}
        matrix_projection = projection.get("study_condition_grouped_result_matrix_projection")
        if template_id and isinstance(matrix_projection, dict):
            if str(matrix_projection.get("context_template_id") or "").strip() != template_id:
                continue
        for ref_key in ("cell_note_refs", "header_note_refs"):
            for ref in candidate.get(ref_key, []) or []:
                if not isinstance(ref, dict):
                    continue
                norm = _markdown_compact_table_text(ref.get("note_text") or "")
                if norm:
                    anchored_following_note_norms.add(norm)
    return note_norms - anchored_following_note_norms


def _markdown_table_has_overview_inventory_schema_projection(block: dict[str, Any]) -> bool:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    overview_projection = projection.get("overview_inventory_schema_projection")
    semantic_profile = str((overview_projection or {}).get("semantic_profile") or "") if isinstance(overview_projection, dict) else ""
    return (
        isinstance(overview_projection, dict)
        and semantic_profile
        in {
            "nonclinical_overview_inventory_table",
            "pk_overview_inventory_table",
            "toxicokinetic_overview_inventory_table",
        }
    )


def _markdown_table_has_study_metric_grouped_matrix_projection(block: dict[str, Any]) -> bool:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    metric_projection = projection.get("study_metric_grouped_matrix_projection")
    return (
        isinstance(metric_projection, dict)
        and str(metric_projection.get("semantic_profile") or "") == "study_metric_grouped_matrix"
    )


def _markdown_table_has_grouped_multilevel_header_projection(block: dict[str, Any]) -> bool:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    grouped_projection = projection.get("grouped_multilevel_header_projection")
    return (
        isinstance(grouped_projection, dict)
        and str(grouped_projection.get("semantic_profile") or "") == "grouped_multilevel_header_table"
    )


def _markdown_table_has_ind_late_semantic_projection(block: dict[str, Any]) -> bool:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    expected = {
        "genotoxicity_assay_matrix_projection": "genotoxicity_assay_matrix",
        "dose_response_result_panel_projection": "dose_response_result_panel",
        "study_metric_grouped_matrix_projection": "study_metric_grouped_matrix",
        "toxicology_summary_schema_projection": "toxicology_summary_schema_table",
    }
    for key, semantic_profile in expected.items():
        payload = projection.get(key)
        if isinstance(payload, dict) and str(payload.get("semantic_profile") or "") == semantic_profile:
            return True
    return False


def _should_export_table_semantic_projection_grid(
    block: dict[str, Any],
    projection: Any,
    semantic_grid: Any,
) -> bool:
    if not isinstance(projection, dict) or not isinstance(semantic_grid, list) or not semantic_grid:
        return False
    if _markdown_table_has_ind_late_semantic_projection(block):
        return True
    if str(projection.get("source") or "") != "table_semantic_projection_v2":
        return False
    semantic_rows = [row for row in semantic_grid if isinstance(row, list) and any(row)]
    if len(semantic_rows) < 2:
        return False
    table_family = str(block.get("table_family") or projection.get("table_family") or "")
    if table_family in {"keyed_long_list", "projected_stub_matrix"}:
        return True
    if table_family == "two_column_spanning_header_table":
        return True
    semantic_transform_keys = {
        "parallel_inventory_list_compaction",
        "keyed_long_description_row_compaction",
        "projected_stub_compaction",
        "multiline_schema_header_projection",
        "leading_boundary_header_projection",
        "compact_single_cell_header_projection",
        "two_column_spanning_header_projection",
        "label_after_value_pair_projection",
        "semantic_row_run_boundary_refinement",
        "blank_stub_column_projection",
        "single_column_keyed_list_projection",
        "compressed_measurement_matrix_projection",
        "flowchart_connector_projection",
        "stacked_numeric_atom_row_expansion",
        "compact_single_cell_body_row_projection",
        "multicolumn_wrapped_record_compaction",
        "rowspan_key_repetition_compaction",
        "external_header_grid_projection",
        "merged_adjacent_numeric_value_projection",
        "sparse_numeric_anchor_column_projection",
        "sparse_body_anchor_column_projection",
        "caption_context_blank_header_leaf_fill",
        "missing_leading_stub_header_projection",
        "grouped_multilevel_header_projection",
        "grouped_multilevel_borderless_projection",
        "study_condition_grouped_result_matrix_projection",
        "overview_inventory_schema_projection",
        "study_metric_grouped_matrix_projection",
        "genotoxicity_assay_matrix_projection",
        "dose_response_result_panel_projection",
        "toxicology_summary_schema_projection",
        "pre_table_parent_header_projection",
    }
    return any(key in projection for key in semantic_transform_keys)


def _markdown_display_rows_after_structural_prefix(block: dict[str, Any], column_count: int) -> list[list[str]]:
    display_grid = block.get("display_grid")
    if not isinstance(display_grid, list) or not display_grid:
        return []

    start_index = _markdown_table_display_data_start_index(block)
    normalized_rows = [
        [_markdown_escape_table_cell(cell) for cell in row]
        for row in display_grid[start_index:]
        if isinstance(row, list)
    ]
    normalized_rows = [row for row in normalized_rows if any(row)]
    if not normalized_rows:
        return []
    if any(len(row) > column_count for row in normalized_rows):
        return []
    data_grid = block.get("data_grid")
    if isinstance(data_grid, list):
        data_row_count = sum(1 for row in data_grid if isinstance(row, list) and any(row))
        if (
            data_row_count
            and len(normalized_rows) < data_row_count
            and not _markdown_data_grid_leading_row_duplicates_structural_header(block)
        ):
            return []
    return normalized_rows


def _markdown_data_grid_leading_row_duplicates_structural_header(block: dict[str, Any]) -> bool:
    display_grid = block.get("display_grid")
    data_grid = block.get("data_grid")
    if not isinstance(display_grid, list) or not isinstance(data_grid, list):
        return False
    if _markdown_table_display_data_start_index(block) <= 0:
        return False

    first_data_row = next((row for row in data_grid if isinstance(row, list) and any(row)), None)
    if first_data_row is None:
        return False

    data_signature = _markdown_row_compact_signature(first_data_row)
    header = [
        item.get("text")
        for item in block.get("header", []) or []
        if isinstance(item, dict)
    ]
    return bool(data_signature and header and data_signature == _markdown_row_compact_signature(header))


def _markdown_row_compact_signature(row: list[Any]) -> tuple[str, ...]:
    return tuple(_markdown_compact_table_text(cell) for cell in row if _markdown_compact_table_text(cell))


def _markdown_table_display_data_start_index(block: dict[str, Any]) -> int:
    if _markdown_table_header_is_external_to_display_grid(block):
        return 0
    data_start = block.get("data_start_row")
    if isinstance(data_start, int) and data_start >= 0:
        return data_start
    header_row = block.get("header_row_index")
    if isinstance(header_row, int) and header_row >= 0:
        return header_row + 1
    title_row = block.get("title_row_index")
    if isinstance(title_row, int) and title_row >= 0:
        return title_row + 1
    return 1


def _markdown_table_header_is_external_to_display_grid(block: dict[str, Any]) -> bool:
    header = [
        str(item.get("text") or "").strip()
        for item in block.get("header", []) or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    display_grid = block.get("display_grid")
    if not header or not isinstance(display_grid, list) or not display_grid:
        return False
    if isinstance(block.get("header_row_index"), int):
        return False
    if isinstance(block.get("title_row_index"), int):
        return False
    if not bool(block.get("header_rebuilt_by_context") or block.get("header_rebuilt_by_guard")):
        return False

    first_display_row = next((row for row in display_grid if isinstance(row, list) and any(row)), None)
    if first_display_row is None:
        return False
    first_texts = [str(cell or "").strip() for cell in first_display_row]
    if len(first_texts) != len(header):
        return False
    header_compact = [_markdown_compact_table_text(text) for text in header]
    first_compact = [_markdown_compact_table_text(text) for text in first_texts]
    overlap_count = sum(1 for text in first_compact if text and text in set(header_compact))
    if overlap_count >= max(2, len(header_compact) // 2):
        return False
    return True


def _markdown_compact_table_text(value: Any) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "").lower())


def _markdown_table_internal_title(block: dict[str, Any]) -> str:
    title_row = block.get("title_row_index")
    if not isinstance(title_row, int) or title_row < 0:
        return ""
    projection = block.get("semantic_projection_v2")
    if isinstance(projection, dict):
        leading_boundary = projection.get("leading_boundary_header_projection")
        if isinstance(leading_boundary, dict):
            try:
                dropped = int(leading_boundary.get("dropped_leading_row_count", 0) or 0)
            except (TypeError, ValueError):
                dropped = 0
            if title_row < dropped:
                return ""

    display_grid = block.get("display_grid")
    if isinstance(display_grid, list) and title_row < len(display_grid):
        row = display_grid[title_row]
        if isinstance(row, list):
            non_empty_cells = [
                _markdown_escape_inline_text(cell)
                for cell in row
                if str(cell or "").strip()
            ]
            if len(non_empty_cells) == 1:
                return non_empty_cells[0]

    return _markdown_escape_inline_text(block.get("title"))


def _markdown_table_external_title(block: dict[str, Any], internal_title: str) -> str:
    projection = block.get("semantic_projection_v2")
    if isinstance(projection, dict) and isinstance(projection.get("leading_boundary_header_projection"), dict):
        return ""
    title = _markdown_escape_inline_text(block.get("title"))
    if not title:
        return ""
    if internal_title and _markdown_compact_table_text(title) == _markdown_compact_table_text(internal_title):
        return ""
    if re.fullmatch(r"(?:table|表)\s*\d+\s*[:.\-]?\s*(?:continued|续表)\.?", title, re.IGNORECASE):
        return ""
    return title


def _markdown_title_reference_policy(block: dict[str, Any]) -> str:
    direct_policy = str(block.get("metadata_title_reference_policy") or "").strip()
    if direct_policy:
        return direct_policy
    composite = block.get("composite_object")
    if isinstance(composite, dict):
        return str(composite.get("metadata_title_reference_policy") or "").strip()
    return ""


def _markdown_object_title_redundant_with_previous_heading(block: dict[str, Any], previous_visible_title: str) -> bool:
    if not previous_visible_title:
        return False
    if _markdown_title_reference_policy(block) != "may_reference_without_visible_rendering":
        return False
    title = str(block.get("title") or block.get("caption_text") or "").strip()
    if not title:
        return False
    return _markdown_compact_table_text(title) == _markdown_compact_table_text(previous_visible_title)


def _markdown_continuation_table_title_redundant_with_previous_visible_title(
    block: dict[str, Any],
    previous_visible_title: str,
) -> bool:
    if not previous_visible_title:
        return False
    if not _markdown_table_is_continuation(block):
        return False
    title = str(block.get("title") or block.get("caption_text") or "").strip()
    if not title:
        return False
    return _markdown_table_titles_are_redundant(previous_visible_title, title)


def _markdown_table_is_continuation(block: dict[str, Any]) -> bool:
    return bool(
        str(block.get("continued_from_table_id") or "").strip()
        or _as_string_list(block.get("continued_from"))
    )


def _markdown_table_caption_ref(text: str) -> str:
    compact = _markdown_compact_table_text(text)
    match = re.match(r"(?:table|琛?)(\d+)", compact, re.IGNORECASE)
    return match.group(1) if match else ""


def _markdown_table_titles_are_redundant(primary_title: str, candidate_title: str) -> bool:
    primary_compact = _markdown_compact_table_text(primary_title)
    candidate_compact = _markdown_compact_table_text(candidate_title)
    if not primary_compact or not candidate_compact:
        return False
    if primary_compact == candidate_compact:
        return True
    primary_ref = _markdown_table_caption_ref(primary_title)
    candidate_ref = _markdown_table_caption_ref(candidate_title)
    if primary_ref and primary_ref == candidate_ref:
        return primary_compact.startswith(candidate_compact) or candidate_compact.startswith(primary_compact)
    return False


def _markdown_structure_template_absorbed_by_business_table(block: dict[str, Any]) -> bool:
    if str(block.get("ownership_domain") or "").strip() == "absorbed_by_business_table":
        return True
    if str(block.get("semantic_role") or "").strip() == "absorbed_structure_template_fragment":
        return True
    return str(block.get("template_profile") or "").strip().startswith("absorbed_")


def _markdown_visible_render_policy(block: dict[str, Any]) -> str:
    direct_policy = str(block.get("visible_render_policy") or "").strip()
    if direct_policy:
        return direct_policy
    composite = block.get("composite_object")
    if isinstance(composite, dict):
        return str(composite.get("visible_render_policy") or "").strip()
    return ""


def _markdown_grid_row_text(row: list[Any]) -> str:
    return " ".join(
        _markdown_escape_inline_text(cell)
        for cell in row
        if str(cell or "").strip()
    ).strip()


def _drop_markdown_table_embedded_title_rows(
    grid: list[list[str]],
    block: dict[str, Any],
    rendered_title: str,
) -> list[list[str]]:
    if not grid or not rendered_title:
        return grid
    title_compact = _markdown_compact_table_text(rendered_title)
    if not title_compact:
        return grid

    rows = list(grid)
    dropped = 0
    while rows and dropped < 3:
        row_text = _markdown_grid_row_text(rows[0])
        row_compact = _markdown_compact_table_text(row_text)
        if not row_compact:
            rows = rows[1:]
            dropped += 1
            continue
        if row_compact == title_compact or row_compact in title_compact:
            rows = rows[1:]
            dropped += 1
            continue
        title_row = block.get("title_row_index")
        if dropped == 0 and isinstance(title_row, int) and title_row == 0 and title_compact.startswith(row_compact):
            rows = rows[1:]
            dropped += 1
            continue
        break
    return rows


def _normalize_markdown_table_grid(block: dict[str, Any]) -> list[list[str]]:
    raw_preserved_grid = _markdown_raw_grid_for_preserved_trailing_structural_rows(block)
    if raw_preserved_grid:
        return raw_preserved_grid

    semantic_grid = _normalize_markdown_table_semantic_header_grid(block)
    if semantic_grid:
        return semantic_grid

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


def _markdown_raw_grid_for_preserved_trailing_structural_rows(block: dict[str, Any]) -> list[list[str]]:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    preserved = projection.get("preserved_reclaimed_trailing_structural_label_rows")
    if not isinstance(preserved, dict):
        return []
    raw_grid = block.get("raw_grid")
    if not isinstance(raw_grid, list) or not raw_grid:
        return []
    expected_texts = [
        _markdown_compact_table_text(text)
        for text in preserved.get("texts", []) or []
        if _markdown_compact_table_text(text)
    ]
    if not expected_texts:
        return []
    normalized_rows = [
        [_markdown_escape_table_cell(cell) for cell in row]
        for row in raw_grid
        if isinstance(row, list)
    ]
    normalized_rows = [row for row in normalized_rows if any(cell for cell in row)]
    if not normalized_rows:
        return []
    row_signatures = {_markdown_row_compact_signature(row) for row in normalized_rows}
    if not all((text,) in row_signatures for text in expected_texts):
        return []
    return normalized_rows


def _normalize_markdown_table_evidence_grid(block: dict[str, Any]) -> list[list[str]]:
    semantic_grid = block.get("semantic_grid")
    projection = block.get("semantic_projection_v2")
    has_pre_table_parent_header_projection = (
        isinstance(projection, dict)
        and isinstance(projection.get("pre_table_parent_header_projection"), dict)
    )
    if (
        (
            _markdown_table_has_word_logical_grid_projection(block)
            or _markdown_table_has_study_condition_result_matrix_projection(block)
            or _markdown_table_has_overview_inventory_schema_projection(block)
            or _markdown_table_has_study_metric_grouped_matrix_projection(block)
            or _markdown_table_has_grouped_multilevel_header_projection(block)
            or _markdown_table_has_ind_late_semantic_projection(block)
            or has_pre_table_parent_header_projection
        )
        and _should_export_table_semantic_projection_grid(block, projection, semantic_grid)
    ):
        semantic_rows = _normalize_markdown_table_semantic_header_grid(block)
        if semantic_rows:
            return semantic_rows

    raw_preserved_grid = _markdown_raw_grid_for_preserved_trailing_structural_rows(block)
    if raw_preserved_grid:
        return raw_preserved_grid

    for key in ("display_grid", "raw_grid", "grid", "data_grid"):
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
    return _normalize_markdown_table_grid(block)


def _markdown_escape_inline_text(value: Any) -> str:
    text = str(project_pdf_math_symbol_display_text(project_table_cell_display_text(value)) or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.replace("\\", "\\\\").replace("*", "\\*").replace("[", "\\[").replace("]", "\\]")


def _markdown_structure_template_note_text(value: Any) -> str:
    text = _markdown_escape_inline_text(value)
    text = re.sub(r"^([+-])(?=\s)", r"\\\1", text)
    text = re.sub(r"^(\d{1,9})([.)])(?=\s)", r"\1\\\2", text)
    return re.sub(r"^([>#])(?=\s|$)", r"\\\1", text)


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


def _markdown_projected_grid_source_row_numbers(
    block: dict[str, Any],
    grid: list[list[str]],
) -> list[int | None]:
    semantic_grid = block.get("semantic_display_grid") or block.get("semantic_grid")
    if not isinstance(semantic_grid, list) or not semantic_grid:
        return [None for _ in grid]
    source_rows_by_signature: dict[tuple[str, ...], list[int]] = {}
    for source_row_number, row in enumerate(semantic_grid, start=1):
        if not isinstance(row, list):
            continue
        normalized = [_markdown_escape_table_cell(cell) for cell in row]
        signature = _markdown_row_compact_signature(normalized)
        if signature and any(signature):
            source_rows_by_signature.setdefault(signature, []).append(source_row_number)

    origins: list[int | None] = []
    consumed_by_signature: dict[tuple[str, ...], int] = {}
    for row in grid:
        signature = _markdown_row_compact_signature(row)
        candidates = source_rows_by_signature.get(signature, [])
        consumed = consumed_by_signature.get(signature, 0)
        if consumed >= len(candidates):
            origins.append(None)
            continue
        origins.append(candidates[consumed])
        consumed_by_signature[signature] = consumed + 1
    return origins


def _markdown_projected_grid_row_provenance(
    block: dict[str, Any],
    grid: list[list[str]],
) -> list[dict[str, Any]]:
    semantic_grid = block.get("semantic_display_grid") or block.get("semantic_grid")
    semantic_lineage = block.get("semantic_row_provenance")
    if (
        not isinstance(semantic_grid, list)
        or not isinstance(semantic_lineage, list)
        or len(semantic_grid) != len(semantic_lineage)
    ):
        return [
            {
                "source_row_refs": [],
                "derivation": "missing_or_unaligned_semantic_row_provenance",
            }
            for _ in grid
        ]

    lineage_by_signature: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row, lineage in zip(semantic_grid, semantic_lineage):
        if not isinstance(row, list) or not isinstance(lineage, dict):
            continue
        normalized = [_markdown_escape_table_cell(cell) for cell in row]
        signature = _markdown_row_compact_signature(normalized)
        if signature:
            lineage_by_signature.setdefault(signature, []).append(lineage)

    consumed_by_signature: dict[tuple[str, ...], int] = {}
    projected: list[dict[str, Any]] = []
    for row in grid:
        signature = _markdown_row_compact_signature(row)
        candidates = lineage_by_signature.get(signature, [])
        consumed = consumed_by_signature.get(signature, 0)
        if consumed < len(candidates):
            lineage = candidates[consumed]
            consumed_by_signature[signature] = consumed + 1
            projected.append(
                {
                    "source_row_refs": list(lineage.get("source_row_refs", []) or []),
                    "derivation": str(lineage.get("derivation") or "semantic_row_provenance"),
                }
            )
            continue
        projected.append(
            {
                "source_row_refs": [],
                "derivation": "presentation_only_or_unmapped_row",
            }
        )
    return projected


def _append_markdown_row_provenance_diagnostic(
    block: dict[str, Any],
    *,
    reason: str,
    merged_row: dict[str, Any],
    candidate_rows: list[int],
) -> None:
    diagnostics = block.setdefault("markdown_row_provenance_diagnostics", [])
    record = {
        "reason": reason,
        "source_row_ref": str(merged_row.get("source_row_ref") or ""),
        "text": str(merged_row.get("text") or ""),
        "candidate_rows": list(candidate_rows),
    }
    if record not in diagnostics:
        diagnostics.append(record)


def _merged_rows_by_projected_grid_row(
    block: dict[str, Any],
    grid: list[list[str]],
) -> dict[int, dict[str, Any]]:
    source_merged_rows = _merged_rows_by_row(block)
    if not source_merged_rows or not grid:
        return source_merged_rows

    row_provenance = _markdown_projected_grid_row_provenance(block, grid)
    projected_rows_by_ref: dict[str, list[int]] = {}
    for projected_row_number, lineage in enumerate(row_provenance, start=1):
        for row_ref in lineage.get("source_row_refs", []) or []:
            source_ref = str(row_ref or "").strip()
            if source_ref:
                projected_rows_by_ref.setdefault(source_ref, []).append(projected_row_number)

    projected: dict[int, dict[str, Any]] = {}
    for metadata in source_merged_rows.values():
        source_ref = str(metadata.get("source_row_ref") or "").strip()
        if source_ref:
            candidate_rows = projected_rows_by_ref.get(source_ref, [])
            if len(candidate_rows) == 1:
                projected[candidate_rows[0]] = metadata
                continue
            _append_markdown_row_provenance_diagnostic(
                block,
                reason="unresolved_source_row_ref" if not candidate_rows else "ambiguous_source_row_ref",
                merged_row=metadata,
                candidate_rows=candidate_rows,
            )
            continue

        text = _markdown_escape_table_cell(metadata.get("text"))
        signature = _markdown_row_compact_signature([text])
        candidate_rows = [
            row_number
            for row_number, row in enumerate(grid, start=1)
            if signature and _markdown_row_compact_signature(row) == signature
        ]
        if len(candidate_rows) == 1:
            projected[candidate_rows[0]] = metadata
            continue
        _append_markdown_row_provenance_diagnostic(
            block,
            reason="unresolved_legacy_signature" if not candidate_rows else "ambiguous_legacy_signature",
            merged_row=metadata,
            candidate_rows=candidate_rows,
        )
    return projected


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


def _html_table_cell_text(value: Any) -> str:
    text = str(project_pdf_math_symbol_display_text(project_table_cell_display_text(value)) or "").strip()
    return re.sub(r"\s+", " ", text)


def _html_table_compact_cell_text(value: Any) -> str:
    return re.sub(r"\s+", "", _html_table_cell_text(value)).lower()


def _semantic_html_table_header_row_count(block: dict[str, Any], row_count: int) -> int:
    header_rows = 1
    if isinstance(block.get("cell_spans"), list):
        header_rows = _markdown_canonical_header_depth(block)
        return max(1, min(header_rows, row_count))
    for cell in block.get("logical_cells", []) or []:
        if not isinstance(cell, dict):
            continue
        source = str(cell.get("source") or "").lower()
        if "header" not in source:
            continue
        try:
            row = int(cell.get("row", 0) or 0)
            rowspan = max(1, int(cell.get("rowspan", 1) or 1))
        except (TypeError, ValueError):
            continue
        header_rows = max(header_rows, row + rowspan)
    data_start = block.get("data_start_row")
    if (
        isinstance(data_start, int)
        and data_start > 0
        and not _markdown_table_has_ind_late_semantic_projection(block)
    ):
        header_rows = max(header_rows, min(data_start, row_count))
    semantic_grid = [
        row for row in block.get("semantic_grid", []) or []
        if isinstance(row, list)
    ]
    if len(semantic_grid) >= 2 and _markdown_grid_rows_are_multilevel_header_pair(
        semantic_grid[0],
        semantic_grid[1],
    ):
        header_rows = max(header_rows, 2)
    return max(1, min(header_rows, row_count))


def _semantic_html_table_span_cells(
    block: dict[str, Any],
    *,
    row_count: int,
    column_count: int,
) -> tuple[dict[tuple[int, int], dict[str, Any]], set[tuple[int, int]]]:
    semantic_rows = [
        row if isinstance(row, list) else []
        for row in (block.get("semantic_grid") or [])
    ]

    def semantic_cell(row: int, col: int) -> str:
        if row < 0 or row >= len(semantic_rows):
            return ""
        values = semantic_rows[row]
        if col < 0 or col >= len(values):
            return ""
        return _html_table_cell_text(values[col])

    def span_matches_semantic_grid(cell: dict[str, Any]) -> bool:
        text = _html_table_compact_cell_text(cell.get("text"))
        if not text:
            return False
        row = int(cell["row"])
        col = int(cell["col"])
        source = str(cell.get("source") or "").strip().lower()
        if source.startswith("sparse_header_"):
            saw_anchor_text = False
            for covered_row in range(row, row + int(cell["rowspan"])):
                for covered_col in range(col, col + int(cell["colspan"])):
                    covered_text = _html_table_compact_cell_text(semantic_cell(covered_row, covered_col))
                    if not covered_text:
                        continue
                    if covered_text != text:
                        return False
                    saw_anchor_text = True
            return saw_anchor_text
        anchor = _html_table_compact_cell_text(semantic_cell(row, col))
        if anchor != text:
            return False
        for covered_row in range(row, row + int(cell["rowspan"])):
            for covered_col in range(col, col + int(cell["colspan"])):
                if covered_row == row and covered_col == col:
                    continue
                covered_text = _html_table_compact_cell_text(semantic_cell(covered_row, covered_col))
                if covered_text and covered_text != text:
                    return False
        return True

    occupied: dict[tuple[int, int], dict[str, Any]] = {}
    covered: set[tuple[int, int]] = set()
    candidates: list[dict[str, Any]] = []
    for cell in _semantic_table_span_cells(block):
        if not isinstance(cell, dict):
            continue
        try:
            row = int(cell.get("row", 0) or 0)
            col = int(cell.get("col", 0) or 0)
            rowspan = max(1, int(cell.get("rowspan", 1) or 1))
            colspan = max(1, int(cell.get("colspan", 1) or 1))
        except (TypeError, ValueError):
            continue
        text = _html_table_cell_text(cell.get("text"))
        if not text or row < 0 or col < 0 or row >= row_count or col >= column_count:
            continue
        if rowspan == 1 and colspan == 1:
            continue
        rowspan = min(rowspan, row_count - row)
        colspan = min(colspan, column_count - col)
        if rowspan <= 1 and colspan <= 1:
            continue
        candidates.append(
            {
                "row": row,
                "col": col,
                "rowspan": rowspan,
                "colspan": colspan,
                "text": text,
                "source": str(cell.get("source") or ""),
            }
        )

    candidates.sort(key=lambda item: (item["row"], item["col"], -(item["rowspan"] * item["colspan"])))
    for cell in candidates:
        if not span_matches_semantic_grid(cell):
            continue
        anchor = (cell["row"], cell["col"])
        if anchor in covered:
            continue
        occupied[anchor] = cell
        for row in range(cell["row"], cell["row"] + cell["rowspan"]):
            for col in range(cell["col"], cell["col"] + cell["colspan"]):
                if (row, col) != anchor:
                    covered.add((row, col))
    return occupied, covered


def _display_grid_header_row_count(block: dict[str, Any], row_count: int) -> int:
    data_start = block.get("data_start_row")
    if isinstance(data_start, int) and data_start > 1:
        return min(data_start, row_count)
    return 0


def _display_grid_header_span_cells(
    rows: list[list[Any]],
    *,
    header_row_count: int,
    column_count: int,
) -> tuple[dict[tuple[int, int], dict[str, Any]], set[tuple[int, int]]]:
    padded_rows = [row + [None] * (column_count - len(row)) for row in rows]
    vertical_covered: set[tuple[int, int]] = set()
    for row_index in range(header_row_count):
        for col_index in range(column_count):
            if (row_index, col_index) in vertical_covered:
                continue
            text = _html_table_cell_text(padded_rows[row_index][col_index])
            if not text:
                continue
            probe_row = row_index + 1
            while (
                probe_row < header_row_count
                and not _html_table_cell_text(padded_rows[probe_row][col_index])
            ):
                vertical_covered.add((probe_row, col_index))
                probe_row += 1

    occupied: dict[tuple[int, int], dict[str, Any]] = {}
    covered: set[tuple[int, int]] = set()
    for row_index in range(header_row_count):
        for col_index in range(column_count):
            if (row_index, col_index) in covered:
                continue
            text = _html_table_cell_text(padded_rows[row_index][col_index])
            if not text:
                continue

            colspan = 1
            while (
                col_index + colspan < column_count
                and not _html_table_cell_text(padded_rows[row_index][col_index + colspan])
                and (row_index, col_index + colspan) not in vertical_covered
            ):
                colspan += 1

            rowspan = 1
            if colspan == 1:
                while (
                    row_index + rowspan < header_row_count
                    and not _html_table_cell_text(padded_rows[row_index + rowspan][col_index])
                ):
                    rowspan += 1

            if rowspan <= 1 and colspan <= 1:
                continue
            cell = {
                "row": row_index,
                "col": col_index,
                "rowspan": rowspan,
                "colspan": colspan,
                "text": text,
                "source": "display_grid_header_span_projection",
            }
            occupied[(row_index, col_index)] = cell
            for covered_row in range(row_index, row_index + rowspan):
                for covered_col in range(col_index, col_index + colspan):
                    if (covered_row, covered_col) != (row_index, col_index):
                        covered.add((covered_row, covered_col))
    return occupied, covered


def _build_display_grid_header_span_html_table(block: dict[str, Any]) -> str | None:
    display_grid = block.get("display_grid")
    if not isinstance(display_grid, list) or not display_grid:
        return None
    rows = [row for row in display_grid if isinstance(row, list)]
    if len(rows) < 3:
        return None
    column_count = max((len(row) for row in rows), default=0)
    if column_count <= 1:
        return None
    header_row_count = _display_grid_header_row_count(block, len(rows))
    if header_row_count < 2:
        return None
    header_rows = rows[:header_row_count]
    if not any(
        not _html_table_cell_text(cell)
        for row in header_rows
        for cell in (row + [None] * (column_count - len(row)))[:column_count]
    ):
        return None

    span_cells, covered = _display_grid_header_span_cells(
        rows,
        header_row_count=header_row_count,
        column_count=column_count,
    )
    if not span_cells:
        return None

    html_lines = ["<table>"]
    for row_index, row in enumerate(rows):
        html_lines.append("  <tr>")
        padded_row = row + [None] * (column_count - len(row))
        for col_index, value in enumerate(padded_row[:column_count]):
            if (row_index, col_index) in covered:
                continue
            span_cell = span_cells.get((row_index, col_index))
            text = span_cell["text"] if span_cell else _html_table_cell_text(value)
            if row_index < header_row_count and not text:
                continue
            tag = "th" if row_index < header_row_count else "td"
            attributes: list[str] = []
            if span_cell:
                if span_cell["rowspan"] > 1:
                    attributes.append(f'rowspan="{span_cell["rowspan"]}"')
                if span_cell["colspan"] > 1:
                    attributes.append(f'colspan="{span_cell["colspan"]}"')
            attr_text = (" " + " ".join(attributes)) if attributes else ""
            html_lines.append(f"    <{tag}{attr_text}>{html.escape(text)}</{tag}>")
        html_lines.append("  </tr>")
    html_lines.append("</table>")
    return "\n".join(html_lines)


def _build_display_header_semantic_body_html_table(block: dict[str, Any]) -> str | None:
    display_grid = block.get("display_grid")
    semantic_grid = block.get("semantic_grid")
    if not isinstance(display_grid, list) or not isinstance(semantic_grid, list):
        return None
    display_rows = [row for row in display_grid if isinstance(row, list)]
    semantic_rows = [row for row in semantic_grid if isinstance(row, list)]
    if len(display_rows) < 3 or len(semantic_rows) < 3:
        return None
    column_count = max((len(row) for row in display_rows), default=0)
    if column_count <= 1:
        return None
    if max((len(row) for row in semantic_rows), default=0) != column_count:
        return None

    header_row_count = _display_grid_header_row_count(block, len(display_rows))
    if header_row_count < 2 or len(semantic_rows) <= header_row_count:
        return None
    display_header_rows = display_rows[:header_row_count]
    if not any(
        not _html_table_cell_text(cell)
        for row in display_header_rows
        for cell in (row + [None] * (column_count - len(row)))[:column_count]
    ):
        return None
    span_cells, covered = _display_grid_header_span_cells(
        display_rows,
        header_row_count=header_row_count,
        column_count=column_count,
    )
    if not span_cells:
        return None

    body_rows = semantic_rows[header_row_count:]
    if not body_rows:
        return None

    html_lines = ["<table>"]
    for row_index, row in enumerate(display_header_rows):
        html_lines.append("  <tr>")
        padded_row = row + [None] * (column_count - len(row))
        for col_index, value in enumerate(padded_row[:column_count]):
            if (row_index, col_index) in covered:
                continue
            span_cell = span_cells.get((row_index, col_index))
            text = span_cell["text"] if span_cell else _html_table_cell_text(value)
            if not text:
                continue
            attributes: list[str] = []
            if span_cell:
                if span_cell["rowspan"] > 1:
                    attributes.append(f'rowspan="{span_cell["rowspan"]}"')
                if span_cell["colspan"] > 1:
                    attributes.append(f'colspan="{span_cell["colspan"]}"')
            attr_text = (" " + " ".join(attributes)) if attributes else ""
            html_lines.append(f"    <th{attr_text}>{html.escape(text)}</th>")
        html_lines.append("  </tr>")

    for row in body_rows:
        html_lines.append("  <tr>")
        padded_row = row + [None] * (column_count - len(row))
        for value in padded_row[:column_count]:
            html_lines.append(f"    <td>{html.escape(_html_table_cell_text(value))}</td>")
        html_lines.append("  </tr>")
    html_lines.append("</table>")
    return "\n".join(html_lines)


def _build_semantic_html_table(block: dict[str, Any]) -> str | None:
    semantic_grid = block.get("semantic_grid")
    if not isinstance(semantic_grid, list) or not semantic_grid:
        return _build_display_grid_header_span_html_table(block)
    rows = [row for row in semantic_grid if isinstance(row, list)]
    if not rows:
        return None
    column_count = max((len(row) for row in rows), default=0)
    if column_count <= 0:
        return None
    table_family = str(block.get("table_family") or "").strip()
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    row_count = len(rows)
    explicit_spans = _semantic_table_span_cells(block)
    if isinstance(block.get("cell_spans"), list):
        inferred_section_span_cells, inferred_section_covered = {}, set()
    else:
        inferred_section_span_cells, inferred_section_covered = _semantic_html_table_two_column_section_divider_span_cells(
            block,
            rows=rows,
            row_count=row_count,
            column_count=column_count,
        )
    has_explicit_span = bool(explicit_spans)
    has_inferred_section_span = bool(inferred_section_span_cells)
    if (
        not has_explicit_span
        and not has_inferred_section_span
        and table_family not in {"projected_stub_matrix"}
    ):
        if (
            "missing_leading_stub_header_projection" not in projection
            and "leading_boundary_header_projection" not in projection
            and "multiline_schema_header_projection" not in projection
        ):
            return _build_display_header_semantic_body_html_table(block)
    if (
        has_explicit_span
        and not has_inferred_section_span
        and not _should_auto_render_table_as_semantic_html(block)
        and "leading_boundary_header_projection" not in projection
        and "multiline_schema_header_projection" not in projection
    ):
        return _build_display_header_semantic_body_html_table(block)
    span_cells, covered = _semantic_html_table_span_cells(
        block,
        row_count=row_count,
        column_count=column_count,
    )
    for anchor, cell in inferred_section_span_cells.items():
        if anchor in covered:
            continue
        span_cells[anchor] = cell
    covered.update(inferred_section_covered)

    header_row_count = _semantic_html_table_header_row_count(block, row_count)
    html_lines = ["<table>"]
    for row_index, row in enumerate(rows):
        html_lines.append("  <tr>")
        padded_row = row + [None] * (column_count - len(row))
        for col_index, value in enumerate(padded_row):
            if (row_index, col_index) in covered:
                continue
            span_cell = span_cells.get((row_index, col_index))
            text = span_cell["text"] if span_cell else _html_table_cell_text(value)
            tag = "th" if row_index < header_row_count else "td"
            attributes: list[str] = []
            if span_cell:
                if span_cell["rowspan"] > 1:
                    attributes.append(f'rowspan="{span_cell["rowspan"]}"')
                if span_cell["colspan"] > 1:
                    attributes.append(f'colspan="{span_cell["colspan"]}"')
            attr_text = (" " + " ".join(attributes)) if attributes else ""
            html_lines.append(f"    <{tag}{attr_text}>{html.escape(text)}</{tag}>")
        html_lines.append("  </tr>")
    html_lines.append("</table>")
    return "\n".join(html_lines)


def _build_keyed_long_list_html_table(block: dict[str, Any]) -> str | None:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    table_family = str(block.get("table_family") or projection.get("table_family") or "").strip()
    if table_family != "keyed_long_list":
        return None
    if "single_column_keyed_list_projection" not in projection:
        return None

    semantic_grid = block.get("semantic_grid")
    if not isinstance(semantic_grid, list) or len(semantic_grid) != 2:
        return None
    rows = [row for row in semantic_grid if isinstance(row, list)]
    if len(rows) != 2 or len(rows[0]) != 2 or len(rows[1]) != 2:
        return None
    if _html_table_compact_cell_text(rows[0][0]) != "field" or _html_table_compact_cell_text(rows[0][1]) != "value":
        return None
    key = _html_table_cell_text(rows[1][0])
    value = _html_table_cell_text(rows[1][1])
    if not key or not value:
        return None

    return "\n".join(
        [
            "<table>",
            "  <tr>",
            f"    <td>{html.escape(key)}</td>",
            f"    <td>{html.escape(value)}</td>",
            "  </tr>",
            "</table>",
        ]
    )


def _semantic_html_table_two_column_section_divider_span_cells(
    block: dict[str, Any],
    *,
    rows: list[list[Any]],
    row_count: int,
    column_count: int,
) -> tuple[dict[tuple[int, int], dict[str, Any]], set[tuple[int, int]]]:
    if column_count != 2 or row_count < 3:
        return {}, set()
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    table_family = str(block.get("table_family") or projection.get("table_family") or "").strip()
    if table_family != "two_column_inventory":
        return {}, set()
    if "label_after_value_pair_projection" not in projection:
        return {}, set()

    def cell_text(row_index: int, col_index: int) -> str:
        if row_index < 0 or row_index >= row_count:
            return ""
        row = rows[row_index]
        if col_index < 0 or col_index >= len(row):
            return ""
        return _html_table_cell_text(row[col_index])

    def is_populated_pair(row_index: int) -> bool:
        return bool(cell_text(row_index, 0) and cell_text(row_index, 1))

    occupied: dict[tuple[int, int], dict[str, Any]] = {}
    covered: set[tuple[int, int]] = set()
    for row_index in range(row_count):
        left_text = cell_text(row_index, 0)
        right_text = cell_text(row_index, 1)
        if not left_text or right_text:
            continue
        if not any(is_populated_pair(idx) for idx in range(0, row_index)):
            continue
        if not any(is_populated_pair(idx) for idx in range(row_index + 1, row_count)):
            continue
        if left_text.rstrip().endswith((".", "!", "?")):
            continue
        occupied[(row_index, 0)] = {
            "row": row_index,
            "col": 0,
            "rowspan": 1,
            "colspan": 2,
            "text": left_text,
            "source": "two_column_section_divider_projection",
        }
        covered.add((row_index, 1))
    return occupied, covered


def _semantic_table_span_cells(block: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(block.get("cell_spans"), list):
        canonical: list[dict[str, Any]] = []
        for cell in block.get("cell_spans", []) or []:
            if not isinstance(cell, dict):
                continue
            try:
                rowspan = max(1, int(cell.get("rowspan", 1) or 1))
                colspan = max(1, int(cell.get("colspan", 1) or 1))
            except (TypeError, ValueError):
                continue
            if rowspan <= 1 and colspan <= 1:
                continue
            canonical.append(
                {
                    **cell,
                    "rowspan": rowspan,
                    "colspan": colspan,
                    "source": str(cell.get("source") or cell.get("evidence") or "canonical_cell_span"),
                }
            )
        return canonical
    spans: list[dict[str, Any]] = []
    for cell in [
        *(block.get("logical_cells") or []),
        *(block.get("presentation_spans") or []),
    ]:
        if not isinstance(cell, dict):
            continue
        try:
            rowspan = max(1, int(cell.get("rowspan", 1) or 1))
            colspan = max(1, int(cell.get("colspan", 1) or 1))
        except (TypeError, ValueError):
            continue
        if rowspan <= 1 and colspan <= 1:
            continue
        spans.append(cell)
    return spans


def _semantic_grid_has_layered_header_signature(semantic_grid: Any) -> bool:
    if not isinstance(semantic_grid, list) or len(semantic_grid) < 3:
        return False
    rows = [row for row in semantic_grid if isinstance(row, list)]
    if len(rows) < 3:
        return False
    column_count = max((len(row) for row in rows), default=0)
    if column_count < 3:
        return False

    header_band = rows[: min(3, len(rows) - 1)]
    data_band = rows[len(header_band):]
    if not data_band:
        return False

    def populated_ratio(row: list[Any]) -> float:
        padded = row + [None] * (column_count - len(row))
        populated = sum(1 for value in padded[:column_count] if _html_table_cell_text(value))
        return populated / column_count

    header_blank_rich_rows = sum(1 for row in header_band if populated_ratio(row) <= 0.75)
    dense_data_rows = sum(1 for row in data_band[:5] if populated_ratio(row) >= 0.55)
    return header_blank_rich_rows > 0 and dense_data_rows > 0


def _should_auto_render_table_as_semantic_html(block: dict[str, Any]) -> bool:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    table_family = str(block.get("table_family") or projection.get("table_family") or "").strip()

    flow_projection_keys = {
        "keyed_long_description_row_compaction",
        "parallel_inventory_list_compaction",
        "projected_stub_compaction",
        "leading_boundary_header_projection",
        "flowchart_connector_projection",
        "single_column_keyed_list_projection",
    }
    if any(key in projection for key in flow_projection_keys):
        return False
    if table_family in {"keyed_long_list", "two_column_inventory", "flowchart_matrix", "projected_stub_matrix"}:
        return False
    if table_family == "two_column_spanning_header_table":
        return True
    if isinstance(projection.get("multiline_schema_header_projection"), dict):
        return True

    if _table_has_source_backed_presentation_spans(block):
        return True

    spans = _semantic_table_span_cells(block)
    if not spans:
        return False

    header_like_span_sources = {
        "header_row_group",
        "header_column_group",
        "multirow_header_projection",
        "multicolumn_header_projection",
        "column_group_header_projection",
        "row_header_projection",
        "stub_header_projection",
    }
    for span in spans:
        source = str(span.get("source") or "").strip().lower()
        try:
            row = int(span.get("row", 0) or 0)
            colspan = max(1, int(span.get("colspan", 1) or 1))
        except (TypeError, ValueError):
            row = 0
            colspan = 1
        if row <= 1 and (colspan > 1 or any(token in source for token in header_like_span_sources)):
            return True

    source = str(block.get("source") or block.get("detection_source") or block.get("detection_method") or "").strip()
    if table_family == "rowspan_grouped_table" and source in {
        "caption_anchored_horizontal_rules",
        "pymupdf_builtin",
        "structured_text_region",
        "visual_structure_grid",
        "embedded_image_ocr",
    }:
        return True

    if table_family == "rowspan_grouped_table" and _semantic_grid_has_layered_header_signature(block.get("semantic_grid")):
        if source != "word_clustering" or any(
            str(span.get("source") or "").strip().lower() != "sparse_body_rowspan_projection"
            for span in spans
        ):
            return True

    return False


def _table_has_source_backed_presentation_spans(block: dict[str, Any]) -> bool:
    return any(
        isinstance(span, dict)
        and str(span.get("source") or "").strip() == "source_body_row_group_presentation_projection"
        for span in block.get("presentation_spans", []) or []
    )


def _append_markdown_table(
    lines: list[str],
    block: dict[str, Any],
    *,
    table_export_mode: str = "markdown",
    previous_visible_title: str = "",
    suppress_note_norms: set[str] | None = None,
) -> None:
    if table_export_mode == "evidence_markdown":
        grid = _normalize_markdown_table_evidence_grid(block)
    else:
        grid = _normalize_markdown_table_grid(block)
    if not grid:
        return
    if isinstance(block.get("cell_spans"), list) and any(
        isinstance(span, dict) and str(span.get("role") or "") == "header"
        for span in block.get("cell_spans", []) or []
    ):
        canonical_header_grid = _project_markdown_canonical_header_grid(block, grid)
        if canonical_header_grid:
            grid = canonical_header_grid

    internal_title = _markdown_table_internal_title(block)
    external_title = _markdown_table_external_title(block, internal_title)
    if (
        external_title
        and _markdown_object_title_redundant_with_previous_heading(block, previous_visible_title)
        and not (
            _markdown_table_is_continuation(block)
            and _markdown_should_preserve_continuation_title(
                block,
                external_title,
                table_export_mode,
                previous_visible_title=previous_visible_title,
            )
        )
    ):
        external_title = ""
    if external_title and _markdown_continuation_table_title_redundant_with_previous_visible_title(
        block,
        previous_visible_title,
    ):
        if not _markdown_should_preserve_continuation_title(
            block,
            external_title,
            table_export_mode,
            previous_visible_title=previous_visible_title,
        ):
            external_title = ""
    if external_title:
        lines.append(f"**{external_title}**")
        lines.append("")
    if internal_title and not _markdown_table_titles_are_redundant(external_title, internal_title):
        lines.append(f"**{internal_title}**")
        lines.append("")

    _append_markdown_table_study_context(lines, block, external_title or internal_title)

    grid = _drop_markdown_table_embedded_title_rows(grid, block, external_title or internal_title)
    rendered_grid_text_norms = _markdown_grid_cell_text_norms(grid)

    if table_export_mode in {"semantic_html", "auto_semantic"}:
        keyed_list_html = _build_keyed_long_list_html_table(block)
        if keyed_list_html:
            lines.append(keyed_list_html)
            lines.append("")
            _append_markdown_float_owned_text(
                lines,
                block,
                roles={"note", "legend"},
                suppress_text_norms=suppress_note_norms,
                table_grid_text_norms=rendered_grid_text_norms,
            )
            return

    if table_export_mode == "evidence_markdown" and (
        _table_has_source_backed_presentation_spans(block)
        or _markdown_table_requires_evidence_semantic_html(block)
    ):
        html_table = _build_semantic_html_table(
            _markdown_table_with_projected_presentation_surface(block, grid)
        )
        if html_table:
            lines.append(html_table)
            lines.append("")
            _append_markdown_float_owned_text(
                lines,
                block,
                roles={"note", "legend"},
                suppress_text_norms=suppress_note_norms,
                table_grid_text_norms=rendered_grid_text_norms,
            )
            return

    if table_export_mode == "semantic_html" or (
        table_export_mode == "auto_semantic" and _should_auto_render_table_as_semantic_html(block)
    ):
        html_block = (
            _markdown_table_with_projected_presentation_surface(block, grid)
            if _table_has_source_backed_presentation_spans(block)
            else block
        )
        html_table = _build_semantic_html_table(html_block)
        if html_table:
            lines.append(html_table)
            lines.append("")
            _append_markdown_float_owned_text(
                lines,
                block,
                roles={"note", "legend"},
                suppress_text_norms=suppress_note_norms,
                table_grid_text_norms=rendered_grid_text_norms,
            )
            return

    merged_rows = (
        {}
        if _markdown_raw_grid_for_preserved_trailing_structural_rows(block)
        else _merged_rows_by_projected_grid_row(block, grid)
    )
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
        _append_markdown_float_owned_text(
            lines,
            block,
            roles={"note", "legend"},
            suppress_text_norms=suppress_note_norms,
            table_grid_text_norms=rendered_grid_text_norms,
        )
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
    _append_markdown_float_owned_text(
        lines,
        block,
        roles={"note", "legend"},
        suppress_text_norms=suppress_note_norms,
        table_grid_text_norms=rendered_grid_text_norms,
    )


def _markdown_table_with_projected_presentation_surface(
    block: dict[str, Any],
    grid: list[list[str]],
) -> dict[str, Any]:
    projected = dict(block)
    projected["semantic_grid"] = [list(row) for row in grid]
    projected["semantic_display_grid"] = [list(row) for row in grid]
    projected["presentation_spans"] = _markdown_projected_presentation_spans(block, grid)
    if isinstance(block.get("cell_spans"), list):
        projected["cell_spans"] = _markdown_projected_canonical_cell_spans(block, grid)
    return projected


def _markdown_table_requires_evidence_semantic_html(block: dict[str, Any]) -> bool:
    if isinstance(block.get("cell_spans"), list) and any(
        isinstance(span, dict)
        and (
            int(span.get("rowspan", 1) or 1) > 1
            or int(span.get("colspan", 1) or 1) > 1
        )
        for span in block.get("cell_spans", []) or []
    ):
        return True
    projection = block.get("semantic_projection_v2")
    if not isinstance(projection, dict):
        return False
    return (
        isinstance(projection.get("genotoxicity_assay_matrix_projection"), dict)
        and isinstance(projection.get("multiline_schema_header_projection"), dict)
    )


def _markdown_table_visible_title(block: dict[str, Any]) -> str:
    internal_title = _markdown_table_internal_title(block)
    external_title = _markdown_table_external_title(block, internal_title)
    if external_title:
        return external_title
    if internal_title:
        return internal_title
    return ""


def _markdown_table_chain_previous_visible_title(
    block: dict[str, Any],
    visible_title_by_table_id: dict[str, str],
) -> str:
    parent_ids: list[str] = []
    parent_id = str(block.get("continued_from_table_id") or "").strip()
    if parent_id:
        parent_ids.append(parent_id)
    parent_ids.extend(_as_string_list(block.get("continued_from")))
    for parent_id in parent_ids:
        title = str(visible_title_by_table_id.get(parent_id) or "").strip()
        if title:
            return title
    return ""


def _markdown_structure_template_chain_previous_visible_title(
    block: dict[str, Any],
    visible_title_by_structure_template_id: dict[str, str],
) -> str:
    parent_id = str(block.get("continued_from_structure_template_id") or "").strip()
    if not parent_id:
        return ""
    return str(visible_title_by_structure_template_id.get(parent_id) or "").strip()


def _append_markdown_table_study_context(
    lines: list[str],
    block: dict[str, Any],
    rendered_title: str,
) -> None:
    context_blocks = [
        item
        for item in block.get("study_context_blocks", []) or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    if not context_blocks:
        return
    title_norm = _markdown_compact_table_text(rendered_title)
    rendered_norms: set[str] = set()
    rows: list[str] = []
    for item in context_blocks:
        text = _markdown_normalize_plain_text(str(item.get("text") or ""))
        if not text:
            continue
        text_norm = _markdown_compact_table_text(text)
        if not text_norm or text_norm in rendered_norms:
            continue
        if title_norm and (text_norm == title_norm or text_norm in title_norm):
            continue
        rendered_norms.add(text_norm)
        rows.append(text)
    if not rows:
        return
    for row in rows:
        lines.append(f"- {_markdown_escape_inline_text(row)}")
    lines.append("")


def _markdown_normalize_plain_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


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


def _should_merge_evidence_markdown_table_chain(
    table: dict[str, Any],
    table_by_id: dict[str, dict[str, Any]],
) -> bool:
    chain = _build_table_chain(table, table_by_id)
    if len(chain) <= 1:
        return False
    root_signature = _genotoxicity_markdown_chain_signature(chain[0])
    if root_signature is None:
        return False
    previous_id = str(chain[0].get("table_id") or "").strip()
    for continuation in chain[1:]:
        continuation_id = str(continuation.get("table_id") or "").strip()
        if not continuation_id:
            return False
        parent_id = str(
            continuation.get("continued_from_table_id")
            or continuation.get("continued_from")
            or ""
        ).strip()
        if parent_id and parent_id != previous_id:
            return False
        if str(continuation.get("title") or continuation.get("caption_text") or "").strip():
            return False
        if _genotoxicity_markdown_chain_signature(continuation) != root_signature:
            return False
        projection = continuation.get("semantic_projection_v2") or {}
        payload = projection.get("genotoxicity_assay_matrix_projection") if isinstance(projection, dict) else None
        if not isinstance(payload, dict) or not bool(payload.get("continuation_schema_inherited")):
            return False
        previous_id = continuation_id
    return True


def _genotoxicity_markdown_chain_signature(table: dict[str, Any]) -> tuple[str, str] | None:
    projection = table.get("semantic_projection_v2") or {}
    payload = projection.get("genotoxicity_assay_matrix_projection") if isinstance(projection, dict) else None
    if not isinstance(payload, dict):
        return None
    semantic_profile = str(payload.get("semantic_profile") or "").strip()
    if semantic_profile != "genotoxicity_assay_matrix":
        return None
    assay_kind = str(payload.get("assay_kind") or "").strip()
    if not assay_kind:
        return None
    return semantic_profile, assay_kind


def _merge_continued_table_chain(
    chain: list[dict[str, Any]],
    *,
    table_export_mode: str = "markdown",
) -> dict[str, Any]:
    if not chain:
        return {}
    merged = dict(chain[0])
    merged_grid: list[list[str]] = []
    merged_row_provenance: list[dict[str, Any]] = []
    merged_rows: list[dict[str, Any]] = []
    merged_presentation_spans: list[dict[str, Any]] = []
    merged_cell_spans: list[dict[str, Any]] = []
    logical_segments: list[tuple[int, int, dict[str, Any]]] = []
    header: list[str] | None = None
    equivalent_headers: list[list[str]] = []
    for index, table in enumerate(chain):
        grid = _normalize_markdown_table_grid(table)
        if not grid:
            continue
        local_row_provenance = _markdown_projected_grid_row_provenance(table, grid)
        local_merged_rows = _merged_rows_by_projected_grid_row(table, grid)
        local_presentation_spans = _markdown_projected_presentation_spans(table, grid)
        local_cell_spans = _markdown_projected_canonical_cell_spans(table, grid)
        if index == 0:
            row_offset = len(merged_grid)
            merged_grid.extend(grid)
            logical_segments.append((row_offset, row_offset + len(grid), table))
            merged_row_provenance.extend(dict(item) for item in local_row_provenance)
            for span in local_presentation_spans:
                span_copy = dict(span)
                span_copy["row"] = row_offset + int(span["row"])
                merged_presentation_spans.append(span_copy)
            for span in local_cell_spans:
                span_copy = dict(span)
                span_copy["row"] = row_offset + int(span["row"])
                merged_cell_spans.append(span_copy)
            for local_row, meta in local_merged_rows.items():
                meta_copy = dict(meta)
                meta_copy["row"] = row_offset + local_row
                merged_rows.append(meta_copy)
            header = grid[0] if grid else None
            canonical_header_depth = _markdown_canonical_header_depth(table)
            equivalent_headers = (
                [list(row) for row in grid[:canonical_header_depth]]
                if canonical_header_depth > 1
                else _markdown_equivalent_continued_table_header_rows(grid)
            )
            continue
        continuation_rows = list(grid)
        continuation_rows, skipped_leading_rows = _markdown_consume_continued_table_header_prefix(
            continuation_rows,
            equivalent_headers or ([header] if header is not None else []),
        )
        before_fragment_merge_count = len(continuation_rows)
        continuation_rows = _merge_leading_continuation_fragments(merged_grid, continuation_rows)
        merged_fragment_count = before_fragment_merge_count - len(continuation_rows)
        skipped_leading_rows += merged_fragment_count
        continuation_title = _markdown_continued_table_chain_boundary_title(
            chain[0],
            table,
            table_export_mode=table_export_mode,
        )
        if not continuation_title:
            _markdown_extend_trailing_presentation_spans_over_continuation_prefix(
                merged_presentation_spans,
                merged_row_count=len(merged_grid),
                continuation=table,
                continuation_rows=continuation_rows,
            )
            _markdown_extend_trailing_presentation_spans_over_continuation_prefix(
                merged_cell_spans,
                merged_row_count=len(merged_grid),
                continuation=table,
                continuation_rows=continuation_rows,
            )
        if continuation_title:
            merged_grid.append([continuation_title])
            merged_row_provenance.append(
                {
                    "source_row_refs": [],
                    "derivation": "continued_table_chain_boundary_title",
                }
            )
            merged_rows.append(
                {
                    "row": len(merged_grid),
                    "text": continuation_title,
                    "source": "continued_table_chain_boundary_title",
                    "source_table_id": str(table.get("table_id") or "").strip(),
                    "source_page": table.get("page"),
                }
            )
        row_offset = len(merged_grid)
        for span in local_presentation_spans:
            local_row = int(span["row"])
            if local_row < skipped_leading_rows:
                continue
            span_copy = dict(span)
            span_copy["row"] = row_offset + local_row - skipped_leading_rows
            merged_presentation_spans.append(span_copy)
        for span in local_cell_spans:
            local_row = int(span["row"])
            if local_row < skipped_leading_rows:
                continue
            span_copy = dict(span)
            span_copy["row"] = row_offset + local_row - skipped_leading_rows
            merged_cell_spans.append(span_copy)
        for local_row, meta in local_merged_rows.items():
            if local_row <= skipped_leading_rows:
                continue
            meta_copy = dict(meta)
            meta_copy["row"] = row_offset + local_row - skipped_leading_rows
            merged_rows.append(meta_copy)
        merged_grid.extend(continuation_rows)
        logical_segments.append((row_offset, row_offset + len(continuation_rows), table))
        continuation_lineage = local_row_provenance[skipped_leading_rows:]
        merged_row_provenance.extend(
            dict(item)
            for item in continuation_lineage[: len(continuation_rows)]
        )
    merged["display_grid"] = merged_grid
    merged["semantic_grid"] = [list(row) for row in merged_grid]
    merged["semantic_display_grid"] = [list(row) for row in merged_grid]
    merged["semantic_row_provenance"] = merged_row_provenance
    merged["row_count"] = len(merged_grid)
    merged["logical_row_count"] = len(merged_grid)
    if merged_rows:
        merged["merged_rows"] = merged_rows
    else:
        merged.pop("merged_rows", None)
    if merged_presentation_spans:
        merged["presentation_spans"] = merged_presentation_spans
    else:
        merged.pop("presentation_spans", None)
    if merged_cell_spans:
        root_id = str(chain[0].get("table_id") or chain[0].get("block_id") or "table").strip() or "table"
        for span_index, span in enumerate(merged_cell_spans, start=1):
            span["coordinate_space"] = "logical_table_chain"
            span["span_id"] = f"{root_id}:logical_cell_span:{span_index}"
            try:
                span_start = int(span.get("row", -1))
                span_end = span_start + max(1, int(span.get("rowspan", 1) or 1))
            except (TypeError, ValueError):
                continue
            covered_tables = [
                table
                for segment_start, segment_end, table in logical_segments
                if span_start < segment_end and span_end > segment_start
            ]
            source_pages = {
                int(page)
                for table in covered_tables
                for page in [table.get("page")]
                if isinstance(page, (int, float)) and int(page) > 0
            }
            source_table_ids = [
                str(table.get("table_id") or table.get("block_id") or "").strip()
                for table in covered_tables
                if str(table.get("table_id") or table.get("block_id") or "").strip()
            ]
            span["source_pages"] = sorted(source_pages)
            span["source_table_ids"] = list(dict.fromkeys(source_table_ids))
        merged["cell_spans"] = merged_cell_spans
    else:
        merged.pop("cell_spans", None)
    _merge_continued_table_chain_semantic_attachments(merged, chain)
    return merged


def _markdown_projected_canonical_cell_spans(
    block: dict[str, Any],
    grid: list[list[str]],
) -> list[dict[str, Any]]:
    canonical = [
        dict(span)
        for span in block.get("cell_spans", []) or []
        if isinstance(span, dict)
    ]
    if not canonical:
        return []
    if all(str(span.get("coordinate_space") or "") == "logical_table_chain" for span in canonical):
        return canonical
    body_spans = [span for span in canonical if str(span.get("role") or "") == "body"]
    projected_body: list[dict[str, Any]] = []
    if body_spans:
        compatibility_block = dict(block)
        compatibility_block["presentation_spans"] = body_spans
        projected_body = _markdown_projected_presentation_spans(compatibility_block, grid)
    header_spans = [span for span in canonical if str(span.get("role") or "") == "header"]
    return [*header_spans, *projected_body]


def _markdown_canonical_header_depth(block: dict[str, Any]) -> int:
    depths: list[int] = []
    for span in block.get("cell_spans", []) or []:
        if not isinstance(span, dict) or str(span.get("role") or "") != "header":
            continue
        try:
            row = int(span.get("row", 0) or 0)
            rowspan = max(1, int(span.get("rowspan", 1) or 1))
            colspan = max(1, int(span.get("colspan", 1) or 1))
        except (TypeError, ValueError):
            continue
        depths.append(row + max(rowspan, 2 if colspan > 1 else 1))
    return max(depths, default=1)


def _markdown_projected_presentation_spans(
    block: dict[str, Any],
    grid: list[list[str]],
) -> list[dict[str, Any]]:
    origins = _markdown_projected_grid_source_row_numbers(block, grid)
    projected_rows_by_source: dict[int, int] = {
        source_row: projected_row
        for projected_row, source_row in enumerate(origins)
        if source_row is not None
    }
    semantic_lineage = block.get("semantic_row_provenance")
    semantic_lineage = semantic_lineage if isinstance(semantic_lineage, list) else []
    projected_lineage = _markdown_projected_grid_row_provenance(block, grid)
    projected_rows_by_ref: dict[str, list[int]] = {}
    for projected_row, lineage in enumerate(projected_lineage):
        for row_ref in lineage.get("source_row_refs", []) or []:
            source_ref = str(row_ref or "").strip()
            if source_ref:
                projected_rows_by_ref.setdefault(source_ref, []).append(projected_row)
    projected: list[dict[str, Any]] = []
    for span in block.get("presentation_spans", []) or []:
        if not isinstance(span, dict):
            continue
        try:
            source_row = int(span.get("row", -1))
            rowspan = max(1, int(span.get("rowspan", 1) or 1))
            col = int(span.get("col", -1))
            colspan = max(1, int(span.get("colspan", 1) or 1))
        except (TypeError, ValueError):
            continue
        semantic_row_indices = list(range(source_row, source_row + rowspan))
        projected_rows: list[int | None] = []
        if semantic_lineage and max(semantic_row_indices, default=-1) < len(semantic_lineage):
            for semantic_row_index in semantic_row_indices:
                lineage = semantic_lineage[semantic_row_index]
                row_refs = list(lineage.get("source_row_refs", []) or []) if isinstance(lineage, dict) else []
                candidates = sorted(
                    {
                        projected_row
                        for row_ref in row_refs
                        for projected_row in projected_rows_by_ref.get(str(row_ref or "").strip(), [])
                    }
                )
                projected_rows.append(candidates[0] if len(candidates) == 1 else None)
        if not projected_rows or any(row is None for row in projected_rows):
            source_row_numbers = [row + 1 for row in semantic_row_indices]
            projected_rows = [
                projected_rows_by_source.get(row_number)
                for row_number in source_row_numbers
            ]
        if any(row is None for row in projected_rows):
            continue
        concrete_rows = [int(row) for row in projected_rows if row is not None]
        if concrete_rows != list(range(concrete_rows[0], concrete_rows[0] + rowspan)):
            continue
        projected.append(
            {
                **span,
                "row": concrete_rows[0],
                "col": col,
                "rowspan": rowspan,
                "colspan": colspan,
            }
        )
    return projected


def _markdown_extend_trailing_presentation_spans_over_continuation_prefix(
    spans: list[dict[str, Any]],
    *,
    merged_row_count: int,
    continuation: dict[str, Any],
    continuation_rows: list[list[str]],
) -> None:
    if not continuation_rows or not _markdown_table_has_inherited_semantic_schema(continuation):
        return
    for span in spans:
        try:
            row = int(span.get("row", -1))
            col = int(span.get("col", -1))
            rowspan = max(1, int(span.get("rowspan", 1) or 1))
        except (TypeError, ValueError):
            continue
        text = _markdown_compact_table_text(span.get("text"))
        if row < 0 or col < 0 or row + rowspan != merged_row_count or not text:
            continue
        if not _markdown_continuation_source_omits_group_text(continuation, text):
            continue
        prefix_count = 0
        for continuation_row in continuation_rows:
            value = continuation_row[col] if col < len(continuation_row) else ""
            if _markdown_compact_table_text(value) != text:
                break
            prefix_count += 1
        if prefix_count:
            span["rowspan"] = rowspan + prefix_count


def _markdown_table_has_inherited_semantic_schema(block: dict[str, Any]) -> bool:
    projection = block.get("semantic_projection_v2") or {}
    if not isinstance(projection, dict):
        return False
    return any(
        isinstance(payload, dict)
        and bool(payload.get("continuation_schema_inherited") or payload.get("schema_inherited"))
        for payload in projection.values()
    )


def _markdown_continuation_source_omits_group_text(
    continuation: dict[str, Any],
    compact_group_text: str,
) -> bool:
    source_grid = continuation.get("display_grid") or continuation.get("raw_grid") or []
    if not isinstance(source_grid, list) or not source_grid:
        return False
    return all(
        _markdown_compact_table_text(cell) != compact_group_text
        for row in source_grid
        if isinstance(row, list)
        for cell in row
    )


def _markdown_equivalent_continued_table_header_rows(grid: list[list[str]]) -> list[list[str]]:
    if not grid:
        return []
    headers = [list(grid[0])]
    if len(grid) >= 2 and _markdown_grid_rows_are_multilevel_header_pair(grid[0], grid[1]):
        headers.append(list(grid[1]))
    return headers


def _markdown_consume_continued_table_header_prefix(
    rows: list[list[str]],
    equivalent_headers: list[list[str]],
) -> tuple[list[list[str]], int]:
    remaining_headers = [list(header) for header in equivalent_headers if header]
    remaining_rows = list(rows)
    skipped = 0
    while remaining_rows and remaining_headers:
        matched_index = next(
            (
                index
                for index, candidate in enumerate(remaining_headers)
                if _markdown_header_rows_equivalent(remaining_rows[0], candidate)
            ),
            None,
        )
        if matched_index is None:
            break
        remaining_rows = remaining_rows[1:]
        remaining_headers.pop(matched_index)
        skipped += 1
    return remaining_rows, skipped


def _markdown_grid_rows_are_multilevel_header_pair(top_row: list[Any], leaf_row: list[Any]) -> bool:
    if not top_row or not leaf_row or len(top_row) != len(leaf_row):
        return False
    top_norms = [_markdown_compact_table_text(cell) for cell in top_row]
    leaf_norms = [_markdown_compact_table_text(cell) for cell in leaf_row]
    if top_norms == leaf_norms:
        return False
    repeated_top = len([text for text in top_norms if text and top_norms.count(text) >= 2])
    if repeated_top < 2:
        return False
    differing_leaf_cells = sum(1 for top, leaf in zip(top_norms, leaf_norms) if top and leaf and top != leaf)
    return differing_leaf_cells >= 2


def _merge_continued_table_chain_semantic_attachments(
    merged: dict[str, Any],
    chain: list[dict[str, Any]],
) -> None:
    if len(chain) <= 1:
        return
    for key in ("note_blocks", "content_segments", "caption_blocks"):
        segments: list[Any] = []
        for table in chain:
            table_id = str(table.get("table_id") or table.get("block_id") or "").strip()
            for segment in table.get(key, []) or []:
                if not isinstance(segment, dict):
                    segments.append(segment)
                    continue
                segment_copy = copy.deepcopy(segment)
                if table_id and not str(segment_copy.get("source_table_id") or "").strip():
                    segment_copy["source_table_id"] = table_id
                if table.get("page") is not None and segment_copy.get("source_page") is None:
                    segment_copy["source_page"] = table.get("page")
                segments.append(segment_copy)
        if segments:
            merged[key] = _dedupe_markdown_table_note_segments(segments)
        else:
            merged.pop(key, None)

    for key in ("header_note_refs", "cell_note_refs"):
        refs: list[Any] = []
        seen: set[str] = set()
        for table in chain:
            table_id = str(table.get("table_id") or table.get("block_id") or "").strip()
            for ref in table.get(key, []) or []:
                if not isinstance(ref, dict):
                    refs.append(ref)
                    continue
                ref_copy = copy.deepcopy(ref)
                if table_id and not str(ref_copy.get("source_table_id") or "").strip():
                    ref_copy["source_table_id"] = table_id
                if table.get("page") is not None and ref_copy.get("source_page") is None:
                    ref_copy["source_page"] = table.get("page")
                ref_key = json.dumps(ref_copy, ensure_ascii=False, sort_keys=True, default=str)
                if ref_key in seen:
                    continue
                seen.add(ref_key)
                refs.append(ref_copy)
        if refs:
            merged[key] = refs
        else:
            merged.pop(key, None)


def _markdown_continued_table_chain_boundary_title(
    root: dict[str, Any],
    continuation: dict[str, Any],
    *,
    table_export_mode: str = "markdown",
) -> str:
    title = str(continuation.get("title") or continuation.get("caption_text") or "").strip()
    if not title:
        return ""
    root_title = str(root.get("title") or root.get("caption_text") or "").strip()
    if not _markdown_should_preserve_continuation_title(continuation, title, table_export_mode):
        return ""
    if not _markdown_table_titles_are_redundant(root_title, title):
        return title
    if int(continuation.get("page", 0) or 0) != int(root.get("page", 0) or 0):
        return title
    return ""


def _markdown_should_preserve_continuation_title(
    block: dict[str, Any],
    title: str,
    table_export_mode: str = "markdown",
    *,
    previous_visible_title: str = "",
) -> bool:
    if str(table_export_mode or "").strip().lower() != "evidence_markdown":
        return False
    if not _markdown_continued_table_title_has_review_locator(title):
        return False
    if _markdown_continued_table_title_is_generic_continued_label(title):
        return False
    if (
        previous_visible_title
        and _markdown_table_titles_are_redundant(previous_visible_title, title)
        and not _markdown_continued_table_title_has_subtable_locator(title)
    ):
        return False
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    if isinstance(projection.get("dose_response_result_panel_projection"), dict):
        return True
    family = str(block.get("table_family") or "").strip()
    if family in {"comparison_matrix", "dose_response_result_panel"}:
        return True
    return False


def _markdown_continued_table_title_has_subtable_locator(title: str) -> bool:
    return bool(re.search(r"\b\d+(?:\.\d+){2,}[A-Za-z]\b", str(title or "")))


def _markdown_continued_table_title_is_generic_continued_label(title: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(title or "").strip())
    if not cleaned:
        return False
    return bool(re.fullmatch(r"(?:table|tab\.?|表)\s*[A-Za-z0-9一二三四五六七八九十]*\s*[:：.\-]?\s*(?:continued|续表|续)\.?", cleaned, re.IGNORECASE))


def _markdown_continued_table_title_has_review_locator(title: str) -> bool:
    cleaned = str(title or "").strip()
    if not cleaned:
        return False
    if re.search(r"(?:续|\(续\)|（续）|continued)", cleaned, re.IGNORECASE):
        return True
    if re.search(r"(?:试验编号|报告编号|study\s*(?:no\.?|number)|report\s*(?:no\.?|number))", cleaned, re.IGNORECASE):
        return True
    if re.search(r"\b\d+(?:\.\d+){2,}[A-Za-z]?\b", cleaned):
        return True
    if re.search(r"示例\s*#?\s*\d+", cleaned, re.IGNORECASE):
        return True
    return False


def _markdown_header_rows_equivalent(left: list[Any], right: list[Any]) -> bool:
    if left == right:
        return True
    return _markdown_row_compact_signature(left) == _markdown_row_compact_signature(right)


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
    toc_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    block_type = str(block.get("block_type") or "").strip().lower()
    if block_type == "table":
        table_id = str(block.get("table_id") or block.get("block_id") or "").strip()
        if table_id and table_id in table_by_id:
            enriched = {**block, **table_by_id[table_id]}
            if block.get("block_id") and not enriched.get("block_id"):
                enriched["block_id"] = block.get("block_id")
            if block.get("section_context") and not enriched.get("section_context"):
                enriched["section_context"] = block.get("section_context")
            return enriched
    if block_type == "image":
        image_id = str(block.get("image_id") or block.get("block_id") or "").strip()
        if image_id and image_id in image_by_id:
            return {**image_by_id[image_id], **block}
    if block_type == "toc":
        toc_id = str(block.get("toc_id") or block.get("block_id") or "").strip()
        if toc_by_id and toc_id and toc_id in toc_by_id:
            return {**toc_by_id[toc_id], **block}
    return block


def _append_markdown_image(
    lines: list[str],
    block: dict[str, Any],
    *,
    embed_images: bool = True,
    suppress_embedded_text: bool = False,
    previous_visible_title: str = "",
) -> None:
    source_path = block.get("_source_path")
    image_id = str(block.get("image_id") or block.get("block_id") or "").strip()
    owned_caption_blocks = [
        item for item in block.get("caption_blocks", []) or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    caption = str(
        block.get("caption_text")
        or block.get("title")
        or block.get("figure_ref")
        or image_id
        or "image"
    ).strip()
    suppress_visible_title = _markdown_object_title_redundant_with_previous_heading(
        block,
        previous_visible_title,
    )
    suppressed_text_norms: set[str] = set()
    if suppress_visible_title:
        title_norm = _markdown_compact_table_text(caption)
        if title_norm:
            suppressed_text_norms.add(title_norm)
    visible_title = "" if suppress_visible_title else _markdown_image_visible_title(block)
    if visible_title:
        visible_title_norm = _markdown_compact_table_text(visible_title)
        if visible_title_norm:
            suppressed_text_norms.add(visible_title_norm)
    alt_source = str(block.get("figure_ref") or image_id or "image").strip() if owned_caption_blocks or suppress_visible_title else caption
    alt_text = alt_source.replace("[", "(").replace("]", ")")
    image_markdown = None
    if embed_images and isinstance(source_path, Path) and source_path.exists():
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
    roles = {"caption", "legend", "note", "nearby_context"}
    if not suppress_embedded_text:
        roles.update({"embedded_text", "embedded_code"})
    if visible_title:
        lines.append(f"**{visible_title}**")
        lines.append("")
    _append_markdown_float_owned_text(
        lines,
        block,
        roles=roles,
        relation_filter={"above"},
        suppress_text_norms=suppressed_text_norms,
    )
    lines.append(image_markdown)
    lines.append("")
    _append_markdown_float_owned_text(
        lines,
        block,
        roles=roles,
        relation_filter=None,
        exclude_relation={"above"},
        suppress_text_norms=suppressed_text_norms,
    )


def _markdown_image_visible_title(block: dict[str, Any]) -> str:
    title = str(block.get("caption_text") or block.get("title") or "").strip()
    if not title:
        return ""
    composite = block.get("composite_object")
    composite = composite if isinstance(composite, dict) else {}
    if (
        composite.get("title_policy") == "owned_object_title"
        or composite.get("visible_title_owner") == "figure"
    ):
        return title
    for segment in block.get("content_segments", []) or []:
        if not isinstance(segment, dict):
            continue
        if (
            str(segment.get("role") or "").strip() == "caption"
            and str(segment.get("relation") or "").strip().lower() == "above"
        ):
            return title
    for segment in block.get("caption_blocks", []) or []:
        if not isinstance(segment, dict):
            continue
        if (
            str(segment.get("role") or "").strip() == "caption"
            and str(segment.get("relation") or "").strip().lower() == "above"
        ):
            return title
    return ""


def _append_markdown_float_owned_text(
    lines: list[str],
    block: dict[str, Any],
    *,
    roles: set[str],
    relation_filter: set[str] | None = None,
    exclude_relation: set[str] | None = None,
    suppress_text_norms: set[str] | None = None,
    table_grid_text_norms: set[str] | None = None,
) -> None:
    segments = _markdown_float_owned_text_segments(block)
    segments = _order_markdown_float_segments(block, segments)
    accepted_roles = set(roles)
    if "note" in accepted_roles:
        accepted_roles.add("table_note")
    paragraph_parts: list[str] = []
    seen: set[str] = set()
    if table_grid_text_norms is None:
        table_grid_text_norms = _markdown_table_grid_cell_text_norms(block)
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        role = str(segment.get("role") or "").strip()
        if role not in accepted_roles:
            continue
        if role == "nearby_context" and not _markdown_nearby_context_segment_is_float_title(block, segment):
            continue
        relation = str(segment.get("relation") or "").strip().lower()
        if relation_filter is not None and relation not in relation_filter:
            continue
        if exclude_relation is not None and relation in exclude_relation:
            continue
        text = _clean_markdown_float_owned_text(str(segment.get("text") or ""))
        if not text:
            continue
        norm = _markdown_compact_table_text(text)
        if not norm or norm in seen:
            continue
        if suppress_text_norms is not None and norm in suppress_text_norms:
            continue
        if norm in table_grid_text_norms and not _markdown_float_owned_segment_should_survive_table_grid_dedupe(segment):
            continue
        seen.add(norm)
        paragraph_parts.append(text)
    if not paragraph_parts:
        return
    if _markdown_float_owned_segments_should_render_as_lines(block, segments):
        for part in paragraph_parts:
            paragraph = _repair_markdown_float_owned_note_paragraph(part.strip())
            paragraph = _markdown_linkify_visible_urls(paragraph)
            if paragraph:
                lines.append(paragraph)
        lines.append("")
        return
    if _markdown_note_texts_form_marker_run(paragraph_parts):
        for part in _sort_markdown_marker_note_texts(paragraph_parts):
            paragraph = _repair_markdown_float_owned_note_paragraph(part.strip())
            paragraph = _markdown_linkify_visible_urls(paragraph)
            if paragraph:
                lines.append(paragraph)
        lines.append("")
        return
    paragraph = _repair_markdown_float_owned_note_paragraph(" ".join(paragraph_parts).strip())
    paragraph = _markdown_linkify_visible_urls(paragraph)
    if paragraph:
        lines.append(paragraph)
        lines.append("")


def _markdown_float_owned_segment_should_survive_table_grid_dedupe(segment: dict[str, Any]) -> bool:
    if str(segment.get("role") or "").strip() not in {"note", "table_note", "legend"}:
        return False
    source = str(segment.get("source") or "").strip()
    relation = str(segment.get("relation") or "").strip().lower()
    if source == "cross_page_result_matrix_statistical_note":
        return _markdown_result_matrix_statistical_note_text(str(segment.get("text") or ""))
    if source in {
        "dose_response_result_panel_note_row",
        "explicit_new_panel_boundary_note_reassignment",
        "dose_response_continuation_note_row",
        "dose_response_top_continuation_note",
        "result_matrix_statistical_note_after_table",
        "trailing_table_note_row",
    }:
        return True
    if relation.startswith("cross_page"):
        return True
    return False


def _markdown_float_owned_segments_should_render_as_lines(block: dict[str, Any], segments: list[Any]) -> bool:
    if str(block.get("block_type") or "").strip().lower() != "table":
        return False
    meaningful_segments = [
        segment
        for segment in segments
        if isinstance(segment, dict)
        and str(segment.get("role") or "").strip() in {"note", "table_note", "legend"}
        and str(segment.get("text") or "").strip()
    ]
    if len(meaningful_segments) < 2:
        return False
    line_groups: dict[str, list[dict[str, Any]]] = {}
    for segment in meaningful_segments:
        group_id = str(segment.get("note_group_id") or "").strip()
        if group_id and str(segment.get("presentation_mode") or "").strip() == "lines":
            line_groups.setdefault(group_id, []).append(segment)
    if any(len(group) >= 2 for group in line_groups.values()):
        return True
    physical_pages = [
        _markdown_float_owned_segment_physical_page(segment)
        for segment in meaningful_segments
        if _markdown_block_bbox(segment) is not None
        and _markdown_float_owned_segment_physical_page(segment) > 0
    ]
    if len(physical_pages) == len(meaningful_segments) and len(set(physical_pages)) >= 2:
        return True
    sources = {str(segment.get("source") or "").strip() for segment in meaningful_segments}
    if (
        "cross_page_result_matrix_statistical_note" in sources
        and "explicit_new_panel_boundary_note_reassignment" in sources
    ):
        return True
    if (
        "study_condition_result_matrix_note_row" in sources
        and "cross_page_result_matrix_statistical_note" in sources
    ):
        return True
    if sources == {"ind_study_result_matrix_word_recovery"} and all(
        _markdown_ind_study_result_matrix_note_row_segment(segment)
        for segment in meaningful_segments
    ):
        return True
    return False


def _markdown_ind_study_result_matrix_note_row_segment(segment: dict[str, Any]) -> bool:
    if _markdown_block_bbox(segment) is None:
        return False
    text = str(segment.get("text") or "").strip()
    if not text:
        return False
    if _markdown_definition_note_segment(segment):
        return True
    if _markdown_result_matrix_statistical_note_text(text):
        return True
    marker = _markdown_note_text_primary_marker(text)
    return bool(marker and re.fullmatch(r"[a-z]|\d{1,3}", marker))


def _markdown_definition_note_segment(segment: dict[str, Any]) -> bool:
    profile = segment.get("note_profile") if isinstance(segment.get("note_profile"), dict) else None
    if isinstance(profile, dict) and profile.get("profile_type") == "definition_note":
        return True
    return _markdown_definition_note_text(str(segment.get("text") or ""))


def _markdown_definition_note_text(text: str) -> bool:
    cleaned = _clean_markdown_float_owned_text(str(text or ""))
    if not cleaned:
        return False
    match = re.match(
        r"^\s*(?P<term>(?:[%A-Za-z][%A-Za-z0-9%/_+\-.]{0,24}|[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9%/_+\-.]{0,24}))\s*[=＝]\s*(?P<definition>.+?)\s*$",
        cleaned,
    )
    if not match:
        return False
    term = str(match.group("term") or "").strip()
    definition = str(match.group("definition") or "").strip()
    if not term or not definition:
        return False
    compact_term = re.sub(r"\s+", "", term)
    return len(compact_term) >= 2 or "%" in compact_term or bool(re.search(r"[\u4e00-\u9fff]", compact_term))


def _markdown_float_owned_text_segments(block: dict[str, Any]) -> list[Any]:
    content_segments = list(block.get("content_segments", []) or [])
    fallback_segments = list(block.get("note_blocks", []) or []) + list(block.get("caption_blocks", []) or [])
    if str(block.get("block_type") or "").strip().lower() == "table":
        return _dedupe_markdown_table_note_segments(
            [
                segment
                for segment in content_segments + fallback_segments
                if not _markdown_float_owned_segment_renders_in_main_flow(segment)
            ]
        )
    return content_segments or fallback_segments


def _markdown_float_owned_segment_renders_in_main_flow(segment: Any) -> bool:
    if not isinstance(segment, dict):
        return False
    return str(segment.get("float_render_policy") or segment.get("render_policy") or "").strip() == "main_flow_at_source"


def _markdown_deferred_main_flow_float_segments(page_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    page_source_ids = {
        str(block.get("block_id") or block.get("source_id") or "").strip()
        for block in page_blocks
        if str(block.get("block_id") or block.get("source_id") or "").strip()
    }
    segments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in page_blocks:
        if str(block.get("block_type") or "").strip().lower() not in {"table", "image"}:
            continue
        for key in ("content_segments", "note_blocks", "caption_blocks"):
            for segment in block.get(key, []) or []:
                if not _markdown_float_owned_segment_renders_in_main_flow(segment):
                    continue
                segment = dict(segment)
                source_ids = _markdown_segment_source_block_ids(segment)
                if source_ids and any(source_id in page_source_ids for source_id in source_ids):
                    continue
                segment_key = _markdown_deferred_float_segment_key(segment)
                if segment_key in seen:
                    continue
                seen.add(segment_key)
                segments.append(segment)
    return [
        segment
        for _, segment in sorted(
            enumerate(segments),
            key=lambda item: (
                (_markdown_block_bbox(item[1]) or (0.0, 0.0, 0.0, 0.0))[1],
                (_markdown_block_bbox(item[1]) or (0.0, 0.0, 0.0, 0.0))[0],
                item[0],
            ),
        )
    ]


def _markdown_segment_source_block_ids(segment: dict[str, Any]) -> set[str]:
    source_ids = {
        str(segment.get(key) or "").strip()
        for key in ("source_block_id", "block_id", "source_id")
        if str(segment.get(key) or "").strip()
    }
    source_ids.update(
        str(source_block_id or "").strip()
        for source_block_id in segment.get("source_block_ids", []) or []
        if str(source_block_id or "").strip()
    )
    return source_ids


def _markdown_deferred_float_segment_key(segment: dict[str, Any]) -> str:
    source_ids = sorted(_markdown_segment_source_block_ids(segment))
    norm = _markdown_compact_table_text(segment.get("text") or "")
    bbox = _markdown_block_bbox(segment)
    bbox_key = ",".join(f"{value:.2f}" for value in bbox) if bbox is not None else ""
    return "|".join([norm, ",".join(source_ids), bbox_key])


def _append_markdown_deferred_main_flow_float_segment(lines: list[str], segment: dict[str, Any]) -> bool:
    text = _clean_markdown_float_owned_text(str(segment.get("text") or ""))
    if not text:
        return False
    paragraph = _repair_markdown_float_owned_note_paragraph(text.strip())
    paragraph = _markdown_linkify_visible_urls(paragraph)
    if not paragraph:
        return False
    lines.append(paragraph)
    lines.append("")
    return True


def _dedupe_markdown_table_note_segments(segments: list[Any]) -> list[Any]:
    ordered = _dedupe_markdown_exact_segment_norms(segments)
    ordered = _drop_markdown_table_combined_marker_note_segments(ordered)
    keep: list[Any] = []
    for index, segment in enumerate(ordered):
        if _markdown_table_note_segment_covered_by_better_segment(segment, ordered, index):
            continue
        keep.append(segment)
    return keep


def _drop_markdown_table_combined_marker_note_segments(segments: list[Any]) -> list[Any]:
    single_markers: set[str] = set()
    marker_sets_by_index: dict[int, set[str]] = {}
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        markers = _markdown_note_text_markers(str(segment.get("text") or ""))
        if markers:
            marker_sets_by_index[index] = markers
        if len(markers) == 1:
            single_markers.update(markers)
    if not single_markers:
        return segments
    keep: list[Any] = []
    for index, segment in enumerate(segments):
        markers = marker_sets_by_index.get(index, set())
        if len(markers) >= 2 and markers.issubset(single_markers):
            continue
        keep.append(segment)
    return keep


def _markdown_note_texts_form_marker_run(texts: list[str]) -> bool:
    markers = [_markdown_note_text_primary_marker(text) for text in texts]
    markers = [marker for marker in markers if marker]
    if len(markers) < 2 or len(markers) != len(texts):
        return False
    if len(set(markers)) != len(markers):
        return False
    if all(re.fullmatch(r"[a-z]", marker) for marker in markers):
        return True
    return True


def _sort_markdown_marker_note_texts(texts: list[str]) -> list[str]:
    def key(text: str) -> tuple[int, str]:
        marker = _markdown_note_text_primary_marker(text)
        if re.fullmatch(r"[a-z]", marker):
            return (ord(marker) - ord("a"), marker)
        if marker.isdigit():
            return (1000 + int(marker), marker)
        return (2000, marker)

    return [text for _, text in sorted(enumerate(texts), key=lambda item: (key(item[1]), item[0]))]


def _markdown_note_text_markers(text: str) -> set[str]:
    markers: set[str] = set()
    for match in re.finditer(r"(?:^|\s)([A-Za-z]|\d+|[*#†‡§])\s*[-–—:：]", str(text or "")):
        markers.add(match.group(1).lower())
    return markers


def _markdown_note_text_primary_marker(text: str) -> str:
    match = re.match(r"^\s*([A-Za-z]|\d+|[*#†‡§])\s*[-–—:：]", str(text or ""))
    if not match:
        return ""
    return match.group(1).lower()


def _dedupe_markdown_exact_segment_norms(segments: list[Any]) -> list[Any]:
    keep: list[Any] = []
    seen: set[str] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            keep.append(segment)
            continue
        norm = _markdown_compact_table_text(segment.get("text") or "")
        source_key = (
            str(segment.get("source_block_id") or "").strip(),
            str(segment.get("source") or "").strip(),
            str(segment.get("role") or "").strip(),
        )
        key = f"{norm}|{source_key}"
        if key in seen:
            continue
        seen.add(key)
        keep.append(segment)
    return keep


def _markdown_table_note_segment_covered_by_better_segment(
    segment: Any,
    segments: list[Any],
    index: int,
) -> bool:
    if not isinstance(segment, dict):
        return False
    role = str(segment.get("role") or "").strip()
    if role not in {"note", "table_note"}:
        return False
    norm = _markdown_compact_table_text(segment.get("text") or "")
    if len(norm) < 4:
        return False
    marker = _markdown_note_segment_marker(segment)
    for other_index, other in enumerate(segments):
        if other_index == index or not isinstance(other, dict):
            continue
        other_role = str(other.get("role") or "").strip()
        if other_role not in {"note", "table_note"}:
            continue
        other_norm = _markdown_compact_table_text(other.get("text") or "")
        if norm == other_norm:
            if _markdown_table_note_label_text(str(segment.get("text") or "")):
                segment_has_bbox = _markdown_block_bbox(segment) is not None
                other_has_bbox = _markdown_block_bbox(other) is not None
                if segment_has_bbox != other_has_bbox:
                    return not segment_has_bbox and other_has_bbox
            return _markdown_table_note_segment_quality(other) > _markdown_table_note_segment_quality(segment)
        if len(other_norm) > len(norm) and norm in other_norm:
            other_marker = _markdown_note_segment_marker(other)
            if marker and other_marker and marker != other_marker:
                continue
            return True
    return False


def _markdown_table_note_label_text(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:附加信息|补充信息|备注|注释|说明|Note|Notes)\s*[:：]?",
            re.sub(r"\s+", " ", str(text or "")).strip(),
            re.IGNORECASE,
        )
    )


def _markdown_note_segment_marker(segment: dict[str, Any]) -> str:
    marker = str(segment.get("marker") or "").strip()
    if marker:
        return marker.lower()
    text = str(segment.get("text") or "").strip()
    match = re.match(r"^\s*([A-Za-z*#†‡§]|\d+)\s*[-–—:：]", text)
    if match:
        return match.group(1).lower()
    return ""


def _markdown_table_note_segment_quality(segment: dict[str, Any]) -> int:
    score = 0
    if str(segment.get("source_block_id") or "").strip():
        score += 4
    if segment.get("source_block_ids"):
        score += 2
    if _markdown_block_bbox(segment) is not None:
        score += 2
    if str(segment.get("relation") or "").strip().lower() in {"below", "above"}:
        score += 2
    if str(segment.get("source") or "").strip():
        score -= 1
    return score


def _normalize_markdown_table_note_fields(document: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(document)
    preferred_note_owner_by_occurrence = _markdown_preferred_table_note_owner_by_occurrence(
        normalized.get("table_asts", []) or []
    )
    for table in normalized.get("table_asts", []) or []:
        if not isinstance(table, dict):
            continue
        note_blocks = list(table.get("note_blocks", []) or [])
        content_segments = list(table.get("content_segments", []) or [])
        if note_blocks:
            table["note_blocks"] = _filter_markdown_table_notes_for_preferred_owner(
                table,
                _dedupe_markdown_table_note_segments(note_blocks),
                preferred_note_owner_by_occurrence,
            )
        if content_segments:
            table["content_segments"] = _filter_markdown_table_notes_for_preferred_owner(
                table,
                _dedupe_markdown_table_note_segments(content_segments),
                preferred_note_owner_by_occurrence,
            )
    table_by_id = {
        str(table.get("table_id") or "").strip(): table
        for table in normalized.get("table_asts", []) or []
        if isinstance(table, dict) and str(table.get("table_id") or "").strip()
    }
    for page in (normalized.get("document_ast", {}) or {}).get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        updated_blocks: list[Any] = []
        for block in page.get("blocks", []) or []:
            if isinstance(block, dict) and str(block.get("block_type") or "").strip().lower() == "table":
                table_id = str(block.get("table_id") or block.get("block_id") or "").strip()
                table = table_by_id.get(table_id)
                if table is None:
                    updated_blocks.append(_normalize_markdown_table_block_note_fields(block))
                else:
                    updated_blocks.append(
                        _normalize_markdown_table_block_note_fields({**block, **table, "block_type": block.get("block_type")})
                    )
            else:
                updated_blocks.append(block)
        page["blocks"] = updated_blocks
    return normalized


def _markdown_preferred_table_note_owner_by_occurrence(tables: list[Any]) -> dict[tuple[str, int], str]:
    candidates: dict[tuple[str, int], list[tuple[int, str]]] = {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_id = str(table.get("table_id") or table.get("block_id") or "").strip()
        if not table_id:
            continue
        occurrence_keys = _markdown_table_renderable_note_occurrence_keys(table)
        if not occurrence_keys:
            continue
        score = _markdown_table_note_owner_score(table)
        for occurrence_key in occurrence_keys:
            candidates.setdefault(occurrence_key, []).append(
                (
                    score + _markdown_explicit_table_note_owner_score(
                        table,
                        occurrence_key[0],
                        occurrence_page=occurrence_key[1],
                    ),
                    table_id,
                )
            )
    preferred: dict[tuple[str, int], str] = {}
    for occurrence_key, scored_tables in candidates.items():
        if len(scored_tables) <= 1:
            continue
        scored_tables.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if scored_tables[0][0] > scored_tables[1][0]:
            preferred[occurrence_key] = scored_tables[0][1]
    return preferred


def _markdown_table_note_owner_score(table: dict[str, Any]) -> int:
    score = 0
    if table.get("cell_note_refs"):
        score += 20
    if table.get("header_note_refs"):
        score += 16
    composite = table.get("composite_object")
    if isinstance(composite, dict) and composite.get("note_anchor_refs"):
        score += 12
    projection = table.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    if isinstance(projection.get("study_condition_grouped_result_matrix_projection"), dict):
        score += 4
    for segment in table.get("content_segments", []) or []:
        if not isinstance(segment, dict):
            continue
        if (
            str(segment.get("source") or "").strip() == "cross_page_result_matrix_statistical_note"
            and not str(segment.get("owner_table_id") or "").strip()
        ):
            score -= 4
    return score


def _markdown_explicit_table_note_owner_score(
    table: dict[str, Any],
    norm: str,
    *,
    occurrence_page: int,
) -> int:
    table_id = str(table.get("table_id") or table.get("block_id") or "").strip()
    if not table_id or not norm:
        return 0
    for key in ("note_blocks", "content_segments"):
        for segment in table.get(key, []) or []:
            if not isinstance(segment, dict):
                continue
            if _markdown_compact_table_text(segment.get("text") or "") != norm:
                continue
            if _markdown_table_note_occurrence_page(segment, table) != occurrence_page:
                continue
            if (
                str(segment.get("owner_table_id") or "").strip() == table_id
                and str(segment.get("note_scope") or "").strip() == "previous_table"
            ):
                return 100
    return 0


def _markdown_table_note_occurrence_page(segment: dict[str, Any], table: dict[str, Any]) -> int:
    for value in (
        segment.get("physical_page"),
        segment.get("continuation_page"),
        segment.get("continued_on_page"),
        segment.get("page"),
        table.get("page"),
    ):
        try:
            page = int(value or 0)
        except (TypeError, ValueError):
            continue
        if page > 0:
            return page
    return 0


def _markdown_table_renderable_note_occurrence_keys(table: dict[str, Any]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for key in ("note_blocks", "content_segments"):
        for item in table.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            if key == "content_segments" and role not in {"note", "table_note"}:
                continue
            norm = _markdown_compact_table_text(item.get("text") or "")
            if norm:
                keys.add((norm, _markdown_table_note_occurrence_page(item, table)))
    return keys


def _markdown_table_renderable_note_norms(table: dict[str, Any]) -> set[str]:
    norms: set[str] = set()
    for key in ("note_blocks", "content_segments"):
        for item in table.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            if key == "content_segments" and role not in {"note", "table_note"}:
                continue
            norm = _markdown_compact_table_text(item.get("text") or "")
            if norm:
                norms.add(norm)
    return norms


def _markdown_table_note_norms(table: dict[str, Any]) -> set[str]:
    norms: set[str] = set()
    norms.update(_markdown_table_renderable_note_norms(table))
    for ref_key in ("cell_note_refs", "header_note_refs"):
        for ref in table.get(ref_key, []) or []:
            if not isinstance(ref, dict):
                continue
            norm = _markdown_compact_table_text(ref.get("note_text") or "")
            if norm:
                norms.add(norm)
    return norms


def _filter_markdown_table_notes_for_preferred_owner(
    table: dict[str, Any],
    segments: list[Any],
    preferred_owner_by_occurrence: dict[tuple[str, int], str],
) -> list[Any]:
    if not preferred_owner_by_occurrence:
        return segments
    table_id = str(table.get("table_id") or table.get("block_id") or "").strip()
    if not table_id:
        return segments
    keep: list[Any] = []
    for segment in segments:
        if not isinstance(segment, dict):
            keep.append(segment)
            continue
        if _markdown_float_owned_segment_can_repeat_across_tables(segment):
            keep.append(segment)
            continue
        norm = _markdown_compact_table_text(segment.get("text") or "")
        occurrence_key = (norm, _markdown_table_note_occurrence_page(segment, table))
        preferred_owner = preferred_owner_by_occurrence.get(occurrence_key)
        if preferred_owner and preferred_owner != table_id:
            continue
        keep.append(segment)
    return keep


def _markdown_float_owned_segment_can_repeat_across_tables(segment: dict[str, Any]) -> bool:
    if str(segment.get("role") or "").strip() not in {"note", "table_note", "legend"}:
        return False
    if str(segment.get("note_group_id") or "").strip():
        return True
    source = str(segment.get("source") or "").strip()
    text = str(segment.get("text") or "")
    cleaned = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(
        r"(?:附加信息|补充信息|备注|注释|说明|Note|Notes)\s*[:：]?",
        cleaned,
        re.IGNORECASE,
    ):
        return True
    if source == "study_condition_result_matrix_note_row":
        return True
    if source == "cross_page_result_matrix_statistical_note":
        return _markdown_result_matrix_statistical_note_text(text)
    if source in {"dose_response_result_panel_note_row", "result_matrix_statistical_note_after_table"}:
        return _markdown_result_matrix_statistical_note_text(text)
    return False


def _markdown_result_matrix_statistical_note_text(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return False
    if cleaned.startswith("-无值得注意的结果"):
        return True
    if re.search(r"\b(?:Dunnett|Fisher)\b", cleaned, re.IGNORECASE):
        return True
    if re.search(r"\*+\s*-\s*p\s*<\s*0\.\d+", cleaned, re.IGNORECASE):
        return True
    return False


def _normalize_markdown_table_block_note_fields(block: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(block)
    note_blocks = list(normalized.get("note_blocks", []) or [])
    content_segments = list(normalized.get("content_segments", []) or [])
    if note_blocks:
        normalized["note_blocks"] = _dedupe_markdown_table_note_segments(note_blocks)
    if content_segments:
        normalized["content_segments"] = _dedupe_markdown_table_note_segments(content_segments)
    return normalized


def _repair_markdown_float_owned_note_paragraph(text: str) -> str:
    repaired = str(text or "").strip()
    repaired = re.sub(r"未检\s+测(?=\s*[；;，,。])", "未检测", repaired)
    repaired = re.sub(r"未检\s*测(?=\s*[；;，,。])", "未检测", repaired)
    repaired = re.sub(r"^([*+-])(?=\s)", r"\\\1", repaired)
    return repaired


def _markdown_table_grid_cell_text_norms(block: dict[str, Any]) -> set[str]:
    if str(block.get("block_type") or "").strip().lower() != "table":
        return set()
    norms: set[str] = set()
    for key in ("semantic_display_grid", "semantic_grid", "display_grid", "raw_grid", "grid", "data_grid"):
        grid = block.get(key)
        if not isinstance(grid, list):
            continue
        norms.update(_markdown_grid_cell_text_norms(grid))
    return norms


def _markdown_grid_cell_text_norms(grid: list[list[Any]]) -> set[str]:
    norms: set[str] = set()
    for row in grid:
        if not isinstance(row, list):
            continue
        row_text_parts: list[str] = []
        for cell in row:
            text = _markdown_escape_table_cell(cell)
            norm = _markdown_compact_table_text(text)
            if norm:
                norms.add(norm)
                row_text_parts.append(text)
        row_norm = _markdown_compact_table_text(" ".join(row_text_parts))
        if row_norm:
            norms.add(row_norm)
    return norms


def _markdown_image_has_renderable_semantics(block: dict[str, Any]) -> bool:
    if str(block.get("caption_text") or "").strip() or str(block.get("title") or "").strip():
        return True
    if block.get("caption_blocks") or block.get("note_blocks"):
        return True
    semantics = block.get("figure_semantics")
    if isinstance(semantics, dict) and any(str(value or "").strip() for value in semantics.values()):
        return True
    for segment in block.get("content_segments", []) or []:
        if not isinstance(segment, dict):
            continue
        role = str(segment.get("role") or "").strip()
        text = str(segment.get("text") or "").strip()
        if role in {"caption", "legend", "note", "embedded_text", "embedded_code"} and text:
            return True
    return False


def _order_markdown_float_segments(block: dict[str, Any], segments: list[Any]) -> list[Any]:
    if not segments:
        return []
    if str(block.get("block_type") or "").strip().lower() == "table":
        return [
            segment
            for _, segment in sorted(
                enumerate(segments),
                key=lambda item: _markdown_table_float_segment_order_key(item[1], item[0]),
            )
        ]
    caption_relation = ""
    for segment in segments:
        if not isinstance(segment, dict) or str(segment.get("role") or "") != "caption":
            continue
        caption_relation = str(segment.get("relation") or "").strip().lower()
        break
    if caption_relation != "below":
        return segments

    role_rank = {
        "embedded_text": 0,
        "embedded_code": 0,
        "caption": 1,
        "legend": 2,
        "note": 2,
    }
    return [
        segment
        for _, segment in sorted(
        enumerate(segments),
        key=lambda item: (
            role_rank.get(str(item[1].get("role") or "").strip(), 1)
            if isinstance(item[1], dict)
            else 1,
            item[0],
        ),
        )
    ]


def _markdown_table_float_segment_order_key(segment: Any, original_index: int) -> tuple[int, float, int]:
    if not isinstance(segment, dict):
        return (9, float(original_index), original_index)
    role = str(segment.get("role") or "").strip()
    relation = str(segment.get("relation") or "").strip().lower()
    source = str(segment.get("source") or "").strip()
    bbox = _markdown_block_bbox(segment)
    note_group_id = str(segment.get("note_group_id") or "").strip()
    if note_group_id and str(segment.get("presentation_mode") or "").strip() == "lines":
        try:
            line_index = int(segment.get("note_line_index", original_index) or 0)
        except (TypeError, ValueError):
            line_index = original_index
        return (19, float(line_index), original_index)
    try:
        source_order_y = float(segment.get("source_order_y"))
        source_order_x = float(segment.get("source_order_x", 0.0) or 0.0)
    except (TypeError, ValueError):
        source_order_y = None
        source_order_x = 0.0
    if role in {"note", "table_note", "legend"} and (
        bbox is not None or source_order_y is not None
    ) and _markdown_float_owned_segment_uses_physical_note_order(segment):
        page_rank = _markdown_float_owned_segment_physical_page(segment)
        order_y = bbox[1] if bbox is not None else source_order_y
        order_x = bbox[0] if bbox is not None else source_order_x
        return (
            20,
            float(page_rank) * 1000000.0 + float(order_y) * 1000.0 + float(order_x),
            original_index,
        )
    role_rank = {
        "embedded_text": 0,
        "embedded_code": 0,
        "caption": 1,
        "legend": 2,
        "note": 2,
        "table_note": 2,
    }.get(role, 3)
    if relation == "above":
        relation_rank = 0
    elif source == "trailing_table_note_row":
        relation_rank = 2
    elif relation == "below":
        relation_rank = 3
    else:
        relation_rank = 1
    y0 = bbox[1] if bbox is not None else float(original_index)
    return (role_rank * 10 + relation_rank, y0, original_index)


def _markdown_float_owned_segment_uses_physical_note_order(segment: dict[str, Any]) -> bool:
    return _markdown_float_owned_segment_physical_page(segment) > 0


def _markdown_float_owned_segment_physical_page(segment: dict[str, Any]) -> int:
    for value in (
        segment.get("physical_page"),
        segment.get("continuation_page"),
        segment.get("continued_on_page"),
        segment.get("page"),
    ):
        try:
            page = int(value or 0)
        except (TypeError, ValueError):
            continue
        if page > 0:
            return page
    return 0


def _markdown_block_bbox_overlap(
    left: dict[str, Any],
    right: dict[str, Any],
) -> float:
    left_bbox = _markdown_block_bbox(left)
    right_bbox = _markdown_block_bbox(right)
    if left_bbox is None or right_bbox is None:
        return 0.0
    x0 = max(left_bbox[0], right_bbox[0])
    y0 = max(left_bbox[1], right_bbox[1])
    x1 = min(left_bbox[2], right_bbox[2])
    y1 = min(left_bbox[3], right_bbox[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    left_area = max(1.0, (left_bbox[2] - left_bbox[0]) * (left_bbox[3] - left_bbox[1]))
    right_area = max(1.0, (right_bbox[2] - right_bbox[0]) * (right_bbox[3] - right_bbox[1]))
    return intersection / min(left_area, right_area)


def _markdown_image_embedded_text_owned_by_structured_table(
    image_block: dict[str, Any],
    table_blocks: list[dict[str, Any]],
) -> bool:
    content_text = str(image_block.get("content_text") or image_block.get("embedded_text") or "").strip()
    if not content_text:
        return False
    image_kind = str(image_block.get("image_kind_guess") or "").strip()
    text_recovery = image_block.get("text_recovery") if isinstance(image_block.get("text_recovery"), dict) else {}
    evidence_only_ocr = (
        bool(text_recovery.get("evidence_only"))
        or str(image_block.get("embedded_text_source") or text_recovery.get("source") or "").strip()
        == "ocr-image-evidence"
    )
    if image_kind not in {
        "captioned_textual_figure",
        "path_screenshot",
        "textual_image",
        "contextual_textual_image",
        "chart_figure",
        "captioned_figure",
    } and not evidence_only_ocr:
        return False

    image_id = str(image_block.get("image_id") or image_block.get("block_id") or "").strip()
    for table in table_blocks:
        if str(table.get("block_type") or "").strip().lower() != "table":
            continue
        bbox_overlap = _markdown_block_bbox_overlap(image_block, table)
        source_image_id = str(table.get("source_image_id") or "").strip()
        detection_source = str(table.get("detection_source") or table.get("detection_method") or "").strip()
        if bbox_overlap >= 0.62 and source_image_id and image_id and source_image_id == image_id:
            return True
        if bbox_overlap >= 0.62 and detection_source in {"embedded_image_ocr", "visual_structure_grid"}:
            return True
        if bbox_overlap >= 0.18 and _markdown_image_text_covers_structured_table_text(content_text, table):
            return True
    return False


def _markdown_image_text_covers_structured_table_text(image_text: str, table: dict[str, Any]) -> bool:
    image_tokens = _markdown_semantic_tokens(image_text)
    table_tokens = _markdown_semantic_tokens(_markdown_structured_table_plain_text(table))
    if len(image_tokens) < 8 or len(table_tokens) < 8:
        return False
    overlap = image_tokens & table_tokens
    return len(overlap) / max(1, len(table_tokens)) >= 0.42


def _markdown_structured_table_plain_text(table: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "caption_text"):
        value = str(table.get(key) or "").strip()
        if value:
            parts.append(value)
    for row_key in ("header_rows", "display_grid", "grid", "rows"):
        rows = table.get(row_key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, list):
                continue
            for cell in row:
                if isinstance(cell, dict):
                    text = str(cell.get("text") or "").strip()
                else:
                    text = str(cell or "").strip()
                if text:
                    parts.append(text)
    return " ".join(parts)


def _markdown_semantic_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", str(text or "").lower())
    return {token for token in tokens if len(token) >= 2}


def _clean_markdown_float_owned_text(text: str) -> str:
    projected = project_pdf_math_symbol_display_text(project_table_cell_display_text(text))
    cleaned = str(projected or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _markdown_candidate_has_safe_boundaries(text: str, candidate: str, start: int, end: int, span: dict[str, Any]) -> bool:
    if not candidate:
        return False
    if str(span.get("formula_complexity") or "").strip() != "inline_symbol":
        return True
    if len(candidate) != 1 or not re.fullmatch(r"[A-Za-z]", candidate):
        return True
    before = text[start - 1 : start]
    after = text[end : end + 1]
    if before and (before.isalnum() or before == "_"):
        return False
    if after and (after.isalnum() or after == "_"):
        return False
    return True


def _render_pdf_bbox_crop_markdown(
    *,
    source_path: Any,
    page_number: int,
    bbox: Any,
    alt_text: str,
    scale: float = 2.0,
) -> str | None:
    if not isinstance(source_path, Path) or not source_path.exists():
        return None
    if page_number <= 0 or not isinstance(bbox, list) or len(bbox) != 4:
        return None
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
            page_rect = page.rect
            clip = clip & page_rect
            if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                return None
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
            image_bytes = pixmap.tobytes("png")
    except Exception:
        return None
    if not image_bytes:
        return None
    safe_alt = str(alt_text or "PDF crop").replace("[", "(").replace("]", ")")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"![{safe_alt}](data:image/png;base64,{encoded})"


def _markdown_text_block_is_unmarked_list_item(block: dict[str, Any]) -> bool:
    return (
        str(block.get("block_type") or "").strip().lower() == "text"
        and str(block.get("semantic_role") or "").strip() == "body_list_item"
        and str(block.get("list_style") or "").strip() == "unmarked_indented"
    )


def _normalize_markdown_explicit_bullet_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    return re.sub(
        r"^\s*[\u2022\u25cf\u25cb\u25aa\u25e6\u2219\uf06c\uf0b7\u00b7\x01]\s*",
        "- ",
        raw,
        count=1,
    )


def _append_markdown_text_block(lines: list[str], block: dict[str, Any], *, text_override: str | None = None) -> None:
    text = (
        _repair_markdown_text_with_contextual_inline_atoms(text_override)
        if text_override is not None
        else _project_markdown_text_with_inline_formulas(block)
    )
    if not text:
        return
    if re.fullmatch(r"[\s\u201c\u201d\"'กฐ]+", text):
        return
    text = _append_markdown_footnote_ref_suffix(text, block)
    text = _markdown_linkify_visible_urls(text)
    if _markdown_text_block_is_unmarked_list_item(block) and not _markdown_text_starts_bullet_item(text):
        text = f"- {text}"
    else:
        text = _normalize_markdown_explicit_bullet_text(text)
    lines.append(text)
    lines.append("")


def _append_markdown_centered_front_matter_title_block(lines: list[str], block: dict[str, Any]) -> None:
    text = _project_markdown_text_with_inline_formulas(block)
    if not text:
        return
    text = _append_markdown_footnote_ref_suffix(text, block)
    text = _markdown_linkify_visible_urls(text)
    lines.append(f"**{text}**")
    lines.append("")


def _toc_block_entries_for_markdown(block: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [entry for entry in block.get("entries", []) or [] if isinstance(entry, dict)]
    if not entries:
        entries = [entry for entry in block.get("_raw_entries", []) or [] if isinstance(entry, dict)]
    return sorted(entries, key=lambda entry: int(entry.get("sequence_entry_index", entry.get("entry_index", 0)) or 0))


def _format_toc_block_entry_markdown(entry: dict[str, Any]) -> str:
    outline_index = str(entry.get("outline_index") or "").strip()
    title = str(entry.get("text") or entry.get("title") or "").strip()
    if outline_index.endswith(".0") and outline_index.count(".") == 1:
        outline_index = f"{outline_index[:-2]}."
    label = " ".join(part for part in (outline_index, title) if part).strip()
    page_locator = str(entry.get("page_locator") or "").strip()
    if page_locator:
        return f"{label} {page_locator}".strip()
    return label


def _append_markdown_toc_block(lines: list[str], block: dict[str, Any]) -> None:
    title = str(block.get("title") or "").strip() or "Table of contents"
    lines.append(f"# {_markdown_linkify_visible_urls(title)}")
    lines.append("")
    for entry in _toc_block_entries_for_markdown(block):
        entry_text = _format_toc_block_entry_markdown(entry)
        if entry_text:
            lines.append(_markdown_linkify_visible_urls(entry_text))
    if lines and lines[-1] != "":
        lines.append("")


def _append_markdown_algorithm_pseudocode(lines: list[str], block: dict[str, Any]) -> None:
    title = str(block.get("title") or block.get("algorithm_ref") or "").strip()
    label = title or str(block.get("algorithm_ref") or "").strip() or "Algorithm pseudocode"
    projected_text = _project_markdown_text_with_inline_formulas(block)
    if not projected_text:
        return

    projected_lines = [line.rstrip() for line in projected_text.splitlines()]
    if title and projected_lines and projected_lines[0].strip() == title:
        projected_lines = projected_lines[1:]
    body_text = "\n".join(line for line in projected_lines if line.strip()).strip()

    lines.append(f"**Algorithm pseudocode: {label}**")
    lines.append("")
    if body_text:
        lines.append(body_text)
        lines.append("")


def _clean_structure_template_markdown_title(value: Any) -> str:
    title = str(value or "").strip()
    if not title:
        return ""
    title = re.sub(r"(?:结构模板|表单模板)\s*$", "", title).strip()
    title = re.sub(
        r"^(?:应按照以下顺序提交|以下顺序提交|按照以下顺序提交|提交|：|:)+",
        "",
        title,
    ).strip()
    title = re.sub(r"[：:]\s*$", "", title).strip()
    return title


def _structure_template_markdown_title_is_instruction_like(value: Any) -> bool:
    compact = _markdown_compact_table_text(value)
    if not compact:
        return False
    prefixes = [
        "\u5e94\u6309\u7167\u4ee5\u4e0b\u987a\u5e8f",
        "\u6309\u7167\u4ee5\u4e0b\u987a\u5e8f",
        "\u4ee5\u4e0b\u987a\u5e8f",
        "\u5efa\u8bae\u91c7\u7528\u4ee5\u4e0b",
        "submitinthefollowingorder",
        "followingorder",
    ]
    return any(compact.startswith(_markdown_compact_table_text(prefix)) for prefix in prefixes)


def _markdown_title_without_leading_outline_marker(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(
        r"^\s*(?:[A-Z]\.?|[IVXLCDM]+\.?|\d+(?:\.\d+)*)\s*[:\uff1a.\-]?\s*",
        "",
        text,
        count=1,
    ).strip()


def _markdown_titles_equivalent_ignoring_outline_marker(
    primary_title: Any,
    candidate_title: Any,
) -> bool:
    primary_norm = _markdown_compact_table_text(
        _markdown_title_without_leading_outline_marker(primary_title)
    )
    candidate_norm = _markdown_compact_table_text(
        _markdown_title_without_leading_outline_marker(candidate_title)
    )
    return bool(primary_norm and candidate_norm and primary_norm == candidate_norm)


def _structure_template_markdown_heading(block: dict[str, Any]) -> str:
    section_context = dict(block.get("section_context", {}) or {})
    section_title = _clean_structure_template_markdown_title(section_context.get("section_title"))
    if section_title:
        section_title = re.sub(r"^\s*(?:[A-Z]\.?|[IVXLCDM]+\.?|\d+(?:\.\d+)*)\s*[:：.-]?\s*", "", section_title).strip()
    title = _clean_structure_template_markdown_title(block.get("title"))
    is_continuation = bool(block.get("is_structure_template_continuation", False))
    if not title and is_continuation:
        title = _clean_structure_template_markdown_title(block.get("continued_from_title"))
    if str(block.get("template_profile") or "") == "tabular_form_template":
        label = title or section_title or "\u8868\u683c\u5f0f\u7ed3\u6784"
        suffix = _structure_template_markdown_continuation_suffix(label, is_continuation)
        return f"{label}{suffix}"
    if title:
        suffix = _structure_template_markdown_continuation_suffix(title, is_continuation)
        return f"{title}{suffix}"
    if not section_title and title:
        section_title = re.sub(r"(?:应按照以下顺序提交|以下顺序提交|按照以下顺序提交|提交|：|:)+", "", title).strip()
    if section_title:
        suffix = _structure_template_markdown_continuation_suffix(section_title, is_continuation)
        return f"{section_title}{suffix}"
    return "\u8868\u683c\u5f0f\u7ed3\u6784"


def _structure_template_markdown_continuation_suffix(heading: Any, is_continuation: bool) -> str:
    if not is_continuation:
        return ""
    text = str(heading or "")
    if re.search(r"(?:\(\s*\u7eed\s*\)|\uff08\s*\u7eed\s*\uff09|continued)", text, re.IGNORECASE):
        return ""
    return "\uff08\u7eed\uff09"


def _structure_template_markdown_heading_is_internal_fallback(
    block: dict[str, Any],
    heading: str,
) -> bool:
    if _markdown_compact_table_text(heading) != _markdown_compact_table_text("\u8868\u683c\u5f0f\u7ed3\u6784"):
        return False
    section_context = dict(block.get("section_context", {}) or {})
    return not any(
        _clean_structure_template_markdown_title(value)
        for value in (
            block.get("title"),
            block.get("continued_from_title"),
            section_context.get("section_title"),
        )
    )


def _structure_template_markdown_heading_redundant_with_previous_visible_title(
    block: dict[str, Any],
    heading: str,
    previous_visible_title: str,
) -> bool:
    if not previous_visible_title:
        return False
    if bool(block.get("is_structure_template_continuation", False)):
        return False
    if not _structure_template_markdown_title_is_instruction_like(block.get("title")):
        return False
    return _markdown_titles_equivalent_ignoring_outline_marker(previous_visible_title, heading)


def _structure_template_markdown_heading_is_visible(
    block: dict[str, Any],
    heading: str,
    previous_visible_title: str = "",
) -> bool:
    if _structure_template_markdown_heading_is_internal_fallback(block, heading):
        return False
    if _markdown_object_title_redundant_with_previous_heading(block, previous_visible_title):
        return False
    if _structure_template_markdown_heading_redundant_with_previous_visible_title(
        block,
        heading,
        previous_visible_title,
    ):
        return False
    return True


def _markdown_structure_template_above_title_prelude(
    block: dict[str, Any],
    nearby_blocks: list[dict[str, Any]],
    *,
    already_rendered_source_ids: set[str] | None = None,
) -> tuple[list[str], set[str]]:
    if str(block.get("block_type") or "").strip().lower() != "structure_template":
        return [], set()
    explicit_preludes = [
        dict(prelude)
        for prelude in block.get("prelude_blocks", []) or []
        if isinstance(prelude, dict) and str(prelude.get("text") or "").strip()
    ]
    if explicit_preludes:
        rendered_source_ids = already_rendered_source_ids or set()
        texts: list[str] = []
        ids: set[str] = set()
        for prelude in sorted(
            explicit_preludes,
            key=lambda item: (
                (_markdown_block_bbox(item) or (0.0, 0.0, 0.0, 0.0))[1],
                (_markdown_block_bbox(item) or (0.0, 0.0, 0.0, 0.0))[0],
            ),
        ):
            source_id = str(
                prelude.get("source_block_id")
                or prelude.get("block_id")
                or prelude.get("source_id")
                or prelude.get("source_row_ref")
                or ""
            ).strip()
            if source_id and source_id in rendered_source_ids:
                continue
            texts.append(str(prelude.get("text") or "").strip())
            if source_id:
                ids.add(source_id)
        return texts, ids
    template_bbox = _markdown_block_bbox(block)
    if template_bbox is None:
        return [], set()
    title_norm = _markdown_compact_table_text(
        _clean_structure_template_markdown_title(block.get("title") or "")
    )
    prelude_blocks: list[dict[str, Any]] = []
    title_top = _markdown_structure_template_title_top(block, title_norm, nearby_blocks)
    previous_blocks = nearby_blocks[: nearby_blocks.index(block)] if block in nearby_blocks else nearby_blocks
    for candidate in reversed(previous_blocks[-8:]):
        if str(candidate.get("block_type") or "").strip().lower() != "text":
            continue
        text = str(candidate.get("display_text") or candidate.get("text") or "").strip()
        if not _markdown_text_looks_like_object_prelude_label(text):
            if prelude_blocks:
                break
            continue
        bbox = _markdown_block_bbox(candidate)
        if bbox is None:
            continue
        if bbox[3] > template_bbox[1] + 72.0:
            continue
        if bbox[3] < template_bbox[1] - 96.0:
            break
        if _markdown_block_bbox_overlap({"bbox": bbox}, {"bbox": template_bbox}) < 0.04 and bbox[0] > template_bbox[2]:
            continue
        role = str(candidate.get("semantic_role") or "").strip()
        if role in {"table_note", "note", "footnote", "footnote_continuation", "page_header", "page_footer"}:
            continue
        prelude_blocks.append(candidate)
    for candidate in nearby_blocks:
        if candidate is block or not isinstance(candidate, dict):
            continue
        if str(candidate.get("block_type") or "").strip().lower() != "text":
            continue
        text = str(candidate.get("display_text") or candidate.get("text") or "").strip()
        if not _markdown_text_looks_like_object_prelude_label(text):
            continue
        bbox = _markdown_block_bbox(candidate)
        if bbox is None:
            continue
        if title_top is not None and bbox[1] >= title_top:
            continue
        if bbox[1] < template_bbox[1] - 8.0 or bbox[3] > template_bbox[3] + 2.0:
            continue
        role = str(candidate.get("semantic_role") or "").strip()
        if role in {"table_note", "note", "footnote", "footnote_continuation", "page_header", "page_footer"}:
            continue
        if candidate not in prelude_blocks:
            prelude_blocks.append(candidate)
    for candidate in block.get("source_blocks", []) or block.get("child_blocks", []) or []:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("block_type") or "").strip().lower() != "text":
            continue
        text = str(candidate.get("display_text") or candidate.get("text") or "").strip()
        if not _markdown_text_looks_like_object_prelude_label(text):
            continue
        bbox = _markdown_block_bbox(candidate)
        if bbox is None:
            continue
        if title_top is not None and bbox[1] >= title_top:
            continue
        if bbox[1] < template_bbox[1] - 8.0 or bbox[3] > template_bbox[3] + 2.0:
            continue
        role = str(candidate.get("semantic_role") or "").strip()
        if role in {"table_note", "note", "footnote", "footnote_continuation", "page_header", "page_footer"}:
            continue
        if candidate not in prelude_blocks:
            prelude_blocks.append(candidate)
    if not prelude_blocks:
        return [], set()
    prelude_blocks.sort(key=lambda item: (_markdown_block_bbox(item) or (0.0, 0.0, 0.0, 0.0))[1])
    texts: list[str] = []
    ids: set[str] = set()
    rendered_source_ids = already_rendered_source_ids or set()
    for candidate in prelude_blocks:
        text = str(candidate.get("display_text") or candidate.get("text") or "").strip()
        if not text:
            continue
        if title_norm and _markdown_compact_table_text(text) == title_norm:
            continue
        candidate_id = str(candidate.get("block_id") or candidate.get("source_id") or "").strip()
        if candidate_id and candidate_id in rendered_source_ids:
            continue
        texts.append(text)
        if candidate_id:
            ids.add(candidate_id)
    return texts, ids


def _markdown_structure_template_title_top(
    block: dict[str, Any],
    title_norm: str,
    nearby_blocks: list[dict[str, Any]] | None = None,
) -> float | None:
    if not title_norm:
        return None
    candidates = []
    source_blocks = list(block.get("source_blocks", []) or block.get("child_blocks", []) or [])
    if nearby_blocks:
        source_blocks.extend(nearby_blocks)
    for source_block in source_blocks:
        if not isinstance(source_block, dict):
            continue
        if source_block is block:
            continue
        text = str(source_block.get("display_text") or source_block.get("text") or "").strip()
        if not text:
            continue
        norm = _markdown_compact_table_text(text)
        if norm and (norm == title_norm or norm in title_norm or title_norm in norm):
            bbox = _markdown_block_bbox(source_block)
            if bbox is not None:
                candidates.append(bbox[1])
    if candidates:
        return min(candidates)
    bbox = _markdown_block_bbox(block)
    return bbox[1] if bbox is not None else None


def _markdown_text_looks_like_object_prelude_label(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or len(raw) > 24:
        return False
    return bool(re.fullmatch(r"示例|实例|例|Example|Examples|EXAMPLE|EXAMPLES", raw))


def _structure_template_markdown_duplicate_title_row(
    raw_row_text: Any,
    *,
    heading: str,
    block: dict[str, Any],
) -> bool:
    row_norm = _markdown_compact_table_text(raw_row_text)
    if not row_norm:
        return False
    section_context = dict(block.get("section_context", {}) or {})
    candidate_titles = [
        heading,
        block.get("title"),
        block.get("continued_from_title"),
        section_context.get("section_title"),
    ]
    for candidate in candidate_titles:
        title_norm = _markdown_compact_table_text(
            _clean_structure_template_markdown_title(candidate)
        )
        if not title_norm:
            continue
        if row_norm == title_norm:
            return True
        if len(row_norm) >= 12 and row_norm in title_norm:
            return True
        if len(title_norm) >= 12 and title_norm in row_norm:
            return True
    return False


def _append_markdown_structure_template(lines: list[str], block: dict[str, Any]) -> None:
    _append_markdown_structure_template_with_deferred_notes(lines, block, deferred_note_texts=set())


def _markdown_structure_template_multilevel_header_grid_rows(block: dict[str, Any]) -> list[list[str]]:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    multilevel_projection = projection.get("ruled_multilevel_template_header_projection")
    if isinstance(multilevel_projection, dict) and str(multilevel_projection.get("semantic_profile") or "") == "ruled_multilevel_ctd_template_header":
        semantic_grid = multilevel_projection.get("semantic_grid")
        if isinstance(semantic_grid, list):
            rows = [
                [
                    _markdown_escape_table_cell(cell)
                    for cell in row
                ]
                for row in semantic_grid
                if isinstance(row, list) and any(str(cell or "").strip() for cell in row)
            ]
            if rows and len(rows[0]) >= 2:
                return rows
    return []


def _markdown_structure_template_projected_header_text_norms(block: dict[str, Any]) -> set[str]:
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    multilevel_projection = projection.get("ruled_multilevel_template_header_projection")
    if not isinstance(multilevel_projection, dict):
        return set()
    if str(multilevel_projection.get("semantic_profile") or "") != "ruled_multilevel_ctd_template_header":
        return set()
    texts: list[Any] = []
    for row in multilevel_projection.get("semantic_grid", []) or []:
        if not isinstance(row, list):
            continue
        texts.extend(row)
    texts.extend(multilevel_projection.get("logical_columns", []) or [])
    for span_cell in multilevel_projection.get("span_header_cells", []) or []:
        if isinstance(span_cell, dict):
            texts.append(span_cell.get("text"))
    for fragment in multilevel_projection.get("header_fragments", []) or []:
        if isinstance(fragment, dict):
            texts.append(fragment.get("text"))
    texts.extend(multilevel_projection.get("consumed_header_row_texts", []) or [])
    texts.extend(multilevel_projection.get("consumed_body_row_texts", []) or [])
    return {
        norm
        for norm in (_markdown_compact_table_text(text) for text in texts)
        if norm
    }


def _markdown_structure_template_blank_header_grid_rows(block: dict[str, Any]) -> list[list[str]]:
    multilevel_rows = _markdown_structure_template_multilevel_header_grid_rows(block)
    if multilevel_rows:
        return multilevel_rows
    projection = block.get("semantic_projection_v2")
    projection = projection if isinstance(projection, dict) else {}
    header_projection = projection.get("ruled_slot_template_header_projection")
    if not isinstance(header_projection, dict):
        header_projection = projection.get("blank_template_header_grid_projection")
    if not isinstance(header_projection, dict):
        return []
    if str(header_projection.get("semantic_profile") or "") not in {
        "blank_ctd_table_template_header",
        "ruled_sparse_ctd_template_header",
    }:
        return []
    columns = [
        _markdown_escape_table_cell(column)
        for column in header_projection.get("logical_columns", []) or []
        if str(column or "").strip()
    ]
    if len(columns) < 2:
        return []
    return [columns]


def _markdown_structure_template_empty_table_grid_rows(
    block: dict[str, Any],
    nearby_blocks: list[dict[str, Any]] | None,
) -> list[list[str]]:
    if str(block.get("template_profile") or "") != "tabular_form_template":
        return []
    if not nearby_blocks:
        return []
    owned_ids = {
        str(item or "").strip()
        for item in block.get("owned_text_block_ids", []) or []
        if str(item or "").strip()
    }
    if not owned_ids:
        return []
    entry_blocks: list[dict[str, Any]] = []
    for candidate in nearby_blocks:
        candidate_id = str(candidate.get("block_id") or candidate.get("source_id") or "").strip()
        if candidate_id not in owned_ids:
            continue
        if str(candidate.get("block_type") or "").strip().lower() != "text":
            continue
        if str(candidate.get("semantic_role") or "").strip() != "structure_template_entry":
            continue
        text = str(candidate.get("display_text") or candidate.get("text") or "").strip()
        bbox = _markdown_block_bbox(candidate)
        if not text or bbox is None:
            continue
        entry_blocks.append(candidate)
    rows = _markdown_structure_template_entry_rows(entry_blocks)
    if len(rows) < 2:
        return []
    top_row = rows[0]
    if len(top_row) < 5:
        return []
    child_row = rows[1]
    if not (1 <= len(child_row) < len(top_row)):
        return []
    parent_index, child_cells = _markdown_structure_template_header_child_group(top_row, child_row)
    if parent_index is None or len(child_cells) < 2:
        return []

    top_headers = [_markdown_structure_template_entry_text(item) for item in top_row]
    if any(not text for text in top_headers):
        return []
    child_headers = [_markdown_structure_template_entry_text(item) for item in child_cells]
    top_header_row = [
        *top_headers[:parent_index],
        *([top_headers[parent_index]] * len(child_headers)),
        *top_headers[parent_index + 1:],
    ]
    leaf_header_row = [
        *top_headers[:parent_index],
        *child_headers,
        *top_headers[parent_index + 1:],
    ]
    column_count = len(top_header_row)
    if column_count < len(top_headers):
        return []

    template_rows: list[list[str]] = []
    first_header_bbox = _markdown_block_bbox(top_row[0])
    parent_bbox = _markdown_block_bbox(top_row[parent_index])
    if first_header_bbox is None or parent_bbox is None:
        return []
    for row in rows[2:]:
        item = row[0]
        text = _markdown_structure_template_entry_text(item)
        bbox = _markdown_block_bbox(item)
        if not text or bbox is None:
            continue
        if _markdown_compact_table_text(text) in {
            _markdown_compact_table_text(header) for header in top_headers + child_headers
        }:
            continue
        # Template row options live in the first template column. Isolated markers under
        # later columns are local note refs, not blank data rows.
        if abs(bbox[0] - first_header_bbox[0]) > max(32.0, (parent_bbox[0] - first_header_bbox[0]) * 0.35):
            continue
        template_rows.append([text, *([""] * (column_count - 1))])
    if not template_rows:
        return []
    return [
        [_markdown_escape_table_cell(cell) for cell in top_header_row],
        [_markdown_escape_table_cell(cell) for cell in leaf_header_row],
        *[[_markdown_escape_table_cell(cell) for cell in row] for row in template_rows],
    ]


def _markdown_structure_template_entry_rows(entry_blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not entry_blocks:
        return []
    items = [
        item for item in entry_blocks
        if _markdown_block_bbox(item) is not None
    ]
    items.sort(key=lambda item: (_markdown_block_bbox(item) or (0.0, 0.0, 0.0, 0.0))[1])
    rows: list[list[dict[str, Any]]] = []
    for item in items:
        bbox = _markdown_block_bbox(item)
        if bbox is None:
            continue
        center_y = (bbox[1] + bbox[3]) / 2.0
        if not rows:
            rows.append([item])
            continue
        previous_row = rows[-1]
        previous_centers = [
            ((_markdown_block_bbox(candidate) or (0.0, 0.0, 0.0, 0.0))[1] + (_markdown_block_bbox(candidate) or (0.0, 0.0, 0.0, 0.0))[3]) / 2.0
            for candidate in previous_row
        ]
        previous_center = sum(previous_centers) / max(1, len(previous_centers))
        if abs(center_y - previous_center) <= 8.0:
            previous_row.append(item)
        else:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda item: (_markdown_block_bbox(item) or (0.0, 0.0, 0.0, 0.0))[0])
    return rows


def _markdown_structure_template_header_child_group(
    top_row: list[dict[str, Any]],
    child_row: list[dict[str, Any]],
) -> tuple[int | None, list[dict[str, Any]]]:
    child_bboxes = [_markdown_block_bbox(item) for item in child_row]
    child_bboxes = [bbox for bbox in child_bboxes if bbox is not None]
    if len(child_bboxes) < 2:
        return None, []
    child_span = (
        min(bbox[0] for bbox in child_bboxes),
        max(bbox[2] for bbox in child_bboxes),
    )
    best_index: int | None = None
    best_score = float("inf")
    for index, item in enumerate(top_row):
        bbox = _markdown_block_bbox(item)
        if bbox is None:
            continue
        center_x = (bbox[0] + bbox[2]) / 2.0
        if child_span[0] - 18.0 <= center_x <= child_span[1] + 18.0:
            span_center = (child_span[0] + child_span[1]) / 2.0
            score = abs(center_x - span_center)
            if score < best_score:
                best_index = index
                best_score = score
    if best_index is None:
        return None, []
    parent_bbox = _markdown_block_bbox(top_row[best_index])
    if parent_bbox is None:
        return None, []
    parent_center = (parent_bbox[0] + parent_bbox[2]) / 2.0
    grouped_children = [
        item for item in child_row
        for bbox in [_markdown_block_bbox(item)]
        if bbox is not None and child_span[0] - 2.0 <= ((bbox[0] + bbox[2]) / 2.0) <= child_span[1] + 2.0
    ]
    if not grouped_children:
        return None, []
    if not (child_span[0] - 24.0 <= parent_center <= child_span[1] + 24.0):
        return None, []
    return best_index, grouped_children


def _markdown_structure_template_entry_text(block: dict[str, Any]) -> str:
    return str(block.get("display_text") or block.get("text") or "").strip()


def _append_markdown_structure_template_form_rows(
    lines: list[str],
    block: dict[str, Any],
    *,
    heading: str,
    fields: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    all_note_blocks: list[dict[str, Any]],
    template_profile: str,
    excluded_row_norms: set[str] | None = None,
) -> bool:
    rendered_form_rows: set[str] = set()
    excluded_norms = excluded_row_norms or set()
    local_form_title_norm = _markdown_compact_table_text(block.get("local_form_title") or "")
    visual_row_texts = _structure_template_markdown_visual_row_texts(block, all_note_blocks)
    if visual_row_texts and template_profile in {
        "tabular_form_template",
        "blank_study_summary_template",
        "blank_study_summary_template_continuation",
        "sparse_tabular_form_skeleton",
        "low_text_ruled_tabular_form_template",
    }:
        iterable_rows = visual_row_texts
    else:
        iterable_rows = [
            str(item.get("text") or "")
            for item in sorted(fields + sections, key=lambda row: int(row.get("row_index", 0) or 0))
        ]
    for raw_row_text in iterable_rows:
        row_norm = _markdown_compact_table_text(raw_row_text)
        if row_norm and row_norm in excluded_norms:
            continue
        if row_norm and row_norm == local_form_title_norm:
            continue
        row_text = _markdown_escape_inline_text(raw_row_text)
        if not row_text or row_text in rendered_form_rows:
            continue
        if _structure_template_markdown_duplicate_title_row(
            raw_row_text,
            heading=heading,
            block=block,
        ):
            continue
        rendered_form_rows.add(row_text)
        lines.append(f"- {_markdown_linkify_visible_urls(row_text)}")
    if rendered_form_rows:
        lines.append("")
        return True
    return False


def _structure_template_markdown_has_renderable_content(
    block: dict[str, Any],
    *,
    entries: list[dict[str, Any]],
    fields: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    note_blocks: list[dict[str, Any]],
) -> bool:
    if entries or fields or sections or note_blocks:
        return True
    if not _clean_structure_template_markdown_title(block.get("title")):
        return False
    if not str(block.get("title_source_block_id") or "").strip():
        return False
    if not bool(block.get("is_structure_template_continuation", False)):
        return False
    signals = block.get("semantic_signals")
    signals = signals if isinstance(signals, dict) else {}
    return (
        str(signals.get("same_page_post_note_continuation_state") or "").strip()
        == "pending_body_on_next_page"
    )


def _append_markdown_structure_template_with_deferred_notes(
    lines: list[str],
    block: dict[str, Any],
    *,
    deferred_note_texts: set[str],
    previous_visible_title: str = "",
    suppress_form_rows: bool = False,
    nearby_blocks: list[dict[str, Any]] | None = None,
) -> None:
    entries = [entry for entry in block.get("entries", []) or [] if isinstance(entry, dict)]
    fields = [field for field in block.get("fields", []) or [] if isinstance(field, dict)]
    sections = [section for section in block.get("sections", []) or [] if isinstance(section, dict)]
    all_note_blocks = _structure_template_markdown_ordered_note_blocks(block)
    note_blocks = list(all_note_blocks)
    if deferred_note_texts:
        note_blocks = [
            note
            for note in note_blocks
            if _markdown_compact_table_text(note.get("text") or "") not in deferred_note_texts
        ]
    if not _structure_template_markdown_has_renderable_content(
        block,
        entries=entries,
        fields=fields,
        sections=sections,
        note_blocks=note_blocks,
    ):
        return

    heading = _structure_template_markdown_heading(block)
    suppress_visible_title = _markdown_object_title_redundant_with_previous_heading(block, previous_visible_title)
    heading_is_visible = _structure_template_markdown_heading_is_visible(
        block,
        heading,
        previous_visible_title,
    )
    if heading_is_visible:
        lines.append(f"#### {_markdown_linkify_visible_urls(_markdown_escape_inline_text(heading))}")
        lines.append("")
    local_form_title = _markdown_escape_inline_text(block.get("local_form_title") or "")
    if (
        local_form_title
        and not _markdown_titles_equivalent_ignoring_outline_marker(heading, local_form_title)
    ):
        lines.append(f"##### {_markdown_linkify_visible_urls(local_form_title)}")
        lines.append("")
    title = _markdown_escape_inline_text(block.get("title") or "")
    if (
        title
        and not suppress_visible_title
        and not bool(block.get("is_structure_template_continuation", False))
        and not _markdown_table_titles_are_redundant(heading, title)
        and str(block.get("template_profile") or "") != "tabular_form_template"
    ):
        lines.append(_markdown_linkify_visible_urls(title))
        lines.append("")

    if suppress_form_rows:
        return

    template_profile = str(block.get("template_profile") or "")
    template_kind = str(block.get("template_kind") or "")
    renders_form_rows = (
        template_profile
        in {
            "tabular_form_template",
            "blank_study_summary_template",
            "blank_study_summary_template_continuation",
            "sparse_tabular_form_skeleton",
            "low_text_ruled_tabular_form_template",
            "populated_study_metadata",
        }
        or (template_kind in {"tabular_form", "study_metadata"} and bool(fields or sections))
    )
    if renders_form_rows:
        empty_table_grid_rows = _markdown_structure_template_empty_table_grid_rows(block, nearby_blocks)
        if empty_table_grid_rows:
            _append_markdown_pipe_table(lines, empty_table_grid_rows, len(empty_table_grid_rows[0]))
            for note in note_blocks:
                note_text = _markdown_structure_template_note_text(note.get("text") or "")
                if note_text:
                    lines.append(_markdown_linkify_visible_urls(note_text))
            if note_blocks:
                lines.append("")
            return
        multilevel_header_grid_rows = _markdown_structure_template_multilevel_header_grid_rows(block)
        if multilevel_header_grid_rows:
            _append_markdown_structure_template_form_rows(
                lines,
                block,
                heading=heading,
                fields=fields,
                sections=sections,
                all_note_blocks=all_note_blocks,
                template_profile=template_profile,
                excluded_row_norms=_markdown_structure_template_projected_header_text_norms(block),
            )
            _append_markdown_pipe_table(
                lines,
                multilevel_header_grid_rows,
                len(multilevel_header_grid_rows[0]),
            )
            for note in note_blocks:
                note_text = _markdown_structure_template_note_text(note.get("text") or "")
                if note_text:
                    lines.append(_markdown_linkify_visible_urls(note_text))
            if note_blocks:
                lines.append("")
            return
        blank_header_grid_rows = _markdown_structure_template_blank_header_grid_rows(block)
        if blank_header_grid_rows:
            _append_markdown_pipe_table(lines, blank_header_grid_rows, len(blank_header_grid_rows[0]))
            for note in note_blocks:
                note_text = _markdown_structure_template_note_text(note.get("text") or "")
                if note_text:
                    lines.append(_markdown_linkify_visible_urls(note_text))
            if note_blocks:
                lines.append("")
            return
        _append_markdown_structure_template_form_rows(
            lines,
            block,
            heading=heading,
            fields=fields,
            sections=sections,
            all_note_blocks=all_note_blocks,
            template_profile=template_profile,
        )
        for note in note_blocks:
            note_text = _markdown_structure_template_note_text(note.get("text") or "")
            if note_text:
                lines.append(_markdown_linkify_visible_urls(note_text))
        if note_blocks:
            lines.append("")
        return

    for entry in entries:
        outline_index = str(entry.get("outline_index") or "").strip()
        title_text = _markdown_escape_inline_text(entry.get("title") or entry.get("text") or "")
        if not outline_index and not title_text:
            continue
        try:
            outline_depth = max(1, int(entry.get("outline_depth", 1) or 1))
        except (TypeError, ValueError):
            outline_depth = 1
        indent = "  " * max(0, outline_depth - 2)
        item_text = " ".join(part for part in (outline_index, title_text) if part).strip()
        lines.append(f"{indent}- {_markdown_linkify_visible_urls(item_text)}")
    if entries:
        lines.append("")

    for note in note_blocks:
        note_text = _markdown_structure_template_note_text(note.get("text") or "")
        if note_text:
            lines.append(_markdown_linkify_visible_urls(note_text))
    if note_blocks:
        lines.append("")


def _structure_template_markdown_ordered_note_blocks(block: dict[str, Any]) -> list[dict[str, Any]]:
    note_blocks = [dict(note) for note in block.get("note_blocks", []) or [] if isinstance(note, dict)]
    if len(note_blocks) <= 1:
        return note_blocks
    numbered_note_keys = _structure_template_numbered_note_order_keys(note_blocks)

    row_positions = {
        _markdown_compact_table_text(row): index
        for index, row in enumerate(block.get("row_texts", []) or [])
        if _markdown_compact_table_text(row)
    }

    def note_order_key(note: dict[str, Any]) -> tuple[float, float, int, str]:
        note_text = str(note.get("text") or "")
        note_signature = _markdown_compact_table_text(note_text)
        numbered_key = numbered_note_keys.get(note_signature)
        if numbered_key is not None:
            return (float(numbered_key), 0.0, int(note.get("note_index", 0) or 0), note_signature)
        if note_signature in row_positions:
            return (float(row_positions[note_signature]), 0.0, int(note.get("note_index", 0) or 0), note_signature)
        bbox = _markdown_block_bbox(note)
        if bbox is not None:
            return (100000.0 + bbox[1], bbox[0], int(note.get("note_index", 0) or 0), note_signature)
        return (200000.0, 0.0, int(note.get("note_index", 0) or 0), note_signature)

    return sorted(note_blocks, key=note_order_key)


def _structure_template_numbered_note_order_keys(note_blocks: list[dict[str, Any]]) -> dict[str, int]:
    numbered: list[tuple[int, int, str]] = []
    for index, note in enumerate(note_blocks):
        number = _structure_template_numbered_note_number(note)
        if number is None:
            continue
        signature = _markdown_compact_table_text(str(note.get("text") or ""))
        if signature:
            numbered.append((number, index, signature))
    if len(numbered) < 2:
        return {}
    if len(numbered) != len(note_blocks):
        return {}
    numbers = {number for number, _index, _signature in numbered}
    if numbers != set(range(1, len(numbers) + 1)):
        return {}
    return {
        signature: number
        for number, _index, signature in numbered
    }


def _structure_template_numbered_note_number(note: dict[str, Any]) -> int | None:
    try:
        value = int(note.get("note_number", 0) or 0)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return value
    text = str(note.get("text") or "").strip()
    match = re.match(
        r"^(?:(?:备注|注释|说明)[:：]\s*)?(?:\((?P<ascii>\d{1,3})\)|（(?P<fullwidth>\d{1,3})）)\s*\S",
        text,
    )
    if not match:
        return None
    marker = str(match.group("ascii") or match.group("fullwidth") or "").strip()
    try:
        number = int(marker)
    except ValueError:
        return None
    return number if number > 0 else None


def _structure_template_context_rendered_by_following_composite_table(
    block: dict[str, Any],
    following_blocks: list[dict[str, Any]],
) -> bool:
    template_id = str(block.get("structure_template_id") or block.get("block_id") or "").strip()
    if not template_id:
        return False
    if str(block.get("template_kind") or "") != "study_metadata":
        return False
    for candidate in following_blocks:
        candidate_type = str(candidate.get("block_type") or "").strip().lower()
        if candidate_type == "structure_template":
            return False
        if candidate_type != "table":
            continue
        projection = candidate.get("semantic_projection_v2")
        projection = projection if isinstance(projection, dict) else {}
        matrix_projection = projection.get("study_condition_grouped_result_matrix_projection")
        if not isinstance(matrix_projection, dict):
            continue
        if str(matrix_projection.get("context_template_id") or "").strip() != template_id:
            continue
        return str(matrix_projection.get("presentation_boundary") or "") == "render_as_single_composite_table"
    return False


def _structure_template_deferred_note_texts_for_internal_following_tables(
    block: dict[str, Any],
    following_blocks: list[dict[str, Any]],
) -> set[str]:
    template_bbox = _markdown_block_bbox(block)
    if template_bbox is None:
        return set()
    note_blocks = _structure_template_markdown_ordered_note_blocks(block)
    if not note_blocks:
        return set()
    internal_table_bboxes: list[tuple[float, float, float, float]] = []
    for candidate in following_blocks:
        block_type = str(candidate.get("block_type") or "").strip().lower()
        if block_type in {"structure_template", "image"}:
            break
        candidate_bbox = _markdown_block_bbox(candidate)
        if candidate_bbox is None:
            continue
        if not _markdown_bbox_inside(candidate_bbox, template_bbox, tolerance=3.0):
            break
        if block_type == "table":
            internal_table_bboxes.append(candidate_bbox)
            continue
        if block_type == "text" and candidate_bbox[1] > template_bbox[3] + 2.0:
            break
    if not internal_table_bboxes:
        return set()
    last_internal_table_bottom = max(bbox[3] for bbox in internal_table_bboxes)
    deferred: set[str] = set()
    for note in note_blocks:
        note_text = str(note.get("text") or "")
        note_signature = _markdown_compact_table_text(note_text)
        if not note_signature:
            continue
        note_bbox = _markdown_block_bbox(note)
        if note_bbox is None:
            continue
        if note_bbox[1] >= last_internal_table_bottom - 2.0:
            deferred.add(note_signature)
    return deferred


def _append_markdown_deferred_structure_template_notes(
    lines: list[str],
    block: dict[str, Any],
    deferred_note_texts: set[str],
) -> None:
    if not deferred_note_texts:
        return
    note_blocks = [
        note
        for note in _structure_template_markdown_ordered_note_blocks(block)
        if _markdown_compact_table_text(note.get("text") or "") in deferred_note_texts
    ]
    if not note_blocks:
        return
    for note in note_blocks:
        note_text = _markdown_structure_template_note_text(note.get("text") or "")
        if note_text:
            lines.append(_markdown_linkify_visible_urls(note_text))
    lines.append("")


def _markdown_bbox_inside(
    bbox: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
    *,
    tolerance: float = 0.0,
) -> bool:
    return (
        container[0] - tolerance <= bbox[0]
        and bbox[1] >= container[1] - tolerance
        and bbox[2] <= container[2] + tolerance
        and bbox[3] <= container[3] + tolerance
    )


def _structure_template_markdown_visual_row_texts(block: dict[str, Any], note_blocks: list[dict[str, Any]]) -> list[str]:
    note_signatures = {
        _markdown_compact_table_text(str(note.get("text") or ""))
        for note in note_blocks
        if _markdown_compact_table_text(str(note.get("text") or ""))
    }
    inline_projection = (block.get("semantic_projection_v2") or {}).get("inline_form_row_projection", {})
    projected_rows = [
        dict(row)
        for row in inline_projection.get("rows", []) or []
        if isinstance(row, dict) and str(row.get("display_text") or "").strip()
    ]
    source_rows = [
        dict(row)
        for row in inline_projection.get("source_visual_rows", []) or []
        if isinstance(row, dict)
        and str(row.get("source_block_id") or "").strip()
        and str(row.get("text") or "").strip()
    ]
    if inline_projection.get("source_visual_rows_complete") and source_rows:
        source_ordered_rows = _structure_template_markdown_source_ordered_rows(
            projected_rows,
            source_rows,
            note_signatures,
        )
        if source_ordered_rows is not None:
            return source_ordered_rows
    consumed_signatures = Counter(
        _markdown_compact_table_text(str(text or ""))
        for text in inline_projection.get("consumed_row_signatures", []) or []
        if _markdown_compact_table_text(str(text or ""))
    )
    projected_row_signature_counts: list[Counter[str]] = []
    for projected_row in projected_rows:
        cell_signatures = Counter(
            _markdown_compact_table_text(str(cell.get("text") or ""))
            for cell in projected_row.get("cells", []) or []
            if isinstance(cell, dict) and _markdown_compact_table_text(str(cell.get("text") or ""))
        )
        projected_row_signature_counts.append(cell_signatures)
    if len(projected_rows) == 1 and not projected_row_signature_counts[0]:
        projected_row_signature_counts[0] = Counter(consumed_signatures)

    rows: list[str] = []
    inserted_projected_row_indexes: set[int] = set()
    for row in block.get("row_texts", []) or []:
        text = str(row or "").strip()
        if not text:
            continue
        signature = _markdown_compact_table_text(text)
        if signature in note_signatures:
            continue
        if consumed_signatures.get(signature, 0) > 0:
            consumed_signatures[signature] -= 1
            projected_row_index = next(
                (
                    index
                    for index, signature_counts in enumerate(projected_row_signature_counts)
                    if index not in inserted_projected_row_indexes and signature_counts.get(signature, 0) > 0
                ),
                None,
            )
            if projected_row_index is not None:
                projected_text = str(projected_rows[projected_row_index].get("display_text") or "").strip()
                if projected_text and _markdown_compact_table_text(projected_text) not in note_signatures:
                    rows.append(projected_text)
                inserted_projected_row_indexes.add(projected_row_index)
            continue
        rows.append(text)
    return rows


def _structure_template_markdown_source_ordered_rows(
    projected_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    note_signatures: set[str],
) -> list[str] | None:
    projected_index_by_source_id: dict[str, int] = {}
    for index, projected in enumerate(projected_rows):
        source_ids = [
            str(source_id or "").strip()
            for source_id in projected.get("source_block_ids", []) or []
            if str(source_id or "").strip()
        ]
        if not source_ids:
            return None
        for source_id in source_ids:
            if source_id in projected_index_by_source_id:
                return None
            projected_index_by_source_id[source_id] = index

    rows: list[str] = []
    inserted: set[int] = set()
    seen_source_ids: set[str] = set()
    ordered_sources = sorted(
        source_rows,
        key=lambda row: (
            float((row.get("bbox") or [0.0, 0.0, 0.0, 0.0])[1]),
            float((row.get("bbox") or [0.0, 0.0, 0.0, 0.0])[0]),
            int(row.get("physical_order_index", 0) or 0),
        ),
    )
    for source in ordered_sources:
        source_id = str(source.get("source_block_id") or "").strip()
        if not source_id or source_id in seen_source_ids:
            return None
        seen_source_ids.add(source_id)
        projected_index = projected_index_by_source_id.get(source_id)
        if projected_index is not None:
            if projected_index not in inserted:
                projected_text = str(projected_rows[projected_index].get("display_text") or "").strip()
                if projected_text and _markdown_compact_table_text(projected_text) not in note_signatures:
                    rows.append(projected_text)
                inserted.add(projected_index)
            continue
        text = str(source.get("text") or "").strip()
        if text and _markdown_compact_table_text(text) not in note_signatures:
            rows.append(text)
    if len(inserted) != len(projected_rows):
        return None
    return rows


def _collect_high_confidence_inline_formula_spans(block: dict[str, Any]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for span in block.get("inline_formula_spans", []) or []:
        if not isinstance(span, dict):
            continue
        if str(span.get("formula_complexity") or "") not in {"inline_formula", "inline_symbol"}:
            continue
        latex_text = str(span.get("latex_text") or "").strip()
        if not latex_text:
            continue
        if "$" in latex_text and len(re.findall(r"\b[A-Za-z]{3,}\b", latex_text)) >= 3:
            continue
        try:
            latex_confidence = float(span.get("latex_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            latex_confidence = 0.0
        if latex_confidence < 0.85:
            continue
        key = (str(span.get("content") or "").strip(), latex_text)
        if key in seen:
            continue
        seen.add(key)
        accepted.append(span)
    def span_complexity_priority(span: dict[str, Any]) -> int:
        if str(span.get("formula_complexity") or "").strip() == "inline_formula":
            return 0
        return 1

    def span_latex_quality_penalty(span: dict[str, Any]) -> int:
        latex_text = str(span.get("latex_text") or "").strip()
        penalty = 0
        if re.search(r"_[A-Za-z0-9]+_", latex_text):
            penalty += 8
        if "…" in latex_text:
            penalty += 4
        if re.search(r"\b[A-Za-z]\s+[A-Za-z]\b", latex_text):
            penalty += 2
        if re.search(r"(?:\\frac|\\sum|\\ldots|\\lVert|\\in|\\ge|\\le|_\{[A-Za-z0-9]+\})", latex_text):
            penalty -= 3
        return penalty

    def span_source_priority(span: dict[str, Any]) -> int:
        source = str(span.get("source") or span.get("latex_source") or "").strip()
        if "cross_block_inline_math_pattern" in source:
            return 0
        if "plain_inline_math_pattern" in source:
            return 1
        if "2d_reconstruction" in source:
            return 2
        return 2

    return sorted(
        accepted,
        key=lambda span: (
            span_complexity_priority(span),
            span_latex_quality_penalty(span),
            span_source_priority(span),
            -max(
                len(str(project_pdf_math_symbol_display_text(span.get("content") or "") or "").strip()),
                len(str(span.get("content") or "").strip()),
            ),
            str(span.get("content") or "").strip(),
            str(span.get("latex_text") or "").strip(),
        ),
    )


def _inline_formula_source_candidates(span: dict[str, Any]) -> list[str]:
    raw_source = str(span.get("raw_content") or span.get("content") or "").strip()
    display_source = str(span.get("content") or "").strip()
    latex_text = str(span.get("latex_text") or "").strip()
    projected_source = str(project_pdf_math_symbol_display_text(raw_source) or "").strip()
    candidates = [projected_source, raw_source]
    compact_source = re.sub(r"\s+", "", raw_source)
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,3}", compact_source):
        spaced_source = " ".join(compact_source)
        if "{" in latex_text or len(compact_source) <= 2:
            candidates.extend(
                [
                    compact_source,
                    str(project_pdf_math_symbol_display_text(spaced_source) or "").strip(),
                    spaced_source,
                ]
            )
    if not str(span.get("raw_content") or "").strip():
        candidates.append(display_source)
    else:
        candidates.append(display_source)

    minus_source = re.sub(r"[−-]", "-", projected_source or raw_source or display_source)
    if minus_source and minus_source not in candidates:
        candidates.append(minus_source)
    unicode_minus_source = minus_source.replace("-", "−")
    if unicode_minus_source and unicode_minus_source not in candidates:
        candidates.append(unicode_minus_source)
    compact_unary_minus_source = re.sub(r"\(\s+([−-])", r"(\1", unicode_minus_source or minus_source)
    if compact_unary_minus_source and compact_unary_minus_source not in candidates:
        candidates.append(compact_unary_minus_source)
    projected_unary_minus_source = compact_unary_minus_source.replace("−", "?")
    if projected_unary_minus_source and projected_unary_minus_source not in candidates:
        candidates.append(projected_unary_minus_source)

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
        projected_candidate = str(project_pdf_math_symbol_display_text(candidate) or "").strip()
        if projected_candidate and projected_candidate not in seen:
            seen.add(projected_candidate)
            out.append(projected_candidate)
        slash_candidate = re.sub(r"\s*[?∕/]\s*", " ∕ ", candidate).strip()
        if slash_candidate and slash_candidate not in seen:
            seen.add(slash_candidate)
            out.append(slash_candidate)
    latex_text = str(span.get("latex_text") or "").strip()
    equation_match = re.fullmatch(r"([A-Za-z]_[A-Za-z0-9{}]+)\s*=\s*([A-Za-z]_[A-Za-z0-9{}]+)", latex_text)
    if equation_match:
        markdown_equation_source = f"${equation_match.group(1)}$ = ${equation_match.group(2)}$"
        if markdown_equation_source not in seen:
            out.append(markdown_equation_source)
    return out


def _markdown_latex_replacement_text(latex_text: str) -> str:
    text = str(latex_text or "").strip()
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text)
    kernel_match = re.fullmatch(
        r"k\(d\)=exp\(([−-]?d)/([σ蟽考])\)",
        compact,
    )
    if kernel_match:
        sign = "-" if kernel_match.group(1).startswith(("−", "-")) else ""
        return rf"k(d)=\exp({sign}d/\sigma)"
    return text


def _inline_formula_bbox_overlap_ratio(a: Any, b: Any) -> float:
    try:
        left = [float(value) for value in list(a or [])[:4]]
        right = [float(value) for value in list(b or [])[:4]]
    except (TypeError, ValueError):
        return 0.0
    if len(left) < 4 or len(right) < 4:
        return 0.0
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    area_left = max(1.0, max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1]))
    area_right = max(1.0, max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1]))
    return intersection / min(area_left, area_right)


def _compact_inline_math_for_overlap(value: Any) -> str:
    text = str(project_pdf_math_symbol_display_text(value or "") or value or "").strip()
    text = text.replace(r"\le", "≤").replace(r"\ge", "≥").replace(r"\ldots", "…")
    return re.sub(r"[^A-Za-z0-9=+\-*/^_≤≥∞]+", "", text)


def _overlapping_inline_formula_source_candidates(block: dict[str, Any], span: dict[str, Any]) -> list[str]:
    source = str(span.get("source") or span.get("latex_source") or "").strip()
    if "plain_inline_math_pattern" not in source and "cross_block_inline_math_pattern" not in source:
        return []
    if str(span.get("formula_complexity") or "").strip() != "inline_formula":
        return []
    latex_text = str(span.get("latex_text") or "").strip()
    if not re.search(r"(?:=|\\sum|\\frac|\\ldots|\\left|\\right|\([^)]{12,}\))", latex_text):
        return []
    latex_atom = _compact_inline_math_for_overlap(_inline_latex_first_atom(latex_text))
    candidates: list[str] = []
    seen: set[str] = set()
    for sibling in block.get("inline_formula_spans", []) or []:
        if not isinstance(sibling, dict) or sibling is span:
            continue
        sibling_source = str(sibling.get("source") or sibling.get("latex_source") or "").strip()
        if "2d_reconstruction" not in sibling_source:
            continue
        if _inline_formula_bbox_overlap_ratio(span.get("bbox"), sibling.get("bbox")) < 0.70:
            continue
        source_text = str(project_pdf_math_symbol_display_text(sibling.get("content") or "") or "").strip()
        if not source_text or source_text in seen:
            continue
        if "$" in source_text and len(re.findall(r"\b[A-Za-z]{3,}\b", source_text)) >= 3:
            continue
        latex_compact = _compact_inline_math_for_overlap(latex_text)
        source_compact = _compact_inline_math_for_overlap(source_text)
        if latex_atom and source_compact and not source_compact.startswith(latex_atom):
            continue
        if (
            latex_compact
            and source_compact.startswith(latex_compact)
            and len(source_compact) > len(latex_compact) + 2
        ):
            continue
        seen.add(source_text)
        candidates.append(source_text)
    return candidates


def _inline_latex_first_atom(latex_text: str) -> str:
    text = str(latex_text or "")
    for match in re.finditer(
        r"(?<!\\)\b[A-Za-z](?:_\{?[A-Za-z0-9]+\}?|_[A-Za-z0-9]+)?(?:\^\{\([A-Za-z0-9]+\)\}|\^\{?[A-Za-z0-9]+\}?|\^\*)?",
        text,
    ):
        atom = match.group(0).strip()
        if atom and atom not in {"R"}:
            return atom
    return ""


def _inline_latex_atom_source_regex(atom: str) -> str:
    text = str(atom or "").strip()
    match = re.fullmatch(
        r"(?P<base>[A-Za-z])(?:_\{?(?P<sub>[A-Za-z0-9]+)\}?)?(?P<sup>\^\{\([A-Za-z0-9]+\)\}|\^\{?[A-Za-z0-9]+\}?|\^\*)?",
        text,
    )
    if not match:
        return re.escape(text)
    base = re.escape(match.group("base"))
    sub = match.group("sub")
    sup = match.group("sup")
    alternatives: list[str] = []
    if sub:
        alternatives.append(re.escape(text))
        sub_escaped = re.escape(sub)
        latex_symbol = f"{match.group('base')}_{sub}"
        alternatives.append(rf"\${re.escape(latex_symbol)}\$")
        alternatives.append(rf"\b{base}\s*{sub_escaped}\b")
    else:
        alternatives.append(rf"\b{base}\b")
    if sup:
        sup_text = sup.lstrip("^").strip("{}")
        sup_raw = re.escape(sup_text.strip("()"))
        base_patterns = alternatives[:]
        alternatives.extend(
            [
                rf"{pattern}\s*\(\s*{sup_raw}\s*\)"
                for pattern in base_patterns
            ]
        )
        alternatives.extend(
            [
                rf"{pattern}\s*\*\b" if sup == "^*" else rf"{pattern}\s*{sup_raw}\b"
                for pattern in base_patterns
            ]
        )
    return "(?:" + "|".join(alternatives) + ")"


def _inline_formula_context_source_candidates(text: str, latex_text: str) -> list[str]:
    """Infer a source substring when the AST only has a readable LaTeX projection."""
    atom = _inline_latex_first_atom(latex_text)
    if not atom:
        return []
    if not re.search(r"(?:=|\\sum|\\frac|\\ldots|\\left|\\right|\\lVert|\\in|\\ge|\\le|[_^].*[_^])", latex_text):
        return []
    start_match = None
    atom_regex = _inline_latex_atom_source_regex(atom)
    relation_tail_regex = ""
    membership_match = re.fullmatch(
        r"(?P<left>[A-Za-z]_\{?[A-Za-z0-9]+\}?)\s+\\in\s+(?P<right>[A-Za-z]_\{?[A-Za-z0-9]+\}?)",
        latex_text,
    )
    if membership_match:
        relation_tail_regex = (
            r"\s*(?:\\in|∈|in)\s*"
            + _inline_latex_atom_source_regex(membership_match.group("right"))
        )
    else:
        domain_match = re.fullmatch(
            r"(?P<left>[A-Za-z])\s+\\in\s+(?P<right>[A-Za-z])\^?\{?(?P<dim>[A-Za-z0-9]+)\}?",
            latex_text,
        )
        if domain_match:
            relation_tail_regex = (
                r"\s*(?:\\in|鈭坾in|¡Ê|in)\s*"
                + re.escape(domain_match.group("right"))
                + r"\s*"
                + re.escape(domain_match.group("dim"))
            )
    for match in re.finditer(_inline_latex_atom_source_regex(atom), text):
        previous_char = text[match.start() - 1 : match.start()]
        if previous_char and previous_char not in {" ", "\t", "\n", "(", "[", "{", ",", ";", ":", "=", "$"}:
            continue
        if previous_char in {"{", "\\"}:
            continue
        if relation_tail_regex and not re.match(re.escape(match.group(0)) + relation_tail_regex, text[match.start() :]):
            continue
        start_match = match
        break
    if not start_match:
        return []
    start = start_match.start()
    tail = text[start:]
    stop_positions: list[int] = []
    if " then " not in f" {latex_text.lower()} ":
        then_match = re.search(r",\s+then\b", tail, re.IGNORECASE)
        if then_match and then_match.start() > 0:
            stop_positions.append(then_match.start())
    if " and " not in f" {latex_text.lower()} ":
        formula_and = re.search(
            r"\s+and\s+(?:\$[^$\n]{1,80}\$|[A-Za-z][A-Za-z0-9_{}^*()\s]{0,80})\s*=",
            tail,
            re.IGNORECASE,
        )
        if formula_and and formula_and.start() > 0:
            stop_positions.append(formula_and.start())
        prose_and = re.search(
            r"\s+and\s+(?:\$[A-Za-z][A-Za-z0-9_{}^*]*\$|[A-Za-z][A-Za-z0-9_{}^*]*)\s+(?:is|are|the)\b",
            tail,
            re.IGNORECASE,
        )
        if prose_and and prose_and.start() > 0:
            stop_positions.append(prose_and.start())
    prose_tail = re.search(
        r"\s+(?:be|is|are|was|were|has|have|being)\s+(?:a|an|the|used|found|obtained|defined|determined)\b",
        tail,
        re.IGNORECASE,
    )
    if prose_tail and prose_tail.start() > 0:
        stop_positions.append(prose_tail.start())
    if not re.search(r"\b(?:with|where)\b", latex_text, re.IGNORECASE):
        explanation_tail = re.search(r"\s*,\s+(?:with|where)\b", tail, re.IGNORECASE)
        if explanation_tail and explanation_tail.start() > 0:
            stop_positions.append(explanation_tail.start() + 1)
    sentence_end = re.search(r"[.;](?:\s|$)", tail)
    if sentence_end and sentence_end.end() > 0:
        stop_positions.append(sentence_end.end())
    if not stop_positions:
        return []
    end = start + min(stop_positions)
    candidate = text[start:end].strip()
    return [candidate] if candidate else []


def _markdown_inline_math_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char != "$":
            continue
        if start is None:
            start = index
        else:
            ranges.append((start, index + 1))
            start = None
    return ranges


def _markdown_match_overlaps_inline_math(match_start: int, match_end: int, text: str) -> bool:
    for range_start, range_end in _markdown_inline_math_ranges(text):
        if match_start < range_end and match_end > range_start:
            return True
    return False


def _markdown_latex_already_inside_inline_math(latex_text: str, text: str) -> bool:
    latex = str(latex_text or "").strip()
    if not latex:
        return False
    for range_start, range_end in _markdown_inline_math_ranges(text):
        math_content = text[range_start + 1 : range_end - 1]
        if latex in math_content:
            return True
    return False


def _markdown_match_is_inside_inline_math(match_start: int, match_end: int, text: str) -> bool:
    for range_start, range_end in _markdown_inline_math_ranges(text):
        if range_start < match_start and match_end < range_end:
            return True
    return False


def _markdown_raw_relation_atom_to_latex(raw_atom: str) -> str:
    text = str(raw_atom or "").strip()
    text = text.strip(" ,.;:")
    if not text:
        return ""
    inline_match = re.fullmatch(r"\$([^$\n]+)\$", text)
    if inline_match:
        return inline_match.group(1).strip()
    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"[A-Za-z]_\{?[A-Za-z0-9]+\}?", compact):
        return compact
    spaced_symbol = re.fullmatch(r"([A-Za-z])\s+([A-Za-z0-9]{1,3})", text)
    if spaced_symbol:
        return f"{spaced_symbol.group(1)}_{spaced_symbol.group(2)}"
    if re.fullmatch(r"[A-Za-z]", compact):
        return compact
    return ""


def _repair_projected_inline_math_relations(text: str) -> str:
    """Merge relation tails left behind after symbol-level Markdown projection."""
    relation_map = {
        r"\in": r"\in",
        "∈": r"\in",
        "（": r"\in",
        "(": r"\in",
        "in": r"\in",
        r"\le": r"\le",
        "≤": r"\le",
        r"\ge": r"\ge",
        "≥": r"\ge",
    }
    relation_alternation = r"(?:\\in|∈|（|\(|\bin\b|\\le|≤|\\ge|≥)"
    inline_atom = r"\$[A-Za-z](?:_\{?[A-Za-z0-9]+\}?|[A-Za-z0-9]{0,3})\^?\{?[A-Za-z0-9]*\}?\$"
    atom = rf"(?:{inline_atom}|[A-Za-z](?:\s+|_)[A-Za-z0-9]{{1,3}}|[A-Za-z])"
    pattern = re.compile(
        rf"(?P<left>{inline_atom})\s*(?P<rel>{relation_alternation})\s*(?P<right>{atom})"
    )

    def replace(match: re.Match[str]) -> str:
        left = _markdown_raw_relation_atom_to_latex(match.group("left"))
        right = _markdown_raw_relation_atom_to_latex(match.group("right"))
        rel = relation_map.get(match.group("rel"), match.group("rel"))
        if not left or not right:
            return match.group(0)
        if re.search(r"\b(?:is|are|the|and|or|where|with)\b", right, re.IGNORECASE):
            return match.group(0)
        return f"${left} {rel} {right}$"

    previous = None
    repaired = str(text or "")
    while previous != repaired:
        previous = repaired
        repaired = pattern.sub(replace, repaired)
    return repaired


def _repair_projected_inline_math_binary_operations(text: str) -> str:
    inline_atom = r"\$[A-Za-z](?:_\{?[A-Za-z0-9]+\}?|[A-Za-z0-9]{0,3})\^?\{?[A-Za-z0-9]*\}?\$"
    raw_atom = r"[A-Za-z](?:\s+|_)[A-Za-z0-9]{1,3}"
    atom = rf"(?:{inline_atom}|{raw_atom})"
    slash = r"[∕/]"
    slash = rf"(?:{slash}|/|∕|⁄)"
    pattern = re.compile(rf"(?<![\w$])(?P<left>{atom})\s*{slash}\s*(?P<right>{atom})(?![\w$])")

    def replace(match: re.Match[str]) -> str:
        left = _markdown_raw_relation_atom_to_latex(match.group("left"))
        right = _markdown_raw_relation_atom_to_latex(match.group("right"))
        if not left or not right:
            return match.group(0)
        if "_" not in left and "_" not in right:
            return match.group(0)
        return f"${left}/{right}$"

    return pattern.sub(replace, str(text or ""))


def _repair_inline_math_segments_with_prose_cues(text: str) -> str:
    repaired = str(text or "")
    repaired = re.sub(
        r"\$\\sigma\s+\\to\s+\+\\in\s+fty,\s*we\s+have\s*([A-Za-z]_[A-Za-z0-9]\s*=\s*[A-Za-z]_[A-Za-z0-9])\$",
        lambda match: rf"$\sigma \to +\infty$, we have ${match.group(1).strip()}$",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r"\$([^$\n]*?\\le\s+i\s+\\le\s+d),\s*then\s*([A-Za-z]\^\*)\$",
        lambda match: f"${match.group(1).strip()}$, then ${match.group(2).strip()}$",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(r"(?<=\$[A-Za-z]\^\*)\s+is\s+a\s+global\s+minimizer", " is a global minimizer", repaired)
    repaired = re.sub(
        r"(?<!then\s)(?<=\.\s)(\$[A-Za-z]\^\*\$\s+is\s+a\s+global\s+minimizer)",
        r"then \1",
        repaired,
    )
    repaired = re.sub(
        r"(\$[A-Za-z](?:_\{?[A-Za-z0-9]+\}?|_[A-Za-z0-9]+)?(?:\^\{\([A-Za-z0-9]+\)\}|\^\{?[A-Za-z0-9]+\}?)\s+\\ne\s+0,\s+1\s+\\le\s+i\s+\\le\s+d\$\s+)(?=is\s+a\s+global\s+minimizer)",
        r"\1then ",
        repaired,
    )
    repaired = re.sub(
        r"\$([A-Za-z])_i\^\{\(0\)\}\s+\\ne\s+0,\s+1\s+\\le\s+i\s+\\le\s+d\$\s+then\s+is\s+a\s+global\s+minimizer",
        lambda match: rf"${match.group(1)}_i^{{(0)}} \ne 0, 1 \le i \le d$ then ${match.group(1)}^*$ is a global minimizer",
        repaired,
    )
    repaired = re.sub(
        r"\$([A-Za-z]_[A-Za-z0-9])\s*=\s*([A-Za-z]_[A-Za-z0-9])\$",
        lambda match: f"${match.group(1)} = {match.group(2)}$",
        repaired,
    )
    return repaired


def _repair_inline_math_residual_delimiters(text: str) -> str:
    repaired = str(text or "")
    if "$" not in repaired:
        return repaired
    repaired = re.sub(
        r"(?m)^[ \t]*(?:\(\s*\)|\[\s*\]|\{\s*\})[ \t]+(?=\$[^$\n]+\$)",
        "",
        repaired,
    )
    repaired = re.sub(
        r"(\$[^$\n]+\$\.)[ \t]+(?:[+\-*/=]\s*)?(?:[\)\]\}]\s*){1,6}(?=$|\n)",
        r"\1",
        repaired,
    )
    repaired = re.sub(r"(?m)^[ \t]*[“”\"']\s*$\n?", "", repaired)
    return repaired


def _repair_nested_inline_math_segments(text: str) -> str:
    repaired = str(text or "")
    def replace_nested(match: re.Match[str]) -> str:
        left = match.group(1)
        middle = match.group(2)
        right = match.group(3)
        if not middle.strip() or re.fullmatch(r"[\s,.;:]+", middle):
            return match.group(0)
        if re.search(
            r"\b(?:where|with|and|or|is|are|be|being|the|a|an|at|to|by|do|if|then|we|have|has|calling|using|via|matrix|class|sample|element)\b",
            middle,
            re.IGNORECASE,
        ):
            return match.group(0)
        if not re.search(r"(?:\\sum|\\frac|[_^{}=]|\\in|\\le|\\ge|[A-Za-z]_[A-Za-z0-9])", f"{left}{middle}{right}"):
            return match.group(0)
        return f"${left}{middle}{right}$"

    previous = None
    while previous != repaired:
        previous = repaired
        repaired = re.sub(r"\$([^$\n]*?)\$\s*([^$\n]{0,80}?)\s*\$([^$\n]*?)\$", replace_nested, repaired)
        repaired = re.sub(
            r"\$([^$\n]*?),+\s*,\s*(where|with)\s*([A-Za-z]_\{?[A-Za-z0-9]+\}?[^$\n]*)\$",
            r"$\1$, \2 $\3$",
            repaired,
            flags=re.IGNORECASE,
        )
    return repaired


def _markdown_source_candidate_sort_key(source_text: str) -> tuple[int, int, int, str]:
    text = str(source_text or "")
    math_signal_count = len(re.findall(r"(?:=|\\sum|\\frac|≤|≥|≠|\bin\b|[_^]|\$)", text))
    math_signal_count = len(re.findall(r"(?:=|\\sum|\\frac|\\le|\\ge|[_^]|\$)", text))
    prose_word_count = len(re.findall(r"\b(?:width|input|parameter|note|that|if|we|have|has|been|used|define|class|number|samples|method|methods)\b", text, re.IGNORECASE))
    return (prose_word_count, -math_signal_count, -len(text), text)


def _markdown_span_first_source_position(text: str, span: dict[str, Any]) -> int:
    positions: list[int] = []
    candidates = list(_inline_formula_source_candidates(span))
    if str(span.get("formula_complexity") or "").strip() == "inline_formula":
        candidates.extend(_inline_formula_context_source_candidates(text, str(span.get("latex_text") or "")))
    for candidate in candidates:
        if not candidate:
            continue
        index = text.find(candidate)
        if index >= 0:
            positions.append(index)
    return min(positions) if positions else len(text)


def _markdown_span_replacement_order_key(text: str, span: dict[str, Any]) -> tuple[int, int, int, int, str]:
    latex_text = str(span.get("latex_text") or "").strip()
    signal_count = len(re.findall(r"(?:=|\\sum|\\frac|\\le|\\ge|\\ne|\\in|[_^]|\{|\})", latex_text))
    source = str(span.get("source") or span.get("latex_source") or "").strip()
    source_priority = 0
    if "plain_inline_math_pattern" in source:
        source_priority = -2
    elif "cross_block_inline_math_pattern" in source:
        source_priority = -1
    elif "2d_reconstruction" in source:
        source_priority = 1
    return (
        _markdown_span_first_source_position(text, span),
        source_priority,
        -signal_count,
        -len(latex_text),
        latex_text,
    )


def _project_markdown_text_with_inline_formulas(block: dict[str, Any]) -> str:
    if (
        str(block.get("text_projection") or "").strip()
        in {"absorbed_inline_math_residue", "complexity_inline_math", "cross_block_inline_math"}
        and not str(block.get("display_text") or "").strip()
    ):
        return ""
    source_display_text = str(block.get("display_text") or block.get("text") or "").strip()
    text = str(project_pdf_math_symbol_display_text(source_display_text) or "").strip()
    text = _normalize_markdown_pdf_inline_symbol_residue(text)
    if not text:
        return ""
    if not _collect_high_confidence_inline_formula_spans(block) and re.fullmatch(r"[\s(){}\[\]*.,;:]+", text):
        return ""
    text = _repair_projected_inline_math_relations(text)
    for span in _collect_high_confidence_inline_formula_spans(block):
        if str(span.get("formula_complexity") or "").strip() != "inline_formula":
            continue
        latex_text = str(span.get("latex_text") or "").strip()
        source = str(span.get("source") or span.get("latex_source") or "").strip()
        if "cross_block_inline_math_pattern" not in source or not latex_text:
            continue
        prefix_match = re.search(r"\bwhere\s+[A-Za-z]\s+[A-Za-z]\s*=\s*\{", text)
        if not prefix_match:
            continue
        math_tail = text[prefix_match.start() :].strip()
        if "$" in math_tail:
            continue
        text = f"{text[:prefix_match.start()].rstrip()} where ${latex_text}$".strip()
        break

    ordered_spans = sorted(
        _collect_high_confidence_inline_formula_spans(block),
        key=lambda span: _markdown_span_replacement_order_key(text, span),
    )
    for span in ordered_spans:
        latex_text = str(span.get("latex_text") or "").strip()
        if not latex_text:
            continue
        latex_text = _markdown_latex_replacement_text(latex_text)
        if not latex_text:
            continue
        replacement = latex_text if "$" in latex_text else f"${latex_text}$"
        replacement_already_present = replacement in text
        if replacement_already_present and str(span.get("formula_complexity") or "").strip() == "inline_formula":
            continue
        source_candidates = [
            *_inline_formula_source_candidates(span),
            *_overlapping_inline_formula_source_candidates(block, span),
        ]
        if str(span.get("formula_complexity") or "").strip() == "inline_formula":
            source_candidates.extend(_inline_formula_context_source_candidates(text, latex_text))
        if (
            source_display_text == text
            and str(span.get("formula_complexity") or "").strip() == "inline_formula"
        ):
            source_candidates.extend(_inline_formula_context_source_candidates(source_display_text, latex_text))
        source_candidates = sorted(source_candidates, key=_markdown_source_candidate_sort_key)
        seen_sources: set[str] = set()
        for source_text in source_candidates:
            if not source_text or source_text in seen_sources:
                continue
            seen_sources.add(source_text)
            if source_text in text:
                search_start = 0
                while True:
                    match_index = text.find(source_text, search_start)
                    if match_index < 0:
                        break
                    match_end = match_index + len(source_text)
                    search_start = match_index + 1
                    if _markdown_match_overlaps_inline_math(match_index, match_end, text):
                        if _markdown_match_is_inside_inline_math(match_index, match_end, text) and source_text == replacement:
                            following_char = text[match_end : match_end + 1]
                            if following_char not in {"^", "_"} and not following_char.isdigit():
                                break
                        continue
                    if not _markdown_candidate_has_safe_boundaries(text, source_text, match_index, match_end, span):
                        continue
                    following_char = text[match_index + len(source_text) : match_index + len(source_text) + 1]
                    if following_char in {"^", "_"} or following_char.isdigit():
                        continue
                    replacement_text = replacement
                    trailing_punctuation = source_text[-1:] if source_text[-1:] in {",", ".", ";", ":"} else ""
                    if trailing_punctuation and not replacement_text.endswith(trailing_punctuation):
                        replacement_text = f"{replacement_text}{trailing_punctuation}"
                    text = f"{text[:match_index]}{replacement_text}{text[match_end:]}"
                    break
                else:
                    continue
                break
    text = re.sub(r"(\b[A-Za-z])\s+-\s+([a-z]{2,}\b)", r"\1-\2", text)
    text = _repair_inline_math_segments_with_prose_cues(text)
    text = _repair_nested_inline_math_segments(text)
    text = re.sub(r"\\in(?=[A-Za-z])", r"\\in ", text)
    text = re.sub(r"(?<=[A-Za-z0-9}])\\in", r" \\in", text)
    text = _repair_projected_inline_math_relations(text)
    text = _repair_projected_inline_math_binary_operations(text)
    text = _repair_inline_math_segments_with_prose_cues(text)
    text = re.sub(r"\bxi\b", r"$x_i$", text)
    text = re.sub(r"\$([^$\n]*?,)\$,", r"$\1$", text)
    text = re.sub(r"\$([^$\n]*?)([.;:])\$\2", r"$\1$\2", text)
    text = re.sub(r"(\$[^$\n]+\$)\s+([,.;:])", r"\1\2", text)
    text = _repair_inline_math_residual_delimiters(text)
    text = _repair_markdown_closed_formula_range_tail(text)
    text = re.sub(r"(\$[^$\n]+\$)\s+([,.;:])", r"\1\2", text)
    return _repair_markdown_text_with_contextual_inline_atoms(text)


def _normalize_markdown_pdf_inline_symbol_residue(text: str) -> str:
    repaired = str(text or "")
    repaired = repaired.replace("กค", "·")
    repaired = repaired.replace("กฐ", "")
    repaired = re.sub(r"(?<!\S)[\u201c\u201d\"'](?!\S)", "", repaired)
    repaired = re.sub(r"[ \t]{2,}", " ", repaired)
    return repaired


def _repair_markdown_text_with_contextual_inline_atoms(text: str) -> str:
    repaired = str(text or "")
    repaired = _normalize_markdown_pdf_inline_symbol_residue(repaired)
    repaired = _repair_markdown_soft_line_hyphenation(repaired)
    repaired = _repair_markdown_scientific_notation(repaired)
    repaired = _repair_markdown_closed_formula_range_tail(repaired)
    repaired = _repair_markdown_intra_line_duplicate_ocr_residue(repaired)
    if re.search(r"\b(?:classes|class|centroid|sample|element|matrix|vector|function)\b", repaired, re.IGNORECASE):
        repaired = re.sub(r"(?<![\w$])uj(?![\w$])", r"$u_j$", repaired)
        repaired = re.sub(
            r"(?<![\w$])u(?![\w$])(?=\s+(?:is|are)\s+the\s+centroid\b)",
            r"$u$",
            repaired,
            flags=re.IGNORECASE,
        )
    repaired = re.sub(r"(?<![\w$])([A-GI-Z])\s+([a-z])\s*\.", r"$\1(\2)$.", repaired)
    return repaired


def _repair_markdown_soft_line_hyphenation(text: str) -> str:
    repaired = str(text or "")

    def replace(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        tail = match.group("tail")
        continuation = match.group("continuation")
        suffix = match.group("suffix")
        if len(tail) < 3 or len(continuation) < 2:
            return match.group(0)
        if tail[:1].isupper() or continuation[:1].isupper():
            return match.group(0)
        if prefix.endswith("-"):
            return match.group(0)
        return f"{prefix}{tail}{continuation}{suffix}"

    return re.sub(
        r"(?P<prefix>(?<![A-Za-z0-9$])|(?<=\s))(?P<tail>[A-Za-z]{3,})-\s+(?P<continuation>[a-z]{2,})(?P<suffix>\b)",
        replace,
        repaired,
    )


def _repair_markdown_scientific_notation(text: str) -> str:
    repaired = str(text or "")

    def replace(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        base = match.group("base")
        exponent = match.group("exponent")
        return f"{prefix} ${base}^{{-{exponent}}}$"

    pattern = re.compile(
        r"(?i)\b(?P<prefix>(?:less\s+than|greater\s+than|smaller\s+than|larger\s+than|at\s+most|at\s+least|approximately|about|e\.g\.,?\s+less\s+than|e\.g\.,?\s+greater\s+than|<=|<|>=|>))\s+(?P<base>\d+)-(?P<exponent>\d+)\b"
    )
    repaired = pattern.sub(replace, repaired)
    return repaired


def _repair_markdown_closed_formula_range_tail(text: str) -> str:
    repaired = str(text or "")

    def replace(match: re.Match[str]) -> str:
        formula = str(match.group("formula") or "")
        inner = formula[1:-1].strip()
        if not inner or re.search(r"_[{][^}]+[}]\\^{[{][^}]+[}]", inner):
            return match.group(0)
        upper = match.group("upper")
        lower = match.group("lower")
        start = match.group("start")
        tail = str(match.group("tail") or "")
        if not lower or not upper or not start:
            return match.group(0)
        if not re.search(r"[\\}\)\]]\s*$", inner):
            return match.group(0)
        tail = re.sub(r"^\s+([,.;:])", r"\1", tail)
        return f"${inner}_{{{lower}={start}}}^{{{upper}}}${tail}"

    repaired = re.sub(
        r"(?P<formula>\$[^$\n]+\$)\s*\$?(?P<upper>[A-Za-z])(?:\s*_?\s*(?P<lower>[A-Za-z]))?\$?\s*=\s*(?P<start>\d+)\b(?P<tail>\s*[,.;:]?\s*)",
        replace,
        repaired,
    )
    return repaired


def _markdown_duplicate_word_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).casefold(), match.start(), match.end())
        for match in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text or ""))
    ]


def _looks_like_markdown_duplicate_ocr_bridge(bridge: str) -> bool:
    value = str(bridge or "")
    stripped = value.strip()
    if not stripped:
        return True
    if len(stripped) > 80:
        return False
    if re.fullmatch(r"[\s,.;:()\"'\-\d]+", value):
        return True
    bridge_words = re.findall(r"[A-Za-z0-9]+", stripped)
    if len(bridge_words) <= 1 and re.fullmatch(r"[\w\s,.;:()\"'\-]+", stripped):
        return True
    if len(bridge_words) <= 5:
        if re.search(r"\b(?:is|are|was|were|be|to|in|on|at|of|the|and|tag|entity|sentence)[a-z]{3,}\b", stripped, re.IGNORECASE):
            return True
        if re.search(r"\b[A-Z]{1,2}\b", stripped):
            return True
    return False


def _repair_markdown_intra_line_duplicate_ocr_residue(text: str) -> str:
    repaired = str(text or "")
    for _ in range(4):
        words = _markdown_duplicate_word_spans(repaired)
        if len(words) < 10:
            return repaired
        changed = False
        max_suffix_len = min(18, len(words) // 2)
        for suffix_len in range(max_suffix_len, 4, -1):
            suffix_start_word = len(words) - suffix_len
            suffix = [word for word, _, _ in words[suffix_start_word:]]
            suffix_start_char = words[suffix_start_word][1]
            for start_word in range(0, suffix_start_word - suffix_len + 1):
                candidate = [word for word, _, _ in words[start_word : start_word + suffix_len]]
                if candidate != suffix:
                    continue
                first_end_char = words[start_word + suffix_len - 1][2]
                bridge = repaired[first_end_char:suffix_start_char]
                if not _looks_like_markdown_duplicate_ocr_bridge(bridge):
                    continue
                repaired = repaired[:first_end_char].rstrip() + repaired[words[-1][2] :]
                changed = True
                break
            if changed:
                break
        if not changed:
            return repaired
    return repaired


def _markdown_block_bbox(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    return x0, y0, x1, y1


def _markdown_text_starts_bullet_item(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:(?:\d{1,4}|[ivxlcdm]{1,12})[.)]\s+|"
            r"[\u2022\u25cf\u25cb\u25aa\u25e6\u2219\uf06c\uf0b7\u00b7\x01]|[-*]\s+)",
            str(text or ""),
            re.IGNORECASE,
        )
    )


def _markdown_text_starts_cross_page_continuation_punctuation(text: str) -> bool:
    stripped = str(text or "").lstrip()
    return bool(stripped) and stripped[0] in {
        "(", "[", "{", "<",
        "\uff08", "\u3010", "\u300a", "\u3008", "\u300c", "\u300e",
        "\u201c", "\u2018", '"', "'",
        ",", ";", ":", "\uff0c", "\uff1b", "\uff1a", "\u3001",
    }


def _markdown_heading_shape_profile(text: str) -> str | None:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw or len(raw) > 96:
        return None
    heading_small_words = {
        "and", "or", "of", "the", "in", "for", "to", "a", "an", "with",
        "at", "by", "from", "on", "vs", "via",
    }
    if re.match(r"^\s*(?:activity|experiment|exercise|task)\s+\d{1,3}\s*:\s+\S", raw, re.IGNORECASE):
        return "activity_section_title"
    if re.match(r"^\s*\d{1,3}\s+[A-Z][A-Za-z0-9][^\n]{2,}$", raw):
        after_marker = re.sub(r"^\s*\d{1,3}\s+", "", raw).strip()
        if len(after_marker) > 72 or re.search(r"\b(?:if|then|when|where|because)\b", after_marker, re.IGNORECASE):
            return None
        alpha_words = re.findall(r"[A-Za-z][A-Za-z'/-]*", after_marker)
        if not alpha_words:
            return None
        title_words = [
            word for word in alpha_words
            if word[:1].isupper() or word.lower() in heading_small_words
        ]
        if len(title_words) == len(alpha_words):
            return "numbered_section_title"
    if re.match(r"^\s*\d{1,3}(?:\.\d{1,3})*\.?\s+[A-Z][^\n]{2,}$", raw):
        after_marker = re.sub(r"^\s*\d{1,3}(?:\.\d{1,3})*\.?\s+", "", raw).strip()
        if len(after_marker) > 72 or re.search(r"\b(?:if|then|when|where|because|the|a|an)\b", after_marker, re.IGNORECASE):
            return None
        return "numbered_section_title"
    if _markdown_text_starts_bullet_item(raw):
        return None
    if re.fullmatch(r"\d{1,3}", raw):
        return "chapter_number"
    if re.fullmatch(r"(?i)chapter\s+\d+\.?", raw):
        return "chapter_label"
    compact = re.sub(r"[^a-z]", "", raw.lower())
    if compact in {"contents", "tableofcontents"}:
        return "contents_title"
    if raw.endswith(":") and len(raw) <= 80 and re.search(r"[A-Za-z\u4e00-\u9fff]", raw):
        return "colon_title"
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", raw)
    visible_chars = re.sub(r"\s+", "", raw)
    if (
        len(cjk_chars) >= 4
        and len(visible_chars) <= 36
        and len(cjk_chars) / max(1, len(visible_chars)) >= 0.55
        and not re.search(r"[。！？!?；;，,、：:]", raw)
        and not re.search(r"https?://|www\.", raw, re.IGNORECASE)
    ):
        return "cjk_unnumbered_title"
    terminal_sentence = bool(re.search(r"[.!?]\s*$", raw))
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not words or len(words) > 10:
        return None
    alpha_words = re.findall(r"[A-Za-z][A-Za-z'/-]*", raw)
    if not alpha_words:
        return None
    upper_words = [word for word in alpha_words if word.upper() == word and len(word) >= 2]
    if len(upper_words) >= max(1, len(alpha_words) - 1):
        return "uppercase_title"
    title_words = [
        word for word in alpha_words
        if word[:1].isupper() or word.lower() in heading_small_words
    ]
    if len(title_words) == len(alpha_words):
        if terminal_sentence:
            return "titlecase_sentence_title"
        return "titlecase_title"
    if (
        not terminal_sentence
        and raw[:1].isupper()
        and len(raw) <= 88
        and len(alpha_words) <= 12
        and _markdown_text_titlecase_ratio(raw) >= 0.25
    ):
        return "sentence_visual_title"
    return None


def _markdown_text_alpha_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z'/-]*", str(text or "")))


def _markdown_text_titlecase_ratio(text: str) -> float:
    alpha_words = re.findall(r"[A-Za-z][A-Za-z'/-]*", str(text or ""))
    if not alpha_words:
        return 0.0
    title_like = [
        word for word in alpha_words
        if word[:1].isupper() or word.lower() in {
            "and", "or", "of", "the", "in", "for", "to", "a", "an", "with",
            "at", "by", "from", "on", "vs", "via",
        }
    ]
    return len(title_like) / max(1, len(alpha_words))


def _markdown_numbered_step_marker(text: str) -> tuple[int, str] | None:
    match = re.match(r"^\s*(?P<number>\d{1,3})[.)]\s+(?P<body>\S.*)$", str(text or "").strip())
    if not match:
        return None
    try:
        number = int(match.group("number"))
    except (TypeError, ValueError):
        return None
    return number, str(match.group("body") or "").strip()


def _markdown_numbered_heading_looks_like_body_step(text: str) -> bool:
    marker = _markdown_numbered_step_marker(text)
    if marker is None:
        return False
    _, body = marker
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", body)
    if len(words) < 2:
        return False
    if re.search(r"[.;!?]\s*$", body):
        return True
    if len(words) >= 3 and _markdown_text_titlecase_ratio(body) < 0.70:
        return True
    return False


def _markdown_blocks_form_compact_numbered_step_sequence(
    blocks: list[dict[str, Any]],
    index: int,
) -> bool:
    block = blocks[index]
    text = str(block.get("display_text") or block.get("text") or "").strip()
    marker = _markdown_numbered_step_marker(text)
    if marker is None or not _markdown_numbered_heading_looks_like_body_step(text):
        return False
    current_number, _ = marker
    bbox = _markdown_block_bbox(block)
    if bbox is None:
        return False
    height = max(1.0, bbox[3] - bbox[1])

    def neighbor_matches(neighbor: dict[str, Any] | None) -> bool:
        if not isinstance(neighbor, dict):
            return False
        if str(neighbor.get("block_type") or "").strip().lower() != "text":
            return False
        neighbor_text = str(neighbor.get("display_text") or neighbor.get("text") or "").strip()
        neighbor_marker = _markdown_numbered_step_marker(neighbor_text)
        if neighbor_marker is None or not _markdown_numbered_heading_looks_like_body_step(neighbor_text):
            return False
        neighbor_number, _ = neighbor_marker
        if abs(neighbor_number - current_number) != 1:
            return False
        neighbor_bbox = _markdown_block_bbox(neighbor)
        if neighbor_bbox is None:
            return False
        neighbor_height = max(1.0, neighbor_bbox[3] - neighbor_bbox[1])
        if abs(neighbor_bbox[0] - bbox[0]) > max(18.0, min(height, neighbor_height) * 1.4):
            return False
        vertical_gap = max(bbox[1] - neighbor_bbox[3], neighbor_bbox[1] - bbox[3])
        return -max(height, neighbor_height) * 0.35 <= vertical_gap <= max(28.0, max(height, neighbor_height) * 2.4)

    previous_text = None
    for previous in reversed(blocks[:index]):
        if str(previous.get("block_type") or "").strip().lower() == "text":
            previous_text = previous
            break
    next_text = None
    for following in blocks[index + 1:]:
        if str(following.get("block_type") or "").strip().lower() == "text":
            next_text = following
            break
    if neighbor_matches(previous_text) or neighbor_matches(next_text):
        return True

    following_bbox = _markdown_block_bbox(next_text or {})
    if following_bbox is not None:
        following_text = str((next_text or {}).get("display_text") or (next_text or {}).get("text") or "").strip()
        vertical_gap = following_bbox[1] - bbox[3]
        continuation_indent = following_bbox[0] >= bbox[0] + max(8.0, height * 0.45)
        same_column_continuation = abs(following_bbox[0] - bbox[0]) <= max(6.0, height * 0.5)
        if (
            0 <= vertical_gap <= max(18.0, height * 1.4)
            and (continuation_indent or same_column_continuation)
            and (
                _starts_like_markdown_body_continuation(following_text)
                or _looks_like_markdown_fill_in_line(following_text)
            )
        ):
            return True
    previous_bbox = _markdown_block_bbox(previous_text or {})
    if previous_bbox is not None:
        previous_text_value = str((previous_text or {}).get("display_text") or (previous_text or {}).get("text") or "").strip()
        previous_gap = bbox[1] - previous_bbox[3]
        if (
            0 <= previous_gap <= max(28.0, height * 2.0)
            and re.search(r"(?i)\b(?:procedure|steps?|activity|experiment|method|instructions?)\s*:?\s*$", previous_text_value)
        ):
            return True
    if _markdown_block_in_numbered_step_region(blocks, index):
        return True
    return False


def _starts_like_markdown_body_continuation(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if _markdown_text_starts_bullet_item(raw):
        return False
    first = re.match(r"[A-Za-z]+", raw)
    if first and first.group(0)[:1].islower():
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if len(words) >= 4 and _markdown_text_titlecase_ratio(raw) < 0.55:
        return True
    return bool(re.search(r"[.;]\s*$", raw) and len(words) >= 3)


def _looks_like_markdown_fill_in_line(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    return bool(re.fullmatch(r"[_\-\s]{8,}", raw))


def _markdown_block_in_numbered_step_region(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    bbox = _markdown_block_bbox(block)
    if bbox is None:
        return False
    text = str(block.get("display_text") or block.get("text") or "").strip()
    marker = _markdown_numbered_step_marker(text)
    if marker is None:
        return False
    current_number, _ = marker
    height = max(1.0, bbox[3] - bbox[1])
    step_numbers: list[int] = [current_number]
    context_seen = False
    for direction in (-1, 1):
        scan = list(enumerate(blocks[:index]))
        if direction > 0:
            scan = list(enumerate(blocks[index + 1:], start=index + 1))
        else:
            scan = list(reversed(scan))
        skipped_continuation_lines = 0
        for _, candidate in scan[:14]:
            if str(candidate.get("block_type") or "").strip().lower() != "text":
                continue
            candidate_text = str(candidate.get("display_text") or candidate.get("text") or "").strip()
            candidate_bbox = _markdown_block_bbox(candidate)
            if candidate_bbox is None or not candidate_text:
                continue
            vertical_distance = abs(candidate_bbox[1] - bbox[1])
            if vertical_distance > max(180.0, height * 11.0):
                break
            if re.search(r"(?i)\b(?:procedure|steps?|activity|experiment|method|instructions?)\s*:?\s*$", candidate_text):
                context_seen = True
                continue
            candidate_marker = _markdown_numbered_step_marker(candidate_text)
            if candidate_marker is not None and _markdown_numbered_heading_looks_like_body_step(candidate_text):
                candidate_number, _ = candidate_marker
                if abs(candidate_number - current_number) <= 6 and abs(candidate_bbox[0] - bbox[0]) <= max(22.0, height * 1.8):
                    step_numbers.append(candidate_number)
                    continue
            if (
                _starts_like_markdown_body_continuation(candidate_text)
                or _looks_like_markdown_fill_in_line(candidate_text)
            ):
                skipped_continuation_lines += 1
                if skipped_continuation_lines <= 5:
                    continue
            break
    unique_numbers = sorted(set(step_numbers))
    has_neighbor_number = any(abs(number - current_number) <= 2 and number != current_number for number in unique_numbers)
    return (context_seen and has_neighbor_number) or len(unique_numbers) >= 2


def _markdown_page_looks_like_landscape_visual_panel(
    blocks: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> bool:
    if page_width <= 0 or page_height <= 0 or page_width < page_height * 1.18:
        return False
    text_blocks = [
        block for block in blocks
        if str(block.get("block_type") or "").strip().lower() == "text"
        and str(block.get("display_text") or block.get("text") or "").strip()
    ]
    if len(text_blocks) < 6:
        return False
    bboxes = [_markdown_block_bbox(block) for block in text_blocks]
    bboxes = [bbox for bbox in bboxes if bbox is not None]
    if len(bboxes) < max(4, int(len(text_blocks) * 0.6)):
        return False
    y_bands: list[float] = []
    x_centers: list[float] = []
    for x0, y0, x1, y1 in bboxes:
        y_bands.append(round(((y0 + y1) / 2.0) / 24.0))
        x_centers.append((x0 + x1) / 2.0)
    repeated_band_count = max((y_bands.count(band) for band in set(y_bands)), default=0)
    x_span = max(x_centers) - min(x_centers) if x_centers else 0.0
    short_title_like = 0
    for block in text_blocks:
        text = str(block.get("display_text") or block.get("text") or "").strip()
        if len(text) <= 80 and _markdown_text_alpha_word_count(text) <= 7 and _markdown_text_titlecase_ratio(text) >= 0.75:
            short_title_like += 1
    return repeated_band_count >= 3 and x_span >= page_width * 0.45 and short_title_like >= 3


def _markdown_block_has_nearby_image_context(
    block: dict[str, Any],
    blocks: list[dict[str, Any]],
) -> bool:
    bbox = _markdown_block_bbox(block)
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    for candidate in blocks:
        if str(candidate.get("block_type") or "").strip().lower() != "image":
            continue
        image_bbox = _markdown_block_bbox(candidate)
        if image_bbox is None:
            continue
        ix0, iy0, ix1, iy1 = image_bbox
        horizontal_overlap = min(x1, ix1) - max(x0, ix0)
        horizontal_overlap_ratio = horizontal_overlap / max(1.0, min(x1 - x0, ix1 - ix0))
        vertical_gap = min(abs(iy0 - y1), abs(y0 - iy1))
        if horizontal_overlap_ratio >= 0.25 and vertical_gap <= 40.0:
            return True
    return False


def _markdown_texts_form_parenthetical_date_continuation(left_text: str, right_text: str) -> bool:
    left = re.sub(r"\s+", " ", str(left_text or "").strip())
    right = re.sub(r"\s+", " ", str(right_text or "").strip())
    if not left or not right:
        return False
    if len(right) > 40:
        return False
    open_parens = left.count("(") + left.count("（")
    closed_parens = left.count(")") + left.count("）")
    if open_parens <= closed_parens:
        return False
    if not re.search(r"[)）]\s*$", right):
        return False
    if re.match(r"^(?:\d{1,2}\s*)?月\s*\d{1,2}\s*日", right):
        return True
    if re.match(r"^\d{1,2}\s*日", right):
        return True
    if re.search(r"(?:19|20)\d{2}\s*年\s*\d{1,2}\s*$", left) and re.match(r"^\d{1,2}\s*月", right):
        return True
    if re.search(r"(?:19|20)\d{2}\s*年\s*$", left) and re.match(r"^\d{1,2}\s*月", right):
        return True
    return False


def _markdown_texts_form_parenthetical_continuation(left_text: str, right_text: str) -> bool:
    left = re.sub(r"\s+", " ", str(left_text or "").strip())
    right = re.sub(r"\s+", " ", str(right_text or "").strip())
    if not left or not right:
        return False
    if _markdown_texts_form_parenthetical_date_continuation(left, right):
        return True
    if len(right) > 40:
        return False
    open_parens = left.count("(") + left.count("（")
    closed_parens = left.count(")") + left.count("）")
    if open_parens <= closed_parens:
        return False
    if not re.search(r"[)）]\s*$", right):
        return False
    if _markdown_text_starts_bullet_item(right):
        return False
    if re.match(r"^(?:\d+(?:\.\d+){1,}|[A-Z]?\d+[.)、]|第\s*\d+\s*[章节条]|附录|表|图|Table|Figure)\b", right):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", right))


def _markdown_block_is_parenthetical_continuation(
    previous_block: dict[str, Any] | None,
    block: dict[str, Any],
    *,
    allow_cross_page: bool = False,
) -> bool:
    if previous_block is None:
        return False
    if str(previous_block.get("block_type") or "").strip().lower() != "text":
        return False
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    previous_text = str(previous_block.get("display_text") or previous_block.get("text") or "").strip()
    text = str(block.get("display_text") or block.get("text") or "").strip()
    if not _markdown_texts_form_parenthetical_continuation(previous_text, text):
        return False
    previous_bbox = _markdown_block_bbox(previous_block)
    bbox = _markdown_block_bbox(block)
    if previous_bbox is None or bbox is None:
        return True
    previous_page = int(previous_block.get("page", 0) or previous_block.get("page_number", 0) or 0)
    page = int(block.get("page", 0) or block.get("page_number", 0) or 0)
    if allow_cross_page and previous_page > 0 and page == previous_page + 1:
        return previous_bbox[3] >= 600.0 and bbox[1] <= 160.0
    previous_height = max(1.0, previous_bbox[3] - previous_bbox[1])
    height = max(1.0, bbox[3] - bbox[1])
    vertical_gap = bbox[1] - previous_bbox[3]
    return -max(previous_height, height) * 0.25 <= vertical_gap <= max(18.0, max(previous_height, height) * 1.5)


def _markdown_block_is_cross_page_body_continuation(
    previous_block: dict[str, Any] | None,
    block: dict[str, Any],
    *,
    page_height: float,
    toc_heading_lookup: dict[str, list[dict[str, Any]]],
) -> bool:
    if previous_block is None:
        return False
    if str(previous_block.get("block_type") or "").strip().lower() != "text":
        return False
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    previous_page = int(previous_block.get("page", 0) or previous_block.get("page_number", 0) or 0)
    current_page = int(block.get("page", 0) or block.get("page_number", 0) or 0)
    if previous_page <= 0 or current_page != previous_page + 1:
        return False
    previous_text = str(previous_block.get("display_text") or previous_block.get("text") or "").strip()
    text = str(block.get("display_text") or block.get("text") or "").strip()
    if not previous_text or not text:
        return False
    if _markdown_text_block_is_unmarked_list_item(previous_block) or _markdown_text_block_is_unmarked_list_item(block):
        return False
    if _markdown_text_starts_bullet_item(previous_text) or _markdown_text_starts_bullet_item(text):
        return False
    if (
        _markdown_heading_shape_profile(text) is not None
        and not _markdown_text_starts_cross_page_continuation_punctuation(text)
    ):
        return False
    previous_boundary = _markdown_text_block_boundary_kind(previous_block, toc_heading_lookup)
    current_boundary = _markdown_text_block_boundary_kind(block, toc_heading_lookup)
    if previous_boundary != "body" or current_boundary != "body":
        return False
    if re.search(r"[.!?。！？；;：:]\s*[)）\]\}】》」』”\"']*\s*$", previous_text):
        return False
    previous_bbox = _markdown_block_bbox(previous_block)
    bbox = _markdown_block_bbox(block)
    if previous_bbox is None or bbox is None:
        return False
    previous_page_height = float(previous_block.get("_markdown_page_height", 0.0) or 0.0)
    previous_near_bottom = (
        previous_bbox[3] >= previous_page_height * 0.84
        if previous_page_height > 0
        else previous_bbox[3] >= 600.0
    )
    current_near_top = (
        bbox[1] <= page_height * 0.20
        if page_height > 0
        else bbox[1] <= 160.0
    )
    if not previous_near_bottom or not current_near_top:
        return False
    if bbox[0] - previous_bbox[0] > 10.0:
        return False
    if abs(bbox[0] - previous_bbox[0]) > 42.0:
        return False
    return True


def _markdown_block_is_parenthetical_date_continuation(
    previous_block: dict[str, Any] | None,
    block: dict[str, Any],
) -> bool:
    if previous_block is None:
        return False
    previous_text = str(previous_block.get("display_text") or previous_block.get("text") or "").strip()
    text = str(block.get("display_text") or block.get("text") or "").strip()
    if not _markdown_texts_form_parenthetical_date_continuation(previous_text, text):
        return False
    return _markdown_block_is_parenthetical_continuation(previous_block, block)


def _looks_like_markdown_visual_standalone_heading(
    block: dict[str, Any],
    previous_block: dict[str, Any] | None,
    next_block: dict[str, Any] | None,
    page_height: float,
    *,
    toc_like_page: bool = False,
    page_blocks: list[dict[str, Any]] | None = None,
    landscape_visual_panel: bool = False,
) -> bool:
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    role = str(block.get("semantic_role") or "").strip()
    if role in {
        "toc_entry",
        "reference_entry",
        "footnote",
        "footnote_continuation",
        "publication_footer",
        "page_number",
        "citation_metadata",
    }:
        return False
    text = str(block.get("display_text") or block.get("text") or "").strip()
    profile = _markdown_heading_shape_profile(text)
    if profile is None:
        return False
    if _markdown_block_is_parenthetical_continuation(previous_block, block, allow_cross_page=True):
        return False
    if toc_like_page:
        return profile == "contents_title"
    if role == "body_list_item" and profile != "numbered_section_title":
        return False
    bbox = _markdown_block_bbox(block)
    if bbox is None:
        return profile in {"contents_title", "chapter_label", "colon_title"}
    x0, y0, x1, y1 = bbox
    height = max(1.0, y1 - y0)
    if page_height > 0 and (y0 <= page_height * 0.075 or y0 >= page_height * 0.88):
        return False
    previous_bbox = _markdown_block_bbox(previous_block or {})
    next_bbox = _markdown_block_bbox(next_block or {})
    previous_gap = y0 - previous_bbox[3] if previous_bbox is not None else float("inf")
    next_gap = next_bbox[1] - y1 if next_bbox is not None else float("inf")
    near_page_top = page_height > 0 and y0 <= page_height * 0.28
    visually_separated_before = previous_bbox is None or previous_gap >= max(8.0, height * 0.65)
    visually_separated_after = next_bbox is None or next_gap >= max(5.0, height * 0.35)
    previous_same_visual_row = (
        previous_bbox is not None
        and abs(((previous_bbox[1] + previous_bbox[3]) / 2.0) - ((y0 + y1) / 2.0)) <= max(8.0, height * 0.45)
    )
    next_same_visual_row = (
        next_bbox is not None
        and abs(((next_bbox[1] + next_bbox[3]) / 2.0) - ((y0 + y1) / 2.0)) <= max(8.0, height * 0.45)
    )
    if profile == "chapter_number":
        next_text = str((next_block or {}).get("display_text") or (next_block or {}).get("text") or "").strip()
        if page_height > 0 and y0 <= page_height * 0.10:
            return False
        return near_page_top and _markdown_heading_shape_profile(next_text) in {"titlecase_title", "uppercase_title"}
    if profile in {"contents_title", "chapter_label"}:
        return near_page_top or visually_separated_before
    if profile == "numbered_section_title":
        compact_section_break_before = previous_bbox is not None and previous_gap >= max(8.0, height * 0.75)
        if not (visually_separated_before or compact_section_break_before):
            return False
        if near_page_top:
            return True
        return visually_separated_after or (next_gap >= max(7.0, height * 0.55))
    if profile == "activity_section_title":
        return visually_separated_before and visually_separated_after
    if profile == "colon_title":
        return visually_separated_before or visually_separated_after
    if profile == "cjk_unnumbered_title":
        if previous_same_visual_row or next_same_visual_row:
            return False
        text_width = x1 - x0
        next_role = str((next_block or {}).get("semantic_role") or "").strip()
        next_text = str((next_block or {}).get("display_text") or (next_block or {}).get("text") or "").strip()
        next_bbox = _markdown_block_bbox(next_block or {})
        next_looks_like_wide_body_line = (
            next_bbox is not None
            and next_role in {"", "text_block", "body"}
            and (next_bbox[2] - next_bbox[0]) >= 340.0
        )
        next_is_body_or_section_boundary = (
            not next_text
            or next_role in {"section_heading", "body_list_item"}
            or _markdown_text_starts_bullet_item(next_text)
            or _markdown_heading_shape_profile(next_text) is None
            or next_looks_like_wide_body_line
        )
        compact_or_indented_title = len(text) <= 28 or text_width <= 320.0
        return (
            compact_or_indented_title
            and next_is_body_or_section_boundary
            and (
                near_page_top
                or visually_separated_before
                or visually_separated_after
            )
        )
    if profile == "titlecase_sentence_title":
        if _markdown_block_has_nearby_image_context(block, page_blocks or []):
            return visually_separated_before or visually_separated_after
        if landscape_visual_panel and len(text) <= 88 and (near_page_top or visually_separated_before):
            return True
        return False
    if profile == "sentence_visual_title":
        if landscape_visual_panel and len(text) <= 88 and (near_page_top or visually_separated_before):
            return True
        return False
    if profile in {"uppercase_title", "titlecase_title"}:
        if page_height > 0 and y0 <= page_height * 0.08:
            return False
        if landscape_visual_panel:
            text_width = x1 - x0
            if (
                len(text) <= 88
                and _markdown_text_alpha_word_count(text) <= 9
                and text_width <= 560.0
                and (
                    visually_separated_before
                    or visually_separated_after
                    or previous_same_visual_row
                    or next_same_visual_row
                )
            ):
                return True
        if previous_block is None and near_page_top and len(text) <= 64:
            return True
        next_text = str((next_block or {}).get("display_text") or (next_block or {}).get("text") or "").strip()
        if (
            visually_separated_before
            and len(text) <= 48
            and _markdown_text_starts_bullet_item(next_text)
        ):
            return True
        return (near_page_top and visually_separated_after) or (visually_separated_before and visually_separated_after)
    return False


def _mark_markdown_centered_front_matter_title_clusters(
    page_blocks: list[dict[str, Any]],
    *,
    page_width: float,
    page_height: float,
    page_previous_flow_block: dict[str, Any] | None = None,
    toc_like_page: bool = False,
    landscape_visual_panel: bool = False,
) -> list[dict[str, Any]]:
    if page_width <= 0 or page_height <= 0 or toc_like_page:
        return page_blocks
    updated_blocks = list(page_blocks)
    for index, block in enumerate(page_blocks):
        if not _markdown_block_is_centered_front_matter_terminal_heading(
            block,
            page_width=page_width,
            page_height=page_height,
        ):
            continue
        previous_block = page_blocks[index - 1] if index > 0 else page_previous_flow_block
        next_block = page_blocks[index + 1] if index + 1 < len(page_blocks) else None
        if not _looks_like_markdown_visual_standalone_heading(
            block,
            previous_block,
            next_block,
            page_height,
            toc_like_page=False,
            page_blocks=page_blocks,
            landscape_visual_panel=landscape_visual_panel,
        ):
            continue
        cluster_indexes: list[int] = []
        scan_index = index - 1
        while scan_index >= 0:
            candidate = page_blocks[scan_index]
            if not _markdown_block_is_centered_front_matter_title_cluster_line(
                candidate,
                page_width=page_width,
                page_height=page_height,
            ):
                break
            lower_block = page_blocks[cluster_indexes[-1]] if cluster_indexes else block
            if not _markdown_centered_front_matter_lines_are_contiguous(candidate, lower_block):
                break
            cluster_indexes.append(scan_index)
            scan_index -= 1
        if len(cluster_indexes) < 2:
            continue
        for cluster_index in cluster_indexes:
            updated_blocks[cluster_index] = {
                **updated_blocks[cluster_index],
                "_markdown_centered_front_matter_title_cluster": True,
            }
    return updated_blocks


def _markdown_block_is_centered_front_matter_terminal_heading(
    block: dict[str, Any],
    *,
    page_width: float,
    page_height: float,
) -> bool:
    if not _markdown_block_is_centered_front_matter_title_cluster_line(
        block,
        page_width=page_width,
        page_height=page_height,
    ):
        return False
    bbox = _markdown_block_bbox(block)
    if bbox is None:
        return False
    return bbox[1] <= page_height * 0.36


def _markdown_block_is_centered_front_matter_title_cluster_line(
    block: dict[str, Any],
    *,
    page_width: float,
    page_height: float,
) -> bool:
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    role = str(block.get("semantic_role") or "").strip()
    if role in {
        "section_heading",
        "toc_entry",
        "reference_entry",
        "footnote",
        "footnote_continuation",
        "publication_footer",
        "page_number",
        "citation_metadata",
    }:
        return False
    text = str(block.get("display_text") or block.get("text") or "").strip()
    if not text or len(text) > 88:
        return False
    if _markdown_text_starts_bullet_item(text):
        return False
    if re.search(r"[,.!?;\uff0c\u3002\uff01\uff1f\uff1b]\s*$", text):
        return False
    if re.search(r"https?://|www\.", text, re.IGNORECASE):
        return False
    bbox = _markdown_block_bbox(block)
    if bbox is None:
        return False
    x0, y0, x1, _y1 = bbox
    width = x1 - x0
    if width <= 0:
        return False
    if y0 <= page_height * 0.055 or y0 >= page_height * 0.45:
        return False
    if width >= page_width * 0.72:
        return False
    page_center = page_width / 2.0
    block_center = (x0 + x1) / 2.0
    if abs(block_center - page_center) > max(18.0, page_width * 0.08):
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", text))


def _markdown_centered_front_matter_lines_are_contiguous(
    upper_block: dict[str, Any],
    lower_block: dict[str, Any],
) -> bool:
    upper_bbox = _markdown_block_bbox(upper_block)
    lower_bbox = _markdown_block_bbox(lower_block)
    if upper_bbox is None or lower_bbox is None:
        return False
    gap = lower_bbox[1] - upper_bbox[3]
    upper_height = max(1.0, upper_bbox[3] - upper_bbox[1])
    lower_height = max(1.0, lower_bbox[3] - lower_bbox[1])
    return 0.0 <= gap <= max(42.0, (upper_height + lower_height) * 1.35)


def _looks_like_markdown_toc_like_page(blocks: list[dict[str, Any]]) -> bool:
    text_blocks = [
        block for block in blocks
        if str(block.get("block_type") or "").strip().lower() == "text"
    ]
    if not text_blocks:
        return False
    texts = [str(block.get("display_text") or block.get("text") or "").strip() for block in text_blocks]
    has_contents_title = any(
        re.sub(r"[^a-z]", "", text.lower()) in {"contents", "tableofcontents"}
        for text in texts[:5]
    )
    if not has_contents_title:
        return False
    locator_count = sum(
        1
        for text in texts
        if re.fullmatch(r"(?:[ivxlcdm]{1,8}|\d{1,4})", text.strip(), re.IGNORECASE)
    )
    entry_like_count = sum(
        1
        for text in texts
        if re.search(r"[A-Za-z]{3,}", text)
        and not re.fullmatch(r"(?i)contents|table\s+of\s+contents", text.strip())
    )
    inline_locator_count = sum(
        1
        for text in texts
        if re.search(r"\s(?:[ivxlcdm]{1,8}|\d{1,4})$", text.strip(), re.IGNORECASE)
    )
    return (locator_count >= 2 and entry_like_count >= 2) or inline_locator_count >= 2


def _markdown_reference_entry_start_label(text: str) -> str:
    match = re.match(
        r"^\s*\[?(?P<label>\d{1,4}|[ivxlcdm]{1,12})\]?[.)]\s+\S",
        str(text or ""),
        re.IGNORECASE,
    )
    return str(match.group("label") or "").strip().lower() if match else ""


def _markdown_reference_entry_identity(block: dict[str, Any]) -> tuple[str, str] | None:
    if str(block.get("semantic_role") or "").strip() != "reference_entry":
        return None
    index = str(block.get("reference_entry_index") or "").strip()
    number = str(block.get("reference_number") or "").strip()
    attributes = block.get("attributes")
    if isinstance(attributes, dict):
        index = index or str(attributes.get("reference_entry_index") or "").strip()
        number = number or str(attributes.get("reference_number") or "").strip()
    text = str(block.get("display_text") or block.get("text") or "")
    if not number:
        number = _markdown_reference_entry_start_label(text)
    if not index and not number:
        return ("", "")
    return (index, number)


def _markdown_reference_entry_start_flag(block: dict[str, Any]) -> bool | None:
    if str(block.get("semantic_role") or "").strip() != "reference_entry":
        return None
    if "reference_entry_start" in block:
        return bool(block.get("reference_entry_start"))
    attributes = block.get("attributes")
    if isinstance(attributes, dict) and "reference_entry_start" in attributes:
        return bool(attributes.get("reference_entry_start"))
    if "reference_continuation" in block:
        return not bool(block.get("reference_continuation"))
    if isinstance(attributes, dict) and "reference_continuation" in attributes:
        return not bool(attributes.get("reference_continuation"))
    return None


def _can_merge_markdown_reference_entry_blocks(
    current_block: dict[str, Any],
    next_block: dict[str, Any],
    current_text: str,
    next_text: str,
    paragraph_base_x0: float | None,
) -> bool | None:
    current_is_reference = str(current_block.get("semantic_role") or "").strip() == "reference_entry"
    next_is_reference = str(next_block.get("semantic_role") or "").strip() == "reference_entry"
    if not current_is_reference and not next_is_reference:
        return None
    if not current_is_reference or not next_is_reference:
        return False

    current_identity = _markdown_reference_entry_identity(current_block)
    next_identity = _markdown_reference_entry_identity(next_block)
    current_start_label = _markdown_reference_entry_start_label(current_text)
    next_start_label = _markdown_reference_entry_start_label(next_text)
    same_known_reference_entry = (
        current_identity is not None
        and next_identity is not None
        and current_identity != ("", "")
        and current_identity == next_identity
    )
    next_start_flag = _markdown_reference_entry_start_flag(next_block)

    if next_start_label:
        if same_known_reference_entry and next_start_flag is False:
            return _same_markdown_text_flow(current_block, next_block, paragraph_base_x0)
        if current_start_label and current_start_label == next_start_label:
            return _same_markdown_text_flow(current_block, next_block, paragraph_base_x0)
        return False
    if current_identity and next_identity and current_identity != ("", "") and next_identity != ("", ""):
        if current_identity != next_identity:
            return False
    return _same_markdown_text_flow(current_block, next_block, paragraph_base_x0)


def _is_markdown_literature_standalone_label_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.search(r"\d", raw):
        return False
    if re.search(r"[.;,]", raw):
        return False
    compact = re.sub(r"[^a-z]", "", raw.lower())
    return compact in {"highlights", "articleinfo", "abstract", "references"}


_MARKDOWN_LITERATURE_SECTION_HEADING_NORMALIZED = {
    "introduction",
    "background",
    "relatedwork",
    "literaturereview",
    "methods",
    "method",
    "materials",
    "materialsandmethods",
    "methodology",
    "model",
    "models",
    "experiments",
    "experiment",
    "experimentalsetup",
    "results",
    "discussion",
    "resultsanddiscussion",
    "conclusion",
    "conclusions",
    "limitations",
    "futurework",
    "acknowledgements",
    "acknowledgments",
    "funding",
    "declarations",
    "authorcontributions",
    "authorscontributions",
    "authordetails",
    "competinginterests",
    "ethicsapproval",
    "consentforpublication",
    "availabilityofdataandmaterials",
    "traditionalpipelinemethods",
    "jointextractionmethods",
    "tablebasedmethods",
    "pointernetworkbasedmethods",
    "sequenceannotationbasedmethods",
    "methodsinthemedicalfield",
    "analysisonmodelefficiency",
}


def _is_markdown_literature_section_heading_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if len(raw) > 96:
        return False
    if re.search(r"[.;,!?]\s*$", raw):
        return False
    if re.search(r"\[[0-9,\s-]+\]", raw):
        return False
    if re.search(r"\b(?:19|20)\d{2}\b", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*", raw)
    if not (1 <= len(words) <= 6):
        return False
    compact = re.sub(r"[^a-z]", "", raw.lower())
    if compact in _MARKDOWN_LITERATURE_SECTION_HEADING_NORMALIZED:
        return True
    if len(words) <= 4 and any(word.lower() in {"methods", "method", "results", "discussion", "analysis"} for word in words):
        titlecase_words = sum(1 for word in words if word[:1].isupper() or word.islower())
        return titlecase_words == len(words)
    return False


def _is_markdown_literature_standalone_label_block(block: dict[str, Any]) -> bool:
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = str(block.get("display_text") or block.get("text") or "").strip()
    return _is_markdown_literature_standalone_label_text(text)


def _is_markdown_literature_section_heading_block(block: dict[str, Any]) -> bool:
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    role = str(block.get("semantic_role") or "").strip()
    if role in {"reference_entry", "footnote", "footnote_continuation", "publication_footer"}:
        return False
    text = str(block.get("display_text") or block.get("text") or "").strip()
    return _is_markdown_literature_section_heading_text(text)


def _is_markdown_body_citation_text_block(block: dict[str, Any]) -> bool:
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    if str(block.get("semantic_role") or "").strip() != "citation_metadata":
        return False
    text = str(block.get("display_text") or block.get("text") or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "journal of theoretical biology" in lowered:
        return False
    if "contents lists available" in lowered or "all rights reserved" in lowered:
        return False
    if "doi.org/" in lowered or re.search(r"\bdoi\s*:", lowered):
        return False
    if re.fullmatch(r"[\d\s.,;:()/-]+", text):
        return False
    return bool(re.search(r"\b[a-z]{3,}\b", lowered))


def _markdown_text_block_boundary_kind(
    block: dict[str, Any],
    toc_heading_lookup: dict[str, list[dict[str, Any]]],
) -> str:
    if str(block.get("block_type") or "").strip().lower() != "text":
        return "other"
    role = str(block.get("semantic_role") or "").strip()
    if role in {"section_heading", "reference_heading"}:
        return "standalone_heading"
    if role == "reference_entry":
        return "reference_entry"
    if role in {"footnote", "footnote_continuation"}:
        return role
    if role == "citation_metadata":
        return "body" if _is_markdown_body_citation_text_block(block) else "publication_metadata"
    if role in {"publication_footer", "publication_masthead", "license_notice", "page_number"}:
        return "publication_metadata"
    if _is_markdown_literature_standalone_label_block(block):
        return "standalone_label"
    if _is_markdown_literature_section_heading_block(block):
        return "standalone_heading"
    if _is_publication_metadata_markdown_block(block) and not _is_markdown_body_citation_text_block(block):
        return "publication_metadata"
    if _markdown_heading_for_text_block(block, toc_heading_lookup) is not None:
        return "standalone_heading"
    return "body"


def _is_markdown_body_paragraph_text_block(
    block: dict[str, Any],
    toc_heading_lookup: dict[str, list[dict[str, Any]]],
) -> bool:
    return _markdown_text_block_boundary_kind(block, toc_heading_lookup) == "body"


def _same_markdown_text_flow(
    left_block: dict[str, Any],
    right_block: dict[str, Any],
    paragraph_base_x0: float | None,
) -> bool:
    left_bbox = _markdown_block_bbox(left_block)
    right_bbox = _markdown_block_bbox(right_block)
    if left_bbox is None or right_bbox is None:
        return True
    left_x0, left_y0, left_x1, left_y1 = left_bbox
    right_x0, right_y0, right_x1, right_y1 = right_bbox
    if right_y0 + 3.0 < left_y0:
        return False
    left_height = max(1.0, left_y1 - left_y0)
    right_height = max(1.0, right_y1 - right_y0)
    vertical_gap = right_y0 - left_y1
    same_visual_line = abs(((left_y0 + left_y1) / 2.0) - ((right_y0 + right_y1) / 2.0)) <= max(6.0, min(left_height, right_height) * 0.55)
    next_text_line = -3.0 <= vertical_gap <= max(8.0, min(left_height, right_height) * 0.85)
    if same_visual_line:
        if right_x0 >= left_x0 - 2.0 and right_x0 <= left_x1 + 120.0:
            return True
    if next_text_line and abs(right_x0 - left_x0) <= 44.0:
        return True
    baseline_x0 = paragraph_base_x0 if paragraph_base_x0 is not None else left_x0
    if abs(right_x0 - baseline_x0) > 36.0 and abs(right_x0 - left_x0) > 36.0:
        return False
    if left_x1 <= left_x0 or right_x1 <= right_x0 or right_y1 <= right_y0 or left_y1 <= left_y0:
        return False
    return True


def _starts_new_indented_markdown_paragraph(
    current_text: str,
    next_text: str,
    current_block: dict[str, Any],
    next_block: dict[str, Any],
    paragraph_base_x0: float | None,
) -> bool:
    if _markdown_text_starts_bullet_item(next_text):
        return True
    if _markdown_text_starts_bullet_item(current_text):
        return False
    if paragraph_base_x0 is None:
        return False
    next_bbox = _markdown_block_bbox(next_block)
    if next_bbox is None:
        return False
    next_x0 = next_bbox[0]
    current_bbox = _markdown_block_bbox(current_block)
    if current_bbox is not None:
        current_center_y = (current_bbox[1] + current_bbox[3]) / 2.0
        next_center_y = (next_bbox[1] + next_bbox[3]) / 2.0
        current_height = max(1.0, current_bbox[3] - current_bbox[1])
        next_height = max(1.0, next_bbox[3] - next_bbox[1])
        if abs(current_center_y - next_center_y) <= max(6.0, min(current_height, next_height) * 0.55):
            return False
        vertical_gap = next_bbox[1] - current_bbox[3]
        same_indent = abs(next_x0 - current_bbox[0]) <= 4.0
        paragraph_gap = vertical_gap >= max(12.0, min(current_height, next_height) * 1.15)
        sentence_boundary = bool(re.search(r"[.!?。！？；;：:)\]）]\s*$", str(current_text or "").strip()))
        if same_indent and paragraph_gap and sentence_boundary:
            return True
        if _markdown_numbered_note_item_indented_continuation(
            current_text=current_text,
            next_text=next_text,
            current_block=current_block,
            next_block=next_block,
            current_bbox=current_bbox,
            next_bbox=next_bbox,
            vertical_gap=vertical_gap,
        ):
            return False
    if current_bbox is not None and abs(next_x0 - current_bbox[0]) <= 4.0:
        return False
    return next_x0 - paragraph_base_x0 >= 8.0


def _markdown_next_block_is_hanging_bullet_continuation(
    *,
    bullet_anchor_block: dict[str, Any],
    current_block: dict[str, Any],
    next_block: dict[str, Any],
    current_text: str,
    next_text: str,
) -> bool:
    if not _markdown_text_starts_bullet_item(current_text):
        return False
    if _markdown_text_starts_bullet_item(next_text):
        return False
    anchor_bbox = _markdown_block_bbox(bullet_anchor_block)
    current_bbox = _markdown_block_bbox(current_block)
    next_bbox = _markdown_block_bbox(next_block)
    if anchor_bbox is None or current_bbox is None or next_bbox is None:
        return False
    current_height = max(1.0, current_bbox[3] - current_bbox[1])
    next_height = max(1.0, next_bbox[3] - next_bbox[1])
    vertical_gap = next_bbox[1] - current_bbox[3]
    if vertical_gap < -3.0 or vertical_gap > max(14.0, min(current_height, next_height) * 1.35):
        return False
    indent_from_bullet = next_bbox[0] - anchor_bbox[0]
    if indent_from_bullet < max(10.0, min(current_height, next_height) * 0.75):
        return False
    if indent_from_bullet > 72.0:
        return False
    if _markdown_heading_shape_profile(str(next_text or "")) in {"numbered_section_title", "ctd_numbered_section_title"}:
        return False
    return True


def _markdown_numbered_note_item_indented_continuation(
    *,
    current_text: str,
    next_text: str,
    current_block: dict[str, Any],
    next_block: dict[str, Any],
    current_bbox: tuple[float, float, float, float],
    next_bbox: tuple[float, float, float, float],
    vertical_gap: float,
) -> bool:
    current_raw = str(current_block.get("display_text") or current_block.get("text") or current_text or "").strip()
    next_raw = str(next_block.get("display_text") or next_block.get("text") or next_text or "").strip()
    if not current_raw or not next_raw:
        return False
    if _markdown_numbered_note_item_marker(next_raw) is not None:
        return False
    if _markdown_numbered_note_item_marker(current_raw) is None:
        return False
    indent = float(next_bbox[0]) - float(current_bbox[0])
    if indent < 6.0 or indent > 48.0:
        return False
    current_height = max(1.0, float(current_bbox[3]) - float(current_bbox[1]))
    next_height = max(1.0, float(next_bbox[3]) - float(next_bbox[1]))
    if vertical_gap < -3.0 or vertical_gap > max(12.0, min(current_height, next_height) * 1.25):
        return False
    if _markdown_heading_shape_profile(next_raw) in {"numbered_section_title", "ctd_numbered_section_title"}:
        return False
    return True


def _markdown_numbered_note_item_marker(text: str) -> tuple[int, str] | None:
    match = re.match(r"^\s*[（(]\s*(?P<number>\d{1,3})\s*[）)]\s*\S", str(text or ""))
    if not match:
        return None
    try:
        number = int(match.group("number"))
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number, match.group(0)


def _can_merge_markdown_body_paragraph_blocks(
    current_block: dict[str, Any],
    next_block: dict[str, Any],
    current_text: str,
    next_text: str,
    paragraph_base_x0: float | None,
) -> bool:
    reference_merge = _can_merge_markdown_reference_entry_blocks(
        current_block,
        next_block,
        current_text,
        next_text,
        paragraph_base_x0,
    )
    if reference_merge is not None:
        return reference_merge
    if not _same_markdown_text_flow(current_block, next_block, paragraph_base_x0):
        return False
    if _starts_new_indented_markdown_paragraph(current_text, next_text, current_block, next_block, paragraph_base_x0):
        return False
    return True


def _can_merge_markdown_text_blocks(left_text: str, right_text: str) -> bool:
    left = str(left_text or "").strip()
    right = str(right_text or "").strip()
    if not left or not right:
        return False
    if _is_markdown_duplicate_continuation_residue(left, right):
        return True
    if _markdown_texts_form_parenthetical_continuation(left, right):
        return True
    if left.endswith(("If", "if")) and re.fullmatch(r"\$[^$\n]+\$", right):
        return True
    if (left.endswith("$") or re.fullmatch(r"\$[^$\n]+\$", left)) and re.match(r"^then\b", right, re.IGNORECASE):
        return True
    if re.search(r"\bof$", left) and re.fullmatch(r"(?:\$[A-Za-z]\([^$\n]*\)\$|\*)\.?", right):
        return True
    if re.search(r"\bwhere$", left, re.IGNORECASE) and re.fullmatch(r"\)?\s*\$[^$\n]+\$.*", right):
        return True
    if re.search(r"\b[A-Za-z]-$", left) and re.match(r"^[a-z]{2,}\b", right):
        return True
    if re.search(r"[A-Za-z]$", left) and re.match(r"^[a-z]{2,}\b", right):
        return True
    return False


def _merge_markdown_same_page_body_continuations_after_cross_page_block(
    page_render_blocks: list[dict[str, Any]],
    start_index: int,
    current_text: str,
    toc_heading_lookup: dict[str, list[dict[str, Any]]],
) -> tuple[str, int]:
    if start_index < 0 or start_index >= len(page_render_blocks):
        return current_text, start_index
    current_merge_block = page_render_blocks[start_index]
    current_text = str(current_text or "").strip()
    if not current_text:
        return current_text, start_index
    current_bbox = _markdown_block_bbox(current_merge_block)
    paragraph_base_x0 = current_bbox[0] if current_bbox is not None else None
    consumed_index = start_index
    next_index = start_index + 1
    while next_index < len(page_render_blocks):
        next_block = page_render_blocks[next_index]
        if str(next_block.get("block_type") or "").strip().lower() != "text":
            break
        next_text = _project_markdown_text_with_inline_formulas(next_block)
        if not next_text:
            next_index += 1
            continue
        if (
            _markdown_text_block_is_unmarked_list_item(current_merge_block)
            or _markdown_text_block_is_unmarked_list_item(next_block)
            or _markdown_text_starts_bullet_item(current_text)
            or _markdown_text_starts_bullet_item(next_text)
        ):
            break
        if (
            current_merge_block.get("_markdown_visual_standalone_heading")
            or next_block.get("_markdown_visual_standalone_heading")
        ):
            break
        current_is_body = (
            bool(current_merge_block.get("_markdown_cross_page_body_continuation"))
            or _is_markdown_body_paragraph_text_block(current_merge_block, toc_heading_lookup)
        )
        next_is_body = _is_markdown_body_paragraph_text_block(next_block, toc_heading_lookup)
        can_merge_special = _can_merge_markdown_text_blocks(current_text, next_text)
        can_merge_body = (
            current_is_body
            and next_is_body
            and _can_merge_markdown_body_paragraph_blocks(
                current_merge_block,
                next_block,
                current_text,
                next_text,
                paragraph_base_x0,
            )
        )
        if not (can_merge_special or can_merge_body):
            break
        current_text = _repair_inline_math_segments_with_prose_cues(
            _merge_markdown_text_fragments(current_text, next_text)
        )
        current_text = _repair_inline_math_residual_delimiters(current_text)
        next_bbox = _markdown_block_bbox(next_block)
        if (
            can_merge_body
            and next_bbox is not None
            and (
                paragraph_base_x0 is None
                or next_bbox[0] < paragraph_base_x0
            )
        ):
            paragraph_base_x0 = next_bbox[0]
        current_merge_block = next_block
        consumed_index = next_index
        next_index += 1
    return current_text, consumed_index


def _normalize_markdown_duplicate_residue_text(value: str) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return text.strip()


def _is_markdown_duplicate_continuation_residue(left_text: str, right_text: str) -> bool:
    left = _normalize_markdown_duplicate_residue_text(left_text)
    right = _normalize_markdown_duplicate_residue_text(right_text)
    if not left or not right or len(right) < 24:
        return False
    if len(right) > len(left):
        return False
    if left.startswith(right):
        return True
    if right in left:
        left_words = re.findall(r"[a-z0-9]+", left)
        right_words = re.findall(r"[a-z0-9]+", right)
        return len(right_words) >= 4 and len(right_words) >= max(4, int(len(left_words) * 0.45))
    return False


def _merge_markdown_text_fragments(left_text: str, right_text: str) -> str:
    left = str(left_text or "").strip()
    right = str(right_text or "").strip()
    if _is_markdown_duplicate_continuation_residue(left, right):
        return left
    if _markdown_texts_form_parenthetical_continuation(left, right):
        return f"{left}{right}"
    if (
        _markdown_text_starts_cross_page_continuation_punctuation(right)
        and re.search(r"[\u4e00-\u9fffA-Za-z0-9)\]\}\uff09\u3011\u300b\u3009\u300d\u300f\u201d\u2019]$", left)
    ):
        return f"{left}{right}"
    if left.endswith(("If", "if")) and re.fullmatch(r"\$[^$\n]+\$", right):
        return f"{left} {right}"
    if (left.endswith("$") or re.fullmatch(r"\$[^$\n]+\$", left)) and re.match(r"^then\b", right, re.IGNORECASE):
        return f"{left} {right}"
    if re.search(r"\bof$", left) and right == "*":
        return f"{left} $F(v)$"
    if re.search(r"\bof$", left) and re.fullmatch(r"\$[A-Za-z]\([^$\n]*\)\$\.?", right):
        return f"{left} {right}"
    if re.search(r"\bwhere$", left, re.IGNORECASE):
        cleaned_right = re.sub(r"^\)+\s*", "", right).strip()
        if re.fullmatch(r"\$[^$\n]+\$.*", cleaned_right):
            return f"{left} {cleaned_right}"
    if re.search(r"\b[A-Za-z]-$", left) and re.match(r"^[a-z]{2,}\b", right):
        return f"{left}{right}"
    if re.search(r"[\u4e00-\u9fff]$", left) and re.match(r"^[\u4e00-\u9fff]", right):
        return f"{left}{right}"
    return f"{left} {right}".strip()


def _markdown_equation_source_block_ids(block: dict[str, Any]) -> set[str]:
    if str(block.get("block_type") or "").strip().lower() != "equation":
        return set()
    return {
        str(source_block_id or "").strip()
        for source_block_id in block.get("source_block_ids", []) or []
        if str(source_block_id or "").strip()
    }


def _markdown_float_owned_text_block_ids(block: dict[str, Any]) -> set[str]:
    block_type = str(block.get("block_type") or "").strip().lower()
    if block_type not in {"table", "image"}:
        return set()
    owned_ids = {
        str(source_block_id or "").strip()
        for source_block_id in block.get("owned_text_block_ids", []) or []
        if str(source_block_id or "").strip()
    }
    for key in ("note_blocks", "caption_blocks", "content_segments"):
        for segment in block.get(key, []) or []:
            if not isinstance(segment, dict):
                continue
            if _markdown_float_owned_segment_renders_in_main_flow(segment):
                for source_block_id in [
                    segment.get("source_block_id"),
                    segment.get("block_id"),
                    segment.get("source_id"),
                    *(segment.get("source_block_ids", []) or []),
                ]:
                    source_block_id = str(source_block_id or "").strip()
                    if source_block_id:
                        owned_ids.discard(source_block_id)
                continue
            role = str(segment.get("role") or "").strip()
            if role == "nearby_context" and _markdown_nearby_context_segment_is_float_title(block, segment):
                source_block_id = str(
                    segment.get("source_block_id")
                    or segment.get("block_id")
                    or segment.get("source_id")
                    or ""
                ).strip()
                if source_block_id:
                    owned_ids.add(source_block_id)
                continue
            if role == "nearby_context":
                continue
            source_block_id = str(segment.get("source_block_id") or "").strip()
            if source_block_id:
                owned_ids.add(source_block_id)
            for source_block_id in segment.get("source_block_ids", []) or []:
                source_block_id = str(source_block_id or "").strip()
                if source_block_id:
                    owned_ids.add(source_block_id)
    return owned_ids


def _markdown_structure_template_owned_text_block_ids(
    structure_template: dict[str, Any],
    page_blocks: list[dict[str, Any]],
) -> set[str]:
    if str(structure_template.get("block_type") or "").strip().lower() != "structure_template":
        return set()
    owned_ids = {
        str(source_block_id or "").strip()
        for source_block_id in structure_template.get("owned_text_block_ids", []) or []
        if str(source_block_id or "").strip()
    }
    template_bbox = _markdown_block_bbox(structure_template)
    if template_bbox is None:
        return owned_ids
    template_text_norms = _markdown_structure_template_owned_text_norms(structure_template)
    if not template_text_norms:
        return owned_ids
    for candidate in page_blocks:
        if str(candidate.get("block_type") or "").strip().lower() != "text":
            continue
        candidate_id = str(candidate.get("block_id") or candidate.get("source_id") or "").strip()
        if not candidate_id:
            continue
        candidate_bbox = _markdown_block_bbox(candidate)
        if candidate_bbox is None or not _markdown_bbox_inside(candidate_bbox, template_bbox, tolerance=3.0):
            continue
        candidate_text = str(candidate.get("display_text") or candidate.get("text") or "").strip()
        if _markdown_compact_table_text(candidate_text) in template_text_norms:
            owned_ids.add(candidate_id)
    return owned_ids


def _markdown_structure_template_owned_text_norms(structure_template: dict[str, Any]) -> set[str]:
    norms: set[str] = set()

    def add_text(value: Any) -> None:
        norm = _markdown_compact_table_text(value)
        if norm:
            norms.add(norm)

    for text in structure_template.get("row_texts", []) or []:
        add_text(text)
    for key in ("fields", "sections", "entries", "note_blocks", "content_segments"):
        for item in structure_template.get(key, []) or []:
            if isinstance(item, dict):
                add_text(item.get("text") or item.get("title") or item.get("label"))
            else:
                add_text(item)
    return norms


def _markdown_structure_template_continuation_marker_owned_ids(
    page_blocks: list[dict[str, Any]],
    structure_template_owned_ids: set[str],
) -> set[str]:
    owned_ids: set[str] = set()
    if not structure_template_owned_ids:
        return owned_ids
    for index, block in enumerate(page_blocks):
        block_id = str(block.get("block_id") or block.get("source_id") or "").strip()
        if not block_id or block_id in structure_template_owned_ids:
            continue
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        text = str(block.get("display_text") or block.get("text") or "").strip()
        if not re.fullmatch(r"[\(（]\s*续\s*[\)）]", text):
            continue
        previous_owned = any(
            str(candidate.get("block_id") or candidate.get("source_id") or "").strip() in structure_template_owned_ids
            for candidate in reversed(page_blocks[max(0, index - 8):index])
            if str(candidate.get("block_type") or "").strip().lower() == "text"
        )
        next_heading = next(
            (
                candidate
                for candidate in page_blocks[index + 1:index + 5]
                if str(candidate.get("block_type") or "").strip().lower() == "text"
                and str(candidate.get("semantic_role") or "").strip() == "section_heading"
            ),
            None,
        )
        if previous_owned and next_heading is not None:
            owned_ids.add(block_id)
    return owned_ids


def _markdown_nearby_context_segment_is_float_title(block: dict[str, Any], segment: dict[str, Any]) -> bool:
    if str(segment.get("relation") or "").strip().lower() != "above":
        return False
    text = str(segment.get("text") or "").strip()
    if not text:
        return False
    title = str(block.get("title") or block.get("caption_text") or "").strip()
    if title and _markdown_table_titles_are_redundant(title, text):
        return True
    outline_marker = parse_outline_heading(text)
    if outline_marker is not None and outline_marker.marker_kind in {"ctd_mixed", "decimal_numeric"}:
        return outline_marker.normalized_marker.count(".") >= 2
    return _markdown_heading_shape_profile(text) in {"numbered_section_title", "ctd_numbered_section_title"}


def _markdown_metadata_only_source_block_ids(block: dict[str, Any]) -> set[str]:
    source_ids: set[str] = set()
    for edge in block.get("metadata_reference_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        if str(edge.get("visible_render_policy") or "").strip() != "metadata_only":
            continue
        source_block_id = str(edge.get("source_block_id") or "").strip()
        if source_block_id:
            source_ids.add(source_block_id)
    return source_ids


def _markdown_block_zone(block: dict[str, Any], page_context: dict[str, Any]) -> str:
    bbox = _markdown_block_bbox(block)
    page_height = float(page_context.get("page_height") or 0.0)
    if bbox is None or page_height <= 0:
        return "body"
    y0, y1 = bbox[1], bbox[3]
    center_y = (y0 + y1) / 2.0
    if center_y <= page_height * 0.075 or y1 <= page_height * 0.09:
        return "header"
    if center_y >= page_height * 0.925 or y0 >= page_height * 0.90:
        return "footer"
    return "body"


def _markdown_role_matches_any(role: str, names: set[str]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(role or "").strip().lower()).strip("_")
    return normalized in names


def _markdown_page_block_role(block: dict[str, Any], page_context: dict[str, Any]) -> dict[str, Any]:
    block_type = str(block.get("block_type") or "").strip().lower()
    semantic_role = str(block.get("semantic_role") or "").strip()
    block_id = str(block.get("block_id") or block.get("source_id") or "").strip()
    zone = _markdown_block_zone(block, page_context)
    toc_heading_lookup = page_context.get("toc_heading_lookup")
    if not isinstance(toc_heading_lookup, dict):
        toc_heading_lookup = {}

    role = "other"
    render_in_main_flow = False
    owned_by = None
    boundary_kind = "other"

    if _markdown_role_matches_any(semantic_role, {"page_number"}):
        role = "page_number"
    elif _markdown_role_matches_any(semantic_role, {"running_header", "page_header", "header"}):
        role = "page_header"
    elif _markdown_role_matches_any(semantic_role, {"running_footer", "page_footer", "footer"}):
        role = "page_footer"
    elif block_id and block_id in set(page_context.get("metadata_only_source_block_ids") or []):
        role = "metadata_only_text" if block_type == "text" else "metadata_only_block"
    elif block_id and block_id in set(page_context.get("equation_source_block_ids") or []):
        role = "equation_source_text"
    elif block_id and block_id in set(page_context.get("float_owned_text_block_ids") or []):
        if _markdown_role_matches_any(semantic_role, {"table_note", "note"}):
            role = "table_note"
        elif _markdown_role_matches_any(semantic_role, {"figure_legend", "image_legend", "legend", "caption", "figure_caption"}):
            role = "figure_legend"
        else:
            role = "float_owned_text"
        owned_by = "float"
    elif block_id and block_id in set(page_context.get("structure_template_owned_text_block_ids") or []):
        role = "structure_template_owned_text"
    elif _markdown_role_matches_any(semantic_role, {"logo", "brand_logo"}):
        role = "logo"
    elif block_type == "toc":
        role = "toc"
    elif block_type == "structure_template":
        if _markdown_visible_render_policy(block) == "metadata_only" or _markdown_structure_template_absorbed_by_business_table(block):
            role = "absorbed_structure_template"
        else:
            role = "structure_template"
            render_in_main_flow = True
    elif block_type == "table":
        if _markdown_visible_render_policy(block) == "metadata_only":
            role = "metadata_only_table"
        else:
            role = "table"
            render_in_main_flow = True
    elif block_type == "image":
        if _markdown_visible_render_policy(block) == "metadata_only":
            role = "metadata_only_figure"
        else:
            role = "figure"
            render_in_main_flow = True
    elif block_type == "algorithm":
        role = "algorithm_pseudocode"
        render_in_main_flow = True
    elif block_type == "equation" or semantic_role == "display_equation":
        role = "display_equation"
        render_in_main_flow = True
    elif block_type == "text":
        if _markdown_visible_render_policy(block) == "metadata_only":
            role = "metadata_only_text"
        else:
            boundary_kind = _markdown_text_block_boundary_kind(block, toc_heading_lookup)
        if role == "metadata_only_text":
            pass
        elif boundary_kind == "publication_metadata":
            if _markdown_role_matches_any(semantic_role, {"page_number"}):
                role = "page_number"
            elif _markdown_role_matches_any(semantic_role, {"running_header", "page_header"}):
                role = "page_header"
            elif _markdown_role_matches_any(semantic_role, {"running_footer", "page_footer", "publication_footer"}):
                role = "page_footer"
            else:
                role = "publication_metadata"
        elif boundary_kind == "standalone_heading":
            if block.get("_markdown_numbered_step_demoted_from_heading"):
                role = "body"
                boundary_kind = "body"
            else:
                role = "heading"
            render_in_main_flow = True
        elif boundary_kind == "standalone_label":
            role = "standalone_label"
            render_in_main_flow = True
        elif boundary_kind == "reference_entry":
            role = "reference_entry"
            render_in_main_flow = True
        elif boundary_kind in {"footnote", "footnote_continuation"}:
            role = boundary_kind
        else:
            role = "body"
            render_in_main_flow = True

    if _markdown_role_matches_any(
        semantic_role,
        {"structure_template_title", "structure_template_entry", "structure_template_note"},
    ) and block_type == "text":
        role = "structure_template_owned_text"
        render_in_main_flow = False

    if role in {"page_number", "page_header", "page_footer", "logo", "toc", "equation_source_text", "table_note", "figure_legend", "float_owned_text", "footnote", "footnote_continuation", "structure_template_owned_text"}:
        render_in_main_flow = False

    return {
        "role": role,
        "zone": zone,
        "render_in_main_flow": render_in_main_flow,
        "owned_by": owned_by,
        "boundary_kind": boundary_kind,
        "block_type": block_type,
    }


def _collect_markdown_page_rendering_audit(
    page_blocks: list[dict[str, Any]],
    *,
    page_number: int,
    page_height: float,
) -> list[dict[str, Any]]:
    metadata_only_source_block_ids = {
        source_block_id
        for page_block in page_blocks or []
        if isinstance(page_block, dict)
        for source_block_id in _markdown_metadata_only_source_block_ids(page_block)
    }
    metadata_only_source_block_edges: dict[str, list[dict[str, Any]]] = {}
    for page_block in page_blocks or []:
        if not isinstance(page_block, dict):
            continue
        for edge in page_block.get("metadata_reference_edges", []) or []:
            if not isinstance(edge, dict):
                continue
            if str(edge.get("visible_render_policy") or "").strip() != "metadata_only":
                continue
            source_block_id = str(edge.get("source_block_id") or "").strip()
            if source_block_id:
                metadata_only_source_block_edges.setdefault(source_block_id, []).append(dict(edge))
    page_context = {
        "page_number": page_number,
        "page_height": page_height,
        "metadata_only_source_block_ids": metadata_only_source_block_ids,
        "metadata_only_source_block_edges": metadata_only_source_block_edges,
    }
    audits: list[dict[str, Any]] = []
    for block in page_blocks or []:
        if not isinstance(block, dict):
            continue
        audit = _markdown_rendering_audit_for_block(block, page_context)
        if audit is not None:
            audits.append(audit)
    return audits


def _markdown_rendering_audit_reason_counts(audits: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for audit in audits or []:
        if not isinstance(audit, dict):
            continue
        reason = str(audit.get("reason") or "").strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return {reason: counts[reason] for reason in sorted(counts)}


def _attach_markdown_rendering_audit_metadata(document: dict[str, Any]) -> dict[str, Any]:
    document_ast = document.get("document_ast")
    document_ast = document_ast if isinstance(document_ast, dict) else {}
    pages = [page for page in document_ast.get("pages", []) or [] if isinstance(page, dict)]
    page_audits: list[dict[str, Any]] = []
    all_suppressed_blocks: list[dict[str, Any]] = []
    suppressed_block_count = 0
    for page in pages:
        page_number = int(page.get("page", 0) or 0)
        page_height = float(page.get("page_height") or page.get("height") or 0.0)
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        suppressed_blocks = _collect_markdown_page_rendering_audit(
            blocks,
            page_number=page_number,
            page_height=page_height,
        )
        if not suppressed_blocks:
            continue
        all_suppressed_blocks.extend(suppressed_blocks)
        suppressed_block_count += len(suppressed_blocks)
        page_audits.append(
            {
                "page": page_number,
                "suppressed_block_count": len(suppressed_blocks),
                "reason_counts": _markdown_rendering_audit_reason_counts(suppressed_blocks),
                "suppressed_blocks": suppressed_blocks,
            }
        )
    metadata = document.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        document["metadata"] = metadata
    metadata["markdown_rendering_audit"] = {
        "page_count": len(page_audits),
        "suppressed_block_count": suppressed_block_count,
        "reason_counts": _markdown_rendering_audit_reason_counts(all_suppressed_blocks),
        "pages": page_audits,
    }
    return document


def _build_markdown_rendering_audit_summary(parsed_documents: list[dict[str, Any]]) -> dict[str, Any]:
    document_summaries: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    suppressed_block_count = 0
    for document in parsed_documents or []:
        if not isinstance(document, dict):
            continue
        working_document = dict(document)
        metadata = document.get("metadata")
        working_document["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
        _attach_markdown_rendering_audit_metadata(working_document)
        audit = working_document.get("metadata", {}).get("markdown_rendering_audit", {})
        if not isinstance(audit, dict):
            continue
        doc_suppressed_count = int(audit.get("suppressed_block_count") or 0)
        if doc_suppressed_count <= 0:
            continue
        doc_reason_counts = {
            str(reason): int(count)
            for reason, count in (audit.get("reason_counts") or {}).items()
            if str(reason).strip() and int(count or 0) > 0
        }
        for reason, count in doc_reason_counts.items():
            reason_counts[reason] = reason_counts.get(reason, 0) + count
        suppressed_block_count += doc_suppressed_count
        doc_metadata = document.get("metadata")
        doc_metadata = doc_metadata if isinstance(doc_metadata, dict) else {}
        document_summaries.append(
            {
                "file_id": str(document.get("file_id") or "").strip(),
                "filename": str(document.get("filename") or "").strip(),
                "page_count": int(doc_metadata.get("page_count") or 0),
                "suppressed_page_count": int(audit.get("page_count") or 0),
                "suppressed_block_count": doc_suppressed_count,
                "reason_counts": {reason: doc_reason_counts[reason] for reason in sorted(doc_reason_counts)},
            }
        )
    return {
        "document_count": len(document_summaries),
        "suppressed_block_count": suppressed_block_count,
        "reason_counts": {reason: reason_counts[reason] for reason in sorted(reason_counts)},
        "documents": document_summaries,
    }


def _markdown_rendering_audit_for_block(
    block: dict[str, Any],
    page_context: dict[str, Any],
) -> dict[str, Any] | None:
    role_info = _markdown_page_block_role(block, page_context)
    role = str(role_info.get("role") or "").strip()
    if role not in {
        "metadata_only_text",
        "metadata_only_block",
        "metadata_only_table",
        "metadata_only_figure",
        "absorbed_structure_template",
    }:
        return None
    block_id = str(block.get("block_id") or block.get("source_id") or "").strip()
    block_type = str(role_info.get("block_type") or block.get("block_type") or "").strip()
    audit = {
        "block_id": block_id,
        "block_type": block_type,
        "page": page_context.get("page_number"),
        "rendered": False,
        "role": role,
        "reason": _markdown_rendering_audit_reason(block, role, page_context),
    }
    if block_id and block_id in set(page_context.get("metadata_only_source_block_ids") or []):
        audit["source_block_id"] = block_id
        edge_map = page_context.get("metadata_only_source_block_edges")
        edges = edge_map.get(block_id, []) if isinstance(edge_map, dict) else []
        first_edge = next((edge for edge in edges if isinstance(edge, dict)), None)
        if first_edge is not None:
            for key in ("relation", "target_object_type", "target_object_id"):
                value = str(first_edge.get(key) or "").strip()
                if value:
                    audit[key] = value
    return audit


def _markdown_rendering_audit_reason(
    block: dict[str, Any],
    role: str,
    page_context: dict[str, Any],
) -> str:
    block_id = str(block.get("block_id") or block.get("source_id") or "").strip()
    if block_id and block_id in set(page_context.get("metadata_only_source_block_ids") or []):
        return "metadata_reference_edge_metadata_only"
    if role == "absorbed_structure_template":
        return "absorbed_structure_template"
    if _markdown_visible_render_policy(block) == "metadata_only":
        return "visible_render_policy_metadata_only"
    return role


def _append_markdown_inline_formula_items(lines: list[str], block: dict[str, Any]) -> None:
    spans = _collect_high_confidence_inline_formula_spans(block)
    if not spans:
        return

    lines.append("**公式项**")
    lines.append("")
    for span in spans:
        latex_text = str(span.get("latex_text") or "").strip()
        source_text = str(project_pdf_math_symbol_display_text(span.get("content") or "") or "").strip()
        if source_text:
            lines.append(f"- `{source_text}` -> ${latex_text}$")
        else:
            lines.append(f"- ${latex_text}$")
    lines.append("")


def _append_markdown_equation(lines: list[str], block: dict[str, Any]) -> None:
    equation_label = str(block.get("equation_label") or "").strip()
    equation_id = str(block.get("equation_id") or block.get("block_id") or "").strip()
    alt_text = f"鍏紡 {equation_label}" if equation_label else "鍏紡"

    latex_text = str(block.get("latex_text") or "").strip()
    try:
        latex_confidence = float(block.get("latex_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        latex_confidence = 0.0
    if latex_text and latex_confidence >= 0.85:
        lines.append("$$")
        lines.append(latex_text)
        lines.append("$$")
        lines.append("")
        return

    image_markdown = _render_pdf_bbox_crop_markdown(
        source_path=block.get("_source_path"),
        page_number=int(block.get("page", 0) or 0),
        bbox=block.get("bbox"),
        alt_text=alt_text,
        scale=3.0,
    )
    if image_markdown is not None:
        lines.append(image_markdown)
        lines.append("")

    if image_markdown is None and not latex_text and equation_id:
        lines.append(f"`{equation_id}`")
        lines.append("")


def _dedupe_markdown_footnote_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        marker = str(ref.get("marker") or "").strip()
        if not marker:
            continue
        footnote_id = str(ref.get("footnote_id") or "").strip()
        key = (marker, footnote_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _collect_markdown_footnote_refs(block: dict[str, Any]) -> list[dict[str, Any]]:
    refs = [
        dict(ref)
        for ref in block.get("footnote_refs", []) or []
        if str(ref.get("marker") or "").strip()
    ]
    return _dedupe_markdown_footnote_refs(refs)


def _project_markdown_footnote_refs_inline(
    text: str,
    refs: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    projected = str(text or "")
    remaining: list[dict[str, Any]] = []
    for ref in refs:
        marker = str(ref.get("marker") or "").strip()
        if not marker:
            continue
        markdown_ref = f"[^{marker}]"
        if markdown_ref in projected:
            continue
        marker_pattern = re.compile(
            rf"(?<![\d\[]){re.escape(marker)}(?![\d\]])"
        )
        matches = list(marker_pattern.finditer(projected))
        if not matches:
            remaining.append(ref)
            continue
        match = matches[-1]
        projected = f"{projected[:match.start()]}{markdown_ref}{projected[match.end():]}"
    return projected, remaining


def _append_markdown_footnote_ref_suffix(text: str, block: dict[str, Any]) -> str:
    refs = _collect_markdown_footnote_refs(block)
    if refs:
        text, remaining_refs = _project_markdown_footnote_refs_inline(text, refs)
        suffix = " ".join(f"[^{str(ref.get('marker') or '').strip()}]" for ref in remaining_refs)
        if not suffix:
            return text
        return f"{text} {suffix}".strip()
    return text


def _merge_markdown_text_block_projection_metadata(
    base_block: dict[str, Any],
    merged_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(merged_blocks) <= 1:
        return base_block
    footnote_refs: list[dict[str, Any]] = []
    linked_footnote_ids: list[str] = []
    linked_seen: set[str] = set()
    source_block_ids: list[str] = []
    source_seen: set[str] = set()
    for item in merged_blocks:
        block_id = str(item.get("block_id") or item.get("source_id") or "").strip()
        if block_id and block_id not in source_seen:
            source_seen.add(block_id)
            source_block_ids.append(block_id)
        footnote_refs.extend(_collect_markdown_footnote_refs(item))
        for footnote_id in item.get("linked_footnote_ids", []) or []:
            footnote_id_text = str(footnote_id or "").strip()
            if not footnote_id_text or footnote_id_text in linked_seen:
                continue
            linked_seen.add(footnote_id_text)
            linked_footnote_ids.append(footnote_id_text)
    merged = {
        **base_block,
        "footnote_refs": _dedupe_markdown_footnote_refs(footnote_refs),
        "linked_footnote_ids": linked_footnote_ids,
    }
    if source_block_ids:
        merged["_markdown_merged_source_block_ids"] = source_block_ids
    return merged


def _append_markdown_footnote_definition(lines: list[str], block: dict[str, Any]) -> None:
    marker = str(block.get("footnote_marker") or "").strip()
    text = str(block.get("footnote_text") or block.get("text") or "").strip()
    if not marker or not text:
        return
    text = re.sub(rf"^\s*{re.escape(marker)}\s*", "", text).strip()
    text = _markdown_linkify_visible_urls(text)
    lines.append(f"[^{marker}]: {text}")
    lines.append("")


_PUBLICATION_METADATA_MARKDOWN_ROLES = {
    "author_affiliation",
    "author_line",
    "author_note",
    "contact_email",
    "contact_name",
    "correspondence",
    "citation_metadata",
    "keyword_metadata",
    "license_notice",
    "page_number",
    "publication_footer",
    "publication_masthead",
}


def _is_publication_metadata_markdown_block(block: dict[str, Any]) -> bool:
    semantic_role = str(block.get("semantic_role") or "").strip()
    unit_role = str(block.get("unit_role") or "").strip()
    return unit_role in {"metadata", "publication_metadata"} or semantic_role in _PUBLICATION_METADATA_MARKDOWN_ROLES


def _is_body_flow_continuation_block(
    previous_block: dict[str, Any] | None,
    block: dict[str, Any],
) -> bool:
    if previous_block is None:
        return False
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    if str(previous_block.get("block_type") or "").strip().lower() != "text":
        return False
    role = str(block.get("semantic_role") or "").strip()
    if role not in {"author_line", "author_note", "citation_metadata", "keyword_metadata"}:
        return False
    previous_role = str(previous_block.get("semantic_role") or "").strip()
    if previous_role in {"publication_masthead", "publication_footer", "license_notice", "page_number"}:
        return False
    if _is_publication_metadata_markdown_block(previous_block) and not _is_markdown_body_citation_text_block(previous_block):
        return False
    text = str(block.get("display_text") or block.get("text") or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in ("doi.org/", "received:", "accepted:", "available online", "corresponding author")):
        return False
    bbox = _markdown_block_bbox(block)
    previous_bbox = _markdown_block_bbox(previous_block)
    if bbox is None or previous_bbox is None:
        return False
    x0, y0, _x1, y1 = bbox
    previous_x0, _previous_y0, _previous_x1, previous_y1 = previous_bbox
    if y0 + 3.0 < previous_y1:
        return False
    vertical_gap = y0 - previous_y1
    if vertical_gap > 22.0:
        return False
    return abs(x0 - previous_x0) <= 52.0 or previous_y1 >= y1 - 18.0


def _should_render_publication_metadata_markdown_block(block: dict[str, Any]) -> bool:
    semantic_role = str(block.get("semantic_role") or "").strip()
    if semantic_role in {"page_number", "publication_masthead", "publication_footer", "license_notice"}:
        return False
    if str(block.get("unit_role") or "").strip() not in {"metadata", "publication_metadata"}:
        return False
    text = str(block.get("display_text") or block.get("text") or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "journal of theoretical biology" in lowered:
        return False
    if "contents lists available" in lowered or "all rights reserved" in lowered:
        return False
    if "doi.org/" in lowered:
        return False
    return semantic_role in {
        "author_affiliation",
        "author_line",
        "author_note",
        "contact_email",
        "contact_name",
        "correspondence",
        "citation_metadata",
        "keyword_metadata",
    }


def _collect_publication_metadata_markdown_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for block in blocks:
        if not _is_publication_metadata_markdown_block(block):
            continue
        if not _should_render_publication_metadata_markdown_block(block):
            continue
        block_id = str(block.get("block_id") or block.get("source_id") or "").strip()
        if block_id and block_id in seen_ids:
            continue
        if block_id:
            seen_ids.add(block_id)
        collected.append(block)
    return collected


def _should_defer_publication_metadata_before_body(block: dict[str, Any]) -> bool:
    semantic_role = str(block.get("semantic_role") or "").strip()
    if semantic_role not in {"author_note", "contact_email", "contact_name", "correspondence"}:
        return False
    return _should_render_publication_metadata_markdown_block(block)


def _canonical_outline_index(value: Any) -> str:
    return normalize_outline_marker(value)


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
    return outline_titles_compatible(_strip_probable_footnote_suffix(body_title), toc_title)


def _markdown_block_is_section_heading_continuation(
    heading_block: dict[str, Any],
    next_block: dict[str, Any] | None,
    page_blocks: list[dict[str, Any]] | None = None,
    heading_index: int | None = None,
) -> bool:
    if not isinstance(next_block, dict):
        return False
    if str(next_block.get("block_type") or "").strip().lower() != "text":
        return False
    next_role = str(next_block.get("semantic_role") or "").strip()
    text = str(next_block.get("display_text") or next_block.get("text") or "").strip()
    if not text or _markdown_text_starts_bullet_item(text):
        return False
    heading_context = heading_block.get("section_context")
    next_context = next_block.get("section_context")
    if isinstance(heading_context, dict) and isinstance(next_context, dict):
        heading_outline = str(heading_context.get("outline_index") or "").strip()
        next_outline = str(next_context.get("outline_index") or "").strip()
        if heading_outline and next_outline and heading_outline != next_outline:
            return False
    if next_role in {"section_heading_continuation", "heading_continuation"}:
        return True
    if next_role:
        return False
    if not _markdown_text_looks_like_ind_heading_metadata_continuation(text):
        return False
    if not _markdown_blocks_are_adjacent_heading_lines(heading_block, next_block):
        return False
    if page_blocks is None or heading_index is None:
        return False
    heading_text = str(heading_block.get("display_text") or heading_block.get("text") or "").strip()
    full_title = f"{heading_text} {text}".strip()
    return _markdown_following_object_references_visible_title(
        page_blocks,
        heading_index + 2,
        full_title,
    )


def _collect_markdown_section_heading_continuation_group(
    page_blocks: list[dict[str, Any]],
    heading_index: int,
) -> list[dict[str, Any]]:
    heading = page_blocks[heading_index]
    continuations: list[dict[str, Any]] = []
    previous = heading
    for candidate in page_blocks[heading_index + 1 : heading_index + 5]:
        if not _markdown_single_heading_continuation_candidate(heading, candidate, previous):
            break
        candidate_texts = [
            str(item.get("display_text") or item.get("text") or "").strip()
            for item in continuations + [candidate]
            if str(item.get("display_text") or item.get("text") or "").strip()
        ]
        heading_text = str(heading.get("display_text") or heading.get("text") or "").strip()
        full_title = " ".join([heading_text, *candidate_texts]).strip()
        start_index = heading_index + 1 + len(candidate_texts)
        if not _markdown_following_object_references_visible_title(page_blocks, start_index, full_title):
            break
        continuations.append(candidate)
        previous = candidate
    return continuations


def _markdown_single_heading_continuation_candidate(
    heading_block: dict[str, Any],
    candidate: dict[str, Any] | None,
    previous_block: dict[str, Any],
) -> bool:
    if not isinstance(candidate, dict):
        return False
    if str(candidate.get("block_type") or "").strip().lower() != "text":
        return False
    role = str(candidate.get("semantic_role") or "").strip()
    text = str(candidate.get("display_text") or candidate.get("text") or "").strip()
    if not text or _markdown_text_starts_bullet_item(text):
        return False
    heading_context = heading_block.get("section_context")
    candidate_context = candidate.get("section_context")
    if isinstance(heading_context, dict) and isinstance(candidate_context, dict):
        heading_outline = str(heading_context.get("outline_index") or "").strip()
        candidate_outline = str(candidate_context.get("outline_index") or "").strip()
        if heading_outline and candidate_outline and heading_outline != candidate_outline:
            return False
    if role in {"section_heading_continuation", "heading_continuation"}:
        return True
    if role:
        return False
    if not _markdown_text_looks_like_ind_heading_metadata_continuation(text):
        return False
    return _markdown_blocks_are_adjacent_heading_lines(previous_block, candidate)


def _markdown_text_looks_like_ind_heading_metadata_continuation(text: str) -> bool:
    compact = str(text or "").strip()
    if not compact:
        return False
    return bool(
        re.search(
            r"(?:报告标题|供试品|试验标题|试验名称|试验编号|Test\s+article|Report\s+title|Study\s+(?:title|number))\s*[:：]",
            compact,
            re.IGNORECASE,
        )
    )


def _markdown_blocks_are_adjacent_heading_lines(
    heading_block: dict[str, Any],
    next_block: dict[str, Any],
) -> bool:
    heading_bbox = _markdown_block_bbox(heading_block)
    next_bbox = _markdown_block_bbox(next_block)
    if heading_bbox is None or next_bbox is None:
        return True
    hx0, hy0, hx1, hy1 = heading_bbox
    nx0, ny0, nx1, ny1 = next_bbox
    if ny0 + 2.0 < hy0:
        return False
    height = max(1.0, hy1 - hy0, ny1 - ny0)
    vertical_gap = ny0 - hy1
    horizontal_overlap = min(hx1, nx1) - max(hx0, nx0)
    overlap_ratio = horizontal_overlap / max(1.0, min(hx1 - hx0, nx1 - nx0))
    return vertical_gap <= max(8.0, height * 0.75) and (
        overlap_ratio >= 0.15 or abs(nx0 - hx0) <= max(36.0, height * 2.0)
    )


def _markdown_following_object_references_visible_title(
    page_blocks: list[dict[str, Any]],
    start_index: int,
    full_title: str,
) -> bool:
    full_norm = _markdown_compact_table_text(full_title)
    if not full_norm:
        return False
    for candidate in page_blocks[start_index : start_index + 4]:
        candidate_type = str(candidate.get("block_type") or "").strip().lower()
        if candidate_type == "text":
            candidate_role = str(candidate.get("semantic_role") or "").strip()
            if candidate_role in {"page_number", "running_header", "running_footer", "page_header", "page_footer"}:
                continue
            candidate_text = str(candidate.get("display_text") or candidate.get("text") or "").strip()
            if not candidate_text:
                continue
            if _markdown_text_looks_like_ind_heading_metadata_continuation(candidate_text):
                continue
            break
        if candidate_type not in {"table", "image", "structure_template"}:
            continue
        candidate_title = str(candidate.get("title") or candidate.get("caption_text") or "").strip()
        candidate_norm = _markdown_compact_table_text(candidate_title)
        return bool(candidate_norm and (candidate_norm == full_norm or full_norm in candidate_norm))
    return False


def _markdown_heading_for_text_block(
    block: dict[str, Any],
    toc_heading_lookup: dict[str, list[dict[str, Any]]],
) -> tuple[str, int] | None:
    text = str(block.get("display_text") or block.get("text") or "").strip()
    if not text:
        return None
    section_context = dict(block.get("section_context", {}) or {})
    semantic_role = str(block.get("semantic_role") or "").strip()
    if block.get("_markdown_numbered_step_demoted_from_heading"):
        return None
    if semantic_role == "section_heading":
        continuation_text = str(block.get("_markdown_heading_continuation_text") or "").strip()
        if continuation_text and continuation_text not in text:
            text = f"{text} {continuation_text}"
        explicit_level = int(block.get("heading_level", 0) or 0)
        section_level = explicit_level or int(section_context.get("section_level", 0) or 0)
        if section_level <= 0:
            outline_index = str(section_context.get("outline_index") or "").strip()
            section_level = len([part for part in outline_index.split(".") if part]) if outline_index else 1
        return text, min(6, 2 + max(1, section_level))
    if block.get("_markdown_visual_standalone_heading"):
        continuation_text = str(block.get("_markdown_heading_continuation_text") or "").strip()
        if continuation_text and continuation_text not in text:
            text = f"{text} {continuation_text}"
        visual_level = int(block.get("_markdown_visual_heading_level", 1) or 1)
        return text, max(1, min(6, visual_level))
    marker_candidate = classify_outline_heading_candidate(
        text,
        toc_heading_lookup=toc_heading_lookup,
        active_parent_marker=section_context.get("outline_index"),
    )
    if marker_candidate is None or marker_candidate.role == "body_list_item":
        return None
    marker = marker_candidate.marker
    outline_index = marker.normalized_marker
    body_title = marker.title
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
    text = _append_markdown_footnote_ref_suffix(text, block)
    lines.append(f"{'#' * level} {_markdown_linkify_visible_urls(text)}")
    lines.append("")


def _build_document_body_markdown_sections(
    document: dict[str, Any],
    *,
    embed_images: bool = True,
    table_export_mode: str = "markdown",
    image_text_mode: str = "semantic",
    body_heading: str | None = "__autoind_default_body_heading__",
    render_toc_blocks: bool = False,
    skip_uncaptioned_image_placeholders: bool = False,
    visual_heading_projection: bool = False,
    visual_heading_projection_profiles: set[str] | None = None,
    merge_safe_evidence_table_chains: bool = False,
) -> list[str]:
    document_ast = document.get("document_ast", {}) or {}
    ast_pages = [
        page
        for page in document_ast.get("pages", []) or []
        if isinstance(page, dict)
    ]
    if not ast_pages:
        return []

    lines = ["### 正文结构化内容", ""]
    if body_heading is None:
        lines = []
    elif body_heading != "__autoind_default_body_heading__":
        lines = [body_heading, ""]
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
    toc_by_id = {
        str(toc.get("toc_id") or toc.get("block_id") or "").strip(): toc
        for toc in document.get("toc_blocks", []) or []
        if isinstance(toc, dict) and str(toc.get("toc_id") or toc.get("block_id") or "").strip()
    }
    toc_heading_lookup = _build_toc_heading_lookup(document)
    uri_link_records_by_page = _build_uri_link_records_by_page(document)
    source_pages_by_number = {
        int(page.get("page_number", page.get("page", 0)) or 0): page
        for page in document.get("pages", []) or []
        if isinstance(page, dict) and int(page.get("page_number", page.get("page", 0)) or 0) > 0
    }
    skipped_table_ids: set[str] = set()
    rendered_footnote_ids: set[str] = set()
    rendered_publication_metadata_ids: set[str] = set()
    footnote_definition_lines: list[str] = []
    source_path_value = document.get("source_path")
    source_path = Path(str(source_path_value)) if str(source_path_value or "").strip() else None
    previous_visible_heading_title = ""
    visible_title_by_table_id: dict[str, str] = {}
    visible_title_by_structure_template_id: dict[str, str] = {}
    rendered_object_prelude_text_ids: set[str] = set()
    rendered_main_flow_text_ids: set[str] = set()
    suppressed_note_norms_by_context_template_id: dict[str, set[str]] = {}
    trailing_metadata_rows_by_context_template_id: dict[str, list[list[Any]]] = {}
    previous_page_last_flow_text_block: dict[str, Any] | None = None
    for page in ast_pages:
        page_number = page.get("page")
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        if not blocks:
            previous_page_last_flow_text_block = None
            continue
        page_render_blocks: list[dict[str, Any]] = []
        for block in blocks:
            if page_number is not None and "page" not in block:
                block = {**block, "page": page_number}
            block = _enrich_document_ast_block(
                block,
                table_by_id=table_by_id,
                image_by_id=image_by_id,
                toc_by_id=toc_by_id,
            )
            if (
                not str(block.get("block_type") or "").strip()
                and str(block.get("role") or block.get("semantic_role") or "").strip() in {"text", "text_block"}
            ):
                block = {**block, "block_type": "text"}
            if source_path is not None:
                block = {**block, "_source_path": source_path}
            page_render_blocks.append(block)
        source_page = source_pages_by_number.get(int(page_number or 0), {})
        page_height = float(page.get("height", source_page.get("height", 0.0)) or 0.0)
        page_width = float(page.get("width", source_page.get("width", 0.0)) or 0.0)
        page_previous_flow_block = previous_page_last_flow_text_block
        for index, block in enumerate(page_render_blocks):
            previous_block = page_render_blocks[index - 1] if index > 0 else None
            if _is_body_flow_continuation_block(previous_block, block):
                page_render_blocks[index] = {
                    **block,
                    "_markdown_body_flow_continuation": True,
                }
        if page_render_blocks and page_previous_flow_block is not None:
            first_flow_index = 0
            first_block = page_render_blocks[first_flow_index]
            if _markdown_block_is_parenthetical_continuation(
                page_previous_flow_block,
                first_block,
                allow_cross_page=True,
            ):
                page_render_blocks[first_flow_index] = {
                    **first_block,
                    "_markdown_cross_page_parenthetical_continuation": True,
                }
            elif _markdown_block_is_cross_page_body_continuation(
                page_previous_flow_block,
                first_block,
                page_height=page_height,
                toc_heading_lookup=toc_heading_lookup,
            ):
                page_render_blocks[first_flow_index] = {
                    **first_block,
                    "_markdown_cross_page_body_continuation": True,
                }
        for index, block in enumerate(page_render_blocks):
            if (
                str(block.get("block_type") or "").strip().lower() == "text"
                and str(block.get("semantic_role") or "").strip() == "section_heading"
                and _markdown_blocks_form_compact_numbered_step_sequence(page_render_blocks, index)
            ):
                page_render_blocks[index] = {
                    **block,
                    "_markdown_numbered_step_demoted_from_heading": True,
                }
        for index, block in enumerate(page_render_blocks):
            if str(block.get("block_type") or "").strip().lower() != "text":
                continue
            if str(block.get("semantic_role") or "").strip() != "section_heading":
                continue
            continuation_group = _collect_markdown_section_heading_continuation_group(
                page_render_blocks,
                index,
            )
            if not continuation_group:
                continue
            continuation_text = " ".join(
                str(item.get("display_text") or item.get("text") or "").strip()
                for item in continuation_group
                if str(item.get("display_text") or item.get("text") or "").strip()
            ).strip()
            page_render_blocks[index] = {
                **block,
                "_markdown_heading_continuation_block_ids": [
                    str(item.get("block_id") or item.get("source_id") or "").strip()
                    for item in continuation_group
                    if str(item.get("block_id") or item.get("source_id") or "").strip()
                ],
                "_markdown_heading_continuation_text": continuation_text,
            }
            for offset in range(1, len(continuation_group) + 1):
                page_render_blocks[index + offset] = {
                    **page_render_blocks[index + offset],
                    "_markdown_heading_continuation_consumed": True,
                }
        toc_like_page = _looks_like_markdown_toc_like_page(page_render_blocks)
        landscape_visual_panel = _markdown_page_looks_like_landscape_visual_panel(
            page_render_blocks,
            page_width,
            page_height,
        )
        if visual_heading_projection:
            page_render_blocks = _mark_markdown_centered_front_matter_title_clusters(
                page_render_blocks,
                page_width=page_width,
                page_height=page_height,
                page_previous_flow_block=page_previous_flow_block,
                toc_like_page=toc_like_page,
                landscape_visual_panel=landscape_visual_panel,
            )
        if toc_like_page:
            for index, block in enumerate(page_render_blocks):
                if str(block.get("block_type") or "").strip().lower() != "text":
                    continue
                text = str(block.get("display_text") or block.get("text") or "").strip()
                if _markdown_heading_shape_profile(text) != "contents_title":
                    page_render_blocks[index] = {
                        **block,
                        "_markdown_numbered_step_demoted_from_heading": True,
                    }
        if visual_heading_projection:
            for index, block in enumerate(page_render_blocks):
                previous_block = page_render_blocks[index - 1] if index > 0 else page_previous_flow_block
                next_block = page_render_blocks[index + 1] if index + 1 < len(page_render_blocks) else None
                heading_profile = _markdown_heading_shape_profile(
                    str(block.get("display_text") or block.get("text") or "").strip()
                )
                if (
                    visual_heading_projection_profiles is not None
                    and heading_profile not in visual_heading_projection_profiles
                ):
                    continue
                if _looks_like_markdown_visual_standalone_heading(
                    block,
                    previous_block,
                    next_block,
                    page_height,
                    toc_like_page=toc_like_page,
                    page_blocks=page_render_blocks,
                    landscape_visual_panel=landscape_visual_panel,
                ):
                    updated_block = {
                        **block,
                        "_markdown_visual_standalone_heading": True,
                    }
                    if heading_profile == "cjk_unnumbered_title":
                        updated_block["_markdown_visual_heading_level"] = 6
                    if (
                        str(block.get("semantic_role") or "").strip() == "body_list_item"
                        and _markdown_heading_shape_profile(str(block.get("text") or "")) == "numbered_section_title"
                    ):
                        bbox = _markdown_block_bbox(block)
                        next_bbox = _markdown_block_bbox(next_block or {})
                        next_text = str((next_block or {}).get("display_text") or (next_block or {}).get("text") or "").strip()
                        if (
                            bbox is not None
                            and next_bbox is not None
                            and next_bbox[0] > bbox[0] + 8.0
                            and 0.0 <= next_bbox[1] - bbox[3] <= max(12.0, (bbox[3] - bbox[1]) * 1.25)
                            and next_text
                            and not _markdown_text_starts_bullet_item(next_text)
                        ):
                            updated_block["_markdown_heading_continuation_block_id"] = str(
                                (next_block or {}).get("block_id") or (next_block or {}).get("source_id") or ""
                            ).strip()
                            updated_block["_markdown_heading_continuation_text"] = next_text
                            if index + 1 < len(page_render_blocks):
                                page_render_blocks[index + 1] = {
                                    **page_render_blocks[index + 1],
                                    "_markdown_heading_continuation_consumed": True,
                                }
                    page_render_blocks[index] = updated_block
                elif (
                    str(block.get("block_type") or "").strip().lower() == "text"
                    and str(block.get("semantic_role") or "").strip() == "body_list_item"
                    and _markdown_heading_shape_profile(str(block.get("text") or "")) == "numbered_section_title"
                ):
                    bbox = _markdown_block_bbox(block)
                    next_bbox = _markdown_block_bbox(next_block or {})
                    next_text = str((next_block or {}).get("display_text") or (next_block or {}).get("text") or "").strip()
                    if (
                        bbox is not None
                        and next_bbox is not None
                        and next_bbox[0] > bbox[0] + 8.0
                        and 0.0 <= next_bbox[1] - bbox[3] <= max(12.0, (bbox[3] - bbox[1]) * 1.25)
                        and _markdown_heading_shape_profile(next_text) in {"titlecase_title", "uppercase_title"}
                    ):
                        page_render_blocks[index] = {
                            **block,
                            "_markdown_visual_standalone_heading": True,
                            "_markdown_heading_continuation_block_id": str(
                                (next_block or {}).get("block_id") or (next_block or {}).get("source_id") or ""
                            ).strip(),
                            "_markdown_heading_continuation_text": next_text,
                        }
                        if index + 1 < len(page_render_blocks):
                            page_render_blocks[index + 1] = {
                                **page_render_blocks[index + 1],
                                "_markdown_heading_continuation_consumed": True,
                            }

        deferred_publication_metadata_blocks = [
            item
            for item in page_render_blocks
            if _should_defer_publication_metadata_before_body(item)
            and not _is_markdown_body_citation_text_block(item)
        ]
        deferred_publication_metadata_ids = {
            str(item.get("block_id") or item.get("source_id") or "").strip()
            for item in deferred_publication_metadata_blocks
            if str(item.get("block_id") or item.get("source_id") or "").strip()
        }
        deferred_publication_metadata_flushed = False
        page_equation_source_block_ids = {
            source_block_id
            for page_block in page_render_blocks
            for source_block_id in _markdown_equation_source_block_ids(page_block)
        }
        page_float_owned_text_block_ids = {
            source_block_id
            for page_block in page_render_blocks
            for source_block_id in _markdown_float_owned_text_block_ids(page_block)
        }
        page_metadata_only_source_block_ids = {
            source_block_id
            for page_block in page_render_blocks
            for source_block_id in _markdown_metadata_only_source_block_ids(page_block)
        }
        page_structure_template_owned_text_block_ids = {
            source_block_id
            for page_block in page_render_blocks
            for source_block_id in _markdown_structure_template_owned_text_block_ids(
                page_block,
                page_render_blocks,
            )
        }
        page_structure_template_owned_text_block_ids.update(
            _markdown_structure_template_continuation_marker_owned_ids(
                page_render_blocks,
                page_structure_template_owned_text_block_ids,
            )
        )
        deferred_main_flow_float_segments = _markdown_deferred_main_flow_float_segments(page_render_blocks)
        rendered_deferred_main_flow_float_segment_keys: set[str] = set()
        page_context = {
            "page_number": page_number,
            "page_height": page_height,
            "toc_heading_lookup": toc_heading_lookup,
            "equation_source_block_ids": page_equation_source_block_ids,
            "float_owned_text_block_ids": page_float_owned_text_block_ids,
            "metadata_only_source_block_ids": page_metadata_only_source_block_ids,
            "structure_template_owned_text_block_ids": page_structure_template_owned_text_block_ids,
        }
        rendered_equation_source_block_ids: set[str] = set()

        def flush_deferred_publication_metadata() -> None:
            nonlocal deferred_publication_metadata_flushed
            if deferred_publication_metadata_flushed:
                return
            for metadata_block in deferred_publication_metadata_blocks:
                block_id = str(metadata_block.get("block_id") or metadata_block.get("source_id") or "").strip()
                if block_id and block_id in rendered_publication_metadata_ids:
                    continue
                if block_id:
                    rendered_publication_metadata_ids.add(block_id)
                _append_markdown_text_block(lines, metadata_block)
            deferred_publication_metadata_flushed = True

        def flush_deferred_main_flow_float_segments_before(anchor_block: dict[str, Any] | None) -> None:
            anchor_bbox = _markdown_block_bbox(anchor_block or {})
            if anchor_block is not None and anchor_bbox is None:
                return
            anchor_y = float(anchor_bbox[1]) if anchor_bbox is not None else float("inf")
            for segment in deferred_main_flow_float_segments:
                key = _markdown_deferred_float_segment_key(segment)
                if key in rendered_deferred_main_flow_float_segment_keys:
                    continue
                segment_bbox = _markdown_block_bbox(segment)
                if segment_bbox is not None and float(segment_bbox[1]) >= anchor_y - 0.5:
                    continue
                if _append_markdown_deferred_main_flow_float_segment(lines, segment):
                    rendered_deferred_main_flow_float_segment_keys.add(key)

        block_index = 0
        while block_index < len(page_render_blocks):
            block = page_render_blocks[block_index]
            flush_deferred_main_flow_float_segments_before(block)
            block_type = str(block.get("block_type") or "").strip().lower()
            if block_type == "text" and (
                block.get("_markdown_cross_page_parenthetical_continuation")
                or block.get("_markdown_cross_page_body_continuation")
            ):
                continuation_text = _project_markdown_text_with_inline_formulas(block)
                if continuation_text:
                    consumed_index = block_index
                    if block.get("_markdown_cross_page_body_continuation"):
                        continuation_text, consumed_index = _merge_markdown_same_page_body_continuations_after_cross_page_block(
                            page_render_blocks,
                            block_index,
                            continuation_text,
                            toc_heading_lookup,
                        )
                    last_line_index = next(
                        (
                            index
                            for index in range(len(lines) - 1, -1, -1)
                            if str(lines[index]).strip()
                        ),
                        None,
                    )
                    if last_line_index is not None:
                        lines[last_line_index] = _markdown_linkify_visible_urls(
                            _merge_markdown_text_fragments(lines[last_line_index], continuation_text)
                        )
                        block_index = consumed_index + 1
                        continue
            boundary_kind = _markdown_text_block_boundary_kind(block, toc_heading_lookup)
            if block.get("_markdown_body_flow_continuation"):
                boundary_kind = "body"
            page_block_role = _markdown_page_block_role(block, page_context)
            if block.get("_markdown_body_flow_continuation"):
                page_block_role = {
                    **page_block_role,
                    "role": "body",
                    "boundary_kind": "body",
                    "render_in_main_flow": True,
                }
            if block.get("_markdown_heading_continuation_consumed"):
                block_index += 1
                continue
            if page_block_role["role"] in {
                "metadata_only_text",
                "metadata_only_block",
                "metadata_only_table",
                "metadata_only_figure",
                "absorbed_structure_template",
            }:
                block_index += 1
                continue
            block_id = str(block.get("block_id") or block.get("source_id") or "").strip()
            structured_table_blocks = [
                item
                for item in page_render_blocks
                if str(item.get("block_type") or "").strip().lower() == "table"
            ]
            if (
                block_type == "text"
                and str(block.get("text_projection") or "").strip()
                in {"absorbed_inline_math_residue", "complexity_inline_math", "cross_block_inline_math"}
                and not str(block.get("display_text") or "").strip()
            ):
                block_index += 1
                continue
            if page_block_role["role"] == "equation_source_text" or (block_id and block_id in page_equation_source_block_ids and not block_id.endswith("_prose_cue")):
                block_index += 1
                continue
            if page_block_role["role"] == "figure_legend" or page_block_role["role"] == "table_note":
                if page_block_role["owned_by"] == "float":
                    block_index += 1
                    continue
            if block_id and block_id in rendered_equation_source_block_ids and not block_id.endswith("_prose_cue"):
                block_index += 1
                continue
            if block_id and block_id in deferred_publication_metadata_ids:
                block_index += 1
                continue
            if block_id and block_id in rendered_object_prelude_text_ids:
                block_index += 1
                continue
            if block_id and block_id in page_float_owned_text_block_ids:
                block_index += 1
                continue
            if block_id and block_id in page_structure_template_owned_text_block_ids:
                block_index += 1
                continue
            if (
                not deferred_publication_metadata_flushed
                and boundary_kind == "standalone_heading"
            ):
                flush_deferred_publication_metadata()
            rendered_text_override: str | None = None
            if (
                block_type == "text"
                and page_block_role["role"] != "heading"
                and not block.get("_markdown_visual_standalone_heading")
                and not block.get("_markdown_centered_front_matter_title_cluster")
            ):
                current_text = _project_markdown_text_with_inline_formulas(block)
                current_merge_block = block
                explicit_bullet_anchor_block: dict[str, Any] | None = (
                    block if _markdown_text_starts_bullet_item(current_text) else None
                )
                merged_text_blocks = [block]
                paragraph_base_x0: float | None = None
                if _is_markdown_body_paragraph_text_block(block, toc_heading_lookup):
                    current_bbox = _markdown_block_bbox(block)
                    if current_bbox is not None:
                        paragraph_base_x0 = current_bbox[0]
                next_index = block_index + 1
                while next_index < len(page_render_blocks):
                    next_block = page_render_blocks[next_index]
                    next_block_id = str(next_block.get("block_id") or next_block.get("source_id") or "").strip()
                    next_block_type = str(next_block.get("block_type") or "").strip().lower()
                    if next_block_type != "text":
                        break
                    next_text = _project_markdown_text_with_inline_formulas(next_block)
                    if not next_text:
                        next_index += 1
                        continue
                    can_merge_special = _can_merge_markdown_text_blocks(current_text, next_text)
                    next_boundary_kind = _markdown_text_block_boundary_kind(next_block, toc_heading_lookup)
                    next_role = _markdown_page_block_role(next_block, page_context)
                    if (
                        _markdown_text_block_is_unmarked_list_item(current_merge_block)
                        or _markdown_text_block_is_unmarked_list_item(next_block)
                    ):
                        break
                    current_starts_explicit_bullet = _markdown_text_starts_bullet_item(current_text)
                    next_starts_explicit_bullet = _markdown_text_starts_bullet_item(next_text)
                    if current_starts_explicit_bullet and explicit_bullet_anchor_block is None:
                        explicit_bullet_anchor_block = current_merge_block
                    if current_starts_explicit_bullet and not next_starts_explicit_bullet:
                        if not _markdown_next_block_is_hanging_bullet_continuation(
                            bullet_anchor_block=explicit_bullet_anchor_block or current_merge_block,
                            current_block=current_merge_block,
                            next_block=next_block,
                            current_text=current_text,
                            next_text=next_text,
                        ):
                            break
                    if not current_starts_explicit_bullet and next_starts_explicit_bullet:
                        break
                    if (
                        current_merge_block.get("_markdown_visual_standalone_heading")
                        or current_merge_block.get("_markdown_centered_front_matter_title_cluster")
                        or
                        next_role["role"] == "heading"
                        or next_block.get("_markdown_visual_standalone_heading")
                        or next_block.get("_markdown_centered_front_matter_title_cluster")
                    ):
                        break
                    if (
                        not can_merge_special
                        and boundary_kind == "reference_entry"
                        and next_boundary_kind == "reference_entry"
                    ):
                        reference_merge = _can_merge_markdown_reference_entry_blocks(
                            current_merge_block,
                            next_block,
                            current_text,
                            next_text,
                            paragraph_base_x0,
                        )
                        if reference_merge is False:
                            break
                        if reference_merge is True:
                            can_merge_special = True
                    if (
                        not can_merge_special
                        and next_block_id
                        and next_block_id in page_equation_source_block_ids
                        and not next_block_id.endswith("_prose_cue")
                    ):
                        break
                    if (
                        not can_merge_special
                        and next_block_id
                        and next_block_id in rendered_equation_source_block_ids
                        and not next_block_id.endswith("_prose_cue")
                    ):
                        break
                    if not can_merge_special and next_block_id and next_block_id in deferred_publication_metadata_ids:
                        break
                    if not can_merge_special and next_block_id and next_block_id in page_float_owned_text_block_ids:
                        break
                    if not can_merge_special and next_block_id and next_block_id in page_structure_template_owned_text_block_ids:
                        break
                    if (
                        not can_merge_special
                        and (
                            _find_uri_for_text_block(current_merge_block, uri_link_records_by_page)
                            or _find_uri_for_text_block(next_block, uri_link_records_by_page)
                        )
                    ):
                        break
                    can_merge_body_paragraph = (
                        _is_markdown_body_paragraph_text_block(block, toc_heading_lookup)
                        and _is_markdown_body_paragraph_text_block(next_block, toc_heading_lookup)
                        and _can_merge_markdown_body_paragraph_blocks(
                            current_merge_block,
                            next_block,
                            current_text,
                            next_text,
                            paragraph_base_x0,
                        )
                    )
                    if not (can_merge_special or can_merge_body_paragraph):
                        break
                    current_text = _repair_inline_math_segments_with_prose_cues(
                        _merge_markdown_text_fragments(current_text, next_text)
                    )
                    current_text = _repair_inline_math_residual_delimiters(current_text)
                    rendered_text_override = current_text
                    merged_text_blocks.append(next_block)
                    next_bbox = _markdown_block_bbox(next_block)
                    if (
                        can_merge_body_paragraph
                        and next_bbox is not None
                        and not _markdown_text_starts_bullet_item(current_text)
                        and (
                            paragraph_base_x0 is None
                            or next_bbox[0] < paragraph_base_x0
                        )
                    ):
                        paragraph_base_x0 = next_bbox[0]
                    current_merge_block = next_block
                    block_index = next_index
                    next_index += 1
                if str(block.get("text_projection") or "").strip() == "complexity_inline_math":
                    residue_index = max(block_index, next_index - 1) + 1
                    if residue_index < len(page_render_blocks):
                        residue_block = page_render_blocks[residue_index]
                        if (
                            str(residue_block.get("block_type") or "").strip().lower() == "text"
                            and str(residue_block.get("text_projection") or "").strip() == "complexity_inline_math"
                            and not str(residue_block.get("display_text") or "").strip()
                            and re.fullmatch(r"\s*\(.*\)\s*\.\s*", str(residue_block.get("text") or ""))
                            and current_text
                            and current_text[-1:] not in {".", ";", ":"}
                        ):
                            current_text = f"{current_text}."
                            rendered_text_override = current_text
                            block_index = residue_index
                            merged_text_blocks.append(residue_block)
                if rendered_text_override is not None:
                    block = _merge_markdown_text_block_projection_metadata(block, merged_text_blocks)
            if block_type == "table":
                table_id = str(block.get("table_id") or block.get("block_id") or "").strip()
                if table_id in skipped_table_ids:
                    block_index += 1
                    continue
                if (
                    table_export_mode == "evidence_markdown"
                    and not (
                        merge_safe_evidence_table_chains
                        and _should_merge_evidence_markdown_table_chain(block, table_by_id)
                    )
                ):
                    chain = [block]
                else:
                    chain = _build_table_chain(block, table_by_id)
                for continuation in chain[1:]:
                    continuation_id = str(continuation.get("table_id") or "").strip()
                    if continuation_id:
                        skipped_table_ids.add(continuation_id)
                if len(chain) > 1:
                    block = _merge_continued_table_chain(chain, table_export_mode=table_export_mode)
                trailing_metadata_rows = _markdown_table_trailing_metadata_rows_for_context(
                    block,
                    trailing_metadata_rows_by_context_template_id,
                )
                if trailing_metadata_rows:
                    block = {
                        **block,
                        "_markdown_trailing_metadata_rows": trailing_metadata_rows,
                    }
                table_previous_visible_title = _markdown_table_chain_previous_visible_title(
                    block,
                    visible_title_by_table_id,
                ) or previous_visible_heading_title
                _append_markdown_table(
                    lines,
                    block,
                    table_export_mode=table_export_mode,
                    previous_visible_title=table_previous_visible_title,
                    suppress_note_norms=_markdown_table_suppress_note_norms_for_context(
                        block,
                        suppressed_note_norms_by_context_template_id,
                    ),
                )
                previous_visible_heading_title = _markdown_table_visible_title(block)
                if table_id:
                    visible_title_by_table_id[table_id] = previous_visible_heading_title
            elif page_block_role["role"] == "structure_template":
                following_blocks = page_render_blocks[block_index + 1 :]
                deferred_note_texts = _structure_template_deferred_note_texts_for_internal_following_tables(
                    block,
                    following_blocks,
                )
                structure_template_previous_visible_title = _markdown_structure_template_chain_previous_visible_title(
                    block,
                    visible_title_by_structure_template_id,
                ) or previous_visible_heading_title
                suppress_form_rows = _structure_template_context_rendered_by_following_composite_table(
                    block,
                    following_blocks,
                )
                prelude_texts, prelude_ids = _markdown_structure_template_above_title_prelude(
                    block,
                    page_render_blocks,
                    already_rendered_source_ids=rendered_main_flow_text_ids,
                )
                for prelude_text in prelude_texts:
                    lines.append(_markdown_linkify_visible_urls(_markdown_escape_inline_text(prelude_text)))
                    lines.append("")
                rendered_object_prelude_text_ids.update(prelude_ids)
                _append_markdown_structure_template_with_deferred_notes(
                    lines,
                    block,
                    deferred_note_texts=deferred_note_texts,
                    previous_visible_title=structure_template_previous_visible_title,
                    suppress_form_rows=suppress_form_rows,
                    nearby_blocks=page_render_blocks,
                )
                if suppress_form_rows:
                    template_id_for_notes = str(block.get("structure_template_id") or block.get("block_id") or "").strip()
                    if template_id_for_notes:
                        note_norms = _markdown_structure_template_note_norms_to_suppress_for_following_table(
                            block,
                            following_blocks,
                        )
                        if note_norms:
                            suppressed_note_norms_by_context_template_id.setdefault(template_id_for_notes, set()).update(note_norms)
                        metadata_rows = _markdown_structure_template_trailing_metadata_rows(block)
                        if metadata_rows:
                            trailing_metadata_rows_by_context_template_id[template_id_for_notes] = metadata_rows
                structure_template_id = str(block.get("structure_template_id") or block.get("block_id") or "").strip()
                structure_template_visible_title = _structure_template_markdown_heading(block)
                if (
                    structure_template_id
                    and _structure_template_markdown_heading_is_visible(
                        block,
                        structure_template_visible_title,
                        structure_template_previous_visible_title,
                    )
                ):
                    visible_title_by_structure_template_id[structure_template_id] = structure_template_visible_title
                if deferred_note_texts:
                    template_bbox = _markdown_block_bbox(block)
                    scan_index = block_index + 1
                    while scan_index < len(page_render_blocks):
                        candidate = page_render_blocks[scan_index]
                        candidate_type = str(candidate.get("block_type") or "").strip().lower()
                        candidate_bbox = _markdown_block_bbox(candidate)
                        if candidate_type in {"structure_template", "image"}:
                            break
                        if (
                            template_bbox is None
                            or candidate_bbox is None
                            or not _markdown_bbox_inside(candidate_bbox, template_bbox, tolerance=3.0)
                        ):
                            break
                        if candidate_type != "table":
                            scan_index += 1
                            continue
                        table_id = str(candidate.get("table_id") or candidate.get("block_id") or "").strip()
                        if table_id not in skipped_table_ids and not _as_string_list(candidate.get("continued_from")):
                            if (
                                table_export_mode == "evidence_markdown"
                                and not (
                                    merge_safe_evidence_table_chains
                                    and _should_merge_evidence_markdown_table_chain(candidate, table_by_id)
                                )
                            ):
                                chain = [candidate]
                            else:
                                chain = _build_table_chain(candidate, table_by_id)
                            for continuation in chain[1:]:
                                continuation_id = str(continuation.get("table_id") or "").strip()
                                if continuation_id:
                                    skipped_table_ids.add(continuation_id)
                            render_table = (
                                _merge_continued_table_chain(chain, table_export_mode=table_export_mode)
                                if len(chain) > 1
                                else candidate
                            )
                            candidate_previous_visible_title = _markdown_table_chain_previous_visible_title(
                                render_table,
                                visible_title_by_table_id,
                            ) or previous_visible_heading_title
                            _append_markdown_table(
                                lines,
                                render_table,
                                table_export_mode=table_export_mode,
                                previous_visible_title=candidate_previous_visible_title,
                            )
                            previous_visible_heading_title = _markdown_table_visible_title(render_table)
                            if table_id:
                                visible_title_by_table_id[table_id] = previous_visible_heading_title
                        if table_id:
                            skipped_table_ids.add(table_id)
                        scan_index += 1
                    _append_markdown_deferred_structure_template_notes(lines, block, deferred_note_texts)
                previous_visible_heading_title = ""
            elif block_type == "image":
                if (
                    skip_uncaptioned_image_placeholders
                    and not embed_images
                    and not _markdown_image_has_renderable_semantics(block)
                ):
                    block_index += 1
                    continue
                suppress_embedded_image_text = (
                    str(image_text_mode or "").strip().lower() in {"caption_only", "review", "none"}
                    or _markdown_image_embedded_text_owned_by_structured_table(
                        block,
                        structured_table_blocks,
                    )
                )
                _append_markdown_image(
                    lines,
                    block,
                    embed_images=embed_images,
                    suppress_embedded_text=suppress_embedded_image_text,
                    previous_visible_title=previous_visible_heading_title,
                )
                previous_visible_heading_title = ""
            elif page_block_role["role"] == "algorithm_pseudocode":
                _append_markdown_algorithm_pseudocode(lines, block)
                previous_visible_heading_title = ""
            elif block_type == "equation" or str(block.get("semantic_role") or "").strip() == "display_equation":
                _append_markdown_equation(lines, block)
                rendered_equation_source_block_ids.update(_markdown_equation_source_block_ids(block))
                previous_visible_heading_title = ""
            elif page_block_role["role"] == "toc":
                if render_toc_blocks:
                    _append_markdown_toc_block(lines, block)
                block_index += 1
                continue
            elif page_block_role["role"] == "footnote":
                footnote_id = str(block.get("footnote_id") or block.get("block_id") or "").strip()
                if footnote_id and footnote_id in rendered_footnote_ids:
                    block_index += 1
                    continue
                _append_markdown_footnote_definition(footnote_definition_lines, block)
                if footnote_id:
                    rendered_footnote_ids.add(footnote_id)
            elif page_block_role["role"] == "footnote_continuation":
                block_index += 1
                continue
            elif page_block_role["role"] == "structure_template_owned_text":
                block_index += 1
                continue
            elif page_block_role["role"] in {"page_number", "page_header", "page_footer", "logo", "publication_metadata"}:
                if not _should_render_publication_metadata_markdown_block(block):
                    block_index += 1
                    continue
                block_id = str(block.get("block_id") or block.get("source_id") or "").strip()
                if block_id and block_id in rendered_publication_metadata_ids:
                    block_index += 1
                    continue
                if block_id:
                    rendered_publication_metadata_ids.add(block_id)
                _append_markdown_text_block(lines, block, text_override=rendered_text_override)
                previous_visible_heading_title = ""
                block_index += 1
                continue
            else:
                if rendered_text_override is not None:
                    _append_markdown_text_block(lines, block, text_override=rendered_text_override)
                    rendered_main_flow_text_ids.update(
                        str(rendered_block.get("block_id") or rendered_block.get("source_id") or "").strip()
                        for rendered_block in merged_text_blocks
                        if str(rendered_block.get("block_id") or rendered_block.get("source_id") or "").strip()
                    )
                    previous_visible_heading_title = ""
                else:
                    if page_block_role["render_in_main_flow"]:
                        heading_for_visible_owner = _markdown_heading_for_text_block(block, toc_heading_lookup)
                        if block.get("_markdown_centered_front_matter_title_cluster"):
                            _append_markdown_centered_front_matter_title_block(lines, block)
                        else:
                            _append_markdown_text_block_with_heading_context(
                                lines,
                                block,
                                toc_heading_lookup,
                                uri_link_records_by_page,
                            )
                        if block_id:
                            rendered_main_flow_text_ids.add(block_id)
                        previous_visible_heading_title = (
                            heading_for_visible_owner[0]
                            if heading_for_visible_owner is not None
                            else ""
                        )
            block_index += 1
        flush_deferred_main_flow_float_segments_before(None)
        previous_page_last_flow_text_block = None
        for candidate in reversed(page_render_blocks):
            if str(candidate.get("block_type") or "").strip().lower() != "text":
                continue
            if candidate.get("_markdown_heading_continuation_consumed"):
                continue
            if candidate.get("_markdown_cross_page_parenthetical_continuation"):
                continue
            if candidate.get("_markdown_cross_page_body_continuation"):
                continue
            candidate_boundary = _markdown_text_block_boundary_kind(candidate, toc_heading_lookup)
            if candidate.get("_markdown_body_flow_continuation"):
                candidate_boundary = "body"
            if candidate_boundary != "body":
                continue
            previous_page_last_flow_text_block = {
                **candidate,
                "_markdown_page_height": page_height,
            }
            break
    if footnote_definition_lines:
        lines.extend(footnote_definition_lines)
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


def _compact_evidence_snapshot_text(value: Any, *, limit: int = 220) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)].rstrip()}..."


def _evidence_snapshot_table_title_is_continuation_marker(title: Any) -> bool:
    text = str(title or "").strip()
    if not text:
        return False
    return bool(
        re.fullmatch(
            r"(?:table|tab\.?|\u8868)\s*[A-Za-z0-9\u4e00-\u9fff]*\s*[:.\-\uff1a]?\s*(?:continued|continuation|\u7eed\u8868|\u7eed)\.?",
            text,
            re.IGNORECASE,
        )
    )


def _format_evidence_snapshot_bbox(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 4:
        return ""
    try:
        parts = [f"{float(item):.1f}".rstrip("0").rstrip(".") for item in value]
    except (TypeError, ValueError):
        return ""
    return f"[{', '.join(parts)}]"


def _collect_ast_blocks(document: dict[str, Any]) -> list[dict[str, Any]]:
    document_ast = document.get("document_ast", {}) or {}
    blocks: list[dict[str, Any]] = []
    for page in document_ast.get("pages", []) or []:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page")
        for block in page.get("blocks", []) or []:
            if not isinstance(block, dict):
                continue
            if page_number is not None and "page" not in block:
                block = {**block, "page": page_number}
            blocks.append(block)
    return blocks


def _unique_evidence_objects(
    primary_items: Iterable[dict[str, Any]],
    fallback_blocks: Iterable[dict[str, Any]],
    *,
    id_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(primary_items) + list(fallback_blocks):
        if not isinstance(item, dict):
            continue
        object_id = ""
        for key in id_keys:
            object_id = str(item.get(key) or "").strip()
            if object_id:
                break
        if not object_id:
            object_id = "|".join(
                [
                    str(item.get("page") or ""),
                    str(item.get("block_type") or ""),
                    str(item.get("bbox") or ""),
                    _compact_evidence_snapshot_text(item.get("title") or item.get("caption_text") or item.get("text"), limit=80),
                ]
            )
        if object_id in seen:
            continue
        seen.add(object_id)
        objects.append(item)
    return objects


def _append_evidence_object_rows(
    lines: list[str],
    objects: list[dict[str, Any]],
    *,
    heading: str,
    id_keys: tuple[str, ...],
    title_keys: tuple[str, ...],
    max_rows: int = 20,
) -> None:
    if not objects:
        return
    lines.append(f"#### {heading}")
    for item in objects[:max_rows]:
        object_id = ""
        for key in id_keys:
            object_id = str(item.get(key) or "").strip()
            if object_id:
                break
        page = item.get("page") or item.get("page_number") or "?"
        bbox = _format_evidence_snapshot_bbox(item.get("bbox"))
        title = ""
        for key in title_keys:
            title = _compact_evidence_snapshot_text(item.get(key), limit=180)
            if title:
                break
        if heading in {"Table Evidence Objects", "Image Evidence Objects"}:
            title = ""
        if heading == "Table Evidence Objects" and _evidence_snapshot_table_title_is_continuation_marker(title):
            title = ""
        descriptors = [f"page {page}"]
        if bbox:
            descriptors.append(f"bbox {bbox}")
        if title:
            descriptors.append(title)
        if object_id:
            lines.append(f"- `{object_id}`: {'; '.join(descriptors)}")
        else:
            lines.append(f"- {'; '.join(descriptors)}")
    if len(objects) > max_rows:
        lines.append(f"- ... {len(objects) - max_rows} more object(s)")
    lines.append("")


def _build_evidence_snapshot_inventory_markdown_section(document: dict[str, Any]) -> list[str]:
    ast_blocks = _collect_ast_blocks(document)
    metadata = document.get("metadata", {}) or {}
    page_count = metadata.get("page_count")
    if not page_count:
        ast_pages = (document.get("document_ast", {}) or {}).get("pages", []) or []
        page_count = len([page for page in ast_pages if isinstance(page, dict)])
    block_counts: Counter[str] = Counter(
        str(block.get("block_type") or "unknown").strip().lower() or "unknown"
        for block in ast_blocks
    )
    table_blocks = [
        block
        for block in ast_blocks
        if str(block.get("block_type") or "").strip().lower() == "table"
    ]
    image_blocks_from_ast = [
        block
        for block in ast_blocks
        if str(block.get("block_type") or "").strip().lower() == "image"
    ]
    equation_blocks = [
        block
        for block in ast_blocks
        if str(block.get("block_type") or "").strip().lower() == "equation"
        or str(block.get("semantic_role") or "").strip() == "display_equation"
    ]
    table_objects = _unique_evidence_objects(
        [item for item in document.get("table_asts", []) or [] if isinstance(item, dict)],
        table_blocks,
        id_keys=("table_id", "block_id", "source_id"),
    )
    image_objects = _unique_evidence_objects(
        [item for item in document.get("image_blocks", []) or [] if isinstance(item, dict)],
        image_blocks_from_ast,
        id_keys=("image_id", "block_id", "source_id"),
    )
    evidence_records = [
        item
        for item in document.get("content_evidence", []) or []
        if isinstance(item, dict)
    ]

    lines = ["### Evidence Snapshot Inventory", ""]
    if page_count:
        lines.append(f"- Pages: {page_count}")
    lines.append(f"- AST blocks: {len(ast_blocks)}")
    if block_counts:
        block_summary = ", ".join(f"{key}={value}" for key, value in sorted(block_counts.items()))
        lines.append(f"- AST block types: {block_summary}")
    lines.append(f"- Tables: {len(table_objects)}")
    lines.append(f"- Images: {len(image_objects)}")
    lines.append(f"- Equations: {len(equation_blocks)}")
    lines.append(f"- Content evidence records: {len(evidence_records)}")
    lines.append("")

    _append_evidence_object_rows(
        lines,
        table_objects,
        heading="Table Evidence Objects",
        id_keys=("table_id", "block_id", "source_id"),
        title_keys=("title", "caption_text", "text"),
    )
    _append_evidence_object_rows(
        lines,
        image_objects,
        heading="Image Evidence Objects",
        id_keys=("image_id", "block_id", "source_id"),
        title_keys=("caption_text", "title", "figure_ref", "content_text"),
    )
    if equation_blocks:
        _append_evidence_object_rows(
            lines,
            equation_blocks,
            heading="Equation Evidence Objects",
            id_keys=("equation_id", "block_id", "source_id"),
            title_keys=("equation_label", "latex_text", "text"),
        )

    embedded_image_evidence: list[tuple[str, str]] = []
    for item in image_objects:
        image_id = str(item.get("image_id") or item.get("block_id") or item.get("source_id") or "").strip()
        embedded_text = _compact_evidence_snapshot_text(
            item.get("content_text") or item.get("embedded_text"),
            limit=320,
        )
        if not embedded_text:
            for segment in item.get("content_segments", []) or []:
                if not isinstance(segment, dict):
                    continue
                role = str(segment.get("role") or "").strip()
                if role not in {"embedded_text", "embedded_code"}:
                    continue
                embedded_text = _compact_evidence_snapshot_text(segment.get("text"), limit=320)
                if embedded_text:
                    break
        if embedded_text and not _looks_like_snapshot_embedded_code_evidence(embedded_text):
            embedded_text = ""
        if embedded_text:
            embedded_image_evidence.append((image_id or "image", embedded_text))
    if embedded_image_evidence:
        lines.append("#### Image embedded evidence")
        for image_id, embedded_text in embedded_image_evidence[:12]:
            lines.append(f"- `{image_id}` embedded evidence: {embedded_text}")
        if len(embedded_image_evidence) > 12:
            lines.append(f"- ... {len(embedded_image_evidence) - 12} more image evidence item(s)")
        lines.append("")

    return lines


def _looks_like_snapshot_embedded_code_evidence(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if re.search(r"<\s*/?\s*[A-Za-z][A-Za-z0-9:_-]*(?:\s|>|/>)", cleaned):
        return True
    if re.search(r"\b(?:xml|dtd|schema|xlink|href|controlled-vocabulary|cn-envelope)\b", cleaned, re.IGNORECASE):
        return True
    return False


def _build_full_markdown(
    parsed_documents: list[dict[str, Any]],
    *,
    markdown_profile: str = "ind-evidence",
) -> str:
    normalized_profile = str(markdown_profile or "ind-evidence").strip().lower()
    if normalized_profile == "ind-review":
        return _build_ind_review_markdown(parsed_documents)
    if normalized_profile not in {"ind-evidence", "legacy", "full"}:
        raise ValueError(f"Unsupported markdown profile: {markdown_profile}")
    lines = [
        "# IND Parse Snapshot (Full)",
        "",
        f"Generated at: {_utc_now()}",
        "",
    ]
    for index, document in enumerate(parsed_documents):
        document = _normalize_markdown_table_note_fields(document)
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

        evidence_inventory_sections = _build_evidence_snapshot_inventory_markdown_section(document)
        if evidence_inventory_sections:
            lines.extend(evidence_inventory_sections)
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

        body_sections = _build_document_body_markdown_sections(document, table_export_mode="auto_semantic")
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


def _build_ind_review_markdown(parsed_documents: list[dict[str, Any]]) -> str:
    lines = [
        "# IND Review Markdown",
        "",
        f"Generated at: {_utc_now()}",
        "",
    ]
    for index, document in enumerate(parsed_documents):
        document = _normalize_markdown_table_note_fields(document)
        filename = str(document.get("filename") or f"document-{index + 1}").strip()
        lines.append(f"## {filename}")
        lines.append("")

        toc_sections = _build_toc_markdown_sections([document], heading_level=3)
        if toc_sections:
            lines.extend(toc_sections)
            lines.append("")

        body_sections = _build_document_body_markdown_sections(
            document,
            embed_images=True,
            table_export_mode="evidence_markdown",
            image_text_mode="caption_only",
            body_heading=None,
            visual_heading_projection=True,
            visual_heading_projection_profiles={"cjk_unnumbered_title"},
            merge_safe_evidence_table_chains=True,
        )
        if body_sections:
            lines.extend(body_sections)
            lines.append("")
            continue

        preview_text = str(document.get("text", "")).strip()
        if preview_text:
            lines.append(preview_text)
            lines.append("")

    markdown = "\n".join(lines).strip()
    _attach_ind_review_render_audits(parsed_documents, markdown)
    return markdown


def _attach_ind_review_render_audits(parsed_documents: list[dict[str, Any]], markdown: str) -> None:
    for document in parsed_documents:
        if not isinstance(document, dict):
            continue
        metadata = document.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            continue
        metadata["ind_review_render_audit"] = _build_ind_review_render_audit(document, markdown)


def _build_ind_review_render_audit(document: dict[str, Any], markdown: str) -> dict[str, Any]:
    architecture = document.get("ind_architecture")
    architecture = architecture if isinstance(architecture, dict) else {}
    render_consumption = architecture.get("render_consumption_audit")
    render_consumption = render_consumption if isinstance(render_consumption, dict) else {}
    study_objects = (architecture.get("study_objects") or {}).get("objects") or []
    normalized_markdown = _normalize_ind_review_visibility_text(markdown)
    heading_duplicate_audit = _build_ind_review_heading_like_duplicate_audit(document, markdown)
    title_violations: list[dict[str, Any]] = []
    skipped_titles: list[dict[str, Any]] = []
    checked_count = 0
    for study in study_objects:
        if not isinstance(study, dict):
            continue
        title = str(study.get("title") or "").strip()
        if not title:
            continue
        render_plan = study.get("render_plan") if isinstance(study.get("render_plan"), dict) else {}
        components = render_plan.get("components") if isinstance(render_plan, dict) else []
        title_components = [
            component
            for component in components or []
            if isinstance(component, dict) and component.get("type") == "title"
        ]
        if not any(
            component.get("source_policy")
            or component.get("source_block_ids")
            or component.get("source_object_ids")
            for component in title_components
        ):
            skipped_titles.append(
                {
                    "study_object_id": str(study.get("study_object_id") or "").strip(),
                    "title": title,
                    "reason": "missing_source_binding",
                }
            )
            continue
        checked_count += 1
        visible_count = normalized_markdown.count(_normalize_ind_review_visibility_text(title))
        if visible_count > 1:
            allowed_evidence = _ind_review_allowed_heading_duplicate_evidence(
                document,
                title,
                markdown=markdown,
            )
            if allowed_evidence.get("reason"):
                continue
            violation = {
                "study_object_id": str(study.get("study_object_id") or "").strip(),
                "title": title,
                "visible_count": visible_count,
            }
            title_violations.append(violation)
            skipped_titles.append(
                {
                    **violation,
                    "reason": "visible_duplicate_requires_render_resolution",
                }
            )
    return {
        "profile": "ind-review",
        "policies": [
            "render_plan_title_components_are_visible_at_most_once",
            "architecture_source_block_consumption_must_be_unique_before_markdown_render",
        ],
        "architecture_visible_block_source_duplicate_count": int(
            render_consumption.get("visible_block_source_duplicate_count", 0) or 0
        ),
        "architecture_visible_block_source_duplicate_violations": list(
            render_consumption.get("visible_block_source_duplicate_violations", []) or []
        )[:20],
        "study_title_eligibility_policy": "only_source_aware_title_components_are_hard_checked",
        "study_title_checked_count": checked_count,
        "study_title_skipped_count": len(skipped_titles),
        "study_title_skipped": skipped_titles[:40],
        "study_title_duplicate_count": len(title_violations),
        "study_title_duplicate_violations": title_violations[:20],
        **heading_duplicate_audit,
    }


def _build_ind_review_heading_like_duplicate_audit(document: dict[str, Any], markdown: str) -> dict[str, Any]:
    duplicate_items = _ind_review_heading_like_duplicate_items(markdown)
    allowed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for item in duplicate_items:
        title = str(item.get("title") or "").strip()
        allowed_evidence = _ind_review_allowed_heading_duplicate_evidence(document, title, markdown=markdown)
        reason = str(allowed_evidence.get("reason") or "")
        if reason:
            allowed.append({**item, **allowed_evidence})
            continue
        if _ind_review_document_scaffold_heading_duplicate(title):
            allowed.append({**item, "reason": "document_scaffold"})
            continue
        unresolved.append({**item, "reason": "requires_review"})
    return {
        "heading_like_duplicate_policy": "classify_duplicate_visible_headings_as_allowed_variants_or_unresolved",
        "heading_like_duplicate_allowed_count": len(allowed),
        "heading_like_duplicate_allowed": allowed[:40],
        "heading_like_duplicate_unresolved_count": len(unresolved),
        "heading_like_duplicate_unresolved": unresolved[:40],
    }


def _ind_review_heading_like_duplicate_items(markdown: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    examples: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(str(markdown or "").splitlines(), start=1):
        text = _ind_review_heading_like_line_text(line)
        if not text:
            continue
        normalized = _normalize_ind_review_visibility_text(text)
        if not normalized or len(normalized) < 6:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
        examples.setdefault(
            normalized,
            {
                "title": text,
                "first_line": line_number,
            },
        )
    return [
        {
            **examples[normalized],
            "visible_count": count,
        }
        for normalized, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count > 1
    ]


def _ind_review_heading_like_line_text(line: str) -> str:
    stripped = str(line or "").strip()
    if not stripped:
        return ""
    if stripped.startswith("#"):
        return re.sub(r"^#+\s*", "", stripped).strip()
    if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
        return stripped.strip("*").strip()
    if re.match(r"^2\.6\.", stripped):
        return stripped
    return ""


def _ind_review_document_scaffold_heading_duplicate(title: str) -> bool:
    normalized = _normalize_ind_review_visibility_text(title)
    if not normalized:
        return False
    if normalized.endswith(".pdf"):
        return True
    if normalized in {"IND Review Markdown"}:
        return True
    return bool(
        re.search(r"(?<![A-Za-z])ICH(?![A-Za-z])", normalized, re.IGNORECASE)
        and "指导原则" in normalized
        and len(normalized) <= 40
    )


def _ind_review_allowed_heading_duplicate_reason(document: dict[str, Any], title: str) -> str:
    return str(_ind_review_allowed_heading_duplicate_evidence(document, title).get("reason") or "")


def _ind_review_allowed_heading_duplicate_evidence(
    document: dict[str, Any],
    title: str,
    *,
    markdown: str | None = None,
) -> dict[str, Any]:
    normalized_title = _normalize_ind_review_visibility_text(title)
    if not normalized_title:
        return {}
    table_image_title_evidence = _ind_review_table_image_title_duplicate_evidence(document, title)
    if table_image_title_evidence:
        return table_image_title_evidence
    matching_templates = [
        template
        for template in document.get("structure_templates", []) or []
        if isinstance(template, dict)
        and _normalize_ind_review_visibility_text(
            _clean_structure_template_markdown_title(template.get("title") or "")
        ) == normalized_title
    ]
    continuation_heading_evidence = _ind_review_guidance_template_continuation_heading_evidence(
        document,
        title,
        markdown=markdown,
    )
    if continuation_heading_evidence:
        return continuation_heading_evidence
    if len(matching_templates) < 2:
        return {}
    profiles = {str(template.get("template_profile") or "").strip() for template in matching_templates}
    domains = {str(template.get("ownership_domain") or "").strip() for template in matching_templates}
    if "tabular_form_template" in profiles:
        reason = "structure_template_variant"
    elif "populated_study_metadata" in profiles and "study_context" in domains:
        reason = "populated_study_metadata_variant"
    else:
        return {}
    return {
        "reason": reason,
        "source_object_family": "structure_template",
        "source_objects": _ind_review_structure_template_duplicate_sources(matching_templates),
        "pages": _ind_review_sorted_unique_values(template.get("page") for template in matching_templates),
        "template_profiles": _ind_review_sorted_unique_values(
            template.get("template_profile") for template in matching_templates
        ),
        "ownership_domains": _ind_review_sorted_unique_values(
            template.get("ownership_domain") for template in matching_templates
        ),
        "data_populations": _ind_review_sorted_unique_values(
            template.get("data_population") for template in matching_templates
        ),
    }


def _ind_review_table_image_title_duplicate_evidence(document: dict[str, Any], title: str) -> dict[str, Any]:
    normalized_title = _normalize_ind_review_visibility_text(title)
    if not normalized_title:
        return {}
    matching_tables = [
        table
        for table in document.get("table_asts", []) or []
        if isinstance(table, dict)
        and _normalize_ind_review_visibility_text(table.get("title") or "") == normalized_title
    ]
    if not matching_tables:
        return {}

    image_title_sources: list[dict[str, Any]] = []
    for image in document.get("image_blocks", []) or []:
        if not isinstance(image, dict):
            continue
        for segment in image.get("content_segments", []) or []:
            if not isinstance(segment, dict):
                continue
            if str(segment.get("role") or "").strip() != "nearby_context":
                continue
            if str(segment.get("relation") or "").strip().lower() != "above":
                continue
            if _normalize_ind_review_visibility_text(segment.get("text") or "") != normalized_title:
                continue
            if not _markdown_nearby_context_segment_is_float_title(image, segment):
                continue
            image_title_sources.append(
                {
                    "source_object_id": str(image.get("image_id") or image.get("block_id") or "").strip(),
                    "page": image.get("page"),
                    "source_block_id": str(
                        segment.get("source_block_id")
                        or segment.get("block_id")
                        or segment.get("source_id")
                        or ""
                    ).strip(),
                    "relation": "above",
                    "gap": segment.get("gap"),
                }
            )
    if not image_title_sources:
        return {}

    table_sources = [
        {
            "source_object_id": str(table.get("table_id") or table.get("block_id") or "").strip(),
            "page": table.get("page"),
            "semantic_role": str(table.get("semantic_role") or "").strip(),
            "table_family": str(
                ((table.get("semantic_projection_v2") or {}).get("table_family") if isinstance(table.get("semantic_projection_v2"), dict) else "")
                or ""
            ).strip(),
        }
        for table in matching_tables[:10]
    ]
    return {
        "reason": "table_image_title_repetition",
        "source_object_family": "table_and_image",
        "source_objects": [
            *table_sources,
            *image_title_sources[:10],
        ],
        "pages": _ind_review_sorted_unique_values(
            [source.get("page") for source in table_sources + image_title_sources]
        ),
        "template_profiles": ["table_title", "image_above_title"],
        "ownership_domains": ["business_table", "figure"],
        "data_populations": [],
    }


def _ind_review_repeated_study_template_continuation_heading(title: str) -> bool:
    normalized = _normalize_ind_review_visibility_text(title)
    if not normalized:
        return False
    return bool(
        re.match(r"^2\.6(?:\.\d+){1,}[A-Z]?\s+\S", normalized)
        and re.search(r"(?:续|continued)", normalized, re.IGNORECASE)
        and re.search(
            r"(?:试验编号|报告标题|致癌性|重复给药毒性|遗传毒性|生殖毒性|毒性|药代动力学)",
            normalized,
            re.IGNORECASE,
        )
    )


def _ind_review_guidance_template_continuation_heading_evidence(
    document: dict[str, Any],
    title: str,
    *,
    markdown: str | None = None,
) -> dict[str, Any]:
    if not _ind_review_repeated_study_template_continuation_heading(title):
        return {}
    document_kind_evidence = _ind_review_guidance_template_example_document_evidence(document, markdown=markdown)
    if not document_kind_evidence:
        return {}
    return {
        "reason": "guidance_template_example_continuation_heading",
        "source_object_family": "heading",
        "source_objects": [],
        "pages": [],
        "template_profiles": ["section_heading_continuation"],
        "ownership_domains": ["document_heading"],
        "data_populations": [],
        "document_kind": "guidance_template_example",
        "allowance_basis": "guidance_template_example_document",
        "continuation_evidence": {
            "title_matches_study_template_continuation_heading": True,
            **document_kind_evidence,
        },
    }


def _ind_review_guidance_template_example_document_evidence(
    document: dict[str, Any],
    *,
    markdown: str | None = None,
) -> dict[str, Any]:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    explicit_kind = _normalize_ind_review_visibility_text(
        " ".join(
            str(metadata.get(key) or "")
            for key in ("document_kind", "submission_kind", "dossier_kind", "content_kind")
        )
    )
    if re.search(r"(?:regulatory_submission|actual_submission|real_submission|申报资料|真实申报|注册申报)", explicit_kind):
        return {}

    haystack = _normalize_ind_review_visibility_text(
        " ".join(
            value
            for value in (
                str(document.get("filename") or ""),
                str(document.get("source_path") or ""),
                str(document.get("text") or "")[:50000],
                str(markdown or "")[:50000],
            )
            if value
        )
    )
    if not haystack:
        return {}

    signals: list[str] = []
    signal_patterns = [
        ("ich_m4s_r2", r"(?:M4S\s*\(?R2\)?|安全性-M4S|ICH\s*三方协调|ICH\s+guideline|ICH)"),
        ("guidance_principle", r"(?:指导原则|指南|guidance|guideline)"),
        ("appendix_template", r"(?:附录\s*B|非临床列表总结-模板|列表总结-模板|模板|template)"),
        ("example_document", r"(?:示例|举例|example)"),
        ("study_template_placeholder", r"(?:供试品[:：]\s*\(?[12]\)?|报告标题[:：]\s*供试品|试验编号[:：]?\s*\(?续\)?)"),
    ]
    for name, pattern in signal_patterns:
        if re.search(pattern, haystack, re.IGNORECASE):
            signals.append(name)

    if "ich_m4s_r2" in signals and ("guidance_principle" in signals or "appendix_template" in signals):
        pass
    elif "appendix_template" in signals and "study_template_placeholder" in signals:
        pass
    elif len(set(signals) & {"guidance_principle", "appendix_template", "example_document"}) >= 2:
        pass
    else:
        return {}

    return {
        "document_kind": "guidance_template_example",
        "document_kind_signals": signals,
    }


def _ind_review_structure_template_duplicate_sources(
    templates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for template in templates:
        object_id = (
            str(template.get("structure_template_id") or "").strip()
            or str(template.get("template_id") or "").strip()
            or str(template.get("id") or "").strip()
        )
        sources.append(
            {
                "source_object_id": object_id,
                "page": template.get("page"),
                "template_profile": str(template.get("template_profile") or "").strip(),
                "ownership_domain": str(template.get("ownership_domain") or "").strip(),
                "data_population": str(template.get("data_population") or "").strip(),
            }
        )
    return sources[:20]


def _ind_review_sorted_unique_values(values: Iterable[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            cleaned: Any = value.strip()
        else:
            cleaned = value
        if cleaned == "":
            continue
        key = str(cleaned)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return sorted(unique, key=lambda item: (str(type(item)), str(item)))


def _normalize_ind_review_visibility_text(text: str) -> str:
    normalized = str(text or "").replace("**", " ")
    normalized = re.sub(r"[`*_]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


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
                    "toc_sequence_id": str(payload.get("toc_sequence_id") or "").strip() or None,
                    "toc_sequence_ids": list(payload.get("toc_sequence_ids", []) or []),
                    "toc_sequence_titles": list(payload.get("toc_sequence_titles", []) or []),
                    "toc_sequence_pages": list(payload.get("toc_sequence_pages", []) or []),
                    "toc_sequence_selection": str(payload.get("toc_sequence_selection") or "").strip() or None,
                    "toc_sequence_alignment_scores": list(payload.get("toc_sequence_alignment_scores", []) or []),
                    "excluded_toc_sequence_count": int(payload.get("excluded_toc_sequence_count", 0) or 0),
                    "excluded_toc_sequence_ids": list(payload.get("excluded_toc_sequence_ids", []) or []),
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


def _build_application_identity_projection(
    package_inventory: dict[str, Any],
    submission_scope: dict[str, Any],
    parsed_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project deterministic root naming and semantic application-type evidence."""

    project_context = dict(submission_scope.get("ectd_project_context", {}) or {})
    sequence_packages = [
        dict(item or {})
        for item in list(project_context.get("sequence_packages", []) or [])
    ]
    vocabulary_contract = build_ectd_vocabulary_rule_contract()
    sequence_semantic_contract = build_ectd_sequence_semantic_contract(
        vocabulary_contract=vocabulary_contract,
    )
    content_signals: list[dict[str, Any]] = []
    semantic_findings: list[dict[str, Any]] = []
    for document in parsed_documents or []:
        metadata = dict(document.get("metadata", {}) or {})
        for signal in list(metadata.get("ectd_application_type_evidence") or []):
            signal_payload = dict(signal or {})
            signal_payload.setdefault("source_path", document.get("source_path"))
            signal_payload.setdefault("filename", document.get("filename"))
            content_signals.append(signal_payload)
    applications: list[dict[str, Any]] = []
    for application in list(package_inventory.get("application_roots", []) or []):
        application_payload = dict(application or {})
        root_name = str(application_payload.get("name") or "").strip()
        matching_packages = [
            package
            for package in sequence_packages
            if str(package.get("application_root_name") or package.get("application_key") or "").strip().lower()
            == root_name.lower()
        ]
        vocabulary_checks: list[dict[str, Any]] = []
        sequence_semantic_rows = [
            {
                "sequence_number": str(package.get("sequence_number") or package.get("sequence_name") or "").strip(),
                "related_sequence": str(package.get("related_sequence_number") or "").strip(),
                "regulatory_activity_type": str(package.get("regulatory_activity_type") or "").strip(),
                "sequence_type": str(package.get("sequence_type") or "").strip(),
                "sequence_description": str(package.get("sequence_description") or "").strip(),
            }
            for package in matching_packages
        ]
        sequence_semantic_validation = validate_ectd_sequence_semantics(
            str(matching_packages[0].get("application_type") or "").strip() if matching_packages else "",
            sequence_semantic_rows,
            contract=sequence_semantic_contract,
        )
        for package in matching_packages:
            envelope = {
                "application-type": package.get("application_type"),
                "product-type": package.get("product_type"),
                "regulatory-activity-type": package.get("regulatory_activity_type"),
                "sequence-type": package.get("sequence_type"),
            }
            if any(str(value or "").strip() for value in envelope.values()):
                vocabulary_checks.append(
                    {
                        "sequence_package_id": str(package.get("sequence_package_id") or "").strip(),
                        "sequence_number": str(package.get("sequence_number") or "").strip(),
                        "relative_path": str(package.get("sequence_root") or "").strip(),
                        "validation": validate_ectd_envelope_vocabulary(
                            envelope,
                            contract=vocabulary_contract,
                        ),
                    }
                )
        assessment = assess_application_identity(
            root_name,
            sequence_packages=matching_packages,
            content_signals=content_signals,
        )
        applications.append(
            {
                **assessment,
                "relative_path": str(application_payload.get("relative_path") or root_name).strip(),
                "sequence_count": len(list(application_payload.get("sequences", []) or [])),
                "controlled_vocabulary_checks": vocabulary_checks,
                "sequence_semantic_validation": sequence_semantic_validation,
            }
        )
        for finding in sequence_semantic_validation.get("findings", []) or []:
            issue_code = str(finding.get("issue_code") or "").strip()
            if issue_code.startswith("sequence_history"):
                semantic_rule_id = "HR-ECTD-001"
            elif issue_code == "sequence_description_too_long":
                semantic_rule_id = "SR-ECTD-002"
            elif issue_code in {"sequence_contact_incomplete", "sequence_contact_email_invalid"}:
                semantic_rule_id = "HR-ECTD-014"
            elif issue_code == "incompatible_type_triplet":
                semantic_rule_id = "HR-ECTD-120"
            elif str(sequence_semantic_validation.get("scenario") or "") == "table2_new_drug_application":
                semantic_rule_id = "HR-ECTD-119"
            elif issue_code.startswith("sequence_description_prohibited_use"):
                semantic_rule_id = "SR-ECTD-055"
            else:
                semantic_rule_id = "HR-ECTD-118"
            semantic_findings.append(
                {
                    "rule_id": semantic_rule_id,
                    "status": "fail",
                    "severity": "error",
                    "scope": "sequence",
                    "relative_path": str(application_payload.get("relative_path") or root_name).strip(),
                    "message": "Clinical-trial sequence semantic evidence conflicts with the Table 1 related-sequence example contract.",
                    "blocking": True,
                    "details": {
                        "issue_family": "clinical_trial_sequence_semantics",
                        "issue_code": issue_code,
                        "sequence_semantic_finding": finding,
                    },
                }
            )
        for review_item in sequence_semantic_validation.get("review_items", []) or []:
            review_issue_code = str(review_item.get("issue_code") or "").strip()
            semantic_findings.append(
                {
                    "rule_id": "SR-ECTD-055" if review_issue_code.startswith("sequence_description_prohibited_use") else ("HR-ECTD-119" if str(sequence_semantic_validation.get("scenario") or "") == "table2_new_drug_application" else "HR-ECTD-118"),
                    "status": "manual_review",
                    "severity": "warning",
                    "scope": "sequence",
                    "relative_path": str(application_payload.get("relative_path") or root_name).strip(),
                    "message": "Clinical-trial sequence description intent needs manual confirmation against the Table 1 scenario.",
                    "blocking": False,
                    "details": {
                        "issue_family": "clinical_trial_sequence_semantics",
                        "review_item": review_item,
                    },
                }
            )

    counts = {
        "supported_count": sum(item.get("status") == "supported" for item in applications),
        "conflict_count": sum(item.get("status") == "conflict" for item in applications),
        "insufficient_evidence_count": sum(
            item.get("status") == "insufficient_evidence" for item in applications
        ),
        "review_required_count": sum(bool(item.get("review_required")) for item in applications),
        "controlled_vocabulary_failure_count": sum(
            1
            for item in applications
            for check in item.get("controlled_vocabulary_checks", []) or []
            if str((check.get("validation") or {}).get("status") or "") == "fail"
        ),
    }
    return {
        "enabled": bool(applications),
        "rule_id": "HR-ECTD-002",
        "basis": {
            "requirement_id": "cn_ectd_technical_specification:req_application_number_format",
            "source_clause_id": "cn_ectd_technical_specification:sec_2_1_1",
            "evidence_policy": "explicit envelope/application-form evidence is strong; title/keyword signals are non-authoritative",
        },
        "controlled_vocabulary_contract": vocabulary_contract,
        "sequence_semantic_contract": sequence_semantic_contract,
        "applications": applications,
        "findings": semantic_findings,
        "summary": counts,
    }


def _build_workbench(
    parsed_documents: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
    markdown_download_url: str | None,
    ind_review_markdown_download_url: str | None = None,
    structure_audit_download_url: str | None = None,
    structure_audit_markdown_download_url: str | None = None,
    demo_report_markdown_download_url: str | None = None,
    demo_script_markdown_download_url: str | None = None,
    compliance_result: dict[str, Any] | None = None,
    package_directory_paths: list[str] | None = None,
    package_source_kind: str | None = None,
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
    package_inventory = None
    package_findings: list[dict[str, Any]] = []
    application_identity: dict[str, Any] = {
        "enabled": False,
        "rule_id": "HR-ECTD-002",
        "applications": [],
        "summary": {
            "supported_count": 0,
            "conflict_count": 0,
            "insufficient_evidence_count": 0,
            "review_required_count": 0,
        },
    }
    try:
        if len(file_records) > 1 or any("/" in str(item.get("relative_path") or "") for item in file_records):
            package_inventory = build_package_inventory(
                PROJECT_ROOT,
                file_records=file_records,
                explicit_directory_paths=package_directory_paths,
            )
            if package_source_kind:
                package_inventory["source_kind"] = package_source_kind
            package_findings = validate_package_structure(package_inventory) + validate_package_naming(package_inventory)
            application_identity = _build_application_identity_projection(
                package_inventory,
                dict(compliance_payload.get("submission_scope", {}) or {}),
                parsed_documents,
            )
            package_findings.extend(application_identity.get("findings", []) or [])
            for assessment in application_identity.get("applications", []) or []:
                if not assessment.get("review_required"):
                    continue
                status = "fail" if assessment.get("status") == "conflict" else "human_review"
                package_findings.append(
                    {
                        "rule_id": "HR-ECTD-002",
                        "status": status,
                        "severity": "error" if status == "fail" else "warning",
                        "scope": "application",
                        "relative_path": str(assessment.get("relative_path") or "").strip(),
                        "message": (
                            "Application-root prefix conflicts with application-type evidence."
                            if status == "fail"
                            else "Application-root prefix lacks sufficient application-type evidence and requires manual review."
                        ),
                        "blocking": status == "fail",
                        "details": {
                            "issue_family": "application_type_semantics",
                            "issue_codes": list(assessment.get("issue_codes", []) or []),
                            "application_identity": assessment,
                        },
                    }
                )
    except (OSError, ValueError) as exc:
        package_findings = [{
            "rule_id": "HR-ECTD-015",
            "status": "human_review",
            "severity": "warning",
            "scope": "application",
            "relative_path": "",
            "message": f"Package inventory could not be built: {exc}",
            "blocking": False,
        }]
    for finding in package_findings:
        if not isinstance(finding, dict):
            continue
        details = dict(finding.get("details", {}) or {})
        details.setdefault("regulatory_provenance", provenance_for_rule(str(finding.get("rule_id") or "")))
        finding["details"] = details
    submission_scope = dict(compliance_payload.get("submission_scope", {}) or {})
    review_scope = str(submission_scope.get("scope") or "") or (
        "application" if submission_scope.get("upload_mode") == "ectd_application_project" else
        "sequence" if submission_scope.get("upload_mode") in {"ectd_sequence_package", "ectd_sequence_batch", "ectd_sequence_candidate"} else
        "document"
    )
    documents = [
        {
            "file_id": document.get("file_id"),
            "filename": document.get("filename"),
            "relative_path": next((item.get("relative_path") for item in file_records if item.get("file_id") == document.get("file_id")), document.get("filename")),
            "source_type": document.get("source_type"),
            "status": "parsed",
            "page_count": len(document.get("pages", []) or []),
            "character_count": len(str(document.get("text") or "")),
            "table_count": len(document.get("table_asts", []) or []),
            "image_count": len(document.get("image_blocks", []) or []),
            "parse_available": True,
            "text_preview": _estimate_first_page_text(document),
            "file_url": f"/api/v1/files/{document.get('file_id')}",
        }
        for document in parsed_documents
    ]
    workbench_files = [
        {
            "file_id": item.get("file_id"),
            "filename": item.get("filename"),
            "relative_path": item.get("relative_path") or item.get("filename"),
            "status": item.get("status"),
            "message": item.get("message"),
            "progress": item.get("progress", 0),
            "document_parse": item.get("document_parse", True),
            "file_url": f"/api/v1/files/{item.get('file_id')}",
        }
        for item in file_records
    ]
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
        "documents": documents,
        "files": workbench_files,
        "selected_file_id": pdf_document.get("file_id") if pdf_document else (documents[0].get("file_id") if documents else None),
        "review_scope": review_scope,
        "package_inventory": package_inventory,
        "package_findings": package_findings,
        "application_identity": application_identity,
        "markdown": _build_ui_markdown(parsed_documents, file_records, consistency_rows),
        "full_markdown_download_url": markdown_download_url,
        "ind_review_markdown_download_url": ind_review_markdown_download_url,
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
        "markdown_rendering_audit_summary": _build_markdown_rendering_audit_summary(parsed_documents),
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


def is_package_support_file(filename: str) -> bool:
    """Return whether an eCTD utility/backbone file needs package validation, not document parsing."""
    suffix = Path(str(filename or "")).suffix.lower()
    name = Path(str(filename or "")).name.lower()
    return suffix in {".dtd", ".xsl", ".xsd", ".txt"} or name in {"index-md5.txt"}


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
            if file_record.get("document_parse") is False or is_package_support_file(
                file_record.get("relative_path") or file_record.get("filename")
            ):
                file_status = "completed"
                file_message = "Package support file received; content parsing skipped"
                file_progress = 100
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
                continue
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
    ind_review_markdown_text = _build_full_markdown(parsed_documents, markdown_profile="ind-review")
    ind_review_markdown_path = IND_REVIEW_MARKDOWN_DIR / f"{job_id}_ind_review.md"
    ind_review_markdown_path.write_text(ind_review_markdown_text, encoding="utf-8")
    ind_review_markdown_download_url = f"/api/v1/jobs/{job_id}/ind-review/markdown/download"

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
            ind_review_markdown_download_url=ind_review_markdown_download_url,
            structure_audit_download_url=structure_audit_download_url,
            structure_audit_markdown_download_url=structure_audit_markdown_download_url,
            demo_report_markdown_download_url=demo_report_markdown_download_url,
            demo_script_markdown_download_url=demo_script_markdown_download_url,
            compliance_result=compliance_result,
            package_directory_paths=list(job.get("package_directory_paths") or []),
            package_source_kind=str(job.get("package_source_kind") or "") or None,
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
        job["ind_review_markdown_path"] = str(ind_review_markdown_path)
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
        directory_paths: list[str] = Form(default=[]),
    ) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        job_id = uuid4().hex
        file_records: list[dict[str, Any]] = []
        package_directory_paths: set[str] = set(directory_paths)
        package_upload_detected = bool(package_directory_paths)
        package_inventory_payload: dict[str, Any] | None = None
        created_at = _utc_now()
        logger.info("Upload request received. job_id=%s, files=%s", job_id, len(files))

        for index, incoming_file in enumerate(files):
            filename = incoming_file.filename or "uploaded_file"
            relative_path = str(relative_paths[index] if index < len(relative_paths) else filename).strip() or filename
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS and not is_directory_upload_path(relative_path, filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
                )

            file_id = uuid4().hex
            payload = await incoming_file.read()
            if suffix == ".zip":
                package_upload_detected = True
                extracted_root = UPLOAD_DIR / f"{file_id}_package"
                try:
                    extracted_records, package_inventory = records_from_zip_bytes(payload, output_root=extracted_root)
                    package_directory_paths.update(package_inventory.get("directory_paths") or [])
                    package_inventory_payload = package_inventory
                except (OSError, ValueError, zipfile.BadZipFile) as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid eCTD ZIP package: {exc}") from exc
                for extracted in extracted_records:
                    extracted_file_id = uuid4().hex
                    extracted_path = Path(extracted["path"])
                    with STORE_LOCK:
                        FILE_STORE[extracted_file_id] = extracted_path
                    file_records.append({
                        "file_id": extracted_file_id,
                        "filename": extracted_path.name,
                        "relative_path": extracted["relative_path"],
                        "suffix": extracted_path.suffix.lower(),
                        "path": str(extracted_path),
                        "status": "queued",
                        "message": "Queued from ZIP package",
                        "progress": 0,
                        "document_parse": extracted_path.suffix.lower() in ALLOWED_EXTENSIONS,
                    })
                continue
            target_path = UPLOAD_DIR / f"{file_id}_{_safe_filename(filename)}"
            target_path.write_bytes(payload)

            if is_directory_upload_path(relative_path, filename):
                package_upload_detected = True

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
                    "document_parse": suffix in ALLOWED_EXTENSIONS,
                }
            )

        if package_upload_detected and file_records and (
            package_inventory_payload is None or package_inventory_payload.get("source_kind") != "zip"
        ):
            package_inventory_payload = build_package_inventory(
                PROJECT_ROOT,
                file_records=file_records,
                explicit_directory_paths=sorted(package_directory_paths),
            )
            if package_inventory_payload.get("source_kind") != "zip" and package_inventory_payload is not None:
                package_inventory_payload["source_kind"] = "folder" if any(
                    is_directory_upload_path(item.get("relative_path"), item.get("filename"))
                    for item in file_records
                ) else "batch"

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
            "package_directory_paths": sorted(package_directory_paths),
            "package_inventory": package_inventory_payload,
            "package_source_kind": (
                str(package_inventory_payload.get("source_kind") or "")
                if package_inventory_payload
                else None
            ),
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
            "package_inventory": package_inventory_payload,
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
                "package_inventory": job.get("package_inventory"),
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

    @app.get("/api/v1/jobs/{job_id}/ind-review/markdown/download")
    def download_ind_review_markdown(job_id: str) -> FileResponse:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            markdown_path_str = str(job.get("ind_review_markdown_path", "")).strip()
        if not markdown_path_str:
            raise HTTPException(status_code=404, detail="IND review markdown file not found")
        markdown_path = Path(markdown_path_str)
        if not markdown_path.exists():
            raise HTTPException(status_code=404, detail="IND review markdown file not found")
        logger.info("IND review markdown download requested for job %s", job_id)
        return FileResponse(
            markdown_path,
            media_type="text/markdown; charset=utf-8",
            filename=f"{job_id}_ind_review.md",
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
