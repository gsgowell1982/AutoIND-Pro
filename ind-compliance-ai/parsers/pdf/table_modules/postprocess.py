# Version: v1.1.5
# Optimization Summary:
# - Prevent inherited headers from rewriting continuation-page data rows unless
#   the current page provides explicit local repeated-header evidence.
# - Keep header inheritance as metadata support rather than a source of truth
#   that can overwrite observed row values.
# - Apply the same contradiction-first local semantic policy to same-page fragment
#   merging and same-title merging.
# - Reject same-page merges when local titles conflict, even if geometry is
#   otherwise similar.
# - Reuse local_context_signal barriers in same-title merge decisions so repeated
#   captions do not collapse independent tables.
# - Guard same-page fragment merging with the preceding-text barrier detected via continuation hints so captions break accidental merges.
# - Rebuild header grid rows when header tokens spill across cells so row_texts keep the correct column labels.
# - Strengthen cross-page stitching with title compatibility and horizontal overlap checks.
# - Allow stitching when current page repeats the same title as previous continuation table.
# - Add stricter gating for low-similarity continuation candidates to reduce false links.
# - Externalize stitching thresholds to config/pdf_parser.toml for enterprise tuning.
# - Add preceding-text guard and continuation metadata reset to avoid false cross-page links.
# - Rebuild headers/column counts from the grid when guard prevents continuation, ensuring correct logical columns.
# - Preserve early-confirmed continuation links unless local contradiction evidence
#   explicitly indicates a new table or narrative barrier.
# - Reuse persisted local context signals so running headers do not break real
#   continuation chains.
# - Keep raw logical rows for audit while rebuilding semantic row views without
#   fully empty rows for business consumers.
# - Repair strong cross-page boundary row splits in the semantic view while
#   preserving raw audit rows unchanged.
#
# v1.1.0 (2026-03-12):
# - Utilize structure_validation information from Step3 for more robust stitching decisions.
# - Check validation_passed, empty_columns, and column_mapping_confidence before stitching.
# - Reduce false positives caused by local misjudgments.

"""Post-processing Module - 表格后处理

Architecture:
    Table ASTs → Post-processing → Final Table ASTs

负责表格的后处理：
- 跨页表格拼接
- 同页片段合并
- 表格 ID 重编号
- 单元格去重与排序
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ..settings import get_pdf_parser_settings
from .cell_text_projection import (
    project_pdf_math_symbol_display_text,
    project_table_grid_display_text,
)

_CONTINUATION_HINT_KEYWORDS = ["continued", "continued from", "续表", "续页"]
_CONTINUATION_HINT_KEYWORDS = ["continued", "continued from", "\u7eed\u8868", "\u7eed\u9875", "\uff08\u7eed", "(\u7eed", "\u7eed\uff09"]
_LOCAL_BARRIER_SIGNALS = {"new_table_title", "narrative_barrier", "section_heading"}
_DATE_PREFIX_RE = re.compile(
    r"^(?P<date>\d{1,2}[/-][A-Za-z]{3,9}[/-]\d{2,4})\s+(?P<rest>.+)$", re.IGNORECASE
)
_STRONG_TERMINAL_PUNCTUATION = {".", "!", "?", ";", "。", "！", "？", "；"}
_TRAILING_CLOSER_CHARS = "\"'”’)]}】》」』）"
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD_CHAR_RE = re.compile(r"[A-Za-z0-9]")
_VECTOR_OCR_SENTENCE_BOUNDARY_CHARS = {".", "!", "?", ";", "\u3002", "\uff01", "\uff1f", "\uff1b"}
_VECTOR_OCR_OPENING_BRACKETS = {"(", "[", "{", "\uff08", "\u3010", "\u300a", "\u3008", "\u300c"}
_VECTOR_OCR_CLOSING_BRACKETS = {")", "]", "}", "\uff09", "\u3011", "\u300b", "\u3009", "\u300d"}
_GENERIC_COLUMN_HEADER_RE = re.compile(r"^column\s+\d+$", re.IGNORECASE)
_HEADER_CONTINUATION_TOKENS = {
    "model",
    "models",
    "content",
    "destination",
    "date",
    "dates",
    "type",
    "types",
    "name",
    "names",
    "id",
    "ids",
    "description",
    "descriptions",
    "method",
    "methods",
    "language",
    "languages",
    "result",
    "results",
    "score",
    "scores",
    "value",
    "values",
    "group",
    "groups",
    "category",
    "categories",
    "item",
    "items",
    "unit",
    "units",
    "path",
    "paths",
    "file",
    "files",
    "folder",
    "folders",
    "\u7f16\u53f7",
    "\u540d\u79f0",
    "\u7c7b\u578b",
    "\u5185\u5bb9",
    "\u8bf4\u660e",
    "\u63cf\u8ff0",
    "\u7ed3\u679c",
    "\u5355\u4f4d",
    "\u65e5\u671f",
    "\u8def\u5f84",
    "\u6587\u4ef6",
    "\u76ee\u7684\u5730",
}

# ============================================================================
# Utility Functions
# ============================================================================

def column_similarity(signature_a: list[float], signature_b: list[float]) -> float:
    """计算两个列签名的相似度"""
    if not signature_a or not signature_b:
        return 0.0

    tolerance = 0.07
    matched = 0

    for value in signature_a:
        if any(abs(value - other) <= tolerance for other in signature_b):
            matched += 1

    return matched / max(len(signature_a), len(signature_b))


def header_similarity(header_a: list[dict[str, Any]], header_b: list[dict[str, Any]]) -> float:
    """计算两个表头的相似度"""
    texts_a = [str(item.get("text", "")).strip().lower() for item in header_a if str(item.get("text", "")).strip()]
    texts_b = [str(item.get("text", "")).strip().lower() for item in header_b if str(item.get("text", "")).strip()]
    if not texts_a or not texts_b:
        return 0.0
    overlap = len(set(texts_a) & set(texts_b))
    return overlap / max(len(set(texts_a)), len(set(texts_b)))


def _headers_show_new_table_boundary(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Detect strong local header contradiction before cross-page stitching."""
    if _get_local_context_signal(current) == "continuation_title":
        return False
    prev_header = [
        str(cell.get("text", "") or "").strip()
        for cell in previous.get("header", []) or []
        if str(cell.get("text", "") or "").strip()
    ]
    curr_header = [
        str(cell.get("text", "") or "").strip()
        for cell in current.get("header", []) or []
        if str(cell.get("text", "") or "").strip()
    ]
    if len(prev_header) < 2 or len(curr_header) < 2:
        return False
    if current.get("header_inherited"):
        return False
    generic_current = sum(1 for text in curr_header if _GENERIC_COLUMN_HEADER_RE.match(text))
    if generic_current >= max(2, len(curr_header) // 2):
        return False

    previous_col_count = int(previous.get("col_count", 0) or len(prev_header))
    current_col_count = int(current.get("col_count", 0) or len(curr_header))
    if abs(previous_col_count - current_col_count) < 2:
        return False
    similarity = header_similarity(previous.get("header", []) or [], current.get("header", []) or [])
    if similarity >= 0.25:
        return False
    prev_tokens = {_compact_text(text) for text in prev_header if _compact_text(text)}
    curr_tokens = {_compact_text(text) for text in curr_header if _compact_text(text)}
    return not bool(prev_tokens & curr_tokens)


# ============================================================================
# Cross-Page Table Stitching
# ============================================================================

def stitch_cross_page_tables(
    table_asts: list[dict[str, Any]],
    page_heights: dict[int, float],
) -> int:
    """??????

    This function:
    1. Detects continuation tables across pages
    2. Sets continued_from/continued_to links
    3. Inherits title and header from parent table
    4. Records inheritance metadata for audit

    Args:
        table_asts: List of table ASTs from all pages
        page_heights: Dict mapping page number to height

    Returns:
        Number of stitched links created
    """
    if not table_asts:
        return 0

    cfg = get_pdf_parser_settings().cross_page_stitching
    stitched_count = 0
    for table in table_asts:
        _apply_date_prefix_split(table)

    sorted_tables = sorted(table_asts, key=lambda item: (item["page"], item["bbox"][1]))

    for previous, current in zip(sorted_tables, sorted_tables[1:]):
        if current["page"] != previous["page"] + 1:
            continue

        previous_bbox = tuple(previous.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        current_bbox = tuple(current.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        overlap_ratio = _horizontal_overlap_ratio(previous_bbox, current_bbox)

        prev_page_height = max(1.0, page_heights.get(previous["page"], 1.0))
        curr_page_height = max(1.0, page_heights.get(current["page"], 1.0))
        prev_near_bottom = previous_bbox[3] >= prev_page_height * cfg.prev_near_bottom_ratio
        curr_near_top = current_bbox[1] <= curr_page_height * cfg.curr_near_top_ratio

        similarity = column_similarity(
            previous.get("column_signature", []),
            current.get("column_signature", [])
        )

        previous_has_title = bool(str(previous.get("title", "")).strip())
        previous_section_hint_text = str(previous.get("section_hint", ""))
        previous_is_glossary_context = _is_glossary_hint_text(previous_section_hint_text) or (
            str(previous.get("continuation_context", "")).strip().lower() == "glossary"
        )
        previous_has_section_hint = bool(previous_section_hint_text.strip()) or previous_is_glossary_context

        previous_col_count = int(previous.get("col_count", 0))
        current_col_count = int(current.get("col_count", 0))

        continuation_hint = current.get("continuation_hint")
        hint_matches_previous = (
            isinstance(continuation_hint, dict)
            and str(continuation_hint.get("table_id", "")).strip() == str(previous.get("table_id", "")).strip()
        )

        previous_title = str(previous.get("title", "")).strip()
        current_title = str(current.get("title", "")).strip()
        if not _titles_compatible(previous_title, current_title):
            continue

        early_confirmed = _is_early_confirmed_continuation(previous, current)
        if early_confirmed:
            if _has_preceding_text_block_for_continuation(current):
                _reset_continuation_flags(current)
                continue
            if not hint_matches_previous and _headers_show_new_table_boundary(previous, current):
                _reset_continuation_flags(current)
                continue
            _materialize_continuation_link(
                previous,
                current,
                similarity=similarity,
                overlap_ratio=overlap_ratio,
                strategy="assessment_confirmed",
            )
            stitched_count += 1
            continue

        if overlap_ratio < cfg.overlap_min_without_hint and not hint_matches_previous:
            continue

        # Context-based continuation detection
        context_continuation = (
            (previous_has_title or previous_has_section_hint)
            and curr_near_top
            and abs(previous_col_count - current_col_count) <= 1
            and overlap_ratio >= cfg.context_overlap_min
            and (
                (previous_is_glossary_context and similarity >= cfg.hint_glossary_similarity_min)
                or (not previous_is_glossary_context and similarity >= cfg.default_similarity_min)
            )
        )

        is_likely_continuation = (
            (prev_near_bottom and curr_near_top)
            or similarity >= cfg.high_similarity_link_threshold
            or ((previous_has_title or previous_has_section_hint) and curr_near_top and similarity >= cfg.title_context_similarity_threshold)
            or context_continuation
            or hint_matches_previous
        )

        min_similarity = cfg.hint_glossary_similarity_min if hint_matches_previous and previous_is_glossary_context else cfg.default_similarity_min
        if similarity < min_similarity or not is_likely_continuation:
            continue
        if (
            similarity < cfg.low_similarity_guard_threshold
            and overlap_ratio < cfg.low_similarity_guard_overlap_min
            and not hint_matches_previous
        ):
            continue

        # v1.1.0: 利用 Step3 的结构校验信息
        # 如果结构校验未通过，需要更强的其他证据
        current_validation = current.get("structure_validation")
        if current_validation:
            if not current_validation.get("validation_passed", True):
                # 结构校验未通过，检查是否有严重问题
                empty_cols = current_validation.get("empty_columns", [])
                if len(empty_cols) > 0:
                    # 有空列，需要更强的其他证据
                    if similarity < cfg.high_similarity_link_threshold and not hint_matches_previous:
                        continue
            
            # 利用列映射置信度
            mapping_confidence = current_validation.get("column_mapping_confidence", 0.5)
            if mapping_confidence < 0.5:
                # 列映射置信度低，需要更强的其他证据
                if similarity < cfg.high_similarity_link_threshold and not hint_matches_previous:
                    continue

        if _has_preceding_text_block_for_continuation(current):
            _reset_continuation_flags(current)
            continue

        if not hint_matches_previous and _headers_show_new_table_boundary(previous, current):
            _reset_continuation_flags(current)
            continue

        if current_title and _compact_text(current_title) != _compact_text(previous_title):
            continue

        _materialize_continuation_link(
            previous,
            current,
            similarity=similarity,
            overlap_ratio=overlap_ratio,
            strategy="cross_page_stitch",
        )
        stitched_count += 1

    return stitched_count


def _is_glossary_hint_text(text: str) -> bool:
    """检查文本是否指示术语表上下文"""
    normalized = text.strip().lower()
    return "术语表" in normalized or "词汇表" in normalized or "glossary" in normalized


# ============================================================================
# Same-Page Fragment Merging
# ============================================================================

def _titles_compatible(previous_title: str, current_title: str) -> bool:
    """Check whether two titles can belong to the same continuation chain."""
    if not previous_title or not current_title:
        return True
    return _compact_text(previous_title) == _compact_text(current_title)


def _is_early_confirmed_continuation(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Return True when earlier pipeline stages already resolved this pair."""
    previous_id = str(previous.get("table_id", "")).strip()
    current_from = str(current.get("continued_from", "")).strip()
    if current_from and current_from == previous_id:
        return True

    assessment = current.get("continuation_assessment")
    if not isinstance(assessment, dict):
        return False

    if not assessment.get("is_continuation"):
        return False
    return str(assessment.get("selected_parent_id", "")).strip() == previous_id


def _materialize_continuation_link(
    previous: dict[str, Any],
    current: dict[str, Any],
    similarity: float,
    overlap_ratio: float,
    strategy: str,
) -> None:
    """Persist a validated continuation relation in a single place."""
    previous.setdefault("continued_to", [])
    if current["table_id"] not in previous["continued_to"]:
        previous["continued_to"].append(current["table_id"])

    current["continued_from"] = previous["table_id"]
    current["cross_page_similarity"] = round(similarity, 3)
    current["cross_page_overlap"] = round(overlap_ratio, 3)
    current["is_continuation"] = True

    continuation_source = current.get("continuation_source")
    if not isinstance(continuation_source, dict):
        continuation_source = {
            "source_table_id": str(previous.get("table_id", "")),
            "strategy": strategy,
            "similarity": round(similarity, 3),
            "horizontal_overlap": round(overlap_ratio, 3),
            "inherited_fields": [],
        }
    else:
        continuation_source["source_table_id"] = str(previous.get("table_id", ""))
        continuation_source["strategy"] = strategy
        continuation_source["similarity"] = round(similarity, 3)
        continuation_source["horizontal_overlap"] = round(overlap_ratio, 3)
        continuation_source.setdefault("inherited_fields", [])

    previous_title = str(previous.get("title", "")).strip()
    current_title = str(current.get("title", "")).strip()
    if not current_title and previous_title:
        current["title"] = previous["title"]
        current["title_inherited"] = True
        if "title" not in continuation_source["inherited_fields"]:
            continuation_source["inherited_fields"].append("title")

    hdr_similarity = header_similarity(previous.get("header", []), current.get("header", []))
    if hdr_similarity < 0.45:
        current["header"] = previous.get("header", [])
        current["header_inherited"] = True
        if "header" not in continuation_source["inherited_fields"]:
            continuation_source["inherited_fields"].append("header")

    current["continuation_source"] = continuation_source

def can_merge_table_fragments_on_same_page(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    """检查同页上的两个表格片段是否可以合并"""
    if int(primary.get("page", 0)) != int(secondary.get("page", 0)):
        return False

    if _can_merge_dense_overlapping_grid_fragments(primary, secondary):
        return True

    cfg = get_pdf_parser_settings().same_page_merge_policy
    if _has_title_conflict(primary, secondary):
        return False
    if cfg.respect_preceding_text_barrier and _has_preceding_text_barrier(primary, secondary):
        return False
    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    vertical_gap = secondary_bbox[1] - primary_bbox[3]

    row_height = _estimate_row_height(primary) or _estimate_row_height(secondary)
    fallback_height = max(primary_bbox[3] - primary_bbox[1], 1.0)
    safe_row_height = row_height if row_height > 0 else fallback_height
    gap_ratio = vertical_gap / safe_row_height
    if gap_ratio < cfg.gap_ratio_min or gap_ratio > cfg.gap_ratio_max:
        return False

    overlap = _horizontal_overlap_ratio(primary_bbox, secondary_bbox)
    if overlap < 0.5:
        return False

    similarity = column_similarity(
        list(primary.get("column_signature", [])),
        list(secondary.get("column_signature", [])),
    )
    if similarity < 0.68:
        return False

    header_similarity_score = header_similarity(
        primary.get("header", []) or [],
        secondary.get("header", []) or [],
    )
    if cfg.header_similarity_threshold > 0.0 and header_similarity_score < cfg.header_similarity_threshold:
        return False

    if float(primary.get("toc_row_ratio", 0.0)) > 0.25 or float(secondary.get("toc_row_ratio", 0.0)) > 0.25:
        return False

    return True


def can_merge_horizontal_table_fragments_on_same_page(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    """Return True when two same-page fragments are left/right pieces of one table."""
    if int(primary.get("page", 0)) != int(secondary.get("page", 0)):
        return False
    if _has_title_conflict(primary, secondary):
        return False

    left, right = _order_horizontal_fragments(primary, secondary)
    left_bbox = tuple(left.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    right_bbox = tuple(right.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(left_bbox) != 4 or len(right_bbox) != 4:
        return False
    horizontal_gap = float(right_bbox[0]) - float(left_bbox[2])
    if horizontal_gap < -10.0 or horizontal_gap > 90.0:
        return False
    if _vertical_overlap_ratio(left_bbox, right_bbox) < 0.62:
        return False

    left_grid = _get_authoritative_raw_grid(left)
    right_grid = _get_authoritative_raw_grid(right)
    if len(left_grid) < 3 or len(right_grid) < 3:
        return False
    if abs(len(left_grid) - len(right_grid)) > 1:
        return False

    left_title = _compact_text(str(left.get("title", "")))
    right_title = _compact_text(str(right.get("title", "")))
    has_title_support = bool(left_title or right_title) and (not left_title or not right_title or left_title == right_title)
    row_profile_support = _horizontal_fragment_row_profiles_match(left_grid, right_grid)
    header_support = _horizontal_fragment_headers_are_complementary(left, right)
    numeric_support = _horizontal_fragment_numeric_density(right_grid) >= 0.55
    return (has_title_support and row_profile_support and (header_support or numeric_support)) or (
        row_profile_support and header_support and numeric_support
    )


def merge_horizontal_table_fragments(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """Merge left/right pieces of the same physical table by row alignment."""
    left, right = _order_horizontal_fragments(primary, secondary)
    left_grid = _get_authoritative_raw_grid(left)
    right_grid = _get_authoritative_raw_grid(right)
    merged_grid = _merge_horizontal_raw_grids(left_grid, right_grid)

    merged = dict(left)
    left_bbox = tuple(left.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    right_bbox = tuple(right.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    merged["bbox"] = [
        min(left_bbox[0], right_bbox[0]),
        min(left_bbox[1], right_bbox[1]),
        max(left_bbox[2], right_bbox[2]),
        max(left_bbox[3], right_bbox[3]),
    ]
    if not merged.get("title") and right.get("title"):
        merged["title"] = right.get("title")
    merged["col_count"] = max((len(row) for row in merged_grid), default=0)
    merged["logical_col_count"] = merged["col_count"]
    merged["physical_col_count"] = merged["col_count"]
    merged["raw_grid"] = merged_grid
    display_grid = project_table_grid_display_text(_clone_grid_rows(merged_grid))
    merged["display_grid"] = display_grid
    merged["grid"] = _clone_grid_rows(display_grid)
    merged["data_grid"] = _clone_grid_rows(display_grid)
    merged["raw_row_texts"] = _render_row_texts(merged_grid)
    merged["display_row_texts"] = _render_row_texts(display_grid)
    merged["row_texts"] = _render_row_texts(display_grid)
    merged["data_row_texts"] = list(merged["row_texts"])
    merged["raw_row_count"] = len(merged_grid)
    merged["display_row_count"] = len(display_grid)
    merged["row_count"] = len(display_grid)
    merged["data_row_count"] = len(display_grid)
    merged["logical_row_count"] = len(display_grid)
    merged["data_start_row"] = 1
    merged["column_signature"] = _build_column_signature(merged["col_count"])
    merged["column_hash"] = _compute_column_hash(merged["column_signature"])
    merged["header"] = _merge_horizontal_headers(left, right, merged_grid)
    merged["cells"] = _build_cells_from_grid(merged["grid"])
    merged.setdefault("merged_from", [])
    secondary_id = str(right.get("table_id") or secondary.get("table_id") or "")
    if secondary_id:
        merged["merged_from"].append(secondary_id)
    merged["horizontal_fragment_merge"] = True
    return merged


def _order_horizontal_fragments(primary: dict[str, Any], secondary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if float(primary_bbox[0]) <= float(secondary_bbox[0]):
        return primary, secondary
    return secondary, primary


def _vertical_overlap_ratio(bbox_a: tuple, bbox_b: tuple) -> float:
    overlap = max(0.0, min(float(bbox_a[3]), float(bbox_b[3])) - max(float(bbox_a[1]), float(bbox_b[1])))
    height_a = max(1.0, float(bbox_a[3]) - float(bbox_a[1]))
    height_b = max(1.0, float(bbox_b[3]) - float(bbox_b[1]))
    return overlap / min(height_a, height_b)


def _horizontal_fragment_row_profiles_match(left_grid: list[list[str | None]], right_grid: list[list[str | None]]) -> bool:
    compare_count = min(len(left_grid), len(right_grid))
    if compare_count < 3:
        return False
    compatible = 0
    for left_row, right_row in zip(left_grid[:compare_count], right_grid[:compare_count]):
        left_non_empty = len(_semantic_non_empty_columns(left_row))
        right_non_empty = len(_semantic_non_empty_columns(right_row))
        if left_non_empty <= 0 or right_non_empty <= 0:
            continue
        if left_non_empty <= 2 and right_non_empty >= 2:
            compatible += 1
            continue
        if right_non_empty <= 2 and left_non_empty >= 2:
            compatible += 1
            continue
        if abs(left_non_empty - right_non_empty) <= max(1, min(left_non_empty, right_non_empty) // 2):
            compatible += 1
    return compatible >= max(3, int(compare_count * 0.72))


def _horizontal_fragment_headers_are_complementary(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    left_header = [
        str(cell.get("text", "") or "").strip()
        for cell in primary.get("header", []) or []
        if str(cell.get("text", "") or "").strip()
    ]
    right_header = [
        str(cell.get("text", "") or "").strip()
        for cell in secondary.get("header", []) or []
        if str(cell.get("text", "") or "").strip()
    ]
    if len(left_header) < 2 or len(right_header) < 2:
        return False
    overlap = set(_compact_text(item) for item in left_header) & set(_compact_text(item) for item in right_header)
    if len(overlap) >= max(2, min(len(left_header), len(right_header)) // 2):
        return False
    return True


def _horizontal_fragment_numeric_density(grid: list[list[str | None]]) -> float:
    values = [_semantic_cell_text(cell) for row in grid for cell in row if _semantic_cell_text(cell)]
    if not values:
        return 0.0
    numeric_like = sum(1 for value in values if _cell_value_profile_kind(value) in {"numeric", "statistical"})
    return numeric_like / len(values)


def _merge_horizontal_raw_grids(left_grid: list[list[str | None]], right_grid: list[list[str | None]]) -> list[list[str | None]]:
    row_count = max(len(left_grid), len(right_grid))
    left_width = max((len(row) for row in left_grid), default=0)
    right_width = max((len(row) for row in right_grid), default=0)
    merged: list[list[str | None]] = []
    for row_idx in range(row_count):
        left_row = list(left_grid[row_idx]) if row_idx < len(left_grid) else []
        right_row = list(right_grid[row_idx]) if row_idx < len(right_grid) else []
        left_row.extend([None] * max(0, left_width - len(left_row)))
        right_row.extend([None] * max(0, right_width - len(right_row)))
        merged.append(left_row + right_row)
    return merged


def _merge_horizontal_headers(primary: dict[str, Any], secondary: dict[str, Any], merged_grid: list[list[str | None]]) -> list[dict[str, Any]]:
    header_texts = [
        str(cell.get("text", "") or "").strip()
        for cell in primary.get("header", []) or []
        if str(cell.get("text", "") or "").strip()
    ] + [
        str(cell.get("text", "") or "").strip()
        for cell in secondary.get("header", []) or []
        if str(cell.get("text", "") or "").strip()
    ]
    col_count = max((len(row) for row in merged_grid), default=len(header_texts))
    if len(header_texts) != col_count and merged_grid:
        header_texts = [str(cell or "").strip() or f"Column {idx + 1}" for idx, cell in enumerate(merged_grid[0])]
    while len(header_texts) < col_count:
        header_texts.append(f"Column {len(header_texts) + 1}")
    return [{"col": idx + 1, "text": text or f"Column {idx + 1}"} for idx, text in enumerate(header_texts[:col_count])]


def _horizontal_overlap_ratio(bbox_a: tuple, bbox_b: tuple) -> float:
    """计算两个 bbox 的水平重叠比例"""
    x_overlap = max(0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]))
    width_a = bbox_a[2] - bbox_a[0]
    width_b = bbox_b[2] - bbox_b[0]
    if width_a <= 0 or width_b <= 0:
        return 0.0
    return x_overlap / min(width_a, width_b)


def _has_preceding_text_block_for_continuation(table: dict[str, Any]) -> bool:
    """Return True if the table has a qualifying intervening text block."""
    signal = _get_local_context_signal(table)
    if signal in _LOCAL_BARRIER_SIGNALS:
        return True
    if signal in {"continuation_title", "running_header", "none", "study_metadata_context"}:
        return False

    block = table.get("preceding_text_block")
    if not block:
        return False
    text = str(block.get("text", "")).strip().lower()
    if not text:
        return False
    for keyword in _CONTINUATION_HINT_KEYWORDS:
        if keyword in text:
            return False
    return True


def _has_preceding_text_barrier(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    """Detect textual captions between fragments that should stop same-page merging."""
    signal = _get_local_context_signal(secondary)
    if signal in _LOCAL_BARRIER_SIGNALS:
        return True
    if signal in {"continuation_title", "running_header", "none", "study_metadata_context"}:
        return False

    block = secondary.get("preceding_text_block")
    if not block:
        return False
    text = str(block.get("text", "")).strip().lower()
    if not text:
        return False
    for keyword in _CONTINUATION_HINT_KEYWORDS:
        if keyword in text:
            return False

    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    block_bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))

    if block_bbox[1] <= primary_bbox[3]:
        return False
    if block_bbox[3] >= secondary_bbox[1]:
        return False

    return True


def _has_title_conflict(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    """Treat incompatible explicit titles as a hard contradiction for fragment identity."""
    primary_title = str(primary.get("title", "")).strip()
    secondary_title = str(secondary.get("title", "")).strip()
    if not primary_title or not secondary_title:
        return False
    return not _titles_compatible(primary_title, secondary_title)


def _get_local_context_signal(table: dict[str, Any]) -> str:
    """Read the persisted local signal or fall back to a conservative legacy guess."""
    signal = str(table.get("local_context_signal", "")).strip()
    if signal:
        return signal

    block = table.get("preceding_text_block")
    if not block:
        return "none"

    text = str(block.get("text", "")).strip().lower()
    if not text:
        return "none"

    for keyword in _CONTINUATION_HINT_KEYWORDS:
        if keyword in text:
            return "continuation_title"

    return "narrative_barrier"


def _reset_continuation_flags(table: dict[str, Any]) -> None:
    """Clear continuation metadata when guard refuses stitching."""
    table["is_continuation"] = False
    table["header_inherited"] = False
    table.pop("continued_from", None)
    table.pop("continuation_source", None)
    table.pop("cross_page_similarity", None)
    table.pop("cross_page_overlap", None)
    table.pop("continued_to", None)

    if _apply_header_from_candidates(table):
        return

    header_cells = _build_header_from_table(table)
    if header_cells:
        table["header"] = header_cells
        table["col_count"] = len(header_cells)
        table["header_rebuilt_by_guard"] = True


def _translate_grid_value(value: str) -> str:
    return value.strip() if isinstance(value, str) else value


def _build_cells_from_grid(grid: list[list[str | None]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for row_idx, row in enumerate(grid, start=1):
        for col_idx, value in enumerate(row, start=1):
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            text = value
            cells.append(
                {
                    "row": row_idx,
                    "col": col_idx,
                    "logical_row": row_idx,
                    "logical_col": col_idx,
                    "physical_row": row_idx,
                    "physical_col": col_idx,
                    "text": text,
                }
            )
    return cells


def _build_header_from_grid(grid: list[list[str | None]] | None) -> list[dict[str, Any]]:
    header_cells: list[dict[str, Any]] = []
    if not grid:
        return header_cells
    first_row = project_table_grid_display_text([list(grid[0])])[0]
    for idx, value in enumerate(first_row):
        text = _translate_grid_value(value)
        if not text:
            text = f"Column {idx + 1}"
        header_cells.append({"col": idx + 1, "text": text})
    return header_cells


def _clone_grid_rows(grid: list[list[str | None]] | None) -> list[list[str | None]]:
    return [list(row) for row in (grid or [])]


def _render_row_texts(grid: list[list[str | None]]) -> list[str]:
    return [
        " | ".join("null" if cell is None else str(cell) for cell in row)
        for row in grid
    ]


def _build_column_signature(col_count: int) -> list[float]:
    if col_count <= 0:
        return []
    return [round((idx + 0.5) / col_count, 3) for idx in range(col_count)]


def _compute_column_hash(signature: list[float]) -> str:
    compact = ",".join(f"{value:.3f}" for value in signature)
    return hashlib.sha1(compact.encode()).hexdigest()[:16] if compact else ""


def _row_has_semantic_content(row: list[str | None]) -> bool:
    for cell in row:
        if cell is None:
            continue
        if isinstance(cell, str):
            if cell.strip():
                return True
            continue
        return True
    return False


def _effective_data_start_row_for_row_views(
    raw_grid: list[list[str | None]],
    *,
    data_start_row: int,
) -> int:
    if len(raw_grid) >= 3 and data_start_row == 2 and _single_schema_header_followed_by_clear_data(raw_grid[0], raw_grid[2]):
        return data_start_row
    if data_start_row != 1 or len(raw_grid) < 2:
        return data_start_row
    if _single_schema_header_followed_by_clear_data(raw_grid[0], raw_grid[1]):
        return 0
    if _rows_share_data_value_profile(raw_grid[0], raw_grid[1]):
        return 0
    return data_start_row


def _single_schema_header_followed_by_clear_data(
    header_row: list[str | None],
    data_row: list[str | None],
) -> bool:
    header_texts = [_semantic_cell_text(cell) for cell in header_row]
    data_texts = [_semantic_cell_text(cell) for cell in data_row]
    if len(header_texts) < 3 or len(header_texts) != len(data_texts):
        return False
    header_non_empty = [text for text in header_texts if text]
    data_non_empty = [text for text in data_texts if text]
    if len(header_non_empty) < 3 or len(data_non_empty) < 3:
        return False
    header_like = sum(1 for text in header_texts if _cell_value_profile_kind(text) == "header_label")
    data_value_like = sum(
        1
        for text in data_texts
        if _cell_value_profile_kind(text) in {"numeric", "statistical", "pathish", "code"}
    )
    if header_like < max(3, len(header_non_empty) - 1):
        return False
    return data_value_like >= max(2, len(data_non_empty) // 2)


def _rows_share_data_value_profile(
    left: list[str | None],
    right: list[str | None],
) -> bool:
    left_texts = [_semantic_cell_text(cell) for cell in left]
    right_texts = [_semantic_cell_text(cell) for cell in right]
    if len(left_texts) < 2 or len(left_texts) != len(right_texts):
        return False

    left_non_empty = [text for text in left_texts if text]
    right_non_empty = [text for text in right_texts if text]
    if len(left_non_empty) < 2 or len(right_non_empty) < 2:
        return False
    if _rows_look_like_header_then_data(left_texts, right_texts):
        return False

    comparable_columns = 0
    matching_profile_columns = 0
    for left_text, right_text in zip(left_texts, right_texts):
        if not left_text or not right_text:
            continue
        left_kind = _cell_value_profile_kind(left_text)
        right_kind = _cell_value_profile_kind(right_text)
        if left_kind == "text" and right_kind == "text":
            continue
        comparable_columns += 1
        if left_kind == right_kind:
            matching_profile_columns += 1
    if comparable_columns < 2:
        return False
    return matching_profile_columns >= max(2, comparable_columns - 1)


def _rows_look_like_header_then_data(
    header_like_row: list[str],
    data_like_row: list[str],
) -> bool:
    header_schema_cells = sum(1 for text in header_like_row if _cell_value_profile_kind(text) == "header_label")
    data_value_cells = sum(
        1
        for text in data_like_row
        if _cell_value_profile_kind(text) in {"numeric", "statistical", "pathish", "code"}
    )
    return header_schema_cells >= max(2, len(header_like_row) // 2) and data_value_cells >= 1


def _cell_value_profile_kind(text: str) -> str:
    cleaned = _semantic_cell_text(text)
    if not cleaned:
        return "empty"
    if _looks_like_pathish_text(cleaned):
        return "pathish"
    normalized = str(project_pdf_math_symbol_display_text(cleaned) or "")
    if re.search(r"\([+=-]\)", normalized) or re.search(r"\d(?:\.\d+)?e\s*-?\s*\d+", cleaned, re.IGNORECASE):
        return "statistical"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\([^)]+\))?", cleaned):
        return "numeric"
    if re.fullmatch(r"[A-Z]{2,}[\w.-]*\d[\w.-]*", cleaned):
        return "code"
    if re.search(r"\b(?:vs\.?|versus)\b", cleaned, re.IGNORECASE):
        return "header_label"
    words = re.findall(r"[A-Za-z]+", cleaned)
    if words and len(words) <= 4 and not re.search(r"\d", cleaned):
        return "header_label"
    return "text"


def _build_row_views_from_raw_grid(
    raw_grid: list[list[str | None]],
    *,
    data_start_row: int = 0,
    raw_audit_grid: list[list[str | None]] | None = None,
) -> tuple[list[list[str | None]], list[list[str | None]], list[str], list[str], list[str], list[int]]:
    data_start_row = _effective_data_start_row_for_row_views(
        raw_grid,
        data_start_row=data_start_row,
    )
    projected_grid, projected_data_start_row = _project_sparse_header_continuation_rows(
        raw_grid,
        data_start_row=data_start_row,
    )
    projected_grid = _project_section_group_rows(projected_grid)
    if not _looks_like_wide_text_aligned_grid(projected_grid):
        projected_grid = _project_sparse_body_wrapped_cell_rows(
            projected_grid,
            data_start_row=projected_data_start_row,
        )
    display_grid: list[list[str | None]] = []
    data_grid: list[list[str | None]] = []
    structural_empty_rows: list[int] = []
    for row_idx, row in enumerate(projected_grid, start=1):
        if _row_has_semantic_content(row):
            display_row = project_table_grid_display_text([list(row)])[0]
            display_grid.append(display_row)
            if row_idx > projected_data_start_row:
                data_grid.append(list(display_row))
            continue
        structural_empty_rows.append(row_idx)

    display_grid = _trim_trailing_empty_columns(display_grid)
    data_grid = _trim_trailing_empty_columns(data_grid)
    raw_row_texts = _render_row_texts(raw_audit_grid if raw_audit_grid is not None else raw_grid)
    display_row_texts = _render_row_texts(display_grid)
    data_row_texts = _render_row_texts(data_grid)
    return display_grid, data_grid, raw_row_texts, display_row_texts, data_row_texts, structural_empty_rows


def _looks_like_wide_text_aligned_grid(grid: list[list[str | None]]) -> bool:
    if len(grid) < 3:
        return False
    col_count = max((len(row) for row in grid if isinstance(row, list)), default=0)
    if col_count < 5:
        return False
    non_empty_counts = [len(_semantic_non_empty_columns(row)) for row in grid if isinstance(row, list)]
    if not non_empty_counts or max(non_empty_counts) < max(4, min(col_count, 7)):
        return False
    dense_rows = sum(1 for count in non_empty_counts if count >= max(3, min(col_count - 1, 6)))
    if dense_rows < 2:
        return False
    first_row = grid[0]
    header_cells = [
        _semantic_cell_text(cell)
        for cell in first_row
        if _semantic_cell_text(cell)
    ]
    schema_cells = [
        text
        for text in header_cells
        if len(text) <= 80
        and len(text.split()) <= 8
        and bool(re.search(r"[A-Za-z\u4e00-\u9fff]", text))
    ]
    return len(schema_cells) >= 3


def _project_section_group_rows(raw_grid: list[list[str | None]]) -> list[list[str | None]]:
    grid = _clone_grid_rows(raw_grid)
    for row_idx, row in enumerate(grid):
        if len(row) < 2:
            continue
        leading_text = _semantic_cell_text(row[0])
        group_title = _extract_section_group_title(leading_text)
        if not group_title:
            continue

        populated_tail = [
            (col_idx, _semantic_cell_text(cell))
            for col_idx, cell in enumerate(row[1:], start=1)
            if _semantic_cell_text(cell)
        ]
        if not populated_tail:
            continue
        if not all(_compact_text(text) in {_compact_text(group_title), _compact_text(leading_text)} for _, text in populated_tail):
            continue

        projected = list(row)
        for col_idx, _ in populated_tail:
            projected[col_idx] = None
        grid[row_idx] = projected
    return grid


def _extract_section_group_title(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    match = re.fullmatch(r"\d+(?:\.\d+)*\s*[-－–—]\s*(\S.*)", cleaned)
    if not match:
        return ""
    return match.group(1).strip()


def _build_merged_row_metadata(grid: list[list[str | None]]) -> list[dict[str, Any]]:
    """Expose full-width semantic rows without changing rectangular grids."""
    merged_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(grid, start=1):
        if len(row) < 2:
            continue
        leading_text = _semantic_cell_text(row[0])
        if any(_semantic_cell_text(cell) for cell in row[1:]):
            continue
        if _extract_section_group_title(leading_text):
            merged_rows.append(
                {
                    "row": row_idx,
                    "kind": "section_group",
                    "text": leading_text,
                    "colspan": len(row),
                    "source": "single_leading_section_group_cell",
                }
            )
            continue
        if _is_table_note_title_row(grid, row_idx - 1):
            merged_rows.append(
                {
                    "row": row_idx,
                    "kind": "table_note_title",
                    "text": leading_text,
                    "colspan": len(row),
                    "source": "single_leading_title_cell_with_following_tabular_rows",
                }
            )
            continue
    return merged_rows


def _display_row_ref(table_id: str, row_number: int) -> str:
    return f"{table_id}:display_row:{row_number}"


def _attach_display_row_provenance(table: dict[str, Any]) -> None:
    rows = [
        row
        for row in table.get("display_grid", []) or []
        if isinstance(row, list)
    ]
    table_id = str(table.get("table_id") or table.get("block_id") or "table").strip()
    provenance = [
        {
            "row_ref": _display_row_ref(table_id, row_number),
            "source_grid": "display_grid",
            "source_row_number": row_number,
            "signature": _row_signature(row),
        }
        for row_number, row in enumerate(rows, start=1)
    ]
    table["display_row_provenance"] = provenance
    by_number = {item["source_row_number"]: item for item in provenance}
    for merged_row in table.get("merged_rows", []) or []:
        if not isinstance(merged_row, dict):
            continue
        try:
            row_number = int(merged_row.get("row", 0) or 0)
        except (TypeError, ValueError):
            continue
        source = by_number.get(row_number)
        if source is None:
            continue
        merged_row["source_grid"] = "display_grid"
        merged_row["source_row_ref"] = source["row_ref"]


def _is_table_note_title_row(grid: list[list[str | None]], row_index: int) -> bool:
    row = grid[row_index] if 0 <= row_index < len(grid) else []
    if len(row) < 2:
        return False
    leading_text = _semantic_cell_text(row[0])
    if not _looks_like_table_note_title_text(leading_text):
        return False
    if any(_semantic_cell_text(cell) for cell in row[1:]):
        return False
    following_rows = grid[row_index + 1 : row_index + 4]
    tabular_following_rows = [
        next_row
        for next_row in following_rows
        if len(_semantic_non_empty_columns(next_row)) >= 2
    ]
    return len(tabular_following_rows) >= 1


def _looks_like_table_note_title_text(text: str) -> bool:
    cleaned = _semantic_cell_text(text)
    if not cleaned:
        return False
    if len(cleaned) > 32:
        return False
    if _looks_like_pathish_text(cleaned):
        return False
    if re.fullmatch(r"[\d.\-_/]+", cleaned):
        return False
    if cleaned.endswith(("\uff1a", ":")):
        return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned[:-1]))
    if re.fullmatch(r"(?:note|notes|legend|remark|remarks)", cleaned, re.IGNORECASE):
        return True
    if re.fullmatch(r"[\u4e00-\u9fff]{1,8}", cleaned):
        note_tokens = {"说明", "备注", "注", "注释", "图例", "释义"}
        return cleaned in note_tokens
    return False


def _trim_trailing_empty_columns(grid: list[list[str | None]]) -> list[list[str | None]]:
    if not grid:
        return grid
    last_content_idx = -1
    for row in grid:
        for col_idx, cell in enumerate(row):
            if _semantic_cell_text(cell):
                last_content_idx = max(last_content_idx, col_idx)
    if last_content_idx < 0:
        return grid
    target_width = last_content_idx + 1
    return [list(row[:target_width]) for row in grid]


def _project_sparse_header_continuation_rows(
    raw_grid: list[list[str | None]],
    *,
    data_start_row: int,
) -> tuple[list[list[str | None]], int]:
    grid = _clone_grid_rows(raw_grid)
    if len(grid) < 3:
        return grid, data_start_row

    header_row_index = data_start_row - 1
    continuation_row_index = data_start_row
    data_probe_index = data_start_row + 1
    if header_row_index < 0 or data_probe_index >= len(grid):
        return grid, data_start_row

    header_row = grid[header_row_index]
    continuation_row = grid[continuation_row_index]
    next_row = grid[data_probe_index]
    if not _is_sparse_header_continuation_projection(
        header_row,
        continuation_row,
        next_row,
    ):
        return grid, data_start_row

    projected_header = list(header_row)
    for col_idx, continuation_cell in enumerate(continuation_row):
        continuation_text = _semantic_cell_text(continuation_cell)
        if not continuation_text:
            continue
        if col_idx >= len(projected_header):
            projected_header.extend([None] * (col_idx - len(projected_header) + 1))
        base_text = _semantic_cell_text(projected_header[col_idx])
        projected_header[col_idx] = _join_header_projection_text(base_text, continuation_text)

    return grid[:header_row_index] + [projected_header] + grid[data_probe_index:], data_start_row


def _is_sparse_header_continuation_projection(
    header_row: list[str | None],
    continuation_row: list[str | None],
    next_row: list[str | None],
) -> bool:
    header_cols = _semantic_non_empty_columns(header_row)
    continuation_cols = _semantic_non_empty_columns(continuation_row)
    next_cols = _semantic_non_empty_columns(next_row)
    if len(header_cols) < 2 or not continuation_cols or len(next_cols) < 2:
        return False
    if len(continuation_cols) > max(2, max(1, len(header_row)) // 2):
        return False
    if not set(continuation_cols).issubset(set(header_cols)):
        return False
    if len(next_cols) < len(header_cols):
        return False
    if _row_has_path_like_text(header_row) or _row_has_path_like_text(continuation_row):
        return False
    if not any(_cell_looks_like_data_value(next_row[col_idx]) for col_idx in next_cols):
        return False

    for col_idx in continuation_cols:
        continuation_text = _semantic_cell_text(continuation_row[col_idx])
        header_text = _semantic_cell_text(header_row[col_idx] if col_idx < len(header_row) else None)
        if not header_text or not _looks_like_header_continuation_text(continuation_text):
            return False
        if _looks_like_body_identifier_value(continuation_text):
            return False
    return True


def _can_merge_dense_overlapping_grid_fragments(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    """Merge fragments of one dense borderless grid split by competing owners.

    Dense text-layer tables can be discovered as a header/first-row fragment
    plus one or more partially overlapping body fragments. The invariant is not
    the detector source; it is a shared wide column lattice with adjacent or
    overlapping vertical coverage and repeated boundary rows.
    """
    if _has_title_conflict(primary, secondary):
        return False
    primary_grid = _get_authoritative_raw_grid(primary)
    secondary_grid = _get_authoritative_raw_grid(secondary)
    if not primary_grid or not secondary_grid:
        return False
    primary_cols = max((len(row) for row in primary_grid if isinstance(row, list)), default=0)
    secondary_cols = max((len(row) for row in secondary_grid if isinstance(row, list)), default=0)
    if primary_cols < 6 or secondary_cols < 6 or abs(primary_cols - secondary_cols) > 1:
        return False

    source_pair = {
        str(primary.get("detection_source") or primary.get("detection_method") or ""),
        str(secondary.get("detection_source") or secondary.get("detection_method") or ""),
    }
    if not source_pair & {"text_aligned_borderless_grid", "pymupdf_builtin", "word_clustering"}:
        return False

    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(primary_bbox) != 4 or len(secondary_bbox) != 4:
        return False
    horizontal_overlap = _horizontal_overlap_ratio(primary_bbox, secondary_bbox)
    if horizontal_overlap < 0.86:
        return False

    row_height = _estimate_row_height(primary) or _estimate_row_height(secondary) or 1.0
    vertical_gap = float(secondary_bbox[1]) - float(primary_bbox[3])
    vertical_overlap = max(0.0, min(float(primary_bbox[3]), float(secondary_bbox[3])) - max(float(primary_bbox[1]), float(secondary_bbox[1])))
    vertically_connected = vertical_overlap > 0.0 or vertical_gap <= max(12.0, row_height * 1.8)
    if not vertically_connected:
        return False

    similarity = column_similarity(
        list(primary.get("column_signature", [])),
        list(secondary.get("column_signature", [])),
    )
    if similarity < 0.82:
        return False

    first_signatures = {_row_signature(row) for row in primary_grid if _row_signature(row)}
    second_signatures = {_row_signature(row) for row in secondary_grid if _row_signature(row)}
    duplicate_support = bool(first_signatures & second_signatures)
    dense_support = _grid_has_dense_numeric_profile(primary_grid) and _grid_has_dense_numeric_profile(secondary_grid)
    return duplicate_support or dense_support


def _looks_like_body_identifier_value(text: str | None) -> bool:
    cleaned = _semantic_cell_text(text)
    if not cleaned:
        return False
    if re.search(r"\d", cleaned) and re.search(r"[A-Za-z]", cleaned):
        return True
    if re.fullmatch(r"[A-Z]{2,}(?:[-_/][A-Z0-9]+)+", cleaned):
        return True
    return False


def _row_has_path_like_text(row: list[str | None]) -> bool:
    return any(_looks_like_pathish_text(_semantic_cell_text(cell)) for cell in row)


def _looks_like_pathish_text(text: str) -> bool:
    cleaned = text.strip()
    lowered = cleaned.lower()
    has_file_extension = bool(re.search(r"\.[A-Za-z0-9]{2,5}\b", cleaned))
    slash_path_like = (
        "/" in cleaned
        and (
            cleaned.count("/") > 1
            or has_file_extension
            or bool(re.search(r"\d", cleaned))
        )
    )
    return (
        "\\" in cleaned
        or slash_path_like
        or has_file_extension
        or lowered.count("_") >= 2
    )


def _cell_looks_like_data_value(value: str | None) -> bool:
    text = _semantic_cell_text(value)
    if not text:
        return False
    return bool(
        re.search(r"\d", text)
        or _looks_like_pathish_text(text)
        or len(text) > 8
        or "\n" in text
    )


def _looks_like_header_continuation_text(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())
    if normalized in _HEADER_CONTINUATION_TOKENS:
        return True

    ascii_words = re.findall(r"[A-Za-z]+", text)
    if ascii_words and len(ascii_words) <= 2:
        compact = "".join(word.lower() for word in ascii_words)
        if compact in _HEADER_CONTINUATION_TOKENS:
            return True

    return text.isupper() and len(text) <= 6


def _join_header_projection_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right or left.endswith(right):
        return left
    return str(project_table_grid_display_text([[f"{left}\n{right}"]])[0][0] or "")


def _project_sparse_body_wrapped_cell_rows(
    raw_grid: list[list[str | None]],
    *,
    data_start_row: int,
) -> list[list[str | None]]:
    grid = _clone_grid_rows(raw_grid)
    if len(grid) < 3:
        return grid

    projected: list[list[str | None]] = []
    row_idx = 0
    while row_idx < len(grid):
        row = list(grid[row_idx])
        if row_idx <= data_start_row:
            projected.append(row)
            row_idx += 1
            continue

        content_col = _find_sparse_wrapped_cell_anchor_column(row)
        if content_col is None:
            projected.append(row)
            row_idx += 1
            continue

        fragments: list[dict[str, Any]] = []
        lookahead_idx = row_idx + 1
        while lookahead_idx < len(grid) and _is_sparse_wrapped_cell_fragment_row(
            anchor_row=row,
            fragment_row=grid[lookahead_idx],
            content_col=content_col,
        ):
            fragments.append(
                {
                    "row_idx": lookahead_idx,
                    "text": _semantic_cell_text(grid[lookahead_idx][content_col]),
                }
            )
            lookahead_idx += 1

        if not fragments:
            projected.append(row)
            row_idx += 1
            continue

        merged_text = _join_sparse_wrapped_cell_fragments(
            [_semantic_cell_text(row[content_col])] + [item["text"] for item in fragments]
        )
        if merged_text:
            row[content_col] = merged_text
        projected.append(row)
        row_idx = lookahead_idx

    return projected


def _find_sparse_wrapped_cell_anchor_column(row: list[str | None]) -> int | None:
    non_empty_columns = _semantic_non_empty_columns(row)
    if len(non_empty_columns) < 2:
        return None

    phrase_columns = [
        col_idx
        for col_idx in non_empty_columns
        if _looks_like_sparse_wrapped_phrase_fragment(_semantic_cell_text(row[col_idx]))
    ]
    if len(phrase_columns) != 1:
        return None

    content_col = phrase_columns[0]
    if not any(
        _cell_looks_like_row_attribute(row[col_idx])
        for col_idx in non_empty_columns
        if col_idx != content_col
    ):
        return None
    return content_col


def _is_sparse_wrapped_cell_fragment_row(
    *,
    anchor_row: list[str | None],
    fragment_row: list[str | None],
    content_col: int,
) -> bool:
    fragment_cols = _semantic_non_empty_columns(fragment_row)
    if fragment_cols != [content_col]:
        return False

    fragment_text = _semantic_cell_text(fragment_row[content_col])
    if not _looks_like_sparse_wrapped_phrase_fragment(fragment_text):
        return False

    anchor_cols = _semantic_non_empty_columns(anchor_row)
    if content_col not in anchor_cols:
        return False
    if any(
        _cell_looks_like_row_attribute(fragment_row[col_idx] if col_idx < len(fragment_row) else None)
        for col_idx in range(len(anchor_row))
        if col_idx != content_col
    ):
        return False

    return any(
        _cell_looks_like_row_attribute(anchor_row[col_idx])
        for col_idx in anchor_cols
        if col_idx != content_col
    )


def _looks_like_sparse_wrapped_phrase_fragment(text: str) -> bool:
    cleaned = _semantic_cell_text(text)
    if not cleaned or "\n" in cleaned:
        return False
    if len(cleaned) > 48:
        return False
    if _looks_like_pathish_text(cleaned):
        return False
    if _cell_looks_like_row_attribute(cleaned):
        return False
    if _ends_with_strong_terminal(cleaned):
        return False
    if re.fullmatch(r"[\d.\-_/]+", cleaned):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z-]*|\d+[A-Za-z-]+|[A-Za-z-]+\d+|[\u4e00-\u9fff]+", cleaned)
    if not words:
        return False
    if len(words) > 4:
        return False
    return True


def _cell_looks_like_row_attribute(value: str | None) -> bool:
    text = _semantic_cell_text(value)
    if not text:
        return False
    if _looks_like_pathish_text(text):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)+", text):
        return True
    if re.fullmatch(r"\d{1,2}[/-][A-Za-z]{3,9}[/-]\d{2,4}", text, re.IGNORECASE):
        return True
    if re.fullmatch(r"[A-Z]{2,}\s*\d[\w.-]*", text):
        return True
    return False


def _join_sparse_wrapped_cell_fragments(parts: list[str]) -> str:
    cleaned_parts = [_semantic_cell_text(part) for part in parts if _semantic_cell_text(part)]
    if not cleaned_parts:
        return ""
    return str(project_table_grid_display_text([["\n".join(cleaned_parts)]])[0][0] or "")


def _get_authoritative_raw_grid(table: dict[str, Any]) -> list[list[str | None]]:
    raw_grid = table.get("raw_grid")
    if isinstance(raw_grid, list):
        return _clone_grid_rows(raw_grid)
    return _clone_grid_rows(table.get("grid") or [])


def _get_display_grid(table: dict[str, Any]) -> list[list[str | None]]:
    display_grid = table.get("display_grid")
    if isinstance(display_grid, list):
        return _clone_grid_rows(display_grid)
    return _clone_grid_rows(table.get("grid") or [])


def _get_data_start_row(table: dict[str, Any]) -> int:
    raw_value = table.get("data_start_row")
    if isinstance(raw_value, int) and raw_value >= 0:
        return raw_value

    header_row_index = table.get("header_row_index")
    if isinstance(header_row_index, int) and header_row_index >= 0:
        return header_row_index + 1

    title_row_index = table.get("title_row_index")
    if isinstance(title_row_index, int) and title_row_index >= 0:
        return title_row_index + 1

    return 0


def _build_header_from_table(table: dict[str, Any]) -> list[dict[str, Any]]:
    raw_grid = _get_authoritative_raw_grid(table)
    if not raw_grid:
        return []

    header_row_index = table.get("header_row_index")
    if isinstance(header_row_index, int) and 0 <= header_row_index < len(raw_grid):
        return _build_header_from_grid([raw_grid[header_row_index]])

    data_start_row = _get_data_start_row(table)
    fallback_row_index = data_start_row - 1
    if 0 <= fallback_row_index < len(raw_grid):
        return _build_header_from_grid([raw_grid[fallback_row_index]])

    display_grid = _get_display_grid(table)
    return _build_header_from_grid(display_grid)


def _apply_header_from_candidates(table: dict[str, Any]) -> bool:
    candidates = table.get("header_candidates") or []
    if not candidates:
        return False
    authoritative_col_count = _authoritative_grid_col_count(table)
    if authoritative_col_count > 0 and len(candidates) != authoritative_col_count:
        return False
    projected = project_table_grid_display_text([[str(text).strip() for text in candidates]])[0]
    table["header"] = [{"col": idx + 1, "text": text} for idx, text in enumerate(projected)]
    table["col_count"] = len(projected)
    table["header_rebuilt_by_guard"] = True
    return True


def _authoritative_grid_col_count(table: dict[str, Any]) -> int:
    """Return the strongest local column count preserved in table grids.

    Header candidates are reconstructed from surrounding text blocks and can be
    under-segmented when adjacent header cells have narrow visual gaps. Raw and
    display grids are stronger evidence because they were projected from the
    accepted table candidate itself. Guard-time header rebuilding must not let a
    weaker candidate collapse those columns.
    """
    counts: list[int] = []
    for key in ("raw_grid", "display_grid", "grid"):
        grid = table.get(key)
        if not isinstance(grid, list):
            continue
        for row in grid:
            if isinstance(row, list):
                counts.append(len(row))
    return max(counts, default=0)


def _align_header_row(table: dict[str, Any]) -> None:
    """Ensure the grid's first row mirrors header metadata when header tokens spill across cells."""
    had_raw_grid = isinstance(table.get("raw_grid"), list)
    grid = _get_authoritative_raw_grid(table)
    header = table.get("header") or []
    if not grid or not header:
        return

    first_row = grid[0]
    header_texts = [str(item.get("text", "")).strip() for item in header]
    normalized_headers = [text for text in (_compact_text(text) for text in header_texts) if text]
    if not normalized_headers:
        return
    if not _has_local_header_evidence(first_row, header_texts, inherited=bool(table.get("header_inherited"))):
        return

    combined = _compact_text(" ".join(str(cell or "") for cell in first_row))
    match_threshold = max(2, len(normalized_headers) // 2)
    match_count = sum(1 for text in normalized_headers if text in combined)
    if match_count < match_threshold:
        return

    updated = False
    for idx, header_text in enumerate(header_texts):
        normalized_header = _compact_text(header_text)
        if not normalized_header:
            continue
        if idx >= len(first_row):
            first_row.extend([None] * (idx - len(first_row) + 1))
        current_value = first_row[idx]
        normalized_current = _compact_text(str(current_value)) if current_value else ""
        if not normalized_current:
            first_row[idx] = header_text
            updated = True
            continue
        if normalized_header in normalized_current and normalized_current != normalized_header:
            first_row[idx] = header_text
            updated = True

    if updated:
        grid[0] = first_row
        table["raw_grid"] = grid
        if not had_raw_grid:
            table["grid"] = _clone_grid_rows(grid)


def extract_trailing_table_note_rows(table: dict[str, Any]) -> bool:
    """Move trailing table-owned note rows into note metadata.

    Borderless word-cluster extraction can pull a below-table footnote into the
    rectangular grid because it is horizontally aligned with the first column.
    Keep the raw note text as table-owned evidence, but remove it from the data
    grid so marked cells can be linked to it.
    """
    raw_grid = _get_authoritative_raw_grid(table)
    if len(raw_grid) < 3:
        return False
    detection_source = str(table.get("detection_source") or table.get("detection_method") or "").strip()
    allow_explanatory_note_rows = detection_source in {
        "caption_anchored_horizontal_rules",
        "text_aligned_borderless_grid",
        "structured_text_region",
        "word_clustering",
    }

    extracted: list[tuple[int, str]] = []
    explanatory_note_row_count = 0
    while raw_grid:
        row = raw_grid[-1]
        non_empty = [
            (idx, _semantic_cell_text(cell))
            for idx, cell in enumerate(row)
            if _semantic_cell_text(cell)
        ]
        if len(non_empty) != 1:
            if (
                explanatory_note_row_count >= 1
                or not allow_explanatory_note_rows
                or not _looks_like_trailing_explanatory_table_note_row(row, raw_grid[:-1])
            ):
                break
            note_text = _semantic_cell_text(" ".join(text for _, text in non_empty))
            extracted.append((len(raw_grid) - 1, note_text))
            explanatory_note_row_count += 1
            raw_grid = raw_grid[:-1]
            continue
        col_idx, note_text = non_empty[0]
        if col_idx <= 1 and _is_explicit_trailing_table_note_label(note_text):
            extracted.append((len(raw_grid) - 1, note_text))
            raw_grid = raw_grid[:-1]
            continue
        marker = _table_note_row_marker(note_text)
        if marker:
            if col_idx > 1:
                break
            previous_grid = raw_grid[:-1]
            if not _grid_has_marked_cell(previous_grid, marker):
                break
            extracted.append((len(raw_grid) - 1, note_text))
            raw_grid = previous_grid
            continue
        if (
            explanatory_note_row_count >= 1
            or col_idx > 1
            or not allow_explanatory_note_rows
            or not _looks_like_trailing_explanatory_table_note_row(row, raw_grid[:-1])
        ):
            break
        extracted.append((len(raw_grid) - 1, note_text))
        explanatory_note_row_count += 1
        raw_grid = raw_grid[:-1]

    if not extracted:
        return False

    extracted.reverse()
    table["raw_grid"] = raw_grid
    table["grid"] = _clone_grid_rows(raw_grid)
    table["trailing_note_rows_extracted"] = True
    table_id = str(table.get("table_id") or "table").strip() or "table"
    try:
        physical_page = int(table.get("page", 0) or 0)
    except (TypeError, ValueError):
        physical_page = 0
    table_bbox = table.get("bbox")
    try:
        source_order_x = float(table_bbox[0]) if len(table_bbox) >= 4 else None
        source_order_y = float(table_bbox[3]) if len(table_bbox) >= 4 else None
    except (TypeError, ValueError):
        source_order_x = None
        source_order_y = None
    existing_notes = [dict(item) for item in table.get("note_blocks", []) or [] if isinstance(item, dict)]
    seen_texts = {_semantic_cell_text(note.get("text")) for note in existing_notes}
    for row_idx, note_text in extracted:
        if _semantic_cell_text(note_text) in seen_texts:
            continue
        existing_notes.append(
            {
                "text": note_text,
                "role": "note",
                "relation": "below",
                "source": "trailing_table_note_row",
                "source_block_id": f"{table_id}:trailing_note_row_{row_idx}",
                "source_grid": "raw_grid",
                "source_row_number": row_idx + 1,
                "source_row_ref": f"{table_id}:raw_row:{row_idx + 1}",
                "physical_page": physical_page,
                **(
                    {
                        "source_order_x": source_order_x,
                        "source_order_y": source_order_y,
                    }
                    if source_order_x is not None and source_order_y is not None
                    else {}
                ),
            }
        )
        seen_texts.add(_semantic_cell_text(note_text))
    table["note_blocks"] = existing_notes
    return True


def _is_explicit_trailing_table_note_label(text: str) -> bool:
    cleaned = _semantic_cell_text(text)
    return bool(cleaned and re.fullmatch(r"(?:附加信息|备注|注释|说明)\s*[:：]?", cleaned))


def _looks_like_trailing_explanatory_table_note_row(
    row: list[str | None],
    previous_grid: list[list[str | None]],
) -> bool:
    if len(previous_grid) < 3:
        return False
    text = _semantic_cell_text(" ".join(_semantic_cell_text(cell) for cell in row if _semantic_cell_text(cell)))
    if not text:
        return False
    if len(text) < 16:
        return False
    if _looks_like_unmarked_section_heading_note_false_positive(text):
        return False
    if re.match(r"^\s*(?:table|tab\.?|表)\s*\d*", text, re.IGNORECASE) or re.match(
        r"^\s*(?:fig(?:ure)?\.?|图)\s*\d+",
        text,
        re.IGNORECASE,
    ):
        return False
    non_empty_cols = _semantic_non_empty_columns(row)
    if len(non_empty_cols) > max(3, len(row) // 2):
        return False
    if sum(1 for cell in row if _cell_value_profile_kind(_semantic_cell_text(cell)) in {"numeric", "statistical", "code"}) > 0:
        return False
    prior_value_rows = 0
    for prior in previous_grid[-8:]:
        values = [_semantic_cell_text(cell) for cell in prior if _semantic_cell_text(cell)]
        if len(values) < 2:
            continue
        value_like = sum(
            1
            for value in values
            if _cell_value_profile_kind(value) in {"numeric", "statistical", "pathish", "code"}
            or bool(re.search(r"\d", value))
        )
        if value_like >= 1:
            prior_value_rows += 1
    if prior_value_rows < 2:
        return False
    prose_like_rows = 0
    body_rows = previous_grid[2:] if len(previous_grid) > 3 else previous_grid
    for prior in body_rows:
        prior_text = _semantic_cell_text(" ".join(_semantic_cell_text(cell) for cell in prior if _semantic_cell_text(cell)))
        if _looks_like_multi_cell_body_prose_row(prior_text):
            prose_like_rows += 1
    if prose_like_rows >= max(3, len(body_rows) // 2):
        return False
    return True


def _looks_like_unmarked_section_heading_note_false_positive(text: str) -> bool:
    cleaned = _semantic_cell_text(text)
    if not cleaned or len(cleaned) > 90:
        return False
    if re.search(r"[.;!?銆傦紱锛沨]$", cleaned):
        return False
    if re.match(r"^\s*\d+(?:\.\d+)*\s+\S+", cleaned):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z-]*", cleaned)
    if 1 <= len(words) <= 7 and len(words) == len(cleaned.split()):
        lower_words = {word.lower() for word in words}
        if any(word in {"note", "notes", "mean", "means", "average", "calculated", "determined", "measured"} for word in lower_words):
            return False
        stop_words = {"the", "and", "or", "of", "to", "in", "for", "with", "by", "on", "a", "an"}
        return any(word not in stop_words for word in lower_words)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    return 2 <= cjk_count <= 18 and not re.search(r"[，,。；;：:]", cleaned)


def _looks_like_multi_cell_body_prose_row(text: str) -> bool:
    cleaned = _semantic_cell_text(text)
    if not cleaned:
        return False
    word_count = len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", cleaned))
    if word_count >= 8:
        return True
    if word_count >= 5 and re.search(r"\b(?:the|and|of|to|in|on|with|from|for|is|are|when|that|this)\b", cleaned, re.IGNORECASE):
        return True
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    return cjk_count >= 16 and bool(re.search(r"[，。；：]", cleaned))


def _table_note_row_marker(text: str) -> str | None:
    cleaned = _semantic_cell_text(text)
    if not cleaned:
        return None
    match = re.match(r"^\s*(?P<marker>[#\$]|\*+|[+\u2020\u2021])(?=\s*[-:：)\]】\u4e00-\u9fffA-Za-z])", cleaned)
    if not match:
        return None
    return str(match.group("marker") or "").strip() or None


def _grid_has_marked_cell(grid: list[list[str | None]], marker: str) -> bool:
    if not marker:
        return False
    for row in grid:
        for cell in row:
            text = _semantic_cell_text(cell)
            if text and _cell_text_has_note_marker(text, marker):
                return True
    return False


def _cell_text_has_note_marker(text: str, marker: str) -> bool:
    cleaned = _semantic_cell_text(text)
    marker = str(marker or "").strip()
    if not cleaned or not marker:
        return False
    escaped = re.escape(marker)
    return bool(re.search(rf"{escaped}(?:\s*$|(?=[,，;；:：)\]】]))", cleaned))


def _project_simple_multiline_header_from_raw_grid(
    table: dict[str, Any],
    raw_grid: list[list[str | None]],
    col_count: int,
) -> bool:
    if table.get("external_header_projection"):
        return False
    header_cells = table.get("header") or []
    if len(header_cells) != col_count:
        return False
    data_start = _get_data_start_row(table)
    if data_start < 2 or data_start >= len(raw_grid):
        return False
    top_idx = data_start - 2
    continuation_idx = data_start - 1
    if top_idx < 0:
        return False
    top_row = raw_grid[top_idx]
    continuation_row = raw_grid[continuation_idx]
    if _semantic_non_empty_columns(top_row) == _semantic_non_empty_columns(continuation_row):
        return False
    continuation_cols = _semantic_non_empty_columns(continuation_row)
    if not continuation_cols or len(continuation_cols) > max(1, col_count // 2):
        return False
    body_row = raw_grid[data_start] if data_start < len(raw_grid) else []
    if not body_row or sum(1 for value in body_row if _cell_value_profile_kind(_semantic_cell_text(value)) in {"numeric", "statistical"}) <= 0:
        return False

    changed = False
    projected_header = [dict(cell) for cell in header_cells]
    for col_idx in continuation_cols:
        if col_idx >= len(projected_header):
            continue
        continuation_text = _semantic_cell_text(continuation_row[col_idx] if col_idx < len(continuation_row) else None)
        if not continuation_text or _cell_value_profile_kind(continuation_text) in {"numeric", "statistical"}:
            continue
        base_text = _semantic_cell_text(projected_header[col_idx].get("text"))
        if not base_text:
            continue
        merged = f"{base_text} {continuation_text}".strip()
        if merged != base_text:
            projected_header[col_idx]["text"] = merged
            changed = True

    if not changed:
        return False
    table["header"] = projected_header
    table.setdefault("header_row_groups", [])
    table["header_row_groups"].append(
        {
            "start_row": top_idx,
            "end_row": continuation_idx,
            "source": "simple_multiline_header_projection",
        }
    )
    return True


def project_table_header_grammar(table: dict[str, Any]) -> bool:
    """Project generic multi-row table header grammar into AST metadata.

    This layer works after leaf columns already exist. It does not infer new
    columns; it classifies header rows into leaf headers and group/spanning
    headers using local evidence: non-placeholder header text, continuation
    header rows, body values under the affected columns, and path/slash guards.
    """
    if table.get("external_header_projection"):
        return False
    raw_grid = _get_authoritative_raw_grid(table)
    if len(raw_grid) < 3:
        return False
    col_count = int(table.get("col_count", 0) or 0)
    if col_count < 3:
        return False
    if any(len(row) < col_count for row in raw_grid[:3]):
        return False
    if _project_simple_multiline_header_from_raw_grid(table, raw_grid, col_count):
        return True

    header_row_index = table.get("header_row_index")
    if isinstance(header_row_index, int) and header_row_index >= 0:
        top_idx = header_row_index
    else:
        data_start = _get_data_start_row(table)
        top_idx = max(0, data_start - 1)
    continuation_idx = top_idx + 1
    body_idx = continuation_idx + 1
    if body_idx >= len(raw_grid):
        return False

    top_row = list(raw_grid[top_idx])
    continuation_row = list(raw_grid[continuation_idx])
    body_rows = raw_grid[body_idx:]
    _repair_shifted_rowspan_stub_header(top_row, continuation_row, body_rows, col_count)
    _split_trailing_leaf_header_from_fragmented_group(top_row, continuation_row, col_count)
    if _project_upper_group_with_lower_stub_header(
        table,
        raw_grid=raw_grid,
        top_row=top_row,
        continuation_row=continuation_row,
        body_rows=body_rows,
        top_idx=top_idx,
        continuation_idx=continuation_idx,
        col_count=col_count,
    ):
        return True
    if not _looks_like_table_header_grammar_rows(top_row, continuation_row, body_rows, col_count):
        return False

    projected_top = list(top_row)
    projected_continuation = list(continuation_row)
    header_texts = ["" for _ in range(col_count)]
    header_groups: list[dict[str, Any]] = []
    header_row_groups: list[dict[str, Any]] = []
    occupied_leaf_cols: set[int] = set()
    changed = False

    for col_idx in range(col_count):
        top_text = _semantic_header_cell_text(projected_top[col_idx], col_idx)
        continuation_text = _semantic_header_cell_text(projected_continuation[col_idx], col_idx)
        if (
            top_text
            and not continuation_text
            and not (
                _count_slashes_outside_grouping(top_text) > 0
                and col_idx + 1 < col_count
                and not _semantic_header_cell_text(projected_top[col_idx + 1], col_idx + 1)
                and _semantic_header_cell_text(projected_continuation[col_idx + 1], col_idx + 1)
            )
            and _column_has_stub_body_values(body_rows, col_idx, col_count)
            and _looks_like_rowspan_stub_header(top_text, body_rows, col_idx)
        ):
            header_texts[col_idx] = top_text
            header_row_groups.append(
                {
                    "col": col_idx,
                    "start_row": top_idx,
                    "end_row": continuation_idx,
                    "rowspan": continuation_idx - top_idx + 1,
                    "text": top_text,
                    "source": "table_header_grammar",
                }
            )
            occupied_leaf_cols.add(col_idx)
            changed = True

    for col_idx in range(col_count):
        if col_idx in occupied_leaf_cols:
            continue
        top_text = _semantic_header_cell_text(projected_top[col_idx], col_idx)
        continuation_text = _semantic_header_cell_text(projected_continuation[col_idx], col_idx)
        if (
            top_text
            and col_idx + 1 < col_count
            and _trailing_slash_outside_grouping(top_text)
            and not _semantic_header_cell_text(projected_top[col_idx + 1], col_idx + 1)
            and continuation_text
            and not _semantic_header_cell_text(projected_continuation[col_idx + 1], col_idx + 1)
            and _column_has_body_values(body_rows, col_idx, col_count)
            and _column_has_body_values(body_rows, col_idx + 1, col_count)
        ):
            left_text = _strip_trailing_slash_header_text(top_text)
            if left_text:
                header_texts[col_idx] = left_text
                header_texts[col_idx + 1] = continuation_text
                projected_top[col_idx] = left_text
                projected_top[col_idx + 1] = continuation_text
                projected_continuation[col_idx] = None
                projected_continuation[col_idx + 1] = None
                header_row_groups.extend(
                    _build_leaf_rowspan_groups(
                        top_idx=top_idx,
                        continuation_idx=continuation_idx,
                        cells=[(col_idx, left_text), (col_idx + 1, continuation_text)],
                    )
                )
                occupied_leaf_cols.update({col_idx, col_idx + 1})
                changed = True
                continue

        if (
            "/" in top_text
            and col_idx + 1 < col_count
            and not _semantic_header_cell_text(projected_top[col_idx + 1], col_idx + 1)
            and _semantic_header_cell_text(projected_continuation[col_idx + 1], col_idx + 1)
            and _column_has_body_values(body_rows, col_idx, col_count)
            and _column_has_body_values(body_rows, col_idx + 1, col_count)
        ):
            next_continuation_text = _semantic_header_cell_text(projected_continuation[col_idx + 1], col_idx + 1)
            left_text, right_seed = _split_slash_adjacent_header_text(top_text)
            if left_text and _header_suffix_matches_continuation(right_seed, next_continuation_text):
                header_texts[col_idx] = left_text
                header_texts[col_idx + 1] = _join_header_projection_text(right_seed, next_continuation_text)
                projected_top[col_idx] = left_text
                projected_top[col_idx + 1] = header_texts[col_idx + 1]
                projected_continuation[col_idx] = None
                projected_continuation[col_idx + 1] = None
                header_row_groups.extend(
                    _build_leaf_rowspan_groups(
                        top_idx=top_idx,
                        continuation_idx=continuation_idx,
                        cells=[(col_idx, left_text), (col_idx + 1, header_texts[col_idx + 1])],
                    )
                )
                occupied_leaf_cols.update({col_idx, col_idx + 1})
                changed = True
                continue

        if (
            top_text
            and "/" not in top_text
            and col_idx + 1 < col_count
            and not _is_inside_fragmented_group_label(projected_top, col_idx)
        ):
            preceding_occupied_leaf_cols = [leaf_col for leaf_col in occupied_leaf_cols if leaf_col < col_idx]
            min_start_col = (max(preceding_occupied_leaf_cols) + 1) if preceding_occupied_leaf_cols else None
            span = _infer_continuation_header_group_span(
                top_row,
                continuation_row,
                body_rows,
                label_col=col_idx,
                min_start_col=min_start_col,
                col_count=col_count,
            )
            if span is not None:
                group_start_col, span_end = span
                group_text = _compose_fragmented_group_header_text(
                    top_row,
                    continuation_row,
                    group_start_col,
                    span_end,
                )
                if not group_text:
                    continue
                header_groups.append(
                    {
                        "row": top_idx,
                        "start_col": group_start_col,
                        "end_col": span_end,
                        "colspan": span_end - group_start_col + 1,
                        "text": group_text,
                        "source": "table_header_grammar",
                    }
                )
                projected_top[group_start_col] = group_text
                if group_start_col != col_idx:
                    projected_top[col_idx] = None
                for covered_col in range(group_start_col + 1, span_end + 1):
                    projected_top[covered_col] = None
                for covered_col in range(group_start_col, span_end + 1):
                    leaf_text = _semantic_header_cell_text(continuation_row[covered_col], covered_col)
                    if leaf_text:
                        header_texts[covered_col] = leaf_text
                occupied_leaf_cols.update(range(group_start_col, span_end + 1))
                changed = True

    for col_idx in range(col_count):
        if header_texts[col_idx]:
            continue
        continuation_text = _semantic_header_cell_text(continuation_row[col_idx], col_idx)
        top_text = _semantic_header_cell_text(top_row[col_idx], col_idx)
        if continuation_text and _column_has_body_values(body_rows, col_idx, col_count):
            header_texts[col_idx] = continuation_text
        elif top_text and _column_has_body_values(body_rows, col_idx, col_count):
            header_texts[col_idx] = top_text

    if not changed:
        return False
    if sum(1 for text in header_texts if text) < max(2, col_count - 1):
        return False

    projected_raw_grid = _clone_grid_rows(raw_grid)
    projected_raw_grid[top_idx] = projected_top
    projected_raw_grid[continuation_idx] = projected_continuation
    table["raw_grid"] = projected_raw_grid
    table["header"] = [
        {"col": idx + 1, "text": text or f"Column {idx + 1}"}
        for idx, text in enumerate(header_texts[:col_count])
    ]
    current_data_start = _get_data_start_row(table)
    table["data_start_row"] = max(current_data_start, continuation_idx + 1)
    if not isinstance(table.get("header_row_index"), int):
        table["header_row_index"] = top_idx
    if header_groups:
        existing = [dict(item) for item in table.get("header_column_groups", []) or [] if isinstance(item, dict)]
        seen = {
            (
                int(item.get("row", -1) or -1),
                int(item.get("start_col", -1) or -1),
                int(item.get("end_col", -1) or -1),
                str(item.get("text", "") or ""),
            )
            for item in existing
        }
        for group in header_groups:
            key = (
                int(group.get("row", -1) or -1),
                int(group.get("start_col", -1) or -1),
                int(group.get("end_col", -1) or -1),
                str(group.get("text", "") or ""),
            )
            if key not in seen:
                existing.append(group)
                seen.add(key)
        table["header_column_groups"] = existing
    if header_row_groups:
        existing_rows = [dict(item) for item in table.get("header_row_groups", []) or [] if isinstance(item, dict)]
        seen_rows = {
            (
                int(item.get("col", -1) or -1),
                int(item.get("start_row", -1) or -1),
                int(item.get("end_row", -1) or -1),
                str(item.get("text", "") or ""),
            )
            for item in existing_rows
        }
        for group in header_row_groups:
            key = (
                int(group.get("col", -1) or -1),
                int(group.get("start_row", -1) or -1),
                int(group.get("end_row", -1) or -1),
                str(group.get("text", "") or ""),
            )
            if key not in seen_rows:
                existing_rows.append(group)
                seen_rows.add(key)
        table["header_row_groups"] = existing_rows
    table["table_header_grammar_projected"] = True
    _refresh_row_texts_from_grid(table)
    _restore_header_grammar_display_rows(
        table,
        projected_top=projected_top,
        projected_continuation=projected_continuation,
        top_idx=top_idx,
        continuation_idx=continuation_idx,
        col_count=col_count,
    )
    table["cells"] = _build_cells_from_grid(table.get("grid", []))
    return True


def project_compound_spanning_table_headers(table: dict[str, Any]) -> bool:
    """Backward-compatible wrapper for the unified table header grammar layer."""
    changed = project_table_header_grammar(table)
    if changed:
        table["compound_header_projected"] = True
    return changed


def _project_upper_group_with_lower_stub_header(
    table: dict[str, Any],
    *,
    raw_grid: list[list[str | None]],
    top_row: list[str | None],
    continuation_row: list[str | None],
    body_rows: list[list[str | None]],
    top_idx: int,
    continuation_idx: int,
    col_count: int,
) -> bool:
    """Handle a one-line group header above a lower-row stub plus leaf headers."""
    top_non_empty = [
        col_idx
        for col_idx in range(col_count)
        if _semantic_header_cell_text(top_row[col_idx], col_idx)
    ]
    if len(top_non_empty) != 1:
        return False
    group_label_col = top_non_empty[0]
    group_text = _semantic_header_cell_text(top_row[group_label_col], group_label_col)
    if not group_text or "/" in group_text or _row_has_document_path_like_header_text([group_text]):
        return False
    stub_text = _semantic_header_cell_text(continuation_row[0], 0)
    if not stub_text:
        return False
    leaf_texts = [
        _semantic_header_cell_text(continuation_row[col_idx], col_idx)
        for col_idx in range(1, col_count)
    ]
    if sum(1 for text in leaf_texts if text) < max(2, col_count - 2):
        return False
    if not _column_has_stub_body_values(body_rows, 0, col_count):
        return False
    if sum(1 for col_idx in range(1, col_count) if _column_has_body_values(body_rows, col_idx, col_count)) < max(2, col_count - 1):
        return False
    if not (1 <= group_label_col <= col_count - 1):
        return False

    header_texts = [stub_text] + [text or f"Column {idx + 2}" for idx, text in enumerate(leaf_texts)]
    projected_top: list[str | None] = [None] * col_count
    projected_continuation: list[str | None] = [None] * col_count
    projected_top[0] = stub_text
    projected_top[1] = group_text
    for col_idx in range(1, col_count):
        projected_continuation[col_idx] = header_texts[col_idx]

    projected_raw_grid = _clone_grid_rows(raw_grid)
    projected_raw_grid[top_idx] = projected_top
    projected_raw_grid[continuation_idx] = projected_continuation
    table["raw_grid"] = projected_raw_grid
    table["header"] = [
        {"col": idx + 1, "text": text}
        for idx, text in enumerate(header_texts)
    ]
    table["header_row_index"] = top_idx
    table["data_start_row"] = max(_get_data_start_row(table), continuation_idx + 1)
    table["header_column_groups"] = _append_unique_header_column_groups(
        table.get("header_column_groups"),
        [
            {
                "row": top_idx,
                "start_col": 1,
                "end_col": col_count - 1,
                "colspan": col_count - 1,
                "text": group_text,
                "source": "table_header_grammar",
            }
        ],
    )
    table["header_row_groups"] = _append_unique_header_row_groups(
        table.get("header_row_groups"),
        [
            {
                "col": 0,
                "start_row": top_idx,
                "end_row": continuation_idx,
                "rowspan": continuation_idx - top_idx + 1,
                "text": stub_text,
                "source": "table_header_grammar",
            }
        ],
    )
    table["table_header_grammar_projected"] = True
    _refresh_row_texts_from_grid(table)
    _restore_header_grammar_display_rows(
        table,
        projected_top=projected_top,
        projected_continuation=projected_continuation,
        top_idx=top_idx,
        continuation_idx=continuation_idx,
        col_count=col_count,
    )
    table["cells"] = _build_cells_from_grid(table.get("grid", []))
    return True


def _append_unique_header_column_groups(
    existing_value: Any,
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = [dict(item) for item in existing_value or [] if isinstance(item, dict)]
    seen = {
        (
            int(item.get("row", -1) or -1),
            int(item.get("start_col", -1) or -1),
            int(item.get("end_col", -1) or -1),
            str(item.get("text", "") or ""),
        )
        for item in existing
    }
    for group in additions:
        key = (
            int(group.get("row", -1) or -1),
            int(group.get("start_col", -1) or -1),
            int(group.get("end_col", -1) or -1),
            str(group.get("text", "") or ""),
        )
        if key not in seen:
            existing.append(group)
            seen.add(key)
    return existing


def _append_unique_header_row_groups(
    existing_value: Any,
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = [dict(item) for item in existing_value or [] if isinstance(item, dict)]
    seen = {
        (
            int(item.get("col", -1) or -1),
            int(item.get("start_row", -1) or -1),
            int(item.get("end_row", -1) or -1),
            str(item.get("text", "") or ""),
        )
        for item in existing
    }
    for group in additions:
        key = (
            int(group.get("col", -1) or -1),
            int(group.get("start_row", -1) or -1),
            int(group.get("end_row", -1) or -1),
            str(group.get("text", "") or ""),
        )
        if key not in seen:
            existing.append(group)
            seen.add(key)
    return existing


def _restore_header_grammar_display_rows(
    table: dict[str, Any],
    *,
    projected_top: list[str | None],
    projected_continuation: list[str | None],
    top_idx: int,
    continuation_idx: int,
    col_count: int,
) -> None:
    """Keep proven header span empty cells from being compacted in display rows."""
    display_grid = _get_display_grid(table)
    if not display_grid:
        return
    raw_grid = _get_authoritative_raw_grid(table)
    visible_raw_indices = [
        row_idx
        for row_idx, row in enumerate(raw_grid)
        if _row_has_semantic_content(row)
    ]
    raw_to_display = {raw_idx: display_idx for display_idx, raw_idx in enumerate(visible_raw_indices)}
    replacement_rows = {
        top_idx: projected_top[:col_count],
        continuation_idx: projected_continuation[:col_count],
    }
    changed = False
    for raw_idx, replacement in replacement_rows.items():
        display_idx = raw_to_display.get(raw_idx)
        if display_idx is None or display_idx >= len(display_grid):
            continue
        display_grid[display_idx] = project_table_grid_display_text([list(replacement)])[0]
        changed = True
    if not changed:
        return
    table["display_grid"] = _trim_trailing_empty_columns(display_grid)
    table["display_row_texts"] = _render_row_texts(table["display_grid"])
    table["display_row_count"] = len(table["display_grid"])
    _attach_display_row_provenance(table)


def _looks_like_table_header_grammar_rows(
    top_row: list[str | None],
    continuation_row: list[str | None],
    body_rows: list[list[str | None]],
    col_count: int,
) -> bool:
    top_cols = _semantic_non_empty_columns(top_row[:col_count])
    top_cols = [
        col_idx
        for col_idx in top_cols
        if _semantic_header_cell_text(top_row[col_idx], col_idx)
    ]
    continuation_cols = [
        col_idx
        for col_idx in _semantic_non_empty_columns(continuation_row[:col_count])
        if _semantic_header_cell_text(continuation_row[col_idx], col_idx)
    ]
    if len(top_cols) < 2 or len(continuation_cols) < 1:
        return False
    has_slash_adjacent_candidate = any("/" in _semantic_cell_text(top_row[col_idx]) for col_idx in top_cols)
    has_group_span_candidate = any(
        _infer_continuation_header_group_span(
            top_row,
            continuation_row,
            body_rows,
            label_col=col_idx,
            min_start_col=None,
            col_count=col_count,
        )
        is not None
        for col_idx in top_cols
        if "/" not in _semantic_header_cell_text(top_row[col_idx], col_idx)
    )
    if not has_slash_adjacent_candidate and not has_group_span_candidate:
        return False
    if sum(1 for col_idx in range(col_count) if _column_has_body_values(body_rows, col_idx, col_count)) < max(2, col_count - 1):
        return False
    if _row_has_document_path_like_header_text(top_row) or _row_has_document_path_like_header_text(continuation_row):
        return False
    return True


def _repair_shifted_rowspan_stub_header(
    top_row: list[str | None],
    continuation_row: list[str | None],
    body_rows: list[list[str | None]],
    col_count: int,
) -> bool:
    """Move a first-column rowspan stub header back into column 0 when shifted."""
    if col_count < 3:
        return False
    if _semantic_header_cell_text(top_row[0], 0) or _semantic_header_cell_text(continuation_row[0], 0):
        return False
    first_text_col = next(
        (
            col_idx
            for col_idx in range(1, col_count)
            if _semantic_header_cell_text(top_row[col_idx], col_idx)
        ),
        None,
    )
    if first_text_col is None or first_text_col > 1:
        return False
    stub_text = _semantic_header_cell_text(top_row[first_text_col], first_text_col)
    if not _looks_like_rowspan_stub_header(stub_text, body_rows, 0):
        return False
    if not _column_has_stub_body_values(body_rows, 0, col_count):
        return False
    top_row[0] = stub_text
    top_row[first_text_col] = None
    return True


def _looks_like_rowspan_stub_header(
    text: str,
    body_rows: list[list[str | None]],
    col_idx: int,
) -> bool:
    cleaned = _semantic_cell_text(text)
    if not cleaned or len(cleaned) > 32:
        return False
    unit_digits_only = bool(re.search(r"\d", cleaned)) and bool(re.search(r"[\(\uff08/]", cleaned))
    if re.search(r"\d", cleaned) and not unit_digits_only:
        return False
    values = [
        _semantic_cell_text(row[col_idx])
        for row in body_rows[:8]
        if col_idx < len(row) and _semantic_cell_text(row[col_idx])
    ]
    if len(values) < 2:
        return False
    numeric_like = sum(1 for value in values if _cell_looks_like_data_value(value) and re.search(r"\d", value))
    text_like = sum(1 for value in values if re.search(r"[A-Za-z\u4e00-\u9fff]", value) and not re.search(r"\d", value))
    if numeric_like >= 2 and _looks_like_unit_or_schema_header_text(cleaned):
        return True
    return text_like >= max(2, numeric_like + 1)


def _looks_like_unit_or_schema_header_text(text: str) -> bool:
    cleaned = _semantic_cell_text(text)
    if not cleaned:
        return False
    has_grouped_unit = bool(re.search(r"[\(\uff08][^)\uff09]*[/][^)\uff09]*[\)\uff09]", cleaned))
    if _looks_like_pathish_text(cleaned) and not has_grouped_unit:
        return False
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned):
        return False
    if re.search(r"\d", cleaned):
        return False
    return len(cleaned) <= 32 and (
        bool(re.search(r"[\(\uff08\[/]", cleaned))
        or bool(re.search(r"(?:dose|amount|level|concentration|reference|route|species|form)\b", cleaned, re.IGNORECASE))
        or bool(re.search(r"(?:剂量|给药|种系|剂型|参考|浓度|暴露量|途径)", cleaned))
    )


def _split_trailing_leaf_header_from_fragmented_group(
    top_row: list[str | None],
    continuation_row: list[str | None],
    col_count: int,
) -> bool:
    """Move a leaf header accidentally attached to a fragmented group unit."""
    changed = False
    for col_idx in range(1, col_count):
        top_text = _semantic_header_cell_text(top_row[col_idx], col_idx)
        if not top_text:
            continue
        if _semantic_header_cell_text(continuation_row[col_idx], col_idx):
            continue
        split = _split_unit_fragment_and_leaf_header(top_text)
        if split is None:
            continue
        unit_fragment, leaf_header = split
        top_row[col_idx] = unit_fragment
        continuation_row[col_idx] = leaf_header
        changed = True
    return changed


def _split_unit_fragment_and_leaf_header(text: str) -> tuple[str, str] | None:
    cleaned = _semantic_cell_text(text)
    if not cleaned:
        return None
    match = re.fullmatch(r"(?P<unit>.*(?:\)|\]|\uff09|\u3011))\s+(?P<leaf>\d+(?:\.\d+)?\s*[A-Za-zμµ%]+)", cleaned)
    if not match:
        return None
    unit = str(match.group("unit") or "").strip()
    leaf = str(match.group("leaf") or "").strip()
    if not unit or not leaf:
        return None
    if len(leaf) > 12:
        return None
    return unit, leaf


def _compose_fragmented_group_header_text(
    top_row: list[str | None],
    continuation_row: list[str | None],
    start_col: int,
    end_col: int,
) -> str:
    _ = continuation_row
    parts = [
        _semantic_header_cell_text(top_row[col_idx], col_idx)
        for col_idx in range(max(0, start_col), min(len(top_row), end_col + 1))
        if _semantic_header_cell_text(top_row[col_idx], col_idx)
    ]
    if not parts:
        return ""
    text = " ".join(parts)
    return _repair_fragmented_group_header_text(text)


def _repair_fragmented_group_header_text(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"\*\s+g(?=\))", "*/g", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bper\s+g(?=\))", "per g", cleaned, flags=re.IGNORECASE)
    return cleaned


def _semantic_header_cell_text(cell: Any, col_idx: int | None = None) -> str:
    text = _semantic_cell_text(cell)
    if not text:
        return ""
    match = re.fullmatch(r"Column\s+(\d+)", text, re.IGNORECASE)
    if match and (col_idx is None or int(match.group(1)) == col_idx + 1):
        return ""
    return text


def _row_has_document_path_like_header_text(row: list[str | None]) -> bool:
    for col_idx, cell in enumerate(row):
        text = _semantic_header_cell_text(cell, col_idx)
        if not text:
            continue
        slash_outside_groups = _count_slashes_outside_grouping(text)
        slash_path_like = "/" in text and (
            slash_outside_groups > 1
            or bool(re.search(r"\.[A-Za-z0-9]{2,5}\b", text))
            or "\\" in text
            or "_" in text
        )
        if slash_path_like or "\\" in text:
            return True
    return False


def _column_has_body_values(
    body_rows: list[list[str | None]],
    col_idx: int,
    col_count: int,
) -> bool:
    evidence = 0
    for row in body_rows:
        if col_idx >= min(len(row), col_count):
            continue
        text = _semantic_cell_text(row[col_idx])
        if not text:
            continue
        if _cell_looks_like_data_value(text) or _cell_looks_like_placeholder_data_value(text):
            evidence += 1
        elif re.fullmatch(r"[A-Za-z.]{1,8}", text):
            evidence += 1
    return evidence >= 1


def _cell_looks_like_placeholder_data_value(value: str | None) -> bool:
    text = _semantic_cell_text(value)
    return bool(re.fullmatch(r"[-–—]+|N/?A|n/?a|ND|N\.D\.|BLQ|BQL", text))


def _column_has_stub_body_values(
    body_rows: list[list[str | None]],
    col_idx: int,
    col_count: int,
) -> bool:
    values = [
        _semantic_cell_text(row[col_idx])
        for row in body_rows[:10]
        if col_idx < min(len(row), col_count) and _semantic_cell_text(row[col_idx])
    ]
    if len(values) < 2:
        return False
    distinct_values = {value for value in values}
    if len(distinct_values) < 2:
        return False
    text_like = sum(1 for value in values if re.search(r"[A-Za-z\u4e00-\u9fff]", value))
    data_like = sum(1 for value in values if _cell_looks_like_data_value(value))
    return text_like >= 2 or data_like >= 2


def _split_slash_adjacent_header_text(text: str) -> tuple[str, str]:
    cleaned = _semantic_cell_text(text)
    split_idx = _last_slash_outside_grouping(cleaned)
    if split_idx <= 0 or split_idx >= len(cleaned) - 1:
        return "", ""
    return cleaned[:split_idx].strip(), cleaned[split_idx + 1 :].strip()


def _trailing_slash_outside_grouping(text: str) -> bool:
    cleaned = _semantic_cell_text(text)
    if not cleaned.endswith("/"):
        return False
    return _is_outside_grouping(cleaned, len(cleaned) - 1)


def _strip_trailing_slash_header_text(text: str) -> str:
    cleaned = _semantic_cell_text(text)
    if not _trailing_slash_outside_grouping(cleaned):
        return ""
    return cleaned[:-1].strip()


def _build_leaf_rowspan_groups(
    *,
    top_idx: int,
    continuation_idx: int,
    cells: list[tuple[int, str]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for col_idx, text in cells:
        cleaned = _semantic_cell_text(text)
        if not cleaned:
            continue
        groups.append(
            {
                "col": col_idx,
                "start_row": top_idx,
                "end_row": continuation_idx,
                "rowspan": continuation_idx - top_idx + 1,
                "text": cleaned,
                "source": "table_header_grammar",
            }
        )
    return groups


def project_rowspan_body_groups(table: dict[str, Any]) -> bool:
    """Record sparse body row groups that visually behave like rowspans.

    The rectangular grid remains unchanged for auditability. This only adds
    semantic metadata when a categorical cell is followed by blank cells in the
    same leading column and those continuation rows carry data in later columns.
    """
    data_grid = _clone_grid_rows(table.get("data_grid") or table.get("grid") or [])
    col_count = int(table.get("col_count", 0) or 0)
    if len(data_grid) < 3 or col_count < 2:
        return False

    header_texts = [
        _semantic_cell_text(cell.get("text") if isinstance(cell, dict) else cell)
        for cell in table.get("header", []) or []
    ]
    candidate_cols = [
        col_idx
        for col_idx in range(min(col_count, 3))
        if col_idx < len(header_texts)
        and header_texts[col_idx]
        and not _is_generic_column_header(header_texts[col_idx])
    ]
    if not candidate_cols:
        candidate_cols = [0]

    groups: list[dict[str, Any]] = []
    for col_idx in candidate_cols:
        row_idx = 0
        while row_idx < len(data_grid):
            row = data_grid[row_idx]
            anchor_text = _semantic_cell_text(row[col_idx] if col_idx < len(row) else None)
            if not anchor_text or not _looks_like_body_rowspan_anchor_text(anchor_text):
                row_idx += 1
                continue

            end_idx = row_idx
            probe_idx = row_idx + 1
            while probe_idx < len(data_grid):
                probe_row = data_grid[probe_idx]
                probe_text = _semantic_cell_text(probe_row[col_idx] if col_idx < len(probe_row) else None)
                if probe_text:
                    break
                if not any(
                    _semantic_cell_text(probe_row[next_col] if next_col < len(probe_row) else None)
                    for next_col in range(col_idx + 1, col_count)
                ):
                    break
                end_idx = probe_idx
                probe_idx += 1

            if end_idx > row_idx:
                groups.append(
                    {
                        "col": col_idx,
                        "col_1based": col_idx + 1,
                        "start_data_row": row_idx + 1,
                        "end_data_row": end_idx + 1,
                        "rowspan": end_idx - row_idx + 1,
                        "text": anchor_text,
                        "source": "sparse_body_rowspan_projection",
                    }
                )
                row_idx = end_idx + 1
                continue
            row_idx += 1

    if not groups:
        return False

    existing = [dict(item) for item in table.get("row_groups", []) or [] if isinstance(item, dict)]
    seen = {
        (
            int(item.get("col", -1) or -1),
            int(item.get("start_data_row", -1) or -1),
            int(item.get("end_data_row", -1) or -1),
            str(item.get("text", "") or ""),
        )
        for item in existing
    }
    changed = False
    for group in groups:
        key = (
            int(group.get("col", -1) or -1),
            int(group.get("start_data_row", -1) or -1),
            int(group.get("end_data_row", -1) or -1),
            str(group.get("text", "") or ""),
        )
        if key in seen:
            continue
        existing.append(group)
        seen.add(key)
        changed = True
    if changed:
        table["row_groups"] = existing
        table["body_rowspan_groups_projected"] = True
    return changed


def _looks_like_body_rowspan_anchor_text(text: str) -> bool:
    cleaned = _semantic_cell_text(text)
    if not cleaned:
        return False
    if _cell_looks_like_placeholder_data_value(cleaned):
        return False
    if _looks_like_pathish_text(cleaned):
        return False
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\([^)]+\))?", cleaned):
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned))


def _count_slashes_outside_grouping(text: str) -> int:
    return sum(1 for idx, char in enumerate(text) if char == "/" and _is_outside_grouping(text, idx))


def _last_slash_outside_grouping(text: str) -> int:
    result = -1
    for idx, char in enumerate(text):
        if char == "/" and _is_outside_grouping(text, idx):
            result = idx
    return result


def _is_outside_grouping(text: str, index: int) -> bool:
    depth = 0
    pairs = {"(": ")", "（": "）", "[": "]", "【": "】"}
    closers = set(pairs.values())
    for pos, char in enumerate(text):
        if pos >= index:
            break
        if char in pairs:
            depth += 1
        elif char in closers and depth > 0:
            depth -= 1
    return depth == 0


def _header_suffix_matches_continuation(prefix: str, continuation: str) -> bool:
    left = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", prefix.lower())
    right = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", continuation.lower())
    if not left or not right:
        return False
    return len(left) <= 16 and len(right) <= 16


def _find_following_header_group_span(
    top_row: list[str | None],
    continuation_row: list[str | None],
    body_rows: list[list[str | None]],
    *,
    start_col: int,
    min_start_col: int | None = None,
    col_count: int,
) -> int | None:
    if start_col >= col_count:
        return None
    group_label_col = start_col
    top_label = _semantic_header_cell_text(top_row[group_label_col], group_label_col)
    if not top_label:
        return None

    previous_top_cols = [
        col_idx
        for col_idx in range(0, group_label_col)
        if _semantic_header_cell_text(top_row[col_idx], col_idx)
    ]
    span_start = (previous_top_cols[-1] + 1) if previous_top_cols else 0
    if min_start_col is not None:
        span_start = max(span_start, int(min_start_col))
    span_cols: list[int] = []
    for col_idx in range(span_start, col_count):
        if col_idx != group_label_col and _semantic_header_cell_text(top_row[col_idx], col_idx):
            break
        continuation_text = _semantic_header_cell_text(continuation_row[col_idx], col_idx)
        if not continuation_text:
            if col_idx == group_label_col:
                continue
            break
        if not _column_has_body_values(body_rows, col_idx, col_count):
            break
        span_cols.append(col_idx)
    if len(span_cols) < 2:
        return None
    if span_cols[0] != span_start:
        return None
    return span_cols[-1]


def _infer_continuation_header_group_span(
    top_row: list[str | None],
    continuation_row: list[str | None],
    body_rows: list[list[str | None]],
    *,
    label_col: int,
    min_start_col: int | None = None,
    col_count: int,
) -> tuple[int, int] | None:
    """Infer a header group span over leaf headers in the continuation row."""
    if label_col < 0 or label_col >= col_count:
        return None
    top_label = _semantic_header_cell_text(top_row[label_col], label_col)
    if not top_label:
        return None
    if _row_has_document_path_like_header_text([top_row[label_col]]):
        return None

    top_label_cols = [
        col_idx
        for col_idx in range(col_count)
        if _semantic_header_cell_text(top_row[col_idx], col_idx)
    ]
    try:
        label_pos = top_label_cols.index(label_col)
    except ValueError:
        return None
    lower_bound = max(0, int(min_start_col or 0))
    if label_pos > 0:
        lower_bound = max(lower_bound, top_label_cols[label_pos - 1] + 1)
    upper_bound = col_count - 1
    fragment_end = _fragmented_group_label_end_col(top_row, label_col, col_count)
    if fragment_end is not None:
        next_label_cols = [col_idx for col_idx in top_label_cols if col_idx > fragment_end]
        if next_label_cols:
            upper_bound = min(upper_bound, next_label_cols[0] - 1)
    elif label_pos + 1 < len(top_label_cols):
        upper_bound = min(upper_bound, top_label_cols[label_pos + 1] - 1)
    if upper_bound < lower_bound:
        return None

    leaf_cols = [
        col_idx
        for col_idx in range(lower_bound, upper_bound + 1)
        if _semantic_header_cell_text(continuation_row[col_idx], col_idx)
        and _column_has_body_values(body_rows, col_idx, col_count)
    ]
    if len(leaf_cols) < 2:
        return None
    if leaf_cols != list(range(leaf_cols[0], leaf_cols[-1] + 1)):
        return None
    if not (leaf_cols[0] <= label_col <= leaf_cols[-1]):
        if label_col != lower_bound and label_col != upper_bound:
            return None
    return leaf_cols[0], leaf_cols[-1]


def _fragmented_group_label_end_col(
    top_row: list[str | None],
    label_col: int,
    col_count: int,
) -> int | None:
    """Return the end col when one group label is split across top-row cells."""
    first = _semantic_header_cell_text(top_row[label_col], label_col)
    if not first:
        return None
    depth = _grouping_balance_delta(first)
    if depth <= 0:
        return None
    end_col = label_col
    for col_idx in range(label_col + 1, col_count):
        text = _semantic_header_cell_text(top_row[col_idx], col_idx)
        if not text:
            continue
        depth += _grouping_balance_delta(text)
        end_col = col_idx
        if depth <= 0:
            return end_col
    return None


def _is_inside_fragmented_group_label(
    top_row: list[str | None],
    col_idx: int,
) -> bool:
    if col_idx <= 0:
        return False
    for start_col in range(0, col_idx):
        end_col = _fragmented_group_label_end_col(top_row, start_col, len(top_row))
        if end_col is not None and start_col < col_idx <= end_col:
            return True
    return False


def _grouping_balance_delta(text: str) -> int:
    pairs = {"(": ")", "\uff08": "\uff09", "[": "]", "\u3010": "\u3011"}
    closers = set(pairs.values())
    delta = 0
    for char in str(text or ""):
        if char in pairs:
            delta += 1
        elif char in closers:
            delta -= 1
    return delta


def _infer_header_group_start_col(
    continuation_row: list[str | None],
    *,
    span_end: int,
    min_start_col: int | None = None,
) -> int:
    lower_bound = max(0, int(min_start_col or 0))
    span_leaf_cols = [
        candidate_col
        for candidate_col in range(lower_bound, span_end + 1)
        if _semantic_header_cell_text(continuation_row[candidate_col], candidate_col)
    ]
    if not span_leaf_cols:
        return span_end
    return span_leaf_cols[0]


def _has_local_header_evidence(
    first_row: list[Any],
    header_texts: list[str],
    *,
    inherited: bool,
) -> bool:
    """Decide whether the current page itself shows repeated-header evidence."""
    normalized_headers = {_compact_text(text) for text in header_texts if _compact_text(text)}
    if not normalized_headers:
        return False

    row_tokens = [
        _compact_text(str(cell))
        for cell in first_row
        if str(cell or "").strip()
    ]
    if not row_tokens:
        return False

    if not inherited:
        return True

    combined_row = _compact_text(" ".join(str(cell).strip() for cell in first_row if str(cell or "").strip()))
    combined_header = _compact_text(" ".join(header_texts))
    if combined_row and combined_row == combined_header:
        return True

    # For continuation pages with inherited headers, only align when every
    # observed non-empty token is itself a header token. This prevents metadata
    # inheritance from overwriting the first local data row.
    return all(token in normalized_headers for token in row_tokens)


def _refresh_row_texts_from_grid(table: dict[str, Any]) -> None:
    raw_grid = _get_authoritative_raw_grid(table)
    semantic_source_grid = _project_semantic_grid_from_cells(raw_grid, table.get("cells") or [], table)
    data_start_row = _get_data_start_row(table)
    external_projection = table.get("external_header_projection")
    external_data_first = isinstance(external_projection, dict) and _raw_grid_is_external_projection_data_first(table, raw_grid)
    if external_data_first:
        data_start_row = 1
    display_grid, data_grid, raw_row_texts, display_row_texts, data_row_texts, structural_empty_rows = _build_row_views_from_raw_grid(
        semantic_source_grid,
        data_start_row=data_start_row,
        raw_audit_grid=raw_grid,
    )
    audit_row_texts = None
    if _should_include_external_title_in_audit_rows(table):
        audit_row_texts = _raw_row_texts_with_external_title(table, raw_grid)
    if audit_row_texts:
        raw_row_texts = audit_row_texts
    if isinstance(external_projection, dict):
        external_audit_rows = _raw_row_texts_with_external_header_projection(table, raw_grid)
        if external_audit_rows:
            raw_row_texts = external_audit_rows

    if external_data_first:
        data_grid = project_table_grid_display_text(_clone_grid_rows(raw_grid[1:]))
        display_grid = project_table_grid_display_text(_clone_grid_rows(raw_grid))
        display_row_texts = _render_row_texts(display_grid)
        data_row_texts = _render_row_texts(data_grid)
        structural_empty_rows = []

    table["raw_grid"] = raw_grid
    table["display_grid"] = display_grid
    table["grid"] = data_grid
    table["data_grid"] = _clone_grid_rows(data_grid)
    table["raw_row_texts"] = raw_row_texts
    table["display_row_texts"] = display_row_texts
    table["row_texts"] = data_row_texts
    table["data_row_texts"] = list(data_row_texts)
    table["raw_row_count"] = len(raw_grid)
    table["display_row_count"] = len(display_grid)
    table["row_count"] = len(data_grid)
    table["data_row_count"] = len(data_grid)
    table["logical_row_count"] = len(data_grid)
    merged_rows = _build_merged_row_metadata(display_grid)
    if merged_rows:
        table["merged_rows"] = merged_rows
    else:
        table.pop("merged_rows", None)
    _attach_display_row_provenance(table)
    if structural_empty_rows:
        table["structural_empty_rows"] = structural_empty_rows
    else:
        table.pop("structural_empty_rows", None)


def _raw_grid_is_external_projection_data_first(
    table: dict[str, Any],
    raw_grid: list[list[str | None]],
) -> bool:
    if len(raw_grid) < 2:
        return False
    header = [
        _semantic_cell_text(cell.get("text") if isinstance(cell, dict) else cell)
        for cell in table.get("header", []) or []
    ]
    if len(header) != len(raw_grid[0]):
        return False
    first_row = [_semantic_cell_text(cell) for cell in raw_grid[0]]
    if not first_row:
        return False
    compatible_header_cells = 0
    observed_header_cells = 0
    for idx, first_text in enumerate(first_row):
        first_compact = _compact_text(first_text)
        if not first_compact:
            continue
        observed_header_cells += 1
        header_compact = _compact_text(header[idx] if idx < len(header) else "")
        if header_compact == first_compact or header_compact.startswith(first_compact) or first_compact in header_compact:
            compatible_header_cells += 1
    if observed_header_cells < 2 or compatible_header_cells < observed_header_cells:
        return False
    second_row = [_semantic_cell_text(cell) for cell in raw_grid[1]]
    value_like = sum(
        1
        for text in second_row
        if _cell_value_profile_kind(text) in {"numeric", "statistical", "code"} or _looks_like_plain_numeric_value(text)
    )
    return value_like >= max(2, len([text for text in second_row if text]) // 2)


def _looks_like_plain_numeric_value(text: str) -> bool:
    cleaned = _semantic_cell_text(text)
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:%|[eE][-+]?\d+)?", cleaned))


def _raw_row_texts_with_external_header_projection(
    table: dict[str, Any],
    raw_grid: list[list[str | None]],
) -> list[str] | None:
    header = [
        _semantic_cell_text(cell.get("text") if isinstance(cell, dict) else cell)
        for cell in table.get("header", []) or []
    ]
    if not header or not raw_grid:
        return None
    title_rows = _external_title_rows_for_audit(table, max(len(header), len(raw_grid[0])))
    base_header = _external_projection_base_header(table, raw_grid, len(header))
    continuation_row = _external_projection_continuation_row(base_header, header)
    if not any(_semantic_cell_text(cell) for cell in continuation_row):
        return None
    data_start = _get_data_start_row(table)
    if data_start > 0 and data_start < len(raw_grid):
        data_rows = _clone_grid_rows(raw_grid[data_start:])
    elif len(raw_grid) >= 2 and _compact_text(" ".join(_semantic_cell_text(cell) for cell in raw_grid[0])) == _compact_text(" ".join(base_header)):
        data_rows = _clone_grid_rows(raw_grid[1:])
    else:
        data_rows = _clone_grid_rows(raw_grid)
    audit_rows = title_rows + [base_header, continuation_row] + data_rows
    return _render_row_texts(audit_rows)


def _external_title_rows_for_audit(table: dict[str, Any], col_count: int) -> list[list[str | None]]:
    title = _semantic_cell_text(table.get("title"))
    if not title:
        return []
    title_bbox = table.get("title_bbox") or (table.get("title_block") or {}).get("bbox") or []
    if isinstance(title_bbox, (list, tuple)) and len(title_bbox) == 4:
        title_height = float(title_bbox[3]) - float(title_bbox[1])
        if title_height > 17.0 and len(title.split()) >= 8:
            first, second = _split_external_title_for_audit(title)
            first_row = [None] * max(1, col_count)
            second_row = [None] * max(1, col_count)
            first_row[0] = first
            second_row[0] = second
            return [first_row, second_row]
    row = [None] * max(1, col_count)
    row[0] = title
    return [row]


def _split_external_title_for_audit(title: str) -> tuple[str, str]:
    words = str(title or "").split()
    if len(words) < 2:
        return title, ""
    midpoint = len(title) / 2.0
    running = 0
    best_idx = 1
    best_distance = float("inf")
    for idx, word in enumerate(words[:-1], start=1):
        running += len(word) + (1 if idx > 1 else 0)
        distance = abs(running - midpoint)
        if distance < best_distance:
            best_idx = idx
            best_distance = distance
    return " ".join(words[:best_idx]).strip(), " ".join(words[best_idx:]).strip()


def _external_projection_base_header(
    table: dict[str, Any],
    raw_grid: list[list[str | None]],
    col_count: int,
) -> list[str | None]:
    if raw_grid and len(raw_grid[0]) == col_count:
        first_row = [_semantic_cell_text(cell) for cell in raw_grid[0]]
        header = [
            _semantic_cell_text(cell.get("text") if isinstance(cell, dict) else cell)
            for cell in table.get("header", []) or []
        ]
        compatible = True
        for idx, first_text in enumerate(first_row):
            first_compact = _compact_text(first_text)
            header_compact = _compact_text(header[idx] if idx < len(header) else "")
            if first_compact and not (
                header_compact == first_compact
                or header_compact.startswith(first_compact)
                or first_compact in header_compact
            ):
                compatible = False
                break
        if compatible:
            return list(raw_grid[0])
    return [
        _semantic_cell_text(cell.get("text") if isinstance(cell, dict) else cell)
        for cell in table.get("header", []) or []
    ]


def _external_projection_continuation_row(
    base_header: list[str | None],
    projected_header: list[str],
) -> list[str | None]:
    continuation: list[str | None] = [None] * max(len(base_header), len(projected_header))
    for idx, projected in enumerate(projected_header):
        if idx >= len(base_header):
            continue
        base = _semantic_cell_text(base_header[idx])
        expanded = _semantic_cell_text(projected)
        if not base or expanded == base:
            continue
        if not expanded.lower().startswith(base.lower()):
            continue
        tail = expanded[len(base) :].strip()
        if tail:
            continuation[idx] = tail
    return continuation


def project_simple_schema_header_data_views(table: dict[str, Any]) -> None:
    """Keep simple one-line schema tables' public row views data-first."""
    raw_grid = _get_authoritative_raw_grid(table)
    if len(raw_grid) < 2:
        return
    if _get_data_start_row(table) != 1:
        return
    if not _single_schema_header_followed_by_clear_data(raw_grid[0], raw_grid[1]):
        return
    if _looks_like_projected_multiline_header_continuation(table, raw_grid):
        return
    data_rows = _clone_grid_rows(raw_grid[1:])
    display_rows = project_table_grid_display_text(data_rows)
    audit_raw_row_texts = _raw_row_texts_with_external_title(table, raw_grid) or _render_row_texts(data_rows)
    table["raw_grid"] = data_rows
    table["display_grid"] = display_rows
    table["grid"] = _clone_grid_rows(display_rows)
    table["data_grid"] = _clone_grid_rows(display_rows)
    table["raw_row_texts"] = audit_raw_row_texts
    table["display_row_texts"] = _render_row_texts(display_rows)
    table["row_texts"] = _render_row_texts(display_rows)
    table["data_row_texts"] = list(table["row_texts"])
    table["raw_row_count"] = len(table["raw_row_texts"])
    table["display_row_count"] = len(display_rows)
    table["row_count"] = len(display_rows)
    table["data_row_count"] = len(display_rows)
    table["logical_row_count"] = len(display_rows)
    _attach_display_row_provenance(table)


def _raw_row_texts_with_external_title(
    table: dict[str, Any],
    raw_grid: list[list[str | None]],
) -> list[str] | None:
    title = _semantic_cell_text(table.get("title"))
    if not title or not raw_grid:
        return None
    title_block = table.get("title_block") or {}
    if str(title_block.get("source", "") or "") not in {"text-layer", "split_title_merge"}:
        return None
    compact_title = _compact_text(title)
    first_row_text = _semantic_cell_text(raw_grid[0][0] if raw_grid and raw_grid[0] else "")
    if first_row_text and compact_title and _compact_text(first_row_text).startswith(compact_title[: min(12, len(compact_title))]):
        return None
    title_row = [None] * max(1, len(raw_grid[0]))
    title_row[0] = title
    return _render_row_texts([title_row] + _clone_grid_rows(raw_grid))


def _should_include_external_title_in_audit_rows(table: dict[str, Any]) -> bool:
    if table.get("external_header_projection"):
        return True
    return str(table.get("detection_source") or table.get("detection_method") or "").strip() in {
        "caption_anchored_horizontal_rules",
        "vector_ocr",
    }


def _looks_like_projected_multiline_header_continuation(
    table: dict[str, Any],
    raw_grid: list[list[str | None]],
) -> bool:
    data_start_row = int(table.get("data_start_row", 0) or 0)
    if data_start_row <= 1 or len(raw_grid) <= data_start_row:
        return False
    first_public_row = raw_grid[0]
    for row in raw_grid[1:data_start_row]:
        non_empty = _semantic_non_empty_columns(row)
        if not non_empty:
            continue
        if len(non_empty) > max(1, len(first_public_row) // 2):
            continue
        if any(_cell_value_profile_kind(_semantic_cell_text(row[col_idx])) in {"numeric", "statistical"} for col_idx in non_empty):
            continue
        if any(_semantic_cell_text(first_public_row[col_idx]) for col_idx in non_empty if col_idx < len(first_public_row)):
            return True
    return False


def _project_semantic_grid_from_cells(
    raw_grid: list[list[str | None]],
    cells: list[dict[str, Any]],
    table: dict[str, Any],
) -> list[list[str | None]]:
    projected = _clone_grid_rows(raw_grid)
    if not projected or not cells:
        return projected
    data_start_row = _get_data_start_row(table)
    for cell in cells:
        if not str(cell.get("inference_reason") or "").startswith("filename_path_text_layer_reconstruction"):
            continue
        text = cell.get("text")
        if text is None:
            continue
        data_row = int(cell.get("logical_row", cell.get("row", 0)) or 0)
        col_idx = int(cell.get("logical_col", cell.get("col", 1)) or 1) - 1
        row_idx = data_start_row + data_row - 1
        if row_idx < 0 or row_idx >= len(projected) or col_idx < 0:
            continue
        while col_idx >= len(projected[row_idx]):
            projected[row_idx].append(None)
        projected[row_idx][col_idx] = str(text)
    return projected


def split_internal_table_segments(table_asts: list[dict[str, Any]]) -> int:
    """Split a physical table when an internal structural break starts a new table.

    The split is schema-driven: a fully empty logical row followed by a compact
    title-like row and a different row pattern is treated as a new table segment.
    This avoids folding explanatory legend tables into a preceding continuation
    table while preserving ordinary continuation rows.
    """
    if not table_asts:
        return 0

    split_count = 0
    result: list[dict[str, Any]] = []
    for table in table_asts:
        split_index = _find_internal_segment_split_index(table)
        if split_index is None:
            result.append(table)
            continue

        raw_grid = _get_authoritative_raw_grid(table)
        leading_grid = raw_grid[:split_index]
        trailing_grid = raw_grid[split_index + 1 :]
        if not leading_grid or not trailing_grid:
            result.append(table)
            continue

        leading = dict(table)
        trailing = dict(table)
        original_table_id = str(table.get("table_id", "") or "")
        if original_table_id:
            trailing["table_id"] = f"{original_table_id}__seg2"
        _rewrite_table_segment(
            leading,
            raw_grid=leading_grid,
            inherited_from=table,
            segment_kind="leading",
        )
        _rewrite_table_segment(
            trailing,
            raw_grid=trailing_grid,
            inherited_from=table,
            segment_kind="trailing",
        )
        result.extend([leading, trailing])
        split_count += 1

    if split_count:
        table_asts[:] = result
    return split_count


def _find_internal_segment_split_index(table: dict[str, Any]) -> int | None:
    raw_grid = _get_authoritative_raw_grid(table)
    if len(raw_grid) < 5:
        return None

    empty_rows = table.get("structural_empty_rows") or _collect_empty_row_indices(raw_grid)
    if not empty_rows:
        return None

    data_start_row = _get_data_start_row(table)
    for row_number in empty_rows:
        split_index = int(row_number) - 1
        if split_index <= data_start_row or split_index >= len(raw_grid) - 2:
            continue
        leading_rows = raw_grid[data_start_row:split_index]
        trailing_rows = raw_grid[split_index + 1 :]
        if _looks_like_internal_segment_boundary(leading_rows, trailing_rows):
            return split_index
    return None


def _collect_empty_row_indices(grid: list[list[str | None]]) -> list[int]:
    return [
        idx
        for idx, row in enumerate(grid, start=1)
        if not _row_has_semantic_content(row)
    ]


def _looks_like_internal_segment_boundary(
    leading_rows: list[list[str | None]],
    trailing_rows: list[list[str | None]],
) -> bool:
    if not leading_rows or len(trailing_rows) < 2:
        return False

    leading_rule_rows = [
        row for row in leading_rows if row and _looks_like_dotted_outline_identifier(_semantic_cell_text(row[0]))
    ]
    if len(leading_rule_rows) < 2:
        return False

    first_trailing = trailing_rows[0]
    first_cols = _semantic_non_empty_columns(first_trailing)
    if first_cols != [0]:
        return False
    first_text = _semantic_cell_text(first_trailing[0])
    if not _looks_like_segment_title_row(first_text):
        return False

    detail_rows = [row for row in trailing_rows[1:] if _row_has_semantic_content(row)]
    if not detail_rows:
        return False
    if any(row and _looks_like_dotted_outline_identifier(_semantic_cell_text(row[0])) for row in detail_rows):
        return False

    sparse_schema_rows = 0
    for row in detail_rows:
        cols = _semantic_non_empty_columns(row)
        if not cols:
            continue
        if 0 in cols and len(cols) <= 3 and (len(row) < 4 or not _semantic_cell_text(row[3])):
            sparse_schema_rows += 1
    return sparse_schema_rows >= max(1, len(detail_rows) // 2)


def _looks_like_segment_title_row(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    if _looks_like_dotted_outline_identifier(cleaned):
        return False
    if len(cleaned) > 40:
        return False
    return cleaned.endswith((':', '：')) or (
        len(re.sub(r"\s+", "", cleaned)) <= 12
        and not any(mark in cleaned for mark in "。；;!?！？")
    )


def _looks_like_dotted_outline_identifier(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)+", str(text or "").strip()))


def _rewrite_table_segment(
    table: dict[str, Any],
    *,
    raw_grid: list[list[str | None]],
    inherited_from: dict[str, Any],
    segment_kind: str,
) -> None:
    table["raw_grid"] = _clone_grid_rows(raw_grid)
    table["cells"] = _build_cells_from_grid(raw_grid)
    table["col_count"] = max((len(row) for row in raw_grid), default=int(table.get("col_count", 0) or 0))
    table["logical_col_count"] = table["col_count"]
    table["physical_row_count"] = len(raw_grid)
    table["raw_row_count"] = len(raw_grid)
    table["column_signature"] = _build_column_signature(table["col_count"])
    table["column_hash"] = _compute_column_hash(table["column_signature"])
    table["internal_segment_source"] = {
        "source_table_id": str(inherited_from.get("table_id", "")),
        "segment_kind": segment_kind,
    }
    if segment_kind == "trailing":
        for key in (
            "continued_from",
            "continued_to",
            "continuation_assessment",
            "continuation_source",
            "cross_page_similarity",
            "cross_page_overlap",
            "header_inherited",
        ):
            table.pop(key, None)
        table["is_continuation"] = False
        table["data_start_row"] = 0
        table["header"] = []
        table["title"] = _semantic_cell_text(raw_grid[0][0]) if raw_grid and raw_grid[0] else ""
        table["local_context_signal"] = "internal_segment"
    _refresh_row_texts_from_grid(table)


def _semantic_cell_text(cell: Any) -> str:
    if cell is None:
        return ""
    return str(cell).strip()


def _semantic_non_empty_columns(row: list[Any]) -> list[int]:
    return [idx for idx, cell in enumerate(row) if _semantic_cell_text(cell)]


def _ends_with_strong_terminal(text: str) -> bool:
    normalized = text.rstrip()
    while normalized and normalized[-1] in _TRAILING_CLOSER_CHARS:
        normalized = normalized[:-1].rstrip()
    return bool(normalized) and normalized[-1] in _STRONG_TERMINAL_PUNCTUATION


def _needs_boundary_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    last_char = left[-1]
    first_char = right[0]
    if last_char.isspace() or first_char.isspace():
        return False
    if _CJK_CHAR_RE.search(last_char) or _CJK_CHAR_RE.search(first_char):
        return False
    return bool(_ASCII_WORD_CHAR_RE.match(last_char) and _ASCII_WORD_CHAR_RE.match(first_char))


def _is_token_like_fragment(text: str) -> bool:
    normalized = text.strip()
    if not normalized or "\n" in normalized:
        return False
    if any(ch.isspace() for ch in normalized):
        return False
    if any(ch in normalized for ch in ",，.。!?！？;；:：()（）[]【】"):
        return False
    return True


def _should_insert_boundary_newline(left: str, right: str) -> bool:
    if "\n" not in right:
        return False
    if "\n" in left:
        return False
    right_lines = [line.strip() for line in right.splitlines() if line.strip()]
    if len(right_lines) < 2:
        return False
    if not _is_token_like_fragment(left):
        return False
    return all(_is_token_like_fragment(line) for line in right_lines)


def _starts_vector_ocr_boundary_segment(text: str) -> bool:
    if not text:
        return False
    first_char = text[0]
    if first_char in _VECTOR_OCR_OPENING_BRACKETS:
        return True
    return bool(_CJK_CHAR_RE.search(first_char) or re.match(r"[A-Za-z]", first_char))


def _needs_vector_ocr_boundary_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    last_char = left[-1]
    first_char = right[0]
    if last_char.isspace() or first_char.isspace():
        return False
    if last_char in _VECTOR_OCR_CLOSING_BRACKETS and first_char in _VECTOR_OCR_OPENING_BRACKETS:
        return True
    if last_char in _VECTOR_OCR_SENTENCE_BOUNDARY_CHARS and _starts_vector_ocr_boundary_segment(right):
        return True
    return False


def _join_boundary_cell_text(left: str, right: str) -> str:
    prefix = left.rstrip()
    suffix = right.lstrip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    if _needs_boundary_space(prefix, suffix):
        return f"{prefix} {suffix}"
    if _should_insert_boundary_newline(prefix, suffix):
        return f"{prefix}\n{suffix}"
    return f"{prefix}{suffix}"


def _is_sparse_anchor_label_text(text: str) -> bool:
    normalized = _semantic_cell_text(text)
    if not normalized:
        return False
    if len(normalized) > 18:
        return False
    if any(mark in normalized for mark in ",，;；:：.!?！？"):
        return False
    return len(normalized.split()) <= 3


def _join_vector_ocr_group_text(left: str, right: str) -> str:
    prefix = left.rstrip()
    suffix = right.lstrip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    if prefix.endswith("-") and suffix[:1].islower():
        return f"{prefix[:-1]}{suffix}"
    if _needs_vector_ocr_boundary_space(prefix, suffix):
        return f"{prefix} {suffix}"
    return _join_boundary_cell_text(prefix, suffix)


def _merge_vector_ocr_group_cells(values: list[str | None]) -> str | None:
    merged = ""
    for value in values:
        text = _semantic_cell_text(value)
        if not text:
            continue
        merged = text if not merged else _join_vector_ocr_group_text(merged, text)
    return merged or None


def _group_rows_by_sparse_anchor_midpoints(
    row_count: int,
    anchor_rows: list[int],
) -> list[list[int]]:
    if row_count <= 0 or not anchor_rows:
        return []

    boundaries: list[int] = [0]
    for idx in range(len(anchor_rows) - 1):
        midpoint = (anchor_rows[idx] + anchor_rows[idx + 1] + 1) // 2
        boundaries.append(midpoint)
    boundaries.append(row_count)

    groups: list[list[int]] = []
    for idx in range(len(anchor_rows)):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        row_indices = list(range(start, end))
        if row_indices:
            groups.append(row_indices)
    return groups


def compact_vector_ocr_sparse_anchor_rows(table: dict[str, Any]) -> bool:
    if str(table.get("detection_method", "") or "") != "vector_ocr":
        return False

    data_grid = _clone_grid_rows(table.get("data_grid") or table.get("grid") or [])
    col_count = int(table.get("col_count", 0) or 0)
    if len(data_grid) < 6 or col_count < 3:
        return False

    first_col_values = [_semantic_cell_text(row[0] if row else None) for row in data_grid]
    anchor_rows = [idx for idx, text in enumerate(first_col_values) if text]
    if len(anchor_rows) < 2 or len(anchor_rows) > max(4, len(data_grid) // 2):
        return False
    if not all(_is_sparse_anchor_label_text(first_col_values[idx]) for idx in anchor_rows):
        return False

    continuation_like_rows = sum(
        1
        for row in data_grid
        if not _semantic_cell_text(row[0] if row else None)
        and any(_semantic_cell_text(cell) for cell in row[1:])
    )
    if continuation_like_rows < max(2, len(data_grid) // 3):
        return False

    grouped_row_indices = _group_rows_by_sparse_anchor_midpoints(len(data_grid), anchor_rows)
    if len(grouped_row_indices) != len(anchor_rows):
        return False
    if not any(len(group) > 1 for group in grouped_row_indices):
        return False

    semantic_grid: list[list[str | None]] = []
    semantic_groups: list[dict[str, Any]] = []
    for anchor_idx, row_indices in zip(anchor_rows, grouped_row_indices):
        anchor_text = first_col_values[anchor_idx]
        if not anchor_text:
            return False

        merged_row: list[str | None] = [None] * col_count
        merged_row[0] = anchor_text
        for col_idx in range(1, col_count):
            merged_row[col_idx] = _merge_vector_ocr_group_cells(
                [data_grid[row_idx][col_idx] if col_idx < len(data_grid[row_idx]) else None for row_idx in row_indices]
            )

        if not any(_semantic_cell_text(cell) for cell in merged_row[1:]):
            return False

        semantic_grid.append(merged_row)
        semantic_groups.append(
            {
                "anchor_text": anchor_text,
                "anchor_data_row": anchor_idx + 1,
                "source_data_row_start": row_indices[0] + 1,
                "source_data_row_end": row_indices[-1] + 1,
                "source_data_row_count": len(row_indices),
            }
        )

    if len(semantic_grid) >= len(data_grid):
        return False

    _write_semantic_grid(table, semantic_grid)
    table["semantic_compaction"] = {
        "applied": True,
        "strategy": "vector_ocr_sparse_anchor_rows",
        "source_data_row_count": len(data_grid),
        "compacted_data_row_count": len(semantic_grid),
        "anchor_column": 1,
        "groups": semantic_groups,
    }
    return True


def _extract_anchor_prefix_values(
    row: list[Any],
    prefix_length: int,
) -> list[str]:
    if prefix_length <= 0:
        return []

    values: list[str] = []
    for col_idx in range(min(prefix_length, len(row))):
        text = _semantic_cell_text(row[col_idx])
        if not text:
            break
        values.append(text)
    return values


def _resolve_continuation_parent_anchor(
    table: dict[str, Any],
    table_lookup: dict[str, dict[str, Any]] | None,
    prefix_length: int,
) -> dict[str, Any] | None:
    if not table_lookup or prefix_length <= 0:
        return None

    parent_id = str(table.get("continued_from", "") or "").strip()
    visited: set[str] = set()
    while parent_id and parent_id not in visited:
        visited.add(parent_id)
        parent_table = table_lookup.get(parent_id)
        if not parent_table:
            break

        parent_grid = _clone_grid_rows(parent_table.get("data_grid") or parent_table.get("grid") or [])
        for row_idx in range(len(parent_grid) - 1, -1, -1):
            prefix_values = _extract_anchor_prefix_values(parent_grid[row_idx], prefix_length)
            if prefix_values:
                return {
                    "table_id": parent_id,
                    "data_row": row_idx + 1,
                    "values": prefix_values,
                }

        parent_id = str(parent_table.get("continued_from", "") or "").strip()

    return None


def compact_leading_key_carry_forward_rows(
    table: dict[str, Any],
    table_lookup: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Carry forward stable leading group keys in hierarchical sparse rows."""
    if table.get("semantic_compaction"):
        return False

    data_grid = _clone_grid_rows(table.get("data_grid") or table.get("grid") or [])
    col_count = int(table.get("col_count", 0) or 0)
    if len(data_grid) < 3 or col_count < 3:
        return False
    original_data_grid = _clone_grid_rows(data_grid)
    data_grid = _compact_sparse_wrapped_content_rows_for_carry_forward(data_grid, col_count)

    header = table.get("header") or []
    header_texts = [str(item.get("text", "")).strip() for item in header]
    if len(header_texts) < 2:
        return False
    if any(not text or _is_generic_column_header(text) for text in header_texts[:2]):
        return False

    max_leading_empty_count = 0
    for row in data_grid:
        non_empty_columns = _semantic_non_empty_columns(row)
        if non_empty_columns:
            max_leading_empty_count = max(max_leading_empty_count, non_empty_columns[0])
    if max_leading_empty_count <= 0:
        return False

    parent_anchor = _resolve_continuation_parent_anchor(table, table_lookup, max_leading_empty_count)

    anchor_rows: list[int] = []
    candidate_rows: list[dict[str, Any]] = []
    current_anchor_idx: int | None = None
    for row_idx, row in enumerate(data_grid):
        non_empty_columns = _semantic_non_empty_columns(row)
        if not non_empty_columns:
            continue

        leading_empty_count = non_empty_columns[0]
        if leading_empty_count == 0:
            current_anchor_idx = row_idx
            anchor_rows.append(row_idx)
            continue

        if current_anchor_idx is None:
            anchor_values = []
            if parent_anchor:
                anchor_values = parent_anchor.get("values", [])[:leading_empty_count]
            if len(anchor_values) < leading_empty_count:
                continue
            anchor_scope = "continuation_parent"
            anchor_group_id = f"parent:{parent_anchor['table_id']}:{parent_anchor['data_row']}"
            anchor_metadata = {
                "anchor_scope": anchor_scope,
                "anchor_table_id": parent_anchor["table_id"],
                "anchor_data_row": parent_anchor["data_row"],
                "anchor_values": list(anchor_values),
                "anchor_group_id": anchor_group_id,
            }
        else:
            anchor_row = data_grid[current_anchor_idx]
            anchor_values = _extract_anchor_prefix_values(anchor_row, leading_empty_count)
            if len(anchor_values) < leading_empty_count:
                continue
            anchor_metadata = {
                "anchor_scope": "local",
                "anchor_data_row": current_anchor_idx + 1,
                "anchor_values": list(anchor_values),
                "anchor_group_id": f"local:{current_anchor_idx + 1}",
            }

        if not any(_semantic_cell_text(row[col_idx]) for col_idx in range(leading_empty_count, col_count)):
            continue

        candidate_rows.append(
            {
                "row_idx": row_idx,
                "leading_empty_count": leading_empty_count,
                **anchor_metadata,
            }
        )

    if not anchor_rows and not parent_anchor:
        return False
    if len(candidate_rows) < max(2, len(data_grid) // 6):
        return False

    anchor_group_count = len({item["anchor_group_id"] for item in candidate_rows})
    parent_only_continuation = (
        bool(parent_anchor)
        and not anchor_rows
        and anchor_group_count == 1
        and all(item["anchor_scope"] == "continuation_parent" for item in candidate_rows)
        and len(candidate_rows) >= 2
    )
    if anchor_group_count < 2 and not parent_only_continuation:
        return False

    semantic_grid = _clone_grid_rows(data_grid)
    semantic_groups: list[dict[str, Any]] = []
    changed = False
    for candidate in candidate_rows:
        row_idx = int(candidate["row_idx"])
        leading_empty_count = int(candidate["leading_empty_count"])
        carried_columns: list[int] = []
        anchor_values = list(candidate.get("anchor_values") or [])
        for col_idx in range(min(leading_empty_count, len(anchor_values))):
            if _semantic_cell_text(semantic_grid[row_idx][col_idx]):
                continue
            anchor_text = _semantic_cell_text(anchor_values[col_idx])
            if not anchor_text:
                break
            semantic_grid[row_idx][col_idx] = anchor_text
            carried_columns.append(col_idx + 1)
            changed = True

        if carried_columns:
            group = {
                "source_data_row": row_idx + 1,
                "carried_columns": carried_columns,
            }
            if candidate.get("anchor_scope") == "continuation_parent":
                group["anchor_scope"] = "continuation_parent"
                group["anchor_table_id"] = candidate.get("anchor_table_id")
                group["anchor_data_row"] = candidate.get("anchor_data_row")
            else:
                group["anchor_data_row"] = candidate.get("anchor_data_row")
            semantic_groups.append(group)

    if not changed:
        return False

    _write_semantic_grid(table, semantic_grid)
    if data_grid != original_data_grid:
        _sync_display_grid_after_sparse_wrapped_content_compaction(
            table,
            original_data_grid=original_data_grid,
            compacted_data_grid=data_grid,
        )
    table["semantic_compaction"] = {
        "applied": True,
        "strategy": "leading_key_carry_forward",
        "source_data_row_count": len(data_grid),
        "compacted_data_row_count": len(semantic_grid),
        "groups": semantic_groups,
    }
    return True


def _compact_sparse_wrapped_content_rows_for_carry_forward(
    grid: list[list[str | None]],
    col_count: int,
) -> list[list[str | None]]:
    """Merge wrapped content fragments before leading-key carry-forward."""
    if len(grid) < 3 or col_count < 3:
        return grid
    projected: list[list[str | None]] = []
    row_idx = 0
    while row_idx < len(grid):
        row = list(grid[row_idx])
        content_col = _find_sparse_wrapped_cell_anchor_column(row)
        if content_col is None:
            projected.append(row)
            row_idx += 1
            continue
        fragments: list[str] = []
        lookahead_idx = row_idx + 1
        while lookahead_idx < len(grid) and _is_sparse_wrapped_cell_fragment_row(
            anchor_row=row,
            fragment_row=grid[lookahead_idx],
            content_col=content_col,
        ):
            fragments.append(_semantic_cell_text(grid[lookahead_idx][content_col]))
            lookahead_idx += 1
        if not fragments:
            projected.append(row)
            row_idx += 1
            continue
        merged_text = _join_sparse_wrapped_cell_fragments([_semantic_cell_text(row[content_col]), *fragments])
        if merged_text:
            row[content_col] = merged_text
        projected.append(row)
        row_idx = lookahead_idx
    return projected


def _sync_display_grid_after_sparse_wrapped_content_compaction(
    table: dict[str, Any],
    *,
    original_data_grid: list[list[str | None]],
    compacted_data_grid: list[list[str | None]],
) -> None:
    if not original_data_grid or original_data_grid == compacted_data_grid:
        return
    display_grid = _clone_grid_rows(table.get("display_grid") or [])
    data_start_row = _get_data_start_row(table)
    if data_start_row < 0 or data_start_row > len(display_grid):
        return
    if len(display_grid) - data_start_row != len(original_data_grid):
        return

    compacted_display = project_table_grid_display_text(_clone_grid_rows(compacted_data_grid))
    table["display_grid"] = display_grid[:data_start_row] + compacted_display
    table["display_row_texts"] = _render_row_texts(table["display_grid"])
    table["display_row_count"] = len(table["display_grid"])
    table["grid"] = _clone_grid_rows(table.get("data_grid") or table.get("grid") or [])
    _attach_display_row_provenance(table)


def _write_semantic_grid(table: dict[str, Any], semantic_grid: list[list[str | None]]) -> None:
    normalized_grid = project_table_grid_display_text(_clone_grid_rows(semantic_grid))
    table["grid"] = normalized_grid
    table["data_grid"] = _clone_grid_rows(normalized_grid)
    data_row_texts = _render_row_texts(normalized_grid)
    table["row_texts"] = data_row_texts
    table["data_row_texts"] = list(data_row_texts)
    table["row_count"] = len(normalized_grid)
    table["data_row_count"] = len(normalized_grid)
    table["logical_row_count"] = len(normalized_grid)
    table["cells"] = _build_cells_from_grid(normalized_grid)


def _write_semantic_projection(
    table: dict[str, Any],
    *,
    header_texts: list[str],
    semantic_grid: list[list[str | None]],
) -> None:
    normalized_header = [
        text.strip() if isinstance(text, str) and text.strip() else f"Column {idx + 1}"
        for idx, text in enumerate(header_texts)
    ]
    normalized_header = project_table_grid_display_text([normalized_header])[0]
    normalized_grid = project_table_grid_display_text(_clone_grid_rows(semantic_grid))
    display_grid = [list(normalized_header)] + _clone_grid_rows(normalized_grid)
    row_texts = _render_row_texts(normalized_grid)

    table["header"] = [{"col": idx + 1, "text": text} for idx, text in enumerate(normalized_header)]
    table["col_count"] = len(normalized_header)
    table["grid"] = normalized_grid
    table["data_grid"] = _clone_grid_rows(normalized_grid)
    table["display_grid"] = display_grid
    table["display_row_texts"] = _render_row_texts(display_grid)
    table["display_row_count"] = len(display_grid)
    table["row_texts"] = row_texts
    table["data_row_texts"] = list(row_texts)
    table["row_count"] = len(normalized_grid)
    table["data_row_count"] = len(normalized_grid)
    table["logical_row_count"] = len(normalized_grid)
    table["cells"] = _build_cells_from_grid(normalized_grid)
    table["header_row_index"] = 0
    table["data_start_row"] = 1
    signature = _build_column_signature(len(normalized_header))
    table["column_signature"] = signature
    table["column_hash"] = _compute_column_hash(signature)
    _attach_display_row_provenance(table)


def _is_generic_column_header(text: str) -> bool:
    return bool(_GENERIC_COLUMN_HEADER_RE.fullmatch(text.strip()))


def _merge_vector_ocr_multiline_values(values: list[str | None]) -> str | None:
    lines: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _semantic_cell_text(value)
        if not text:
            continue
        normalized = _compact_text(text)
        if normalized and normalized in seen:
            continue
        if normalized:
            seen.add(normalized)
        lines.append(text)
    if not lines:
        return None
    return "\n".join(lines)


def compact_vector_ocr_placeholder_tail_column_pairs(table: dict[str, Any]) -> bool:
    if str(table.get("detection_method", "") or "") != "vector_ocr":
        return False

    raw_grid = _get_authoritative_raw_grid(table)
    if _compact_two_column_vector_ocr_wrapped_values(table, raw_grid):
        return True
    if len(raw_grid) < 5:
        return False

    col_count = int(table.get("col_count", 0) or 0)
    physical_col_count = int(table.get("physical_col_count", 0) or 0)
    if col_count != 3 or physical_col_count < 5:
        return False

    header = table.get("header") or []
    header_texts = [str(item.get("text", "")).strip() for item in header]
    if len(header_texts) != 3:
        return False
    if not header_texts[0] or not header_texts[1] or not _is_generic_column_header(header_texts[2]):
        return False

    data_start_row = _get_data_start_row(table)
    if data_start_row <= 0 or data_start_row >= len(raw_grid):
        return False

    data_rows = [list(row[:3]) + [None] * max(0, 3 - len(row)) for row in raw_grid[data_start_row:]]
    if len(data_rows) < 4:
        return False

    anchor_rows = [idx for idx, row in enumerate(data_rows) if _semantic_cell_text(row[0])]
    if len(anchor_rows) < 2:
        return False

    first_anchor_idx = anchor_rows[0]
    if first_anchor_idx <= 0:
        return False

    header_continuation_rows = data_rows[:first_anchor_idx]
    if not header_continuation_rows:
        return False
    for row in header_continuation_rows:
        if _semantic_cell_text(row[0]) or _semantic_cell_text(row[2]) or not _semantic_cell_text(row[1]):
            return False

    tail_fragment_rows: list[int] = []
    for idx, row in enumerate(data_rows):
        tail_text = _semantic_cell_text(row[2])
        if not tail_text:
            continue
        if _semantic_cell_text(row[0]) or not _semantic_cell_text(row[1]):
            return False
        tail_fragment_rows.append(idx)
    if not tail_fragment_rows:
        return False

    semantic_grid: list[list[str | None]] = []
    semantic_groups: list[dict[str, Any]] = []
    remaining_rows = data_rows[first_anchor_idx:]
    local_anchor_rows = [idx - first_anchor_idx for idx in anchor_rows]

    for anchor_pos, local_anchor_idx in enumerate(local_anchor_rows):
        anchor_row = remaining_rows[local_anchor_idx]
        anchor_text = _semantic_cell_text(anchor_row[0])
        if not anchor_text:
            return False
        next_local_anchor = (
            local_anchor_rows[anchor_pos + 1]
            if anchor_pos + 1 < len(local_anchor_rows)
            else len(remaining_rows)
        )
        group_rows = remaining_rows[local_anchor_idx:next_local_anchor]
        value_lines: list[str | None] = []
        for row_offset, row in enumerate(group_rows):
            if row_offset > 0 and _semantic_cell_text(row[0]):
                return False
            merged_row_value = _merge_vector_ocr_group_cells([row[1], row[2]])
            if merged_row_value:
                value_lines.append(merged_row_value)
        merged_value = _merge_vector_ocr_multiline_values(value_lines)
        if not merged_value:
            return False
        semantic_grid.append([anchor_text, merged_value])
        semantic_groups.append(
            {
                "anchor_text": anchor_text,
                "source_data_row_start": first_anchor_idx + local_anchor_idx + 1,
                "source_data_row_end": first_anchor_idx + next_local_anchor,
                "source_data_row_count": len(group_rows),
            }
        )

    if len(semantic_grid) >= len(data_rows):
        return False

    merged_header_value = _merge_vector_ocr_multiline_values(
        [header_texts[1]]
        + [_merge_vector_ocr_group_cells([row[1], row[2]]) for row in header_continuation_rows]
    )
    if not merged_header_value:
        return False

    _write_semantic_projection(
        table,
        header_texts=[header_texts[0], merged_header_value],
        semantic_grid=semantic_grid,
    )
    table["semantic_compaction"] = {
        "applied": True,
        "strategy": "vector_ocr_placeholder_tail_column_pairs",
        "source_data_row_count": len(data_rows),
        "compacted_data_row_count": len(semantic_grid),
        "header_continuation_row_count": len(header_continuation_rows),
        "groups": semantic_groups,
    }
    return True


def _compact_two_column_vector_ocr_wrapped_values(
    table: dict[str, Any],
    raw_grid: list[list[Any]],
) -> bool:
    col_count = int(table.get("col_count", 0) or 0)
    physical_col_count = int(table.get("physical_col_count", 0) or 0)
    if col_count != 2 or physical_col_count != 2 or len(raw_grid) < 4:
        return False
    header = table.get("header") or []
    header_texts = [str(item.get("text", "")).strip() for item in header]
    if len(header_texts) != 2 or not header_texts[0] or not header_texts[1]:
        return False

    data_start_row = _get_data_start_row(table)
    if data_start_row <= 0:
        return False
    data_rows = [list(row[:2]) + [None] * max(0, 2 - len(row)) for row in raw_grid[data_start_row:]]
    if len(data_rows) < 2:
        return False

    header_continuations: list[str] = []
    while data_rows and not _semantic_cell_text(data_rows[0][0]) and _semantic_cell_text(data_rows[0][1]):
        header_continuations.append(_semantic_cell_text(data_rows.pop(0)))
    if not data_rows:
        return False

    semantic_grid: list[list[str | None]] = []
    idx = 0
    while idx < len(data_rows):
        anchor = _semantic_cell_text(data_rows[idx][0])
        value = _semantic_cell_text(data_rows[idx][1])
        if not anchor or not value:
            return False
        idx += 1
        value_parts = [value]
        while idx < len(data_rows) and not _semantic_cell_text(data_rows[idx][0]) and _semantic_cell_text(data_rows[idx][1]):
            value_parts.append(_semantic_cell_text(data_rows[idx][1]))
            idx += 1
        semantic_grid.append([anchor, _merge_vector_ocr_multiline_values(value_parts)])

    source_row_count = len(data_rows) + len(header_continuations)
    if len(semantic_grid) > source_row_count:
        return False
    merged_header_value = _merge_vector_ocr_multiline_values([header_texts[1], *header_continuations])
    _write_semantic_projection(
        table,
        header_texts=[header_texts[0], merged_header_value],
        semantic_grid=semantic_grid,
    )
    table["semantic_compaction"] = {
        "applied": True,
        "strategy": "vector_ocr_placeholder_tail_column_pairs",
        "source_data_row_count": source_row_count,
        "compacted_data_row_count": len(semantic_grid),
        "header_continuation_row_count": len(header_continuations),
        "groups": [
            {
                "anchor_text": row[0],
                "source_data_row_count": 1,
            }
            for row in semantic_grid
        ],
    }
    return True


def _select_cross_page_boundary_merge(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> tuple[int, str] | None:
    previous_grid = previous.get("data_grid") or previous.get("grid") or []
    current_grid = current.get("data_grid") or current.get("grid") or []
    if not previous_grid or not current_grid:
        return None

    previous_last_row = previous_grid[-1]
    current_first_row = current_grid[0]
    current_non_empty_columns = _semantic_non_empty_columns(current_first_row)
    if len(current_non_empty_columns) != 1:
        return None

    target_col = current_non_empty_columns[0]
    if target_col <= 0:
        return None
    if any(_semantic_cell_text(current_first_row[idx]) for idx in range(target_col)):
        return None

    previous_non_empty_columns = _semantic_non_empty_columns(previous_last_row)
    if not previous_non_empty_columns:
        return None
    if previous_non_empty_columns[-1] != target_col:
        return None
    if not any(idx < target_col for idx in previous_non_empty_columns):
        return None

    previous_text = _semantic_cell_text(previous_last_row[target_col] if target_col < len(previous_last_row) else None)
    current_text = _semantic_cell_text(current_first_row[target_col] if target_col < len(current_first_row) else None)
    if not previous_text or not current_text:
        return None
    if _ends_with_strong_terminal(previous_text):
        return None

    merged_text = _join_boundary_cell_text(previous_text, current_text)
    if not merged_text or merged_text == previous_text:
        return None
    return target_col, merged_text


def repair_cross_page_boundary_row_splits(table_asts: list[dict[str, Any]]) -> int:
    """Repair semantic-only row splits at validated continuation boundaries."""
    if not table_asts:
        return 0

    table_by_id = {
        str(table.get("table_id", "")).strip(): table
        for table in table_asts
        if str(table.get("table_id", "")).strip()
    }
    repaired = 0

    for current in sorted(
        table_asts,
        key=lambda item: (int(item.get("page", 0)), item.get("bbox", [0.0, 0.0])[1]),
    ):
        parent_id = str(current.get("continued_from", "")).strip()
        if not parent_id:
            continue
        previous = table_by_id.get(parent_id)
        if previous is None:
            continue

        candidate = _select_cross_page_boundary_merge(previous, current)
        if candidate is None:
            continue

        target_col, merged_text = candidate
        previous_grid = _clone_grid_rows(previous.get("data_grid") or previous.get("grid") or [])
        current_grid = _clone_grid_rows(current.get("data_grid") or current.get("grid") or [])
        if not previous_grid or not current_grid:
            continue

        previous_grid[-1][target_col] = merged_text
        current_grid = current_grid[1:]

        _write_semantic_grid(previous, previous_grid)
        _write_semantic_grid(current, current_grid)
        repaired += 1

    return repaired


def _apply_date_prefix_split(table: dict[str, Any]) -> bool:
    """Split rows where the second column starts with a date + rest."""
    grid = _get_authoritative_raw_grid(table)
    if not grid or not grid[0]:
        return False
    header_row = grid[0]
    if header_row[0] not in (None, "", "null"):
        return False
    matched_rows = 0
    for row in grid[1:]:
        if len(row) < 2:
            continue
        second = row[1]
        if not isinstance(second, str):
            continue
        if not second.strip():
            continue
        if not _DATE_PREFIX_RE.match(second.strip()):
            continue
        matched_rows += 1
    if matched_rows < 3:
        return False

    for idx, row in enumerate(grid):
        if len(row) < 2:
            continue
        text = row[1]
        if not isinstance(text, str) or not text.strip():
            continue
        match = _DATE_PREFIX_RE.match(text.strip())
        if not match:
            continue
        row[0] = match.group("date")
        row[1] = match.group("rest")
    table["raw_grid"] = grid
    _align_header_row(table)
    _refresh_row_texts_from_grid(table)
    table["cells"] = _build_cells_from_grid(table.get("grid", []))
    return True


def _estimate_row_height(table: dict[str, Any]) -> float:
    """根据逻辑行数与 bbox 估算行高"""
    row_count = int(
        table.get("display_row_count")
        or table.get("row_count")
        or table.get("logical_row_count")
        or 0
    )
    if row_count <= 0:
        return 0.0
    bbox = tuple(table.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    height = bbox[3] - bbox[1]
    if height <= 0:
        return 0.0
    return height / row_count


def merge_two_table_fragments(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """合并两个表格片段"""
    dense_overlap_merge = _can_merge_dense_overlapping_grid_fragments(primary, secondary)
    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))

    # 合并 bbox
    primary["bbox"] = [
        min(primary_bbox[0], secondary_bbox[0]),
        min(primary_bbox[1], secondary_bbox[1]),
        max(primary_bbox[2], secondary_bbox[2]),
        max(primary_bbox[3], secondary_bbox[3]),
    ]

    # 计算行偏移
    current_max_row = max((int(cell.get("logical_row", cell.get("row", 0))) for cell in primary.get("cells", [])), default=0)

    # 合并单元格
    secondary_cells: list[dict[str, Any]] = []
    for cell in secondary.get("cells", []):
        cell_copy = dict(cell)
        cell_copy["physical_row"] = cell_copy.get("row", 0)
        cell_copy["logical_row"] = cell_copy.get("row", 0) + current_max_row
        cell_copy["row"] = cell_copy["logical_row"]
        secondary_cells.append(cell_copy)

    primary.setdefault("cells", [])
    primary["cells"].extend(secondary_cells)

    # 更新计数
    primary["col_count"] = max(int(primary.get("col_count", 0)), int(secondary.get("col_count", 0)))

    # 更新分数
    primary["structure_score"] = round(
        max(float(primary.get("structure_score", 0.0)), float(secondary.get("structure_score", 0.0))),
        3,
    )

    # 更新位置标记
    primary["near_page_bottom"] = bool(primary.get("near_page_bottom")) or bool(secondary.get("near_page_bottom"))
    primary["near_page_top"] = bool(primary.get("near_page_top")) and bool(secondary.get("near_page_top"))

    # 继承标题
    if not primary.get("title") and secondary.get("title"):
        primary["title"] = secondary.get("title")

    # 记录合并历史
    primary.setdefault("merged_from", [])
    primary["merged_from"].append(str(secondary.get("table_id", "")))

    primary_raw_grid = _get_authoritative_raw_grid(primary)
    secondary_raw_grid = _get_authoritative_raw_grid(secondary)
    if primary_raw_grid or secondary_raw_grid:
        if dense_overlap_merge:
            primary["raw_grid"] = _merge_grid_rows_without_duplicate_boundaries(primary_raw_grid, secondary_raw_grid)
            primary["dense_overlap_fragment_merge"] = True
        else:
            primary["raw_grid"] = primary_raw_grid + secondary_raw_grid
        _refresh_row_texts_from_grid(primary)
        primary["cells"] = _build_cells_from_grid(primary.get("grid", []))
    else:
        primary.setdefault("row_texts", [])
        primary["row_texts"].extend([str(item) for item in secondary.get("row_texts", [])])
        primary["row_count"] = int(primary.get("row_count", 0)) + int(secondary.get("row_count", 0))

    return primary


def _merge_grid_rows_without_duplicate_boundaries(
    primary_grid: list[list[str | None]],
    secondary_grid: list[list[str | None]],
) -> list[list[str | None]]:
    merged = _clone_grid_rows(primary_grid)
    seen = {_row_signature(row) for row in merged if _row_signature(row)}
    for row in _clone_grid_rows(secondary_grid):
        signature = _row_signature(row)
        if signature and signature in seen:
            continue
        merged.append(row)
        if signature:
            seen.add(signature)
    return merged


def _row_signature(row: list[Any]) -> str:
    parts = [
        re.sub(r"\s+", "", _semantic_cell_text(cell)).lower()
        for cell in row
        if _semantic_cell_text(cell)
    ]
    if not parts:
        return ""
    return "|".join(parts)


def _grid_has_dense_numeric_profile(grid: list[list[str | None]]) -> bool:
    rows = [row for row in grid if isinstance(row, list)]
    if len(rows) < 3:
        return False
    wide_rows = 0
    numeric_like = 0
    filled = 0
    for row in rows:
        values = [_semantic_cell_text(cell) for cell in row if _semantic_cell_text(cell)]
        if len(values) >= 6:
            wide_rows += 1
        for value in values:
            filled += 1
            if re.search(r"\d", value):
                numeric_like += 1
    if wide_rows < max(2, len(rows) // 2):
        return False
    return numeric_like >= max(4, filled // 3)


def merge_same_page_table_fragments(
    page_tables: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """合并同页表格片段"""
    if not page_tables:
        return [], 0

    sorted_tables = sorted(page_tables, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    merged_tables: list[dict[str, Any]] = [dict(sorted_tables[0])]
    merge_count = 0

    for table in sorted_tables[1:]:
        candidate = dict(table)
        previous = merged_tables[-1]
        if can_merge_horizontal_table_fragments_on_same_page(previous, candidate):
            merged_tables[-1] = merge_horizontal_table_fragments(previous, candidate)
            merge_count += 1
            continue
        if can_merge_table_fragments_on_same_page(previous, candidate):
            merged_tables[-1] = merge_two_table_fragments(previous, candidate)
            merge_count += 1
            continue
        merged_tables.append(candidate)

    return merged_tables, merge_count


# ============================================================================
# Table ID Management
# ============================================================================

def renumber_table_ids(table_asts: list[dict[str, Any]]) -> None:
    """重新编号表格 ID"""
    old_to_new_id: dict[str, str] = {}
    for i, table in enumerate(table_asts, start=1):
        old_id = str(table.get("table_id", ""))
        new_id = f"tbl_{i:03d}"
        old_to_new_id[old_id] = new_id
        table["table_id"] = new_id

    # 更新交叉引用
    for table in table_asts:
        old_from = table.get("continued_from")
        if old_from and old_from in old_to_new_id:
            table["continued_from"] = old_to_new_id[old_from]

        old_to_list = table.get("continued_to")
        if old_to_list:
            table["continued_to"] = [
                old_to_new_id.get(t, t) for t in old_to_list
            ]


# ============================================================================
# Cell Management
# ============================================================================

def deduplicate_cells(table_ast: dict[str, Any]) -> int:
    """去除重复单元格"""
    cells = table_ast.get("cells", [])
    if not cells:
        return 0

    seen: set[tuple[int, int]] = set()
    unique_cells: list[dict[str, Any]] = []
    duplicates = 0

    for cell in cells:
        key = (cell.get("logical_row", cell.get("row", 0)), cell.get("col", 0))
        if key not in seen:
            seen.add(key)
            unique_cells.append(cell)
        else:
            duplicates += 1

    table_ast["cells"] = unique_cells
    return duplicates


def sort_cells_by_position(table_ast: dict[str, Any]) -> None:
    """按位置排序单元格"""
    cells = table_ast.get("cells", [])
    if cells:
        table_ast["cells"] = sorted(
            cells,
            key=lambda c: (c.get("logical_row", c.get("row", 0)), c.get("col", 0))
        )


# ============================================================================
# Same-title Table Merging
# ============================================================================

def can_merge_tables_with_same_title(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    """检查两个具有相同标题的表格是否可以合并"""
    if int(primary.get("page", 0)) != int(secondary.get("page", 0)):
        return False
    if _has_title_conflict(primary, secondary):
        return False

    primary_title = _compact_text(str(primary.get("title", "")))
    secondary_title = _compact_text(str(secondary.get("title", "")))
    if not primary_title or not secondary_title or primary_title != secondary_title:
        return False
    if _has_preceding_text_barrier(primary, secondary):
        return False
    if can_merge_table_fragments_on_same_page(primary, secondary):
        return True

    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    vertical_gap = secondary_bbox[1] - primary_bbox[3]

    if vertical_gap < -12 or vertical_gap > 320:
        return False

    overlap = _horizontal_overlap_ratio(primary_bbox, secondary_bbox)
    if overlap < 0.4:
        return False

    similarity = column_similarity(
        list(primary.get("column_signature", [])),
        list(secondary.get("column_signature", [])),
    )
    if similarity >= 0.45:
        return True

    # Allow merge for same title when fragments are vertically adjacent and almost same width.
    return overlap >= 0.92 and abs(vertical_gap) <= 10


def _compact_text(text: str) -> str:
    """Compact text by removing extra whitespace."""
    return " ".join(text.split()).strip().lower()


def merge_same_title_tables(
    page_tables: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """合并具有相同标题的表格

    Args:
        page_tables: 表格列表

    Returns:
        Tuple of (merged tables, merge count)
    """
    if not page_tables:
        return [], 0

    sorted_tables = sorted(page_tables, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    merged_tables: list[dict[str, Any]] = []
    merge_count = 0

    for table in sorted_tables:
        candidate = dict(table)
        if not merged_tables:
            merged_tables.append(candidate)
            continue
        previous = merged_tables[-1]
        if can_merge_tables_with_same_title(previous, candidate):
            merged_tables[-1] = merge_two_table_fragments(previous, candidate)
            merge_count += 1
            continue
        merged_tables.append(candidate)

    return merged_tables, merge_count


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Utilities
    "column_similarity",
    "header_similarity",
    "compact_vector_ocr_placeholder_tail_column_pairs",
    "compact_vector_ocr_sparse_anchor_rows",
    "project_table_header_grammar",
    "project_compound_spanning_table_headers",
    "project_rowspan_body_groups",
    "project_simple_schema_header_data_views",
    "extract_trailing_table_note_rows",
    "compact_leading_key_carry_forward_rows",
    # Cross-page stitching
    "stitch_cross_page_tables",
    "repair_cross_page_boundary_row_splits",
    "split_internal_table_segments",
    # Same-page merging
    "can_merge_table_fragments_on_same_page",
    "merge_two_table_fragments",
    "merge_same_page_table_fragments",
    # Same-title merging
    "can_merge_tables_with_same_title",
    "merge_same_title_tables",
    # Table ID management
    "renumber_table_ids",
    # Cell management
    "deduplicate_cells",
    "sort_cells_by_position",
]
