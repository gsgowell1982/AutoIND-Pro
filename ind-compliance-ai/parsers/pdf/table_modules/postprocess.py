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
from .cell_text_projection import project_table_grid_display_text

_CONTINUATION_HINT_KEYWORDS = ["continued", "continued from", "续表", "续页"]
_CONTINUATION_HINT_KEYWORDS = ["continued", "continued from", "\u7eed\u8868", "\u7eed\u9875"]
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
    if signal in {"continuation_title", "running_header", "none"}:
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
    if signal in {"continuation_title", "running_header", "none"}:
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
    first_row = grid[0]
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


def _build_row_views_from_raw_grid(
    raw_grid: list[list[str | None]],
    *,
    data_start_row: int = 0,
    raw_audit_grid: list[list[str | None]] | None = None,
) -> tuple[list[list[str | None]], list[list[str | None]], list[str], list[str], list[str], list[int]]:
    projected_grid, projected_data_start_row = _project_sparse_header_continuation_rows(
        raw_grid,
        data_start_row=data_start_row,
    )
    projected_grid = _project_section_group_rows(projected_grid)
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
    return True


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
    table["header"] = [{"col": idx + 1, "text": text} for idx, text in enumerate(candidates)]
    table["col_count"] = len(candidates)
    table["header_rebuilt_by_guard"] = True
    return True


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
    display_grid, data_grid, raw_row_texts, display_row_texts, data_row_texts, structural_empty_rows = _build_row_views_from_raw_grid(
        semantic_source_grid,
        data_start_row=data_start_row,
        raw_audit_grid=raw_grid,
    )

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
    if structural_empty_rows:
        table["structural_empty_rows"] = structural_empty_rows
    else:
        table.pop("structural_empty_rows", None)


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
    table["semantic_compaction"] = {
        "applied": True,
        "strategy": "leading_key_carry_forward",
        "source_data_row_count": len(data_grid),
        "compacted_data_row_count": len(semantic_grid),
        "groups": semantic_groups,
    }
    return True


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
        primary["raw_grid"] = primary_raw_grid + secondary_raw_grid
        _refresh_row_texts_from_grid(primary)
        primary["cells"] = _build_cells_from_grid(primary.get("grid", []))
    else:
        primary.setdefault("row_texts", [])
        primary["row_texts"].extend([str(item) for item in secondary.get("row_texts", [])])
        primary["row_count"] = int(primary.get("row_count", 0)) + int(secondary.get("row_count", 0))

    return primary


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
