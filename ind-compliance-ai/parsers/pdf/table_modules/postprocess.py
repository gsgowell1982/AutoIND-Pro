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

import re
from typing import Any


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
    """跨页表格拼接

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

    stitched_count = 0
    sorted_tables = sorted(table_asts, key=lambda item: (item["page"], item["bbox"][1]))

    for previous, current in zip(sorted_tables, sorted_tables[1:]):
        if current["page"] != previous["page"] + 1:
            continue

        prev_page_height = max(1.0, page_heights.get(previous["page"], 1.0))
        curr_page_height = max(1.0, page_heights.get(current["page"], 1.0))
        prev_near_bottom = previous["bbox"][3] >= prev_page_height * 0.72
        curr_near_top = current["bbox"][1] <= curr_page_height * 0.28

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

        # Context-based continuation detection
        context_continuation = (
            (previous_has_title or previous_has_section_hint)
            and curr_near_top
            and abs(previous_col_count - current_col_count) <= 1
            and (
                (previous_is_glossary_context and similarity >= 0.22)
                or (not previous_is_glossary_context and similarity >= 0.28)
            )
        )

        is_likely_continuation = (
            (prev_near_bottom and curr_near_top)
            or similarity >= 0.82
            or ((previous_has_title or previous_has_section_hint) and curr_near_top and similarity >= 0.42)
            or context_continuation
            or hint_matches_previous
        )

        min_similarity = 0.22 if hint_matches_previous and previous_is_glossary_context else 0.28
        if similarity < min_similarity or not is_likely_continuation:
            continue

        if str(current.get("title", "")).strip():
            continue

        # Set cross-page links
        previous.setdefault("continued_to", [])
        previous["continued_to"].append(current["table_id"])
        current["continued_from"] = previous["table_id"]
        current["cross_page_similarity"] = round(similarity, 3)
        current["is_continuation"] = True

        # Record inheritance metadata
        continuation_source: dict[str, Any] = {
            "source_table_id": str(previous.get("table_id", "")),
            "strategy": "cross_page_stitch",
            "similarity": round(similarity, 3),
            "inherited_fields": [],
        }

        # Inherit title
        if not str(current.get("title", "")).strip() and str(previous.get("title", "")).strip():
            current["title"] = previous["title"]
            current["title_inherited"] = True
            continuation_source["inherited_fields"].append("title")

        # Inherit header if different
        hdr_similarity = header_similarity(previous.get("header", []), current.get("header", []))
        if hdr_similarity < 0.45:
            current["header"] = previous.get("header", [])
            current["header_inherited"] = True
            continuation_source["inherited_fields"].append("header")

        current["continuation_source"] = continuation_source
        stitched_count += 1

    return stitched_count


def _is_glossary_hint_text(text: str) -> bool:
    """检查文本是否指示术语表上下文"""
    normalized = text.strip().lower()
    return "术语表" in normalized or "词汇表" in normalized or "glossary" in normalized


# ============================================================================
# Same-Page Fragment Merging
# ============================================================================

def can_merge_table_fragments_on_same_page(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    """检查同页上的两个表格片段是否可以合并"""
    if int(primary.get("page", 0)) != int(secondary.get("page", 0)):
        return False

    primary_bbox = tuple(primary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    secondary_bbox = tuple(secondary.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    vertical_gap = secondary_bbox[1] - primary_bbox[3]

    if vertical_gap < -10 or vertical_gap > 52:
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
    primary["row_count"] = int(primary.get("row_count", 0)) + int(secondary.get("row_count", 0))
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

    primary.setdefault("row_texts", [])
    primary["row_texts"].extend([str(item) for item in secondary.get("row_texts", [])])

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

    primary_title = _compact_text(str(primary.get("title", "")))
    secondary_title = _compact_text(str(secondary.get("title", "")))
    if not primary_title or not secondary_title or primary_title != secondary_title:
        return False

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
    # Cross-page stitching
    "stitch_cross_page_tables",
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
