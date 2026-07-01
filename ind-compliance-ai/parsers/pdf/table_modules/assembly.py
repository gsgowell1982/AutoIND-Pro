"""Table Instance Assembly Layer - 表格实例组装层

Architecture:
    Normalized Evidence → Assembly → Table Instance

负责将规范化证据组装为表格实例：
- 识别表头行
- 处理单元格合并 (rowspan/colspan)
- 构建表格结构
- 生成表格 AST 雏形

关键特性：
- 表头检测：首行或继承父表
- 单元格合并：续表行合并、括号缩写合并
- 空值处理：显式 null 占位符
"""

# Version: v1.0.6
# Optimization Summary:
# - Preserve auditability for inferred cell values by exposing inference metadata.
# - Split row representation into raw audit view and semantic business view.
# - Filter fully empty logical rows out of the main AST-facing row_texts/grid path.
# - Persist opening-row semantics and explicit data-row views so downstream
#   compliance checks can consume title/header/data as separate layers.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .normalization import NormalizedTable, NormalizedCell, NormalizedRow
from .raw_objects import RawTableEvidence

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
    "编号",
    "名称",
    "类型",
    "内容",
    "说明",
    "描述",
    "结果",
    "单位",
    "日期",
    "路径",
    "文件",
    "目的地",
}


@dataclass
class TableHeader:
    """表格表头"""
    cells: list[dict[str, Any]]  # [{col: int, text: str}]
    row_index: int = 0  # 表头所在行索引
    inherited: bool = False  # 是否继承自父表
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": self.cells,
            "row_index": self.row_index,
            "inherited": self.inherited,
        }


@dataclass
class TableOpeningStructure:
    """Opening-row structure for tables with optional internal title/header rows."""
    title_row_index: int | None = None
    title_text: str | None = None
    header_row_index: int | None = None
    header_cells: list[dict[str, Any]] = field(default_factory=list)
    header_texts: list[str] = field(default_factory=list)


@dataclass
class TableInstance:
    """表格实例
    
    组装完成的表格实例，包含：
    - 表头信息
    - 单元格数据
    - 结构元数据
    - 状态标记
    
    这是 Continuum Engine 的输入对象。
    """
    # 基础信息
    table_id: str
    page_number: int
    bbox: tuple[float, float, float, float]
    
    # 结构信息
    col_count: int  # 逻辑列数
    row_count: int  # 展示层语义行数（保留 opening rows，兼容内部流程）
    header: TableHeader | None = None
    
    # 单元格
    cells: list[dict[str, Any]] = field(default_factory=list)
    grid: list[list[str | None]] = field(default_factory=list)
    raw_grid: list[list[str | None]] = field(default_factory=list)
    data_grid: list[list[str | None]] = field(default_factory=list)
    
    # 行文本 (用于验证)
    row_texts: list[str] = field(default_factory=list)
    raw_row_texts: list[str] = field(default_factory=list)
    data_row_texts: list[str] = field(default_factory=list)
    raw_row_count: int = 0
    data_row_count: int = 0
    structural_empty_rows: list[int] = field(default_factory=list)
    
    # 物理信息
    physical_col_count: int = 0
    physical_row_count: int = 0
    
    # 页面位置
    near_page_top: bool = False
    near_page_bottom: bool = False
    
    # 续表标记
    is_continuation: bool = False
    needs_header_inheritance: bool = False
    
    # 列签名
    column_signature: list[float] = field(default_factory=list)
    column_hash: str = ""
    
    # 检测方法
    detection_method: str = "pymupdf_builtin"
    normalization_strategy: str = "column_clustering"
    
    # 上下文 (由 validation 层填充)
    title: str | None = None
    section_hint: str | None = None
    toc_context: bool = False
    grid_line_score: float = 0.0
    title_row_index: int | None = None
    header_row_index: int | None = None
    data_start_row: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "block_type": "table",
            "table_id": self.table_id,
            "page": self.page_number,
            "bbox": list(self.bbox),
            "col_count": self.col_count,
            "row_count": self.row_count,
            "raw_row_count": self.raw_row_count,
            "header": self.header.to_dict() if self.header else None,
            "cells": self.cells,
            "grid": self.grid,
            "raw_grid": self.raw_grid,
            "data_grid": self.data_grid,
            "row_texts": self.row_texts,
            "raw_row_texts": self.raw_row_texts,
            "data_row_texts": self.data_row_texts,
            "structural_empty_rows": self.structural_empty_rows,
            "physical_col_count": self.physical_col_count,
            "physical_row_count": self.physical_row_count,
            "near_page_top": self.near_page_top,
            "near_page_bottom": self.near_page_bottom,
            "is_continuation": self.is_continuation,
            "data_row_count": self.data_row_count,
            "column_hash": self.column_hash,
            "detection_method": self.detection_method,
            "title_row_index": self.title_row_index,
            "header_row_index": self.header_row_index,
            "data_start_row": self.data_start_row,
        }


# ============================================================================
# Assembly Functions
# ============================================================================

def assemble_table_instance(
    normalized: NormalizedTable,
    table_id: str,
    parent_header: list[dict[str, Any]] | None = None,
    assessment: Any = None,  # ContinuationAssessment 类型，避免循环导入
) -> TableInstance:
    """组装表格实例 (v1.3.0 更新)

    核心步骤：
    1. 检测是否为续表 - 优先使用 Step2 的评估结果
    2. 确定表头 - 续表继承父表表头
    3. 处理单元格合并
    4. 构建列签名

    Args:
        normalized: 规范化表格数据
        table_id: 表格 ID
        parent_header: 父表表头 (用于续表继承)
        assessment: Step2 的评估结果

    Returns:
        TableInstance 表格实例
    """
    # Step 1: 检测续表 - 优先使用 Step2 的评估结果
    is_continuation = _detect_continuation(normalized, parent_header, assessment)
    opening_structure = analyze_table_opening_structure(
        normalized.grid,
        normalized.logical_col_count,
    )

    # Step 2: 确定表头 - 续表继承父表表头
    header, data_start_row = _determine_header(
        normalized,
        is_continuation,
        parent_header,
        assessment,
        opening_structure=opening_structure,
    )

    # Step 3: 处理单元格
    cells = _build_cells(
        normalized,
        data_start_row,
        header.row_index if header else 0,
    )

    # Step 4: 单元格合并处理
    cells = _apply_cell_merging(cells, normalized.grid)

    # Step 5: 构建原始/语义双轨行视图
    raw_grid = [list(row) for row in normalized.grid]
    raw_row_texts = _build_row_texts(raw_grid)
    semantic_grid, semantic_cells, row_texts, data_grid, data_row_texts, structural_empty_rows = _build_semantic_table_views(
        raw_grid,
        cells,
        data_start_row=data_start_row,
    )

    # Step 6: 构建列签名
    column_signature = _build_column_signature(normalized)
    column_hash = _compute_column_hash(column_signature)

    # 构建表格实例
    instance = TableInstance(
        table_id=table_id,
        page_number=normalized.page_number,
        bbox=normalized.bbox,
        col_count=normalized.logical_col_count,
        row_count=len(semantic_grid),
        header=header,
        cells=semantic_cells,
        grid=semantic_grid,
        raw_grid=raw_grid,
        data_grid=data_grid,
        row_texts=row_texts,
        raw_row_texts=raw_row_texts,
        data_row_texts=data_row_texts,
        raw_row_count=len(raw_grid),
        data_row_count=len(data_grid),
        structural_empty_rows=structural_empty_rows,
        physical_col_count=normalized.physical_col_count,
        physical_row_count=normalized.physical_row_count,
        near_page_top=normalized.near_page_top,
        near_page_bottom=normalized.near_page_bottom,
        is_continuation=is_continuation,
        needs_header_inheritance=is_continuation and header and not header.inherited,
        column_signature=column_signature,
        column_hash=column_hash,
        detection_method=normalized.source,
        normalization_strategy=normalized.normalization_strategy,
        title=opening_structure.title_text,
        title_row_index=opening_structure.title_row_index,
        header_row_index=opening_structure.header_row_index,
        data_start_row=data_start_row,
    )

    return instance


def _detect_continuation(
    normalized: NormalizedTable,
    parent_header: list[dict[str, Any]] | None,
    assessment: Any = None,  # ContinuationAssessment 类型
) -> bool:
    """检测是否为续表 (v1.3.0 更新)

    优化要点：
    1. 如果 Step2 已确认为续表，直接返回 True
    2. 否则降级到原有判断逻辑
    
    续表特征：
    1. 页面顶部位置
    2. 首行像数据行（不是表头）
    3. 有父表上下文
    """
    # 如果 Step2 已确认为续表，直接返回
    if assessment and hasattr(assessment, 'is_continuation') and assessment.is_continuation:
        return True
    
    # 降级判断：基本条件
    if not normalized.near_page_top:
        return False

    if not parent_header:
        return False

    # 检查首行是否像数据行
    first_row = normalized.rows[0] if normalized.rows else None
    if not first_row:
        return False

    # 首行特征分析
    first_row_texts = [c.text for c in first_row.cells if c.text]
    if not first_row_texts:
        return False
    
    combined_text = " ".join(first_row_texts)
    
    # 数据行特征
    data_indicators = ["回复", "撤回", "提交", "模块", "m1", "m2", "m3", "文件夹"]
    is_data_like = any(ind in combined_text.lower() for ind in data_indicators)
    
    # 短数字开头
    starts_with_number = combined_text.strip().isdigit() and len(combined_text.strip()) <= 4
    
    # 非空单元格数量较少
    sparse_content = first_row.non_empty_count <= 2
    
    return is_data_like or starts_with_number or sparse_content


def _determine_header(
    normalized: NormalizedTable,
    is_continuation: bool,
    parent_header: list[dict[str, Any]] | None,
    assessment: Any = None,  # ContinuationAssessment 类型
    opening_structure: TableOpeningStructure | None = None,
) -> tuple[TableHeader | None, int]:
    """确定表头 (v1.3.0 更新)

    优化要点：
    1. 续表优先继承父表表头
    2. 符合 IND 文档中"首表提供表头、后续页面延续数据"的实际模式

    Returns:
        (header, data_start_row)
    """
    col_count = normalized.logical_col_count
    opening = opening_structure or analyze_table_opening_structure(normalized.grid, col_count)
    title_row_end = (opening.title_row_index + 1) if opening.title_row_index is not None else 0

    if is_continuation and parent_header:
        # 续表继承父表表头
        header = TableHeader(
            cells=parent_header,
            row_index=0,
            inherited=True,
        )
        if opening.header_row_index is not None and _header_rows_are_compatible(
            opening.header_cells,
            parent_header,
        ):
            _, data_start_row = _merge_sparse_header_continuation_row(
                opening.header_cells,
                normalized.grid,
                opening.header_row_index,
                col_count,
            )
            return header, data_start_row
        if opening.title_row_index is not None:
            return header, title_row_end
        return header, 0  # data_start_row = 0，因为续表没有表头行

    if opening.header_cells:
        header_cells, data_start_row = _merge_sparse_header_continuation_row(
            opening.header_cells,
            normalized.grid,
            opening.header_row_index or 0,
            col_count,
        )
        header_cells, data_start_row = _merge_dense_header_continuation_row(
            header_cells,
            normalized.grid,
            data_start_row,
            col_count,
        )
        header = TableHeader(
            cells=header_cells,
            row_index=opening.header_row_index or 0,
            inherited=False,
        )
        return header, data_start_row

    # 非续表：降级到旧规则，使用首个有效行作为表头。
    fallback_row_index = title_row_end if title_row_end < len(normalized.rows) else 0
    first_row = normalized.rows[fallback_row_index] if normalized.rows and fallback_row_index < len(normalized.rows) else None
    if not first_row:
        return None, title_row_end

    header_cells = _build_header_cells_from_row(
        normalized.grid[fallback_row_index] if fallback_row_index < len(normalized.grid) else [],
        col_count,
    )
    header = TableHeader(
        cells=header_cells,
        row_index=fallback_row_index,
        inherited=False,
    )

    return header, fallback_row_index + 1


def analyze_table_opening_structure(
    grid: list[list[str | None]],
    col_count: int,
) -> TableOpeningStructure:
    """Classify opening rows into optional table-title row and column-header row."""
    structure = TableOpeningStructure()
    if not grid or col_count <= 0:
        return structure

    row_summaries = [_summarize_row(row) for row in grid[:3]]
    if not row_summaries:
        return structure

    header_candidate_index = 0
    title_row_texts: list[str] = []
    title_probe_index = 0
    if len(row_summaries) >= 2 and _looks_like_explicit_tabular_title_text(row_summaries[0].get("merged_text", "")):
        while title_probe_index < len(row_summaries) - 1:
            current_row = row_summaries[title_probe_index]
            if not _is_internal_title_line_candidate(current_row):
                break

            merged_text = str(current_row.get("merged_text", "")).strip()
            if merged_text:
                title_row_texts.append(merged_text)

            next_row = row_summaries[title_probe_index + 1]
            next_next_row = (
                row_summaries[title_probe_index + 2]
                if title_probe_index + 2 < len(row_summaries)
                else None
            )
            if _is_column_header_row_candidate(next_row, next_next_row, col_count):
                structure.title_row_index = 0
                structure.title_text = " ".join(title_row_texts).strip() or None
                header_candidate_index = title_probe_index + 1
                break

            title_probe_index += 1

    if structure.title_text is None and len(row_summaries) >= 2 and _is_internal_title_row_candidate(
        row_summaries[0],
        row_summaries[1],
        col_count,
    ):
        structure.title_row_index = 0
        structure.title_text = row_summaries[0]["merged_text"] or None
        header_candidate_index = 1

    if header_candidate_index < len(grid):
        current_summary = row_summaries[header_candidate_index]
        next_summary = (
            row_summaries[header_candidate_index + 1]
            if header_candidate_index + 1 < len(row_summaries)
            else None
        )
        if _is_column_header_row_candidate(current_summary, next_summary, col_count):
            structure.header_row_index = header_candidate_index
            structure.header_cells = _build_header_cells_from_row(
                grid[header_candidate_index],
                col_count,
            )
            structure.header_texts = [
                str(cell.get("text", "")).strip()
                for cell in structure.header_cells
                if str(cell.get("text", "")).strip()
            ]

    return structure


def _summarize_row(row: list[str | None]) -> dict[str, Any]:
    non_empty: list[tuple[int, str]] = []
    path_like_count = 0
    numeric_like_count = 0
    textual_count = 0
    lengths: list[int] = []

    for col_idx, cell in enumerate(row):
        text = _clean_text(cell)
        if not text:
            continue
        non_empty.append((col_idx, text))
        lengths.append(len(text))
        if _looks_like_pathish_text(text):
            path_like_count += 1
        if _looks_like_numeric_marker(text):
            numeric_like_count += 1
        if re.search(r"[A-Za-z\u4e00-\u9fff]", text):
            textual_count += 1

    merged_text = " ".join(text for _, text in non_empty).strip()
    return {
        "non_empty": non_empty,
        "non_empty_count": len(non_empty),
        "merged_text": merged_text,
        "path_like_count": path_like_count,
        "numeric_like_count": numeric_like_count,
        "textual_count": textual_count,
        "max_len": max(lengths, default=0),
        "avg_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
    }


def _is_internal_title_row_candidate(
    first_row: dict[str, Any],
    second_row: dict[str, Any],
    col_count: int,
) -> bool:
    if not _is_internal_title_line_candidate(first_row):
        return False
    if second_row.get("non_empty_count", 0) < min(2, col_count):
        return False
    return _is_column_header_row_candidate(second_row, None, col_count)


def _is_internal_title_line_candidate(row: dict[str, Any]) -> bool:
    if row.get("non_empty_count", 0) != 1:
        return False
    title_text = str(row.get("merged_text", "")).strip()
    if not _looks_like_title_text(title_text):
        return False
    title_col = row.get("non_empty", [(0, "")])[0][0]
    return title_col == 0


def _is_column_header_row_candidate(
    row: dict[str, Any],
    next_row: dict[str, Any] | None,
    col_count: int,
) -> bool:
    non_empty_count = int(row.get("non_empty_count", 0) or 0)
    if non_empty_count < min(2, col_count):
        return False
    if int(row.get("path_like_count", 0) or 0) > 0:
        return False
    if int(row.get("numeric_like_count", 0) or 0) > max(0, non_empty_count - 2):
        return False
    if int(row.get("textual_count", 0) or 0) < min(2, non_empty_count):
        return False
    short_label_count = sum(
        1
        for _, text in row.get("non_empty", [])
        if len(str(text).strip()) <= 32
    )
    if non_empty_count <= 2 and short_label_count < non_empty_count:
        return False
    if non_empty_count > 2 and short_label_count < 2:
        return False
    if float(row.get("avg_len", 0.0) or 0.0) > 32.0:
        return False
    if int(row.get("max_len", 0) or 0) > 72:
        return False

    if next_row is None:
        return True
    if int(next_row.get("path_like_count", 0) or 0) > 0:
        return True
    if int(next_row.get("numeric_like_count", 0) or 0) > 0:
        return True
    return non_empty_count >= int(next_row.get("non_empty_count", 0) or 0)


def _looks_like_title_text(text: str | None) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    if len(cleaned) < 8:
        return False
    if _looks_like_pathish_text(cleaned):
        return False
    if re.fullmatch(r"[\d.\-_/]+", cleaned):
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned))


def _looks_like_explicit_tabular_title_text(text: str | None) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    return bool(
        re.search(r"\btable\b", cleaned, re.IGNORECASE)
        or re.search(r"表\s*\d+", cleaned)
    )


def _looks_like_pathish_text(text: str | None) -> bool:
    cleaned = _clean_text(text) or ""
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


def _looks_like_numeric_marker(text: str | None) -> bool:
    cleaned = _clean_text(text) or ""
    return bool(re.fullmatch(r"[-–—]|\(?\d+(?:\.\d+)*\)?", cleaned))


def _build_header_cells_from_row(
    row: list[str | None],
    col_count: int,
) -> list[dict[str, Any]]:
    header_cells: list[dict[str, Any]] = []
    for logical_col in range(col_count):
        text = _clean_text(row[logical_col]) if logical_col < len(row) else None
        header_cells.append({
            "col": logical_col + 1,
            "text": text or f"Column {logical_col + 1}",
        })
    return header_cells


def _merge_sparse_header_continuation_row(
    header_cells: list[dict[str, Any]],
    grid: list[list[str | None]],
    header_row_index: int,
    col_count: int,
) -> tuple[list[dict[str, Any]], int]:
    data_start_row = header_row_index + 1
    if not header_cells or data_start_row >= len(grid):
        return list(header_cells), data_start_row

    header_row = grid[header_row_index] if 0 <= header_row_index < len(grid) else []
    continuation_row = grid[data_start_row]
    next_row = grid[data_start_row + 1] if data_start_row + 1 < len(grid) else None
    if not _is_sparse_header_continuation_row_candidate(
        header_row,
        continuation_row,
        next_row,
        col_count,
    ):
        return list(header_cells), data_start_row

    merged_header_cells = [dict(cell) for cell in header_cells]
    for col_idx, cell in enumerate(continuation_row[:col_count]):
        continuation_text = _clean_text(cell)
        if not continuation_text:
            continue
        base_text = _clean_text(merged_header_cells[col_idx].get("text", "")) or ""
        merged_header_cells[col_idx]["text"] = _join_header_text(base_text, continuation_text)

    return merged_header_cells, data_start_row + 1


def _merge_dense_header_continuation_row(
    header_cells: list[dict[str, Any]],
    grid: list[list[str | None]],
    data_start_row: int,
    col_count: int,
) -> tuple[list[dict[str, Any]], int]:
    """Merge compact two-tier headers such as group label + metric/unit row."""
    if not header_cells or data_start_row >= len(grid) or col_count <= 1:
        return list(header_cells), data_start_row

    header_row_index = data_start_row - 1
    header_row = grid[header_row_index] if 0 <= header_row_index < len(grid) else []
    continuation_row = grid[data_start_row]
    next_row = grid[data_start_row + 1] if data_start_row + 1 < len(grid) else None
    if not _is_dense_header_continuation_row_candidate(
        header_row,
        continuation_row,
        next_row,
        col_count,
    ):
        return list(header_cells), data_start_row

    merged_header_cells = [dict(cell) for cell in header_cells]
    for col_idx in range(min(col_count, len(merged_header_cells))):
        continuation_text = _clean_text(continuation_row[col_idx] if col_idx < len(continuation_row) else None) or ""
        if not continuation_text:
            continue
        base_text = _clean_text(merged_header_cells[col_idx].get("text", "")) or ""
        if not base_text:
            merged_header_cells[col_idx]["text"] = continuation_text
            continue
        if base_text == continuation_text:
            continue
        merged_header_cells[col_idx]["text"] = _join_header_text(base_text, continuation_text)

    return merged_header_cells, data_start_row + 1


def _is_dense_header_continuation_row_candidate(
    header_row: list[str | None],
    continuation_row: list[str | None],
    next_row: list[str | None] | None,
    col_count: int,
) -> bool:
    header_summary = _summarize_row(header_row)
    continuation_summary = _summarize_row(continuation_row)
    if not _is_column_header_row_candidate(header_summary, next_row=None, col_count=col_count):
        return False
    continuation_non_empty = int(continuation_summary.get("non_empty_count", 0) or 0)
    if continuation_non_empty < max(2, min(col_count, 3)):
        return False
    if continuation_non_empty < int(header_summary.get("non_empty_count", 0) or 0) - 1:
        return False
    if int(continuation_summary.get("path_like_count", 0) or 0) > 0:
        return False
    if int(continuation_summary.get("numeric_like_count", 0) or 0) > 0:
        return False
    if int(continuation_summary.get("textual_count", 0) or 0) < max(1, continuation_non_empty - 1):
        return False
    if int(continuation_summary.get("max_len", 0) or 0) > 28:
        return False
    if not _looks_like_data_row_after_header(next_row, col_count):
        return False

    continuation_columns = {col_idx for col_idx, _ in continuation_summary.get("non_empty", [])}
    if not continuation_columns:
        return False
    for col_idx, text in continuation_summary.get("non_empty", []):
        if not _looks_like_header_metric_or_unit_text(text):
            return False
        header_text = _clean_text(header_row[col_idx] if col_idx < len(header_row) else None) or ""
        if not header_text and col_idx != 0:
            return False
    return True


def _looks_like_header_metric_or_unit_text(text: str | None) -> bool:
    cleaned = _clean_text(text) or ""
    if not cleaned:
        return False
    if len(cleaned) > 28:
        return False
    if re.search(r"\([^)]{1,12}\)", cleaned):
        return True
    if re.search(r"\d", cleaned) and re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned):
        return True
    if re.fullmatch(r"[A-Za-z]{1,8}(?:[/\-][A-Za-z]{1,8})?", cleaned):
        return True
    return bool(re.fullmatch(r"[\u4e00-\u9fffA-Za-z]{1,12}", cleaned))


def _is_sparse_header_continuation_row_candidate(
    header_row: list[str | None],
    continuation_row: list[str | None],
    next_row: list[str | None] | None,
    col_count: int,
) -> bool:
    header_summary = _summarize_row(header_row)
    continuation_summary = _summarize_row(continuation_row)
    if not _is_column_header_row_candidate(header_summary, next_row=None, col_count=col_count):
        return False

    continuation_non_empty = int(continuation_summary.get("non_empty_count", 0) or 0)
    if continuation_non_empty <= 0 or continuation_non_empty > max(2, col_count // 2):
        return False
    if int(continuation_summary.get("path_like_count", 0) or 0) > 0:
        return False
    if int(continuation_summary.get("numeric_like_count", 0) or 0) > 0:
        return False
    if int(continuation_summary.get("textual_count", 0) or 0) != continuation_non_empty:
        return False
    if int(continuation_summary.get("max_len", 0) or 0) > 24:
        return False

    header_columns = {col_idx for col_idx, _ in header_summary.get("non_empty", [])}
    continuation_columns = {col_idx for col_idx, _ in continuation_summary.get("non_empty", [])}
    if not continuation_columns or not continuation_columns.issubset(header_columns):
        return False

    for col_idx, text in continuation_summary.get("non_empty", []):
        header_text = _clean_text(header_row[col_idx] if col_idx < len(header_row) else None) or ""
        if len(header_text) < 8 and " " not in header_text and "-" not in header_text:
            return False
        if _looks_like_body_identifier_value(text):
            return False
        if not _looks_like_header_continuation_text(text):
            return False

    return _looks_like_data_row_after_header(next_row, col_count)


def _looks_like_body_identifier_value(text: str | None) -> bool:
    cleaned = _clean_text(text) or ""
    if not cleaned:
        return False
    if re.search(r"\d", cleaned) and re.search(r"[A-Za-z]", cleaned):
        return True
    if re.fullmatch(r"[A-Z]{2,}(?:[-_/][A-Z0-9]+)+", cleaned):
        return True
    return False


def _looks_like_header_continuation_text(text: str | None) -> bool:
    cleaned = _clean_text(text) or ""
    if not cleaned:
        return False

    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", cleaned.lower())
    if normalized in _HEADER_CONTINUATION_TOKENS:
        return True

    ascii_words = re.findall(r"[A-Za-z]+", cleaned)
    if ascii_words and len(ascii_words) <= 2:
        compact = "".join(word.lower() for word in ascii_words)
        if compact in _HEADER_CONTINUATION_TOKENS:
            return True

    return cleaned.isupper() and len(cleaned) <= 6


def _looks_like_data_row_after_header(
    row: list[str | None] | None,
    col_count: int,
) -> bool:
    if not row:
        return False

    summary = _summarize_row(row)
    non_empty_count = int(summary.get("non_empty_count", 0) or 0)
    if non_empty_count < min(2, col_count):
        return False
    if int(summary.get("path_like_count", 0) or 0) > 0:
        return True
    return int(summary.get("numeric_like_count", 0) or 0) > 0


def _join_header_text(left: str, right: str) -> str:
    left_clean = _clean_text(left) or ""
    right_clean = _clean_text(right) or ""
    if not left_clean:
        return right_clean
    if not right_clean:
        return left_clean
    if left_clean.endswith(right_clean):
        return left_clean
    left_clean, right_clean = _repair_trailing_leading_word_order(left_clean, right_clean)
    return f"{left_clean} {right_clean}".strip()


def _repair_trailing_leading_word_order(left: str, right: str) -> tuple[str, str]:
    left_parts = left.split()
    right_parts = right.split()
    if not left_parts or not right_parts:
        return left, right
    right_trailing = right_parts[-1]
    if (
        re.fullmatch(r"[A-Z][a-z]{3,}", right_trailing)
        and re.search(r"[\u4e00-\u9fff]", left)
        and re.search(r"[.!?。！？]$", left)
    ):
        return left, " ".join([right_trailing, *right_parts[:-1]]).strip()
    trailing = left_parts[-1]
    if not re.fullmatch(r"[A-Z][a-z]{3,}", trailing):
        return left, right
    if any(re.search(r"[\u4e00-\u9fff]", part) for part in left_parts[:-1]) and re.search(r"[.!?。！？]$", right):
        return " ".join(left_parts[:-1]).strip(), " ".join([trailing, *right_parts]).strip()
    return left, right


def _header_rows_are_compatible(
    current_header: list[dict[str, Any]],
    parent_header: list[dict[str, Any]] | None,
) -> bool:
    if not current_header or not parent_header:
        return False

    current_texts = {
        _normalize_header_token(str(item.get("text", "")))
        for item in current_header
        if _normalize_header_token(str(item.get("text", "")))
    }
    parent_texts = {
        _normalize_header_token(str(item.get("text", "")))
        for item in parent_header
        if _normalize_header_token(str(item.get("text", "")))
    }
    if not current_texts or not parent_texts:
        return False

    overlap = len(current_texts & parent_texts)
    similarity = overlap / max(len(current_texts), len(parent_texts))
    return similarity >= 0.5


def _normalize_header_token(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.strip().lower())


def _build_cells(
    normalized: NormalizedTable,
    data_start_row: int,
    header_row_count: int,
) -> list[dict[str, Any]]:
    """构建单元格列表"""
    cells = []
    logical_row_index = 0
    
    for row in normalized.rows[data_start_row:]:
        for cell in row.cells:
            cell_dict = {
                "row": logical_row_index + 1,
                "col": cell.logical_col + 1,
                "logical_row": logical_row_index + 1,
                "logical_col": cell.logical_col + 1,
                "physical_row": cell.physical_row,
                "physical_col": cell.physical_col_start + 1,
                "physical_col_start": cell.physical_col_start + 1,
                "physical_col_end": cell.physical_col_end + 1,
                "physical_colspan": cell.physical_colspan,
                "text": _clean_text(cell.text) if cell.text else None,
                "bbox": list(cell.bbox) if cell.bbox else None,
                "inferred": bool(cell.supplemented),
                "inference_reason": cell.supplement_reason,
                "rowspan": 1,
                "colspan": max(1, int(cell.physical_colspan or 1)),
            }
            cells.append(cell_dict)
        
        logical_row_index += 1
    
    return cells


def _apply_cell_merging(
    cells: list[dict[str, Any]],
    grid: list[list[str | None]],
) -> list[dict[str, Any]]:
    """应用单元格合并
    
    处理场景：
    1. 括号缩写续行合并 (eCTD) -> 上一行
    2. 单列文本续行合并
    """
    if not cells or len(grid) < 2:
        return cells
    
    # 按 logical_row 排序
    sorted_cells = sorted(cells, key=lambda c: (c.get("logical_row", 0), c.get("col", 0)))
    
    merged_cells = []
    latest_by_col: dict[int, dict[str, Any]] = {}
    
    for cell in sorted_cells:
        col_idx = cell.get("col", 1) - 1
        text = cell.get("text")
        
        if not text:
            merged_cells.append(cell)
            continue
        
        prev_cell = latest_by_col.get(col_idx)
        
        # 检查括号缩写
        if prev_cell and _is_parenthetical_abbreviation(text):
            # 合并到上一行
            prev_cell["text"] = f"{prev_cell['text']}\n{text}"
            prev_cell["rowspan"] = prev_cell.get("rowspan", 1) + 1
            continue
        
        merged_cells.append(cell)
        latest_by_col[col_idx] = cell
    
    return merged_cells


def _is_parenthetical_abbreviation(text: str | None) -> bool:
    """检查是否为括号缩写"""
    if not text:
        return False
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    return bool(re.fullmatch(r"[（(][A-Za-z0-9.\-_/]{2,28}[)）]", cleaned))


def _build_row_texts(grid: list[list[str | None]]) -> list[str]:
    """构建行文本表示"""
    row_texts = []
    for row in grid:
        parts = []
        for cell in row:
            text = _clean_text(cell) if cell else "null"
            parts.append(text if text else "null")
        row_texts.append(" | ".join(parts))
    return row_texts


def _row_has_semantic_content(row: list[str | None]) -> bool:
    """Return True when a logical row carries business content."""
    for cell in row:
        if _clean_text(cell):
            return True
    return False


def _build_semantic_table_views(
    raw_grid: list[list[str | None]],
    cells: list[dict[str, Any]],
    *,
    data_start_row: int,
) -> tuple[
    list[list[str | None]],
    list[dict[str, Any]],
    list[str],
    list[list[str | None]],
    list[str],
    list[int],
]:
    """Derive display/data row views while preserving raw audit rows."""
    semantic_grid: list[list[str | None]] = []
    data_grid: list[list[str | None]] = []
    structural_empty_rows: list[int] = []
    semantic_data_row_by_raw_grid_row: dict[int, int] = {}
    semantic_data_row_count = 0
    projected_grid = _project_grid_from_cells(raw_grid, cells, data_start_row=data_start_row)

    for raw_grid_row_idx, row in enumerate(projected_grid, start=1):
        if _row_has_semantic_content(row):
            semantic_grid.append(list(row))
            if raw_grid_row_idx > data_start_row:
                data_grid.append(list(row))
                semantic_data_row_count += 1
                semantic_data_row_by_raw_grid_row[raw_grid_row_idx] = semantic_data_row_count
            continue
        structural_empty_rows.append(raw_grid_row_idx)

    semantic_cells: list[dict[str, Any]] = []
    for cell in cells:
        raw_data_row = int(cell.get("logical_row", cell.get("row", 0)) or 0)
        if raw_data_row <= 0:
            continue
        raw_grid_row_idx = raw_data_row + data_start_row
        semantic_data_row = semantic_data_row_by_raw_grid_row.get(raw_grid_row_idx)
        if semantic_data_row is None:
            continue
        cell_copy = dict(cell)
        cell_copy["logical_row"] = semantic_data_row
        cell_copy["row"] = semantic_data_row
        semantic_cells.append(cell_copy)

    semantic_row_texts = _build_row_texts(semantic_grid)
    data_row_texts = _build_row_texts(data_grid)
    return semantic_grid, semantic_cells, semantic_row_texts, data_grid, data_row_texts, structural_empty_rows


def _project_grid_from_cells(
    grid: list[list[str | None]],
    cells: list[dict[str, Any]],
    *,
    data_start_row: int,
) -> list[list[str | None]]:
    projected = [list(row) for row in grid]
    if not projected:
        return projected
    for cell in cells:
        if not str(cell.get("inference_reason") or "").startswith("filename_path_text_layer_reconstruction"):
            continue
        data_row_idx = int(cell.get("logical_row", cell.get("row", 0)) or 0)
        row_idx = data_start_row + data_row_idx - 1
        col_idx = int(cell.get("logical_col", cell.get("col", 1)) or 1) - 1
        if row_idx < 0 or row_idx >= len(projected) or col_idx < 0:
            continue
        while col_idx >= len(projected[row_idx]):
            projected[row_idx].append(None)
        text = cell.get("text")
        if text is None:
            continue
        projected[row_idx][col_idx] = str(text)
    return projected


def _build_column_signature(normalized: NormalizedTable) -> list[float]:
    """构建列签名"""
    col_count = normalized.logical_col_count
    return [round((i + 0.5) / col_count, 3) for i in range(col_count)]


def _compute_column_hash(signature: list[float]) -> str:
    """计算列签名哈希"""
    import hashlib
    sig = ",".join(f"{v:.3f}" for v in signature)
    return hashlib.sha1(sig.encode()).hexdigest()[:16]


def _clean_text(text: str | None) -> str | None:
    """清理文本"""
    if not text:
        return None
    cleaned = text.strip().replace("\n", " ")
    return cleaned if cleaned else None


# ============================================================================
# Edge Row Pruning (整合自 validation.py)
# ============================================================================

def is_heading_like_row_text(text: str) -> bool:
    """检查文本是否像表格标题行"""
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    return bool(
        re.match(r"^\d+(?:\.\d+)*\s*[\u4e00-\u9fffA-Za-z]{1,20}表$", cleaned)
        or re.match(r"^表\s*[0-9A-Za-z一二三四五六七八九十零〇.\-]+\b", cleaned)
        or "术语表" in cleaned
    )


def is_section_heading_like_tail(words: list[Any]) -> bool:
    """检查单词列表是否像章节标题尾部"""
    if not words:
        return False
    parts = [w.text if hasattr(w, 'text') else str(w) for w in words]
    if len(parts) == 1:
        return bool(re.match(r"^\d+(?:\.\d+){1,4}$", parts[0]))
    if len(parts) == 2 and re.match(r"^\d+(?:\.\d+){1,4}$", parts[0]):
        return len(parts[1]) <= 12 and bool(re.search(r"[\u4e00-\u9fffA-Za-z]", parts[1]))
    merged = " ".join(parts)
    return bool(re.match(r"^\d+(?:\.\d+){1,4}\s+\S+$", merged)) and len(parts) <= 3


def prune_non_table_edge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """修剪表格边缘的非表格行

    此函数对于过滤以下内容至关重要：
    - 章节标题如 "6. 术语表"
    - 引入表格的叙述性行
    - 表格标题行
    - 章节标题尾部如 "2.3 序列信息"

    Args:
        rows: 包含 'words' 和 'bbox' 键的行字典列表

    Returns:
        修剪后的行列表
    """
    if len(rows) < 2:
        return rows

    pruned = list(rows)

    # 删除开头的表格标题行如 "6. 术语表"
    while len(pruned) >= 2:
        first_words = pruned[0]["words"]
        second_words = pruned[1]["words"]
        first_text = _row_text_from_words(first_words)
        if len(first_words) <= 3 and len(second_words) >= 2 and is_heading_like_row_text(first_text):
            pruned = pruned[1:]
            continue
        break

    # 删除尾部的章节标题行如 "2.3 序列信息"
    while len(pruned) >= 2:
        last_words = pruned[-1]["words"]
        prev_words = pruned[-2]["words"]
        last_bbox = tuple(float(item) for item in pruned[-1]["bbox"])
        prev_bbox = tuple(float(item) for item in pruned[-2]["bbox"])
        vertical_gap = last_bbox[1] - prev_bbox[3]
        avg_height = max(1.0, ((last_bbox[3] - last_bbox[1]) + (prev_bbox[3] - prev_bbox[1])) / 2)
        overlap = _horizontal_overlap_ratio(prev_bbox, last_bbox)
        if is_section_heading_like_tail(last_words) and (
            len(prev_words) >= 2
            or vertical_gap > max(8.0, avg_height * 0.55)
            or overlap < 0.3
        ):
            pruned = pruned[:-1]
            continue
        break

    # 删除开头引入表格的叙述性行和表格标题行
    while len(pruned) >= 2:
        first_text = _row_text_from_words(pruned[0]["words"])
        second_text = _row_text_from_words(pruned[1]["words"])
        if not first_text:
            pruned = pruned[1:]
            continue
        is_intro_line = bool(re.search(r"(如下表|见下表|详见表|见表)", first_text))
        is_caption_line = bool(re.match(r"^\s*(?:附?表\s*[0-9A-Za-z一二三四五六七八九十零〇.\-]+|table\s*[0-9A-Za-z.\-]+)\s*(?:[.:：、\-]|\s+)", first_text, re.IGNORECASE))
        if is_intro_line or is_caption_line:
            pruned = pruned[1:]
            continue
        break

    # 删除尾部引入图片的叙述性行
    while len(pruned) >= 2:
        last_text = _row_text_from_words(pruned[-1]["words"])
        if not last_text:
            pruned = pruned[:-1]
            continue
        if re.search(r"(如下图|见下图|如图|见图)", last_text) or (re.search(r"图\s*\d+", last_text) and "所示" in last_text):
            pruned = pruned[:-1]
            continue
        break

    return pruned


def _row_text_from_words(words: list[Any]) -> str:
    """从单词列表提取文本"""
    return _clean_text(" ".join(word.text if hasattr(word, 'text') else str(word) for word in words))


def _horizontal_overlap_ratio(bbox_a: tuple, bbox_b: tuple) -> float:
    """计算两个 bbox 的水平重叠比例"""
    x_overlap = max(0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]))
    width_a = bbox_a[2] - bbox_a[0]
    width_b = bbox_b[2] - bbox_b[0]
    if width_a <= 0 or width_b <= 0:
        return 0.0
    return x_overlap / min(width_a, width_b)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "TableHeader",
    "TableOpeningStructure",
    "TableInstance",
    "assemble_table_instance",
    "analyze_table_opening_structure",
    # Edge row pruning (整合自 validation.py)
    "is_heading_like_row_text",
    "is_section_heading_like_tail",
    "prune_non_table_edge_rows",
]
