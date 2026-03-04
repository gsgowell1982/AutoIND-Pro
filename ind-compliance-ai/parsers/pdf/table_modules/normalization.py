# Version: v1.2.1
# Optimization Summary:
# - Migrate semantic repair logic to continuum semantic repair module.
# - Execute semantic repairs through unified rule-engine apply path.
# - Keep defaults non-destructive and enterprise-auditable.
# - Delegate semantic orchestration and legacy supplement to continuum modules.
# - Remove local wrapper duplication and call continuum semantic rules directly.
# - Consume continuum unified exports to align with layered package entrypoint.
# - Migrate column inference/mapping logic into continuum column layout utilities.
# - Migrate row projection and column-cluster builders into continuum utilities.

"""Physical Evidence Normalization Layer - 物理证据规范化层

Architecture:
    Raw Objects → Normalization → Normalized Evidence
负责将原始证据规范化为可处理的格式：
- 分析物理列分布，推断逻辑列数
- 构建物理列到逻辑列的映射
- 规范化单元格内容
- 补充遗漏的表格内容
- 处理空值和占位符

关键特性：
- 列聚类：检测哪些物理列应合并为逻辑列
- 列扩展：处理 PyMuPDF 遗漏空列的情况
- 内容补充：从原始 spans 补充遗漏的表格内容（整合自 recovery.py）
- 续表列继承：从父表继承逻辑列数

整合说明：
    本模块整合了原 recovery.py 的核心功能：
    - supplement_table_from_page_text() → _supplement_from_page_text()
    - merge_pymupdf_columns() → 整合到 _normalize_rows_and_cells()
    - detect_continuation_table() → 整合到 _detect_continuation()
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .raw_objects import (
    RawTableEvidence,
    RawCell,
    RawRow,
    RawSpan,
    RawChar,
    RawDrawing,
    DrawingType,
)
from ..settings import get_pdf_parser_settings
from .continuum import (
    supplement_missing_content as _legacy_supplement_missing_content,
    build_default_semantic_rules,
    run_apply_semantic_rules,
    run_shadow_semantic_rules,
    infer_logical_column_count as _column_infer_logical_column_count,
    analyze_content_distribution as _column_analyze_content_distribution,
    analyze_column_clustering as _column_analyze_column_clustering,
    build_column_mapping as _column_build_column_mapping,
    build_column_mapping_with_parent_bbox as _column_build_column_mapping_with_parent_bbox,
    build_column_clusters as _projection_build_column_clusters,
    normalize_rows_and_cells as _projection_normalize_rows_and_cells,
    recover_key_identifier_cells as _rule_recover_key_identifier_cells,
    repair_directory_listing_structure as _rule_repair_directory_listing_structure,
    merge_filename_continuations as _rule_merge_filename_continuations,
)


@dataclass
class ColumnCluster:
    """列聚类
    
    表示一组应合并为同一逻辑列的物理列。
    """
    logical_col: int  # 逻辑列索引 (0-based)
    physical_cols: list[int]  # 物理列索引列表
    center_x: float = 0.0  # 逻辑列的 x 中心位置
    
    @property
    def physical_colspan(self) -> int:
        """跨物理列数"""
        return len(self.physical_cols)


@dataclass
class NormalizedCell:
    """规范化单元格
    
    已完成物理列到逻辑列映射的单元格。
    """
    logical_row: int  # 逻辑行索引 (0-based)
    logical_col: int  # 逻辑列索引 (0-based)
    physical_row: int  # 原始物理行索引
    physical_col_start: int  # 原始物理列起始索引
    physical_col_end: int  # 原始物理列结束索引
    physical_colspan: int  # 跨物理列数
    text: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    supplemented: bool = False  # 是否为补充内容
    supplement_reason: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_row": self.logical_row,
            "logical_col": self.logical_col,
            "physical_row": self.physical_row,
            "physical_col_start": self.physical_col_start,
            "physical_col_end": self.physical_col_end,
            "physical_colspan": self.physical_colspan,
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox else None,
            "supplemented": self.supplemented,
            "supplement_reason": self.supplement_reason,
        }


@dataclass
class NormalizedRow:
    """规范化行"""
    logical_row: int
    physical_row: int
    cells: list[NormalizedCell] = field(default_factory=list)
    
    @property
    def non_empty_count(self) -> int:
        return sum(1 for c in self.cells if c.text and c.text.strip())


@dataclass
class NormalizedTable:
    """规范化表格
    
    完成物理到逻辑映射的表格数据。
    """
    # 基础信息
    page_number: int
    bbox: tuple[float, float, float, float]
    
    # 逻辑结构 (规范化后)
    logical_col_count: int
    logical_row_count: int
    
    # 物理结构 (原始)
    physical_col_count: int
    physical_row_count: int
    
    # 列映射
    column_clusters: list[ColumnCluster] = field(default_factory=list)
    column_mapping: list[list[int]] = field(default_factory=list)  # logical_col -> [physical_cols]
    
    # 规范化数据
    rows: list[NormalizedRow] = field(default_factory=list)
    grid: list[list[str | None]] = field(default_factory=list)  # 2D grid
    
    # 补充数据统计
    supplemented_cells_count: int = 0
    missing_content_candidates_count: int = 0
    missing_content_candidates: list[dict[str, Any]] = field(default_factory=list)
    supplement_writeback_enabled: bool = False
    semantic_rule_engine: dict[str, Any] | None = None
    
    # 绘图分析结果
    grid_line_score: float = 0.0
    horizontal_lines_count: int = 0
    vertical_lines_count: int = 0
    
    # 页面上下文
    page_height: float = 0.0
    near_page_top: bool = False
    near_page_bottom: bool = False
    
    # 元数据
    source: str = "pymupdf_builtin"
    normalization_strategy: str = "column_clustering"  # 或 "inherit", "expand"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "bbox": list(self.bbox),
            "logical_col_count": self.logical_col_count,
            "logical_row_count": self.logical_row_count,
            "physical_col_count": self.physical_col_count,
            "physical_row_count": self.physical_row_count,
            "column_mapping": self.column_mapping,
            "supplemented_cells": self.supplemented_cells_count,
            "missing_content_candidates_count": self.missing_content_candidates_count,
            "supplement_writeback_enabled": self.supplement_writeback_enabled,
            "semantic_rule_engine": self.semantic_rule_engine,
            "grid_line_score": self.grid_line_score,
            "horizontal_lines": self.horizontal_lines_count,
            "vertical_lines": self.vertical_lines_count,
            "page_height": self.page_height,
            "near_page_top": self.near_page_top,
            "near_page_bottom": self.near_page_bottom,
            "source": self.source,
            "normalization_strategy": self.normalization_strategy,
        }


# ============================================================================
# Normalization Functions
# ============================================================================

def normalize_raw_evidence(
    raw_evidence: RawTableEvidence,
    parent_col_count: int | None = None,
    parent_bbox: tuple[float, float, float, float] | None = None,
) -> NormalizedTable:
    """规范化原始表格证据
    
    核心算法：
    1. 分析绘图元素，计算网格分数
    2. 推断逻辑列数
    3. 构建物理列到逻辑列的映射
    4. 规范化所有单元格
    5. 补充遗漏的表格内容
    6. 检测并修复重复值问题（合并单元格场景）
    
    Args:
        raw_evidence: 原始表格证据
        parent_col_count: 父表逻辑列数 (用于续表)
        parent_bbox: 父表边界框 (用于续表列边界继承)
        
    Returns:
        NormalizedTable 规范化表格
    """
    # Step 1: 分析绘图元素
    grid_score, h_lines, v_lines = _analyze_drawings(raw_evidence)
    
    # Step 2: 推断逻辑列数
    logical_col_count, strategy = _infer_logical_column_count(
        raw_evidence, parent_col_count, grid_score
    )
    
    # Step 3: 构建列映射（传递 row_patterns 用于内容列聚类）
    # 对于续表，传递 parent_bbox 进行精确的列映射
    column_mapping = _build_column_mapping(
        raw_evidence.physical_col_count,
        logical_col_count,
        raw_evidence.row_patterns,
        raw_evidence.bbox,
        parent_bbox,
    )
    
    # Step 4: 构建列聚类
    column_clusters = _build_column_clusters(
        column_mapping,
        raw_evidence.bbox,
        raw_evidence.physical_col_count,
    )
    
    # Step 5: 规范化行和单元格
    rows, grid = _normalize_rows_and_cells(
        raw_evidence,
        column_mapping,
        logical_col_count,
    )
    
    # Step 6: semantic repairs through unified rule-engine (apply mode)
    semantic_repair_count = _apply_semantic_rule_engine(
        rows=rows,
        grid=grid,
        raw_evidence=raw_evidence,
        logical_col_count=logical_col_count,
    )

    # Optional legacy supplement (default off for non-destructive enterprise mode)
    table_policy = get_pdf_parser_settings().table_content_policy
    supplemented_count = 0
    if table_policy.enable_supplement_writeback:
        supplemented_count = _supplement_missing_content(
            rows, grid, raw_evidence, column_mapping, logical_col_count, parent_bbox
        )
    semantic_rule_engine_report = _run_semantic_rule_engine_shadow(
        rows=rows,
        grid=grid,
        raw_evidence=raw_evidence,
        logical_col_count=logical_col_count,
    )
    
    # Step 7: 检测并修复重复值问题（合并单元格场景）
    # 当 PyMuPDF 把同一内容放到多个物理列时，可能导致重复值
    _detect_and_fix_duplicate_values(rows, grid, logical_col_count)
    
    # 构建规范化表格
    normalized = NormalizedTable(
        page_number=raw_evidence.page_number,
        bbox=raw_evidence.bbox,
        logical_col_count=logical_col_count,
        logical_row_count=len(rows),
        physical_col_count=raw_evidence.physical_col_count,
        physical_row_count=raw_evidence.physical_row_count,
        column_clusters=column_clusters,
        column_mapping=column_mapping,
        rows=rows,
        grid=grid,
        supplemented_cells_count=supplemented_count + semantic_repair_count,
        missing_content_candidates_count=0,
        missing_content_candidates=[],
        supplement_writeback_enabled=table_policy.enable_supplement_writeback,
        semantic_rule_engine=semantic_rule_engine_report,
        grid_line_score=grid_score,
        horizontal_lines_count=len(h_lines),
        vertical_lines_count=len(v_lines),
        page_height=raw_evidence.page_height,
        near_page_top=raw_evidence.near_page_top,
        near_page_bottom=raw_evidence.near_page_bottom,
        source=raw_evidence.source,
        normalization_strategy=strategy,
    )
    
    return normalized


def _analyze_drawings(
    raw_evidence: RawTableEvidence,
) -> tuple[float, list[RawDrawing], list[RawDrawing]]:
    """分析绘图元素
    
    Returns:
        (grid_score, horizontal_lines, vertical_lines)
    """
    drawings = raw_evidence.drawings
    
    # 获取水平和垂直线
    h_lines = raw_evidence.horizontal_lines
    v_lines = raw_evidence.vertical_lines
    
    # 计算网格分数
    # 基于线条数量和分布
    table_area = (
        (raw_evidence.bbox[2] - raw_evidence.bbox[0]) *
        (raw_evidence.bbox[3] - raw_evidence.bbox[1])
    )
    
    if table_area <= 0:
        return 0.0, h_lines, v_lines
    
    # 网格密度分数
    line_count = len(h_lines) + len(v_lines)
    density_score = min(1.0, line_count / 10.0)
    
    # 均匀分布分数
    distribution_score = 0.0
    if h_lines and v_lines:
        # 检查垂直线是否均匀分布
        v_positions = sorted([d.center[0] for d in v_lines])
        if len(v_positions) >= 2:
            gaps = [v_positions[i+1] - v_positions[i] for i in range(len(v_positions)-1)]
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
                cv = (variance ** 0.5) / avg_gap if avg_gap > 0 else 1.0
                distribution_score = max(0, 1.0 - cv)
    
    grid_score = 0.6 * density_score + 0.4 * distribution_score
    
    return grid_score, h_lines, v_lines


def _infer_logical_column_count(
    raw_evidence: RawTableEvidence,
    parent_col_count: int | None,
    grid_score: float,
) -> tuple[int, str]:
    return _column_infer_logical_column_count(raw_evidence, parent_col_count, grid_score)


def _analyze_content_distribution(
    raw_evidence: RawTableEvidence,
) -> dict[str, Any]:
    return _column_analyze_content_distribution(raw_evidence)


def _analyze_column_clustering(
    row_patterns: list[tuple[int, ...]],
    physical_col_count: int,
) -> dict[str, Any]:
    return _column_analyze_column_clustering(row_patterns, physical_col_count)


def _build_column_mapping(
    physical_col_count: int,
    logical_col_count: int,
    row_patterns: list[tuple[int, ...]] | None = None,
    table_bbox: tuple[float, float, float, float] | None = None,
    parent_bbox: tuple[float, float, float, float] | None = None,
) -> list[list[int]]:
    return _column_build_column_mapping(
        physical_col_count,
        logical_col_count,
        row_patterns,
        table_bbox,
        parent_bbox,
    )


def _build_column_mapping_with_parent_bbox(
    physical_col_count: int,
    logical_col_count: int,
    table_bbox: tuple[float, float, float, float],
    parent_bbox: tuple[float, float, float, float],
) -> list[list[int]]:
    return _column_build_column_mapping_with_parent_bbox(
        physical_col_count,
        logical_col_count,
        table_bbox,
        parent_bbox,
    )


def _build_column_clusters(
    column_mapping: list[list[int]],
    table_bbox: tuple[float, float, float, float],
    physical_col_count: int,
) -> list[ColumnCluster]:
    return _projection_build_column_clusters(
        column_mapping=column_mapping,
        table_bbox=table_bbox,
        physical_col_count=physical_col_count,
        cluster_cls=ColumnCluster,
    )


def _normalize_rows_and_cells(
    raw_evidence: RawTableEvidence,
    column_mapping: list[list[int]],
    logical_col_count: int,
) -> tuple[list[NormalizedRow], list[list[str | None]]]:
    rows_any, grid = _projection_normalize_rows_and_cells(
        raw_evidence=raw_evidence,
        column_mapping=column_mapping,
        logical_col_count=logical_col_count,
        cell_cls=NormalizedCell,
        row_cls=NormalizedRow,
    )
    return rows_any, grid


def _supplement_missing_content(
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    raw_evidence: RawTableEvidence,
    column_mapping: list[list[int]],
    logical_col_count: int,
    parent_bbox: tuple[float, float, float, float] | None = None,
) -> int:
    _ = column_mapping
    return _legacy_supplement_missing_content(
        rows=rows,
        grid=grid,
        raw_evidence=raw_evidence,
        logical_col_count=logical_col_count,
        parent_bbox=parent_bbox,
    )


def _detect_and_fix_duplicate_values(
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    logical_col_count: int,
) -> None:
    """Non-destructive duplicate handling for enterprise IND safety."""
    _ = rows
    _ = grid
    _ = logical_col_count
    return


def _apply_semantic_rule_engine(
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    raw_evidence: RawTableEvidence,
    logical_col_count: int,
) -> int:
    rules = build_default_semantic_rules(
        recover_key_identifier_cells_fn=_rule_recover_key_identifier_cells,
        repair_directory_listing_structure_fn=_rule_repair_directory_listing_structure,
        merge_filename_continuations_fn=_rule_merge_filename_continuations,
    )
    return run_apply_semantic_rules(
        rows=rows,
        grid=grid,
        raw_evidence=raw_evidence,
        logical_col_count=logical_col_count,
        rules=rules,
    )


def _run_semantic_rule_engine_shadow(
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    raw_evidence: RawTableEvidence,
    logical_col_count: int,
) -> dict[str, Any] | None:
    policy = get_pdf_parser_settings().rule_engine_policy
    if not policy.enabled:
        return None

    rules = build_default_semantic_rules(
        recover_key_identifier_cells_fn=_rule_recover_key_identifier_cells,
        repair_directory_listing_structure_fn=_rule_repair_directory_listing_structure,
        merge_filename_continuations_fn=_rule_merge_filename_continuations,
    )
    return run_shadow_semantic_rules(
        rows=rows,
        grid=grid,
        raw_evidence=raw_evidence,
        logical_col_count=logical_col_count,
        rules=rules,
        emit_diagnostics=policy.emit_diagnostics,
    )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "ColumnCluster",
    "NormalizedCell",
    "NormalizedRow",
    "NormalizedTable",
    "normalize_raw_evidence",
]






