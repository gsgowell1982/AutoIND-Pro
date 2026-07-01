# Version: v1.2.2
# Optimization Summary:
# - Migrate semantic repair logic to continuum semantic repair module.
# - Execute semantic repairs through unified rule-engine apply path.
# - Keep defaults non-destructive and enterprise-auditable.
# - Delegate semantic orchestration and legacy supplement to continuum modules.
# - Remove local wrapper duplication and call continuum semantic rules directly.
# - Consume continuum unified exports to align with layered package entrypoint.
# - Migrate column inference/mapping logic into continuum column layout utilities.
# - Migrate row projection and column-cluster builders into continuum utilities.
# - Anchor sparse continuation column mapping to observed cell geometry before any
#   bbox-based fallback.
# - Reuse parent structural skeleton even when only parent_bbox is available.

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

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from .raw_objects import (
    RawTableEvidence,
    RawCell,
    RawRow,
    RawSpan,
    RawWord,
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
    reconstruct_filename_path_cells_from_text_layer as _rule_reconstruct_filename_path_cells_from_text_layer,
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
    
    # 结构校验信息 (v1.3.0 新增)
    structure_validation: Any = None  # StructureValidation 类型，避免循环导入

    def to_dict(self) -> dict[str, Any]:
        result = {
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
        
        # 添加结构校验信息
        if self.structure_validation:
            result["structure_validation"] = self.structure_validation.to_dict()
        
        return result


# ============================================================================
# Normalization Functions
# ============================================================================

def normalize_raw_evidence(
    raw_evidence: RawTableEvidence,
    parent_col_count: int | None = None,
    parent_bbox: tuple[float, float, float, float] | None = None,
    parent_column_boundaries: list[float] | None = None,
    assessment: Any = None,  # ContinuationAssessment 类型，避免循环导入
) -> NormalizedTable:
    """规范化原始表格证据 - 结构继承优先版本 (v1.3.0)

    核心算法：
    1. 分析绘图元素，计算网格分数
    2. 推断逻辑列数 - 优先继承父表结构
    3. 构建物理列到逻辑列的映射 - 基于父表列边界
    4. 规范化所有单元格
    5. 补充遗漏的表格内容
    6. 检测并修复重复值问题（合并单元格场景）
    7. 记录结构校验信息

    Args:
        raw_evidence: 原始表格证据
        parent_col_count: 父表逻辑列数 (用于续表)
        parent_bbox: 父表边界框 (用于续表列边界继承)
        parent_column_boundaries: 父表列边界 (用于精确列映射)
        assessment: Step2 的评估结果

    Returns:
        NormalizedTable 规范化表格
    """
    # 导入 StructureValidation（延迟导入避免循环依赖）
    from ..types import StructureValidation
    
    # 初始化结构校验信息
    validation = StructureValidation()
    validation.original_physical_col_count = raw_evidence.physical_col_count
    
    # Step 1: 分析绘图元素
    grid_score, h_lines, v_lines = _analyze_drawings(raw_evidence)

    # Step 2: 推断逻辑列数 - 优先继承父表结构
    if parent_col_count is not None and parent_col_count > 0:
        validation.col_count_inherited = True
        validation.parent_col_count = parent_col_count
        logical_col_count = parent_col_count
        strategy = "inherit"
        
        # 校验继承合理性
        max_content_cols = raw_evidence.max_content_cols_per_row
        validation.inferred_col_count = max_content_cols
        
        if max_content_cols > parent_col_count + 1:
            if assessment and hasattr(assessment, 'is_continuation') and assessment.is_continuation:
                # 信任评估结果，保持继承
                validation.col_count_match = True
            else:
                # 降级为内容推断
                logical_col_count, strategy = _infer_logical_column_count(
                    raw_evidence, None, grid_score
                )
                validation.col_count_match = (logical_col_count == parent_col_count)
                validation.structure_adjusted = True
                validation.adjustment_reason = "content_cols_exceed_parent"
        else:
            validation.col_count_match = True
    else:
        validation.col_count_inherited = False
        logical_col_count, strategy = _infer_logical_column_count(
            raw_evidence, parent_col_count, grid_score
        )
        validation.inferred_col_count = logical_col_count
    
    validation.final_col_count = logical_col_count

    # Step 3: 构建列映射 - 父表结构骨架优先
    if parent_bbox and logical_col_count > 0:
        column_mapping = _build_column_mapping_from_parent_boundaries(
            raw_evidence=raw_evidence,
            parent_bbox=parent_bbox,
            parent_column_boundaries=parent_column_boundaries,
            logical_col_count=logical_col_count,
        )
        validation.column_mapping_method = (
            "parent_boundaries" if parent_column_boundaries else "parent_bbox"
        )
        validation.column_mapping_confidence = _calculate_mapping_confidence(
            column_mapping, logical_col_count
        )
    else:
        column_mapping = _build_column_mapping(
            raw_evidence.physical_col_count,
            logical_col_count,
            raw_evidence.row_patterns,
            raw_evidence.bbox,
            None,
        )
        validation.column_mapping_method = "content_clustering"
        validation.column_mapping_confidence = 0.5
    
    # 检查空列
    validation.empty_columns = [
        i for i, mapping in enumerate(column_mapping) if not mapping
    ]
    
    if validation.empty_columns:
        validation.validation_warnings.append(
            f"存在空列: {validation.empty_columns}"
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
    visual_row_rebuild_count = _rebuild_sparse_ruled_table_rows_from_word_geometry(
        raw_evidence=raw_evidence,
        rows=rows,
        grid=grid,
        logical_col_count=logical_col_count,
    )
    if parent_column_boundaries and len(parent_column_boundaries) >= logical_col_count + 1:
        _split_spanning_identifier_cells_from_words(
            raw_evidence=raw_evidence,
            rows=rows,
            grid=grid,
            parent_boundaries=parent_column_boundaries,
            logical_col_count=logical_col_count,
        )
        _supplement_empty_grid_cells_from_words(
            raw_evidence=raw_evidence,
            rows=rows,
            grid=grid,
            parent_boundaries=parent_column_boundaries,
            logical_col_count=logical_col_count,
        )
    grouped_header_report = _project_dense_leaf_table_header_groups(
        raw_evidence=raw_evidence,
        rows=rows,
        grid=grid,
        logical_col_count=logical_col_count,
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
    
    # Step 8: 校验结果
    validation.validation_passed = (
        logical_col_count > 0 and
        len(validation.empty_columns) == 0 and
        validation.column_mapping_confidence >= 0.5
    )
    
    # 记录 bbox 修正信息
    if hasattr(raw_evidence, 'bbox_correction') and raw_evidence.bbox_correction:
        validation.bbox_corrected = True
        validation.original_bbox = tuple(raw_evidence.bbox_correction.get("original_bbox", (0, 0, 0, 0)))
        validation.corrected_bbox = tuple(raw_evidence.bbox_correction.get("corrected_bbox", (0, 0, 0, 0)))

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
        supplemented_cells_count=supplemented_count + semantic_repair_count + visual_row_rebuild_count + (1 if grouped_header_report else 0),
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
        structure_validation=validation,  # 新增
    )
    if grouped_header_report:
        if normalized.semantic_rule_engine is None:
            normalized.semantic_rule_engine = {}
        normalized.semantic_rule_engine["dense_leaf_header_groups"] = grouped_header_report

    return normalized


def _rebuild_sparse_ruled_table_rows_from_word_geometry(
    *,
    raw_evidence: RawTableEvidence,
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    logical_col_count: int,
) -> int:
    """Rebuild visual data rows when a ruled table lacks row separators.

    PyMuPDF often collapses sparse ruled tables into a header row plus one
    multi-line data row. When the table has stable column evidence and the words
    inside the bbox form more visual y-rows than the physical grid, use word
    geometry as the display/data grid while preserving raw evidence upstream.
    """
    if logical_col_count < 2 or not raw_evidence.words or not rows or not grid:
        return 0
    if raw_evidence.physical_row_count > 3:
        return 0
    if not _has_sparse_ruled_visual_row_shape(raw_evidence, grid, logical_col_count):
        return 0

    column_boundaries = _infer_current_table_column_boundaries(
        raw_evidence=raw_evidence,
        logical_col_count=logical_col_count,
    )
    if len(column_boundaries) < logical_col_count + 1:
        return 0

    visual_word_rows = _cluster_table_words_into_visual_rows(raw_evidence.words)
    if len(visual_word_rows) <= len(grid) + 1:
        return 0

    header_limit_y = _estimate_sparse_table_header_bottom(raw_evidence, grid)
    data_word_rows = [
        word_row
        for word_row in visual_word_rows
        if _word_row_center_y(word_row) > header_limit_y + 1.0
    ]
    if len(data_word_rows) <= max(1, len(grid)):
        return 0

    rebuilt_pairs: list[tuple[list[str | None], list[RawWord]]] = []
    for word_row in data_word_rows:
        projected = _project_visual_word_row_to_columns(
            word_row=word_row,
            column_boundaries=column_boundaries,
            logical_col_count=logical_col_count,
        )
        if _visual_projected_row_is_note(projected, logical_col_count):
            rebuilt_pairs.append((projected, word_row))
            continue
        if sum(1 for value in projected if str(value or "").strip()) == 0:
            continue
        rebuilt_pairs.append((projected, word_row))

    header_row = _project_sparse_table_header_row(
        raw_evidence=raw_evidence,
        column_boundaries=column_boundaries,
        logical_col_count=logical_col_count,
    )
    rebuilt_grid = ([header_row] if header_row else []) + [row for row, _word_row in rebuilt_pairs]
    if len(rebuilt_grid) <= len(grid):
        return 0
    if not _rebuilt_grid_is_better_sparse_projection(grid, rebuilt_grid, logical_col_count):
        return 0

    _replace_normalized_rows_with_visual_grid(
        rows=rows,
        grid=grid,
        rebuilt_grid=rebuilt_grid,
        source_word_rows=(
            [_header_words_for_sparse_table(raw_evidence, header_row)] if header_row else []
        ) + [word_row for _row, word_row in rebuilt_pairs],
        column_boundaries=column_boundaries,
        logical_col_count=logical_col_count,
    )
    return sum(1 for row in rebuilt_grid for value in row if str(value or "").strip())


def _project_sparse_table_header_row(
    *,
    raw_evidence: RawTableEvidence,
    column_boundaries: list[float],
    logical_col_count: int,
) -> list[str | None]:
    if not raw_evidence.rows:
        return []
    first_row = raw_evidence.rows[0]
    header_cells = [cell for cell in first_row.cells if str(cell.text or "").strip()]
    if header_cells:
        projected: list[str | None] = [None] * logical_col_count
        for cell in header_cells:
            x0, _y0, x1, _y1 = cell.bbox
            logical_col = _assign_center_to_parent_skeleton(
                center_x=(float(x0) + float(x1)) / 2.0,
                parent_boundaries=column_boundaries,
                logical_col_count=logical_col_count,
            )
            if 0 <= logical_col < logical_col_count:
                text = _clean_cell_text(cell.text)
                if text:
                    existing = str(projected[logical_col] or "").strip()
                    projected[logical_col] = f"{existing} {text}".strip() if existing else text
        if sum(1 for value in projected if str(value or "").strip()) >= min(2, logical_col_count):
            return projected

    header_limit_y = _estimate_sparse_table_header_bottom(raw_evidence, [])
    header_words = [
        word for word in raw_evidence.words
        if float(word.y_center) <= header_limit_y + 1.0
    ]
    if not header_words:
        return []
    projected = _project_visual_word_row_to_columns(
        word_row=sorted(header_words, key=lambda item: float(item.x0)),
        column_boundaries=column_boundaries,
        logical_col_count=logical_col_count,
    )
    if sum(1 for value in projected if str(value or "").strip()) >= min(2, logical_col_count):
        return projected
    return []


def _header_words_for_sparse_table(
    raw_evidence: RawTableEvidence,
    header_row: list[str | None],
) -> list[RawWord]:
    if not header_row:
        return []
    header_limit_y = _estimate_sparse_table_header_bottom(raw_evidence, [])
    return [
        word for word in raw_evidence.words
        if float(word.y_center) <= header_limit_y + 1.0
    ]


def _clean_cell_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return _normalize_merged_word_spacing(text)


def _has_sparse_ruled_visual_row_shape(
    raw_evidence: RawTableEvidence,
    grid: list[list[str | None]],
    logical_col_count: int,
) -> bool:
    v_lines = len(raw_evidence.vertical_lines)
    unique_h_line_count = len(
        _cluster_numeric_positions(
            [float(line.center[1]) for line in raw_evidence.horizontal_lines],
            tolerance=2.0,
        )
    )
    if v_lines < max(2, logical_col_count - 1):
        return False
    if unique_h_line_count > max(4, raw_evidence.physical_row_count + 3):
        return False
    multiline_cells = 0
    max_lines = 0
    for row in grid:
        for cell in row:
            lines = [line.strip() for line in str(cell or "").splitlines() if line.strip()]
            if len(lines) >= 3:
                multiline_cells += 1
            max_lines = max(max_lines, len(lines))
    return multiline_cells >= 2 and max_lines >= 4


def _infer_current_table_column_boundaries(
    *,
    raw_evidence: RawTableEvidence,
    logical_col_count: int,
) -> list[float]:
    x0, _, x1, _ = raw_evidence.bbox
    if x1 <= x0 or logical_col_count <= 0:
        return []

    vertical_positions = sorted(
        float(line.center[0])
        for line in raw_evidence.vertical_lines
        if x0 - 2.0 <= float(line.center[0]) <= x1 + 2.0
    )
    clustered_positions = _cluster_numeric_positions(vertical_positions, tolerance=2.0)
    if len(clustered_positions) >= logical_col_count + 1:
        candidates = _choose_column_boundaries_from_positions(
            positions=clustered_positions,
            table_x0=x0,
            table_x1=x1,
            logical_col_count=logical_col_count,
        )
        if len(candidates) >= logical_col_count + 1:
            return candidates

    word_centers_by_col = _cluster_word_x_centers(raw_evidence.words, logical_col_count)
    if len(word_centers_by_col) == logical_col_count:
        boundaries = [x0]
        for left_values, right_values in zip(word_centers_by_col, word_centers_by_col[1:]):
            boundaries.append((max(left_values) + min(right_values)) / 2)
        boundaries.append(x1)
        if len(boundaries) == logical_col_count + 1:
            return boundaries

    return [
        x0 + index * (x1 - x0) / logical_col_count
        for index in range(logical_col_count + 1)
    ]


def _cluster_numeric_positions(values: list[float], *, tolerance: float) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - statistics.median(clusters[-1])) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(statistics.median(cluster)) for cluster in clusters if cluster]


def _choose_column_boundaries_from_positions(
    *,
    positions: list[float],
    table_x0: float,
    table_x1: float,
    logical_col_count: int,
) -> list[float]:
    needed = logical_col_count + 1
    if len(positions) < needed:
        return []
    if len(positions) == needed:
        return positions

    best: list[float] = []
    best_score = float("inf")
    for start in range(0, len(positions) - needed + 1):
        candidate = positions[start : start + needed]
        span_penalty = abs(candidate[0] - table_x0) + abs(candidate[-1] - table_x1)
        gaps = [candidate[idx + 1] - candidate[idx] for idx in range(len(candidate) - 1)]
        positive_gaps = [gap for gap in gaps if gap > 0]
        if len(positive_gaps) != logical_col_count:
            continue
        median_gap = statistics.median(positive_gaps)
        gap_variance = sum((gap - median_gap) ** 2 for gap in positive_gaps) / len(positive_gaps)
        score = span_penalty + gap_variance * 0.02
        if score < best_score:
            best_score = score
            best = list(candidate)
    return best


def _cluster_word_x_centers(words: list[RawWord], logical_col_count: int) -> list[list[float]]:
    centers = sorted(float(word.x_center) for word in words if str(word.text or "").strip())
    if len(centers) < logical_col_count:
        return []
    clusters = [[center] for center in centers[:logical_col_count]]
    for center in centers[logical_col_count:]:
        nearest_idx = min(
            range(len(clusters)),
            key=lambda idx: abs(center - statistics.median(clusters[idx])),
        )
        clusters[nearest_idx].append(center)
    ordered = sorted(clusters, key=lambda cluster: statistics.median(cluster))
    if any(not cluster for cluster in ordered):
        return []
    return ordered


def _cluster_table_words_into_visual_rows(words: list[RawWord]) -> list[list[RawWord]]:
    valid_words = [word for word in words if str(word.text or "").strip()]
    if not valid_words:
        return []
    heights = [max(1.0, float(word.y1) - float(word.y0)) for word in valid_words]
    median_height = statistics.median(heights) if heights else 10.0
    tolerance = max(3.0, median_height * 0.55)
    rows: list[list[RawWord]] = []
    for word in sorted(valid_words, key=lambda item: (float(item.y_center), float(item.x0))):
        center_y = float(word.y_center)
        if not rows or abs(center_y - _word_row_center_y(rows[-1])) > tolerance:
            rows.append([word])
        else:
            rows[-1].append(word)
    return [sorted(row, key=lambda item: float(item.x0)) for row in rows]


def _word_row_center_y(word_row: list[RawWord]) -> float:
    if not word_row:
        return 0.0
    return float(statistics.median(float(word.y_center) for word in word_row))


def _estimate_sparse_table_header_bottom(
    raw_evidence: RawTableEvidence,
    grid: list[list[str | None]],
) -> float:
    first_raw_row = raw_evidence.rows[0] if raw_evidence.rows else None
    if first_raw_row and first_raw_row.bbox:
        return float(first_raw_row.bbox[3])
    header_text = " ".join(str(cell or "") for cell in (grid[0] if grid else []))
    header_tokens = {token for token in re.split(r"\s+", header_text) if token}
    matching_words = [
        word for word in raw_evidence.words
        if str(word.text or "").strip() in header_tokens
    ]
    if matching_words:
        return max(float(word.y1) for word in matching_words)
    return float(raw_evidence.bbox[1])


def _project_visual_word_row_to_columns(
    *,
    word_row: list[RawWord],
    column_boundaries: list[float],
    logical_col_count: int,
) -> list[str | None]:
    words_by_col: dict[int, list[RawWord]] = {idx: [] for idx in range(logical_col_count)}
    for word in word_row:
        logical_col = _assign_center_to_parent_skeleton(
            center_x=float(word.x_center),
            parent_boundaries=column_boundaries,
            logical_col_count=logical_col_count,
        )
        if 0 <= logical_col < logical_col_count:
            words_by_col[logical_col].append(word)
    projected: list[str | None] = []
    for logical_col in range(logical_col_count):
        text = _merge_words_for_cell(words_by_col[logical_col])
        projected.append(text if text else None)
    return projected


def _visual_projected_row_is_note(row: list[str | None], logical_col_count: int) -> bool:
    non_empty = [(idx, str(value or "").strip()) for idx, value in enumerate(row) if str(value or "").strip()]
    if len(non_empty) != 1:
        return False
    idx, text = non_empty[0]
    if idx != 0:
        return False
    return bool(re.match(r"^\s*(?:[*†‡]|\d+\)|注[:：]|note\b)", text, re.IGNORECASE))


def _rebuilt_grid_is_better_sparse_projection(
    current_grid: list[list[str | None]],
    rebuilt_grid: list[list[str | None]],
    logical_col_count: int,
) -> bool:
    current_bad = _grid_embedded_cross_column_text_count(current_grid)
    rebuilt_bad = _grid_embedded_cross_column_text_count(rebuilt_grid)
    current_filled = sum(1 for row in current_grid for value in row if str(value or "").strip())
    rebuilt_filled = sum(1 for row in rebuilt_grid for value in row if str(value or "").strip())
    dense_rebuilt_rows = sum(
        1 for row in rebuilt_grid
        if sum(1 for value in row if str(value or "").strip()) >= min(2, logical_col_count)
    )
    return rebuilt_bad <= current_bad and rebuilt_filled >= current_filled and dense_rebuilt_rows >= 2


def _grid_embedded_cross_column_text_count(grid: list[list[str | None]]) -> int:
    count = 0
    for row in grid:
        for value in row:
            text = str(value or "").strip()
            if not text:
                continue
            if "\n" in text:
                count += 1
            if len(re.findall(r"[\u4e00-\u9fff]{2,}", text)) >= 2 and " " in text:
                count += 1
    return count


def _replace_normalized_rows_with_visual_grid(
    *,
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    rebuilt_grid: list[list[str | None]],
    source_word_rows: list[list[RawWord]],
    column_boundaries: list[float],
    logical_col_count: int,
) -> None:
    grid[:] = [list(row) for row in rebuilt_grid]
    rows[:] = []
    for row_idx, row_values in enumerate(rebuilt_grid):
        word_row = source_word_rows[row_idx] if row_idx < len(source_word_rows) else []
        row_cells: list[NormalizedCell] = []
        row_y0 = min((float(word.y0) for word in word_row), default=0.0)
        row_y1 = max((float(word.y1) for word in word_row), default=0.0)
        for logical_col in range(logical_col_count):
            left = column_boundaries[logical_col] if logical_col < len(column_boundaries) else 0.0
            right = column_boundaries[logical_col + 1] if logical_col + 1 < len(column_boundaries) else left
            col_words = [
                word for word in word_row
                if _assign_center_to_parent_skeleton(
                    center_x=float(word.x_center),
                    parent_boundaries=column_boundaries,
                    logical_col_count=logical_col_count,
                ) == logical_col
            ]
            bbox = None
            if col_words:
                bbox = (
                    min(float(word.x0) for word in col_words),
                    min(float(word.y0) for word in col_words),
                    max(float(word.x1) for word in col_words),
                    max(float(word.y1) for word in col_words),
                )
            elif row_y1 > row_y0 and right > left:
                bbox = (left, row_y0, right, row_y1)
            row_cells.append(
                NormalizedCell(
                    logical_row=row_idx,
                    logical_col=logical_col,
                    physical_row=row_idx,
                    physical_col_start=logical_col,
                    physical_col_end=logical_col,
                    physical_colspan=1,
                    text=row_values[logical_col] if logical_col < len(row_values) else None,
                    bbox=bbox,
                    supplemented=True,
                    supplement_reason="sparse_ruled_visual_word_row_rebuild",
                )
            )
        rows.append(NormalizedRow(logical_row=row_idx, physical_row=row_idx, cells=row_cells))


def _project_dense_leaf_table_header_groups(
    *,
    raw_evidence: RawTableEvidence,
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    logical_col_count: int,
) -> dict[str, Any] | None:
    """Repair grouped headers and unit continuations in dense leaf-column tables."""
    if logical_col_count < 4 or not raw_evidence.words or not rows or not grid:
        return None
    if str(raw_evidence.source or "") != "caption_anchored_horizontal_rules":
        return None

    visual_rows = _cluster_table_words_into_visual_rows(raw_evidence.words)
    if len(visual_rows) < 4:
        return None

    dense_rows = [
        word_row
        for word_row in visual_rows
        if _dense_leaf_numeric_word_count(word_row) >= max(3, logical_col_count - 2)
    ]
    if len(dense_rows) < 2:
        return None

    leaf_anchors = _infer_dense_leaf_anchors_from_visual_rows(dense_rows, logical_col_count)
    if len(leaf_anchors) != logical_col_count:
        return None
    boundaries = _boundaries_from_column_anchors(leaf_anchors, raw_evidence.bbox)
    if len(boundaries) != logical_col_count + 1:
        return None

    projected_rows: list[list[str | None]] = []
    source_rows: list[list[RawWord]] = []
    row_idx = 0
    merged_unit_rows = 0
    while row_idx < len(visual_rows):
        word_row = visual_rows[row_idx]
        projected = _project_visual_word_row_to_columns(
            word_row=word_row,
            column_boundaries=boundaries,
            logical_col_count=logical_col_count,
        )
        if (
            projected_rows
            and _is_stub_unit_continuation_row(projected)
            and _row_has_dense_value_cells(projected_rows[-1])
        ):
            projected_rows[-1][0] = _join_stub_unit_text(
                str(projected_rows[-1][0] or ""),
                str(projected[0] or ""),
            )
            source_rows[-1].extend(word_row)
            merged_unit_rows += 1
            row_idx += 1
            continue
        if projected_rows and _is_stub_header_unit_continuation_row(projected_rows[-1], projected):
            projected_rows[-1][0] = _join_stub_unit_text(
                str(projected_rows[-1][0] or ""),
                str(projected[0] or ""),
            )
            source_rows[-1].extend(word_row)
            merged_unit_rows += 1
            row_idx += 1
            continue
        projected_rows.append(projected)
        source_rows.append(list(word_row))
        row_idx += 1

    if len(projected_rows) >= len(grid):
        row_count_gain = len(projected_rows) - len(grid)
    else:
        row_count_gain = 0
    if merged_unit_rows <= 0 and row_count_gain <= 0:
        return None

    non_empty_counts = [sum(1 for cell in row if _clean_cell_text(cell)) for row in projected_rows]
    if max(non_empty_counts, default=0) < max(3, logical_col_count - 1):
        return None

    header_groups = _infer_header_column_groups_from_projected_rows(
        projected_rows,
        source_rows,
        leaf_anchors,
        max_header_rows=_first_dense_value_row_index(projected_rows),
    )
    _apply_header_group_projection(projected_rows, header_groups)
    _replace_normalized_rows_with_visual_grid(
        rows=rows,
        grid=grid,
        rebuilt_grid=projected_rows,
        source_word_rows=source_rows,
        column_boundaries=boundaries,
        logical_col_count=logical_col_count,
    )
    return {
        "source": "dense_numeric_leaf_columns",
        "column_anchors": [round(value, 2) for value in leaf_anchors],
        "merged_unit_rows": merged_unit_rows,
        "header_column_groups": header_groups,
    }


def _dense_leaf_numeric_word_count(word_row: list[RawWord]) -> int:
    return sum(1 for word in word_row if _looks_like_numeric_cell_text(str(word.text or "")))


def _looks_like_numeric_cell_text(text: str) -> bool:
    cleaned = str(text or "").strip().replace(",", "")
    if not cleaned:
        return False
    if re.fullmatch(r"[<>≤≥~+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?%?", cleaned):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?", cleaned):
        return True
    return False


def _infer_dense_leaf_anchors_from_visual_rows(
    dense_rows: list[list[RawWord]],
    logical_col_count: int,
) -> list[float]:
    import statistics

    numeric_count = logical_col_count - 1
    candidates: list[tuple[float, list[float]]] = []
    for word_row in dense_rows:
        ordered = sorted(word_row, key=lambda item: float(item.x0))
        numeric_words = [word for word in ordered if _looks_like_numeric_cell_text(str(word.text or ""))]
        if len(numeric_words) < numeric_count:
            continue
        numeric_words = numeric_words[:numeric_count]
        label_words = [word for word in ordered if float(word.x1) <= float(numeric_words[0].x0) + 1.0]
        label_words = [word for word in label_words if word not in numeric_words]
        if not label_words:
            continue
        candidates.append(
            (
                float(statistics.median(float(word.x0) for word in label_words)),
                [float(word.x0) for word in numeric_words],
            )
        )
    if len(candidates) < 2:
        return []
    label_anchor = float(statistics.median(item[0] for item in candidates))
    value_anchors = [
        float(statistics.median(item[1][value_idx] for item in candidates))
        for value_idx in range(numeric_count)
    ]
    anchors = [label_anchor] + value_anchors
    if any(right <= left + 4.0 for left, right in zip(anchors, anchors[1:])):
        return []
    return anchors


def _boundaries_from_column_anchors(
    anchors: list[float],
    bbox: tuple[float, float, float, float],
) -> list[float]:
    if not anchors:
        return []
    x0, _y0, x1, _y1 = bbox
    boundaries = [min(float(x0), float(anchors[0]) - 8.0)]
    for left, right in zip(anchors, anchors[1:]):
        boundaries.append((float(left) + float(right)) / 2.0)
    if len(anchors) >= 2:
        tail_gap = max(8.0, float(anchors[-1]) - float(anchors[-2]))
    else:
        tail_gap = 24.0
    boundaries.append(max(float(x1), float(anchors[-1]) + tail_gap * 0.7))
    return boundaries


def _is_stub_unit_continuation_row(row: list[str | None]) -> bool:
    non_empty = [(idx, _clean_cell_text(value)) for idx, value in enumerate(row) if _clean_cell_text(value)]
    if len(non_empty) != 1:
        return False
    idx, text = non_empty[0]
    if idx != 0:
        return False
    if len(text) > 32:
        return False
    if re.fullmatch(r"\([^)]{1,24}\)", text):
        return True
    return bool(re.fullmatch(r"[\[(]?[A-Za-z%μµ·./\-\s]{1,24}[\])]?|[\uff08][^\uff09]{1,24}[\uff09]", text))


def _row_has_dense_value_cells(row: list[str | None]) -> bool:
    return sum(1 for cell in row[1:] if _clean_cell_text(cell)) >= 3


def _is_stub_header_unit_continuation_row(
    previous_row: list[str | None],
    current_row: list[str | None],
) -> bool:
    if not _is_stub_unit_continuation_row(current_row):
        return False
    previous_non_empty = [(idx, _clean_cell_text(value)) for idx, value in enumerate(previous_row) if _clean_cell_text(value)]
    if not previous_non_empty:
        return False
    return previous_non_empty[0][0] == 0 and not _row_has_dense_value_cells(previous_row)


def _join_stub_unit_text(base: str, continuation: str) -> str:
    left = _clean_cell_text(base)
    right = _clean_cell_text(continuation)
    if not left:
        return right
    if not right:
        return left
    if right.startswith(("(", "（")):
        return f"{left}{right}"
    return f"{left} {right}".strip()


def _first_dense_value_row_index(grid: list[list[str | None]]) -> int:
    for row_idx, row in enumerate(grid):
        if _row_has_dense_value_cells(row) and sum(
            1 for cell in row[1:] if _looks_like_numeric_cell_text(_clean_cell_text(cell))
        ) >= 3:
            return row_idx
    return min(len(grid), 3)


def _infer_header_column_groups_from_projected_rows(
    projected_rows: list[list[str | None]],
    source_rows: list[list[RawWord]],
    leaf_anchors: list[float],
    *,
    max_header_rows: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    _ = source_rows
    _ = leaf_anchors
    header_limit = max(0, min(max_header_rows, len(projected_rows)))
    for row_idx in range(header_limit):
        row = projected_rows[row_idx]
        value_label_cols = [
            col_idx
            for col_idx, value in enumerate(row)
            if col_idx >= 1 and _clean_cell_text(value)
        ]
        if not value_label_cols:
            continue
        for label_idx, col_idx in enumerate(value_label_cols):
            text = _clean_cell_text(row[col_idx])
            previous_col = value_label_cols[label_idx - 1] if label_idx > 0 else None
            next_col = value_label_cols[label_idx + 1] if label_idx + 1 < len(value_label_cols) else None
            start_col = 1 if previous_col is None else int((previous_col + col_idx) // 2) + 1
            end_col = len(row) - 1 if next_col is None else int((col_idx + next_col) // 2)
            if end_col <= start_col:
                continue
            groups.append(
                {
                    "row": row_idx,
                    "start_col": start_col,
                    "end_col": end_col,
                    "colspan": end_col - start_col + 1,
                    "text": text,
                    "source": "dense_leaf_header_geometry",
                }
            )
    return groups


def _apply_header_group_projection(
    projected_rows: list[list[str | None]],
    header_groups: list[dict[str, Any]],
) -> None:
    for group in header_groups:
        row_idx = int(group.get("row", -1))
        start_col = int(group.get("start_col", -1))
        end_col = int(group.get("end_col", -1))
        text = _clean_cell_text(group.get("text"))
        if row_idx < 0 or row_idx >= len(projected_rows) or not text:
            continue
        row = projected_rows[row_idx]
        if start_col < 0 or start_col >= len(row):
            continue
        row[start_col] = text
        for col_idx in range(start_col + 1, min(end_col + 1, len(row))):
            row[col_idx] = None


def _build_uniform_column_mapping(
    physical_col_count: int,
    logical_col_count: int,
) -> list[list[int]]:
    """Build a simple left-to-right fallback mapping."""
    mapping: list[list[int]] = [[] for _ in range(max(logical_col_count, 0))]
    if physical_col_count <= 0 or logical_col_count <= 0:
        return mapping

    for physical_col in range(physical_col_count):
        rel_pos = (physical_col + 0.5) / physical_col_count
        logical_col = min(int(rel_pos * logical_col_count), logical_col_count - 1)
        mapping[logical_col].append(physical_col)

    return mapping


def _build_parent_skeleton_boundaries(
    parent_bbox: tuple[float, float, float, float] | None,
    logical_col_count: int,
    explicit_boundaries: list[float] | None,
) -> list[float]:
    """Prefer explicit parent boundaries; otherwise synthesize a logical skeleton."""
    if explicit_boundaries and len(explicit_boundaries) >= 2:
        return list(explicit_boundaries)

    if not parent_bbox or logical_col_count <= 0:
        return []

    parent_width = parent_bbox[2] - parent_bbox[0]
    if parent_width <= 0:
        return []

    return [
        parent_bbox[0] + i * parent_width / logical_col_count
        for i in range(logical_col_count + 1)
    ]


def _estimate_observed_physical_column_centers(
    raw_evidence: RawTableEvidence,
) -> dict[int, float]:
    """Estimate each physical column center from observed cell geometry."""
    non_empty_centers: dict[int, list[float]] = {
        physical_col: [] for physical_col in range(raw_evidence.physical_col_count)
    }
    any_centers: dict[int, list[float]] = {
        physical_col: [] for physical_col in range(raw_evidence.physical_col_count)
    }

    for row in raw_evidence.rows:
        for cell in row.cells:
            if cell.physical_col < 0 or cell.physical_col >= raw_evidence.physical_col_count:
                continue
            center_x = _estimate_cell_text_center_x(raw_evidence, cell)
            if center_x is None and cell.bbox:
                x0, _, x1, _ = cell.bbox
                if x1 <= x0:
                    continue
                center_x = (x0 + x1) / 2
            if center_x is None:
                continue
            any_centers[cell.physical_col].append(center_x)
            if cell.text and cell.text.strip():
                non_empty_centers[cell.physical_col].append(center_x)

    observed_centers: dict[int, float] = {}
    for physical_col in range(raw_evidence.physical_col_count):
        samples = non_empty_centers[physical_col] or any_centers[physical_col]
        if samples:
            observed_centers[physical_col] = float(statistics.median(samples))

    return observed_centers


def _estimate_cell_text_center_x(
    raw_evidence: RawTableEvidence,
    cell: RawCell,
) -> float | None:
    """Prefer actual word geometry over a wide merged-cell bbox."""
    text = str(cell.text or "").strip()
    if not text or not cell.bbox:
        return None

    x0, y0, x1, y1 = cell.bbox
    if x1 <= x0 or y1 <= y0:
        return None

    normalized_cell_text = _normalize_geometry_text(text)
    word_centers: list[float] = []
    for word in raw_evidence.words:
        try:
            wx0 = float(word.x0)
            wy0 = float(word.y0)
            wx1 = float(word.x1)
            wy1 = float(word.y1)
        except (TypeError, ValueError):
            continue
        word_center_x = (wx0 + wx1) / 2
        word_center_y = (wy0 + wy1) / 2
        if not (x0 - 1.0 <= word_center_x <= x1 + 1.0 and y0 - 1.0 <= word_center_y <= y1 + 1.0):
            continue
        word_text = _normalize_geometry_text(str(getattr(word, "text", "") or ""))
        if word_text and word_text not in normalized_cell_text and normalized_cell_text not in word_text:
            continue
        word_centers.append(word_center_x)

    if not word_centers:
        return None
    return float(statistics.median(word_centers))


def _normalize_geometry_text(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def _supplement_empty_grid_cells_from_words(
    *,
    raw_evidence: RawTableEvidence,
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    parent_boundaries: list[float] | None,
    logical_col_count: int,
) -> int:
    """Fill empty projected cells from words inside the same physical row.

    This is intentionally narrow: it only uses words already inside the detected
    table region, assigns them through inherited/observed column boundaries, and
    refuses to overwrite existing cell text.
    """
    if not raw_evidence.words or not grid or logical_col_count <= 0:
        return 0

    boundaries = list(parent_boundaries or [])
    if len(boundaries) < logical_col_count + 1:
        return 0

    row_by_physical: dict[int, NormalizedRow] = {
        row.physical_row: row for row in rows
    }
    changed = 0
    for raw_row in raw_evidence.rows:
        if raw_row.physical_row < 0 or raw_row.physical_row >= len(grid):
            continue
        if not raw_row.bbox:
            continue
        existing_non_empty_cols = [
            col_idx
            for col_idx, value in enumerate(grid[raw_row.physical_row])
            if _normalize_geometry_text(value or "")
        ]
        if existing_non_empty_cols != [0]:
            continue
        first_cell_text = str(grid[raw_row.physical_row][0] or "").strip()
        if _looks_like_section_group_label(first_cell_text):
            continue
        row_y0, row_y1 = float(raw_row.bbox[1]), float(raw_row.bbox[3])
        if row_y1 <= row_y0:
            continue

        words_by_col: dict[int, list[RawWord]] = {idx: [] for idx in range(logical_col_count)}
        for word in raw_evidence.words:
            word_center_y = (float(word.y0) + float(word.y1)) / 2
            if not (row_y0 - 1.0 <= word_center_y <= row_y1 + 1.0):
                continue
            text = str(word.text or "").strip()
            if not text:
                continue
            word_center_x = (float(word.x0) + float(word.x1)) / 2
            logical_col = _assign_center_to_parent_skeleton(
                center_x=word_center_x,
                parent_boundaries=boundaries,
                logical_col_count=logical_col_count,
            )
            if 0 <= logical_col < logical_col_count:
                words_by_col[logical_col].append(word)

        normalized_row = row_by_physical.get(raw_row.physical_row)
        if normalized_row is None:
            continue

        for logical_col, col_words in words_by_col.items():
            if _normalize_geometry_text(grid[raw_row.physical_row][logical_col] or ""):
                continue
            ordered_words = sorted(col_words, key=lambda item: (float(item.y0), float(item.x0)))
            text = _merge_words_for_cell(ordered_words)
            if not text:
                continue
            grid[raw_row.physical_row][logical_col] = text
            for cell in normalized_row.cells:
                if cell.logical_col != logical_col:
                    continue
                cell.text = text
                cell.bbox = (
                    min(float(word.x0) for word in ordered_words),
                    min(float(word.y0) for word in ordered_words),
                    max(float(word.x1) for word in ordered_words),
                    max(float(word.y1) for word in ordered_words),
                )
                cell.supplemented = True
                cell.supplement_reason = "same_row_word_projection"
                changed += 1
                break

    return changed


def _split_spanning_identifier_cells_from_words(
    *,
    raw_evidence: RawTableEvidence,
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    parent_boundaries: list[float] | None,
    logical_col_count: int,
) -> int:
    """Split merged identifier/description cells using inherited column geometry.

    Some PDFs expose a visually two-column rule row as one wide physical cell
    that spans the identifier and description columns. When the row also carries
    downstream columns such as details/severity, the parent table skeleton plus
    word geometry can safely project that wide cell into logical cells. Section
    group rows like "3 - ICH ..." are intentionally left untouched.
    """
    if not raw_evidence.words or not grid or logical_col_count < 2:
        return 0

    boundaries = list(parent_boundaries or [])
    if len(boundaries) < logical_col_count + 1:
        return 0

    row_by_physical: dict[int, NormalizedRow] = {
        row.physical_row: row for row in rows
    }
    changed = 0

    for raw_row in raw_evidence.rows:
        if raw_row.physical_row < 0 or raw_row.physical_row >= len(grid):
            continue
        if not raw_row.bbox:
            continue

        grid_row = grid[raw_row.physical_row]
        existing_non_empty_cols = [
            col_idx
            for col_idx, value in enumerate(grid_row)
            if _normalize_geometry_text(value or "")
        ]
        if len(existing_non_empty_cols) < 2:
            continue

        normalized_row = row_by_physical.get(raw_row.physical_row)
        if normalized_row is None:
            continue

        for raw_cell in raw_row.cells:
            cell_text = str(raw_cell.text or "").strip()
            if not cell_text or not raw_cell.bbox:
                continue
            cell_cols = _logical_columns_overlapped_by_bbox(
                raw_cell.bbox,
                parent_boundaries=boundaries,
                logical_col_count=logical_col_count,
            )
            if len(cell_cols) < 2:
                continue

            words_by_col = _words_by_parent_column_for_cell(
                raw_evidence=raw_evidence,
                raw_cell=raw_cell,
                parent_boundaries=boundaries,
                logical_col_count=logical_col_count,
            )
            populated_cols = [col for col, words in words_by_col.items() if words]
            if len(populated_cols) < 2:
                continue
            if not set(populated_cols).issubset(set(cell_cols)):
                continue

            first_col = min(populated_cols)
            first_text = _merge_words_for_cell(words_by_col[first_col])
            if not _looks_like_dotted_outline_identifier(first_text):
                continue

            split_texts: dict[int, str] = {}
            for logical_col in populated_cols:
                text = _merge_words_for_cell(words_by_col[logical_col])
                if text:
                    split_texts[logical_col] = text
            if len(split_texts) < 2:
                continue

            joined_split_text = _normalize_geometry_text("".join(split_texts[col] for col in sorted(split_texts)))
            normalized_cell_text = _normalize_geometry_text(cell_text)
            if joined_split_text and normalized_cell_text and joined_split_text != normalized_cell_text:
                continue

            source_logical_cols = [
                col
                for col in cell_cols
                if col < len(grid_row)
                and _normalize_geometry_text(grid_row[col] or "") == normalized_cell_text
            ]
            source_logical_col = source_logical_cols[0] if source_logical_cols else cell_cols[0]
            can_project = True
            for logical_col, text in split_texts.items():
                current = _normalize_geometry_text(grid_row[logical_col] if logical_col < len(grid_row) else "")
                if current and logical_col != source_logical_col:
                    can_project = False
                    break
                if logical_col == source_logical_col and current and current != normalized_cell_text:
                    can_project = False
                    break
            if not can_project:
                continue

            for logical_col, text in split_texts.items():
                if logical_col >= len(grid_row):
                    continue
                grid_row[logical_col] = text
                _update_normalized_cell_from_words(
                    normalized_row,
                    logical_col=logical_col,
                    words=words_by_col[logical_col],
                    text=text,
                    reason="parent_boundary_identifier_split",
                )
            changed += 1

    return changed


def _logical_columns_overlapped_by_bbox(
    bbox: tuple[float, float, float, float],
    *,
    parent_boundaries: list[float],
    logical_col_count: int,
) -> list[int]:
    x0, _, x1, _ = bbox
    overlapped: list[int] = []
    for logical_col in range(min(logical_col_count, len(parent_boundaries) - 1)):
        left = parent_boundaries[logical_col]
        right = parent_boundaries[logical_col + 1]
        if max(0.0, min(float(x1), right) - max(float(x0), left)) > 1.0:
            overlapped.append(logical_col)
    return overlapped


def _words_by_parent_column_for_cell(
    *,
    raw_evidence: RawTableEvidence,
    raw_cell: RawCell,
    parent_boundaries: list[float],
    logical_col_count: int,
) -> dict[int, list[RawWord]]:
    words_by_col: dict[int, list[RawWord]] = {idx: [] for idx in range(logical_col_count)}
    if not raw_cell.bbox:
        return words_by_col
    x0, y0, x1, y1 = raw_cell.bbox
    for word in raw_evidence.words:
        text = str(word.text or "").strip()
        if not text:
            continue
        word_center_x = (float(word.x0) + float(word.x1)) / 2
        word_center_y = (float(word.y0) + float(word.y1)) / 2
        if not (x0 - 1.0 <= word_center_x <= x1 + 1.0 and y0 - 1.0 <= word_center_y <= y1 + 1.0):
            continue
        logical_col = _assign_center_to_parent_skeleton(
            center_x=word_center_x,
            parent_boundaries=parent_boundaries,
            logical_col_count=logical_col_count,
        )
        if 0 <= logical_col < logical_col_count:
            words_by_col[logical_col].append(word)
    return words_by_col


def _looks_like_dotted_outline_identifier(text: str | None) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)+", str(text or "").strip()))


def _looks_like_section_group_label(text: str | None) -> bool:
    return bool(re.match(r"^\s*\d+(?:\.\d+)*\s*[-－–—]\s+\S+", str(text or "").strip()))


def _update_normalized_cell_from_words(
    row: NormalizedRow,
    *,
    logical_col: int,
    words: list[RawWord],
    text: str,
    reason: str,
) -> None:
    ordered_words = sorted(words, key=lambda item: (float(item.y0), float(item.x0)))
    bbox = None
    if ordered_words:
        bbox = (
            min(float(word.x0) for word in ordered_words),
            min(float(word.y0) for word in ordered_words),
            max(float(word.x1) for word in ordered_words),
            max(float(word.y1) for word in ordered_words),
        )
    for cell in row.cells:
        if cell.logical_col != logical_col:
            continue
        cell.text = text
        cell.bbox = bbox
        cell.supplemented = True
        cell.supplement_reason = reason
        return


def _merge_words_for_cell(words: list[RawWord]) -> str | None:
    text = " ".join(str(word.text or "").strip() for word in words if str(word.text or "").strip()).strip()
    if not text:
        return None
    return _normalize_merged_word_spacing(text)


def _normalize_merged_word_spacing(text: str) -> str:
    """Separate CJK labels from dotted outline numbers glued by PDF word extraction."""
    text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
    return re.sub(r"([\u4e00-\u9fff])(\d+(?:\.\d+)+)", r"\1 \2", text)


def _assign_center_to_parent_skeleton(
    center_x: float,
    parent_boundaries: list[float],
    logical_col_count: int,
) -> int:
    """Map an observed center onto the nearest logical column interval."""
    interval_count = min(logical_col_count, max(0, len(parent_boundaries) - 1))
    if interval_count <= 0:
        return 0

    for logical_col in range(interval_count):
        left_boundary = parent_boundaries[logical_col]
        right_boundary = parent_boundaries[logical_col + 1]
        if left_boundary <= center_x < right_boundary:
            return logical_col

    nearest_col = 0
    min_dist = float("inf")
    for logical_col in range(interval_count):
        left_boundary = parent_boundaries[logical_col]
        right_boundary = parent_boundaries[logical_col + 1]
        logical_center = (left_boundary + right_boundary) / 2
        distance = abs(center_x - logical_center)
        if distance < min_dist:
            min_dist = distance
            nearest_col = logical_col

    return nearest_col


def _build_column_mapping_from_parent_boundaries(
    raw_evidence: RawTableEvidence,
    parent_bbox: tuple[float, float, float, float],
    parent_column_boundaries: list[float] | None,
    logical_col_count: int,
) -> list[list[int]]:
    """基于父表列边界构建列映射
    
    统一规则：
    1. 优先使用父表逻辑列骨架作为续表结构约束
    2. 优先使用当前页真实单元格几何中心定位物理列
    3. 仅在当前页缺少可用几何时，退化到 bbox 均分估计

    这样可以在续表页保留前导空逻辑列，避免将稀疏列强行均匀铺满纠偏后的 bbox。
    """
    mapping: list[list[int]] = [[] for _ in range(logical_col_count)]
    if logical_col_count <= 0:
        return mapping

    parent_boundaries = _build_parent_skeleton_boundaries(
        parent_bbox=parent_bbox,
        logical_col_count=logical_col_count,
        explicit_boundaries=parent_column_boundaries,
    )
    if len(parent_boundaries) < 2:
        return _build_uniform_column_mapping(
            raw_evidence.physical_col_count,
            logical_col_count,
        )

    observed_centers = _estimate_observed_physical_column_centers(raw_evidence)
    current_bbox = raw_evidence.bbox
    current_width = current_bbox[2] - current_bbox[0]

    for physical_col in range(raw_evidence.physical_col_count):
        col_x_center = observed_centers.get(physical_col)

        # 稀疏续表优先依赖真实 cell 几何；缺失时才退化到 bbox 均分。
        if col_x_center is None and current_width > 0 and raw_evidence.physical_col_count > 0:
            col_x_start = current_bbox[0] + physical_col * current_width / raw_evidence.physical_col_count
            col_x_end = current_bbox[0] + (physical_col + 1) * current_width / raw_evidence.physical_col_count
            col_x_center = (col_x_start + col_x_end) / 2

        if col_x_center is None:
            return _build_uniform_column_mapping(
                raw_evidence.physical_col_count,
                logical_col_count,
            )

        logical_col = _assign_center_to_parent_skeleton(
            center_x=col_x_center,
            parent_boundaries=parent_boundaries,
            logical_col_count=logical_col_count,
        )
        if logical_col < logical_col_count:
            mapping[logical_col].append(physical_col)

    return mapping


def _calculate_mapping_confidence(
    column_mapping: list[list[int]],
    logical_col_count: int,
) -> float:
    """计算列映射置信度"""
    if not column_mapping or logical_col_count <= 0:
        return 0.0
    
    # 检查每个逻辑列是否都有物理列映射
    non_empty_count = sum(1 for mapping in column_mapping if mapping)
    
    # 检查映射是否均匀分布
    mapping_sizes = [len(mapping) for mapping in column_mapping]
    if not mapping_sizes:
        return 0.0
    
    avg_size = sum(mapping_sizes) / len(mapping_sizes)
    variance = sum((s - avg_size) ** 2 for s in mapping_sizes) / len(mapping_sizes)
    
    # 置信度 = 非空列比例 * 均匀度
    coverage = non_empty_count / logical_col_count
    uniformity = 1.0 / (1.0 + variance)
    
    return coverage * uniformity


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
        reconstruct_filename_path_cells_from_text_layer_fn=_rule_reconstruct_filename_path_cells_from_text_layer,
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
        reconstruct_filename_path_cells_from_text_layer_fn=_rule_reconstruct_filename_path_cells_from_text_layer,
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






