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
    
    # Step 6: 补充遗漏内容
    supplemented_count = _supplement_missing_content(
        rows, grid, raw_evidence, column_mapping, logical_col_count, parent_bbox
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
        supplemented_cells_count=supplemented_count,
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
    """推断逻辑列数
    
    核心算法（来自老架构 _merge_pymupdf_columns）：
    PyMuPDF 的 find_tables() 有时会将垂直分隔线检测为单独的列，
    导致很多空列出现在实际数据列之间。需要基于内容分析合并列。
    
    策略优先级：
    1. 父表列继承 (续表场景) - 最高优先级
    2. 内容分布分析 - 分析每行的非空列模式
    3. 列聚类分析 - 将相邻有内容列聚类
    4. 回退到物理列数（不推荐）
    
    Returns:
        (logical_col_count, strategy_name)
    """
    physical_col_count = raw_evidence.physical_col_count
    row_patterns = raw_evidence.row_patterns
    
    if not row_patterns:
        return max(physical_col_count, 1), "fallback"
    
    # Step 1: 分析每行的非空列数量
    cols_per_row = [len(pattern) for pattern in row_patterns if pattern]
    if not cols_per_row:
        return max(physical_col_count, 1), "fallback"
    
    max_cols_in_any_row = max(cols_per_row)
    
    # 数据行分析（跳过可能的表头行）
    data_row_patterns = row_patterns[1:] if len(row_patterns) > 1 else []
    data_cols_per_row = [len(pattern) for pattern in data_row_patterns if pattern]
    max_cols_in_data_rows = max(data_cols_per_row) if data_cols_per_row else 0
    
    # Step 2: 父表列继承（续表场景）- 最高优先级
    if parent_col_count is not None and parent_col_count > 0:
        return parent_col_count, "inherit"
    
    # Step 3: 收集所有有内容的列
    all_content_cols: set[int] = set()
    for pattern in row_patterns:
        all_content_cols.update(pattern)
    
    if not all_content_cols:
        return max(physical_col_count, 1), "fallback"
    
    sorted_content_cols = sorted(all_content_cols)
    
    # Step 4: 计算内容列聚类
    # 相邻的有内容列（间隔=1）可能属于同一逻辑列
    # 有间隔的列（间隔>1）属于不同逻辑列
    clusters: list[list[int]] = []
    current_cluster: list[int] = []
    
    for col_idx in sorted_content_cols:
        if current_cluster and col_idx > current_cluster[-1] + 1:
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(col_idx)
    if current_cluster:
        clusters.append(current_cluster)
    
    num_clusters = len(clusters)
    
    # Step 5: 确定逻辑列数
    # 如果聚类数量等于最大内容列数，直接使用
    if num_clusters == max_cols_in_any_row:
        return num_clusters, "content_clusters"
    
    # 如果聚类数量是1，但最大内容列数>1，使用最大内容列数
    if num_clusters == 1 and max_cols_in_any_row > 1:
        return max_cols_in_any_row, "max_content_cols"
    
    # 否则使用聚类数量（更保守）
    if num_clusters >= 2:
        return num_clusters, "content_clusters"
    
    # 回退到最大内容列数
    return max(max_cols_in_any_row, 1), "max_content_cols"


def _analyze_content_distribution(
    raw_evidence: RawTableEvidence,
) -> dict[str, Any]:
    """分析内容分布以推断逻辑列数
    
    原理：如果某些物理列总是同时有内容，说明它们属于同一逻辑列。
    """
    row_patterns = raw_evidence.row_patterns
    physical_col_count = raw_evidence.physical_col_count
    
    if not row_patterns:
        return {"logical_col_count": physical_col_count, "confidence": 0.0}
    
    # 统计每个物理列的内容频率
    col_frequency: dict[int, int] = {}
    for pattern in row_patterns:
        for col in pattern:
            col_frequency[col] = col_frequency.get(col, 0) + 1
    
    if not col_frequency:
        return {"logical_col_count": physical_col_count, "confidence": 0.0}
    
    sorted_cols = sorted(col_frequency.keys())
    
    # 检测列间间隔
    # 如果相邻物理列总是同时有内容，说明它们属于同一逻辑列
    clusters = [[sorted_cols[0]]]
    for col in sorted_cols[1:]:
        prev_col = clusters[-1][-1]
        # 检查是否总是同时出现
        co_occurrence = sum(
            1 for pattern in row_patterns
            if col in pattern and prev_col in pattern
        )
        solo_occurrence = sum(
            1 for pattern in row_patterns
            if col in pattern or prev_col in pattern
        )
        
        if solo_occurrence > 0 and co_occurrence / solo_occurrence >= 0.5:
            # 高共现率，合并到同一逻辑列
            clusters[-1].append(col)
        else:
            clusters.append([col])
    
    logical_col_count = len(clusters)
    confidence = min(1.0, len(row_patterns) / 5.0)  # 样本越多越可信
    
    return {
        "logical_col_count": logical_col_count,
        "confidence": confidence,
        "clusters": clusters,
    }


def _analyze_column_clustering(
    row_patterns: list[tuple[int, ...]],
    physical_col_count: int,
) -> dict[str, Any]:
    """通过列聚类分析推断逻辑列数
    
    使用间隔聚类算法：相邻的物理列（间隔=1）可能属于同一逻辑列，
    有间隔的物理列（间隔>1）可能属于不同逻辑列。
    """
    if not row_patterns:
        return {"logical_col_count": physical_col_count, "confidence": 0.0}
    
    # 收集所有非空列
    all_non_empty_cols = set()
    for pattern in row_patterns:
        all_non_empty_cols.update(pattern)
    
    if not all_non_empty_cols:
        return {"logical_col_count": physical_col_count, "confidence": 0.0}
    
    sorted_cols = sorted(all_non_empty_cols)
    
    # 使用间隔聚类
    clusters = [[sorted_cols[0]]]
    for col in sorted_cols[1:]:
        prev_col = clusters[-1][-1]
        gap = col - prev_col
        
        if gap <= 1:
            # 紧邻的列，可能属于同一逻辑列
            clusters[-1].append(col)
        else:
            # 有间隔，新开一个逻辑列
            clusters.append([col])
    
    logical_col_count = len(clusters)
    
    # 置信度基于聚类的稳定性
    max_cols_per_row = max(len(p) for p in row_patterns) if row_patterns else 1
    confidence = min(1.0, logical_col_count / max(1, max_cols_per_row))
    
    return {
        "logical_col_count": logical_col_count,
        "confidence": confidence,
        "clusters": clusters,
    }


def _build_column_mapping(
    physical_col_count: int,
    logical_col_count: int,
    row_patterns: list[tuple[int, ...]] | None = None,
    table_bbox: tuple[float, float, float, float] | None = None,
    parent_bbox: tuple[float, float, float, float] | None = None,
) -> list[list[int]]:
    """构建物理列到逻辑列的映射
    
    核心算法（来自老架构 _merge_pymupdf_columns）：
    1. 收集所有有内容的物理列
    2. 对内容列进行聚类（相邻列属于同一逻辑列）
    3. 构建物理列到逻辑列的映射
    
    对于续表，使用父表的边界来确定正确的列映射。
    
    Args:
        physical_col_count: 物理列总数
        logical_col_count: 逻辑列总数
        row_patterns: 每行非空列的模式（用于确定内容列聚类）
        table_bbox: 当前表格的边界框
        parent_bbox: 父表边界框（用于续表列边界继承）
    
    Returns:
        mapping[logical_col] = [physical_cols]
    """
    if logical_col_count <= 0 or physical_col_count <= 0:
        return []
    
    # 对于续表，使用父表边界进行精确的列映射
    if parent_bbox and table_bbox:
        return _build_column_mapping_with_parent_bbox(
            physical_col_count,
            logical_col_count,
            table_bbox,
            parent_bbox,
        )
    
    # 如果没有提供行模式，使用简单的位置映射
    if not row_patterns:
        mapping: list[list[int]] = [[] for _ in range(logical_col_count)]
        for physical_col in range(physical_col_count):
            rel_pos = (physical_col + 0.5) / physical_col_count
            logical_col = min(int(rel_pos * logical_col_count), logical_col_count - 1)
            mapping[logical_col].append(physical_col)
        return mapping
    
    # Step 1: 收集所有有内容的列
    all_content_cols: set[int] = set()
    for pattern in row_patterns:
        all_content_cols.update(pattern)
    
    if not all_content_cols:
        # 没有内容列，使用位置映射
        mapping = [[] for _ in range(logical_col_count)]
        for physical_col in range(physical_col_count):
            rel_pos = (physical_col + 0.5) / physical_col_count
            logical_col = min(int(rel_pos * logical_col_count), logical_col_count - 1)
            mapping[logical_col].append(physical_col)
        return mapping
    
    sorted_content_cols = sorted(all_content_cols)
    
    # Step 2: 计算内容列聚类
    clusters: list[list[int]] = []
    current_cluster: list[int] = []
    
    for col_idx in sorted_content_cols:
        if current_cluster and col_idx > current_cluster[-1] + 1:
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(col_idx)
    if current_cluster:
        clusters.append(current_cluster)
    
    # Step 3: 如果聚类数量等于逻辑列数，直接使用
    if len(clusters) == logical_col_count:
        # 每个聚类对应一个逻辑列
        # 还需要处理聚类之间的空列
        mapping = [[] for _ in range(logical_col_count)]
        
        # 分配聚类内的列
        for logical_col, cluster in enumerate(clusters):
            mapping[logical_col] = list(cluster)
        
        # 分配空列到相邻的逻辑列
        all_mapped = set()
        for cluster in clusters:
            all_mapped.update(cluster)
        
        for physical_col in range(physical_col_count):
            if physical_col not in all_mapped:
                # 找到最近的逻辑列
                min_dist = float('inf')
                nearest_logical = 0
                for logical_col, cluster in enumerate(clusters):
                    for cluster_col in cluster:
                        dist = abs(physical_col - cluster_col)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_logical = logical_col
                mapping[nearest_logical].append(physical_col)
        
        # 排序每个逻辑列的物理列
        for logical_col in range(logical_col_count):
            mapping[logical_col].sort()
        
        return mapping
    
    # Step 4: 否则使用位置映射
    mapping = [[] for _ in range(logical_col_count)]
    for physical_col in range(physical_col_count):
        rel_pos = (physical_col + 0.5) / physical_col_count
        logical_col = min(int(rel_pos * logical_col_count), logical_col_count - 1)
        mapping[logical_col].append(physical_col)
    
    return mapping


def _build_column_mapping_with_parent_bbox(
    physical_col_count: int,
    logical_col_count: int,
    table_bbox: tuple[float, float, float, float],
    parent_bbox: tuple[float, float, float, float],
) -> list[list[int]]:
    """使用父表边界构建精确的列映射
    
    核心思路：
    续表的物理列位置可能与父表不对齐，因此不能简单地基于物理列索引映射。
    应该基于父表的列边界，确定每个物理列应该映射到哪个逻辑列。
    
    特别处理：
    如果续表的起始 x 位置比父表偏右（第一列为空），需要调整映射。
    
    Args:
        physical_col_count: 物理列总数
        logical_col_count: 逻辑列总数
        table_bbox: 当前表格的边界框
        parent_bbox: 父表边界框
    
    Returns:
        mapping[logical_col] = [physical_cols]
    """
    mapping: list[list[int]] = [[] for _ in range(logical_col_count)]
    
    table_width = table_bbox[2] - table_bbox[0]
    parent_width = parent_bbox[2] - parent_bbox[0]
    
    if table_width <= 0 or parent_width <= 0:
        # 回退到简单映射
        for physical_col in range(physical_col_count):
            rel_pos = (physical_col + 0.5) / physical_col_count
            logical_col = min(int(rel_pos * logical_col_count), logical_col_count - 1)
            mapping[logical_col].append(physical_col)
        return mapping
    
    # 检测续表是否偏右（第一列为空）
    # 如果续表的起始 x 比父表偏右超过一定阈值，说明第一列被截断
    x_offset = table_bbox[0] - parent_bbox[0]
    first_col_width = parent_width / logical_col_count
    
    # 如果偏移量超过第一列宽度的 20%，认为第一列被截断
    first_col_truncated = x_offset > first_col_width * 0.2
    
    if first_col_truncated:
        # 第一列被截断，所有物理列向右偏移一列
        # 物理列 0, 1, 2... 映射到逻辑列 1, 2, 2...
        for physical_col in range(physical_col_count):
            # 计算物理列的中心 x 坐标
            col_x_start = table_bbox[0] + physical_col * table_width / physical_col_count
            col_x_end = table_bbox[0] + (physical_col + 1) * table_width / physical_col_count
            col_x_center = (col_x_start + col_x_end) / 2
            
            # 相对于父表的起始位置计算逻辑列
            # 注意：使用父表的 x0 作为基准，而不是续表的 x0
            rel_x = (col_x_center - parent_bbox[0]) / parent_width
            logical_col = int(rel_x * logical_col_count)
            logical_col = max(1, min(logical_col, logical_col_count - 1))  # 至少从第 2 列开始
            
            mapping[logical_col].append(physical_col)
    else:
        # 正常映射：使用父表的列边界
        parent_col_boundaries = []
        for i in range(logical_col_count + 1):
            boundary = parent_bbox[0] + i * parent_width / logical_col_count
            parent_col_boundaries.append(boundary)
        
        for physical_col in range(physical_col_count):
            col_x_start = table_bbox[0] + physical_col * table_width / physical_col_count
            col_x_end = table_bbox[0] + (physical_col + 1) * table_width / physical_col_count
            col_x_center = (col_x_start + col_x_end) / 2
            
            logical_col = 0
            for i in range(logical_col_count):
                if col_x_center >= parent_col_boundaries[i] and col_x_center < parent_col_boundaries[i + 1]:
                    logical_col = i
                    break
                if col_x_center >= parent_col_boundaries[-1]:
                    logical_col = logical_col_count - 1
            
            mapping[logical_col].append(physical_col)
    
    return mapping


def _build_column_clusters(
    column_mapping: list[list[int]],
    table_bbox: tuple[float, float, float, float],
    physical_col_count: int,
) -> list[ColumnCluster]:
    """构建列聚类对象"""
    clusters = []
    table_width = table_bbox[2] - table_bbox[0]
    
    for logical_col, physical_cols in enumerate(column_mapping):
        if not physical_cols:
            continue
        
        # 计算逻辑列的中心位置
        avg_physical = sum(physical_cols) / len(physical_cols)
        center_x = table_bbox[0] + table_width * (avg_physical + 0.5) / physical_col_count
        
        cluster = ColumnCluster(
            logical_col=logical_col,
            physical_cols=physical_cols,
            center_x=center_x,
        )
        clusters.append(cluster)
    
    return clusters


def _normalize_rows_and_cells(
    raw_evidence: RawTableEvidence,
    column_mapping: list[list[int]],
    logical_col_count: int,
) -> tuple[list[NormalizedRow], list[list[str | None]]]:
    """规范化行和单元格
    
    将物理单元格映射到逻辑单元格。
    """
    # 构建反向映射: physical_col -> logical_col
    physical_to_logical: dict[int, int] = {}
    for logical_col, physical_cols in enumerate(column_mapping):
        for physical_col in physical_cols:
            physical_to_logical[physical_col] = logical_col
    
    # 处理每一行
    rows = []
    grid = []
    
    for raw_row in raw_evidence.rows:
        # 初始化逻辑行
        normalized_cells: list[NormalizedCell] = [None] * logical_col_count  # type: ignore
        grid_row: list[str | None] = [None] * logical_col_count
        
        # 累积每个逻辑列的文本
        cell_texts: dict[int, list[str]] = {i: [] for i in range(logical_col_count)}
        
        for raw_cell in raw_row.cells:
            physical_col = raw_cell.physical_col
            logical_col = physical_to_logical.get(physical_col, physical_col)
            
            if raw_cell.text:
                cell_texts[logical_col].append(raw_cell.text)
        
        # 构建规范化单元格
        for logical_col in range(logical_col_count):
            physical_cols = column_mapping[logical_col] if logical_col < len(column_mapping) else []
            text = " ".join(cell_texts[logical_col]) if cell_texts[logical_col] else None
            
            cell = NormalizedCell(
                logical_row=raw_row.physical_row,
                logical_col=logical_col,
                physical_row=raw_row.physical_row,
                physical_col_start=physical_cols[0] if physical_cols else logical_col,
                physical_col_end=physical_cols[-1] if physical_cols else logical_col,
                physical_colspan=len(physical_cols),
                text=text,
            )
            normalized_cells[logical_col] = cell  # type: ignore
            grid_row[logical_col] = text
        
        # 创建规范化行
        normalized_row = NormalizedRow(
            logical_row=raw_row.physical_row,
            physical_row=raw_row.physical_row,
            cells=[c for c in normalized_cells if c is not None],
        )
        rows.append(normalized_row)
        grid.append(grid_row)
    
    return rows, grid


def _supplement_missing_content(
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    raw_evidence: RawTableEvidence,
    column_mapping: list[list[int]],
    logical_col_count: int,
    parent_bbox: tuple[float, float, float, float] | None = None,
) -> int:
    """补充遗漏的表格内容
    
    整合自原 recovery.py 的 supplement_table_from_page_text()。
    从原始 spans 和页面文本中补充 PyMuPDF 可能遗漏的内容。
    
    Algorithm:
    1. Lock logical row y-range
    2. Merge spans using same-row rules (y-overlap > 0.7, x-gap < threshold)
    3. Map to columns based on relative x-position
    4. Structure placeholder for empty cells
    
    Key improvement: Use parent_bbox for column boundary calculation when
    processing continuation tables. This ensures column alignment with the
    parent table across pages.
    
    Returns:
        补充的单元格数量
    """
    if not raw_evidence.spans:
        return 0
    
    supplemented_count = 0
    
    # 计算列宽
    table_bbox = raw_evidence.bbox
    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]
    
    if table_width <= 0 or table_height <= 0:
        return 0
    
    # Use parent table's bbox for column boundaries if available
    # This ensures continuation tables align with their parent
    if parent_bbox and len(parent_bbox) == 4:
        reference_x0 = parent_bbox[0]
        reference_x1 = parent_bbox[2]
    else:
        reference_x0 = table_bbox[0]
        reference_x1 = table_bbox[2]
    
    reference_width = reference_x1 - reference_x0
    
    # 计算每个逻辑列的 x 范围 (使用父表边界)
    col_x_ranges: list[tuple[float, float]] = []
    for logical_col in range(logical_col_count):
        # 使用父表的边界计算列位置
        x_start = reference_x0 + reference_width * logical_col / logical_col_count
        x_end = reference_x0 + reference_width * (logical_col + 1) / logical_col_count
        col_x_ranges.append((x_start, x_end))
    
    # 使用原始 spans 进行补充
    all_spans = raw_evidence.spans
    
    # 计算中位高度用于行分组
    median_height = sorted([s.height for s in all_spans])[len(all_spans)//2] if all_spans else 8.0
    x_gap_threshold = table_width / logical_col_count * 0.25
    
    def y_overlap_ratio(s1: RawSpan, s2: RawSpan) -> float:
        """计算两个 span 的 y 重叠比例"""
        y0_max = max(s1.y0, s2.y0)
        y1_min = min(s1.y1, s2.y1)
        if y1_min <= y0_max:
            return 0.0
        return (y1_min - y0_max) / min(s1.height, s2.height)
    
    # 按 y 位置分组
    y_tolerance = median_height * 0.5
    y_groups: dict[int, list[RawSpan]] = {}
    for span in all_spans:
        y_key = int(span.y_center / y_tolerance)
        y_groups.setdefault(y_key, []).append(span)
    
    # 合并同一 y 组内的 spans
    merged_cells = []
    for group in y_groups.values():
        group_sorted = sorted(group, key=lambda s: s.x0)
        clusters: list[list[RawSpan]] = []
        
        for span in group_sorted:
            merged = False
            for cluster in clusters:
                for existing in cluster:
                    y_overlap = y_overlap_ratio(span, existing)
                    x_gap = span.x0 - existing.x1
                    size_diff = abs(span.size - existing.size)
                    
                    if y_overlap > 0.7 and x_gap < x_gap_threshold and size_diff < 2:
                        cluster.append(span)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                clusters.append([span])
        
        for cluster in clusters:
            cluster_sorted = sorted(cluster, key=lambda s: s.x0)
            text = " ".join(s.text.strip() for s in cluster_sorted)
            x0 = min(s.x0 for s in cluster)
            x1 = max(s.x1 for s in cluster)
            y0 = min(s.y0 for s in cluster)
            y1 = max(s.y1 for s in cluster)
            merged_cells.append({
                "text": text,
                "x_center": (x0 + x1) / 2,
                "y_center": (y0 + y1) / 2,
                "y0": y0, "y1": y1,
            })
    
    # 按行分组
    cell_rows: list[list[dict]] = []
    for cell in sorted(merged_cells, key=lambda c: c["y_center"]):
        found = False
        for row in cell_rows:
            row_y0 = min(c["y0"] for c in row)
            row_y1 = max(c["y1"] for c in row)
            cell_h = cell["y1"] - cell["y0"]
            if cell_h > 0:
                overlap = (min(row_y1, cell["y1"]) - max(row_y0, cell["y0"])) / cell_h
                if overlap > 0.5:
                    row.append(cell)
                    found = True
                    break
        if not found:
            cell_rows.append([cell])
    
    # 映射到逻辑列并补充
    for row_cells in cell_rows:
        # 找到对应的逻辑行
        row_y_center = sum(c["y_center"] for c in row_cells) / len(row_cells)
        logical_row = None
        for row_idx in range(len(rows)):
            row_y0 = table_bbox[1] + table_height * row_idx / len(rows)
            row_y1 = table_bbox[1] + table_height * (row_idx + 1) / len(rows)
            row_center = (row_y0 + row_y1) / 2
            if abs(row_y_center - row_center) < table_height / len(rows) * 0.6:
                logical_row = row_idx
                break
        
        if logical_row is None or logical_row >= len(grid):
            continue
        
        for cell in row_cells:
            # 映射到逻辑列
            # Use parent table's boundaries for column mapping
            # This ensures continuation tables align with parent's column structure
            rel_x = (cell["x_center"] - reference_x0) / reference_width
            logical_col = max(0, min(int(rel_x * logical_col_count), logical_col_count - 1))
            
            cell_text = cell["text"].strip()
            if not cell_text:
                continue
            
            current_text = grid[logical_row][logical_col]
            
            # 只在单元格为空时补充内容
            # 注意：不再追加到已有内容，避免将不同行的数据混合
            if current_text is None:
                grid[logical_row][logical_col] = cell_text
                for norm_cell in rows[logical_row].cells:
                    if norm_cell.logical_col == logical_col:
                        norm_cell.text = cell_text
                        norm_cell.supplemented = True
                        break
                supplemented_count += 1
            # 如果单元格已有内容，不进行追加
            # 原因：PyMuPDF 已正确提取内容，追加会导致不同行数据混合
    
    return supplemented_count


def _detect_and_fix_duplicate_values(
    rows: list[NormalizedRow],
    grid: list[list[str | None]],
    logical_col_count: int,
) -> None:
    """检测并修复重复值问题（合并单元格场景）
    
    问题场景：
    当表格中有合并单元格（视觉上第一列跨越多行）时，PyMuPDF 可能会把
    同一个值放到多个相邻的物理列中。经过列映射后，这会导致第一列和第二列
    显示相同的值。
    
    例如：
    原始期望：
        新药申请 | 首次申请 | ...
        null     | 补充申请 | ...  <- 第一列应为空
        null     | 备案     | ...  <- 第一列应为空
    
    错误结果：
        新药申请 | 首次申请 | ...
        补充申请 | 补充申请 | ...  <- 第一列重复了第二列
        备案     | 备案     | ...  <- 第一列重复了第二列
    
    修复逻辑：
    1. 检测每行中第一列和第二列是否有相同的值
    2. 检查该行上方是否有"大类"行（第一列有值，且与当前行不同）
    3. 如果是，说明当前行应该属于上方的"大类"，将第一列置为 None
    
    Args:
        rows: 规范化行列表
        grid: 规范化网格数据
        logical_col_count: 逻辑列数
    """
    if logical_col_count < 2 or len(grid) < 2:
        return
    
    # 跳过表头行（通常是第一行）
    start_row = 1 if len(grid) > 1 else 0
    
    # 记录最近的"大类"行的第一列值
    last_category_value: str | None = None
    
    for row_idx in range(start_row, len(grid)):
        row = grid[row_idx]
        
        # 获取第一列和第二列的值
        col0_value = row[0] if len(row) > 0 else None
        col1_value = row[1] if len(row) > 1 else None
        
        # 清理文本（去除空白）
        col0_clean = col0_value.strip() if col0_value else None
        col1_clean = col1_value.strip() if col1_value else None
        
        # 检测重复值：第一列和第二列有相同的非空值
        if col0_clean and col1_clean and col0_clean == col1_clean:
            # 检查是否应该属于上方的"大类"
            # 条件：上方有"大类"行（第一列有值），且当前行的第一列值与"大类"不同
            if last_category_value and col0_clean != last_category_value:
                # 这是一个"子项"行，第一列应该为空
                # 修复：将第一列置为 None
                grid[row_idx][0] = None
                
                # 更新对应的 NormalizedCell
                for cell in rows[row_idx].cells:
                    if cell.logical_col == 0:
                        cell.text = None
                        break
                
                continue
        
        # 更新最近的"大类"值
        # 如果第一列有值且不为空，认为是新的"大类"
        if col0_clean:
            last_category_value = col0_clean


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
