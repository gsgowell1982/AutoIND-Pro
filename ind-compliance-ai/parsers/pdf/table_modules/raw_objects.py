"""Raw Objects Layer - 原始证据提取层

Architecture:
    PDF → PyMuPDF → Raw Objects

负责从PDF页面提取原始证据数据：
- RawChar: 原始字符 (PyMuPDF char)
- RawSpan: 原始文本片段 (PyMuPDF span)
- RawDrawing: 原始绘图元素 (PyMuPDF drawing)
- RawCell: 原始单元格 (PyMuPDF cell)
- RawRow: 原始行 (PyMuPDF row)
- RawTableEvidence: 完整的原始表格证据

关键特性：
- 保留 PyMuPDF 原始检测结果，不做任何推断
- 区分物理行列（PyMuPDF检测）和逻辑行列（待推断）
- 提供完整的原始证据供后续层处理
- 包含 chars, spans, drawings 三类原始数据
"""

# Version: v1.0.9
# Optimization Summary:
# - Fix PyMuPDF row/cell bbox alignment by reading per-row cell geometry directly.
# - Avoid incorrect flat-index mapping for sparse/merged table cell bboxes.
# - Preserve row bbox/y-range metadata for downstream recovery logic.
# - Add row-aligned table words as a secondary evidence source for robust recovery.

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any
from enum import Enum


class DrawingType(Enum):
    """绘图类型"""
    LINE = "line"
    RECTANGLE = "rectangle"
    CURVE = "curve"
    UNKNOWN = "unknown"


@dataclass
class RawChar:
    """原始字符 (PyMuPDF char)
    
    PyMuPDF 检测的最小字符单位。
    """
    char: str
    x0: float
    y0: float
    x1: float
    y1: float
    origin: tuple[float, float] = (0.0, 0.0)  # 字符原点
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "char": self.char,
            "bbox": [self.x0, self.y0, self.x1, self.y1],
            "origin": list(self.origin),
        }


@dataclass
class RawSpan:
    """原始文本片段 (PyMuPDF span)
    
    PyMuPDF 检测的文本片段，包含多个字符。
    """
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float = 0.0
    font: str = ""
    flags: int = 0
    chars: list[RawChar] = field(default_factory=list)
    origin: tuple[float, float] = (0.0, 0.0)
    
    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": [self.x0, self.y0, self.x1, self.y1],
            "size": self.size,
            "font": self.font,
            "flags": self.flags,
            "chars": [c.to_dict() for c in self.chars],
            "origin": list(self.origin),
        }


@dataclass
class RawWord:
    """原始单词证据 (PyMuPDF words)"""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": [self.x0, self.y0, self.x1, self.y1],
        }


@dataclass
class RawDrawing:
    """原始绘图元素 (PyMuPDF drawing)
    
    包含线条、矩形、曲线等绘图元素。
    用于检测表格边框和网格线。
    """
    drawing_type: DrawingType
    x0: float
    y0: float
    x1: float
    y1: float
    color: tuple[float, ...] = ()
    fill: tuple[float, ...] = ()
    width: float = 1.0
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_horizontal_line(self) -> bool:
        """是否为水平线"""
        h = self.y1 - self.y0
        w = self.x1 - self.x0
        return w > h * 3 and h < 5
    
    @property
    def is_vertical_line(self) -> bool:
        """是否为垂直线"""
        h = self.y1 - self.y0
        w = self.x1 - self.x0
        return h > w * 3 and w < 5
    
    @property
    def is_rectangular(self) -> bool:
        """是否为矩形"""
        h = self.y1 - self.y0
        w = self.x1 - self.x0
        return h > 5 and w > 5
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.drawing_type.value,
            "bbox": [self.x0, self.y0, self.x1, self.y1],
            "color": list(self.color),
            "fill": list(self.fill),
            "width": self.width,
            "is_horizontal": self.is_horizontal_line,
            "is_vertical": self.is_vertical_line,
        }


@dataclass
class RawCell:
    """原始单元格 (PyMuPDF cell)
    
    PyMuPDF find_tables() 检测的单元格，可能包含多个span。
    注意：这是物理单元格，可能需要合并为逻辑单元格。
    """
    physical_col: int  # PyMuPDF 检测的物理列索引 (0-based)
    physical_row: int  # PyMuPDF 检测的物理行索引 (0-based)
    text: str | None = None
    spans: list[RawSpan] = field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_col": self.physical_col,
            "physical_row": self.physical_row,
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox else None,
            "spans": [s.to_dict() for s in self.spans],
        }


@dataclass
class RawRow:
    """原始行 (PyMuPDF row)
    
    PyMuPDF find_tables() 检测的行，包含多个单元格。
    注意：这是物理行，可能需要处理 rowspan。
    """
    physical_row: int  # 物理行索引 (0-based)
    cells: list[RawCell] = field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None
    y0: float = 0.0
    y1: float = 0.0
    
    @property
    def non_empty_cell_count(self) -> int:
        """非空单元格数量"""
        return sum(1 for c in self.cells if c.text and c.text.strip())
    
    @property
    def non_empty_cell_indices(self) -> tuple[int, ...]:
        """非空单元格的物理列索引"""
        return tuple(c.physical_col for c in self.cells if c.text and c.text.strip())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_row": self.physical_row,
            "cells": [c.to_dict() for c in self.cells],
            "bbox": list(self.bbox) if self.bbox else None,
            "non_empty_count": self.non_empty_cell_count,
        }


@dataclass
class RawTableEvidence:
    """完整的原始表格证据
    
    从 PyMuPDF 提取的原始数据，包含：
    - 物理行列信息 (PyMuPDF 直接检测结果)
    - 原始 chars, spans, drawings 数据
    - 页面上下文信息
    
    关键区分：
    - physical_col_count: PyMuPDF 检测的列数 (可能因垂直分隔线而过多)
    - logical_col_count: 待后续层推断的逻辑列数
    """
    # 基础信息
    page_number: int
    bbox: tuple[float, float, float, float]
    
    # 物理结构 (PyMuPDF 直接检测)
    physical_col_count: int  # PyMuPDF 检测的列数
    physical_row_count: int  # PyMuPDF 检测的行数
    rows: list[RawRow] = field(default_factory=list)
    
    # 原始数据 - 三类核心数据
    chars: list[RawChar] = field(default_factory=list)      # 字符级数据
    spans: list[RawSpan] = field(default_factory=list)      # 片段级数据
    words: list[RawWord] = field(default_factory=list)      # 单词级数据
    drawings: list[RawDrawing] = field(default_factory=list)  # 绘图数据
    
    # 原始表格数据
    raw_data: list[list[str | None]] = field(default_factory=list)
    
    # 页面上下文
    page_height: float = 0.0
    page_width: float = 0.0
    near_page_top: bool = False
    near_page_bottom: bool = False
    
    # 元数据
    source: str = "pymupdf_builtin"  # 或 "word_clustering"
    
    @property
    def row_patterns(self) -> list[tuple[int, ...]]:
        """每行非空单元格的模式"""
        return [row.non_empty_cell_indices for row in self.rows]
    
    @property
    def max_content_cols_per_row(self) -> int:
        """所有行中最大非空单元格数"""
        return max((row.non_empty_cell_count for row in self.rows), default=0)
    
    @property
    def horizontal_lines(self) -> list[RawDrawing]:
        """水平线（潜在表格边框）"""
        return [d for d in self.drawings if d.is_horizontal_line]
    
    @property
    def vertical_lines(self) -> list[RawDrawing]:
        """垂直线（潜在表格分隔线）"""
        return [d for d in self.drawings if d.is_vertical_line]
    
    @property
    def rectangles(self) -> list[RawDrawing]:
        """矩形（潜在单元格边框）"""
        return [d for d in self.drawings if d.is_rectangular]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "bbox": list(self.bbox),
            "physical_col_count": self.physical_col_count,
            "physical_row_count": self.physical_row_count,
            "rows": [r.to_dict() for r in self.rows],
            "chars_count": len(self.chars),
            "spans_count": len(self.spans),
            "words_count": len(self.words),
            "drawings_count": len(self.drawings),
            "horizontal_lines": len(self.horizontal_lines),
            "vertical_lines": len(self.vertical_lines),
            "page_height": self.page_height,
            "near_page_top": self.near_page_top,
            "near_page_bottom": self.near_page_bottom,
            "source": self.source,
            "max_content_cols": self.max_content_cols_per_row,
        }


# ============================================================================
# Extraction Functions
# ============================================================================

def extract_raw_evidence_from_pymupdf(
    page: Any,
    page_number: int,
    page_height: float,
    page_width: float,
    table_index: int = 0,
) -> RawTableEvidence | None:
    """从 PyMuPDF 页面提取原始表格证据
    
    这是 Raw Objects Layer 的核心函数，负责：
    1. 调用 PyMuPDF find_tables() 获取原始检测
    2. 提取所有 chars, spans, drawings 信息
    3. 构建 RawTableEvidence 对象
    
    不做任何推断，保留原始数据。
    
    Args:
        page: PyMuPDF page object
        page_number: 页码 (1-indexed)
        page_height: 页面高度
        page_width: 页面宽度
        table_index: 要提取的表格索引 (0-based)
        
    Returns:
        RawTableEvidence 或 None
    """
    try:
        tables = page.find_tables()
        if not tables or not hasattr(tables, 'tables') or not tables.tables:
            return None
        
        # 支持多表格：检查索引是否有效
        if table_index >= len(tables.tables):
            return None
        
        pymupdf_table = tables.tables[table_index]
        raw_table_data = pymupdf_table.extract()
        
        if not raw_table_data or len(raw_table_data) < 2:
            return None
        
        table_bbox = pymupdf_table.bbox
        if not table_bbox:
            return None
        
        physical_col_count = pymupdf_table.col_count
        physical_row_count = pymupdf_table.row_count
        
        if physical_col_count < 1 or physical_row_count < 2:
            return None
        
        # Step 1: 提取 chars 和 spans
        chars, spans = _extract_chars_and_spans_from_page(page, table_bbox)
        words = _extract_words_from_page(page, table_bbox)
        
        # Step 2: 提取 drawings
        drawings = _extract_drawings_from_page(page, table_bbox)
        
        # Step 3: 构建行数据
        # 优先使用逐行 cells 几何，避免合并单元格导致的扁平索引错位。
        table_row_objects = pymupdf_table.rows if hasattr(pymupdf_table, "rows") else None
        table_cells_bboxes = pymupdf_table.cells if hasattr(pymupdf_table, "cells") else None
        rows = []
        for row_idx, row_data in enumerate(raw_table_data):
            row_bbox = None
            row_cells_bboxes = None
            if table_row_objects and row_idx < len(table_row_objects):
                row_obj = table_row_objects[row_idx]
                row_bbox = _coerce_bbox(getattr(row_obj, "bbox", None))
                row_cells_bboxes = getattr(row_obj, "cells", None)
            cells = []
            for col_idx, cell_text in enumerate(row_data):
                # 优先用行内列 bbox；仅在致密网格时退回扁平索引。
                cell_bbox = None
                if isinstance(row_cells_bboxes, (list, tuple)) and col_idx < len(row_cells_bboxes):
                    cell_bbox = _coerce_bbox(row_cells_bboxes[col_idx])
                elif table_cells_bboxes and len(table_cells_bboxes) == physical_col_count * physical_row_count:
                    cell_index = row_idx * physical_col_count + col_idx
                    cell_bbox = _coerce_bbox(table_cells_bboxes[cell_index])
                cell = RawCell(
                    physical_col=col_idx,
                    physical_row=row_idx,
                    text=str(cell_text).strip() if cell_text else None,
                    bbox=cell_bbox,
                )
                cells.append(cell)
            
            row = RawRow(
                physical_row=row_idx,
                cells=cells,
                bbox=row_bbox,
                y0=float(row_bbox[1]) if row_bbox else 0.0,
                y1=float(row_bbox[3]) if row_bbox else 0.0,
            )
            rows.append(row)
        
        # Step 4: 构建证据
        evidence = RawTableEvidence(
            page_number=page_number,
            bbox=table_bbox,
            physical_col_count=physical_col_count,
            physical_row_count=physical_row_count,
            rows=rows,
            chars=chars,
            spans=spans,
            words=words,
            drawings=drawings,
            raw_data=raw_table_data,
            page_height=page_height,
            page_width=page_width,
            near_page_top=table_bbox[1] <= page_height * 0.28,
            near_page_bottom=table_bbox[3] >= page_height * 0.72,
            source="pymupdf_builtin",
        )
        
        return evidence
        
    except Exception as e:
        import traceback
        print(f"[Raw Objects Extraction Error] {e}")
        traceback.print_exc()
        return None


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """Normalize PyMuPDF Rect-like and tuple-like bboxes."""
    if value is None:
        return None
    if all(hasattr(value, attr) for attr in ("x0", "y0", "x1", "y1")):
        try:
            return (float(value.x0), float(value.y0), float(value.x1), float(value.y1))
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return (
                float(value[0]),
                float(value[1]),
                float(value[2]),
                float(value[3]),
            )
        except (TypeError, ValueError):
            return None
    return None


def _extract_chars_and_spans_from_page(
    page: Any,
    table_bbox: tuple[float, float, float, float],
) -> tuple[list[RawChar], list[RawSpan]]:
    """从页面提取表格区域内的所有字符和文本片段"""
    chars = []
    spans = []
    
    try:
        # 使用 dict 格式获取文本块（flags=0 保留所有内容）
        # 注意：rawdict 需要 flags=0 才能获取 chars
        text_dict = page.get_text("dict", flags=0)
        
        # 确保 table_bbox 是 tuple
        tb = tuple(table_bbox) if not isinstance(table_bbox, tuple) else table_bbox
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
                
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    bbox = span.get("bbox", (0, 0, 0, 0))
                    
                    # 检查是否在表格区域内
                    if not _is_in_table_region(bbox, tb):
                        continue
                    
                    # 构建 RawSpan（简化版，不提取 chars 以提高性能）
                    raw_span = RawSpan(
                        text=text,
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3],
                        size=span.get("size", 0),
                        font=span.get("font", ""),
                        flags=span.get("flags", 0),
                        chars=[],  # 暂不提取字符级数据
                        origin=span.get("origin", (0, 0)),
                    )
                    spans.append(raw_span)
                    
    except Exception as e:
        print(f"[Char/Span Extraction Warning] {e}")
    
    return chars, spans


def _extract_words_from_page(
    page: Any,
    table_bbox: tuple[float, float, float, float],
) -> list[RawWord]:
    """提取表格区域内的 words 证据，作为 spans 的稳健补充。"""
    words: list[RawWord] = []
    try:
        for item in page.get_text("words"):
            if len(item) < 5:
                continue
            x0, y0, x1, y1, text = item[0], item[1], item[2], item[3], str(item[4]).strip()
            if not text:
                continue
            if not _is_in_table_region((x0, y0, x1, y1), table_bbox):
                continue
            words.append(RawWord(text=text, x0=x0, y0=y0, x1=x1, y1=y1))
    except Exception:
        pass
    return words


def _extract_drawings_from_page(
    page: Any,
    table_bbox: tuple[float, float, float, float],
) -> list[RawDrawing]:
    """从页面提取表格区域内的所有绘图元素"""
    drawings = []
    
    try:
        raw_drawings = page.get_drawings()
        
        for draw in raw_drawings:
            rect = draw.get("rect")
            if not rect:
                continue
            
            # 检查是否在表格区域内或与表格边界相交
            if not _is_in_table_region(rect, table_bbox, margin=15):
                continue
            
            # 确定绘图类型
            drawing_type = _classify_drawing(draw)
            
            # 提取颜色信息
            color = draw.get("color") or ()
            fill = draw.get("fill") or ()
            
            raw_drawing = RawDrawing(
                drawing_type=drawing_type,
                x0=rect[0],
                y0=rect[1],
                x1=rect[2],
                y1=rect[3],
                color=tuple(color) if color else (),
                fill=tuple(fill) if fill else (),
                width=draw.get("width", 1.0),
                raw_data=draw,
            )
            drawings.append(raw_drawing)
            
    except Exception as e:
        print(f"[Drawing Extraction Warning] {e}")
    
    return drawings


def _is_in_table_region(
    bbox: tuple[float, float, float, float],
    table_bbox: tuple[float, float, float, float],
    margin: float = 10.0,
) -> bool:
    """检查 bbox 是否在表格区域内"""
    return (
        table_bbox[0] - margin <= bbox[0] <= table_bbox[2] + margin and
        table_bbox[1] - margin <= bbox[1] <= table_bbox[3] + margin and
        table_bbox[0] - margin <= bbox[2] <= table_bbox[2] + margin and
        table_bbox[1] - margin <= bbox[3] <= table_bbox[3] + margin
    )


def _classify_drawing(draw: dict[str, Any]) -> DrawingType:
    """分类绘图元素"""
    items = draw.get("items", [])
    
    if not items:
        return DrawingType.UNKNOWN
    
    # 检查是否为线段
    for item in items:
        item_type = item[0] if isinstance(item, (list, tuple)) else str(item)
        
        if "l" == str(item_type).lower():
            return DrawingType.LINE
        elif "c" == str(item_type).lower():
            return DrawingType.CURVE
        elif "re" in str(item_type).lower():
            return DrawingType.RECTANGLE
    
    # 根据形状判断
    rect = draw.get("rect", (0, 0, 0, 0))
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    
    if w < 3 or h < 3:
        return DrawingType.LINE
    elif w * h < 100:
        return DrawingType.LINE
    else:
        return DrawingType.RECTANGLE


def extract_raw_evidence_from_words(
    words: list[Any],
    page_number: int,
    page_height: float,
    page_width: float,
) -> RawTableEvidence | None:
    """从 word 列表提取原始表格证据 (fallback 策略)
    
    整合自原 builder.py 的 word-clustering 策略。
    当 PyMuPDF find_tables() 失败时使用。
    通过 word 聚类检测表格结构。
    
    Args:
        words: 单词列表 (来自 _Word 或 PyMuPDF words)
        page_number: 页码
        page_height: 页面高度
        page_width: 页面宽度
        
    Returns:
        RawTableEvidence 或 None
    """
    if not words:
        return None
    
    try:
        # Step 1: 聚类行为行组
        rows = _cluster_words_into_rows(words)
        if len(rows) < 2:
            return None
        
        # Step 2: 分组为表格组
        table_groups = _group_rows_into_tables(rows)
        if not table_groups:
            return None
        
        # Step 3: 选择结构最强的候选组。无框线表格的关键不是“最大块”，
        # 而是“跨多行保持稳定列锚点的二维文本网格”。
        best_group = _select_best_table_group(table_groups)
        if not best_group:
            return None

        raw_rows, raw_data, spans, grouped_words = _build_raw_rows_from_group(
            best_group["rows"],
            best_group["column_anchors"],
        )
        if len(raw_rows) < 2:
            return None

        table_bbox = best_group["bbox"]

        # Step 4: 构建 RawTableEvidence
        evidence = RawTableEvidence(
            page_number=page_number,
            bbox=table_bbox,
            physical_col_count=len(best_group["column_anchors"]),
            physical_row_count=len(raw_rows),
            rows=raw_rows,
            chars=[],
            spans=spans,
            words=grouped_words,
            drawings=[],
            raw_data=raw_data,
            page_height=page_height,
            page_width=page_width,
            near_page_top=table_bbox[1] <= page_height * 0.28,
            near_page_bottom=table_bbox[3] >= page_height * 0.72,
            source="word_clustering",
        )
        
        return evidence
        
    except Exception as e:
        print(f"[Word Clustering Extraction Error] {e}")
        return None


def _select_best_table_group(
    table_groups: list[list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Select the strongest borderless-grid candidate from row groups."""
    best_candidate: dict[str, Any] | None = None
    best_score = 0.0

    for group in table_groups:
        segmented_rows = _segment_group_rows(group)
        if len(segmented_rows) < 2:
            continue

        column_anchors = _detect_segment_column_anchors(segmented_rows)
        col_count = len(column_anchors)
        if col_count < 2:
            continue

        segment_counts = [len(row.get("segments", [])) for row in segmented_rows]
        multi_segment_rows = sum(1 for count in segment_counts if count >= 2)
        if multi_segment_rows < 2:
            continue

        row_anchor_matches = [
            _count_row_segments_near_column_anchors(row.get("segments", []), column_anchors)
            for row in segmented_rows
        ]
        strong_anchor_rows = sum(1 for count in row_anchor_matches if count >= max(2, int(col_count * 0.6)))
        stable_anchor_ratio = strong_anchor_rows / max(1, len(segmented_rows))

        aligned_rows = sum(1 for count in segment_counts if count >= max(2, min(col_count, 3)))
        coverage = sum(segment_counts) / max(1, len(segmented_rows) * col_count)
        if coverage < 0.2:
            continue

        dense_grid = stable_anchor_ratio >= 0.5
        sparse_listing = _looks_like_sparse_listing_table(
            segmented_rows=segmented_rows,
            column_anchors=column_anchors,
            segment_counts=segment_counts,
            row_anchor_matches=row_anchor_matches,
            coverage=coverage,
        )
        if not dense_grid and not sparse_listing:
            continue

        if dense_grid:
            score = (
                0.25 * min(1.0, len(segmented_rows) / 8.0)
                + 0.25 * min(1.0, stable_anchor_ratio)
                + 0.25 * (multi_segment_rows / max(1, len(segmented_rows)))
                + 0.15 * (aligned_rows / max(1, len(segmented_rows)))
                + 0.10 * min(1.0, coverage)
            )
        else:
            complete_rows = sum(1 for count in row_anchor_matches if count >= max(3, col_count - 1))
            continuation_rows = sum(
                1
                for count in row_anchor_matches
                if 1 <= count < max(3, col_count - 1)
            )
            score = (
                0.20 * min(1.0, len(segmented_rows) / 12.0)
                + 0.20 * min(1.0, complete_rows / 4.0)
                + 0.20 * min(1.0, continuation_rows / 8.0)
                + 0.20 * min(1.0, coverage / 0.35)
                + 0.20 * min(1.0, aligned_rows / max(1, len(segmented_rows)))
            )

        if score <= best_score:
            continue

        bbox = (
            min(row["bbox"][0] for row in group),
            min(row["bbox"][1] for row in group),
            max(row["bbox"][2] for row in group),
            max(row["bbox"][3] for row in group),
        )
        best_score = score
        best_candidate = {
            "rows": segmented_rows,
            "column_anchors": column_anchors,
            "bbox": bbox,
            "score": score,
        }

    return best_candidate


def _looks_like_sparse_listing_table(
    *,
    segmented_rows: list[dict[str, Any]],
    column_anchors: list[float],
    segment_counts: list[int],
    row_anchor_matches: list[int],
    coverage: float,
) -> bool:
    """Admit borderless listing tables with sparse continuation rows.

    Some IND roadmap/directory pages use a visible column schema followed by
    child rows that populate only one or two interior columns. They are real
    tabular evidence, but their stable-anchor ratio is low because many rows
    intentionally inherit leading keys from the previous full row.
    """
    row_count = len(segmented_rows)
    col_count = len(column_anchors)
    if row_count < 6 or col_count < 3:
        return False
    if coverage < 0.28:
        return False

    complete_threshold = max(3, col_count - 1)
    complete_rows = [idx for idx, count in enumerate(row_anchor_matches) if count >= complete_threshold]
    if len(complete_rows) < 3:
        return False
    if not any(idx <= 2 for idx in complete_rows):
        return False

    header_rows = segmented_rows[: min(2, row_count)]
    header_segments = [
        str(segment.get("text", "") or "").strip()
        for row in header_rows
        for segment in row.get("segments", [])
        if str(segment.get("text", "") or "").strip()
    ]
    if len(header_segments) < min(3, col_count):
        return False
    if not _segments_look_like_listing_header(header_segments):
        return False

    continuation_rows = [
        idx
        for idx, count in enumerate(row_anchor_matches[1:], start=1)
        if 1 <= count < complete_threshold
    ]
    if len(continuation_rows) < 4:
        return False
    if not any(count <= 2 for count in row_anchor_matches[1:]):
        return False

    body_rows = segmented_rows[1:]
    body_segment_texts = [
        str(segment.get("text", "") or "").strip()
        for row in body_rows
        for segment in row.get("segments", [])
        if str(segment.get("text", "") or "").strip()
    ]
    if not body_segment_texts:
        return False

    long_sentence_ratio = sum(
        1 for text in body_segment_texts if _looks_like_narrative_sentence(text)
    ) / max(1, len(body_segment_texts))
    if long_sentence_ratio > 0.18:
        return False

    structured_value_count = sum(1 for text in body_segment_texts if _looks_like_structured_listing_value(text))
    if structured_value_count < 3:
        return False

    multi_segment_rows = sum(1 for count in segment_counts if count >= 2)
    return multi_segment_rows >= max(4, row_count // 3)


def _segments_look_like_listing_header(texts: list[str]) -> bool:
    """Return true when opening segments look like a multi-column schema."""
    if len(texts) < 3:
        return False

    alphaish = 0
    compact = 0
    repeated_schema_words = 0
    schema_terms = {
        "date",
        "content",
        "file",
        "folder",
        "document",
        "submission",
        "sequence",
        "item",
        "description",
        "destination",
        "section",
        "module",
        "type",
        "name",
        "title",
        "link",
        "location",
        "version",
        "status",
    }
    for text in texts:
        lowered = text.lower()
        if re.search(r"[A-Za-z]", text):
            alphaish += 1
        if len(text.split()) <= 4 and len(text) <= 40:
            compact += 1
        tokens = re.findall(r"[A-Za-z]+", lowered)
        if any(token in schema_terms for token in tokens):
            repeated_schema_words += 1

    return alphaish >= 2 and compact >= max(2, len(texts) - 1) and repeated_schema_words >= 2


def _looks_like_narrative_sentence(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    word_count = len(cleaned.split())
    if word_count >= 11:
        return True
    if word_count >= 7 and re.search(r"[.;:!?]$", cleaned):
        return True
    return False


def _looks_like_structured_listing_value(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if re.search(r"\b\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}\b", cleaned):
        return True
    if re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", cleaned):
        return True
    if re.search(r"\b\d+(?:\.\d+){1,3}\b", cleaned):
        return True
    if re.search(r"\b[\w.-]+\.(?:pdf|xml|xsd|dtd|docx?|xlsx?|txt|zip)\b", cleaned, re.IGNORECASE):
        return True
    if re.search(r"\b[A-Z]{2,}[ -]?\d[\w.-]*\b", cleaned):
        return True
    if re.search(r"\b\d{3,}\b", cleaned) and len(cleaned.split()) <= 4:
        return True
    return False


def _count_row_segments_near_column_anchors(
    segments: list[dict[str, Any]],
    column_anchors: list[float],
) -> int:
    """Count segments that corroborate the global column-anchor lattice."""
    if not segments or not column_anchors:
        return 0
    if len(column_anchors) == 1:
        tolerance = 12.0
    else:
        gaps = [
            abs(float(right) - float(left))
            for left, right in zip(column_anchors, column_anchors[1:])
            if abs(float(right) - float(left)) > 0
        ]
        median_gap = _percentile(gaps, 0.5) if gaps else 24.0
        tolerance = max(8.0, min(18.0, median_gap * 0.35))
    matched_anchors: set[int] = set()
    for segment in segments:
        text = str(segment.get("text", "") or "").strip()
        if not text:
            continue
        x0 = float((segment.get("bbox") or (0.0,))[0])
        nearest = min(
            range(len(column_anchors)),
            key=lambda idx: abs(x0 - float(column_anchors[idx])),
        )
        if abs(x0 - float(column_anchors[nearest])) <= tolerance:
            matched_anchors.add(nearest)
    return len(matched_anchors)


def _segment_group_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split each visual row into cell-like segments using adaptive gap evidence."""
    import statistics

    if not rows:
        return []

    positive_gaps: list[float] = []
    heights: list[float] = []
    for row in rows:
        row_words = row.get("words", [])
        heights.extend(_word_height(word) for word in row_words if _word_height(word) > 0)
        for previous, current in zip(row_words, row_words[1:]):
            gap = _word_x0(current) - _word_x1(previous)
            if gap > 0:
                positive_gaps.append(gap)

    lower_gap = _percentile(positive_gaps, 0.25) if positive_gaps else 0.0
    median_height = statistics.median(heights) if heights else 8.0
    gap_threshold = max(8.0, median_height * 0.8, lower_gap * 2.5)

    segmented_rows: list[dict[str, Any]] = []
    for row in rows:
        row_words = list(row.get("words", []))
        segments = _segment_row_words(row_words, gap_threshold)
        if len(segments) == 1 and len(row_words) >= 4:
            positive_row_gaps = [
                (_word_x0(current) - _word_x1(previous))
                for previous, current in zip(row_words, row_words[1:])
                if (_word_x0(current) - _word_x1(previous)) > 0
            ]
            large_gap_floor = max(18.0, median_height * 2.2)
            large_gap_count = sum(1 for gap in positive_row_gaps if gap >= large_gap_floor)
            if large_gap_count >= 2 and positive_row_gaps:
                refined_gap_threshold = max(
                    12.0,
                    min(gap_threshold * 0.6, statistics.median(positive_row_gaps) * 0.82),
                )
                if refined_gap_threshold < gap_threshold:
                    refined_segments = _segment_row_words(row_words, refined_gap_threshold)
                    if len(refined_segments) > len(segments):
                        segments = refined_segments
        segmented_rows.append({
            "segments": segments,
            "bbox": row.get("bbox", (0.0, 0.0, 0.0, 0.0)),
            "y0": float(row.get("y0", 0.0)),
            "y1": float(row.get("y1", 0.0)),
        })

    return segmented_rows


def _segment_row_words(
    row_words: list[Any],
    gap_threshold: float,
) -> list[dict[str, Any]]:
    """Group adjacent words on the same row into cell-like text segments."""
    if not row_words:
        return []

    ordered_words = sorted(row_words, key=lambda item: (_word_x0(item), _word_y0(item)))
    segments: list[list[Any]] = []
    current_segment: list[Any] = [ordered_words[0]]

    for previous, current in zip(ordered_words, ordered_words[1:]):
        gap = _word_x0(current) - _word_x1(previous)
        if gap > gap_threshold:
            segments.append(current_segment)
            current_segment = [current]
            continue
        current_segment.append(current)

    if current_segment:
        segments.append(current_segment)

    return [_build_segment(segment_words) for segment_words in segments if segment_words]


def _build_segment(segment_words: list[Any]) -> dict[str, Any]:
    """Convert grouped words into a segment record."""
    # Segment words originate from a single clustered visual row, so horizontal
    # reading order is the most stable primary key for rebuilding line text.
    ordered_words = sorted(segment_words, key=lambda item: (_word_x0(item), _word_y0(item)))
    bbox = (
        min(_word_x0(word) for word in ordered_words),
        min(_word_y0(word) for word in ordered_words),
        max(_word_x1(word) for word in ordered_words),
        max(_word_y1(word) for word in ordered_words),
    )
    return {
        "words": ordered_words,
        "bbox": bbox,
        "text": " ".join(_word_text(word) for word in ordered_words).strip(),
    }


def _detect_segment_column_anchors(
    segmented_rows: list[dict[str, Any]],
) -> list[float]:
    """Detect stable column anchors from segment left edges."""
    import statistics

    segments = [
        segment
        for row in segmented_rows
        for segment in row.get("segments", [])
        if segment.get("text")
    ]
    if not segments:
        return []

    heights = [
        max(0.1, float(segment["bbox"][3]) - float(segment["bbox"][1]))
        for segment in segments
    ]
    anchor_tolerance = max(12.0, statistics.median(heights) * 1.4)
    anchors: list[list[float]] = []

    for x0 in sorted(float(segment["bbox"][0]) for segment in segments):
        if not anchors or abs(x0 - anchors[-1][-1]) > anchor_tolerance:
            anchors.append([x0])
            continue
        anchors[-1].append(x0)

    return [sum(cluster) / len(cluster) for cluster in anchors if cluster]


def _build_raw_rows_from_group(
    segmented_rows: list[dict[str, Any]],
    column_anchors: list[float],
) -> tuple[list[RawRow], list[list[str | None]], list[RawSpan], list[RawWord]]:
    """Project segmented rows onto stable column anchors."""
    raw_rows: list[RawRow] = []
    raw_data: list[list[str | None]] = []
    spans: list[RawSpan] = []
    grouped_words: list[RawWord] = []
    col_count = len(column_anchors)

    for row_idx, row in enumerate(segmented_rows):
        row_segments = _refine_row_segments_against_column_anchors(
            row.get("segments", []),
            column_anchors,
        )
        cell_segments: list[list[dict[str, Any]]] = [[] for _ in range(col_count)]
        for segment in row_segments:
            col_idx = _find_column_for_segment(segment, column_anchors)
            if col_idx is None:
                continue
            cell_segments[col_idx].append(segment)

        row_cells: list[RawCell] = []
        row_data_row: list[str | None] = [None] * col_count
        for col_idx, segments in enumerate(cell_segments):
            cell_text: str | None = None
            cell_bbox: tuple[float, float, float, float] | None = None
            if segments:
                ordered_segments = sorted(segments, key=lambda item: (item["bbox"][1], item["bbox"][0]))
                cell_text = " ".join(
                    segment.get("text", "").strip()
                    for segment in ordered_segments
                    if str(segment.get("text", "")).strip()
                ).strip() or None
                cell_bbox = (
                    min(float(segment["bbox"][0]) for segment in ordered_segments),
                    min(float(segment["bbox"][1]) for segment in ordered_segments),
                    max(float(segment["bbox"][2]) for segment in ordered_segments),
                    max(float(segment["bbox"][3]) for segment in ordered_segments),
                )
                for segment in ordered_segments:
                    for word in segment.get("words", []):
                        text = _word_text(word).strip()
                        if not text:
                            continue
                        spans.append(
                            RawSpan(
                                text=text,
                                x0=_word_x0(word),
                                y0=_word_y0(word),
                                x1=_word_x1(word),
                                y1=_word_y1(word),
                            )
                        )
                        grouped_words.append(
                            RawWord(
                                text=text,
                                x0=_word_x0(word),
                                y0=_word_y0(word),
                                x1=_word_x1(word),
                                y1=_word_y1(word),
                            )
                        )
            row_cells.append(
                RawCell(
                    physical_col=col_idx,
                    physical_row=row_idx,
                    text=cell_text,
                    bbox=cell_bbox,
                )
            )
            row_data_row[col_idx] = cell_text

        row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        raw_rows.append(
            RawRow(
                physical_row=row_idx,
                cells=row_cells,
                bbox=row_bbox,
                y0=float(row.get("y0", row_bbox[1] if len(row_bbox) == 4 else 0.0)),
                y1=float(row.get("y1", row_bbox[3] if len(row_bbox) == 4 else 0.0)),
            )
        )
        raw_data.append(row_data_row)

    return raw_rows, raw_data, spans, grouped_words


def _refine_row_segments_against_column_anchors(
    segments: list[dict[str, Any]],
    column_anchors: list[float],
) -> list[dict[str, Any]]:
    """Split under-segmented rows using stable column anchors and large-gap cues."""
    if len(segments) <= 1 or not column_anchors or len(segments) >= len(column_anchors):
        return list(segments)

    refined_segments: list[dict[str, Any]] = []
    for segment in segments:
        refined_segments.extend(_split_segment_on_anchor_transitions(segment, column_anchors))

    if len(refined_segments) <= len(segments) or len(refined_segments) > len(column_anchors):
        return list(segments)
    return refined_segments


def _split_segment_on_anchor_transitions(
    segment: dict[str, Any],
    column_anchors: list[float],
) -> list[dict[str, Any]]:
    """Split a segment when its words show large gaps across distinct column anchors."""
    import statistics

    words = list(segment.get("words", []))
    if len(words) < 2 or not column_anchors:
        return [segment]

    heights = [_word_height(word) for word in words if _word_height(word) > 0]
    median_height = statistics.median(heights) if heights else 8.0
    large_gap_floor = max(18.0, median_height * 2.2)
    word_anchor_indices = [_find_nearest_column_anchor(_word_x0(word), column_anchors) for word in words]

    split_words: list[list[Any]] = []
    current_chunk: list[Any] = [words[0]]
    current_anchor = word_anchor_indices[0]

    for previous_word, current_word, next_anchor in zip(words, words[1:], word_anchor_indices[1:]):
        gap = _word_x0(current_word) - _word_x1(previous_word)
        if gap >= large_gap_floor and next_anchor != current_anchor:
            split_words.append(current_chunk)
            current_chunk = [current_word]
            current_anchor = next_anchor
            continue
        current_chunk.append(current_word)
        current_anchor = next_anchor

    if current_chunk:
        split_words.append(current_chunk)

    if len(split_words) <= 1:
        return [segment]

    split_segments = [_build_segment(chunk) for chunk in split_words if chunk]
    populated_columns = {
        _find_column_for_segment(split_segment, column_anchors)
        for split_segment in split_segments
        if _find_column_for_segment(split_segment, column_anchors) is not None
    }
    if len(populated_columns) <= 1:
        return [segment]
    return split_segments


def _find_column_for_segment(
    segment: dict[str, Any],
    column_anchors: list[float],
) -> int | None:
    """Assign a segment to the nearest left-edge column anchor."""
    if not column_anchors:
        return None

    segment_x0 = float(segment.get("bbox", (0.0, 0.0, 0.0, 0.0))[0])
    min_dist = float("inf")
    closest_col: int | None = None
    for col_idx, anchor in enumerate(column_anchors):
        dist = abs(segment_x0 - anchor)
        if dist < min_dist:
            min_dist = dist
            closest_col = col_idx

    return closest_col


def _find_nearest_column_anchor(
    x0: float,
    column_anchors: list[float],
) -> int | None:
    if not column_anchors:
        return None

    min_dist = float("inf")
    closest_col: int | None = None
    for col_idx, anchor in enumerate(column_anchors):
        dist = abs(float(x0) - float(anchor))
        if dist < min_dist:
            min_dist = dist
            closest_col = col_idx

    return closest_col


def _word_x0(word: Any) -> float:
    if hasattr(word, "x0"):
        return float(word.x0)
    if isinstance(word, (list, tuple)) and len(word) >= 1:
        return float(word[0])
    return 0.0


def _word_y0(word: Any) -> float:
    if hasattr(word, "y0"):
        return float(word.y0)
    if isinstance(word, (list, tuple)) and len(word) >= 2:
        return float(word[1])
    return 0.0


def _word_x1(word: Any) -> float:
    if hasattr(word, "x1"):
        return float(word.x1)
    if isinstance(word, (list, tuple)) and len(word) >= 3:
        return float(word[2])
    return 0.0


def _word_y1(word: Any) -> float:
    if hasattr(word, "y1"):
        return float(word.y1)
    if isinstance(word, (list, tuple)) and len(word) >= 4:
        return float(word[3])
    return 0.0


def _word_text(word: Any) -> str:
    if hasattr(word, "text"):
        return str(word.text)
    if isinstance(word, (list, tuple)) and len(word) >= 5:
        return str(word[4])
    return str(word)


def _word_height(word: Any) -> float:
    return max(0.0, _word_y1(word) - _word_y0(word))


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    ratio = max(0.0, min(1.0, ratio))
    position = ratio * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_words_into_rows(words: list[Any]) -> list[dict[str, Any]]:
    """将单词聚类为行
    
    整合自原 builder.py 的 cluster_rows()。
    """
    import statistics
    
    if not words:
        return []
    
    # 获取高度信息
    heights = []
    for w in words:
        height = _word_height(w)
        if height > 0:
            heights.append(height)
    
    median_height = statistics.median(heights) if heights else 8.0
    row_tol = max(2.5, median_height * 0.55)
    
    rows: list[dict[str, Any]] = []
    
    for word in sorted(words, key=lambda item: (
        (_word_y0(item) + _word_y1(item)) / 2,
        _word_x0(item),
    )):
        word_yc = (_word_y0(word) + _word_y1(word)) / 2
        
        placed = False
        for row in rows:
            if abs(word_yc - row["yc"]) <= row_tol:
                row["words"].append(word)
                row["yc"] = (row["yc"] * row["n"] + word_yc) / (row["n"] + 1)
                row["n"] += 1
                placed = True
                break
        
        if not placed:
            rows.append({
                "yc": word_yc,
                "n": 1,
                "words": [word],
            })
    
    # 规范化行
    normalized_rows = []
    for row in rows:
        row_words = sorted(row["words"], key=lambda item: _word_x0(item))
        
        # 计算 bbox
        x0 = min((_word_x0(w) for w in row_words), default=0.0)
        y0 = min((_word_y0(w) for w in row_words), default=0.0)
        x1 = max((_word_x1(w) for w in row_words), default=0.0)
        y1 = max((_word_y1(w) for w in row_words), default=0.0)
        
        normalized_rows.append({
            "words": row_words,
            "bbox": (x0, y0, x1, y1),
            "y0": y0,
            "y1": y1,
            "height": max(0.1, y1 - y0),
        })
    
    normalized_rows.sort(key=lambda item: item["y0"])
    return normalized_rows


def _group_rows_into_tables(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """将行分组为表格组
    
    整合自原 builder.py 的 group_table_rows()。
    """
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    dense_rows_in_current = 0
    
    for row in rows:
        is_dense_row = len(row.get("words", [])) >= 2
        
        if not current:
            if is_dense_row:
                current = [row]
                dense_rows_in_current = 1
            continue
        
        previous = current[-1]
        
        if is_dense_row:
            vertical_gap = row["y0"] - previous["y1"]
            avg_height = (row["height"] + previous["height"]) / 2
            
            # 水平重叠检查
            overlap = _horizontal_overlap(previous["bbox"], row["bbox"])
            
            if vertical_gap <= max(7.0, avg_height * 1.35) and overlap >= 0.18:
                current.append(row)
                dense_rows_in_current += 1
                continue
            
            if len(current) >= 2 and dense_rows_in_current >= 2:
                groups.append(current)
            
            current = [row]
            dense_rows_in_current = 1
            continue
        
        # 稀疏行处理
        vertical_gap = row["y0"] - previous["y1"]
        avg_height = (row["height"] + previous["height"]) / 2
        overlap = _horizontal_overlap(previous["bbox"], row["bbox"])
        
        if vertical_gap <= max(4.5, avg_height * 0.9) and overlap >= 0.35:
            current.append(row)
            continue
        
        if len(current) >= 2 and dense_rows_in_current >= 2:
            groups.append(current)
        
        current = []
        dense_rows_in_current = 0
    
    if len(current) >= 2 and dense_rows_in_current >= 2:
        groups.append(current)
    
    return [g for g in groups if sum(1 for r in g if len(r.get("words", [])) >= 2) >= 2]


def _horizontal_overlap(bbox_a: tuple, bbox_b: tuple) -> float:
    """计算两个 bbox 的水平重叠比例"""
    x_overlap = max(0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]))
    width_a = bbox_a[2] - bbox_a[0]
    width_b = bbox_b[2] - bbox_b[0]
    if width_a <= 0 or width_b <= 0:
        return 0.0
    return x_overlap / min(width_a, width_b)


def _detect_column_centers(words: list[Any]) -> list[float]:
    """检测列中心位置"""
    import statistics
    
    if not words:
        return []
    
    x_centers = sorted((w.x0 + w.x1) / 2 for w in words if hasattr(w, 'x0'))
    
    if not x_centers:
        return []
    
    median_width = statistics.median((w.x1 - w.x0) for w in words if hasattr(w, 'x0'))
    col_tol = max(8.0, median_width * 0.75) if median_width else 12.0
    
    clusters = []
    for xc in x_centers:
        if not clusters or abs(xc - clusters[-1][-1]) > col_tol:
            clusters.append([xc])
        else:
            clusters[-1].append(xc)
    
    return [sum(c) / len(c) for c in clusters]


def _find_column_for_word(word_x_center: float, column_centers: list[float]) -> int | None:
    """找到单词所属的列"""
    if not column_centers:
        return None
    
    # 找到最近的列中心
    min_dist = float('inf')
    closest_col = None
    
    for col_idx, center in enumerate(column_centers):
        dist = abs(word_x_center - center)
        if dist < min_dist:
            min_dist = dist
            closest_col = col_idx
    
    # 容差检查
    tolerance = 50.0  # 像素
    if min_dist > tolerance:
        return None
    
    return closest_col


# ============================================================================
# Utility Functions
# ============================================================================

def filter_drawings_by_type(
    drawings: list[RawDrawing],
    drawing_type: DrawingType,
) -> list[RawDrawing]:
    """按类型筛选绘图元素"""
    return [d for d in drawings if d.drawing_type == drawing_type]


def get_grid_lines(
    drawings: list[RawDrawing],
) -> tuple[list[RawDrawing], list[RawDrawing]]:
    """获取网格线（水平和垂直线）
    
    Returns:
        (horizontal_lines, vertical_lines)
    """
    horizontal = [d for d in drawings if d.is_horizontal_line]
    vertical = [d for d in drawings if d.is_vertical_line]
    return horizontal, vertical


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Drawing types
    "DrawingType",
    # Data classes
    "RawChar",
    "RawSpan",
    "RawWord",
    "RawDrawing",
    "RawCell",
    "RawRow",
    "RawTableEvidence",
    # Extraction functions
    "extract_raw_evidence_from_pymupdf",
    "extract_raw_evidence_from_words",
    # Utility functions
    "filter_drawings_by_type",
    "get_grid_lines",
]
