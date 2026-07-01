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
    caption_text: str = ""
    caption_bbox: tuple[float, float, float, float] | None = None
    caption_source: str = ""
    
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
        
        if not raw_table_data:
            return None
        
        table_bbox = pymupdf_table.bbox
        if not table_bbox:
            return None
        
        physical_col_count = pymupdf_table.col_count
        physical_row_count = pymupdf_table.row_count
        
        if physical_col_count < 1 or physical_row_count < 1:
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

        best_group = _augment_word_cluster_group_with_preceding_spanner_rows(rows, best_group)
        best_group = _augment_word_cluster_group_with_trailing_aligned_rows(rows, best_group)

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


def extract_caption_anchored_horizontal_rule_tables(
    *,
    words: list[Any],
    drawings: list[dict[str, Any]] | None,
    page_number: int,
    page_height: float,
    page_width: float,
    occupied_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[RawTableEvidence]:
    """Recover caption-anchored booktabs/three-line tables.

    This detector covers tables whose evidence is explicit horizontal rules plus
    stable word rows, but no vertical grid. It deliberately stays in the raw
    evidence layer: it only constructs an auditable candidate from objective
    geometry, then the normal table pipeline decides acceptance.
    """
    if not words:
        return []
    blank_form_candidates = _extract_caption_anchored_blank_form_tables(
        words=words,
        page_number=page_number,
        page_height=page_height,
        page_width=page_width,
        occupied_bboxes=occupied_bboxes,
    )
    if not drawings:
        return _deduplicate_rule_table_candidates(blank_form_candidates)

    full_width_candidates = _extract_caption_anchored_horizontal_rule_tables_single_lane(
        words=words,
        drawings=drawings,
        page_number=page_number,
        page_height=page_height,
        page_width=page_width,
        occupied_bboxes=occupied_bboxes,
    )
    if full_width_candidates and all(
        _caption_rule_candidate_is_strong_full_width_table(candidate, page_width=page_width)
        for candidate in full_width_candidates
    ):
        return _deduplicate_rule_table_candidates(full_width_candidates + blank_form_candidates)

    lane_word_groups = _text_aligned_candidate_word_lanes(
        words=words,
        page_width=page_width,
        page_height=page_height,
    )
    if len(lane_word_groups) > 1:
        candidates: list[RawTableEvidence] = []
        for lane_words in lane_word_groups:
            lane_bbox = _words_bbox_any(lane_words)
            if lane_bbox is None:
                continue
            lane_drawings = _drawings_overlapping_x_range(
                drawings,
                x0=lane_bbox[0],
                x1=lane_bbox[2],
            )
            candidates.extend(
                _extract_caption_anchored_horizontal_rule_tables_single_lane(
                    words=lane_words,
                    drawings=lane_drawings,
                    page_number=page_number,
                    page_height=page_height,
                    page_width=page_width,
                    occupied_bboxes=occupied_bboxes,
                )
            )
        if candidates:
            return _deduplicate_rule_table_candidates(candidates + blank_form_candidates)

    return _deduplicate_rule_table_candidates(full_width_candidates + blank_form_candidates)


def extract_visual_structure_grid_tables(
    *,
    words: list[Any],
    drawings: list[dict[str, Any]] | None,
    page_number: int,
    page_height: float,
    page_width: float,
    occupied_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[RawTableEvidence]:
    """Recover visual table grids that do not carry an explicit caption.

    Some source PDFs expose table structure through horizontal rules, vertical
    separators, or shaded row bands, but have no ``Table N`` caption nearby.
    This detector deliberately emits only raw geometric evidence. Normalization,
    header grammar, continuation, and semantic acceptance stay in the shared
    table pipeline.
    """
    if not words or not drawings:
        return []

    occupied = list(occupied_bboxes or [])
    rows = _cluster_words_into_rows(words)
    if len(rows) < 4:
        return []

    horizontal_bands = _cluster_horizontal_rule_bands(drawings, page_width)
    vertical_bands = _cluster_vertical_rule_bands(drawings, page_height)
    if len(horizontal_bands) < 2 and len(vertical_bands) < 2:
        return []

    candidates: list[RawTableEvidence] = []
    for region in _visual_structure_candidate_regions(
        rows=rows,
        horizontal_bands=horizontal_bands,
        vertical_bands=vertical_bands,
        page_height=page_height,
        page_width=page_width,
        occupied_bboxes=occupied,
    ):
        body_rows = region["body_rows"]
        all_rows = _trim_visual_structure_rows_to_structural_region(
            region["all_rows"],
            body_rows,
            region["bbox"],
        )
        table_bbox = region["bbox"]
        all_rows, body_rows, table_bbox = _shrink_visual_structure_candidate_to_internal_header(
            all_rows=all_rows,
            body_rows=body_rows,
            table_bbox=table_bbox,
        )
        if len(all_rows) < 4:
            continue
        if any(_bbox_overlap_ratio(table_bbox, item) >= 0.25 for item in occupied):
            continue

        body_segmented = _segment_group_rows(body_rows)
        all_segmented = _segment_group_rows(all_rows)
        column_anchors = _detect_rule_table_column_anchors(
            body_segmented,
            table_bbox,
            region.get("rule_bands") or [],
        )
        column_anchors = _prefer_visual_structure_body_column_anchors(
            column_anchors,
            body_segmented,
            table_bbox,
            page_width=page_width,
        )
        if len(column_anchors) < 2:
            continue

        raw_rows, raw_data, spans, grouped_words = _build_raw_rows_from_group(
            all_segmented,
            column_anchors,
            split_single_segment_by_column_anchors=True,
        )
        raw_rows, raw_data = _drop_visual_structure_footer_rows(
            raw_rows,
            raw_data,
            page_height=page_height,
            page_width=page_width,
        )
        if not _visual_structure_candidate_has_table_shape(
            raw_data=raw_data,
            col_count=len(column_anchors),
        ):
            continue
        row_bboxes = [row.bbox for row in raw_rows if row.bbox]
        if row_bboxes:
            table_bbox = _bbox_union_loose(row_bboxes)
        if table_bbox == (0.0, 0.0, 0.0, 0.0):
            continue

        candidates.append(
            RawTableEvidence(
                page_number=page_number,
                bbox=table_bbox,
                physical_col_count=len(column_anchors),
                physical_row_count=len(raw_rows),
                rows=raw_rows,
                chars=[],
                spans=spans,
                words=grouped_words,
                drawings=_raw_drawings_from_visual_structure_region(region),
                raw_data=raw_data,
                page_height=page_height,
                page_width=page_width,
                near_page_top=table_bbox[1] <= page_height * 0.28,
                near_page_bottom=table_bbox[3] >= page_height * 0.72,
                source="visual_structure_grid",
            )
        )

    return _deduplicate_rule_table_candidates(candidates)


def _extract_caption_anchored_horizontal_rule_tables_single_lane(
    *,
    words: list[Any],
    drawings: list[dict[str, Any]] | None,
    page_number: int,
    page_height: float,
    page_width: float,
    occupied_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[RawTableEvidence]:
    if not words or not drawings:
        return []

    occupied = list(occupied_bboxes or [])
    rows = _cluster_words_into_rows(words)
    rule_bands = _cluster_horizontal_rule_bands(drawings, page_width)
    if len(rule_bands) < 2:
        return []

    candidates: list[RawTableEvidence] = []
    for start_idx in range(0, len(rule_bands) - 1):
        for end_idx in range(start_idx + 1, min(len(rule_bands), start_idx + 5)):
            band_slice = rule_bands[start_idx : end_idx + 1]
            table_bbox = _horizontal_rule_candidate_bbox(band_slice)
            if table_bbox is None:
                continue
            top_y = table_bbox[1]
            bottom_y = table_bbox[3]
            if bottom_y <= top_y:
                continue
            if any(_bbox_overlap_ratio(table_bbox, item) >= 0.25 for item in occupied):
                continue

            caption_rows = _caption_rows_above_rule(rows, top_y, table_bbox, page_height)
            if not caption_rows:
                continue

            inside_rows = [
                row
                for row in rows
                if row["y0"] >= top_y - 3.0
                and ((float(row["y0"]) + float(row["y1"])) / 2.0) <= bottom_y + 1.0
                and _horizontal_overlap(row["bbox"], table_bbox) >= 0.18
            ]
            if len(inside_rows) < 4:
                continue

            segmented_rows = _segment_group_rows(inside_rows)
            column_anchors = _detect_rule_table_column_anchors(segmented_rows, table_bbox, band_slice)
            if len(column_anchors) < 2:
                continue

            raw_rows, raw_data, spans, grouped_words = _build_raw_rows_from_group(
                segmented_rows,
                column_anchors,
                split_single_segment_by_column_anchors=True,
            )
            raw_rows, raw_data, spans, grouped_words, inside_rows = _shrink_caption_rule_candidate_to_table_run(
                raw_rows=raw_rows,
                raw_data=raw_data,
                spans=spans,
                grouped_words=grouped_words,
                source_rows=inside_rows,
                col_count=len(column_anchors),
            )
            if len(raw_rows) < 4:
                continue
            if _candidate_internal_caption_row_count(
                RawTableEvidence(
                    page_number=page_number,
                    bbox=table_bbox,
                    physical_col_count=len(column_anchors),
                    physical_row_count=len(raw_rows),
                    rows=raw_rows,
                    raw_data=raw_data,
                    caption_text=_caption_rows_text(caption_rows),
                )
            ) > 0:
                continue
            if not _caption_rule_candidate_has_table_shape(
                raw_data=raw_data,
                rule_bands=band_slice,
                col_count=len(column_anchors),
            ):
                continue

            drawing_evidence = _raw_drawings_from_rule_bands(band_slice)
            caption_text = _caption_rows_text(caption_rows)
            caption_bbox = _rows_bbox(caption_rows)
            grouped_x0 = min((word.x0 for word in grouped_words), default=table_bbox[0])
            grouped_x1 = max((word.x1 for word in grouped_words), default=table_bbox[2])
            if _candidate_is_lane_scoped(page_width=page_width, grouped_x0=grouped_x0, grouped_x1=grouped_x1):
                max_x1 = grouped_x1
            else:
                max_x1 = max(table_bbox[2], grouped_x1)
            expanded_bbox = (
                min(table_bbox[0], grouped_x0),
                min(row["y0"] for row in inside_rows),
                max_x1,
                max(row["y1"] for row in inside_rows),
            )
            candidate = RawTableEvidence(
                page_number=page_number,
                bbox=expanded_bbox,
                physical_col_count=len(column_anchors),
                physical_row_count=len(raw_rows),
                rows=raw_rows,
                chars=[],
                spans=spans,
                words=grouped_words,
                drawings=drawing_evidence,
                raw_data=raw_data,
                page_height=page_height,
                page_width=page_width,
                near_page_top=expanded_bbox[1] <= page_height * 0.28,
                near_page_bottom=expanded_bbox[3] >= page_height * 0.72,
                source="caption_anchored_horizontal_rules",
                caption_text=caption_text,
                caption_bbox=caption_bbox,
                caption_source="caption_anchored_horizontal_rules",
            )
            split_candidates = _split_side_by_side_caption_rule_candidate(
                candidate,
                column_anchors=column_anchors,
            )
            candidates.extend(split_candidates or [candidate])

    return _deduplicate_rule_table_candidates(candidates)


def _extract_caption_anchored_blank_form_tables(
    *,
    words: list[Any],
    page_number: int,
    page_height: float,
    page_width: float,
    occupied_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[RawTableEvidence]:
    rows = _cluster_words_into_rows(words)
    if len(rows) < 4:
        return []
    candidates: list[RawTableEvidence] = []
    for title_index, title_row in enumerate(rows[:-3]):
        title_text = _row_text_from_segmented_source(title_row)
        if not _looks_like_table_caption_row_text(title_text):
            continue
        header_index = title_index + 1
        while header_index < len(rows) and not _row_text_from_segmented_source(rows[header_index]):
            header_index += 1
        if header_index >= len(rows) - 2:
            continue
        header_text = _row_text_from_segmented_source(rows[header_index])
        if not _caption_blank_form_header_looks_structural(header_text):
            continue
        body_rows: list[dict[str, Any]] = []
        for row in rows[header_index + 1 :]:
            row_text = _row_text_from_segmented_source(row)
            if not row_text:
                continue
            if _blank_form_key_row_has_separable_value_column(row):
                break
            if _caption_rule_row_is_blank_form_key(row_text):
                body_rows.append(row)
                continue
            if _caption_rule_tail_row_is_non_table([row_text, None], 2):
                break
            break
        if len(body_rows) < 3:
            continue
        candidate_rows = [rows[header_index], *body_rows]
        candidate_bbox = _rows_bbox(candidate_rows)
        if any(_bbox_overlap_ratio(candidate_bbox, item) >= 0.25 for item in occupied_bboxes or []):
            continue
        raw_rows, raw_data, spans, grouped_words = _build_blank_form_raw_rows_from_rows(
            candidate_rows,
            page_width=page_width,
        )
        if len(raw_data) < 4:
            continue
        candidates.append(
            RawTableEvidence(
                page_number=page_number,
                bbox=candidate_bbox,
                physical_col_count=2,
                physical_row_count=len(raw_rows),
                rows=raw_rows,
                chars=[],
                spans=spans,
                words=grouped_words,
                drawings=[],
                raw_data=raw_data,
                page_height=page_height,
                page_width=page_width,
                near_page_top=candidate_bbox[1] <= page_height * 0.28,
                near_page_bottom=candidate_bbox[3] >= page_height * 0.72,
                source="caption_anchored_blank_form",
                caption_text=title_text,
                caption_bbox=tuple(title_row.get("bbox", candidate_bbox)),
                caption_source="caption_anchored_blank_form",
            )
        )
    return candidates


def _caption_blank_form_header_looks_structural(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned or len(cleaned) > 180:
        return False
    if _looks_like_narrative_sentence(cleaned):
        return False
    words = re.findall(r"[A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff&/+()-]*", cleaned)
    return len(words) >= 3 and not re.match(r"^\s*(?:\d+|[A-Za-z])[\.)]\s+\S+", cleaned)


def _build_blank_form_raw_rows_from_rows(
    rows: list[dict[str, Any]],
    *,
    page_width: float,
) -> tuple[list[RawRow], list[list[str | None]], list[RawSpan], list[RawWord]]:
    header_text = _row_text_from_segmented_source(rows[0])
    raw_data: list[list[str | None]] = [_split_blank_form_header(header_text)]
    for row in rows[1:]:
        raw_data.append([_row_text_from_segmented_source(row), None])
    raw_rows, _projected_raw_data, spans, grouped_words = _build_raw_rows_from_projected_entries(
        [
            {
                "bbox": row.get("bbox", (0.0, 0.0, 0.0, 0.0)),
                "data": raw_data[idx],
                "rows": [row],
            }
            for idx, row in enumerate(rows)
        ],
        2,
    )
    return raw_rows, raw_data, spans, grouped_words


def _split_blank_form_header(text: str) -> list[str | None]:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return [None, None]
    words = cleaned.split()
    best_index: int | None = None
    for idx in range(2, len(words)):
        if not re.match(
            r"^(?:relative|size|rate|rates|result|results|observation|observations|value|values|response|assessment|settling|mean|total)\b",
            words[idx],
            re.IGNORECASE,
        ):
            continue
        left = " ".join(words[:idx])
        right = " ".join(words[idx:])
        if len(left.split()) <= 4 and len(right.split()) >= 3:
            best_index = idx
            break
    if best_index is None:
        best_index = max(1, min(len(words) - 1, len(words) // 2))
    return [" ".join(words[:best_index]).strip(), " ".join(words[best_index:]).strip()]


def _row_text_from_segmented_source(row: dict[str, Any]) -> str:
    words = [
        _word_text(word).strip()
        for word in row.get("words", []) or []
        if _word_text(word).strip()
    ]
    return " ".join(words).strip()


def _split_side_by_side_caption_rule_candidate(
    candidate: RawTableEvidence,
    *,
    column_anchors: list[float],
) -> list[RawTableEvidence]:
    caption_text = str(getattr(candidate, "caption_text", "") or "")
    table_label_count = len(
        re.findall(
            r"\b(?:table|表|附表)\s*[0-9A-Za-z一二三四五六七八九十零〇.\-]+",
            caption_text,
            re.IGNORECASE,
        )
    )
    if table_label_count < 2:
        return []
    if len(column_anchors) < 6 or len(candidate.raw_data or []) < 3:
        return []

    gaps = [
        (idx, float(right) - float(left))
        for idx, (left, right) in enumerate(zip(column_anchors, column_anchors[1:]))
    ]
    if not gaps:
        return []
    split_idx, split_gap = max(gaps, key=lambda item: item[1])
    typical_gaps = sorted(gap for _idx, gap in gaps)
    median_gap = typical_gaps[len(typical_gaps) // 2]
    if split_idx < 1 or split_idx + 2 >= len(column_anchors):
        return []
    if split_gap < max(44.0, median_gap * 1.75):
        return []

    left_rows = _slice_raw_data_columns(candidate.raw_data or [], 0, split_idx + 1)
    right_rows = _slice_raw_data_columns(candidate.raw_data or [], split_idx + 1, len(column_anchors))
    left_col_count = split_idx + 1
    right_col_count = len(column_anchors) - left_col_count
    if not _caption_rule_candidate_has_table_shape(raw_data=left_rows, rule_bands=[{"segments": []}, {"segments": []}], col_count=left_col_count):
        return []
    if not _caption_rule_candidate_has_table_shape(raw_data=right_rows, rule_bands=[{"segments": []}, {"segments": []}], col_count=right_col_count):
        return []

    caption_parts = _split_caption_text_by_table_labels(caption_text, expected_count=2)
    anchor_ranges = [(0, split_idx), (split_idx + 1, len(column_anchors) - 1)]
    row_slices = [left_rows, right_rows]
    result: list[RawTableEvidence] = []
    for side_index, (start_col, end_col) in enumerate(anchor_ranges):
        sliced_rows = row_slices[side_index]
        trimmed_rows = _drop_empty_outer_rows(sliced_rows)
        if len(trimmed_rows) < 3:
            continue
        side_bbox = _side_by_side_split_bbox(candidate, column_anchors, start_col, end_col)
        side_caption = caption_parts[side_index] if side_index < len(caption_parts) else caption_text
        result.append(
            RawTableEvidence(
                page_number=candidate.page_number,
                bbox=side_bbox,
                physical_col_count=len(trimmed_rows[0]) if trimmed_rows else 0,
                physical_row_count=len(trimmed_rows),
                rows=[],
                chars=[],
                spans=[],
                words=[],
                drawings=list(candidate.drawings or []),
                raw_data=trimmed_rows,
                page_height=candidate.page_height,
                page_width=candidate.page_width,
                near_page_top=side_bbox[1] <= candidate.page_height * 0.28,
                near_page_bottom=side_bbox[3] >= candidate.page_height * 0.72,
                source="caption_anchored_horizontal_rules",
                caption_text=side_caption,
                caption_bbox=candidate.caption_bbox,
                caption_source="caption_anchored_horizontal_rules",
            )
        )
    return result if len(result) >= 2 else []


def _caption_rule_candidate_is_strong_full_width_table(
    candidate: RawTableEvidence,
    *,
    page_width: float,
) -> bool:
    raw_data = candidate.raw_data or []
    col_count = int(candidate.physical_col_count or 0)
    if col_count < 5 or len(raw_data) < 6:
        return False
    if _candidate_internal_caption_row_count(candidate) > 0:
        return False
    bbox_width = float(candidate.bbox[2]) - float(candidate.bbox[0])
    if bbox_width < page_width * 0.70:
        return False
    filled_counts = [sum(1 for cell in row if str(cell or "").strip()) for row in raw_data]
    if not filled_counts or min(filled_counts[: min(4, len(filled_counts))]) < max(3, col_count // 2):
        return False
    full_rows = sum(1 for count in filled_counts if count >= max(4, int(col_count * 0.72)))
    if full_rows < max(5, len(raw_data) // 2):
        return False
    long_text_cells = 0
    statistical_or_numeric_cells = 0
    for row in raw_data:
        for cell in row:
            text = str(cell or "").strip()
            if not text:
                continue
            if _looks_like_numeric_cell(text) or re.search(r"\([þ¼+=-]\)|\d(?:\.\d+)?e\s*-?\s*\d+", text, re.IGNORECASE):
                statistical_or_numeric_cells += 1
                continue
            if _looks_like_narrative_sentence(text):
                long_text_cells += 1
    if long_text_cells > max(1, len(raw_data) // 5):
        return False
    return statistical_or_numeric_cells >= max(6, len(raw_data))


def _slice_raw_data_columns(
    raw_data: list[list[str | None]],
    start_col: int,
    end_col: int,
) -> list[list[str | None]]:
    return [list(row[start_col:end_col]) for row in raw_data]


def _drop_empty_outer_rows(raw_data: list[list[str | None]]) -> list[list[str | None]]:
    rows = [list(row) for row in raw_data]
    while rows and not any(str(cell or "").strip() for cell in rows[0]):
        rows.pop(0)
    while rows and not any(str(cell or "").strip() for cell in rows[-1]):
        rows.pop()
    return rows


def _split_caption_text_by_table_labels(text: str, *, expected_count: int) -> list[str]:
    matches = list(
        re.finditer(
            r"\b(?:table|表|附表)\s*[0-9A-Za-z一二三四五六七八九十零〇.\-]+",
            str(text or ""),
            re.IGNORECASE,
        )
    )
    if len(matches) < expected_count:
        return [str(text or "").strip()]
    parts: list[str] = []
    for idx, match in enumerate(matches[:expected_count]):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        part = str(text[start:end]).strip()
        if part:
            parts.append(part)
    return parts


def _words_bbox_any(words: list[Any]) -> tuple[float, float, float, float] | None:
    valid = [word for word in words if _word_text(word).strip()]
    if not valid:
        return None
    return (
        min(_word_x0(word) for word in valid),
        min(_word_y0(word) for word in valid),
        max(_word_x1(word) for word in valid),
        max(_word_y1(word) for word in valid),
    )


def _candidate_is_lane_scoped(
    *,
    page_width: float,
    grouped_x0: float,
    grouped_x1: float,
) -> bool:
    width = float(grouped_x1) - float(grouped_x0)
    if width <= 0:
        return False
    split_x = float(page_width) / 2.0
    guard = max(10.0, min(24.0, page_width * 0.032))
    return width <= page_width * 0.58 and (grouped_x1 <= split_x + guard or grouped_x0 >= split_x - guard)


def _drawings_overlapping_x_range(
    drawings: list[dict[str, Any]] | None,
    *,
    x0: float,
    x1: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    if not drawings:
        return selected
    width = max(1.0, float(x1) - float(x0))
    for drawing in drawings:
        bbox = _coerce_bbox(drawing.get("rect") if isinstance(drawing, dict) else None)
        if bbox is None:
            continue
        overlap = max(0.0, min(float(x1), bbox[2]) - max(float(x0), bbox[0]))
        drawing_width = max(1.0, bbox[2] - bbox[0])
        if overlap / min(width, drawing_width) < 0.25:
            continue
        selected.append(drawing)
    return selected


def _side_by_side_split_bbox(
    candidate: RawTableEvidence,
    column_anchors: list[float],
    start_col: int,
    end_col: int,
) -> tuple[float, float, float, float]:
    x0 = float(column_anchors[start_col]) - 2.0
    if end_col + 1 < len(column_anchors):
        x1 = float(column_anchors[end_col + 1]) - 4.0
    else:
        x1 = float(candidate.bbox[2])
    return (
        max(float(candidate.bbox[0]), x0),
        float(candidate.bbox[1]),
        min(float(candidate.bbox[2]), x1),
        float(candidate.bbox[3]),
    )


def extract_text_aligned_borderless_grid_tables(
    *,
    words: list[Any],
    page_number: int,
    page_height: float,
    page_width: float,
    occupied_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[RawTableEvidence]:
    """Recover text-layer aligned borderless tables.

    This raw-evidence strategy covers IND/CTD pages whose tables have no
    reliable vector grid, no PyMuPDF table object, and no caption/rule anchor,
    but do expose a stable text-column lattice. It constructs only candidate
    evidence; the normal normalization, AST, header grammar, and note ownership
    layers remain responsible for semantics.
    """
    if not words:
        return []

    occupied = list(occupied_bboxes or [])
    candidates: list[RawTableEvidence] = []

    rows = _cluster_words_into_rows(words)
    if len(rows) >= 4:
        full_width_candidates = _extract_text_aligned_candidates_from_segmented_rows(
            segmented_rows=_segment_group_rows(rows),
            occupied=occupied,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
        )
        candidates.extend(full_width_candidates)
        occupied = occupied + [candidate.bbox for candidate in full_width_candidates]

    for lane_words in _text_aligned_candidate_word_lanes(
        words=words,
        page_width=page_width,
        page_height=page_height,
    ):
        rows = _cluster_words_into_rows(lane_words)
        if len(rows) < 4:
            continue
        segmented_rows = _segment_group_rows(rows)
        candidates.extend(
            _extract_text_aligned_candidates_from_segmented_rows(
                segmented_rows=segmented_rows,
                occupied=occupied,
                page_number=page_number,
                page_height=page_height,
                page_width=page_width,
            )
        )

    return _deduplicate_text_aligned_candidates(candidates)


def extract_structured_text_region_tables(
    *,
    words: list[Any],
    drawings: list[dict[str, Any]] | None,
    page_number: int,
    page_height: float,
    page_width: float,
    occupied_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[RawTableEvidence]:
    """Recover low-column structured text regions as table evidence.

    The higher-recall text-aligned fallback focuses on wide, multi-column
    tables. Real manuals, appendices, and IND support documents also contain
    legitimate one-column and two-column table regions: boxed checklists,
    two-column reagent/supply lists, species/name lists with captions below,
    and compact appendix matrices bounded by horizontal rules. This discovery
    source admits only locally structured regions with objective geometry
    support, then hands them to the shared table pipeline.
    """
    if not words:
        return []

    occupied = list(occupied_bboxes or [])
    rows = _cluster_words_into_rows(words)
    if len(rows) < 3:
        return []

    drawing_bboxes = [
        bbox
        for bbox in (_coerce_bbox(draw.get("rect") if isinstance(draw, dict) else None) for draw in (drawings or []))
        if bbox is not None
    ]
    candidates: list[RawTableEvidence] = []
    candidates.extend(
        _extract_rule_bounded_structured_text_regions(
            rows=rows,
            drawing_bboxes=drawing_bboxes,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
            occupied=occupied,
        )
    )
    candidates.extend(
        _extract_contiguous_structured_text_runs(
            rows=rows,
            drawing_bboxes=drawing_bboxes,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
            occupied=occupied,
        )
    )
    return _deduplicate_structured_text_region_candidates(candidates)


def _extract_text_aligned_candidates_from_segmented_rows(
    *,
    segmented_rows: list[dict[str, Any]],
    occupied: list[tuple[float, float, float, float]],
    page_number: int,
    page_height: float,
    page_width: float,
) -> list[RawTableEvidence]:
    candidates: list[RawTableEvidence] = []
    for start_index, start_row in enumerate(segmented_rows):
        start_bbox = tuple(start_row.get("bbox", (0, 0, 0, 0)))
        if any(
            _bbox_overlap_ratio(start_bbox, bbox) >= 0.20
            and not _occupied_bbox_is_weak_row_fragment_for_text_aligned_start(start_row, bbox)
            for bbox in occupied
        ):
            continue
        if not _looks_like_text_aligned_table_opening_row(start_row, page_width):
            continue

        anchor_window = _text_aligned_anchor_window_before_layout_break(
            segmented_rows,
            start_index,
            max_rows=24,
        )
        column_anchors = _detect_text_aligned_grid_column_anchors(anchor_window, page_width)
        if len(column_anchors) < 4:
            continue
        if float(column_anchors[-1]) - float(column_anchors[0]) < page_width * 0.42:
            continue

        selected: list[dict[str, Any]] = []
        projected_entries: list[dict[str, Any]] = []
        last_bottom: float | None = None
        seen_multi_col = False
        stable_table_rows = 0

        for row_index in range(start_index, len(segmented_rows)):
            row = segmented_rows[row_index]
            row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            if len(row_bbox) != 4:
                continue
            if any(
                _bbox_overlap_ratio(row_bbox, bbox) >= 0.20
                and not _occupied_bbox_is_weak_row_fragment_for_text_aligned_start(row, bbox)
                for bbox in occupied
            ):
                break

            row_text = _row_text_from_segmented(row)
            if not row_text:
                continue
            if selected and _looks_like_text_aligned_table_stop_row(row, page_height):
                break
            if selected and _looks_like_text_aligned_table_note_start(row_text):
                break
            if selected and _text_aligned_caption_row_bounds_selected_table(
                selected,
                row,
                stable_table_rows=stable_table_rows,
            ):
                break
            if selected and stable_table_rows >= 4 and _text_aligned_row_breaks_after_stable_table_run(
                row,
                previous_bottom=last_bottom,
            ):
                break

            if last_bottom is not None:
                gap = float(row_bbox[1]) - last_bottom
                row_height = max(1.0, float(row_bbox[3]) - float(row_bbox[1]))
                if gap > max(30.0, row_height * 2.8):
                    break

            projected_row, matched_cols = _project_segmented_row_to_text_grid(row, column_anchors)
            if row_index == start_index:
                if (
                    len(matched_cols) < max(4, min(len(column_anchors), 6))
                    and not _is_sparse_stub_value_table_opening(
                        projected_row,
                        matched_cols,
                        len(column_anchors),
                    )
                ):
                    break
            else:
                if len(matched_cols) >= 2:
                    pass
                elif _is_text_aligned_group_row(row, projected_row, matched_cols, segmented_rows, row_index, column_anchors):
                    pass
                elif _is_text_aligned_continuation_row(
                    projected_entries,
                    projected_row,
                    matched_cols,
                    row_bbox,
                    column_anchors,
                ):
                    pass
                else:
                    if len(selected) >= 4:
                        break
                    continue

            selected.append(row)
            projected_entries.append(
                {
                    "row": row,
                    "data": projected_row,
                    "matched_cols": set(matched_cols),
                    "bbox": row_bbox,
                }
            )
            if len(matched_cols) >= 2:
                seen_multi_col = True
            if len(matched_cols) >= max(4, min(len(column_anchors), 6)):
                stable_table_rows += 1
            last_bottom = float(row_bbox[3])

        boundary_caption = _following_bottom_caption_for_text_aligned_candidate(
            segmented_rows,
            start_index + len(selected),
            selected,
        )
        min_entry_count = 2 if boundary_caption is not None else 4
        if not seen_multi_col or len(projected_entries) < min_entry_count:
            continue

        if boundary_caption is not None:
            projected_entries = _prepend_upstream_text_aligned_header_entries(
                segmented_rows=segmented_rows,
                start_index=start_index,
                projected_entries=projected_entries,
                column_anchors=column_anchors,
                page_width=page_width,
            )
        else:
            projected_entries = _prepend_immediate_header_for_text_aligned_value_matrix(
                segmented_rows=segmented_rows,
                start_index=start_index,
                projected_entries=projected_entries,
                column_anchors=column_anchors,
                page_width=page_width,
            )
        merged_entries = _merge_text_aligned_continuation_entries(projected_entries, len(column_anchors))
        min_merged_count = 2 if boundary_caption is not None else 3
        if len(merged_entries) < min_merged_count:
            continue
        if not _text_aligned_candidate_has_table_shape(
            merged_entries,
            len(column_anchors),
            boundary_caption=boundary_caption,
        ):
            continue

        raw_rows, raw_data, spans, grouped_words = _build_raw_rows_from_projected_entries(
            merged_entries,
            len(column_anchors),
        )
        if len(raw_rows) < 3:
            continue
        table_bbox = _bbox_union_loose([entry["bbox"] for entry in merged_entries])
        if table_bbox == (0.0, 0.0, 0.0, 0.0):
            continue

        candidates.append(
            RawTableEvidence(
                page_number=page_number,
                bbox=table_bbox,
                physical_col_count=len(column_anchors),
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
                source="text_aligned_borderless_grid",
                caption_text=str(boundary_caption.get("text", "") or "") if boundary_caption else "",
                caption_bbox=tuple(boundary_caption.get("bbox")) if boundary_caption else None,
                caption_source="bottom_caption_text_aligned_grid" if boundary_caption else "",
            )
        )
    candidates.extend(
        _extract_anchor_following_small_text_aligned_tables(
            segmented_rows=segmented_rows,
            occupied=occupied,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
        )
    )
    return candidates


def _is_sparse_stub_value_table_opening(
    projected_row: list[str | None],
    matched_cols: set[int],
    col_count: int,
) -> bool:
    """Allow label-plus-dose/value openings with blank spacer anchors."""

    if col_count < 5 or len(matched_cols) < 5:
        return False
    cells = [str(cell or "").strip() for cell in projected_row[:col_count]]
    filled_indices = [idx for idx, text in enumerate(cells) if text]
    if len(filled_indices) < 5:
        return False
    if filled_indices[0] > 1:
        return False
    stub_text = cells[filled_indices[0]]
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", stub_text):
        return False
    if re.search(r"[.。！？!?；;]\s*$", " ".join(cells[idx] for idx in filled_indices if cells[idx])):
        return False
    right_side_values = [
        cells[idx]
        for idx in filled_indices
        if idx >= 2 and cells[idx]
    ]
    value_like = sum(1 for text in right_side_values if _looks_like_text_aligned_value_atom(text))
    return value_like >= 3


def _looks_like_text_aligned_value_atom(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if re.search(r"\d", candidate):
        return True
    if candidate.lower() in {"yes", "no", "na", "n/a", "nd"}:
        return True
    if re.fullmatch(r"[-+*#]+", candidate):
        return True
    return bool(re.fullmatch(r"[A-Z]{1,4}[-/]?[A-Z0-9]{0,6}", candidate))


def _text_aligned_caption_row_bounds_selected_table(
    selected: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    stable_table_rows: int,
) -> bool:
    row_text = _row_text_from_segmented(row)
    if not _looks_like_table_caption_row_text(row_text):
        return False
    if stable_table_rows >= 4:
        return True
    if len(selected) < 2:
        return False
    row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    table_bbox = _bbox_union_loose([tuple(item.get("bbox", (0.0, 0.0, 0.0, 0.0))) for item in selected])
    if table_bbox == (0.0, 0.0, 0.0, 0.0) or len(row_bbox) != 4:
        return False
    if _horizontal_overlap(row_bbox, table_bbox) < 0.55:
        return False
    data_like_rows = sum(1 for item in selected if len(_row_anchor_candidate_lefts(item)) >= 4)
    return data_like_rows >= 2


def _prepend_upstream_text_aligned_header_entries(
    *,
    segmented_rows: list[dict[str, Any]],
    start_index: int,
    projected_entries: list[dict[str, Any]],
    column_anchors: list[float],
    page_width: float,
) -> list[dict[str, Any]]:
    """Recover multi-row headers immediately above a detected data lattice.

    Borderless scientific and IND tables often have spanning header rows whose
    text is centered across several columns. Those rows may not expose enough
    anchors to start the text-aligned detector by themselves, but they are still
    part of the same table when they sit directly above the detected lattice.
    """
    if not projected_entries or start_index <= 0:
        return projected_entries

    first_bbox = tuple(projected_entries[0].get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(first_bbox) != 4:
        return projected_entries

    prepend: list[dict[str, Any]] = []
    next_top = float(first_bbox[1])
    table_x0 = float(first_bbox[0])
    table_x1 = float(first_bbox[2])
    for row_index in range(start_index - 1, max(-1, start_index - 4), -1):
        row = segmented_rows[row_index]
        row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if len(row_bbox) != 4:
            break
        row_text = _row_text_from_segmented(row)
        if not row_text:
            break
        if _looks_like_table_caption_row_text(row_text) or _looks_like_text_aligned_table_note_start(row_text):
            break
        if re.search(r"[.。！？!?；;]\s*$", row_text):
            break

        row_height = max(1.0, float(row_bbox[3]) - float(row_bbox[1]))
        gap = next_top - float(row_bbox[3])
        if gap > max(18.0, row_height * 1.8):
            break
        row_center = (float(row_bbox[0]) + float(row_bbox[2])) / 2.0
        overlaps_or_inside = (
            _horizontal_overlap(row_bbox, (table_x0, first_bbox[1], table_x1, first_bbox[3])) >= 0.18
            or table_x0 - page_width * 0.05 <= row_center <= table_x1 + page_width * 0.05
        )
        if not overlaps_or_inside:
            break

        projected_row, matched_cols = _project_segmented_row_to_text_grid(row, column_anchors)
        compact_header_lefts = _compact_header_word_column_lefts(row, page_width)
        if len(matched_cols) < 1 and len(compact_header_lefts) < 2:
            break
        if len(row_text.split()) > 14 and len(matched_cols) <= 1:
            break

        prepend.append(
            {
                "row": row,
                "data": projected_row,
                "matched_cols": set(matched_cols),
                "bbox": row_bbox,
            }
        )
        next_top = float(row_bbox[1])
        table_x0 = min(table_x0, float(row_bbox[0]))
        table_x1 = max(table_x1, float(row_bbox[2]))

    if not prepend:
        return projected_entries
    return list(reversed(prepend)) + projected_entries


def _prepend_immediate_header_for_text_aligned_value_matrix(
    *,
    segmented_rows: list[dict[str, Any]],
    start_index: int,
    projected_entries: list[dict[str, Any]],
    column_anchors: list[float],
    page_width: float,
) -> list[dict[str, Any]]:
    """Attach a tight label row above a captionless numeric matrix.

    Some text-layer borderless tables are introduced by prose and then start
    with a header row followed by compact numeric rows. If anchor learning starts
    on the first numeric row, the detector sees a strong matrix but loses the
    header. This repairs that evidence layer only when the body rows are
    value-dense and the immediate previous row projects as label-dense table
    header text.
    """

    if start_index <= 0 or len(projected_entries) < 3 or len(column_anchors) < 4:
        return projected_entries
    if not _text_aligned_entries_are_value_matrix_body(projected_entries, len(column_anchors)):
        return projected_entries

    first_bbox = tuple(projected_entries[0].get("bbox", (0.0, 0.0, 0.0, 0.0)))
    previous_row = segmented_rows[start_index - 1]
    previous_bbox = tuple(previous_row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(first_bbox) != 4 or len(previous_bbox) != 4:
        return projected_entries

    previous_text = _row_text_from_segmented(previous_row)
    if not previous_text:
        return projected_entries
    if _looks_like_table_caption_row_text(previous_text) or _looks_like_text_aligned_table_note_start(previous_text):
        return projected_entries
    if _looks_like_narrative_sentence(previous_text) or re.search(r"[.。！？!?；;]\s*$", previous_text):
        return projected_entries

    row_height = max(1.0, float(first_bbox[3]) - float(first_bbox[1]))
    gap = float(first_bbox[1]) - float(previous_bbox[3])
    if gap < -2.0 or gap > max(18.0, row_height * 1.8):
        return projected_entries

    table_x0 = float(first_bbox[0])
    table_x1 = float(first_bbox[2])
    header_center = (float(previous_bbox[0]) + float(previous_bbox[2])) / 2.0
    overlaps_or_inside = (
        _horizontal_overlap(previous_bbox, (table_x0, first_bbox[1], table_x1, first_bbox[3])) >= 0.35
        or table_x0 - page_width * 0.05 <= header_center <= table_x1 + page_width * 0.05
    )
    if not overlaps_or_inside:
        return projected_entries

    projected_header, matched_cols = _project_segmented_row_to_text_grid(previous_row, column_anchors)
    if not _projected_row_is_label_header_for_value_matrix(projected_header, matched_cols, len(column_anchors)):
        ordered_projection = _project_text_aligned_header_row_by_visual_order(
            previous_row,
            col_count=len(column_anchors),
        )
        if ordered_projection is not None:
            projected_header, matched_cols = ordered_projection
    if not _projected_row_is_label_header_for_value_matrix(projected_header, matched_cols, len(column_anchors)):
        return projected_entries

    header_entry = {
        "row": previous_row,
        "data": projected_header,
        "matched_cols": set(matched_cols),
        "bbox": previous_bbox,
    }
    return [header_entry] + projected_entries


def _text_aligned_entries_are_value_matrix_body(entries: list[dict[str, Any]], col_count: int) -> bool:
    if len(entries) < 3 or col_count < 4:
        return False
    dense_value_rows = 0
    first_col_values = 0
    for entry in entries[: min(len(entries), 6)]:
        cells = [str(cell or "").strip() for cell in entry.get("data", [])[:col_count]]
        filled = [cell for cell in cells if cell]
        if len(filled) < max(3, min(col_count, 5) - 1):
            continue
        value_like = sum(1 for cell in filled if _text_aligned_cell_is_numeric_value(cell))
        if value_like >= max(3, len(filled) - 1):
            dense_value_rows += 1
        if cells and _text_aligned_cell_is_numeric_value(cells[0]):
            first_col_values += 1
    return dense_value_rows >= 3 and first_col_values >= 2


def _projected_row_is_label_header_for_value_matrix(
    projected_row: list[str | None],
    matched_cols: set[int],
    col_count: int,
) -> bool:
    if col_count < 4 or len(matched_cols) < max(3, min(col_count, 5) - 1):
        return False
    cells = [str(cell or "").strip() for cell in projected_row[:col_count]]
    filled = [cell for cell in cells if cell]
    if len(filled) < max(3, min(col_count, 5) - 1):
        return False
    joined = " ".join(filled)
    if re.search(r"[.。！？!?；;]\s*$", joined):
        return False
    label_like = sum(1 for cell in filled if re.search(r"[A-Za-z\u4e00-\u9fff]", cell))
    numeric_like = sum(1 for cell in filled if _text_aligned_cell_is_numeric_value(cell))
    return label_like >= max(2, len(filled) // 2) and numeric_like <= max(1, len(filled) // 3)


def _project_text_aligned_header_row_by_visual_order(
    row: dict[str, Any],
    *,
    col_count: int,
) -> tuple[list[str | None], set[int]] | None:
    if col_count < 4:
        return None
    segments = [
        segment
        for segment in row.get("segments", []) or []
        if str(segment.get("text", "") or "").strip()
    ]
    if len(segments) != col_count:
        return None
    texts = [str(segment.get("text", "") or "").strip() for segment in segments]
    if any(not text for text in texts):
        return None
    joined = " ".join(texts)
    if re.search(r"[.。！？!?；;]\s*$", joined):
        return None
    label_like = sum(1 for text in texts if re.search(r"[A-Za-z\u4e00-\u9fff]", text))
    numeric_like = sum(1 for text in texts if _text_aligned_cell_is_numeric_value(text))
    if label_like < max(2, col_count // 2) or numeric_like > max(1, col_count // 3):
        return None

    bboxes = [tuple(segment.get("bbox", (0.0, 0.0, 0.0, 0.0))) for segment in segments]
    if any(len(bbox) != 4 for bbox in bboxes):
        return None
    ordered = sorted(zip(bboxes, texts), key=lambda item: float(item[0][0]))
    centers = [(float(bbox[0]) + float(bbox[2])) / 2.0 for bbox, _text in ordered]
    if any(right <= left for left, right in zip(centers, centers[1:])):
        return None

    return [text for _bbox, text in ordered], set(range(col_count))


def _text_aligned_cell_is_numeric_value(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if re.search(r"[$￥€£%]", candidate):
        return True
    return bool(re.search(r"\d", candidate))


def _following_bottom_caption_for_text_aligned_candidate(
    segmented_rows: list[dict[str, Any]],
    next_index: int,
    selected: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not selected or next_index >= len(segmented_rows):
        return None
    table_bbox = _bbox_union_loose([tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0))) for row in selected])
    if table_bbox == (0.0, 0.0, 0.0, 0.0):
        return None
    caption_rows: list[dict[str, Any]] = []
    previous_bottom = float(table_bbox[3])
    for row in segmented_rows[next_index : min(len(segmented_rows), next_index + 4)]:
        row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if len(row_bbox) != 4:
            continue
        gap = float(row_bbox[1]) - previous_bottom
        row_height = max(1.0, float(row_bbox[3]) - float(row_bbox[1]))
        if gap > max(26.0, row_height * 2.6):
            break
        text = _row_text_from_segmented(row)
        if not text:
            continue
        if not caption_rows:
            if not _looks_like_table_caption_row_text(text):
                return None
            if _horizontal_overlap(row_bbox, table_bbox) < 0.45:
                return None
            caption_rows.append(row)
            previous_bottom = float(row_bbox[3])
            continue
        if _looks_like_table_caption_row_text(text):
            break
        if (
            len(_row_anchor_candidate_lefts(row)) >= 4
            or len(_compact_header_word_column_lefts(row, max(1.0, float(table_bbox[2])))) >= 4
        ):
            break
        if len(text.split()) > 3 or re.search(r"[.。]\s*$", text):
            caption_rows.append(row)
            previous_bottom = float(row_bbox[3])
            continue
        break
    if not caption_rows:
        return None
    caption_bbox = _bbox_union_loose([tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0))) for row in caption_rows])
    return {
        "text": " ".join(_row_text_from_segmented(row) for row in caption_rows if _row_text_from_segmented(row)).strip(),
        "bbox": caption_bbox,
    }


def _extract_anchor_following_small_text_aligned_tables(
    *,
    segmented_rows: list[dict[str, Any]],
    occupied: list[tuple[float, float, float, float]],
    page_number: int,
    page_height: float,
    page_width: float,
) -> list[RawTableEvidence]:
    candidates: list[RawTableEvidence] = []
    for index, row in enumerate(segmented_rows[:-1]):
        anchor = _small_table_anchor_from_rows(segmented_rows, index)
        if anchor is None:
            continue
        for start_index in range(index + 1, min(len(segmented_rows), index + 6)):
            start_row = segmented_rows[start_index]
            start_bbox = tuple(start_row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            if any(_bbox_overlap_ratio(start_bbox, bbox) >= 0.20 for bbox in occupied):
                break
            column_anchors = _word_lefts_as_table_anchors(start_row, page_width)
            if len(column_anchors) < 3 and start_index + 1 < len(segmented_rows):
                next_bbox = tuple(segmented_rows[start_index + 1].get("bbox", (0.0, 0.0, 0.0, 0.0)))
                row_height = max(1.0, float(start_bbox[3]) - float(start_bbox[1])) if len(start_bbox) == 4 else 8.0
                if len(next_bbox) == 4 and float(next_bbox[1]) - float(start_bbox[3]) <= max(22.0, row_height * 2.0):
                    column_anchors = _word_lefts_as_table_anchors(segmented_rows[start_index + 1], page_width)
            if len(column_anchors) < 3:
                continue
            selected: list[dict[str, Any]] = []
            entries: list[dict[str, Any]] = []
            last_bottom: float | None = None
            for row_index in range(start_index, min(len(segmented_rows), start_index + 6)):
                current = segmented_rows[row_index]
                current_bbox = tuple(current.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                if len(current_bbox) != 4:
                    continue
                if any(_bbox_overlap_ratio(current_bbox, bbox) >= 0.20 for bbox in occupied):
                    break
                text = _row_text_from_segmented(current)
                if not text or _looks_like_table_caption_row_text(text):
                    break
                if last_bottom is not None:
                    row_height = max(1.0, float(current_bbox[3]) - float(current_bbox[1]))
                    if float(current_bbox[1]) - last_bottom > max(22.0, row_height * 2.0):
                        break
                projected, matched = _project_segmented_row_to_text_grid(current, column_anchors)
                if row_index == start_index and len(matched) < len(column_anchors) - 1:
                    break
                if row_index > start_index and len(matched) < 2:
                    break
                selected.append(current)
                entries.append({
                    "row": current,
                    "data": projected,
                    "matched_cols": set(matched),
                    "bbox": current_bbox,
                })
                last_bottom = float(current_bbox[3])
            if len(entries) < 2:
                continue
            if not _small_anchor_table_entries_have_shape(entries, len(column_anchors)):
                continue
            raw_rows, raw_data, spans, grouped_words = _build_raw_rows_from_projected_entries(entries, len(column_anchors))
            table_bbox = _bbox_union_loose([entry["bbox"] for entry in entries])
            if table_bbox == (0.0, 0.0, 0.0, 0.0):
                continue
            candidates.append(
                RawTableEvidence(
                    page_number=page_number,
                    bbox=table_bbox,
                    physical_col_count=len(column_anchors),
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
                    source="text_aligned_borderless_grid",
                    caption_text=str(anchor.get("text", "") or ""),
                    caption_bbox=tuple(anchor.get("bbox")) if anchor.get("bbox") else None,
                    caption_source="footnote_following_table_anchor",
                )
            )
            break
    return candidates


def _small_table_anchor_from_rows(
    segmented_rows: list[dict[str, Any]],
    index: int,
) -> dict[str, Any] | None:
    rows = segmented_rows[index : min(len(segmented_rows), index + 2)]
    texts = [_row_text_from_segmented(row) for row in rows if _row_text_from_segmented(row)]
    if not texts:
        return None
    joined = " ".join(texts)
    if not re.search(r"\b(?:see|shown|listed|given)\s+(?:the\s+)?table\s+below\b|\btable\s+below\b|如下表|见下表|下表", joined, re.IGNORECASE):
        return None
    bbox = _bbox_union_loose([tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0))) for row in rows])
    return {"text": joined.strip(), "bbox": bbox}


def _word_lefts_as_table_anchors(row: dict[str, Any], page_width: float) -> list[float]:
    words = _segmented_row_words(row)
    if len(words) < 3:
        return []
    lefts: list[float] = []
    previous = None
    for word in words:
        text = _word_text(word).strip()
        if not text:
            previous = word
            continue
        if previous is None:
            lefts.append(_word_x0(word))
        else:
            gap = _word_x0(word) - _word_x1(previous)
            if gap >= max(10.0, min(24.0, page_width * 0.018)):
                lefts.append(_word_x0(word))
        previous = word
    return lefts


def _small_anchor_table_entries_have_shape(entries: list[dict[str, Any]], col_count: int) -> bool:
    if len(entries) < 2 or col_count < 3:
        return False
    header = [str(cell or "").strip() for cell in entries[0].get("data", [])]
    data_rows = [[str(cell or "").strip() for cell in entry.get("data", [])] for entry in entries[1:]]
    if sum(1 for cell in header if cell) < max(3, col_count - 1):
        return False
    if not any(sum(1 for cell in row if cell) >= max(3, col_count - 1) for row in data_rows):
        return False
    header_text = " ".join(cell for cell in header if cell)
    if re.search(r"[.。！？!?；;]\s*$", header_text):
        return False
    return True


def _occupied_bbox_is_weak_row_fragment_for_text_aligned_start(
    row: dict[str, Any],
    occupied_bbox: tuple[float, float, float, float],
) -> bool:
    row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(row_bbox) != 4 or len(occupied_bbox) != 4:
        return False
    row_height = max(1.0, float(row_bbox[3]) - float(row_bbox[1]))
    occupied_height = max(1.0, float(occupied_bbox[3]) - float(occupied_bbox[1]))
    if occupied_height > row_height * 1.8:
        return False
    row_support = max(
        len([segment for segment in row.get("segments", []) or [] if str(segment.get("text", "")).strip()]),
        len(_row_anchor_candidate_lefts(row)),
    )
    if row_support < 4:
        return False
    vertical_overlap = min(float(row_bbox[3]), float(occupied_bbox[3])) - max(float(row_bbox[1]), float(occupied_bbox[1]))
    return vertical_overlap > 0.0


def _text_aligned_row_breaks_after_stable_table_run(
    row: dict[str, Any],
    *,
    previous_bottom: float | None,
) -> bool:
    row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(row_bbox) != 4:
        return False
    row_text = _row_text_from_segmented(row)
    if not row_text:
        return False
    row_height = max(1.0, float(row_bbox[3]) - float(row_bbox[1]))
    gap = float(row_bbox[1]) - float(previous_bottom if previous_bottom is not None else row_bbox[1])
    segments = [segment for segment in row.get("segments", []) or [] if str(segment.get("text", "")).strip()]
    anchorish_lefts = _row_anchor_candidate_lefts(row)
    support = max(len(segments), len(anchorish_lefts))
    if support <= 2 and (
        _looks_like_table_caption_row_text(row_text)
        or re.search(r"[.。！？!?；;]\s*$", row_text)
        or len(row_text.split()) >= 8
    ):
        return True
    return gap > max(10.0, row_height * 1.35)


def _text_aligned_anchor_window_before_layout_break(
    segmented_rows: list[dict[str, Any]],
    start_index: int,
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    """Limit column-anchor learning to the local table run.

    Residual pages can mix a full-width table followed by a caption and
    two-column prose. If anchor detection looks too far past the table, prose
    left edges become fake columns and the real table is lost. The window is
    therefore allowed to stop once a compact multi-column run has enough
    support and the next row collapses into narrative/caption geometry.
    """
    window: list[dict[str, Any]] = []
    stable_rows = 0
    previous_bottom: float | None = None
    for row in segmented_rows[start_index : min(len(segmented_rows), start_index + max_rows)]:
        row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if len(row_bbox) != 4:
            continue
        row_text = _row_text_from_segmented(row)
        segments = [segment for segment in row.get("segments", []) or [] if str(segment.get("text", "")).strip()]
        anchorish_lefts = _row_anchor_candidate_lefts(row)
        row_support = max(len(segments), len(anchorish_lefts))
        if window and stable_rows >= 4:
            gap = float(row_bbox[1]) - float(previous_bottom or row_bbox[1])
            row_height = max(1.0, float(row_bbox[3]) - float(row_bbox[1]))
            collapsed_to_prose = row_support <= 2 and (
                _looks_like_table_caption_row_text(row_text)
                or re.search(r"[.。！？!?；;]\s*$", row_text)
                or len(row_text.split()) >= 8
            )
            if collapsed_to_prose or gap > max(10.0, row_height * 1.35):
                break
        if window and stable_rows >= 2 and _looks_like_table_caption_row_text(row_text):
            break
        window.append(row)
        if row_support >= 4 and not re.search(r"[.。！？!?；;]\s*$", row_text):
            stable_rows += 1
        previous_bottom = float(row_bbox[3])
    return window


def _extract_rule_bounded_structured_text_regions(
    *,
    rows: list[dict[str, Any]],
    drawing_bboxes: list[tuple[float, float, float, float]],
    page_number: int,
    page_height: float,
    page_width: float,
    occupied: list[tuple[float, float, float, float]],
) -> list[RawTableEvidence]:
    if not drawing_bboxes:
        return []
    horizontal_rules = [
        bbox
        for bbox in drawing_bboxes
        if _bbox_width(bbox) >= page_width * 0.12 and _bbox_height(bbox) <= max(5.0, _bbox_width(bbox) * 0.08)
    ]
    if len(horizontal_rules) < 2:
        return []

    candidates: list[RawTableEvidence] = []
    for top_index, top_rule in enumerate(sorted(horizontal_rules, key=lambda item: (item[1], item[0]))):
        for bottom_rule in sorted(horizontal_rules, key=lambda item: (item[1], item[0]))[top_index + 1 : top_index + 8]:
            top_y = float(top_rule[1])
            bottom_y = float(bottom_rule[3])
            if bottom_y <= top_y or bottom_y - top_y > page_height * 0.36:
                continue
            x0 = min(float(top_rule[0]), float(bottom_rule[0]))
            x1 = max(float(top_rule[2]), float(bottom_rule[2]))
            if x1 - x0 < page_width * 0.14:
                continue
            region_bbox = (x0, top_y, x1, bottom_y)
            if any(_bbox_overlap_ratio(region_bbox, item) >= 0.25 for item in occupied):
                continue
            inside_rows = [
                row
                for row in rows
                if row["y1"] >= top_y - 4.0
                and row["y0"] <= bottom_y + 4.0
                and _horizontal_overlap(row["bbox"], region_bbox) >= 0.12
            ]
            if len(inside_rows) < 2:
                continue
            candidate = _build_structured_region_candidate(
                rows=inside_rows,
                drawing_bboxes=drawing_bboxes,
                bbox_hint=region_bbox,
                page_number=page_number,
                page_height=page_height,
                page_width=page_width,
                source="structured_text_region",
                require_rule_or_box=False,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _extract_contiguous_structured_text_runs(
    *,
    rows: list[dict[str, Any]],
    drawing_bboxes: list[tuple[float, float, float, float]],
    page_number: int,
    page_height: float,
    page_width: float,
    occupied: list[tuple[float, float, float, float]],
) -> list[RawTableEvidence]:
    candidates: list[RawTableEvidence] = []
    for start_index in range(len(rows)):
        selected: list[dict[str, Any]] = []
        last_bottom: float | None = None
        for row in rows[start_index:]:
            row_bbox = tuple(row.get("bbox") or ())
            if len(row_bbox) != 4:
                continue
            row_text = _row_text(row)
            if not row_text:
                continue
            if any(_bbox_overlap_ratio(row_bbox, item) >= 0.25 for item in occupied):
                break
            if last_bottom is not None:
                gap = float(row_bbox[1]) - last_bottom
                row_height = max(1.0, float(row_bbox[3]) - float(row_bbox[1]))
                if gap > max(28.0, row_height * 2.2):
                    break
            selected.append(row)
            last_bottom = float(row_bbox[3])
            if len(selected) >= 16:
                break
            candidate = _build_structured_region_candidate(
                rows=selected,
                drawing_bboxes=drawing_bboxes,
                bbox_hint=None,
                page_number=page_number,
                page_height=page_height,
                page_width=page_width,
                source="structured_text_region",
                require_rule_or_box=True,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _build_structured_region_candidate(
    *,
    rows: list[dict[str, Any]],
    drawing_bboxes: list[tuple[float, float, float, float]],
    bbox_hint: tuple[float, float, float, float] | None,
    page_number: int,
    page_height: float,
    page_width: float,
    source: str,
    require_rule_or_box: bool,
) -> RawTableEvidence | None:
    rows = [row for row in rows if _row_text(row)]
    rows = _trim_structured_region_rows(rows)
    if len(rows) < 2:
        return None
    rows_bbox = _rows_bbox(rows)
    if rows_bbox is None:
        return None
    table_bbox = _bbox_union_loose([rows_bbox, bbox_hint]) if bbox_hint is not None else rows_bbox
    if table_bbox == (0.0, 0.0, 0.0, 0.0):
        return None

    segmented_rows = _segment_group_rows(rows)
    column_anchors = _detect_structured_region_column_anchors(segmented_rows, table_bbox)
    if not column_anchors:
        return None
    if len(column_anchors) == 1 and not _structured_region_accepts_single_column(segmented_rows, table_bbox, drawing_bboxes):
        return None

    raw_rows, raw_data, spans, grouped_words = _build_raw_rows_from_group(
        segmented_rows,
        column_anchors,
        split_single_segment_by_column_anchors=True,
    )
    if not _structured_region_candidate_has_table_shape(
        raw_data=raw_data,
        col_count=len(column_anchors),
        table_bbox=table_bbox,
        drawing_bboxes=drawing_bboxes,
        page_width=page_width,
        require_rule_or_box=require_rule_or_box,
    ):
        return None
    row_bboxes = [row.bbox for row in raw_rows if row.bbox]
    if row_bboxes:
        table_bbox = _bbox_union_loose([*row_bboxes, bbox_hint])
    drawings = _raw_drawings_from_bbox_list(
        [
            bbox
            for bbox in drawing_bboxes
            if _bbox_overlap_ratio(bbox, table_bbox) >= 0.35
            or (
                _horizontal_overlap(bbox, table_bbox) >= 0.45
                and _bbox_height(bbox) <= 6.0
                and (
                    abs(float(bbox[1]) - float(table_bbox[1])) <= 6.0
                    or abs(float(bbox[3]) - float(table_bbox[3])) <= 6.0
                    or max(0.0, min(float(bbox[3]), float(table_bbox[3])) - max(float(bbox[1]), float(table_bbox[1]))) > 0.0
                )
            )
        ],
        source=source,
    )
    return RawTableEvidence(
        page_number=page_number,
        bbox=table_bbox,
        physical_col_count=len(column_anchors),
        physical_row_count=len(raw_rows),
        rows=raw_rows,
        chars=[],
        spans=spans,
        words=grouped_words,
        drawings=drawings,
        raw_data=raw_data,
        page_height=page_height,
        page_width=page_width,
        near_page_top=table_bbox[1] <= page_height * 0.28,
        near_page_bottom=table_bbox[3] >= page_height * 0.72,
        source=source,
    )


def _trim_structured_region_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim prose and captions around a compact structured table region."""

    if not rows:
        return rows
    trimmed = list(rows)

    while len(trimmed) >= 3:
        first_text = _row_text(trimmed[0])
        next_texts = [_row_text(row) for row in trimmed[1: min(len(trimmed), 4)]]
        if _looks_like_table_caption_row_text(first_text):
            break
        if _looks_like_narrative_sentence(first_text) and any(
            _row_has_multi_segment_table_shape(row) or _looks_like_compact_table_row_text(text)
            for row, text in zip(trimmed[1: min(len(trimmed), 4)], next_texts)
        ):
            trimmed = trimmed[1:]
            continue
        if (
            not _row_has_multi_segment_table_shape(trimmed[0])
            and not _looks_like_compact_table_row_text(first_text)
            and sum(1 for row in trimmed[1: min(len(trimmed), 5)] if _row_has_multi_segment_table_shape(row)) >= 2
        ):
            trimmed = trimmed[1:]
            continue
        break

    stop_index: int | None = None
    for idx, row in enumerate(trimmed):
        if idx < 2:
            continue
        text = _row_text(row)
        if _looks_like_table_caption_row_text(text):
            stop_index = idx
            break
        if _looks_like_narrative_sentence(text):
            following = trimmed[idx + 1 : min(len(trimmed), idx + 4)]
            if not following or sum(1 for item in following if _row_has_multi_segment_table_shape(item)) == 0:
                stop_index = idx
                break
    if stop_index is not None:
        trimmed = trimmed[:stop_index]

    return trimmed


def _row_has_multi_segment_table_shape(row: dict[str, Any]) -> bool:
    segments = [
        segment
        for segment in row.get("segments", []) or []
        if str(segment.get("text", "") or "").strip()
    ]
    if len(segments) >= 2:
        return True
    words = row.get("words", []) or []
    return len([word for word in words if _word_text(word).strip()]) >= 2 and not _looks_like_narrative_sentence(_row_text(row))


def _looks_like_compact_table_row_text(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if _looks_like_table_caption_row_text(candidate):
        return True
    if len(candidate) > 120:
        return False
    if re.search(r"[.。！？!?；;]\s*$", candidate):
        return False
    return True


def _detect_structured_region_column_anchors(
    segmented_rows: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
) -> list[float]:
    anchors = _detect_segment_column_anchors(segmented_rows)
    stable = _detect_stable_segment_column_anchors_allow_low_columns(segmented_rows, table_bbox)
    if len(stable) >= 2:
        return stable
    if len(anchors) >= 2 and _structured_region_anchors_have_support(segmented_rows, anchors):
        return anchors
    if _structured_region_rows_have_single_column_shape(segmented_rows):
        row_lefts = [
            float((row.get("bbox") or table_bbox)[0])
            for row in segmented_rows
            if str(_row_text_from_segmented(row)).strip()
        ]
        if row_lefts:
            import statistics
            return [float(statistics.median(row_lefts))]
    return []


def _detect_stable_segment_column_anchors_allow_low_columns(
    segmented_rows: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
) -> list[float]:
    import statistics

    row_candidates: list[list[float]] = []
    heights: list[float] = []
    for row in segmented_rows:
        segments = [
            segment
            for segment in row.get("segments", []) or []
            if str(segment.get("text", "") or "").strip()
        ]
        if len(segments) < 2:
            continue
        segment_x0s = [float((segment.get("bbox") or (0.0,))[0]) for segment in segments]
        if segment_x0s[0] < table_bbox[0] - 10.0 or segment_x0s[-1] > table_bbox[2] + 10.0:
            continue
        row_candidates.append(segment_x0s)
        for segment in segments:
            bbox = segment.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            if len(bbox) == 4:
                heights.append(max(0.1, float(bbox[3]) - float(bbox[1])))
    if len(row_candidates) < 2:
        return []
    counts: dict[int, int] = {}
    for candidate in row_candidates:
        counts[len(candidate)] = counts.get(len(candidate), 0) + 1
    col_count, support = max(counts.items(), key=lambda item: (item[1], item[0]))
    if col_count < 2 or support < 2:
        return []
    stable_rows = [candidate for candidate in row_candidates if len(candidate) == col_count]
    median_height = statistics.median(heights) if heights else 8.0
    tolerance = max(6.0, min(18.0, median_height * 1.7))
    anchors: list[float] = []
    for idx in range(col_count):
        values = [row[idx] for row in stable_rows]
        if max(values) - min(values) > tolerance * 2.8:
            return []
        anchors.append(float(statistics.median(values)))
    if any(right <= left + 10.0 for left, right in zip(anchors, anchors[1:])):
        return []
    return anchors


def _structured_region_anchors_have_support(
    segmented_rows: list[dict[str, Any]],
    anchors: list[float],
) -> bool:
    if len(anchors) < 2:
        return False
    supported = 0
    for row in segmented_rows:
        projected, matched = _project_segmented_row_to_text_grid(row, anchors)
        filled = sum(1 for cell in projected if str(cell or "").strip())
        if len(matched) >= min(2, len(anchors)) and filled >= min(2, len(anchors)):
            supported += 1
    return supported >= 2


def _structured_region_rows_have_single_column_shape(segmented_rows: list[dict[str, Any]]) -> bool:
    texts = [_row_text_from_segmented(row) for row in segmented_rows if _row_text_from_segmented(row)]
    if len(texts) < 4:
        return False
    list_marker_rows = sum(1 for text in texts if re.match(r"^\s*(?:[#*+\-•]?\s*)?(?:#?\d+[:.)]|[A-Za-z][.)])\s+", text))
    compact_rows = sum(1 for text in texts if len(text) <= 96 and not re.search(r"[.。！？!?；;]\s*$", text))
    return list_marker_rows >= max(3, len(texts) // 2) or compact_rows >= max(4, len(texts) - 1)


def _structured_region_accepts_single_column(
    segmented_rows: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
    drawing_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    if not _structured_region_rows_have_single_column_shape(segmented_rows):
        return False
    return _structured_region_has_rule_or_box_support(table_bbox, drawing_bboxes)


def _structured_region_candidate_has_table_shape(
    *,
    raw_data: list[list[str | None]],
    col_count: int,
    table_bbox: tuple[float, float, float, float],
    drawing_bboxes: list[tuple[float, float, float, float]],
    page_width: float,
    require_rule_or_box: bool,
) -> bool:
    rows = [
        [str(cell or "").strip() for cell in row]
        for row in raw_data
        if any(str(cell or "").strip() for cell in row)
    ]
    if len(rows) < 2 or col_count < 1:
        return False
    if require_rule_or_box and col_count <= 2 and not _structured_region_has_rule_or_box_support(table_bbox, drawing_bboxes):
        return False
    row_texts = [" ".join(cell for cell in row if cell).strip() for row in rows]
    narrative_rows = sum(1 for text in row_texts if _looks_like_narrative_sentence(text))
    caption_rows = sum(1 for text in row_texts if _looks_like_table_caption_row_text(text))
    if narrative_rows >= max(2, len(row_texts) // 3) and caption_rows == 0:
        return False
    if require_rule_or_box and not _structured_region_has_rule_or_box_support(table_bbox, drawing_bboxes):
        return False
    if col_count == 1:
        if len(rows) < 4:
            return False
        if not _structured_region_rows_have_single_column_texts(row_texts):
            return False
        return True
    filled_counts = [sum(1 for cell in row if cell) for row in rows]
    if len(rows) < 4 and not any(_looks_like_table_caption_row_text(text) for text in row_texts):
        return False
    multi_rows = sum(1 for count in filled_counts if count >= min(2, col_count))
    if multi_rows < 2:
        return False
    compact_rows = sum(
        1
        for text in row_texts
        if len(text) <= 160 and not re.search(r"[.。！？!?；;]\s*$", text)
    )
    if compact_rows < max(2, len(row_texts) // 2):
        return False
    if (float(table_bbox[2]) - float(table_bbox[0])) > page_width * 0.82 and not _structured_region_has_rule_or_box_support(table_bbox, drawing_bboxes):
        return False
    return True


def _structured_region_rows_have_single_column_texts(row_texts: list[str]) -> bool:
    if len(row_texts) < 4:
        return False
    marker_rows = sum(1 for text in row_texts if re.match(r"^\s*(?:[#*+\-•]?\s*)?(?:#?\d+[:.)]|[A-Za-z][.)])\s+", text))
    compact_rows = sum(1 for text in row_texts if len(text) <= 96 and not re.search(r"[.。！？!?；;]\s*$", text))
    return marker_rows >= max(3, len(row_texts) // 2) or compact_rows >= max(4, len(row_texts) - 1)


def _structured_region_has_rule_or_box_support(
    table_bbox: tuple[float, float, float, float],
    drawing_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    if not drawing_bboxes:
        return False
    horizontal = 0
    vertical = 0
    for bbox in drawing_bboxes:
        y_overlap = max(0.0, min(float(bbox[3]), float(table_bbox[3])) - max(float(bbox[1]), float(table_bbox[1])))
        near_top_or_bottom = (
            abs(float(bbox[1]) - float(table_bbox[1])) <= 6.0
            or abs(float(bbox[3]) - float(table_bbox[3])) <= 6.0
        )
        x_overlap = _horizontal_overlap(bbox, table_bbox)
        if y_overlap <= 0.0 and not near_top_or_bottom:
            continue
        if _bbox_overlap_ratio(bbox, table_bbox) < 0.08 and x_overlap < 0.35:
            continue
        width = _bbox_width(bbox)
        height = _bbox_height(bbox)
        if width >= max(24.0, (table_bbox[2] - table_bbox[0]) * 0.45) and height <= max(5.0, width * 0.08):
            horizontal += 1
        elif height >= max(24.0, (table_bbox[3] - table_bbox[1]) * 0.35) and width <= max(5.0, height * 0.08):
            vertical += 1
    return horizontal >= 2 or (horizontal >= 1 and vertical >= 2)


def _raw_drawings_from_bbox_list(
    bboxes: list[tuple[float, float, float, float]],
    *,
    source: str,
) -> list[RawDrawing]:
    drawings: list[RawDrawing] = []
    for bbox in bboxes:
        width = _bbox_width(bbox)
        height = _bbox_height(bbox)
        drawings.append(
            RawDrawing(
                drawing_type=DrawingType.LINE if min(width, height) <= 5.0 else DrawingType.RECTANGLE,
                x0=float(bbox[0]),
                y0=float(bbox[1]),
                x1=float(bbox[2]),
                y1=float(bbox[3]),
                raw_data={"source": source},
            )
        )
    return drawings


def _deduplicate_structured_text_region_candidates(candidates: list[RawTableEvidence]) -> list[RawTableEvidence]:
    result: list[RawTableEvidence] = []
    for candidate in sorted(
        candidates,
        key=_structured_text_region_candidate_sort_key,
    ):
        if any(_bbox_overlap_ratio(candidate.bbox, existing.bbox) >= 0.65 for existing in result):
            continue
        result.append(candidate)
    return result


def _structured_text_region_candidate_sort_key(candidate: RawTableEvidence) -> tuple[float, int, int, int, float]:
    raw_data = candidate.raw_data or []
    row_texts = [
        " ".join(str(cell or "").strip() for cell in row if str(cell or "").strip()).strip()
        for row in raw_data
        if isinstance(row, list)
    ]
    narrative_rows = sum(1 for text in row_texts if _looks_like_narrative_sentence(text))
    caption_rows = sum(1 for text in row_texts if _looks_like_table_caption_row_text(text))
    compact_rows = sum(1 for text in row_texts if text and len(text) <= 120 and not re.search(r"[.。！？!?；;]\s*$", text))
    line_support = len(candidate.drawings or [])
    return (
        float(candidate.bbox[1]),
        narrative_rows,
        -line_support,
        -caption_rows,
        -compact_rows,
    )


def _text_aligned_candidate_word_lanes(
    *,
    words: list[Any],
    page_width: float,
    page_height: float,
) -> list[list[Any]]:
    """Split text-aligned fallback evidence by layout lane when the page is two-column.

    The text-aligned detector is a residual fallback. On two-column literature
    pages, full-page row clustering can merge unrelated left/right lane content
    that merely shares a y position, then invent a wide grid. Only split when
    the text layer itself shows sustained left-only and right-only rows; full
    width IND tables keep flowing through as one lane.
    """
    if len(words) < 160:
        return [words]

    rows = _cluster_words_into_rows(words)
    if len(rows) < 18:
        return [words]

    split_x = float(page_width) / 2.0
    guard = max(12.0, min(28.0, page_width * 0.035))
    left_lane_rows = _cluster_words_into_rows(
        [word for word in words if _word_x1(word) <= split_x - guard and _word_text(word).strip()]
    )
    right_lane_rows = _cluster_words_into_rows(
        [word for word in words if _word_x0(word) >= split_x + guard and _word_text(word).strip()]
    )
    left_flow_rows = [
        row
        for row in left_lane_rows
        if len(row.get("words", []) or []) >= 3
        and (float(row.get("bbox", (0.0, 0.0, 0.0, 0.0))[2]) - float(row.get("bbox", (0.0, 0.0, 0.0, 0.0))[0]))
        >= page_width * 0.11
    ]
    right_flow_rows = [
        row
        for row in right_lane_rows
        if len(row.get("words", []) or []) >= 3
        and (float(row.get("bbox", (0.0, 0.0, 0.0, 0.0))[2]) - float(row.get("bbox", (0.0, 0.0, 0.0, 0.0))[0]))
        >= page_width * 0.11
    ]
    if min(len(left_flow_rows), len(right_flow_rows)) < 6:
        return [words]

    overlapping_flow_rows = 0
    for left in left_flow_rows:
        left_y0 = float(left.get("y0", 0.0))
        left_y1 = float(left.get("y1", 0.0))
        for right in right_flow_rows:
            right_y0 = float(right.get("y0", 0.0))
            right_y1 = float(right.get("y1", 0.0))
            if min(left_y1, right_y1) - max(left_y0, right_y0) > 0.0:
                overlapping_flow_rows += 1
                break
    if overlapping_flow_rows < 4:
        return [words]

    mid_words = [
        word
        for word in words
        if _word_x1(word) > split_x - guard
        and _word_x0(word) < split_x + guard
        and _word_text(word).strip()
    ]
    if len(mid_words) > len(words) * 0.22:
        return [words]

    left_words: list[Any] = []
    right_words: list[Any] = []
    for row in rows:
        row_words = [word for word in row.get("words", []) or [] if _word_text(word).strip()]
        if not row_words:
            continue
        row_left = [word for word in row_words if _word_x1(word) <= split_x - guard]
        row_right = [word for word in row_words if _word_x0(word) >= split_x + guard]
        for word in row_words:
            center_x = (_word_x0(word) + _word_x1(word)) / 2.0
            if _word_x1(word) <= split_x - guard:
                left_words.append(word)
            elif _word_x0(word) >= split_x + guard:
                right_words.append(word)
            elif row_right and not row_left:
                right_words.append(word)
            elif row_left and not row_right:
                left_words.append(word)
            elif center_x < split_x:
                left_words.append(word)
            else:
                right_words.append(word)

    left_words = _trim_lane_words_to_supported_edge(left_words, keep="left")
    right_words = _trim_lane_words_to_supported_edge(right_words, keep="right")
    lanes = [lane for lane in (left_words, right_words) if len(lane) >= 8]
    return lanes or [words]


def _trim_lane_words_to_supported_edge(words: list[Any], *, keep: str) -> list[Any]:
    if len(words) < 24:
        return words
    rows = _cluster_words_into_rows(words)
    row_edges = [
        float(row.get("bbox", (0.0, 0.0, 0.0, 0.0))[2 if keep == "left" else 0])
        for row in rows
        if len(row.get("words", []) or []) >= 3
    ]
    if len(row_edges) < 6:
        return words
    edge = _percentile(row_edges, 0.82 if keep == "left" else 0.18)
    tolerance = 14.0
    if keep == "left":
        return [word for word in words if _word_x0(word) <= edge + tolerance]
    return [word for word in words if _word_x1(word) >= edge - tolerance]


def _looks_like_text_aligned_table_opening_row(row: dict[str, Any], page_width: float) -> bool:
    segments = [segment for segment in row.get("segments", []) or [] if str(segment.get("text", "")).strip()]
    anchorish_lefts = _row_anchor_candidate_lefts(row)
    if len(anchorish_lefts) < 4:
        anchorish_lefts = _merge_anchor_lefts(
            anchorish_lefts,
            _compact_header_word_column_lefts(row, page_width),
            tolerance=max(8.0, min(18.0, page_width * 0.02)),
        )
    if len(anchorish_lefts) < 4 and len(segments) < 4:
        return False
    row_text = _row_text_from_segmented(row)
    if not row_text:
        return False
    if _looks_like_table_caption_row_text(row_text):
        return False
    if re.search(r"(?:目录|contents|table of contents)", row_text, re.IGNORECASE):
        return False
    if re.search(r"[.。！？!?；;]\s*$", row_text):
        return False
    bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(bbox) != 4 or float(bbox[2]) - float(bbox[0]) < page_width * 0.42:
        return False
    compact_segments = sum(1 for segment in segments if len(str(segment.get("text", "")).strip()) <= 64)
    if len(anchorish_lefts) >= 4 and compact_segments >= max(1, len(segments) - 1):
        return True
    return compact_segments >= max(4, len(segments) - 1)


def _detect_text_aligned_grid_column_anchors(
    rows: list[dict[str, Any]],
    page_width: float,
) -> list[float]:
    import statistics

    lefts: list[float] = []
    support_lefts: list[tuple[float, int]] = []
    first_row_lefts: list[float] = []
    for row_index, row in enumerate(rows):
        row_text = _row_text_from_segmented(row)
        if row_index > 0 and _looks_like_text_aligned_table_note_start(row_text):
            break
        row_lefts = _row_anchor_candidate_lefts(row)
        if row_index == 0 and len(row_lefts) < 4:
            row_lefts = _merge_anchor_lefts(
                row_lefts,
                _compact_header_word_column_lefts(row, page_width),
                tolerance=max(8.0, min(18.0, page_width * 0.02)),
            )
        for left in row_lefts:
            lefts.append(left)
            if row_index == 0:
                first_row_lefts.append(left)
        for word in _segmented_row_words(row):
            text = _word_text(word).strip()
            if not text:
                continue
            support_lefts.append((_word_x0(word), row_index))

    if len(first_row_lefts) < 4:
        return []
    all_lefts = sorted(lefts)
    clusters: list[list[float]] = []
    tolerance = max(22.0, min(34.0, page_width * 0.04))
    for left in all_lefts:
        if not clusters or abs(left - statistics.median(clusters[-1])) > tolerance:
            clusters.append([left])
        else:
            clusters[-1].append(left)

    anchors: list[float] = []
    for cluster in clusters:
        support = len(cluster)
        median_left = float(statistics.median(cluster))
        first_support = any(abs(value - median_left) <= tolerance for value in first_row_lefts)
        if support >= 2 or first_support:
            anchors.append(median_left)

    anchors = sorted(anchors)
    anchors = _add_repeated_word_left_anchors(
        anchors,
        support_lefts,
        first_row_lefts,
        page_width=page_width,
    )
    anchors = _prune_low_support_inserted_text_anchors(
        anchors,
        support_lefts,
        first_row_lefts,
        page_width=page_width,
    )
    pruned: list[float] = []
    min_gap = max(18.0, min(32.0, page_width * 0.035))
    for anchor in anchors:
        if not pruned or anchor - pruned[-1] >= min_gap:
            pruned.append(anchor)
            continue
        pruned[-1] = (pruned[-1] + anchor) / 2.0
    return pruned


def _merge_anchor_lefts(lefts: list[float], extra_lefts: list[float], *, tolerance: float) -> list[float]:
    merged = sorted([*lefts, *extra_lefts])
    if not merged:
        return []
    result: list[float] = []
    for left in merged:
        if result and abs(float(left) - float(result[-1])) <= tolerance:
            result[-1] = (float(result[-1]) + float(left)) / 2.0
            continue
        result.append(float(left))
    return result


def _compact_header_word_column_lefts(row: dict[str, Any], page_width: float) -> list[float]:
    words = _segmented_row_words(row)
    if len(words) < 4:
        return []
    row_text = _row_text_from_segmented(row)
    if not row_text or _looks_like_table_caption_row_text(row_text) or re.search(r"[.。！？!?；;]\s*$", row_text):
        return []
    row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(row_bbox) != 4 or float(row_bbox[2]) - float(row_bbox[0]) < page_width * 0.35:
        return []
    heights = [_word_height(word) for word in words if _word_height(word) > 0]
    median_height = _percentile(heights, 0.5) if heights else 8.0
    min_gap = max(6.0, min(14.0, median_height * 0.75))
    lefts: list[float] = []
    previous = None
    for word in words:
        text = _word_text(word).strip()
        if not text:
            previous = word
            continue
        if previous is None:
            lefts.append(_word_x0(word))
            previous = word
            continue
        gap = _word_x0(word) - _word_x1(previous)
        if gap >= min_gap:
            lefts.append(_word_x0(word))
        previous = word
    return lefts


def _prune_low_support_inserted_text_anchors(
    anchors: list[float],
    support_lefts: list[tuple[float, int]],
    first_row_lefts: list[float],
    *,
    page_width: float,
) -> list[float]:
    if len(anchors) <= 4:
        return list(anchors)
    tolerance = max(8.0, min(16.0, page_width * 0.018))
    first_tol = max(10.0, min(22.0, page_width * 0.026))
    result: list[float] = []
    for idx, anchor in enumerate(anchors):
        rows = {
            row_index
            for left, row_index in support_lefts
            if abs(float(left) - float(anchor)) <= tolerance
        }
        first_supported = any(abs(float(anchor) - float(first_left)) <= first_tol for first_left in first_row_lefts)
        previous_gap = float(anchor) - float(anchors[idx - 1]) if idx > 0 else 999.0
        next_gap = float(anchors[idx + 1]) - float(anchor) if idx + 1 < len(anchors) else 999.0
        narrow_between_strong_neighbors = previous_gap < 72.0 or next_gap < 72.0
        if (
            idx > 0
            and idx + 1 < len(anchors)
            and not first_supported
            and narrow_between_strong_neighbors
            and len(rows) <= 2
        ):
            continue
        result.append(anchor)
    return result


def _add_repeated_word_left_anchors(
    anchors: list[float],
    support_lefts: list[tuple[float, int]],
    first_row_lefts: list[float],
    *,
    page_width: float,
) -> list[float]:
    import statistics

    if not support_lefts:
        return list(anchors)
    tolerance = max(8.0, min(16.0, page_width * 0.018))
    clusters: list[list[tuple[float, int]]] = []
    for left, row_index in sorted(support_lefts, key=lambda item: item[0]):
        if not clusters or abs(left - statistics.median(value for value, _ in clusters[-1])) > tolerance:
            clusters.append([(left, row_index)])
        else:
            clusters[-1].append((left, row_index))

    result = list(anchors)
    existing_tol = max(10.0, min(20.0, page_width * 0.024))
    for cluster in clusters:
        row_support = {row_index for _left, row_index in cluster}
        if len(row_support) < 2:
            continue
        median_left = float(statistics.median(left for left, _row_index in cluster))
        if any(abs(median_left - existing) <= existing_tol for existing in result):
            continue
        first_row_supported = any(abs(median_left - first_left) <= existing_tol for first_left in first_row_lefts)
        compact_value_support = sum(
            1
            for left, row_index in cluster
            if row_index > 0 and abs(left - median_left) <= tolerance
        )
        if not first_row_supported and compact_value_support < 2:
            continue
        if result and (median_left < min(result) - 6.0 or median_left > max(result) + page_width * 0.12):
            continue
        result.append(median_left)
    return sorted(result)


def _row_anchor_candidate_lefts(row: dict[str, Any]) -> list[float]:
    """Return word left edges that behave like starts of visual columns.

    A segment start alone is too coarse for compact borderless IND tables
    because several columns can be printed with small but still meaningful word
    gaps. Conversely, every word left edge is too noisy because multi-word cells
    such as ``Sponsor Inc.`` should not create a new column at the second word.
    """
    words = _segmented_row_words(row)
    if not words:
        return []
    heights = [_word_height(word) for word in words if _word_height(word) > 0]
    median_height = _percentile(heights, 0.5) if heights else 8.0
    large_gap_floor = max(14.0, min(28.0, median_height * 1.75))
    lefts: list[float] = []
    previous = None
    for word in words:
        text = _word_text(word).strip()
        if not text:
            previous = word
            continue
        if previous is None:
            lefts.append(_word_x0(word))
            previous = word
            continue
        gap = _word_x0(word) - _word_x1(previous)
        if gap >= large_gap_floor:
            lefts.append(_word_x0(word))
        previous = word
    return lefts


def _project_segmented_row_to_text_grid(
    row: dict[str, Any],
    column_anchors: list[float],
) -> tuple[list[str | None], set[int]]:
    col_count = len(column_anchors)
    projected: list[list[str]] = [[] for _ in range(col_count)]
    matched_cols: set[int] = set()
    segments = _refine_row_segments_against_column_anchors(
        row.get("segments", []) or [],
        column_anchors,
        split_single_segment_by_column_anchors=True,
    )
    if len(segments) < min(col_count, 4):
        word_segments = _segments_from_words_by_column_anchors(row, column_anchors)
        if len(word_segments) > len(segments):
            segments = word_segments
    tolerance = _text_grid_anchor_tolerance(column_anchors)
    for segment in segments:
        text = str(segment.get("text", "") or "").strip()
        if not text:
            continue
        bbox = tuple(segment.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if len(bbox) != 4:
            continue
        x0 = float(bbox[0])
        if _place_segment_between_neighbor_anchors(
            projected,
            matched_cols,
            text,
            x0,
            column_anchors,
            tolerance,
        ):
            continue
        nearest = min(range(col_count), key=lambda idx: abs(x0 - float(column_anchors[idx])))
        nearest_gap = abs(x0 - float(column_anchors[nearest]))
        if nearest_gap > tolerance:
            if (
                nearest_gap > tolerance * 1.35
                or nearest in matched_cols
                or projected[nearest]
                or not _is_compact_text_aligned_cell_text(text)
            ):
                continue
        projected[nearest].append(text)
        matched_cols.add(nearest)
    return ([" ".join(parts).strip() or None for parts in projected], matched_cols)


def _is_compact_text_aligned_cell_text(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if not re.search(r"[A-Za-z\u4e00-\u9fff0-9]", candidate):
        return False
    if len(candidate) > 64 or len(candidate.split()) > 4:
        return False
    if re.search(r"[.。！？!?；;]\s*$", candidate):
        return False
    return True


def _place_segment_between_neighbor_anchors(
    projected: list[list[str]],
    matched_cols: set[int],
    text: str,
    x0: float,
    column_anchors: list[float],
    tolerance: float,
) -> bool:
    if not text or len(column_anchors) < 3:
        return False
    nearest = min(range(len(column_anchors)), key=lambda col_idx: abs(x0 - float(column_anchors[col_idx])))
    if abs(x0 - float(column_anchors[nearest])) <= tolerance:
        return False
    for idx in range(1, len(column_anchors) - 1):
        anchor = float(column_anchors[idx])
        if abs(x0 - anchor) <= tolerance:
            return False
        left = float(column_anchors[idx - 1])
        right = float(column_anchors[idx + 1])
        if not (left < x0 < right):
            continue
        left_gap = abs(x0 - left)
        right_gap = abs(right - x0)
        if min(left_gap, right_gap) > tolerance * 1.35:
            continue
        if idx in matched_cols:
            continue
        if projected[idx]:
            continue
        if not _is_compact_text_aligned_cell_text(text):
            return False
        projected[idx].append(text)
        matched_cols.add(idx)
        return True
    return False


def _segments_from_words_by_column_anchors(
    row: dict[str, Any],
    column_anchors: list[float],
) -> list[dict[str, Any]]:
    words = _segmented_row_words(row)
    if not words or not column_anchors:
        return []
    start_tol = max(10.0, min(22.0, _text_grid_anchor_tolerance(column_anchors) * 0.55))
    chunks: list[list[Any]] = []
    current_chunk: list[Any] = []
    current_anchor_index: int | None = None
    for word in words:
        x0 = _word_x0(word)
        nearest = min(range(len(column_anchors)), key=lambda idx: abs(x0 - float(column_anchors[idx])))
        starts_new_anchor = abs(x0 - float(column_anchors[nearest])) <= start_tol
        if current_chunk and starts_new_anchor and nearest != current_anchor_index:
            if _is_short_text_connector_word(word):
                current_chunk.append(word)
                continue
            chunks.append(current_chunk)
            current_chunk = [word]
            current_anchor_index = nearest
            continue
        if not current_chunk:
            current_anchor_index = nearest if starts_new_anchor else current_anchor_index
        current_chunk.append(word)
    return [_build_segment(chunk) for chunk in chunks if chunk]


def _is_short_text_connector_word(word: Any) -> bool:
    text = _word_text(word).strip()
    if not text:
        return False
    return bool(re.fullmatch(r"(?:和|及|与|或|and|or|to|of|for|&)", text, re.IGNORECASE))


def _segmented_row_words(row: dict[str, Any]) -> list[Any]:
    words = list(row.get("words", []) or [])
    if not words:
        words = [
            word
            for segment in row.get("segments", []) or []
            for word in segment.get("words", []) or []
        ]
    return sorted(words, key=lambda item: (_word_x0(item), _word_y0(item)))


def _text_grid_anchor_tolerance(column_anchors: list[float]) -> float:
    gaps = [
        abs(float(right) - float(left))
        for left, right in zip(column_anchors, column_anchors[1:])
        if abs(float(right) - float(left)) > 0
    ]
    median_gap = _percentile(gaps, 0.5) if gaps else 48.0
    return max(18.0, min(42.0, median_gap * 0.45))


def _is_text_aligned_group_row(
    row: dict[str, Any],
    projected_row: list[str | None],
    matched_cols: set[int],
    all_rows: list[dict[str, Any]],
    row_index: int,
    column_anchors: list[float],
) -> bool:
    if matched_cols != {0}:
        return False
    text = str(projected_row[0] or "").strip()
    if not text or len(text) > 80:
        return False
    if _looks_like_text_aligned_table_note_start(text):
        return False
    if re.search(r"[.。！？!?；;]\s*$", text):
        return False
    for next_row in all_rows[row_index + 1 : min(len(all_rows), row_index + 4)]:
        next_projected, next_cols = _project_segmented_row_to_text_grid(next_row, column_anchors)
        if len(next_cols) >= 2 and any(str(cell or "").strip() for cell in next_projected[1:]):
            return True
    return False


def _is_text_aligned_continuation_row(
    previous_entries: list[dict[str, Any]],
    projected_row: list[str | None],
    matched_cols: set[int],
    row_bbox: tuple[float, float, float, float],
    column_anchors: list[float],
) -> bool:
    if not previous_entries or not matched_cols:
        return False
    if len(matched_cols) > max(2, len(column_anchors) // 3):
        return False
    previous = previous_entries[-1]
    previous_bbox = tuple(previous.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(previous_bbox) != 4:
        return False
    row_height = max(1.0, float(row_bbox[3]) - float(row_bbox[1]))
    if float(row_bbox[1]) - float(previous_bbox[3]) > max(8.0, row_height * 1.15):
        return False
    text_values = [str(projected_row[col] or "").strip() for col in matched_cols]
    if not any(text_values):
        return False
    if matched_cols == {0}:
        text = text_values[0]
        return bool(text and (len(text) > 28 or re.search(r"[,，、(/（]", text)))
    if 0 not in matched_cols:
        return True
    return len(matched_cols) >= 2


def _merge_text_aligned_continuation_entries(
    entries: list[dict[str, Any]],
    col_count: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        data = list(entry.get("data") or [])
        while len(data) < col_count:
            data.append(None)
        matched_cols = set(entry.get("matched_cols") or [])
        should_merge = False
        if merged:
            previous = merged[-1]
            previous_data = list(previous.get("data") or [])
            previous_nonempty = {idx for idx, value in enumerate(previous_data) if str(value or "").strip()}
            next_entry = entries[index + 1] if index + 1 < len(entries) else None
            current_col0 = str(data[0] or "").strip() if data else ""
            current_is_group = (
                matched_cols == {0}
                and current_col0
                and len(current_col0) <= 80
                and next_entry is not None
                and len(set(next_entry.get("matched_cols") or [])) >= 2
                and not re.search(r"[.。！？!?；;]\s*$", current_col0)
            )
            if not current_is_group:
                if 0 not in matched_cols and len(matched_cols) <= max(2, col_count // 3):
                    should_merge = True
                elif matched_cols and matched_cols.issubset(previous_nonempty) and len(matched_cols) <= 2:
                    should_merge = True
                elif 0 in matched_cols and len(matched_cols) >= 2 and len(matched_cols) <= 2:
                    should_merge = True

        if should_merge and merged:
            previous = merged[-1]
            previous_data = list(previous.get("data") or [])
            for col_index, value in enumerate(data):
                text = str(value or "").strip()
                if not text:
                    continue
                existing = str(previous_data[col_index] or "").strip()
                previous_data[col_index] = f"{existing} {text}".strip() if existing else text
            previous["data"] = previous_data
            previous["matched_cols"] = set(previous.get("matched_cols") or set()) | matched_cols
            previous["bbox"] = _bbox_union_loose([tuple(previous.get("bbox")), tuple(entry.get("bbox"))])
            previous["rows"] = list(previous.get("rows") or []) + [entry.get("row")]
            continue

        new_entry = dict(entry)
        new_entry["data"] = data
        new_entry["rows"] = [entry.get("row")]
        merged.append(new_entry)
    return merged


def _text_aligned_candidate_has_table_shape(
    entries: list[dict[str, Any]],
    col_count: int,
    *,
    boundary_caption: dict[str, Any] | None = None,
) -> bool:
    min_rows = 2 if boundary_caption is not None else 3
    if len(entries) < min_rows or col_count < 4:
        return False
    filled_counts = [
        sum(1 for cell in entry.get("data", []) if str(cell or "").strip())
        for entry in entries
    ]
    fullish_rows = sum(1 for count in filled_counts if count >= max(3, int(col_count * 0.55)))
    multi_rows = sum(1 for count in filled_counts if count >= 2)
    min_multi_rows = 2 if boundary_caption is not None else max(3, len(entries) // 2)
    if fullish_rows < 2 or multi_rows < min_multi_rows:
        return False
    first_row_text = " ".join(str(cell or "").strip() for cell in entries[0].get("data", []) if str(cell or "").strip())
    if re.search(r"[.。！？!?；;]\s*$", first_row_text):
        return False
    if boundary_caption is None and _text_aligned_entries_look_like_multicolumn_prose_projection(entries, col_count):
        return False
    right_tail_values = [
        str(cell or "").strip()
        for entry in entries[1:]
        for cell in (entry.get("data", [])[-3:] if entry.get("data") else [])
        if str(cell or "").strip()
    ]
    structured_tail = sum(
        1
        for text in right_tail_values
        if re.search(r"\b\d{1,6}\b", text)
        or re.search(r"\b(?:inc|ltd|llc|corp|sponsor|glp)\b", text, re.IGNORECASE)
        or len(text) <= 12
    )
    matrix_like = fullish_rows >= max(4, len(entries) // 2)
    if boundary_caption is not None and fullish_rows >= 2:
        return True
    return matrix_like or structured_tail >= 3


def _text_aligned_entries_look_like_multicolumn_prose_projection(
    entries: list[dict[str, Any]],
    col_count: int,
) -> bool:
    if col_count < 8 or len(entries) < 4:
        return False
    total_cells = 0
    filled_cells = 0
    prose_rows = 0
    numeric_rows = 0
    for entry in entries:
        cells = [str(cell or "").strip() for cell in entry.get("data", [])]
        total_cells += len(cells)
        filled = [cell for cell in cells if cell]
        filled_cells += len(filled)
        joined = " ".join(filled)
        word_count = len(re.findall(r"[A-Za-z\u4e00-\u9fff]+", joined))
        numeric_count = len(re.findall(r"\d+(?:[.,]\d+)?%?", joined))
        sentence_fragment_cells = sum(
            1
            for cell in filled
            if len(cell.split()) >= 2
            and re.search(r"\b(?:the|and|of|to|in|that|with|from|when|where|as|for)\b", cell, re.IGNORECASE)
        )
        sentence_punct_cells = sum(1 for cell in filled if re.search(r"[.。！？!?；;]\s*$", cell))
        if word_count >= 8 and (sentence_fragment_cells >= 1 or sentence_punct_cells >= 1):
            prose_rows += 1
        if numeric_count >= 2 and len(filled) >= 3:
            numeric_rows += 1

    if total_cells <= 0:
        return False
    empty_ratio = 1.0 - (filled_cells / total_cells)
    return empty_ratio >= 0.30 and prose_rows >= max(3, len(entries) // 2) and numeric_rows <= 2


def _build_raw_rows_from_projected_entries(
    entries: list[dict[str, Any]],
    col_count: int,
) -> tuple[list[RawRow], list[list[str | None]], list[RawSpan], list[RawWord]]:
    raw_rows: list[RawRow] = []
    raw_data: list[list[str | None]] = []
    spans: list[RawSpan] = []
    grouped_words: list[RawWord] = []
    for row_idx, entry in enumerate(entries):
        row_bbox = tuple(entry.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        data = list(entry.get("data") or [])
        while len(data) < col_count:
            data.append(None)
        row_cells: list[RawCell] = []
        for col_idx, text in enumerate(data[:col_count]):
            cell_text = str(text or "").strip() or None
            row_cells.append(
                RawCell(
                    physical_col=col_idx,
                    physical_row=row_idx,
                    text=cell_text,
                    bbox=row_bbox if cell_text else None,
                )
            )
        for source_row in entry.get("rows") or []:
            if not isinstance(source_row, dict):
                continue
            for word in source_row.get("words", []) or []:
                text = _word_text(word).strip()
                if not text:
                    continue
                spans.append(RawSpan(text=text, x0=_word_x0(word), y0=_word_y0(word), x1=_word_x1(word), y1=_word_y1(word)))
                grouped_words.append(RawWord(text=text, x0=_word_x0(word), y0=_word_y0(word), x1=_word_x1(word), y1=_word_y1(word)))
        raw_rows.append(
            RawRow(
                physical_row=row_idx,
                cells=row_cells,
                bbox=row_bbox,
                y0=float(row_bbox[1]),
                y1=float(row_bbox[3]),
            )
        )
        raw_data.append(data[:col_count])
    return raw_rows, raw_data, spans, grouped_words


def _looks_like_text_aligned_table_stop_row(row: dict[str, Any], page_height: float) -> bool:
    row_text = _row_text_from_segmented(row)
    bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(bbox) != 4:
        return False
    if float(bbox[1]) >= page_height * 0.86 and re.fullmatch(r"\d{1,4}", row_text):
        return True
    return False


def _looks_like_text_aligned_table_note_start(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return False
    if re.match(r"^[#*$+\u2020\u2021]?\s*[a-zA-Z]\s*[-:：.)、]", cleaned):
        return True
    if re.match(r"^[#*$+\u2020\u2021]\s*", cleaned):
        return True
    return bool(re.match(r"^(?:附加信息|备注|注|说明|Note|Notes)\s*[:：]", cleaned, re.IGNORECASE))


def _row_text_from_segmented(row: dict[str, Any]) -> str:
    return " ".join(
        str(segment.get("text", "") or "").strip()
        for segment in row.get("segments", []) or []
        if str(segment.get("text", "") or "").strip()
    ).strip()


def _bbox_union_loose(
    bboxes: list[tuple[float, float, float, float] | None],
) -> tuple[float, float, float, float]:
    valid = [bbox for bbox in bboxes if bbox and len(bbox) == 4]
    if not valid:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(float(bbox[0]) for bbox in valid),
        min(float(bbox[1]) for bbox in valid),
        max(float(bbox[2]) for bbox in valid),
        max(float(bbox[3]) for bbox in valid),
    )


def _deduplicate_text_aligned_candidates(candidates: list[RawTableEvidence]) -> list[RawTableEvidence]:
    result: list[RawTableEvidence] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            float(item.bbox[1]),
            -int(item.physical_row_count or 0),
            -int(item.physical_col_count or 0),
            -(float(item.bbox[3]) - float(item.bbox[1])),
        ),
    ):
        if any(_bbox_overlap_ratio(candidate.bbox, existing.bbox) >= 0.55 for existing in result):
            continue
        result.append(candidate)
    return result


def _cluster_horizontal_rule_bands(
    drawings: list[dict[str, Any]],
    page_width: float,
) -> list[dict[str, Any]]:
    raw_segments: list[tuple[float, float, float, float]] = []
    min_width = max(24.0, page_width * 0.035)
    for drawing in drawings:
        bbox = _coerce_bbox(drawing.get("rect") if isinstance(drawing, dict) else None)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        if width < min_width or height > max(4.0, width * 0.08):
            continue
        raw_segments.append(bbox)
    if not raw_segments:
        return []

    bands: list[list[tuple[float, float, float, float]]] = []
    for segment in sorted(raw_segments, key=lambda item: ((item[1] + item[3]) / 2.0, item[0])):
        yc = (segment[1] + segment[3]) / 2.0
        if not bands:
            bands.append([segment])
            continue
        last_yc = sum((item[1] + item[3]) / 2.0 for item in bands[-1]) / len(bands[-1])
        if abs(yc - last_yc) <= 3.0:
            bands[-1].append(segment)
        else:
            bands.append([segment])

    merged: list[dict[str, Any]] = []
    for segments in bands:
        x0 = min(item[0] for item in segments)
        y0 = min(item[1] for item in segments)
        x1 = max(item[2] for item in segments)
        y1 = max(item[3] for item in segments)
        coverage = sum(max(0.0, item[2] - item[0]) for item in segments) / max(1.0, x1 - x0)
        if x1 - x0 < page_width * 0.18:
            continue
        if coverage < 0.45:
            continue
        merged.append(
            {
                "bbox": (x0, y0, x1, y1),
                "segments": segments,
                "yc": (y0 + y1) / 2.0,
                "coverage": coverage,
            }
        )
    return merged


def _cluster_vertical_rule_bands(
    drawings: list[dict[str, Any]],
    page_height: float,
) -> list[dict[str, Any]]:
    raw_segments: list[tuple[float, float, float, float]] = []
    min_height = max(24.0, page_height * 0.035)
    for drawing in drawings:
        bbox = _coerce_bbox(drawing.get("rect") if isinstance(drawing, dict) else None)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        if height < min_height or width > max(4.0, height * 0.08):
            continue
        raw_segments.append(bbox)
    if not raw_segments:
        return []

    bands: list[list[tuple[float, float, float, float]]] = []
    for segment in sorted(raw_segments, key=lambda item: ((item[0] + item[2]) / 2.0, item[1])):
        xc = (segment[0] + segment[2]) / 2.0
        if not bands:
            bands.append([segment])
            continue
        last_xc = sum((item[0] + item[2]) / 2.0 for item in bands[-1]) / len(bands[-1])
        if abs(xc - last_xc) <= 3.0:
            bands[-1].append(segment)
        else:
            bands.append([segment])

    merged: list[dict[str, Any]] = []
    for segments in bands:
        x0 = min(item[0] for item in segments)
        y0 = min(item[1] for item in segments)
        x1 = max(item[2] for item in segments)
        y1 = max(item[3] for item in segments)
        coverage = sum(max(0.0, item[3] - item[1]) for item in segments) / max(1.0, y1 - y0)
        if y1 - y0 < page_height * 0.12:
            continue
        if coverage < 0.45:
            continue
        merged.append(
            {
                "bbox": (x0, y0, x1, y1),
                "segments": segments,
                "xc": (x0 + x1) / 2.0,
                "coverage": coverage,
            }
        )
    return merged


def _visual_structure_candidate_regions(
    *,
    rows: list[dict[str, Any]],
    horizontal_bands: list[dict[str, Any]],
    vertical_bands: list[dict[str, Any]],
    page_height: float,
    page_width: float,
    occupied_bboxes: list[tuple[float, float, float, float]],
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for start_idx in range(0, len(horizontal_bands) - 1):
        for end_idx in range(start_idx + 1, min(len(horizontal_bands), start_idx + 6)):
            band_slice = horizontal_bands[start_idx : end_idx + 1]
            top_band = band_slice[0]
            bottom_band = band_slice[-1]
            x0 = min(float(band["bbox"][0]) for band in band_slice)
            x1 = max(float(band["bbox"][2]) for band in band_slice)
            top_y = float(top_band["bbox"][1])
            bottom_y = float(bottom_band["bbox"][3])
            if x1 - x0 < page_width * 0.18 or bottom_y - top_y < 22.0:
                continue
            region_bbox = (x0, top_y, x1, bottom_y)
            if any(_bbox_overlap_ratio(region_bbox, item) >= 0.25 for item in occupied_bboxes):
                continue
            inside_rows = [
                row
                for row in rows
                if row["y1"] >= top_y - 3.0
                and row["y0"] <= bottom_y + 3.0
                and _horizontal_overlap(row["bbox"], region_bbox) >= 0.10
            ]
            if len(inside_rows) < 3:
                continue
            body_rows = [
                row
                for row in inside_rows
                if row["y0"] >= top_y - 3.0
                and ((float(row["y0"]) + float(row["y1"])) / 2.0) <= bottom_y + 1.0
            ]
            if len(body_rows) < 3:
                continue
            regions.append(
                {
                    "bbox": _expand_visual_region_bbox(region_bbox, body_rows),
                    "all_rows": inside_rows,
                    "body_rows": body_rows,
                    "rule_bands": band_slice,
                    "vertical_bands": [
                        band
                        for band in vertical_bands
                        if _vertical_band_overlaps_region(band, region_bbox)
                    ],
                }
            )

    if len(vertical_bands) >= 2:
        vertical_regions = _visual_structure_regions_from_vertical_rules(
            rows=rows,
            vertical_bands=vertical_bands,
            horizontal_bands=horizontal_bands,
            page_width=page_width,
            occupied_bboxes=occupied_bboxes,
        )
        regions.extend(vertical_regions)

    return _deduplicate_visual_structure_regions(regions)


def _visual_structure_regions_from_vertical_rules(
    *,
    rows: list[dict[str, Any]],
    vertical_bands: list[dict[str, Any]],
    horizontal_bands: list[dict[str, Any]],
    page_width: float,
    occupied_bboxes: list[tuple[float, float, float, float]],
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for start_idx in range(0, len(vertical_bands) - 1):
        for end_idx in range(start_idx + 1, min(len(vertical_bands), start_idx + 6)):
            band_slice = vertical_bands[start_idx : end_idx + 1]
            x0 = min(float(band["bbox"][0]) for band in band_slice)
            x1 = max(float(band["bbox"][2]) for band in band_slice)
            y0 = min(float(band["bbox"][1]) for band in band_slice)
            y1 = max(float(band["bbox"][3]) for band in band_slice)
            if x1 - x0 < page_width * 0.18 or y1 <= y0:
                continue
            region_bbox = (x0, y0, x1, y1)
            if any(_bbox_overlap_ratio(region_bbox, item) >= 0.25 for item in occupied_bboxes):
                continue
            body_rows = [
                row
                for row in rows
                if row["y1"] >= y0 - 5.0
                and row["y0"] <= y1 + 5.0
                and _horizontal_overlap(row["bbox"], region_bbox) >= 0.10
            ]
            if len(body_rows) < 4:
                continue
            matching_horizontal = [
                band
                for band in horizontal_bands
                if float(band["bbox"][3]) >= y0 - 4.0
                and float(band["bbox"][1]) <= y1 + 4.0
                and _horizontal_overlap(tuple(band["bbox"]), region_bbox) >= 0.30
            ]
            if len(matching_horizontal) < 2:
                continue
            horizontal_y0 = min(float(band["bbox"][1]) for band in matching_horizontal)
            horizontal_y1 = max(float(band["bbox"][3]) for band in matching_horizontal)
            effective_verticals = [
                band
                for band in band_slice
                if _vertical_band_overlaps_y_span(band, horizontal_y0, horizontal_y1)
            ]
            if len(effective_verticals) < 2:
                continue
            x0 = min(
                min(float(band["bbox"][0]) for band in effective_verticals),
                min(float(band["bbox"][0]) for band in matching_horizontal),
            )
            x1 = max(
                max(float(band["bbox"][2]) for band in effective_verticals),
                max(float(band["bbox"][2]) for band in matching_horizontal),
            )
            effective_x0 = min(float(band["bbox"][0]) for band in matching_horizontal)
            effective_x1 = max(float(band["bbox"][2]) for band in matching_horizontal)
            header_y0 = _visual_structure_header_top_from_rows(
                rows,
                x0=effective_x0,
                x1=effective_x1,
                top_y=horizontal_y0,
                column_anchors=_visual_structure_content_column_anchors(
                    rows,
                    x0=effective_x0,
                    x1=effective_x1,
                    y0=horizontal_y0,
                    y1=horizontal_y1,
                ),
            )
            y0 = header_y0 if header_y0 is not None else horizontal_y0
            y1 = horizontal_y1
            region_bbox = (x0, y0, x1, y1)
            body_rows = [
                row
                for row in rows
                if row["y1"] >= y0 - 5.0
                and row["y0"] <= y1 + 5.0
                and _horizontal_overlap(row["bbox"], region_bbox) >= 0.10
            ]
            if len(body_rows) < 4:
                continue
            regions.append(
                {
                    "bbox": _expand_visual_region_bbox(region_bbox, body_rows),
                    "all_rows": body_rows,
                    "body_rows": body_rows,
                    "rule_bands": matching_horizontal,
                    "vertical_bands": effective_verticals,
                }
            )
    return regions


def _vertical_band_overlaps_y_span(
    band: dict[str, Any],
    y0: float,
    y1: float,
) -> bool:
    bbox = tuple(band.get("bbox") or ())
    if len(bbox) != 4:
        return False
    overlap = max(0.0, min(float(bbox[3]), y1) - max(float(bbox[1]), y0))
    span_height = max(1.0, y1 - y0)
    band_height = max(1.0, float(bbox[3]) - float(bbox[1]))
    return overlap >= min(18.0, span_height * 0.15) or overlap / band_height >= 0.20


def _visual_structure_header_top_from_rows(
    rows: list[dict[str, Any]],
    *,
    x0: float,
    x1: float,
    top_y: float,
    column_anchors: list[float],
) -> float | None:
    if not column_anchors:
        return None
    candidate_rows = [
        row
        for row in rows
        if 0.0 <= top_y - float(row.get("y1", 0.0)) <= 32.0
        and _horizontal_overlap(tuple(row.get("bbox") or (0.0, 0.0, 0.0, 0.0)), (x0, top_y, x1, top_y + 1.0)) >= 0.10
    ]
    if not candidate_rows:
        return None
    segmented_rows = _segment_group_rows(candidate_rows)
    supported: list[dict[str, Any]] = []
    anchor_candidates = _normalize_visual_structure_header_anchors(column_anchors, x0, x1)
    for row, segmented in zip(candidate_rows, segmented_rows):
        matched_cols = _visual_header_row_matched_columns(segmented, anchor_candidates)
        filled = len(matched_cols)
        row_text = _row_text(row)
        if filled >= 2 and len(matched_cols) >= 2 and not re.search(r"[.。！？!?；;]\s*$", row_text):
            supported.append(row)
    if not supported:
        return None
    return min(float(row.get("y0", top_y)) for row in supported)


def _visual_structure_content_column_anchors(
    rows: list[dict[str, Any]],
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> list[float]:
    body_rows = [
        row
        for row in rows
        if row["y1"] >= y0 - 4.0
        and row["y0"] <= y1 + 4.0
        and _horizontal_overlap(tuple(row.get("bbox") or (0.0, 0.0, 0.0, 0.0)), (x0, y0, x1, y1)) >= 0.10
    ]
    segmented = _segment_group_rows(body_rows)
    anchors = _detect_stable_segment_column_anchors(segmented, (x0, y0, x1, y1))
    if len(anchors) >= 2:
        return anchors
    return _detect_segment_column_anchors(segmented)


def _normalize_visual_structure_header_anchors(
    anchors: list[float],
    x0: float,
    x1: float,
) -> list[float]:
    values = sorted(float(anchor) for anchor in anchors if x0 - 8.0 <= float(anchor) <= x1 + 8.0)
    if len(values) >= 2:
        return values
    return [x0, x1]


def _visual_header_row_matched_columns(
    row: dict[str, Any],
    anchors: list[float],
) -> set[int]:
    if not anchors:
        return set()
    matched: set[int] = set()
    tolerance = max(_text_grid_anchor_tolerance(anchors), _visual_header_anchor_center_tolerance(anchors))
    for segment in row.get("segments", []) or []:
        text = str(segment.get("text", "") or "").strip()
        if not text:
            continue
        bbox = tuple(segment.get("bbox") or ())
        if len(bbox) != 4:
            continue
        x0 = float(bbox[0])
        center = (float(bbox[0]) + float(bbox[2])) / 2.0
        nearest_left = min(range(len(anchors)), key=lambda idx: abs(x0 - float(anchors[idx])))
        nearest_center = min(range(len(anchors)), key=lambda idx: abs(center - float(anchors[idx])))
        if abs(x0 - float(anchors[nearest_left])) <= tolerance:
            matched.add(nearest_left)
        elif abs(center - float(anchors[nearest_center])) <= tolerance:
            matched.add(nearest_center)
    return matched


def _visual_header_anchor_center_tolerance(anchors: list[float]) -> float:
    gaps = [
        abs(float(right) - float(left))
        for left, right in zip(anchors, anchors[1:])
        if abs(float(right) - float(left)) > 0
    ]
    median_gap = _percentile(gaps, 0.5) if gaps else 80.0
    return max(28.0, min(96.0, median_gap * 0.38))


def _vertical_band_overlaps_region(
    band: dict[str, Any],
    region_bbox: tuple[float, float, float, float],
) -> bool:
    bbox = tuple(band.get("bbox") or ())
    if len(bbox) != 4:
        return False
    return (
        float(bbox[0]) <= region_bbox[2] + 4.0
        and float(bbox[2]) >= region_bbox[0] - 4.0
        and float(bbox[1]) <= region_bbox[3] + 4.0
        and float(bbox[3]) >= region_bbox[1] - 4.0
    )


def _expand_visual_region_bbox(
    region_bbox: tuple[float, float, float, float],
    rows: list[dict[str, Any]],
) -> tuple[float, float, float, float]:
    if not rows:
        return region_bbox
    rows_bbox = _rows_bbox(rows)
    if rows_bbox is None:
        return region_bbox
    return (
        min(float(region_bbox[0]), float(rows_bbox[0])),
        min(float(region_bbox[1]), float(rows_bbox[1])),
        max(float(region_bbox[2]), float(rows_bbox[2])),
        max(float(region_bbox[3]), float(rows_bbox[3])),
    )


def _deduplicate_visual_structure_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for region in sorted(
        regions,
        key=lambda item: (
            float(item["bbox"][1]),
            -(float(item["bbox"][3]) - float(item["bbox"][1])),
            -len(item.get("body_rows") or []),
        ),
    ):
        bbox = tuple(region.get("bbox") or ())
        if len(bbox) != 4:
            continue
        if any(_bbox_overlap_ratio(bbox, tuple(existing["bbox"])) >= 0.65 for existing in result):
            continue
        result.append(region)
    return result


def _horizontal_rule_candidate_bbox(
    rule_bands: list[dict[str, Any]],
) -> tuple[float, float, float, float] | None:
    if len(rule_bands) < 2:
        return None
    x0 = max(float(band["bbox"][0]) for band in rule_bands)
    x1 = min(float(band["bbox"][2]) for band in rule_bands)
    if x1 - x0 < 40.0:
        x0 = min(float(band["bbox"][0]) for band in rule_bands)
        x1 = max(float(band["bbox"][2]) for band in rule_bands)
    return (
        x0,
        min(float(band["bbox"][1]) for band in rule_bands),
        x1,
        max(float(band["bbox"][3]) for band in rule_bands),
    )


def _caption_rows_above_rule(
    rows: list[dict[str, Any]],
    top_y: float,
    table_bbox: tuple[float, float, float, float],
    page_height: float,
) -> list[dict[str, Any]]:
    max_gap = max(60.0, page_height * 0.075)
    candidates = [
        row
        for row in rows
        if 0.0 <= top_y - row["y1"] <= max_gap
        and _horizontal_overlap(row["bbox"], table_bbox) >= 0.08
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda row: row["y0"])
    label_indices = [
        idx
        for idx, row in enumerate(candidates)
        if _looks_like_table_caption_row_text(_row_text(row))
    ]
    if not label_indices:
        return []
    first_label = label_indices[-1]
    selected: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    caption_text_so_far = ""
    for row in candidates[first_label:]:
        row_text = _repair_caption_intraword_spacing(_row_text(row))
        if not row_text:
            continue
        if not selected:
            selected.append(row)
            previous = row
            caption_text_so_far = row_text
            continue
        if previous is None:
            break
        vertical_gap = float(row["y0"]) - float(previous["y1"])
        if vertical_gap < -1.5 or vertical_gap > 24.0:
            break
        overlaps_table = _horizontal_overlap(row["bbox"], table_bbox) >= 0.08
        aligns_with_caption = _horizontal_overlap(row["bbox"], previous["bbox"]) >= 0.10
        short_bracket_tail = _looks_like_caption_bracket_tail(row_text, caption_text_so_far)
        if not (overlaps_table or aligns_with_caption or short_bracket_tail):
            break
        selected.append(row)
        previous = row
        caption_text_so_far = f"{caption_text_so_far} {row_text}".strip()
    return selected


def _caption_rows_text(rows: list[dict[str, Any]]) -> str:
    line_texts = [
        _repair_caption_intraword_spacing(_row_text(row))
        for row in rows
        if _row_text(row)
    ]
    joined = " ".join(text for text in line_texts if text).strip()
    return _repair_caption_cross_line_bracket_spacing(joined)


def _looks_like_caption_bracket_tail(text: str, previous_text: str) -> bool:
    cleaned = str(text or "").strip()
    previous = str(previous_text or "").strip()
    if not cleaned or not previous or len(cleaned) > 16:
        return False
    opens = previous.count("[") + previous.count("【") + previous.count("(") + previous.count("（")
    closes = previous.count("]") + previous.count("】") + previous.count(")") + previous.count("）")
    if opens <= closes:
        return False
    return bool(re.search(r"[\]】)）]\s*$", cleaned))


def _repair_caption_cross_line_bracket_spacing(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"([\[【(（][^\]】)）\s]{1,12})\s+([^\[【(（\s]{1,12}[\]】)）])", r"\1\2", cleaned)
    return cleaned


def _repair_caption_intraword_spacing(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    # PDF word extraction often inserts spaces inside CJK/Latin-number mixed
    # title phrases. Keep row-level separation elsewhere, but remove spaces
    # where both sides are word characters from the same continuous title.
    cleaned = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fffA-Za-z0-9])", r"\1\2", cleaned)
    cleaned = re.sub(r"([A-Za-z0-9])\s+([\u4e00-\u9fff])", r"\1\2", cleaned)
    return cleaned


def _rows_bbox(rows: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    bboxes = [
        row.get("bbox")
        for row in rows
        if isinstance(row.get("bbox"), tuple) and len(row.get("bbox")) == 4
    ]
    if not bboxes:
        return None
    return (
        min(float(bbox[0]) for bbox in bboxes),
        min(float(bbox[1]) for bbox in bboxes),
        max(float(bbox[2]) for bbox in bboxes),
        max(float(bbox[3]) for bbox in bboxes),
    )


def _looks_like_table_caption_row_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if _looks_like_table_note_or_source_text(cleaned):
        return False
    return bool(
        re.match(
            r"^\s*(?:(?:附)?表\s*[0-9A-Za-z一二三四五六七八九十零〇Xx.\-]*|table\s*[0-9A-Za-z.\-]*)\s*$",
            cleaned,
            re.IGNORECASE,
        )
        or re.match(
            r"^\s*(?:(?:附)?表\s*[0-9A-Za-z一二三四五六七八九十零〇Xx.\-]+|table\s*[0-9A-Za-z.\-]+)\b",
            cleaned,
            re.IGNORECASE,
        )
    )


def _looks_like_table_note_or_source_text(text: str) -> bool:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return False
    return bool(
        re.match(
            r"(?i)^\s*table\s+(?:adapted|source|sources|note|notes|caption|legend|continued\s+from|from)\b",
            cleaned,
        )
        or re.match(r"^\s*(?:表格?来源|资料来源|数据来源|注|备注|说明)\s*[:：]", cleaned)
    )


def _detect_rule_table_column_anchors(
    segmented_rows: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
    rule_bands: list[dict[str, Any]] | None = None,
) -> list[float]:
    rule_anchors = _column_anchors_from_rule_segments(rule_bands or [])
    segment_anchors = _detect_segment_column_anchors(segmented_rows)
    stable_segment_anchors = _detect_stable_segment_column_anchors(segmented_rows, table_bbox)
    dense_leaf_anchors = _detect_dense_numeric_leaf_column_anchors(segmented_rows, table_bbox)
    numeric_word_anchors = _detect_numeric_word_leaf_column_anchors(segmented_rows, table_bbox)
    if len(dense_leaf_anchors) > len(rule_anchors):
        return dense_leaf_anchors
    if len(numeric_word_anchors) > len(rule_anchors):
        return numeric_word_anchors
    if len(stable_segment_anchors) >= len(rule_anchors) and _stable_anchors_extend_leaf_span(
        stable_segment_anchors,
        rule_anchors,
    ):
        return stable_segment_anchors
    if len(segment_anchors) > len(rule_anchors) and _segment_anchors_have_stable_row_support(
        segmented_rows,
        segment_anchors,
    ):
        return segment_anchors
    if len(rule_anchors) >= 2:
        return rule_anchors

    anchors = segment_anchors
    if len(dense_leaf_anchors) > len(anchors):
        return dense_leaf_anchors
    if len(numeric_word_anchors) > len(anchors):
        return numeric_word_anchors
    if len(stable_segment_anchors) > len(anchors):
        return stable_segment_anchors
    if len(anchors) >= 2:
        return anchors

    word_lefts = [
        _word_x0(word)
        for row in segmented_rows
        for segment in row.get("segments", [])
        for word in segment.get("words", [])
        if table_bbox[0] - 4.0 <= _word_x0(word) <= table_bbox[2] + 4.0
    ]
    if not word_lefts:
        return []
    clusters: list[list[float]] = []
    for x0 in sorted(word_lefts):
        if not clusters or abs(x0 - clusters[-1][-1]) > 18.0:
            clusters.append([x0])
        else:
            clusters[-1].append(x0)
    anchors = [sum(cluster) / len(cluster) for cluster in clusters if len(cluster) >= 2]
    return anchors


def _detect_stable_segment_column_anchors(
    segmented_rows: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
) -> list[float]:
    """Infer leaf columns from repeated visual row segments.

    Horizontal rules may describe only group spans or table extents. When most
    rows already split into the same number of left-edge segments, those segment
    starts are stronger leaf-column evidence than sparse rule segments.
    """
    import statistics

    row_candidates: list[list[float]] = []
    heights: list[float] = []
    for row in segmented_rows:
        segments = [
            segment
            for segment in row.get("segments", []) or []
            if str(segment.get("text", "") or "").strip()
        ]
        if len(segments) < 3:
            continue
        segment_x0s = [float((segment.get("bbox") or (0.0,))[0]) for segment in segments]
        if not segment_x0s:
            continue
        if segment_x0s[0] < table_bbox[0] - 8.0 or segment_x0s[-1] > table_bbox[2] + 8.0:
            continue
        row_candidates.append(segment_x0s)
        for segment in segments:
            bbox = segment.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            if len(bbox) == 4:
                heights.append(max(0.1, float(bbox[3]) - float(bbox[1])))

    if len(row_candidates) < 2:
        return []

    count_frequencies: dict[int, int] = {}
    for candidate in row_candidates:
        count_frequencies[len(candidate)] = count_frequencies.get(len(candidate), 0) + 1
    col_count, support = max(count_frequencies.items(), key=lambda item: (item[1], item[0]))
    if col_count < 3 or support < 2:
        return []

    stable_rows = [candidate for candidate in row_candidates if len(candidate) == col_count]
    if len(stable_rows) < max(2, min(4, len(row_candidates) // 2)):
        return []

    median_height = statistics.median(heights) if heights else 8.0
    tolerance = max(5.0, min(14.0, median_height * 1.35))
    anchors: list[float] = []
    for col_idx in range(col_count):
        values = [row[col_idx] for row in stable_rows]
        if max(values) - min(values) > tolerance * 2.4:
            return []
        anchors.append(float(statistics.median(values)))

    if any(right <= left + 4.0 for left, right in zip(anchors, anchors[1:])):
        return []
    return anchors


def _segment_anchors_have_stable_row_support(
    segmented_rows: list[dict[str, Any]],
    anchors: list[float],
) -> bool:
    if len(anchors) < 3:
        return False
    supported_rows = 0
    for row in segmented_rows:
        segments = [
            segment
            for segment in row.get("segments", []) or []
            if str(segment.get("text", "") or "").strip()
        ]
        if len(segments) < len(anchors):
            continue
        matched = _count_row_segments_near_column_anchors(row.get("segments", []), anchors)
        if matched >= len(anchors):
            supported_rows += 1
    return supported_rows >= 2


def _stable_anchors_extend_leaf_span(stable_anchors: list[float], rule_anchors: list[float]) -> bool:
    if len(stable_anchors) < 3:
        return False
    if len(stable_anchors) > len(rule_anchors):
        return True
    if not rule_anchors:
        return True
    return float(stable_anchors[-1]) > float(rule_anchors[-1]) + 18.0


def _detect_numeric_word_leaf_column_anchors(
    segmented_rows: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
) -> list[float]:
    """Infer leaf columns from repeated numeric word x positions."""
    import statistics

    row_candidates: list[tuple[float, list[float]]] = []
    heights: list[float] = []
    for row in segmented_rows:
        row_words = [
            word
            for segment in row.get("segments", []) or []
            for word in segment.get("words", []) or []
            if str(_word_text(word) or "").strip()
        ]
        if len(row_words) < 4:
            continue
        ordered = sorted(row_words, key=lambda item: (_word_x0(item), _word_y0(item)))
        if ordered and re.match(r"^\s*(?:category|model|dataset|relations?|sentences?|triples|类别|项目|参数|组织|物种)\b", str(_word_text(ordered[0]) or ""), re.IGNORECASE):
            continue
        numeric_words = [
            word
            for word in ordered
            if _looks_like_numeric_cell(str(_word_text(word) or "").strip())
        ]
        if len(numeric_words) < 3:
            continue
        if len(numeric_words) >= 4 and numeric_words[0] is ordered[0]:
            label_anchor = _word_x0(numeric_words[0])
            value_numeric_words = numeric_words[1:]
        else:
            label_anchor_word = next(
                (
                    word
                    for word in ordered
                    if not _looks_like_numeric_cell(str(_word_text(word) or "").strip())
                    and _word_x0(word) <= _word_x0(numeric_words[0]) - 4.0
                ),
                None,
            )
            if label_anchor_word is None:
                continue
            label_anchor = _word_x0(label_anchor_word)
            value_numeric_words = numeric_words
        if len(value_numeric_words) < 3:
            continue
        row_candidates.append(
            (
                label_anchor,
                [_word_x0(word) for word in value_numeric_words],
            )
        )
        heights.extend(_word_height(word) for word in row_words if _word_height(word) > 0)

    if len(row_candidates) < 2:
        return []

    count_frequencies: dict[int, int] = {}
    for _, numeric_x0s in row_candidates:
        count_frequencies[len(numeric_x0s)] = count_frequencies.get(len(numeric_x0s), 0) + 1
    numeric_count, support = max(count_frequencies.items(), key=lambda item: (item[1], item[0]))
    if numeric_count < 3 or support < 2:
        return []

    stable_rows = [(label_x0, numeric_x0s) for label_x0, numeric_x0s in row_candidates if len(numeric_x0s) == numeric_count]
    median_height = statistics.median(heights) if heights else 8.0
    tolerance = max(4.0, min(12.0, median_height * 0.9))
    label_anchor = statistics.median(label_x0 for label_x0, _ in stable_rows)
    anchors = [float(label_anchor)]
    for idx in range(numeric_count):
        values = [numeric_x0s[idx] for _, numeric_x0s in stable_rows]
        if max(values) - min(values) > tolerance * 2.4:
            return []
        anchors.append(float(statistics.median(values)))
    if any(right <= left + 4.0 for left, right in zip(anchors, anchors[1:])):
        return []
    if anchors[0] < table_bbox[0] - 8.0 or anchors[-1] > table_bbox[2] + 24.0:
        return []
    return anchors


def _detect_dense_numeric_leaf_column_anchors(
    segmented_rows: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
) -> list[float]:
    """Infer hidden leaf columns in booktabs-style tables from numeric rows.

    Horizontal-rule tables often draw only group-level rules. In that layout the
    visible rules identify header groups, while the real leaf columns are only
    visible through repeated numeric tokens in the body rows. Use those repeated
    token x positions as objective geometry before falling back to rule anchors.
    """
    import statistics

    row_candidates: list[dict[str, Any]] = []
    heights: list[float] = []
    for row in segmented_rows:
        row_words = [
            word
            for segment in row.get("segments", [])
            for word in segment.get("words", [])
            if str(_word_text(word) or "").strip()
        ]
        if len(row_words) < 4:
            continue
        heights.extend(_word_height(word) for word in row_words if _word_height(word) > 0)
        ordered = sorted(row_words, key=lambda item: (_word_x0(item), _word_y0(item)))
        numeric_words = [
            word
            for word in ordered
            if _looks_like_numeric_cell(str(_word_text(word) or "").strip())
        ]
        if len(numeric_words) < 3:
            continue
        leading_words = [
            word
            for word in ordered
            if _word_x1(word) <= _word_x0(numeric_words[0]) + 1.0
        ]
        label_words = [word for word in leading_words if word not in numeric_words]
        if not label_words:
            continue
        label_text = " ".join(str(_word_text(word) or "").strip() for word in label_words).strip()
        if not label_text or _looks_like_numeric_cell(label_text):
            continue
        row_candidates.append(
            {
                "label_x0": min(_word_x0(word) for word in label_words),
                "numeric_x0s": [_word_x0(word) for word in numeric_words],
                "numeric_count": len(numeric_words),
            }
        )

    if len(row_candidates) < 2:
        return []

    count_frequencies: dict[int, int] = {}
    for candidate in row_candidates:
        count = int(candidate["numeric_count"])
        count_frequencies[count] = count_frequencies.get(count, 0) + 1
    numeric_count, support = max(count_frequencies.items(), key=lambda item: (item[1], item[0]))
    if numeric_count < 3 or support < 2:
        return []

    stable_rows = [
        candidate
        for candidate in row_candidates
        if int(candidate["numeric_count"]) == numeric_count
    ]
    if len(stable_rows) < 2:
        return []

    median_height = statistics.median(heights) if heights else 8.0
    tolerance = max(4.0, min(12.0, median_height * 0.75))
    label_anchor = statistics.median(float(candidate["label_x0"]) for candidate in stable_rows)
    numeric_anchors: list[float] = []
    for value_idx in range(numeric_count):
        values = [float(candidate["numeric_x0s"][value_idx]) for candidate in stable_rows]
        if max(values) - min(values) > tolerance * 2.4:
            return []
        numeric_anchors.append(float(statistics.median(values)))

    anchors = [float(label_anchor)] + numeric_anchors
    if any(right <= left + 4.0 for left, right in zip(anchors, anchors[1:])):
        return []
    if anchors[0] < table_bbox[0] - 8.0 or anchors[-1] > table_bbox[2] + 8.0:
        return []
    return anchors


def _column_anchors_from_rule_segments(rule_bands: list[dict[str, Any]]) -> list[float]:
    segment_lefts: list[float] = []
    for band in rule_bands:
        for segment in band.get("segments", []) or []:
            if len(segment) != 4:
                continue
            if float(segment[2]) - float(segment[0]) < 12.0:
                continue
            segment_lefts.append(float(segment[0]))
    if not segment_lefts:
        return []

    clusters: list[list[float]] = []
    for x0 in sorted(segment_lefts):
        if not clusters or abs(x0 - clusters[-1][-1]) > 8.0:
            clusters.append([x0])
        else:
            clusters[-1].append(x0)
    min_support = max(1, min(2, len(rule_bands)))
    return [sum(cluster) / len(cluster) for cluster in clusters if len(cluster) >= min_support]


def _caption_rule_candidate_has_table_shape(
    *,
    raw_data: list[list[str | None]],
    rule_bands: list[dict[str, Any]],
    col_count: int,
) -> bool:
    if len(rule_bands) < 2 or len(raw_data) < 4 or col_count < 2:
        return False

    filled_counts = [sum(1 for cell in row if str(cell or "").strip()) for row in raw_data]
    body_rows = raw_data[1:]
    numeric_like_rows = 0
    for row in body_rows:
        values = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        if len(values) < max(2, min(3, col_count)):
            continue
        numeric_count = sum(1 for value in values if _looks_like_numeric_cell(value))
        if numeric_count >= max(1, len(values) - 1):
            numeric_like_rows += 1
    if numeric_like_rows >= 2:
        return True

    blank_key_rows = [
        row
        for row in body_rows
        if sum(1 for cell in row if str(cell or "").strip()) == 1
        and _caption_rule_row_is_blank_form_key(next(str(cell or "").strip() for cell in row if str(cell or "").strip()))
    ]
    if len(blank_key_rows) >= 3:
        header_values = [str(cell or "").strip() for cell in raw_data[0] if str(cell or "").strip()]
        if len(header_values) >= 2 and all(len(value) <= 96 for value in header_values):
            return True

    fullish_rows = sum(1 for count in filled_counts if count >= max(2, int(col_count * 0.6)))
    if fullish_rows < max(3, len(raw_data) // 2):
        return False

    compact_rows = sum(
        1
        for row in raw_data
        if all(len(str(cell or "").strip()) <= 48 for cell in row if str(cell or "").strip())
    )
    return compact_rows >= max(3, len(raw_data) - 1)


def _shrink_caption_rule_candidate_to_table_run(
    *,
    raw_rows: list[RawRow],
    raw_data: list[list[str | None]],
    spans: list[RawSpan],
    grouped_words: list[RawWord],
    source_rows: list[dict[str, Any]],
    col_count: int,
) -> tuple[list[RawRow], list[list[str | None]], list[RawSpan], list[RawWord], list[dict[str, Any]]]:
    """Trim prose/figure rows appended below a caption-anchored ruled table.

    Horizontal-rule tables often have only top/header rules, not a bottom rule.
    A later figure or page rule can then look like the table's lower boundary.
    The stable signal is the row run: headers plus compact/numeric data rows
    followed by narrative or figure-caption rows with a different value profile.
    """
    if len(raw_data) < 6 or col_count < 2:
        return raw_rows, raw_data, spans, grouped_words, source_rows

    data_like_indices = [
        idx
        for idx, row in enumerate(raw_data)
        if idx >= 1 and _caption_rule_row_is_data_like(row, col_count)
    ]
    if len(data_like_indices) < 2:
        return raw_rows, raw_data, spans, grouped_words, source_rows

    numeric_data_indices = [
        idx
        for idx, row in enumerate(raw_data)
        if idx >= 1 and _caption_rule_row_has_numeric_data(row)
    ]
    if len(numeric_data_indices) >= 2:
        first_data_idx = numeric_data_indices[0]
        last_data_idx = numeric_data_indices[-1]
    else:
        first_data_idx = data_like_indices[0]
        last_data_idx = data_like_indices[-1]
    if last_data_idx < max(2, first_data_idx + 1):
        return raw_rows, raw_data, spans, grouped_words, source_rows

    cut_idx: int | None = None
    for idx in range(last_data_idx + 1, len(raw_data)):
        row = raw_data[idx]
        row_text = " ".join(str(cell or "").strip() for cell in row if str(cell or "").strip()).strip()
        if not row_text:
            continue
        if _caption_rule_tail_row_is_non_table(row, col_count):
            cut_idx = idx
            break
        if _caption_rule_row_is_data_like(row, col_count):
            cut_idx = None
            break

    if cut_idx is None or cut_idx < 3:
        return raw_rows, raw_data, spans, grouped_words, source_rows

    kept_raw_rows = raw_rows[:cut_idx]
    kept_raw_data = raw_data[:cut_idx]
    kept_source_rows = source_rows[:cut_idx]
    row_bottom = max((float(row.y1) for row in kept_raw_rows), default=0.0)
    kept_spans = [span for span in spans if float(span.y1) <= row_bottom + 1.0]
    kept_words = [word for word in grouped_words if float(word.y1) <= row_bottom + 1.0]
    return kept_raw_rows, kept_raw_data, kept_spans, kept_words, kept_source_rows


def _caption_rule_row_has_numeric_data(row: list[str | None]) -> bool:
    values = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
    if not values:
        return False
    numeric_count = sum(1 for value in values if _looks_like_numeric_cell(value))
    return numeric_count >= max(1, min(2, len(values)))


def _caption_rule_row_is_data_like(row: list[str | None], col_count: int) -> bool:
    values = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
    if not values:
        return False
    if _caption_rule_row_has_numeric_data(row):
        return True
    if len(values) == 1 and _caption_rule_row_is_blank_form_key(values[0]):
        return True
    filled_count = len(values)
    if filled_count >= max(2, min(3, col_count)):
        compact_values = sum(1 for value in values if len(value) <= 42)
        narrative_values = sum(1 for value in values if _looks_like_narrative_sentence(value))
        return compact_values >= max(2, filled_count - 1) and narrative_values == 0
    return False


def _caption_rule_row_is_blank_form_key(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return False
    if len(cleaned) > 48 or len(cleaned.split()) > 5:
        return False
    if _looks_like_narrative_sentence(cleaned):
        return False
    if re.match(r"^\s*(?:\d+|[A-Za-z])[\.)]\s+\S+", cleaned):
        return False
    if re.search(r"[.;:!?。；：！？]\s*$", cleaned):
        return False
    if re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned):
        return True
    return bool(re.search(r"[+\-#*$]", cleaned))


def _blank_form_key_row_has_separable_value_column(row: dict[str, Any]) -> bool:
    words = [
        word
        for word in row.get("words", []) or []
        if _word_text(word).strip()
    ]
    if len(words) < 2:
        return False
    trailing_text = _word_text(words[-1]).strip()
    leading_text = " ".join(_word_text(word).strip() for word in words[:-1] if _word_text(word).strip())
    if not trailing_text or not leading_text:
        return False
    if not _looks_like_numeric_cell(trailing_text):
        return False
    gap = float(_word_x0(words[-1])) - float(_word_x1(words[-2]))
    row_width = max(1.0, float(_word_x1(words[-1])) - float(_word_x0(words[0])))
    return gap >= max(10.0, row_width * 0.08)


def _caption_rule_tail_row_is_non_table(row: list[str | None], col_count: int) -> bool:
    values = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
    if not values:
        return False
    joined = " ".join(values)
    if _looks_like_table_caption_row_text(joined):
        return True
    if re.match(r"^\s*(?:Figure|Fig\.?|图)\s*\d+(?:\.\d+)*\b", joined, re.IGNORECASE):
        return True
    if _caption_rule_tail_row_is_unmarked_section_heading(joined):
        return True
    narrative_cells = sum(1 for value in values if _looks_like_narrative_sentence(value))
    long_cells = sum(1 for value in values if len(value.split()) >= 6 or len(value) >= 55)
    numeric_count = sum(1 for value in values if _looks_like_numeric_cell(value))
    if narrative_cells >= 1 and numeric_count == 0:
        return True
    if long_cells >= 1 and len(values) <= max(3, col_count // 2) and numeric_count == 0:
        return True
    if len(values) >= max(3, min(5, col_count)) and numeric_count == 0:
        joined_words = len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", joined))
        if joined_words >= 6 and re.search(r"\b(?:the|and|of|to|in|on|with|from|for|is|are)\b", joined, re.IGNORECASE):
            return True
    return False


def _caption_rule_tail_row_is_unmarked_section_heading(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned or len(cleaned) > 90:
        return False
    if re.search(r"[.;!?。；;]$", cleaned):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z-]*", cleaned)
    if 1 <= len(words) <= 7 and len(words) == len(cleaned.split()):
        lower_words = {word.lower() for word in words}
        if any(word in {"note", "notes", "mean", "means", "average", "calculated", "determined", "measured"} for word in lower_words):
            return False
        stop_words = {"the", "and", "or", "of", "to", "in", "for", "with", "by", "on", "a", "an"}
        return any(word not in stop_words for word in lower_words)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    return 2 <= cjk_count <= 18 and not re.search(r"[，,。；;：:]", cleaned)


def _prefer_visual_structure_body_column_anchors(
    anchors: list[float],
    segmented_rows: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
    *,
    page_width: float,
) -> list[float]:
    body_anchors = _detect_stable_segment_column_anchors(segmented_rows, table_bbox)
    if len(body_anchors) >= 2 and _visual_body_anchors_have_row_support(segmented_rows, body_anchors):
        if not anchors or len(body_anchors) <= max(len(anchors), 3):
            return body_anchors
        if len(anchors) - len(body_anchors) >= 2:
            return body_anchors
    if len(anchors) > 4:
        projected = _project_visual_headers_to_body_anchors(segmented_rows, anchors, table_bbox)
        if len(projected) >= 2 and len(projected) < len(anchors):
            return projected
    return list(anchors)


def _trim_visual_structure_rows_to_structural_region(
    all_rows: list[dict[str, Any]],
    body_rows: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    if not all_rows or not body_rows:
        return list(all_rows)
    body_top = min(float(row.get("y0", 0.0)) for row in body_rows)
    body_bottom = max(float(row.get("y1", 0.0)) for row in body_rows)
    body_left = min(float((row.get("bbox") or table_bbox)[0]) for row in body_rows)
    body_right = max(float((row.get("bbox") or table_bbox)[2]) for row in body_rows)
    retained: list[dict[str, Any]] = []
    for row in all_rows:
        bbox = tuple(row.get("bbox") or ())
        if len(bbox) != 4:
            continue
        if float(bbox[1]) < body_top - 8.0 and _horizontal_overlap(bbox, (body_left, body_top, body_right, body_bottom)) < 0.45:
            continue
        retained.append(row)
    return retained


def _shrink_visual_structure_candidate_to_internal_header(
    *,
    all_rows: list[dict[str, Any]],
    body_rows: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[float, float, float, float]]:
    """Release leading body prose when a visual candidate contains an internal table header.

    Visual rules can span explanatory paragraphs plus an embedded structure
    grid. The raw table candidate should start at the first compact schema row,
    leaving the preceding prose to the normal reading-order text pipeline.
    """
    if len(all_rows) < 6:
        return list(all_rows), list(body_rows), table_bbox

    segmented_rows = _segment_group_rows(all_rows)
    for row_index, segmented in enumerate(segmented_rows):
        if row_index < 2:
            continue
        preceding_rows = all_rows[:row_index]
        following_rows = all_rows[row_index + 1 :]
        if len(following_rows) < 2:
            continue
        if not _visual_prefix_rows_are_body_prose(preceding_rows):
            continue
        if not _segmented_row_looks_like_internal_visual_header(segmented):
            continue
        if not _visual_rows_after_internal_header_support_table(segmented_rows[row_index + 1 :]):
            continue

        retained_all = list(all_rows[row_index:])
        retained_bbox = _rows_bbox(retained_all)
        if retained_bbox is None:
            return list(all_rows), list(body_rows), table_bbox
        retained_body = [
            row
            for row in body_rows
            if float(row.get("y1", 0.0)) >= float(retained_bbox[1]) - 2.0
        ]
        if len(retained_body) < 3:
            retained_body = retained_all
        return retained_all, retained_body, retained_bbox

    return list(all_rows), list(body_rows), table_bbox


def _visual_prefix_rows_are_body_prose(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 2:
        return False
    prose_rows = 0
    long_rows = 0
    continuation_rows = 0
    compact_schema_rows = 0
    for row in rows:
        text = _row_text(row)
        if not text:
            continue
        words = re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", text)
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        if _looks_like_narrative_sentence(text) or len(words) >= 8 or cjk_chars >= 16:
            prose_rows += 1
        if len(words) >= 8 or len(text) >= 64:
            long_rows += 1
        if re.search(
            r"\b(?:and|or|of|in|the|that|for|with|because|compared|results?|shows?|inside|through)\b",
            text,
            re.IGNORECASE,
        ):
            continuation_rows += 1
        row_segments = _segment_group_rows([row])
        if row_segments and _segmented_row_looks_like_internal_visual_header(row_segments[0]):
            compact_schema_rows += 1

    return (
        prose_rows >= max(2, len(rows) // 2)
        and long_rows >= 1
        and continuation_rows >= 1
        and compact_schema_rows <= 1
    )


def _segmented_row_looks_like_internal_visual_header(row: dict[str, Any]) -> bool:
    segments = [
        segment
        for segment in row.get("segments", []) or []
        if str(segment.get("text", "") or "").strip()
    ]
    if len(segments) < 3:
        return False
    texts = [str(segment.get("text", "") or "").strip() for segment in segments]
    joined = " ".join(texts)
    if _looks_like_narrative_sentence(joined):
        return False
    alpha_segments = sum(1 for text in texts if re.search(r"[A-Za-z\u4e00-\u9fff]", text))
    symbol_segments = sum(1 for text in texts if _visual_segment_is_connector_symbol(text))
    compact_segments = sum(1 for text in texts if len(text) <= 42 and len(text.split()) <= 5)
    long_segments = sum(1 for text in texts if len(text) > 64 or len(text.split()) > 8)
    if alpha_segments < 2:
        return False
    if compact_segments < max(2, len(texts) - 1):
        return False
    if long_segments:
        return False
    return len(texts) >= 4 or symbol_segments >= 1 or alpha_segments >= 3


def _visual_rows_after_internal_header_support_table(
    rows: list[dict[str, Any]],
) -> bool:
    if len(rows) < 2:
        return False
    supported_rows = 0
    compact_or_symbol_rows = 0
    narrative_rows = 0
    for row in rows[: min(len(rows), 12)]:
        segments = [
            segment
            for segment in row.get("segments", []) or []
            if str(segment.get("text", "") or "").strip()
        ]
        texts = [str(segment.get("text", "") or "").strip() for segment in segments]
        if not texts:
            continue
        joined = " ".join(texts)
        if _looks_like_narrative_sentence(joined) and len(segments) <= 1:
            narrative_rows += 1
        if len(segments) >= 2 or any(_visual_segment_is_connector_symbol(text) for text in texts):
            supported_rows += 1
        if all(len(text) <= 72 for text in texts) or any(_visual_segment_is_connector_symbol(text) for text in texts):
            compact_or_symbol_rows += 1
    return supported_rows >= 2 and compact_or_symbol_rows >= 2 and narrative_rows <= max(2, supported_rows)


def _visual_segment_is_connector_symbol(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    return bool(re.fullmatch(r"(?:[-+=>→⇒↔←]+|[•·])+", cleaned))


def _visual_body_anchors_have_row_support(
    segmented_rows: list[dict[str, Any]],
    anchors: list[float],
) -> bool:
    if len(anchors) < 2:
        return False
    supported_rows = 0
    for row in segmented_rows:
        projected, matched_cols = _project_segmented_row_to_text_grid(row, anchors)
        filled = sum(1 for cell in projected if str(cell or "").strip())
        if filled >= min(len(anchors), 2) and len(matched_cols) >= min(len(anchors), 2):
            supported_rows += 1
    return supported_rows >= max(2, min(4, len(segmented_rows) // 2))


def _project_visual_headers_to_body_anchors(
    segmented_rows: list[dict[str, Any]],
    anchors: list[float],
    table_bbox: tuple[float, float, float, float],
) -> list[float]:
    candidate_rows = []
    for row in segmented_rows:
        segments = [
            segment
            for segment in row.get("segments", []) or []
            if str(segment.get("text", "") or "").strip()
        ]
        if 2 <= len(segments) <= 4:
            candidate_rows.append([float(segment["bbox"][0]) for segment in segments])
    if not candidate_rows:
        return []
    counts: dict[int, int] = {}
    for row_anchors in candidate_rows:
        counts[len(row_anchors)] = counts.get(len(row_anchors), 0) + 1
    col_count, support = max(counts.items(), key=lambda item: (item[1], item[0]))
    if col_count < 2 or support < 2:
        return []
    rows = [row for row in candidate_rows if len(row) == col_count]
    projected: list[float] = []
    for idx in range(col_count):
        values = sorted(row[idx] for row in rows)
        projected.append(values[len(values) // 2])
    if projected[0] < table_bbox[0] - 8.0 or projected[-1] > table_bbox[2] + 8.0:
        return []
    return projected


def _visual_structure_candidate_has_table_shape(
    *,
    raw_data: list[list[str | None]],
    col_count: int,
) -> bool:
    if len(raw_data) < 3 or col_count < 2:
        return False
    rows = [
        [str(cell or "").strip() for cell in row]
        for row in raw_data
        if any(str(cell or "").strip() for cell in row)
    ]
    if len(rows) < 3:
        return False
    filled_counts = [sum(1 for cell in row if cell) for row in rows]
    multi_col_rows = sum(1 for count in filled_counts if count >= min(2, col_count))
    if multi_col_rows < max(2, len(rows) // 2):
        return False
    compact_label_rows = sum(
        1
        for row in rows
        if 1 <= sum(1 for cell in row if cell) <= col_count
        and all(len(cell) <= 96 for cell in row if cell)
    )
    if compact_label_rows < max(2, len(rows) // 3):
        return False
    first_text = " ".join(cell for cell in rows[0] if cell)
    if len(first_text) > 180 and re.search(r"[.。！？!?；;]\s*$", first_text):
        return False
    return True


def _drop_visual_structure_footer_rows(
    raw_rows: list[RawRow],
    raw_data: list[list[str | None]],
    *,
    page_height: float,
    page_width: float,
) -> tuple[list[RawRow], list[list[str | None]]]:
    trimmed_rows = list(raw_rows)
    trimmed_data = list(raw_data)
    while trimmed_rows and trimmed_data:
        row = trimmed_rows[-1]
        data = trimmed_data[-1]
        values = [str(cell or "").strip() for cell in data if str(cell or "").strip()]
        if len(values) != 1:
            break
        text = values[0]
        bbox = tuple(row.bbox or ())
        if len(bbox) == 4 and re.fullmatch(r"\d{1,4}", text) and (
            float(bbox[1]) >= page_height * 0.90
            or float(bbox[0]) >= page_width * 0.82
        ):
            trimmed_rows.pop()
            trimmed_data.pop()
            continue
        break
    for index, row in enumerate(trimmed_rows):
        row.physical_row = index
        for cell in row.cells:
            cell.physical_row = index
    return trimmed_rows, trimmed_data


def _raw_drawings_from_visual_structure_region(region: dict[str, Any]) -> list[RawDrawing]:
    drawings: list[RawDrawing] = []
    for band in list(region.get("rule_bands") or []) + list(region.get("vertical_bands") or []):
        for segment in band.get("segments", []) or []:
            if len(segment) != 4:
                continue
            width = float(segment[2]) - float(segment[0])
            height = float(segment[3]) - float(segment[1])
            drawing_type = DrawingType.LINE
            drawings.append(
                RawDrawing(
                    drawing_type=drawing_type,
                    x0=float(segment[0]),
                    y0=float(segment[1]),
                    x1=float(segment[2]),
                    y1=float(segment[3]),
                    raw_data={
                        "source": "visual_structure_grid",
                        "orientation": "vertical" if height > width else "horizontal",
                    },
                )
            )
    return drawings


def _looks_like_numeric_cell(text: str) -> bool:
    cleaned = str(text or "").strip().replace(",", "")
    if not cleaned:
        return False
    if re.fullmatch(r"[<>≤≥~+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?%?", cleaned):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?", cleaned):
        return True
    return False


def _raw_drawings_from_rule_bands(rule_bands: list[dict[str, Any]]) -> list[RawDrawing]:
    drawings: list[RawDrawing] = []
    for band in rule_bands:
        for segment in band.get("segments", []):
            drawings.append(
                RawDrawing(
                    drawing_type=DrawingType.LINE,
                    x0=float(segment[0]),
                    y0=float(segment[1]),
                    x1=float(segment[2]),
                    y1=float(segment[3]),
                    raw_data={"source": "caption_anchored_horizontal_rule"},
                )
            )
    return drawings


def _deduplicate_rule_table_candidates(
    candidates: list[RawTableEvidence],
) -> list[RawTableEvidence]:
    result: list[RawTableEvidence] = []
    for candidate in sorted(candidates, key=_rule_table_candidate_dedup_sort_key):
        duplicate = False
        for existing_index, existing in enumerate(result):
            if _bbox_overlap_ratio(candidate.bbox, existing.bbox) >= 0.70:
                preferred = _preferred_overlapping_rule_table_candidate(candidate, existing)
                if preferred is candidate:
                    result[existing_index] = candidate
                duplicate = True
                break
        if not duplicate:
            result.append(candidate)
    return result


def _rule_table_candidate_dedup_sort_key(candidate: RawTableEvidence) -> tuple[float, int, int, int, float]:
    internal_caption_count = _candidate_internal_caption_row_count(candidate)
    row_count = int(candidate.physical_row_count or len(candidate.raw_data or []))
    height = float(candidate.bbox[3]) - float(candidate.bbox[1])
    blank_form_rank = 0 if str(candidate.source or "") == "caption_anchored_blank_form" else 1
    return (float(candidate.bbox[1]), internal_caption_count, blank_form_rank, -row_count, -height)


def _preferred_overlapping_rule_table_candidate(
    candidate: RawTableEvidence,
    existing: RawTableEvidence,
) -> RawTableEvidence:
    candidate_source = str(candidate.source or "")
    existing_source = str(existing.source or "")
    blank_source = "caption_anchored_blank_form"
    if candidate_source == blank_source and existing_source != blank_source:
        return existing if _rule_candidate_has_strong_line_structure(existing) else candidate
    if existing_source == blank_source and candidate_source != blank_source:
        return candidate if _rule_candidate_has_strong_line_structure(candidate) else existing

    candidate_key = _rule_table_candidate_dedup_sort_key(candidate)
    existing_key = _rule_table_candidate_dedup_sort_key(existing)
    return candidate if candidate_key < existing_key else existing


def _rule_candidate_has_strong_line_structure(candidate: RawTableEvidence) -> bool:
    if str(candidate.source or "") != "caption_anchored_horizontal_rules":
        return False
    line_count = len(candidate.drawings or [])
    col_count = int(candidate.physical_col_count or 0)
    row_count = int(candidate.physical_row_count or len(candidate.raw_data or []))
    if col_count >= 4 and row_count >= 3 and line_count >= 3:
        return True
    return line_count >= max(4, min(8, row_count))


def _candidate_internal_caption_row_count(candidate: RawTableEvidence) -> int:
    caption_norm = _compact_caption_text(str(getattr(candidate, "caption_text", "") or ""))
    count = 0
    for row_index, row in enumerate(candidate.raw_data or []):
        row_text = " ".join(str(cell or "").strip() for cell in row if str(cell or "").strip()).strip()
        if not row_text or not _looks_like_table_caption_row_text(row_text):
            continue
        row_norm = _compact_caption_text(row_text)
        if row_index <= 1 and row_norm and (row_norm == caption_norm or row_norm in caption_norm):
            continue
        count += 1
    return count


def _compact_caption_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _bbox_overlap_ratio(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter_area = inter_w * inter_h
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    denom = min(area_a, area_b)
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _bbox_width(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0]))


def _bbox_height(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(_word_text(word).strip() for word in row.get("words", []) if _word_text(word).strip()).strip()


def _select_best_table_group(
    table_groups: list[list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Select the strongest borderless-grid candidate from row groups."""
    table_groups = _merge_keyed_long_description_table_groups(table_groups)
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

        segmented_rows = _trim_word_cluster_group_to_schema_region_when_followed_by_body_prompt(
            segmented_rows,
            column_anchors,
        )
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
        keyed_long_description = _looks_like_keyed_long_description_group(
            segmented_rows=segmented_rows,
            column_anchors=column_anchors,
            segment_counts=segment_counts,
            row_anchor_matches=row_anchor_matches,
            coverage=coverage,
        )
        if not dense_grid and not sparse_listing and not keyed_long_description:
            continue

        if dense_grid:
            score = (
                0.25 * min(1.0, len(segmented_rows) / 8.0)
                + 0.25 * min(1.0, stable_anchor_ratio)
                + 0.25 * (multi_segment_rows / max(1, len(segmented_rows)))
                + 0.15 * (aligned_rows / max(1, len(segmented_rows)))
                + 0.10 * min(1.0, coverage)
            )
        elif keyed_long_description:
            keyed_rows = sum(1 for count in row_anchor_matches if count >= max(3, col_count - 1))
            continuation_rows = sum(1 for count in row_anchor_matches if count == 1)
            score = (
                0.20 * min(1.0, len(segmented_rows) / 20.0)
                + 0.25 * min(1.0, keyed_rows / 4.0)
                + 0.20 * min(1.0, continuation_rows / 8.0)
                + 0.20 * min(1.0, coverage / 0.30)
                + 0.15 * min(1.0, aligned_rows / 4.0)
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
            "profile": "keyed_long_description" if keyed_long_description else ("dense_grid" if dense_grid else "sparse_listing"),
        }

    return best_candidate


def _trim_word_cluster_group_to_schema_region_when_followed_by_body_prompt(
    segmented_rows: list[dict[str, Any]],
    column_anchors: list[float],
) -> list[dict[str, Any]]:
    """Trim external headers/body prompts from a word-cluster table group."""
    if len(segmented_rows) < 3 or len(column_anchors) < 2:
        return segmented_rows

    projected = [
        _project_segmented_row_to_text_grid(row, column_anchors)
        for row in segmented_rows
    ]
    segment_counts = [len(row.get("segments", []) or []) for row in segmented_rows]
    row_anchor_matches = [
        _count_row_segments_near_column_anchors(row.get("segments", []) or [], column_anchors)
        for row in segmented_rows
    ]
    coverage = sum(segment_counts) / max(1, len(segmented_rows) * len(column_anchors))
    if _looks_like_keyed_long_description_group(
        segmented_rows=segmented_rows,
        column_anchors=column_anchors,
        segment_counts=segment_counts,
        row_anchor_matches=row_anchor_matches,
        coverage=coverage,
    ):
        return segmented_rows

    body_boundary_index: int | None = None
    for idx, row in enumerate(segmented_rows[1:], start=1):
        row_text = _row_text_from_segmented(row)
        cells = [str(cell or "").strip() for cell in projected[idx][0] if str(cell or "").strip()]
        matched_cols = projected[idx][1]
        previous_bbox = tuple(segmented_rows[idx - 1].get("bbox", (0.0, 0.0, 0.0, 0.0)))
        row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        vertical_gap = float(row_bbox[1]) - float(previous_bbox[3]) if len(previous_bbox) == 4 and len(row_bbox) == 4 else 0.0
        if _word_cluster_row_is_external_body_boundary(
            row_text=row_text,
            cells=cells,
            matched_cols=matched_cols,
            col_count=len(column_anchors),
            vertical_gap=vertical_gap,
        ):
            body_boundary_index = idx
            break
    if body_boundary_index is None:
        return segmented_rows

    start = 0
    for idx, row in enumerate(segmented_rows):
        if idx >= body_boundary_index:
            break
        cells = [str(cell or "").strip() for cell in projected[idx][0] if str(cell or "").strip()]
        matched_cols = projected[idx][1]
        if _word_cluster_row_is_spanning_schema_region_start(
            row_index=idx,
            segmented_rows=segmented_rows,
            projected=projected,
            column_anchors=column_anchors,
        ):
            start = idx
            break
        if _word_cluster_row_is_schema_region_start(
            row=row,
            cells=cells,
            matched_cols=matched_cols,
            col_count=len(column_anchors),
        ):
            start = idx
            break
    else:
        return segmented_rows

    table_region_row_count = body_boundary_index - start
    if table_region_row_count < 2 or table_region_row_count > 8:
        return segmented_rows

    trimmed = list(segmented_rows[start:])
    projected = projected[start:]
    end = body_boundary_index - start
    for idx, row in enumerate(trimmed[1:], start=1):
        if idx >= end:
            break
        row_text = _row_text_from_segmented(row)
        cells = [str(cell or "").strip() for cell in projected[idx][0] if str(cell or "").strip()]
        matched_cols = projected[idx][1]
        previous_bbox = tuple(trimmed[idx - 1].get("bbox", (0.0, 0.0, 0.0, 0.0)))
        row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        vertical_gap = float(row_bbox[1]) - float(previous_bbox[3]) if len(previous_bbox) == 4 and len(row_bbox) == 4 else 0.0
        if _word_cluster_row_is_external_body_boundary(
            row_text=row_text,
            cells=cells,
            matched_cols=matched_cols,
            col_count=len(column_anchors),
            vertical_gap=vertical_gap,
        ):
            end = idx
            break

    return trimmed[:end] if end >= 2 else segmented_rows


def _word_cluster_row_is_schema_region_start(
    *,
    row: dict[str, Any],
    cells: list[str],
    matched_cols: set[int],
    col_count: int,
) -> bool:
    if len(matched_cols) < max(2, min(4, col_count)):
        return False
    if _segments_look_like_listing_header(cells):
        return True
    if len(cells) < max(3, min(4, col_count)):
        return False
    if _row_has_page_locator_header_cues(row):
        return False
    compact_label_cells = [
        text for text in cells
        if len(text) <= 28 and len(text.split()) <= 3 and re.search(r"[A-Za-z\u4e00-\u9fff]", text)
    ]
    numeric_cells = [text for text in cells if re.fullmatch(r"[+\-−]?\d+(?:\.\d+)?(?:\s*[A-Za-z/%]+)?", text)]
    if len(compact_label_cells) >= max(3, min(4, col_count)):
        return True
    return len(compact_label_cells) >= 2 and len(numeric_cells) >= 1


def _word_cluster_row_is_external_body_boundary(
    *,
    row_text: str,
    cells: list[str],
    matched_cols: set[int],
    col_count: int,
    vertical_gap: float,
) -> bool:
    cleaned = " ".join(str(row_text or "").split())
    if not cleaned:
        return False
    filled_count = len(cells)
    if filled_count == 1 and re.fullmatch(r"[_\-—–]{12,}", cleaned):
        return True
    if filled_count <= 1 and _looks_like_external_body_heading_or_prompt(cleaned):
        return True
    if filled_count <= 1 and re.match(r"^\d+\.\s+[A-Z\u4e00-\u9fff]", cleaned) and _looks_like_narrative_sentence(cleaned):
        return True
    if filled_count <= 1 and vertical_gap >= 10.0 and _looks_like_narrative_sentence(cleaned):
        return True
    if filled_count <= 1 and len(cleaned.split()) >= 5 and col_count >= 3:
        return True
    return False


def _word_cluster_row_is_spanning_schema_region_start(
    *,
    row_index: int,
    segmented_rows: list[dict[str, Any]],
    projected: list[tuple[list[str | None], set[int]]],
    column_anchors: list[float],
) -> bool:
    if row_index + 2 >= len(segmented_rows):
        return False
    row = segmented_rows[row_index]
    row_text = _row_text_from_segmented(row)
    cells = [str(cell or "").strip() for cell in projected[row_index][0] if str(cell or "").strip()]
    if len(cells) != 1:
        return False
    if _row_has_page_locator_header_cues(row):
        return False
    if _looks_like_external_body_heading_or_prompt(row_text):
        return False
    tokens = re.findall(r"[A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff'/-]*|\d+(?:\.\d+)?", row_text)
    if not (3 <= len(tokens) <= max(10, len(column_anchors) * 3)):
        return False
    next_matches = [
        len(projected[next_index][1])
        for next_index in range(row_index + 1, min(len(projected), row_index + 4))
    ]
    if sum(1 for count in next_matches if count >= max(3, min(4, len(column_anchors) - 1))) < 2:
        return False
    next_cells = [
        str(cell or "").strip()
        for next_index in range(row_index + 1, min(len(projected), row_index + 4))
        for cell in projected[next_index][0]
        if str(cell or "").strip()
    ]
    compact_values = sum(1 for text in next_cells if len(text) <= 18 and len(text.split()) <= 3)
    return compact_values >= max(4, min(8, len(next_cells)))


def _looks_like_external_body_heading_or_prompt(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+(?:\.\d+)*|[\u4e00-\u9fff]+", cleaned)
    if len(tokens) < 2 or len(tokens) > 14:
        return False
    if re.match(r"^\d+\.\s+", cleaned):
        return True
    if cleaned.endswith(":"):
        alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z\u4e00-\u9fff]", token)]
        if len(alpha_tokens) >= 1 and len(tokens) <= 9:
            return True
    title_like = 0
    alpha_tokens = 0
    for token in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)*", token):
            continue
        alpha_tokens += 1
        if token[:1].isupper() or re.fullmatch(r"[\u4e00-\u9fff]+", token):
            title_like += 1
    return alpha_tokens >= 2 and title_like / max(1, alpha_tokens) >= 0.55 and len(tokens) <= 8


def _merge_keyed_long_description_table_groups(
    table_groups: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Join adjacent header/body groups for sparse long-description tables.

    Borderless regulatory and legal tables often have a compact multi-row
    header followed by rows with short key/status columns and a long narrative
    description column. Row grouping may split the header from the body because
    the first body line only populates the description column. Keep this as a
    raw-evidence repair so later normalization/header grammar still owns the
    semantic interpretation.
    """
    if len(table_groups) < 2:
        return table_groups

    ordered = sorted(
        [list(group) for group in table_groups if group],
        key=lambda group: float(group[0].get("y0", 0.0)),
    )
    merged: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(ordered):
        current = list(ordered[index])
        if index + 1 < len(ordered) and _looks_like_keyed_long_description_pair(current, ordered[index + 1]):
            current.extend(ordered[index + 1])
            index += 2
        else:
            index += 1
        merged.append(current)
    return merged


def _looks_like_keyed_long_description_pair(
    header_group: list[dict[str, Any]],
    body_group: list[dict[str, Any]],
) -> bool:
    if len(header_group) < 2 or len(body_group) < 4:
        return False

    header_segmented = _segment_group_rows(header_group)
    body_segmented = _segment_group_rows(body_group)
    header_anchors = _detect_segment_column_anchors(header_segmented)
    body_anchors = _detect_segment_column_anchors(body_segmented)
    if len(header_anchors) < 4 or len(body_anchors) < 3:
        return False

    shared_anchor_count = _shared_column_anchor_count(header_anchors, body_anchors)
    if shared_anchor_count < max(3, min(len(header_anchors), len(body_anchors)) - 1):
        return False

    vertical_gap = float(body_group[0].get("y0", 0.0)) - float(header_group[-1].get("y1", 0.0))
    median_height = _median_row_height(header_group + body_group)
    if vertical_gap < -1.0 or vertical_gap > max(18.0, median_height * 1.8):
        return False

    header_segments = [
        str(segment.get("text", "") or "").strip()
        for row in header_segmented
        for segment in row.get("segments", []) or []
        if str(segment.get("text", "") or "").strip()
    ]
    if len(header_segments) < len(header_anchors):
        return False
    if not _segments_look_like_long_description_header(header_segments):
        return False

    body_matches = [
        _count_row_segments_near_column_anchors(row.get("segments", []) or [], body_anchors)
        for row in body_segmented
    ]
    keyed_rows = sum(1 for count in body_matches if count >= max(3, len(body_anchors) - 1))
    continuation_rows = sum(1 for count in body_matches if count == 1)
    if keyed_rows < 2 or continuation_rows < 2:
        return False

    body_segments = [
        str(segment.get("text", "") or "").strip()
        for row in body_segmented
        for segment in row.get("segments", []) or []
        if str(segment.get("text", "") or "").strip()
    ]
    if not body_segments:
        return False
    short_key_values = sum(1 for text in body_segments if _looks_like_compact_key_or_flag_cell(text))
    long_description_values = sum(1 for text in body_segments if _looks_like_long_description_cell(text))
    if short_key_values < 4 or long_description_values < 4:
        return False

    return True


def _looks_like_keyed_long_description_group(
    *,
    segmented_rows: list[dict[str, Any]],
    column_anchors: list[float],
    segment_counts: list[int],
    row_anchor_matches: list[int],
    coverage: float,
) -> bool:
    row_count = len(segmented_rows)
    col_count = len(column_anchors)
    if row_count < 8 or col_count < 4:
        return False
    if coverage < 0.24:
        return False

    header_rows = segmented_rows[: min(4, row_count)]
    header_segments = [
        str(segment.get("text", "") or "").strip()
        for row in header_rows
        for segment in row.get("segments", []) or []
        if str(segment.get("text", "") or "").strip()
    ]
    if not _segments_look_like_long_description_header(header_segments):
        return False

    complete_threshold = max(3, col_count - 1)
    body_matches = row_anchor_matches[len(header_rows) :]
    keyed_rows = [idx for idx, count in enumerate(body_matches, start=len(header_rows)) if count >= complete_threshold]
    continuation_rows = [idx for idx, count in enumerate(body_matches, start=len(header_rows)) if count == 1]
    if len(keyed_rows) < 2 or len(continuation_rows) < 4:
        return False

    first_keyed = min(keyed_rows)
    if first_keyed > len(header_rows) + 6:
        return False

    body_segments = [
        str(segment.get("text", "") or "").strip()
        for row in segmented_rows[len(header_rows) :]
        for segment in row.get("segments", []) or []
        if str(segment.get("text", "") or "").strip()
    ]
    if not body_segments:
        return False
    short_key_values = sum(1 for text in body_segments if _looks_like_compact_key_or_flag_cell(text))
    long_description_values = sum(1 for text in body_segments if _looks_like_long_description_cell(text))
    if short_key_values < 4 or long_description_values < 4:
        return False

    multi_segment_rows = sum(1 for count in segment_counts if count >= 2)
    if multi_segment_rows < max(4, len(keyed_rows)):
        return False

    return True


def _shared_column_anchor_count(left: list[float], right: list[float]) -> int:
    if not left or not right:
        return 0
    gaps = [
        abs(float(b) - float(a))
        for anchors in (left, right)
        for a, b in zip(anchors, anchors[1:])
        if abs(float(b) - float(a)) > 0
    ]
    tolerance = max(8.0, min(18.0, (_percentile(gaps, 0.5) if gaps else 32.0) * 0.22))
    used_right: set[int] = set()
    shared = 0
    for value in left:
        candidates = [
            (idx, abs(float(value) - float(other)))
            for idx, other in enumerate(right)
            if idx not in used_right and abs(float(value) - float(other)) <= tolerance
        ]
        if not candidates:
            continue
        idx, _distance = min(candidates, key=lambda item: item[1])
        used_right.add(idx)
        shared += 1
    return shared


def _median_row_height(rows: list[dict[str, Any]]) -> float:
    heights = sorted(float(row.get("height", 0.0) or 0.0) for row in rows if float(row.get("height", 0.0) or 0.0) > 0)
    if not heights:
        return 10.0
    return heights[len(heights) // 2]


def _segments_look_like_long_description_header(texts: list[str]) -> bool:
    cleaned = [" ".join(str(text or "").split()) for text in texts if str(text or "").strip()]
    if len(cleaned) < 4:
        return False
    compact_headers = sum(1 for text in cleaned if len(text) <= 64 and len(text.split()) <= 6)
    if compact_headers < max(3, len(cleaned) - 1):
        return False
    schema_terms = {
        "jurisdiction",
        "ownership",
        "restriction",
        "requirement",
        "reservation",
        "permitted",
        "reporting",
        "category",
        "type",
        "item",
        "description",
        "result",
        "endpoint",
        "dose",
        "route",
        "species",
        "study",
        "reference",
        "parameter",
        "unit",
        "value",
        "status",
    }
    term_hits = 0
    for text in cleaned:
        tokens = re.findall(r"[A-Za-z\u4e00-\u9fff]+", text.lower())
        if any(token in schema_terms for token in tokens):
            term_hits += 1
    return term_hits >= 2


def _looks_like_compact_key_or_flag_cell(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return False
    if len(cleaned) <= 3 and re.fullmatch(r"[A-Za-z0-9()+/\-]+", cleaned):
        return True
    if len(cleaned) <= 32 and len(cleaned.split()) <= 3 and re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned):
        return True
    return False


def _looks_like_long_description_cell(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return False
    if len(re.findall(r"[\u4e00-\u9fff]", cleaned)) >= 10:
        return True
    return len(cleaned) >= 28 and len(cleaned.split()) >= 4


def _augment_word_cluster_group_with_preceding_spanner_rows(
    rows: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Include a close one-line spanning header above a borderless word table.

    Word-clustering starts table groups from dense rows, so a centered group
    header printed alone above the leaf-header row can be dropped before the
    header grammar sees it. This recovers that row only when geometry supports
    it as part of the table, not as a free-standing caption or paragraph.
    """
    segmented_rows = list(candidate.get("rows") or [])
    column_anchors = list(candidate.get("column_anchors") or [])
    bbox = tuple(candidate.get("bbox") or ())
    if not segmented_rows or len(column_anchors) < 3 or len(bbox) != 4:
        return candidate

    first_row = segmented_rows[0]
    if len(first_row.get("segments", []) or []) < max(3, min(5, len(column_anchors))):
        return candidate
    if _row_has_page_locator_header_cues(first_row):
        return candidate

    top_y = float(first_row.get("y0", bbox[1]))
    preceding_rows = [
        row
        for row in rows
        if float(row.get("y1", 0.0)) <= top_y + 1.0
    ]
    if not preceding_rows:
        return candidate
    preceding = max(preceding_rows, key=lambda row: float(row.get("y1", 0.0)))
    gap = top_y - float(preceding.get("y1", 0.0))
    row_height = max(1.0, float(preceding.get("height", 0.0)))
    if gap < -1.0 or gap > max(16.0, row_height * 1.6):
        return candidate

    segmented_preceding = _segment_group_rows([preceding])
    if len(segmented_preceding) != 1:
        return candidate
    preceding_segmented = segmented_preceding[0]
    segments = [segment for segment in preceding_segmented.get("segments", []) if str(segment.get("text", "")).strip()]
    if len(segments) != 1:
        return candidate
    segment = segments[0]
    text = str(segment.get("text", "") or "").strip()
    if not _looks_like_preceding_table_spanner_text(text):
        return candidate

    seg_bbox = tuple(segment.get("bbox") or ())
    if len(seg_bbox) != 4:
        return candidate
    seg_center = (float(seg_bbox[0]) + float(seg_bbox[2])) / 2.0
    value_region_left = float(column_anchors[1]) if len(column_anchors) > 1 else float(bbox[0])
    value_region_right = float(bbox[2])
    if not (value_region_left - 18.0 <= seg_center <= value_region_right + 18.0):
        return candidate
    if _horizontal_overlap(seg_bbox, bbox) < 0.05 and not (float(bbox[0]) <= seg_center <= float(bbox[2])):
        return candidate
    if _preceding_spanner_is_external_title_for_word_cluster(
        text=text,
        first_row=first_row,
        column_anchors=column_anchors,
    ):
        return candidate

    augmented = dict(candidate)
    augmented_rows = [preceding_segmented, *segmented_rows]
    augmented["rows"] = augmented_rows
    augmented["bbox"] = (
        min(float(bbox[0]), float(preceding.get("bbox", bbox)[0])),
        min(float(bbox[1]), float(preceding.get("bbox", bbox)[1])),
        max(float(bbox[2]), float(preceding.get("bbox", bbox)[2])),
        max(float(bbox[3]), float(preceding.get("bbox", bbox)[3])),
    )
    augmented["leading_spanner_row_recovered"] = True
    return augmented


def _augment_word_cluster_group_with_trailing_aligned_rows(
    rows: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Keep late rows owned by a borderless word-cluster table when anchors persist.

    Word-clustering builds the initial region from locally dense rows. Real
    service matrices and IND summary tables often end with a sparse final
    section whose first/left key remains aligned with the accepted table lattice
    after a larger vertical gap. The ownership invariant is the stable column
    profile, not the local row density.
    """
    if str(candidate.get("profile") or "") == "keyed_long_description":
        return candidate

    segmented_rows = list(candidate.get("rows") or [])
    column_anchors = list(candidate.get("column_anchors") or [])
    bbox = tuple(candidate.get("bbox") or ())
    if not segmented_rows or len(column_anchors) < 2 or len(bbox) != 4:
        return candidate

    last_row = segmented_rows[-1]
    last_bbox = tuple(last_row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(last_bbox) != 4:
        return candidate
    candidate_width = max(1.0, float(bbox[2]) - float(bbox[0]))
    page_like_width = max(candidate_width, float(column_anchors[-1]) - float(column_anchors[0]))
    trailing_rows = [
        row
        for row in rows
        if float(row.get("y0", 0.0)) > float(last_bbox[3]) - 0.5
    ]
    if not trailing_rows:
        return candidate

    augmented_rows = list(segmented_rows)
    current_bottom = float(last_bbox[3])
    for row in trailing_rows:
        row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        if len(row_bbox) != 4:
            continue
        row_height = max(1.0, float(row_bbox[3]) - float(row_bbox[1]))
        gap = float(row_bbox[1]) - current_bottom
        if gap > max(46.0, row_height * 5.2):
            break

        segmented = _segment_group_rows([row])[0]
        row_text = _row_text_from_segmented(segmented)
        if not row_text:
            continue
        if _trailing_aligned_row_is_external_boundary(row_text):
            break

        projected_row, matched_cols = _project_segmented_row_to_text_grid(segmented, column_anchors)
        if not _trailing_row_matches_word_cluster_profile(
            row=segmented,
            projected_row=projected_row,
            matched_cols=matched_cols,
            column_anchors=column_anchors,
            table_bbox=bbox,
            page_like_width=page_like_width,
        ):
            break
        augmented_rows.append(segmented)
        current_bottom = float(row_bbox[3])

    if len(augmented_rows) == len(segmented_rows):
        return candidate
    augmented = dict(candidate)
    augmented["rows"] = augmented_rows
    augmented["bbox"] = _bbox_union_loose([tuple(row.get("bbox", bbox)) for row in augmented_rows])
    augmented["trailing_aligned_rows_recovered"] = len(augmented_rows) - len(segmented_rows)
    return augmented


def _trailing_aligned_row_is_external_boundary(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return True
    if re.fullmatch(r"[_\-—–]{12,}", cleaned):
        return True
    if _looks_like_external_body_heading_or_prompt(cleaned):
        return True
    if _looks_like_table_caption_row_text(cleaned):
        return True
    if re.search(r"(?:^|\s)(?:figure|fig\.|table)\s+\d+[:.]", cleaned, re.IGNORECASE):
        return True
    if re.match(r"^\s*\d+(?:\.\d+){0,4}\s+[A-Z\u4e00-\u9fff]", cleaned) and len(cleaned.split()) <= 12:
        return True
    return False


def _trailing_row_matches_word_cluster_profile(
    *,
    row: dict[str, Any],
    projected_row: list[str | None],
    matched_cols: set[int],
    column_anchors: list[float],
    table_bbox: tuple[float, float, float, float],
    page_like_width: float,
) -> bool:
    if len(matched_cols) >= max(2, min(3, len(column_anchors))):
        return True
    if not matched_cols:
        return False
    row_bbox = tuple(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(row_bbox) != 4:
        return False
    horizontal_overlap = _horizontal_overlap(tuple(table_bbox), row_bbox)
    if horizontal_overlap < 0.16:
        return False
    row_width = max(1.0, float(row_bbox[2]) - float(row_bbox[0]))
    filled = [str(cell or "").strip() for cell in projected_row if str(cell or "").strip()]
    if len(filled) < 1:
        return False
    if len(matched_cols) == 1:
        only_col = next(iter(matched_cols))
        if only_col > 0 and float(table_bbox[0]) - 8.0 <= float(row_bbox[0]) <= float(table_bbox[2]) + 8.0:
            return True
    if row_width < max(40.0, page_like_width * 0.18):
        return False
    text = " ".join(filled)
    if len(text) > 220 and re.search(r"[.。！？!?]\s*$", text):
        return False
    return True


def _preceding_spanner_is_external_title_for_word_cluster(
    *,
    text: str,
    first_row: dict[str, Any],
    column_anchors: list[float],
) -> bool:
    """Keep standalone captions out of the raw evidence grid.

    Borderless tables often have a centered title immediately above a dense
    header row. That title is table context, not a physical table row, when the
    next row already exposes a complete column schema. True internal spanning
    headers remain recoverable when the opening row is sparse or header-like
    evidence is not already present below it.
    """
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return False
    if _looks_like_table_caption_row_text(cleaned):
        return False
    projected_row, matched_cols = _project_segmented_row_to_text_grid(first_row, column_anchors)
    first_cells = [str(cell or "").strip() for cell in projected_row if str(cell or "").strip()]
    if len(matched_cols) < max(3, min(len(column_anchors), 5)):
        return False
    if not _segments_look_like_listing_header(first_cells):
        return False
    if _looks_like_narrative_sentence(cleaned):
        return True
    word_count = len(re.findall(r"[A-Za-z\u4e00-\u9fff0-9]+", cleaned))
    return word_count <= 8 and bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned))


def _looks_like_preceding_table_spanner_text(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return False
    if len(cleaned) > 48:
        return False
    if re.match(r"^(?:table|fig(?:ure)?|表|图|琛|鍥)", cleaned, re.IGNORECASE):
        return False
    if re.match(r"^[*#$+\u2020\u2021]", cleaned):
        return False
    if re.search(r"[。！？!?；;。]$", cleaned):
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned))


def _row_has_page_locator_header_cues(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(segment.get("text", "") or "").strip()
        for segment in row.get("segments", []) or []
        if str(segment.get("text", "") or "").strip()
    )
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    return bool(
        re.search(r"(?:页码|頁碼|卷|volume|vol\.?|page|pages)", compact, re.IGNORECASE)
        and re.search(r"(?:试验|試驗|位置|编号|編號|study|location|number|no\.)", compact, re.IGNORECASE)
    )


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
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    if cjk_count >= 18:
        return True
    if cjk_count >= 10 and re.search(r"[，,。；;：:、]$", cleaned):
        return True
    word_count = len(cleaned.split())
    if word_count >= 11:
        return True
    if word_count >= 7 and re.search(r"[.;:!?，,。；;：:]$", cleaned):
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
    *,
    split_single_segment_by_column_anchors: bool = False,
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
            split_single_segment_by_column_anchors=split_single_segment_by_column_anchors,
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
    *,
    split_single_segment_by_column_anchors: bool = False,
) -> list[dict[str, Any]]:
    """Split under-segmented rows using stable column anchors and large-gap cues."""
    if len(segments) <= 1 and not split_single_segment_by_column_anchors:
        return list(segments)
    if not segments or not column_anchors or len(segments) >= len(column_anchors):
        return list(segments)

    refined_segments: list[dict[str, Any]] = []
    for segment in segments:
        refined_segments.extend(
            _split_segment_on_anchor_transitions(
                segment,
                column_anchors,
                split_by_word_anchor=split_single_segment_by_column_anchors,
            )
        )

    if len(refined_segments) <= len(segments) or len(refined_segments) > len(column_anchors):
        return list(segments)
    return refined_segments


def _split_segment_on_anchor_transitions(
    segment: dict[str, Any],
    column_anchors: list[float],
    *,
    split_by_word_anchor: bool = False,
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
    populated_word_anchors = {idx for idx in word_anchor_indices if idx is not None}
    if split_by_word_anchor and len(populated_word_anchors) >= 2:
        split_words_by_anchor: list[list[Any]] = []
        current_chunk: list[Any] = [words[0]]
        current_anchor = word_anchor_indices[0]
        for current_word, next_anchor in zip(words[1:], word_anchor_indices[1:]):
            if next_anchor != current_anchor:
                if _is_short_text_connector_word(current_word):
                    current_chunk.append(current_word)
                    continue
                split_words_by_anchor.append(current_chunk)
                current_chunk = [current_word]
                current_anchor = next_anchor
                continue
            current_chunk.append(current_word)
        if current_chunk:
            split_words_by_anchor.append(current_chunk)
        split_segments = [_build_segment(chunk) for chunk in split_words_by_anchor if chunk]
        if len(split_segments) >= 2:
            return split_segments

    split_words: list[list[Any]] = []
    current_chunk: list[Any] = [words[0]]
    current_anchor = word_anchor_indices[0]

    for previous_word, current_word, next_anchor in zip(words, words[1:], word_anchor_indices[1:]):
        gap = _word_x0(current_word) - _word_x1(previous_word)
        if gap >= large_gap_floor and next_anchor != current_anchor:
            if _is_short_text_connector_word(current_word):
                current_chunk.append(current_word)
                continue
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
    "extract_caption_anchored_horizontal_rule_tables",
    "extract_visual_structure_grid_tables",
    "extract_text_aligned_borderless_grid_tables",
    "extract_structured_text_region_tables",
    # Utility functions
    "filter_drawings_by_type",
    "get_grid_lines",
]
