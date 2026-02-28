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

from __future__ import annotations

from dataclasses import dataclass, field
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
        
        # Step 2: 提取 drawings
        drawings = _extract_drawings_from_page(page, table_bbox)
        
        # Step 3: 构建行数据
        rows = []
        for row_idx, row_data in enumerate(raw_table_data):
            cells = []
            for col_idx, cell_text in enumerate(row_data):
                cell = RawCell(
                    physical_col=col_idx,
                    physical_row=row_idx,
                    text=str(cell_text).strip() if cell_text else None,
                )
                cells.append(cell)
            
            row = RawRow(
                physical_row=row_idx,
                cells=cells,
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
        
        # 取最大的表格组
        largest_group = max(table_groups, key=len)
        if len(largest_group) < 2:
            return None
        
        # Step 3: 计算列中心
        all_words_in_group = [w for row in largest_group for w in row.get("words", [])]
        if not all_words_in_group:
            return None
        
        column_centers = _detect_column_centers(all_words_in_group)
        if not column_centers or len(column_centers) > 20:
            return None
        
        col_count = len(column_centers)
        
        # Step 4: 构建原始数据
        raw_data = []
        spans = []
        
        for row in largest_group:
            row_data = [None] * col_count
            row_words = row.get("words", [])
            
            for word in row_words:
                # 找到对应的列
                word_x_center = (word.x0 + word.x1) / 2
                col_idx = _find_column_for_word(word_x_center, column_centers)
                
                if col_idx is not None and col_idx < col_count:
                    text = word.text if hasattr(word, 'text') else str(word)
                    if row_data[col_idx] is None:
                        row_data[col_idx] = text
                    else:
                        row_data[col_idx] += " " + text
                    
                    # 构建 span
                    span = RawSpan(
                        text=text,
                        x0=word.x0,
                        y0=word.y0,
                        x1=word.x1,
                        y1=word.y1,
                    )
                    spans.append(span)
            
            raw_data.append(row_data)
        
        # Step 5: 构建 bbox
        all_x0 = [w.x0 for w in all_words_in_group]
        all_y0 = [w.y0 for w in all_words_in_group]
        all_x1 = [w.x1 for w in all_words_in_group]
        all_y1 = [w.y1 for w in all_words_in_group]
        
        table_bbox = (min(all_x0), min(all_y0), max(all_x1), max(all_y1))
        
        # Step 6: 构建 RawTableEvidence
        evidence = RawTableEvidence(
            page_number=page_number,
            bbox=table_bbox,
            physical_col_count=col_count,
            physical_row_count=len(raw_data),
            chars=[],
            spans=spans,
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
        if hasattr(w, 'y1') and hasattr(w, 'y0'):
            heights.append(w.y1 - w.y0)
    
    median_height = statistics.median(heights) if heights else 8.0
    row_tol = max(2.5, median_height * 0.55)
    
    rows: list[dict[str, Any]] = []
    
    for word in sorted(words, key=lambda item: (
        (item.y0 + item.y1) / 2 if hasattr(item, 'y0') else 0,
        item.x0 if hasattr(item, 'x0') else 0
    )):
        word_yc = (word.y0 + word.y1) / 2 if hasattr(word, 'y0') else 0
        
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
        row_words = sorted(row["words"], key=lambda item: item.x0 if hasattr(item, 'x0') else 0)
        
        # 计算 bbox
        x0 = min((w.x0 for w in row_words), default=0)
        y0 = min((w.y0 for w in row_words), default=0)
        x1 = max((w.x1 for w in row_words), default=0)
        y1 = max((w.y1 for w in row_words), default=0)
        
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
