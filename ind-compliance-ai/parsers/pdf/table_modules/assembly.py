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

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .normalization import NormalizedTable, NormalizedCell, NormalizedRow
from .raw_objects import RawTableEvidence


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
    row_count: int  # 逻辑行数
    header: TableHeader | None = None
    
    # 单元格
    cells: list[dict[str, Any]] = field(default_factory=list)
    grid: list[list[str | None]] = field(default_factory=list)
    
    # 行文本 (用于验证)
    row_texts: list[str] = field(default_factory=list)
    
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
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "block_type": "table",
            "table_id": self.table_id,
            "page": self.page_number,
            "bbox": list(self.bbox),
            "col_count": self.col_count,
            "row_count": self.row_count,
            "header": self.header.to_dict() if self.header else None,
            "cells": self.cells,
            "grid": self.grid,
            "row_texts": self.row_texts,
            "physical_col_count": self.physical_col_count,
            "physical_row_count": self.physical_row_count,
            "near_page_top": self.near_page_top,
            "near_page_bottom": self.near_page_bottom,
            "is_continuation": self.is_continuation,
            "column_hash": self.column_hash,
            "detection_method": self.detection_method,
        }


# ============================================================================
# Assembly Functions
# ============================================================================

def assemble_table_instance(
    normalized: NormalizedTable,
    table_id: str,
    parent_header: list[dict[str, Any]] | None = None,
) -> TableInstance:
    """组装表格实例
    
    核心步骤：
    1. 检测是否为续表
    2. 确定表头
    3. 处理单元格合并
    4. 构建列签名
    
    Args:
        normalized: 规范化表格数据
        table_id: 表格 ID
        parent_header: 父表表头 (用于续表继承)
        
    Returns:
        TableInstance 表格实例
    """
    # Step 1: 检测续表
    is_continuation = _detect_continuation(normalized, parent_header)
    
    # Step 2: 确定表头
    header, data_start_row = _determine_header(
        normalized,
        is_continuation,
        parent_header,
    )
    
    # Step 3: 处理单元格
    cells = _build_cells(
        normalized,
        data_start_row,
        header.row_index if header else 0,
    )
    
    # Step 4: 单元格合并处理
    cells = _apply_cell_merging(cells, normalized.grid)
    
    # Step 5: 构建行文本
    row_texts = _build_row_texts(normalized.grid)
    
    # Step 6: 构建列签名
    column_signature = _build_column_signature(normalized)
    column_hash = _compute_column_hash(column_signature)
    
    # 构建表格实例
    instance = TableInstance(
        table_id=table_id,
        page_number=normalized.page_number,
        bbox=normalized.bbox,
        col_count=normalized.logical_col_count,
        row_count=len(normalized.rows),
        header=header,
        cells=cells,
        grid=normalized.grid,
        row_texts=row_texts,
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
    )
    
    return instance


def _detect_continuation(
    normalized: NormalizedTable,
    parent_header: list[dict[str, Any]] | None,
) -> bool:
    """检测是否为续表
    
    续表特征：
    1. 页面顶部位置
    2. 首行像数据行（不是表头）
    3. 有父表上下文
    """
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
) -> tuple[TableHeader | None, int]:
    """确定表头
    
    Returns:
        (header, data_start_row)
    """
    col_count = normalized.logical_col_count
    
    if is_continuation and parent_header:
        # 续表继承父表表头
        header = TableHeader(
            cells=parent_header,
            row_index=0,
            inherited=True,
        )
        return header, 0
    
    # 使用首行作为表头
    first_row = normalized.rows[0] if normalized.rows else None
    if not first_row:
        return None, 0
    
    header_cells = []
    for cell in first_row.cells:
        header_cells.append({
            "col": cell.logical_col + 1,
            "text": cell.text or f"Column {cell.logical_col + 1}",
        })
    
    # 确保所有列都有表头
    existing_cols = {c["col"] for c in header_cells}
    for col in range(1, col_count + 1):
        if col not in existing_cols:
            header_cells.append({
                "col": col,
                "text": f"Column {col}",
            })
    
    header_cells.sort(key=lambda c: c["col"])
    
    header = TableHeader(
        cells=header_cells,
        row_index=0,
        inherited=False,
    )
    
    return header, 1


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
                "rowspan": 1,
                "colspan": 1,
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
        normalized_cells = [_clean_text(cell) for cell in row]
        non_empty_parts = [text for text in normalized_cells if text]
        if not non_empty_parts:
            continue
        if len(non_empty_parts) == 1:
            row_texts.append(non_empty_parts[0])
            continue
        row_texts.append(" ".join(non_empty_parts))
    return row_texts


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
    "TableInstance",
    "assemble_table_instance",
    # Edge row pruning (整合自 validation.py)
    "is_heading_like_row_text",
    "is_section_heading_like_tail",
    "prune_non_table_edge_rows",
]
