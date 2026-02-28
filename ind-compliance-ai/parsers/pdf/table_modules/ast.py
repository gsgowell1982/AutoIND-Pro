"""Logical Table AST - 逻辑表格抽象语法树

Architecture:
    Continuum Engine → Logical Table AST

负责生成最终的表格 AST 表示：
- 标准化的表格数据结构
- 兼容现有 API
- 包含完整元数据

关键特性：
- 向后兼容现有 AST 格式
- 包含物理和逻辑信息
- 记录处理过程元数据
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .assembly import TableInstance, TableHeader
from .continuum import (
    ContinuumResult,
    IdentityResolution,
    TableIdentity,
    CellSemantics,
    CellState,
)


@dataclass
class LogicalTableAST:
    """逻辑表格 AST
    
    最终的表格表示，包含：
    - 标准表格数据 (兼容现有 API)
    - 物理与逻辑信息
    - 处理元数据
    - 置信度和审核信息
    """
    # 标准字段 (兼容现有 API)
    block_type: str = "table"
    table_id: str = ""
    page: int = 0
    bbox: list[float] = field(default_factory=list)
    header: list[dict[str, Any]] = field(default_factory=list)
    cells: list[dict[str, Any]] = field(default_factory=list)
    grid: list[list[str | None]] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    
    # 物理与逻辑信息
    logical_row_count: int = 0
    logical_col_count: int = 0
    physical_row_count: int = 0
    physical_col_count: int = 0
    
    # 列签名
    column_signature: list[float] = field(default_factory=list)
    column_hash: str = ""
    
    # 位置信息
    near_page_top: bool = False
    near_page_bottom: bool = False
    
    # 续表信息
    is_continuation: bool = False
    continued_from: str | None = None
    continued_to: list[str] = field(default_factory=list)
    header_inherited: bool = False
    title_inherited: bool = False
    
    # 继承元数据
    continuation_source: dict[str, Any] | None = None
    
    # 上下文
    title: str | None = None
    section_hint: str | None = None
    toc_context: bool = False
    grid_line_score: float = 0.0
    
    # 处理元数据
    detection_method: str = "pymupdf_builtin"
    normalization_strategy: str = "column_clustering"
    
    # 置信度和审核
    confidence: float = 0.0
    risk_flags: list[str] = field(default_factory=list)
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    
    # 行文本
    row_texts: list[str] = field(default_factory=list)
    
    # 结构分数
    structure_score: float = 0.0
    toc_row_ratio: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式 (兼容现有 API)"""
        result = {
            "block_type": self.block_type,
            "table_id": self.table_id,
            "page": self.page,
            "bbox": self.bbox,
            "header": self.header,
            "cells": self.cells,
            "grid": self.grid,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "logical_row_count": self.logical_row_count,
            "logical_col_count": self.logical_col_count,
            "physical_row_count": self.physical_row_count,
            "physical_col_count": self.physical_col_count,
            "column_signature": self.column_signature,
            "column_hash": self.column_hash,
            "near_page_top": self.near_page_top,
            "near_page_bottom": self.near_page_bottom,
            "detection_method": self.detection_method,
            "row_texts": self.row_texts,
            "structure_score": self.structure_score,
            "toc_row_ratio": self.toc_row_ratio,
        }
        
        # 可选字段
        if self.is_continuation:
            result["is_continuation"] = True
        if self.continued_from:
            result["continued_from"] = self.continued_from
        if self.continued_to:
            result["continued_to"] = self.continued_to
        if self.header_inherited:
            result["header_inherited"] = True
        if self.title_inherited:
            result["title_inherited"] = True
        if self.continuation_source:
            result["continuation_source"] = self.continuation_source
        if self.title:
            result["title"] = self.title
        if self.section_hint:
            result["section_hint"] = self.section_hint
        if self.confidence > 0:
            result["confidence"] = round(self.confidence, 3)
        if self.risk_flags:
            result["risk_flags"] = self.risk_flags
        if self.review_required:
            result["review_required"] = True
        if self.review_reasons:
            result["review_reasons"] = self.review_reasons
        
        return result


# ============================================================================
# AST Builder
# ============================================================================

def build_logical_ast(
    instance: TableInstance,
    continuum_result: ContinuumResult,
) -> LogicalTableAST:
    """构建逻辑表格 AST
    
    将 TableInstance 和 ContinuumResult 合并为最终 AST。
    
    Args:
        instance: 表格实例
        continuum_result: 连续体引擎处理结果
        
    Returns:
        LogicalTableAST 最终表格表示
    """
    # 构建表头列表
    header_list = []
    if instance.header:
        header_list = instance.header.cells.copy()
    
    # 计算结构分数
    structure_score = _calculate_structure_score(instance)
    
    # 计算 TOC 行比例
    toc_row_ratio = _calculate_toc_row_ratio(instance)
    
    # 构建 AST
    ast = LogicalTableAST(
        block_type="table",
        table_id=instance.table_id,
        page=instance.page_number,
        bbox=list(instance.bbox),
        header=header_list,
        cells=instance.cells,
        grid=instance.grid,
        row_count=instance.row_count,
        col_count=instance.col_count,
        logical_row_count=instance.row_count,
        logical_col_count=instance.col_count,
        physical_row_count=instance.physical_row_count,
        physical_col_count=instance.physical_col_count,
        column_signature=instance.column_signature,
        column_hash=instance.column_hash,
        near_page_top=instance.near_page_top,
        near_page_bottom=instance.near_page_bottom,
        is_continuation=instance.is_continuation or continuum_result.continuity.is_continuation,
        header_inherited=instance.header.inherited if instance.header else False,
        title=instance.title,
        section_hint=instance.section_hint,
        toc_context=instance.toc_context,
        grid_line_score=instance.grid_line_score,
        detection_method=instance.detection_method,
        normalization_strategy=instance.normalization_strategy,
        confidence=continuum_result.confidence.overall_confidence,
        risk_flags=continuum_result.confidence.risk_flags,
        review_required=continuum_result.confidence.review_required,
        review_reasons=continuum_result.confidence.review_reasons,
        row_texts=instance.row_texts,
        structure_score=structure_score,
        toc_row_ratio=toc_row_ratio,
    )
    
    # 设置续表信息
    if continuum_result.continuity.is_continuation:
        ast.continued_from = continuum_result.continuity.parent_table_id
        
        # 构建继承元数据
        if continuum_result.continuity.inherited_fields:
            ast.continuation_source = {
                "source_table_id": continuum_result.continuity.parent_table_id,
                "strategy": "continuum_engine",
                "similarity": continuum_result.continuity.similarity_score,
                "inherited_fields": continuum_result.continuity.inherited_fields,
            }
    
    return ast


def _calculate_structure_score(instance: TableInstance) -> float:
    """计算结构分数"""
    if not instance.cells:
        return 0.0
    
    # 非空单元格比例
    non_null_cells = sum(1 for c in instance.cells if c.get("text"))
    total_slots = instance.row_count * instance.col_count
    coverage = non_null_cells / max(1, total_slots)
    
    # 行对齐度
    aligned_rows = sum(1 for row in instance.grid if any(cell for cell in row))
    alignment = aligned_rows / max(1, len(instance.grid))
    
    # 数值单元格比例
    numeric_cells = sum(
        1 for c in instance.cells
        if c.get("text") and any(ch.isdigit() for ch in c["text"])
    )
    numeric_ratio = numeric_cells / max(1, non_null_cells)
    
    return min(1.0, 0.45 * coverage + 0.35 * alignment + 0.2 * numeric_ratio)


def _calculate_toc_row_ratio(instance: TableInstance) -> float:
    """计算 TOC 行比例"""
    if not instance.row_texts:
        return 0.0
    
    import re
    
    toc_patterns = [
        r"[\.·…]{4,}\s*\d{1,3}\s*$",  # 点号引导
        r"^\s*\d+(?:\.\d+){1,4}\s+\S+\s+\d{1,3}\s*$",  # 编号格式
    ]
    
    toc_rows = 0.0
    for row_text in instance.row_texts:
        for pattern in toc_patterns:
            if re.search(pattern, row_text):
                toc_rows += 1.0
                break
    
    return min(1.0, toc_rows / len(instance.row_texts))


# ============================================================================
# Compatibility Layer
# ============================================================================

def ast_to_legacy_dict(ast: LogicalTableAST) -> dict[str, Any]:
    """将 AST 转换为旧格式字典 (向后兼容)
    
    确保现有 API 消费者不受影响。
    """
    return ast.to_dict()


def merge_ast_with_context(
    ast: LogicalTableAST,
    title_block: dict[str, Any] | None = None,
    section_hint_block: dict[str, Any] | None = None,
    toc_context: bool = False,
    grid_line_score: float = 0.0,
    continuation_hint: dict[str, Any] | None = None,
) -> LogicalTableAST:
    """合并上下文信息到 AST
    
    用于整合 validation 层的上下文检测结果。
    """
    if title_block:
        ast.title = title_block.get("text", "").strip()
    
    if section_hint_block:
        ast.section_hint = section_hint_block.get("text", "").strip()
    
    ast.toc_context = toc_context
    ast.grid_line_score = grid_line_score
    
    if continuation_hint:
        ast.continued_from = continuation_hint.get("table_id")
        ast.is_continuation = True
    
    return ast


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "LogicalTableAST",
    "build_logical_ast",
    "ast_to_legacy_dict",
    "merge_ast_with_context",
    # 内部工具函数 (供新架构使用)
    "_calculate_structure_score",
    "_calculate_toc_row_ratio",
]
