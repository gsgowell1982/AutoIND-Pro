"""Continuum Engine - 逻辑表格连续体引擎

Architecture:
    Table Instance → Continuum Engine (6 Phases) → Logical Table AST

6 Phase 处理流程:
    Phase 1: Table Identity Resolution - 表格身份判定 (identity.py)
    Phase 2: Logical Grid Stabilization - 逻辑网格稳定 (grid.py)
    Phase 3: Cross-Page Continuity - 跨页连续性 (continuity.py)
    Phase 4: Cell Semantics & State Machine - 单元格语义状态机 (semantics.py)
    Phase 5: Nested Structure Detection - 嵌套结构识别 (nested.py)
    Phase 6: Confidence, Risk & Review Policy - 置信度评估 (confidence.py)

关键特性：
- 多阶段流水线处理
- 状态机驱动单元格语义分析
- 跨页表格拼接与继承
- 置信度评估与风险标记

Usage:
    from table_modules.continuum import run_continuum_engine

    result = run_continuum_engine(instance, prev_instances, context)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ============================================================================
# Phase 1: Table Identity Resolution
# ============================================================================

from .identity import (
    TableIdentity,
    IdentityResolution,
    resolve_table_identity,
)

# ============================================================================
# Phase 2: Logical Grid Stabilization
# ============================================================================

from .grid import (
    GridStabilization,
    stabilize_logical_grid,
)

# ============================================================================
# Phase 3: Cross-Page Continuity
# ============================================================================

from .continuity import (
    ContinuityResult,
    establish_cross_page_continuity,
)

# ============================================================================
# Phase 4: Cell Semantics & State Machine
# ============================================================================

from .semantics import (
    CellState,
    CellSemantics,
    analyze_cell_semantics,
)

# ============================================================================
# Phase 5: Nested Structure Detection
# ============================================================================

from .nested import (
    NestedStructure,
    detect_nested_structure,
)

# ============================================================================
# Phase 6: Confidence, Risk & Review Policy
# ============================================================================

from .confidence import (
    TOC_LINE_PATTERN,
    REFERENCE_ROW_PATTERN,
    ConfidenceAssessment,
    assess_confidence,
)


# ============================================================================
# Continuum Engine Pipeline
# ============================================================================

@dataclass
class ContinuumResult:
    """连续体引擎处理结果"""
    identity: IdentityResolution
    grid: GridStabilization
    continuity: ContinuityResult
    cell_semantics: dict[tuple[int, int], CellSemantics]
    nested: NestedStructure
    confidence: ConfidenceAssessment


def run_continuum_engine(
    instance: Any,
    prev_instances: list[Any],
    context: dict[str, Any] | None = None,
) -> ContinuumResult:
    """运行连续体引擎

    执行所有 6 个 Phase，返回完整结果。

    Args:
        instance: 表格实例 (TableInstance)
        prev_instances: 前序表格列表（用于续表检测）
        context: 上下文信息（title_block, section_hint, toc_context, grid_line_score）

    Returns:
        ContinuumResult 包含所有阶段结果
    """
    # Phase 1: Table Identity Resolution
    identity = resolve_table_identity(instance, prev_instances)

    # Phase 2: Logical Grid Stabilization
    grid = stabilize_logical_grid(instance, identity)

    # Phase 3: Cross-Page Continuity
    continuity = establish_cross_page_continuity(
        instance, identity, prev_instances
    )

    # Phase 4: Cell Semantics & State Machine
    cell_semantics = analyze_cell_semantics(instance)

    # Phase 5: Nested Structure Detection
    nested = detect_nested_structure(instance, cell_semantics)

    # Phase 6: Confidence, Risk & Review Policy
    confidence = assess_confidence(
        instance, identity, grid, continuity, nested, context
    )

    return ContinuumResult(
        identity=identity,
        grid=grid,
        continuity=continuity,
        cell_semantics=cell_semantics,
        nested=nested,
        confidence=confidence,
    )


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
# Exports
# ============================================================================

__all__ = [
    # Phase 1
    "TableIdentity",
    "IdentityResolution",
    "resolve_table_identity",
    # Phase 2
    "GridStabilization",
    "stabilize_logical_grid",
    # Phase 3
    "ContinuityResult",
    "establish_cross_page_continuity",
    # Phase 4
    "CellState",
    "CellSemantics",
    "analyze_cell_semantics",
    # Phase 5
    "NestedStructure",
    "detect_nested_structure",
    # Phase 6
    "TOC_LINE_PATTERN",
    "REFERENCE_ROW_PATTERN",
    "ConfidenceAssessment",
    "assess_confidence",
    # Pipeline
    "ContinuumResult",
    "run_continuum_engine",
    # Utilities
    "column_similarity",
    "header_similarity",
]
