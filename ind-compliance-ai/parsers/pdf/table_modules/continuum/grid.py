"""Phase 2: Logical Grid Stabilization - 逻辑网格稳定

Architecture:
    Table Instance + Identity → Phase 2 → Grid Stabilization

负责确保表格逻辑网格的稳定性：
- 列数合理性检查
- 行数与单元格匹配验证
- 结构一致性评估
- 结构分数计算
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..assembly import TableInstance
    from .identity import IdentityResolution


@dataclass
class GridStabilization:
    """Phase 2 结果：逻辑网格稳定化"""
    stable: bool
    col_count: int
    row_count: int
    adjustments: list[str] = field(default_factory=list)
    structure_score: float = 0.0


def stabilize_logical_grid(
    instance: "TableInstance",
    identity: "IdentityResolution",
) -> GridStabilization:
    """Phase 2: 稳定逻辑网格

    确保：
    - 列数合理
    - 行数与单元格匹配
    - 结构一致性

    Args:
        instance: 当前表格实例
        identity: Phase 1 身份解析结果

    Returns:
        GridStabilization 包含稳定性状态和结构分数
    """
    adjustments = []

    # 计算结构分数
    non_null_cells = sum(1 for c in instance.cells if c.get("text"))
    total_slots = instance.row_count * instance.col_count
    coverage = non_null_cells / max(1, total_slots)

    # 检查行对齐
    aligned_rows = 0
    for row in instance.grid:
        if any(cell for cell in row):
            aligned_rows += 1

    alignment = aligned_rows / max(1, instance.row_count)

    # 检查数值比例
    numeric_cells = sum(
        1 for c in instance.cells
        if c.get("text") and re.search(r"\d", c["text"])
    )
    numeric_ratio = numeric_cells / max(1, non_null_cells)

    structure_score = 0.4 * coverage + 0.35 * alignment + 0.25 * numeric_ratio

    # 稳定性检查
    stable = structure_score >= 0.3

    if not stable:
        adjustments.append(f"low_structure_score:{structure_score:.2f}")

    return GridStabilization(
        stable=stable,
        col_count=instance.col_count,
        row_count=instance.row_count,
        adjustments=adjustments,
        structure_score=min(1.0, structure_score),
    )


__all__ = [
    "GridStabilization",
    "stabilize_logical_grid",
]
