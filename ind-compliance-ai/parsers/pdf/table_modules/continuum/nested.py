"""Phase 5: Nested Structure Detection - 嵌套结构识别

Architecture:
    Table Instance + Cell Semantics → Phase 5 → Nested Structure

负责检测表格中的嵌套结构：
- 单元格内的嵌套表格
- 层级结构的行分组
- 合并单元格的嵌套含义
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..assembly import TableInstance
    from .semantics import CellSemantics


@dataclass
class NestedStructure:
    """Phase 5 结果：嵌套结构"""
    has_nested: bool
    nested_regions: list[dict[str, Any]] = field(default_factory=list)
    nesting_type: str | None = None


def detect_nested_structure(
    instance: "TableInstance",
    cell_semantics: dict[tuple[int, int], "CellSemantics"],
) -> NestedStructure:
    """Phase 5: 检测嵌套结构

    识别：
    - 单元格内的嵌套表格
    - 层级结构的行分组
    - 合并单元格的嵌套含义

    Args:
        instance: 当前表格实例
        cell_semantics: Phase 4 的单元格语义结果

    Returns:
        NestedStructure 包含嵌套信息
    """
    nested_regions = []

    for cell in instance.cells:
        rowspan = cell.get("rowspan", 1)
        colspan = cell.get("colspan", 1)

        if rowspan > 1 or colspan > 1:
            nested_regions.append({
                "row": cell.get("row"),
                "col": cell.get("col"),
                "rowspan": rowspan,
                "colspan": colspan,
                "type": "merged_cell",
            })

    has_nested = len(nested_regions) > 0
    nesting_type = "merged_cells" if has_nested else None

    return NestedStructure(
        has_nested=has_nested,
        nested_regions=nested_regions,
        nesting_type=nesting_type,
    )


__all__ = [
    "NestedStructure",
    "detect_nested_structure",
]
