"""Phase 4: Cell Semantics & State Machine - 单元格语义状态机

Architecture:
    Table Instance → Phase 4 → Cell Semantics Map

负责分析单元格语义：
- 状态机驱动单元格分类
- 续行合并检测
- 汇总行识别
- 注释单元格识别
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..assembly import TableInstance


class CellState(Enum):
    """单元格状态"""
    EMPTY = "empty"            # 空单元格
    DATA = "data"              # 数据单元格
    HEADER = "header"          # 表头单元格
    CONTINUATION = "continuation"  # 续行单元格
    AGGREGATE = "aggregate"    # 汇总单元格
    ANNOTATION = "annotation"  # 注释单元格


@dataclass
class CellSemantics:
    """Phase 4 结果：单元格语义"""
    state: CellState
    confidence: float
    signals: list[str] = field(default_factory=list)


def analyze_cell_semantics(
    instance: "TableInstance",
) -> dict[tuple[int, int], CellSemantics]:
    """Phase 4: 分析单元格语义

    状态机驱动：
    - 根据位置、内容、上下文确定单元格语义
    - 处理续行合并
    - 识别汇总行

    Args:
        instance: 当前表格实例

    Returns:
        Dict mapping (row, col) -> CellSemantics
    """
    semantics = {}

    for cell in instance.cells:
        row = cell.get("logical_row", cell.get("row", 1))
        col = cell.get("col", 1)
        text = cell.get("text", "")

        key = (row, col)

        # 状态判断
        if not text:
            semantics[key] = CellSemantics(
                state=CellState.EMPTY,
                confidence=1.0,
            )
            continue

        # 表头行
        if instance.header and row <= instance.header.row_index + 1:
            semantics[key] = CellSemantics(
                state=CellState.HEADER,
                confidence=0.9,
            )
            continue

        # 续行检查
        if _is_continuation_text(text):
            semantics[key] = CellSemantics(
                state=CellState.CONTINUATION,
                confidence=0.8,
                signals=["parenthetical_abbreviation"],
            )
            continue

        # 汇总行检查
        if _is_aggregate_text(text):
            semantics[key] = CellSemantics(
                state=CellState.AGGREGATE,
                confidence=0.7,
                signals=["aggregate_pattern"],
            )
            continue

        # 注释检查
        if _is_annotation_text(text):
            semantics[key] = CellSemantics(
                state=CellState.ANNOTATION,
                confidence=0.6,
                signals=["annotation_pattern"],
            )
            continue

        # 默认为数据单元格
        semantics[key] = CellSemantics(
            state=CellState.DATA,
            confidence=0.7,
        )

    return semantics


def _is_continuation_text(text: str) -> bool:
    """检查是否为续行文本"""
    if not text:
        return False
    # 括号缩写
    if re.fullmatch(r"[（(][A-Za-z0-9.\-_/]{2,28}[)）]", text.strip()):
        return True
    return False


def _is_aggregate_text(text: str) -> bool:
    """检查是否为汇总文本"""
    if not text:
        return False
    aggregate_patterns = [
        r"^(合计|总计|小计|共计|平均|mean|total|sum)",
        r"[总共计]\s*\d+",
    ]
    text_lower = text.lower()
    for pattern in aggregate_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def _is_annotation_text(text: str) -> bool:
    """检查是否为注释文本"""
    if not text:
        return False
    # 注释特征：星号、数字上标开头
    if re.match(r"^[*†‡§¶]|\d+\)", text.strip()):
        return True
    if re.match(r"^(注|说明|备注|note):", text.lower()):
        return True
    return False


__all__ = [
    "CellState",
    "CellSemantics",
    "analyze_cell_semantics",
]
