"""Phase 1: Table Identity Resolution - 表格身份判定

Architecture:
    Table Instance → Phase 1 → Identity Resolution

负责判定表格的身份类型：
- NEW_TABLE: 新表格
- CONTINUATION: 续表
- FRAGMENT: 表格片段
- NESTED: 嵌套表格

判断依据：
- 页面位置（顶部/底部）
- 前序表格关联
- 表头特征
- 续表标记
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..assembly import TableInstance


class TableIdentity(Enum):
    """表格身份类型"""
    NEW_TABLE = "new_table"        # 新表格
    CONTINUATION = "continuation"  # 续表
    FRAGMENT = "fragment"          # 表格片段
    NESTED = "nested"              # 嵌套表格


@dataclass
class IdentityResolution:
    """Phase 1 结果：表格身份解析"""
    identity: TableIdentity
    confidence: float
    parent_table_id: str | None = None
    signals: list[str] = field(default_factory=list)


def resolve_table_identity(
    instance: "TableInstance",
    prev_instances: list["TableInstance"],
) -> IdentityResolution:
    """Phase 1: 解析表格身份

    判断当前表格是：
    - 新表格
    - 续表 (继续上一页的表格)
    - 表格片段 (需要与其他片段合并)
    - 嵌套表格

    Args:
        instance: 当前表格实例
        prev_instances: 前序表格列表（用于续表检测）

    Returns:
        IdentityResolution 包含身份类型和置信度
    """
    signals = []

    # Signal 1: 页面顶部位置
    if instance.near_page_top:
        signals.append("near_page_top")

    # Signal 2: 有父表候选
    parent_candidate = None
    if prev_instances:
        for prev in reversed(prev_instances[-10:]):
            if (prev.page_number == instance.page_number - 1 and
                prev.near_page_bottom):
                # 检查列签名相似度
                if _column_signature_similarity(
                    instance.column_signature,
                    prev.column_signature
                ) >= 0.7:
                    parent_candidate = prev
                    signals.append("parent_candidate_found")
                    break

    # Signal 3: 表头特征
    has_header = instance.header and not instance.header.inherited
    if has_header:
        signals.append("has_explicit_header")
    else:
        signals.append("no_explicit_header")

    # Signal 4: 续表标记
    if instance.is_continuation:
        signals.append("marked_as_continuation")

    # 判定身份
    if parent_candidate and instance.near_page_top and not has_header:
        return IdentityResolution(
            identity=TableIdentity.CONTINUATION,
            confidence=0.9,
            parent_table_id=parent_candidate.table_id,
            signals=signals,
        )

    if instance.needs_header_inheritance:
        return IdentityResolution(
            identity=TableIdentity.CONTINUATION,
            confidence=0.8,
            parent_table_id=parent_candidate.table_id if parent_candidate else None,
            signals=signals,
        )

    if instance.col_count >= 3 and instance.row_count <= 3:
        return IdentityResolution(
            identity=TableIdentity.FRAGMENT,
            confidence=0.6,
            signals=signals,
        )

    return IdentityResolution(
        identity=TableIdentity.NEW_TABLE,
        confidence=0.8,
        signals=signals,
    )


def _column_signature_similarity(
    sig_a: list[float],
    sig_b: list[float],
) -> float:
    """计算列签名相似度"""
    if not sig_a or not sig_b:
        return 0.0

    tolerance = 0.07
    matched = 0

    for value in sig_a:
        if any(abs(value - other) <= tolerance for other in sig_b):
            matched += 1

    return matched / max(len(sig_a), len(sig_b))


__all__ = [
    "TableIdentity",
    "IdentityResolution",
    "resolve_table_identity",
]
