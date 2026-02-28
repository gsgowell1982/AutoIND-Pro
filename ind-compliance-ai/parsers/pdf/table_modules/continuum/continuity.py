"""Phase 3: Cross-Page Continuity - 跨页连续性

Architecture:
    Table Instance + Identity → Phase 3 → Continuity Result

负责处理表格的跨页连续性：
- 设置 continued_from/continued_to 链接
- 继承表头和标题
- 记录继承元数据
- 跨页相似度计算
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..assembly import TableInstance, TableHeader
    from .identity import IdentityResolution, TableIdentity


@dataclass
class ContinuityResult:
    """Phase 3 结果：跨页连续性"""
    is_continuation: bool
    parent_table_id: str | None = None
    inherited_fields: list[str] = field(default_factory=list)
    similarity_score: float = 0.0


def establish_cross_page_continuity(
    instance: "TableInstance",
    identity: "IdentityResolution",
    prev_instances: list["TableInstance"],
) -> ContinuityResult:
    """Phase 3: 建立跨页连续性

    处理：
    - 设置 continued_from/continued_to 链接
    - 继承表头和标题
    - 记录继承元数据

    Args:
        instance: 当前表格实例
        identity: Phase 1 身份解析结果
        prev_instances: 前序表格列表

    Returns:
        ContinuityResult 包含连续性信息
    """
    from .identity import TableIdentity

    if identity.identity != TableIdentity.CONTINUATION:
        return ContinuityResult(is_continuation=False)

    if not identity.parent_table_id:
        return ContinuityResult(is_continuation=True)

    # 找到父表
    parent = None
    for prev in prev_instances:
        if prev.table_id == identity.parent_table_id:
            parent = prev
            break

    if not parent:
        return ContinuityResult(
            is_continuation=True,
            parent_table_id=identity.parent_table_id,
        )

    # 计算相似度
    similarity = _column_signature_similarity(
        instance.column_signature,
        parent.column_signature,
    )

    inherited_fields = []

    # 继承表头
    if parent.header and (not instance.header or instance.header.inherited):
        instance.header = _create_inherited_header(parent.header)
        inherited_fields.append("header")

    # 继承标题
    if parent.title and not instance.title:
        instance.title = parent.title
        inherited_fields.append("title")

    # 设置续表标记
    instance.is_continuation = True

    return ContinuityResult(
        is_continuation=True,
        parent_table_id=parent.table_id,
        inherited_fields=inherited_fields,
        similarity_score=similarity,
    )


def _create_inherited_header(parent_header: "TableHeader") -> "TableHeader":
    """创建继承的表头"""
    from ..assembly import TableHeader

    return TableHeader(
        cells=parent_header.cells,
        row_index=0,
        inherited=True,
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
    "ContinuityResult",
    "establish_cross_page_continuity",
]
