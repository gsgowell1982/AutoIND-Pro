"""
eCTD Leaf操作分析模块

负责：
1. Leaf节点操作类型识别（new/delete/replace/append）
2. Replace操作精确语义分析
3. 操作影响范围评估
4. 元数据变更关联分析

版本: v1.0
创建日期: 2026-09-11
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Leaf操作类型枚举
# ============================================================================

class LeafOperationType(str, Enum):
    """
    Leaf节点操作类型

    根据ICH eCTD DTD 3.2规范：
    - new: 新增leaf节点
    - delete: 删除leaf节点
    - replace: 替换leaf节点
    - append: 追加leaf节点（某些情况视为new）
    """
    NEW = "new"
    DELETE = "delete"
    REPLACE = "replace"
    APPEND = "append"


class ReplaceSemantics(str, Enum):
    """
    Replace操作的语义类型

    根据eCTD 3.6章节规范：
    - FULL_CONTENT_UPDATE: 完整内容更新（元数据未变，内容全量更新）
    - METADATA_CONTENT_COUPLING: 元数据-内容耦合更新（元数据变更，内容必须全量更新）
    - MINOR_CORRECTION: 轻微修正（错别字、格式修正，元数据不变）
    - UNKNOWN: 无法判断（需要人工审核）
    """
    FULL_CONTENT_UPDATE = "full_content_update"
    METADATA_CONTENT_COUPLING = "metadata_content_coupling"
    MINOR_CORRECTION = "minor_correction"
    UNKNOWN = "unknown"


# ============================================================================
# Leaf操作数据结构
# ============================================================================

@dataclass
class LeafOperation:
    """
    单个Leaf节点的操作信息

    Attributes:
        leaf_id: Leaf的ID属性（XML中的ID）
        operation_type: 操作类型（new/delete/replace/append）
        checksum: Leaf文件的MD5校验和
        file_path: Leaf文件的相对路径（相对于sequence根目录）
        section_path: Leaf所属的section路径（用于关联section元数据）
        title: Leaf的标题（可选）

    Examples:
        >>> op = LeafOperation(
        ...     leaf_id="l-m2-3-s-substance-api-a-001",
        ...     operation_type=LeafOperationType.REPLACE,
        ...     checksum="abc123def456",
        ...     file_path="m2/23s/substance-api-a.pdf",
        ...     section_path="m2-3-s-drug-substance[substance='API-A']"
        ... )
    """
    leaf_id: str
    operation_type: LeafOperationType
    checksum: str
    file_path: str
    section_path: str
    title: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"LeafOperation(id={self.leaf_id}, "
            f"type={self.operation_type.value}, "
            f"path={self.file_path})"
        )


@dataclass
class LeafOperationAnalysis:
    """
    Leaf操作的详细分析结果

    用于判断Replace操作的语义和影响范围

    Attributes:
        leaf_operation: 原始Leaf操作信息
        replace_semantics: Replace操作的语义类型（仅当operation_type=REPLACE时有效）
        previous_leaf_id: 前序列中对应的leaf_id（用于跨序列比对）
        previous_checksum: 前序列中的校验和（用于判断内容是否变更）
        metadata_changed: 所属section的元数据是否变更
        metadata_changes: 元数据变更详情（哪些属性变了）
        requires_full_update: 是否需要完整内容更新（基于元数据生命周期耦合规则）
        is_compliant: 是否符合规范要求
        violation_reason: 如果不符合，违规原因

    Examples:
        >>> analysis = LeafOperationAnalysis(
        ...     leaf_operation=LeafOperation(...),
        ...     replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING,
        ...     previous_leaf_id="l-m2-3-s-substance-api-a-001",
        ...     previous_checksum="old_checksum",
        ...     metadata_changed=True,
        ...     metadata_changes={"manufacturer": ("MFR-X", "MFR-Y")},
        ...     requires_full_update=True,
        ...     is_compliant=True,
        ...     violation_reason=None
        ... )
    """
    leaf_operation: LeafOperation
    replace_semantics: Optional[ReplaceSemantics] = None
    previous_leaf_id: Optional[str] = None
    previous_checksum: Optional[str] = None
    metadata_changed: bool = False
    metadata_changes: Dict[str, tuple] = field(default_factory=dict)
    requires_full_update: bool = False
    is_compliant: bool = True
    violation_reason: Optional[str] = None

    def get_operation_summary(self) -> str:
        """
        获取操作的简要描述

        Returns:
            描述字符串

        Examples:
            >>> analysis.get_operation_summary()
            'REPLACE (metadata_content_coupling): manufacturer changed, full update compliant'
        """
        op_type = self.leaf_operation.operation_type.value.upper()

        if self.leaf_operation.operation_type == LeafOperationType.REPLACE:
            semantics = self.replace_semantics.value if self.replace_semantics else "unknown"
            compliance = "compliant" if self.is_compliant else f"VIOLATION: {self.violation_reason}"
            if self.metadata_changed:
                changed_attrs = ", ".join(self.metadata_changes.keys())
                return f"{op_type} ({semantics}): {changed_attrs} changed, {compliance}"
            else:
                return f"{op_type} ({semantics}): {compliance}"
        else:
            return f"{op_type}: {self.leaf_operation.file_path}"

    def __repr__(self) -> str:
        return f"LeafOperationAnalysis({self.get_operation_summary()})"


# ============================================================================
# Replace操作语义判断逻辑
# ============================================================================

def determine_replace_semantics(
    current_checksum: str,
    previous_checksum: Optional[str],
    metadata_changed: bool,
    section_requires_lifecycle_tracking: bool
) -> ReplaceSemantics:
    """
    判断Replace操作的语义类型

    Args:
        current_checksum: 当前序列的leaf文件校验和
        previous_checksum: 前序列的leaf文件校验和（如果存在）
        metadata_changed: 所属section的元数据是否变更
        section_requires_lifecycle_tracking: 所属section是否需要元数据生命周期追踪

    Returns:
        Replace操作的语义类型

    Decision Logic:
        1. 如果元数据变更 + section需要lifecycle tracking
           → METADATA_CONTENT_COUPLING（必须完整更新内容）

        2. 如果元数据未变 + 内容变更（checksum不同）
           → FULL_CONTENT_UPDATE（完整内容更新）

        3. 如果元数据未变 + 内容未变（checksum相同）
           → MINOR_CORRECTION（可能是轻微修正，但需人工确认）

        4. 其他情况
           → UNKNOWN（需要人工审核）

    Examples:
        >>> # 场景1：元数据变更，内容也变更（符合规范）
        >>> semantics = determine_replace_semantics(
        ...     current_checksum="new_hash",
        ...     previous_checksum="old_hash",
        ...     metadata_changed=True,
        ...     section_requires_lifecycle_tracking=True
        ... )
        >>> semantics == ReplaceSemantics.METADATA_CONTENT_COUPLING
        True

        >>> # 场景2：元数据未变，内容变更（正常更新）
        >>> semantics = determine_replace_semantics(
        ...     current_checksum="new_hash",
        ...     previous_checksum="old_hash",
        ...     metadata_changed=False,
        ...     section_requires_lifecycle_tracking=True
        ... )
        >>> semantics == ReplaceSemantics.FULL_CONTENT_UPDATE
        True

        >>> # 场景3：元数据未变，内容也未变（可能是格式修正）
        >>> semantics = determine_replace_semantics(
        ...     current_checksum="same_hash",
        ...     previous_checksum="same_hash",
        ...     metadata_changed=False,
        ...     section_requires_lifecycle_tracking=True
        ... )
        >>> semantics == ReplaceSemantics.MINOR_CORRECTION
        True
    """
    # 场景1：元数据变更 + 需要生命周期追踪
    if metadata_changed and section_requires_lifecycle_tracking:
        return ReplaceSemantics.METADATA_CONTENT_COUPLING

    # 场景2：内容变更（无论元数据是否变更）
    if previous_checksum and current_checksum != previous_checksum:
        if metadata_changed and not section_requires_lifecycle_tracking:
            # 元数据变更但不需要追踪，仍然是完整内容更新
            return ReplaceSemantics.FULL_CONTENT_UPDATE
        elif not metadata_changed:
            # 元数据未变，内容变更
            return ReplaceSemantics.FULL_CONTENT_UPDATE
        else:
            # 其他情况（理论上不会到这）
            return ReplaceSemantics.FULL_CONTENT_UPDATE

    # 场景3：元数据未变 + 内容未变（或无法比对）
    if not metadata_changed:
        if previous_checksum and current_checksum == previous_checksum:
            return ReplaceSemantics.MINOR_CORRECTION
        elif not previous_checksum:
            # 没有前序列数据，无法判断
            return ReplaceSemantics.UNKNOWN

    # 其他情况
    return ReplaceSemantics.UNKNOWN


def validate_replace_operation(
    current_checksum: str,
    previous_checksum: Optional[str],
    metadata_changed: bool,
    section_requires_lifecycle_tracking: bool,
    replace_semantics: ReplaceSemantics
) -> tuple[bool, Optional[str]]:
    """
    验证Replace操作是否符合eCTD规范

    Args:
        current_checksum: 当前序列的leaf文件校验和
        previous_checksum: 前序列的leaf文件校验和
        metadata_changed: 元数据是否变更
        section_requires_lifecycle_tracking: section是否需要生命周期追踪
        replace_semantics: Replace操作的语义类型

    Returns:
        (is_compliant, violation_reason) 元组
        - is_compliant: True如果符合规范
        - violation_reason: 如果不符合，返回违规原因字符串

    Validation Rules:
        1. 元数据变更 + 需要lifecycle tracking → 内容必须变更（checksum不同）
        2. METADATA_CONTENT_COUPLING语义 → 内容必须变更
        3. 元数据变更但内容未变 → 违规（"Metadata changed but content not updated"）

    Examples:
        >>> # 符合规范：元数据变更，内容也变更
        >>> is_compliant, reason = validate_replace_operation(
        ...     current_checksum="new_hash",
        ...     previous_checksum="old_hash",
        ...     metadata_changed=True,
        ...     section_requires_lifecycle_tracking=True,
        ...     replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING
        ... )
        >>> is_compliant
        True
        >>> reason is None
        True

        >>> # 违规：元数据变更，但内容未变
        >>> is_compliant, reason = validate_replace_operation(
        ...     current_checksum="same_hash",
        ...     previous_checksum="same_hash",
        ...     metadata_changed=True,
        ...     section_requires_lifecycle_tracking=True,
        ...     replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING
        ... )
        >>> is_compliant
        False
        >>> "Metadata changed but content not updated" in reason
        True
    """
    # 规则1：元数据变更 + 需要lifecycle tracking → 内容必须变更
    if metadata_changed and section_requires_lifecycle_tracking:
        if previous_checksum and current_checksum == previous_checksum:
            return False, (
                "Metadata changed but content not updated. "
                "eCTD 3.6 requires full content update when metadata changes."
            )

    # 规则2：METADATA_CONTENT_COUPLING语义 → 内容必须变更
    if replace_semantics == ReplaceSemantics.METADATA_CONTENT_COUPLING:
        if previous_checksum and current_checksum == previous_checksum:
            return False, (
                "Replace operation marked as metadata-content coupling, "
                "but content (checksum) is identical to previous sequence."
            )

    # 规则3：内容未变但操作标记为REPLACE（可能合规，但需审查）
    if previous_checksum and current_checksum == previous_checksum:
        if replace_semantics == ReplaceSemantics.MINOR_CORRECTION:
            # 允许轻微修正（如格式调整）
            return True, None
        elif replace_semantics == ReplaceSemantics.UNKNOWN:
            # 未知语义，建议人工审核，但不标记为违规
            logger.warning(
                f"Replace operation with identical checksum and unknown semantics. "
                f"Manual review recommended."
            )
            return True, None

    # 其他情况默认符合规范
    return True, None


# ============================================================================
# Leaf操作分析主函数
# ============================================================================

def analyze_leaf_operations(
    current_leaf_ops: List[LeafOperation],
    previous_leaf_ops: Optional[List[LeafOperation]],
    metadata_changes: Dict[str, Dict[str, tuple]],
    section_lifecycle_tracking: Dict[str, bool]
) -> List[LeafOperationAnalysis]:
    """
    分析Leaf操作列表，生成详细分析结果

    Args:
        current_leaf_ops: 当前序列的Leaf操作列表
        previous_leaf_ops: 前序列的Leaf操作列表（如果存在）
        metadata_changes: Section元数据变更详情
            格式：{section_path: {attr_name: (old_value, new_value)}}
        section_lifecycle_tracking: Section是否需要生命周期追踪
            格式：{section_path: True/False}

    Returns:
        Leaf操作分析结果列表

    Examples:
        >>> current_ops = [
        ...     LeafOperation(
        ...         leaf_id="l-001",
        ...         operation_type=LeafOperationType.REPLACE,
        ...         checksum="new_hash",
        ...         file_path="m2/file.pdf",
        ...         section_path="m2-3-s[substance='API-A']"
        ...     )
        ... ]
        >>> previous_ops = [
        ...     LeafOperation(
        ...         leaf_id="l-001",
        ...         operation_type=LeafOperationType.NEW,
        ...         checksum="old_hash",
        ...         file_path="m2/file.pdf",
        ...         section_path="m2-3-s[substance='API-A']"
        ...     )
        ... ]
        >>> metadata_changes = {
        ...     "m2-3-s[substance='API-A']": {"manufacturer": ("MFR-X", "MFR-Y")}
        ... }
        >>> section_lifecycle = {
        ...     "m2-3-s[substance='API-A']": True
        ... }
        >>> analyses = analyze_leaf_operations(
        ...     current_ops, previous_ops, metadata_changes, section_lifecycle
        ... )
        >>> len(analyses)
        1
        >>> analyses[0].metadata_changed
        True
        >>> analyses[0].is_compliant
        True
    """
    # 构建前序列leaf索引（按leaf_id）
    previous_leaf_map: Dict[str, LeafOperation] = {}
    if previous_leaf_ops:
        for leaf_op in previous_leaf_ops:
            previous_leaf_map[leaf_op.leaf_id] = leaf_op

    analyses: List[LeafOperationAnalysis] = []

    for current_op in current_leaf_ops:
        # 获取前序列对应的leaf
        previous_op = previous_leaf_map.get(current_op.leaf_id)
        previous_checksum = previous_op.checksum if previous_op else None

        # 获取section的元数据变更信息
        section_path = current_op.section_path
        section_metadata_changes = metadata_changes.get(section_path, {})
        metadata_changed = len(section_metadata_changes) > 0

        # 获取section是否需要生命周期追踪
        requires_lifecycle = section_lifecycle_tracking.get(section_path, False)

        # 如果是REPLACE操作，进行详细分析
        if current_op.operation_type == LeafOperationType.REPLACE:
            # 判断Replace语义
            replace_semantics = determine_replace_semantics(
                current_checksum=current_op.checksum,
                previous_checksum=previous_checksum,
                metadata_changed=metadata_changed,
                section_requires_lifecycle_tracking=requires_lifecycle
            )

            # 验证是否符合规范
            is_compliant, violation_reason = validate_replace_operation(
                current_checksum=current_op.checksum,
                previous_checksum=previous_checksum,
                metadata_changed=metadata_changed,
                section_requires_lifecycle_tracking=requires_lifecycle,
                replace_semantics=replace_semantics
            )

            # 创建分析结果
            analysis = LeafOperationAnalysis(
                leaf_operation=current_op,
                replace_semantics=replace_semantics,
                previous_leaf_id=previous_op.leaf_id if previous_op else None,
                previous_checksum=previous_checksum,
                metadata_changed=metadata_changed,
                metadata_changes=section_metadata_changes,
                requires_full_update=(
                    metadata_changed and requires_lifecycle
                ),
                is_compliant=is_compliant,
                violation_reason=violation_reason
            )

        else:
            # NEW/DELETE/APPEND操作，默认符合规范
            analysis = LeafOperationAnalysis(
                leaf_operation=current_op,
                replace_semantics=None,
                previous_leaf_id=previous_op.leaf_id if previous_op else None,
                previous_checksum=previous_checksum,
                metadata_changed=metadata_changed,
                metadata_changes=section_metadata_changes,
                requires_full_update=False,
                is_compliant=True,
                violation_reason=None
            )

        analyses.append(analysis)

    return analyses


# ============================================================================
# 批量操作分析工具函数
# ============================================================================

def filter_violations(
    analyses: List[LeafOperationAnalysis]
) -> List[LeafOperationAnalysis]:
    """
    筛选出违规的操作

    Args:
        analyses: 分析结果列表

    Returns:
        违规的分析结果列表

    Examples:
        >>> all_analyses = [...]  # 包含符合和违规的分析
        >>> violations = filter_violations(all_analyses)
        >>> all(not analysis.is_compliant for analysis in violations)
        True
    """
    return [analysis for analysis in analyses if not analysis.is_compliant]


def group_by_section(
    analyses: List[LeafOperationAnalysis]
) -> Dict[str, List[LeafOperationAnalysis]]:
    """
    按section路径分组

    Args:
        analyses: 分析结果列表

    Returns:
        按section_path分组的字典

    Examples:
        >>> analyses = [...]
        >>> grouped = group_by_section(analyses)
        >>> for section_path, section_analyses in grouped.items():
        ...     print(f"{section_path}: {len(section_analyses)} operations")
    """
    grouped: Dict[str, List[LeafOperationAnalysis]] = {}
    for analysis in analyses:
        section_path = analysis.leaf_operation.section_path
        if section_path not in grouped:
            grouped[section_path] = []
        grouped[section_path].append(analysis)
    return grouped


def get_operation_statistics(
    analyses: List[LeafOperationAnalysis]
) -> Dict[str, int]:
    """
    获取操作统计信息

    Args:
        analyses: 分析结果列表

    Returns:
        统计字典，包含各类操作的数量

    Examples:
        >>> analyses = [...]
        >>> stats = get_operation_statistics(analyses)
        >>> stats
        {
            'total': 10,
            'new': 3,
            'delete': 1,
            'replace': 6,
            'append': 0,
            'violations': 2,
            'metadata_content_coupling': 4,
            'full_content_update': 2
        }
    """
    stats = {
        "total": len(analyses),
        "new": 0,
        "delete": 0,
        "replace": 0,
        "append": 0,
        "violations": 0,
        "metadata_content_coupling": 0,
        "full_content_update": 0,
        "minor_correction": 0,
        "unknown": 0
    }

    for analysis in analyses:
        # 统计操作类型
        op_type = analysis.leaf_operation.operation_type.value
        stats[op_type] += 1

        # 统计违规
        if not analysis.is_compliant:
            stats["violations"] += 1

        # 统计Replace语义
        if analysis.replace_semantics:
            semantics_key = analysis.replace_semantics.value
            stats[semantics_key] += 1

    return stats
