"""
eCTD元数据生命周期耦合规则验证器

Stage 2 Task 2 - 违规检测逻辑

负责:
1. 跨序列元数据生命周期耦合规则验证
2. 生成详细的违规报告
3. 与material_assessment.py集成

版本: v1.0
创建日期: 2026-09-14
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from pathlib import Path

from core.ectd_metadata_extractor import (
    SequenceMetadataIndex,
    extract_sequence_metadata_index
)
from core.ectd_metadata_comparator import (
    ECTDMetadataComparator,
    SectionUpdateAnalysis,
    MetadataChange
)
from core.ectd_sequence_resolver import (
    resolve_previous_sequence,
    SequenceResolverError
)


# ============================================================================
# 违规详情数据结构
# ============================================================================

@dataclass
class ViolationDetail:
    """
    元数据生命周期耦合违规详情

    Attributes:
        section_identifier: Section标识符（element_name + attributes）
        section_path: Section在XML中的路径（显示用）
        sequence_number: 违规发生的序列号
        previous_sequence_number: 前序列号

        metadata_changes: 元数据变更列表
        changed_attributes: 变更的属性名列表（快速访问）

        leaf_operations: 相关的leaf操作列表
        leaf_ids: 涉及的leaf ID列表

        violation_type: 违规类型
        violation_message: 违规描述
        severity: 严重程度（error/warning）

        rule_id: 对应的规则ID（如HR-ECTD-XXX）
        rule_citation: 规则引用（eCTD规范章节）

    Examples:
        >>> violation = ViolationDetail(
        ...     section_identifier="m2-3-s-drug-substance?substance=API-A",
        ...     section_path="m2/m2-3-body-data/m2-3-s-drug-substance[@substance='API-A']",
        ...     sequence_number="0005",
        ...     previous_sequence_number="0004",
        ...     metadata_changes=[
        ...         MetadataChange(
        ...             attribute_name="manufacturer",
        ...             old_value="MFR-X",
        ...             new_value="MFR-Y",
        ...             change_type="modified"
        ...         )
        ...     ],
        ...     changed_attributes=["manufacturer"],
        ...     leaf_ids=["l-m2-3-s-001"],
        ...     violation_type="metadata_changed_content_not_updated",
        ...     violation_message="Metadata 'manufacturer' changed from 'MFR-X' to 'MFR-Y', "
        ...                       "but leaf content (checksum) remains identical. "
        ...                       "eCTD 3.6 requires full content update when metadata changes.",
        ...     severity="error",
        ...     rule_id="HR-ECTD-200",
        ...     rule_citation="cn_ectd_technical_specification#sec_3_6"
        ... )
    """
    section_identifier: str
    section_path: str
    sequence_number: str
    previous_sequence_number: str

    metadata_changes: List[MetadataChange]
    changed_attributes: List[str]

    leaf_operations: List[Dict]  # {leaf_id, operation, checksum, file_path}
    leaf_ids: List[str]

    violation_type: str
    violation_message: str
    severity: str  # "error" or "warning"

    rule_id: str
    rule_citation: str

    def to_dict(self) -> Dict:
        """转换为字典格式（用于JSON序列化）"""
        return {
            "section_identifier": self.section_identifier,
            "section_path": self.section_path,
            "sequence_number": self.sequence_number,
            "previous_sequence_number": self.previous_sequence_number,
            "metadata_changes": [
                {
                    "attribute": change.attribute_name,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "change_type": change.change_type
                }
                for change in self.metadata_changes
            ],
            "changed_attributes": self.changed_attributes,
            "leaf_operations": self.leaf_operations,
            "leaf_ids": self.leaf_ids,
            "violation_type": self.violation_type,
            "violation_message": self.violation_message,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "rule_citation": self.rule_citation
        }

    def get_summary(self) -> str:
        """获取简短摘要"""
        attrs = ", ".join(self.changed_attributes)
        return f"[{self.rule_id}] Section {self.section_identifier}: {attrs} changed, {self.violation_type}"


@dataclass
class ValidationResult:
    """
    验证结果汇总

    Attributes:
        sequence_number: 当前序列号
        previous_sequence_number: 前序列号
        total_sections_analyzed: 分析的section总数
        compliant_sections: 符合规范的section数量
        violation_count: 违规数量
        violations: 违规详情列表
        warning_count: 警告数量
        warnings: 警告详情列表
    """
    sequence_number: str
    previous_sequence_number: str
    total_sections_analyzed: int
    compliant_sections: int
    violation_count: int
    violations: List[ViolationDetail]
    warning_count: int = 0
    warnings: List[ViolationDetail] = field(default_factory=list)

    def is_fully_compliant(self) -> bool:
        """是否完全符合规范（无违规）"""
        return self.violation_count == 0

    def get_summary(self) -> str:
        """获取验证摘要"""
        return (
            f"Validation Summary (Sequence {self.previous_sequence_number} → {self.sequence_number}):\n"
            f"  Total sections analyzed: {self.total_sections_analyzed}\n"
            f"  Compliant: {self.compliant_sections}\n"
            f"  Violations: {self.violation_count}\n"
            f"  Warnings: {self.warning_count}"
        )


# ============================================================================
# 违规检测逻辑
# ============================================================================

def validate_metadata_lifecycle_coupling(
    previous_index: SequenceMetadataIndex,
    current_index: SequenceMetadataIndex
) -> ValidationResult:
    """
    验证元数据生命周期耦合规则

    核心规则（eCTD 3.6章节）:
    当Section的元数据属性变更时，其下的Leaf文件内容必须完整更新（checksum变化）

    Args:
        previous_index: 前序列的元数据索引
        current_index: 当前序列的元数据索引

    Returns:
        ValidationResult: 验证结果汇总

    Examples:
        >>> # 准备测试数据
        >>> previous_index = extract_sequence_metadata_index("/path/to/0004")
        >>> current_index = extract_sequence_metadata_index("/path/to/0005")
        >>>
        >>> # 运行验证
        >>> result = validate_metadata_lifecycle_coupling(previous_index, current_index)
        >>>
        >>> # 检查结果
        >>> if result.is_fully_compliant():
        ...     print("All sections compliant")
        ... else:
        ...     for violation in result.violations:
        ...         print(violation.get_summary())
    """
    # 创建比对器
    comparator = ECTDMetadataComparator(previous_index, current_index)

    # 执行比对，获取所有Section的更新分析
    section_analyses: List[SectionUpdateAnalysis] = comparator.detect_metadata_changes()

    # 统计和收集违规
    violations: List[ViolationDetail] = []
    warnings: List[ViolationDetail] = []
    compliant_count = 0

    for analysis in section_analyses:
        if analysis.is_compliant():
            compliant_count += 1
        else:
            # 生成违规详情
            violation = _create_violation_detail(analysis, current_index, previous_index)

            if violation.severity == "error":
                violations.append(violation)
            else:
                warnings.append(violation)

    # 构建验证结果
    result = ValidationResult(
        sequence_number=current_index.sequence_number,
        previous_sequence_number=previous_index.sequence_number,
        total_sections_analyzed=len(section_analyses),
        compliant_sections=compliant_count,
        violation_count=len(violations),
        violations=violations,
        warning_count=len(warnings),
        warnings=warnings
    )

    return result


def _create_violation_detail(
    analysis: SectionUpdateAnalysis,
    current_index: SequenceMetadataIndex,
    previous_index: SequenceMetadataIndex
) -> ViolationDetail:
    """
    根据SectionUpdateAnalysis创建ViolationDetail

    Args:
        analysis: Section更新分析结果
        current_index: 当前序列元数据索引
        previous_index: 前序列元数据索引

    Returns:
        ViolationDetail: 违规详情
    """
    # 提取元数据变更信息
    changed_attributes = [
        change.attribute_name
        for change in analysis.metadata_changes
        if change.change_type in ("added", "removed", "modified")
    ]

    # 提取leaf操作信息
    leaf_operations = []
    leaf_ids = []

    if hasattr(analysis.leaf_analysis, 'leaf_operation'):
        leaf_op = analysis.leaf_analysis.leaf_operation
        leaf_ids.append(leaf_op.leaf_id)
        leaf_operations.append({
            "leaf_id": leaf_op.leaf_id,
            "operation": leaf_op.operation_type.value,
            "checksum": leaf_op.checksum,
            "file_path": leaf_op.file_path
        })

    # 构建违规消息
    violation_message = _build_violation_message(analysis)

    # 确定违规类型
    violation_type = "metadata_changed_content_not_updated"
    if analysis.leaf_analysis.replace_semantics:
        if "identical" in analysis.leaf_analysis.violation_reason or "":
            violation_type = "metadata_changed_content_not_updated"

    # 创建ViolationDetail
    violation = ViolationDetail(
        section_identifier=analysis.section_identifier.get_matching_key(),
        section_path=analysis.section_identifier.to_display_path(),
        sequence_number=analysis.sequence_number,
        previous_sequence_number=analysis.previous_sequence_number or "N/A",
        metadata_changes=analysis.metadata_changes,
        changed_attributes=changed_attributes,
        leaf_operations=leaf_operations,
        leaf_ids=leaf_ids,
        violation_type=violation_type,
        violation_message=violation_message,
        severity="error",  # 元数据生命周期耦合违规为硬性错误
        rule_id="HR-ECTD-200",  # 待在material_assessment.py中正式定义
        rule_citation="cn_ectd_technical_specification#sec_3_6"
    )

    return violation


def _build_violation_message(analysis: SectionUpdateAnalysis) -> str:
    """
    构建详细的违规消息

    Args:
        analysis: Section更新分析结果

    Returns:
        str: 违规消息
    """
    # 提取变更的属性
    changes_desc = []
    for change in analysis.metadata_changes:
        if change.change_type in ("added", "removed", "modified"):
            changes_desc.append(str(change))

    changes_text = "; ".join(changes_desc)

    # 基础违规消息
    message = (
        f"Section metadata changed ({changes_text}), "
        f"but leaf content (checksum) remains identical to previous sequence. "
        f"eCTD 3.6 requires full content update when metadata changes."
    )

    # 如果有具体的违规原因，追加
    if analysis.leaf_analysis.violation_reason:
        message += f" Details: {analysis.leaf_analysis.violation_reason}"

    return message


# ============================================================================
# 便捷函数：从序列目录进行验证
# ============================================================================

def validate_sequences_from_paths(
    previous_sequence_path: str,
    current_sequence_path: str
) -> ValidationResult:
    """
    从序列目录路径进行验证（便捷函数）

    Args:
        previous_sequence_path: 前序列目录路径
        current_sequence_path: 当前序列目录路径
        resolution_strategy: 序列解析策略

    Returns:
        ValidationResult: 验证结果

    Examples:
        >>> result = validate_sequences_from_paths(
        ...     previous_sequence_path="/path/to/application/0004",
        ...     current_sequence_path="/path/to/application/0005"
        ... )
        >>> print(result.get_summary())
    """
    # 提取元数据索引
    previous_index = extract_sequence_metadata_index(previous_sequence_path)
    current_index = extract_sequence_metadata_index(current_sequence_path)

    # 执行验证
    result = validate_metadata_lifecycle_coupling(previous_index, current_index)

    return result


# ============================================================================
# 批量验证：应用级别
# ============================================================================

def validate_application_sequences(
    application_root: str,
    start_sequence: Optional[str] = None,
    end_sequence: Optional[str] = None
) -> List[ValidationResult]:
    """
    验证应用下的多个序列（批量验证）

    Args:
        application_root: 应用根目录
        start_sequence: 起始序列号（如"0004"），None表示从第二个序列开始
        end_sequence: 结束序列号（如"0010"），None表示到最后一个序列

    Returns:
        List[ValidationResult]: 每对相邻序列的验证结果列表

    Examples:
        >>> results = validate_application_sequences(
        ...     application_root="/path/to/x2023001234"
        ... )
        >>> for result in results:
        ...     print(result.get_summary())
    """
    # 发现应用下的所有序列目录
    app_path = Path(application_root)
    if not app_path.exists():
        return []

    # 查找所有序列目录（0000, 0001, 0002等）
    sequence_dirs = []
    for item in app_path.iterdir():
        if item.is_dir() and item.name.isdigit() and len(item.name) == 4:
            sequence_dirs.append(item.name)

    sequence_dirs.sort()

    if len(sequence_dirs) < 2:
        return []  # 少于2个序列，无法比对

    # 筛选序列范围
    sequences_to_validate = sequence_dirs
    if start_sequence:
        sequences_to_validate = [
            s for s in sequences_to_validate if int(s) >= int(start_sequence)
        ]
    if end_sequence:
        sequences_to_validate = [
            s for s in sequences_to_validate if int(s) <= int(end_sequence)
        ]

    # 相邻序列两两验证
    validation_results: List[ValidationResult] = []

    for i in range(len(sequences_to_validate) - 1):
        prev_seq = sequences_to_validate[i]
        curr_seq = sequences_to_validate[i + 1]

        prev_path = str(Path(application_root) / prev_seq)
        curr_path = str(Path(application_root) / curr_seq)

        result = validate_sequences_from_paths(prev_path, curr_path)
        validation_results.append(result)

    return validation_results
