"""
eCTD元数据比对器

跨序列比对元数据变更，检测Section属性变化和Leaf操作。

Stage 1 Task 1.2 - 元数据比对核心模块
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple, Literal
from enum import Enum

from core.ectd_metadata_extractor import (
    SequenceMetadataIndex,
    SectionMetadataSnapshot,
    LeafMetadata
)
from core.ectd_section_identifier import SectionIdentifier
from core.ectd_leaf_operation_analyzer import (
    analyze_leaf_operations,
    LeafOperationAnalysis,
    ReplaceSemantics
)


@dataclass
class MetadataChange:
    """元数据变更记录"""
    attribute_name: str
    old_value: Optional[str]
    new_value: Optional[str]
    change_type: Literal["added", "removed", "modified", "unchanged"]

    def __str__(self) -> str:
        if self.change_type == "added":
            return f"{self.attribute_name}: (新增) → '{self.new_value}'"
        elif self.change_type == "removed":
            return f"{self.attribute_name}: '{self.old_value}' → (删除)"
        elif self.change_type == "modified":
            return f"{self.attribute_name}: '{self.old_value}' → '{self.new_value}'"
        else:
            return f"{self.attribute_name}: '{self.old_value}' (未变更)"


@dataclass
class SectionUpdateAnalysis:
    """Section更新分析结果"""
    section_identifier: SectionIdentifier
    sequence_number: str
    previous_sequence_number: Optional[str]

    # 元数据变更
    metadata_changes: List[MetadataChange]
    has_metadata_change: bool

    # Leaf操作分析
    leaf_analysis: LeafOperationAnalysis

    # Section生命周期
    is_new_section: bool = False      # 新增section（前序列没有）
    is_deleted_section: bool = False  # 删除section（当前序列没有）
    is_updated_section: bool = False  # 更新section（前后都有）

    def is_compliant(self) -> bool:
        """
        判定是否符合元数据生命周期耦合规则

        规则:
        1. 新增/删除section不受约束 → 合规
        2. 元数据未变更 → 合规
        3. 元数据变更时:
           - 有new/delete操作 → 合规（结构变更）
           - 全部replace操作 → 必须有实质内容变更，否则违规
        """
        if self.is_new_section or self.is_deleted_section:
            return True

        if not self.has_metadata_change:
            return True

        # 元数据变更时，检查leaf操作分析结果
        return self.leaf_analysis.is_compliant

    def get_violation_summary(self) -> Optional[str]:
        """获取违规摘要"""
        if self.is_compliant():
            return None

        return self.leaf_analysis.violation_reason


class ECTDMetadataComparator:
    """eCTD元数据比对器"""

    def __init__(
        self,
        previous_index: SequenceMetadataIndex,
        current_index: SequenceMetadataIndex
    ):
        """
        初始化比对器

        参数:
            previous_index: 前序列的元数据索引
            current_index: 当前序列的元数据索引
        """
        self.previous_index = previous_index
        self.current_index = current_index

    def detect_metadata_changes(self) -> List[SectionUpdateAnalysis]:
        """
        检测序列间元数据变更

        返回:
            SectionUpdateAnalysis对象列表，每个对象代表一个section的变更分析
        """
        analyses = []

        # 构建匹配键映射
        previous_sections_by_key = {
            section.get_matching_key(): section
            for section in self.previous_index.get_all_sections()
        }

        current_sections_by_key = {
            section.get_matching_key(): section
            for section in self.current_index.get_all_sections()
        }

        # 1. 处理前后都存在的section（配对匹配）
        matched_keys = set(previous_sections_by_key.keys()) & set(current_sections_by_key.keys())
        for key in matched_keys:
            prev_section = previous_sections_by_key[key]
            curr_section = current_sections_by_key[key]

            analysis = self._analyze_section_update(prev_section, curr_section)
            analysis.is_updated_section = True
            analyses.append(analysis)

        # 2. 处理新增section（只在当前序列）
        new_keys = set(current_sections_by_key.keys()) - set(previous_sections_by_key.keys())
        for key in new_keys:
            curr_section = current_sections_by_key[key]
            analysis = self._create_new_section_analysis(curr_section)
            analyses.append(analysis)

        # 3. 处理删除section（只在前序列）
        deleted_keys = set(previous_sections_by_key.keys()) - set(current_sections_by_key.keys())
        for key in deleted_keys:
            prev_section = previous_sections_by_key[key]
            analysis = self._create_deleted_section_analysis(prev_section)
            analyses.append(analysis)

        return analyses

    def _analyze_section_update(
        self,
        prev_section: SectionMetadataSnapshot,
        curr_section: SectionMetadataSnapshot
    ) -> SectionUpdateAnalysis:
        """
        分析配对section的更新

        参数:
            prev_section: 前序列的section快照
            curr_section: 当前序列的section快照

        返回:
            SectionUpdateAnalysis对象
        """
        # 对比属性，识别变更
        metadata_changes = self._compare_attributes(
            prev_section.identifier.attributes,
            curr_section.identifier.attributes
        )

        has_metadata_change = any(
            change.change_type in ("added", "removed", "modified")
            for change in metadata_changes
        )

        # 转换LeafMetadata到LeafOperation对象
        from core.ectd_leaf_operation_analyzer import (
            LeafOperation,
            LeafOperationType,
            analyze_leaf_operations
        )

        current_leaf_ops = [
            LeafOperation(
                leaf_id=leaf.leaf_id,
                operation_type=LeafOperationType(leaf.operation),
                checksum=leaf.checksum or "",
                file_path=leaf.xlink_href or "",
                section_path=curr_section.section_path,
                title=leaf.title
            )
            for leaf in curr_section.leaf_metadata
        ]

        prev_leaf_ops = [
            LeafOperation(
                leaf_id=leaf.leaf_id,
                operation_type=LeafOperationType(leaf.operation),
                checksum=leaf.checksum or "",
                file_path=leaf.xlink_href or "",
                section_path=prev_section.section_path,
                title=leaf.title
            )
            for leaf in prev_section.leaf_metadata
        ]

        # 构建metadata_changes字典格式
        metadata_changes_dict = {
            curr_section.section_path: {
                change.attribute_name: (change.old_value, change.new_value)
                for change in metadata_changes
                if change.change_type in ("added", "removed", "modified")
            }
        }

        # 构建section_lifecycle_tracking字典
        section_lifecycle_tracking = {
            curr_section.section_path: True  # 假设所有tracked section都需要lifecycle tracking
        }

        # 分析leaf操作
        leaf_analyses = analyze_leaf_operations(
            current_leaf_ops=current_leaf_ops,
            previous_leaf_ops=prev_leaf_ops,
            metadata_changes=metadata_changes_dict,
            section_lifecycle_tracking=section_lifecycle_tracking
        )

        # Stage 3 增强：检查部分更新违规
        # 当元数据变更时，验证所有前序列的leaf是否都被处理了
        leaf_analysis = self._validate_leaf_completeness(
            prev_section=prev_section,
            curr_section=curr_section,
            has_metadata_change=has_metadata_change,
            leaf_analyses=leaf_analyses,
            prev_leaf_ops=prev_leaf_ops,
            current_leaf_ops=current_leaf_ops
        )

        return SectionUpdateAnalysis(
            section_identifier=curr_section.identifier,
            sequence_number=curr_section.sequence_number,
            previous_sequence_number=prev_section.sequence_number,
            metadata_changes=metadata_changes,
            has_metadata_change=has_metadata_change,
            leaf_analysis=leaf_analysis,
            is_updated_section=True
        )

    def _validate_leaf_completeness(
        self,
        prev_section: SectionMetadataSnapshot,
        curr_section: SectionMetadataSnapshot,
        has_metadata_change: bool,
        leaf_analyses: List,
        prev_leaf_ops: List,
        current_leaf_ops: List
    ):
        """
        验证leaf更新的完整性（Stage 3 增强）

        当元数据变更时，检查前序列的所有leaf是否都被正确处理：
        - 必须被删除（delete操作）
        - 或被替换（replace操作）

        如果有leaf既没有被删除也没有被替换，则为"部分更新"违规

        Args:
            prev_section: 前序列section
            curr_section: 当前序列section
            has_metadata_change: 是否有元数据变更
            leaf_analyses: leaf操作分析结果
            prev_leaf_ops: 前序列leaf操作列表
            current_leaf_ops: 当前序列leaf操作列表

        Returns:
            LeafOperationAnalysis: 汇总的分析结果
        """
        from core.ectd_leaf_operation_analyzer import (
            LeafOperationAnalysis,
            LeafOperation,
            LeafOperationType
        )

        # 如果没有元数据变更，使用原有逻辑
        if not has_metadata_change:
            if leaf_analyses:
                return leaf_analyses[0]
            else:
                placeholder_leaf = LeafOperation(
                    leaf_id="placeholder",
                    operation_type=LeafOperationType.NEW,
                    checksum="",
                    file_path="",
                    section_path=curr_section.section_path
                )
                return LeafOperationAnalysis(
                    leaf_operation=placeholder_leaf,
                    is_compliant=True
                )

        # 元数据变更时，检查完整性
        if not current_leaf_ops:
            # 元数据变更但没有任何leaf操作 - 违规
            placeholder_leaf = LeafOperation(
                leaf_id="placeholder",
                operation_type=LeafOperationType.NEW,
                checksum="",
                file_path="",
                section_path=curr_section.section_path
            )
            return LeafOperationAnalysis(
                leaf_operation=placeholder_leaf,
                is_compliant=False,
                violation_reason="Metadata changed but no leaf operations present"
            )

        # 构建当前序列的操作映射
        current_ops_by_id = {op.leaf_id: op for op in current_leaf_ops}

        # 构建modified_file映射：哪些前序列leaf被replace/delete/append操作引用了
        # 需要从curr_section的LeafMetadata中获取modified_file信息
        modified_file_targets = set()
        for leaf_meta in curr_section.leaf_metadata:
            if leaf_meta.modified_file:
                modified_file_targets.add(leaf_meta.modified_file)

        # 检查前序列的每个leaf是否都被处理了
        unprocessed_leafs = []
        for prev_leaf in prev_leaf_ops:
            # 情况1：前序列leaf通过modified_file被引用（replace/delete/append）
            if prev_leaf.leaf_id in modified_file_targets:
                continue

            # 情况2：前序列leaf ID在当前序列中有对应操作
            curr_op = current_ops_by_id.get(prev_leaf.leaf_id)
            if curr_op:
                # 检查操作类型
                if curr_op.operation_type in (LeafOperationType.REPLACE, LeafOperationType.DELETE):
                    # 正确处理：replace或delete
                    continue
                else:
                    # 其他操作类型（理论上不应该出现在这里）
                    continue

            # 情况3：检查leaf分析结果中是否有delete操作引用了这个leaf
            has_delete = any(
                op.leaf_operation.operation_type == LeafOperationType.DELETE
                for op in leaf_analyses
                if hasattr(op, 'leaf_operation') and op.leaf_operation.leaf_id == prev_leaf.leaf_id
            )

            if not has_delete:
                unprocessed_leafs.append(prev_leaf.leaf_id)

        # 如果有未处理的leaf，则为部分更新违规
        if unprocessed_leafs:
            # 创建违规分析结果
            representative_leaf = current_leaf_ops[0] if current_leaf_ops else LeafOperation(
                leaf_id="placeholder",
                operation_type=LeafOperationType.NEW,
                checksum="",
                file_path="",
                section_path=curr_section.section_path
            )

            violation_msg = (
                f"Partial update violation: Metadata changed but {len(unprocessed_leafs)} "
                f"leaf(s) from previous sequence were not deleted or replaced. "
                f"Unprocessed leafs: {', '.join(unprocessed_leafs[:5])}"
            )
            if len(unprocessed_leafs) > 5:
                violation_msg += f" (and {len(unprocessed_leafs) - 5} more)"

            return LeafOperationAnalysis(
                leaf_operation=representative_leaf,
                is_compliant=False,
                violation_reason=violation_msg
            )

        # 没有部分更新问题，返回原有分析结果
        if leaf_analyses:
            # 汇总所有leaf的合规性
            all_compliant = all(analysis.is_compliant for analysis in leaf_analyses)
            if not all_compliant:
                # 返回第一个不合规的分析
                for analysis in leaf_analyses:
                    if not analysis.is_compliant:
                        return analysis
            return leaf_analyses[0]
        else:
            # 理论上不应该到这里，但提供默认返回
            placeholder_leaf = LeafOperation(
                leaf_id="placeholder",
                operation_type=LeafOperationType.NEW,
                checksum="",
                file_path="",
                section_path=curr_section.section_path
            )
            return LeafOperationAnalysis(
                leaf_operation=placeholder_leaf,
                is_compliant=True
            )

    def _create_new_section_analysis(
        self,
        curr_section: SectionMetadataSnapshot
    ) -> SectionUpdateAnalysis:
        """
        创建新增section的分析结果

        参数:
            curr_section: 当前序列的section快照

        返回:
            SectionUpdateAnalysis对象（标记为新增）
        """
        # 新增section，所有属性都是"新增"
        metadata_changes = [
            MetadataChange(
                attribute_name=attr_name,
                old_value=None,
                new_value=attr_value,
                change_type="added"
            )
            for attr_name, attr_value in curr_section.identifier.attributes.items()
        ]

        # 创建空的leaf分析（新增section不需要详细分析）
        from core.ectd_leaf_operation_analyzer import LeafOperationAnalysis, LeafOperation, LeafOperationType

        # 为新增section创建一个占位符leaf operation
        placeholder_leaf = LeafOperation(
            leaf_id="placeholder",
            operation_type=LeafOperationType.NEW,
            checksum="",
            file_path="",
            section_path=curr_section.section_path
        )

        leaf_analysis = LeafOperationAnalysis(
            leaf_operation=placeholder_leaf,
            replace_semantics=None,
            is_compliant=True,
            violation_reason=None
        )

        return SectionUpdateAnalysis(
            section_identifier=curr_section.identifier,
            sequence_number=curr_section.sequence_number,
            previous_sequence_number=None,
            metadata_changes=metadata_changes,
            has_metadata_change=True,
            leaf_analysis=leaf_analysis,
            is_new_section=True
        )

    def _create_deleted_section_analysis(
        self,
        prev_section: SectionMetadataSnapshot
    ) -> SectionUpdateAnalysis:
        """
        创建删除section的分析结果

        参数:
            prev_section: 前序列的section快照

        返回:
            SectionUpdateAnalysis对象（标记为删除）
        """
        # 删除section，所有属性都是"删除"
        metadata_changes = [
            MetadataChange(
                attribute_name=attr_name,
                old_value=attr_value,
                new_value=None,
                change_type="removed"
            )
            for attr_name, attr_value in prev_section.identifier.attributes.items()
        ]

        # 创建空的leaf分析
        from core.ectd_leaf_operation_analyzer import LeafOperationAnalysis, LeafOperation, LeafOperationType

        # 为删除section创建一个占位符leaf operation
        placeholder_leaf = LeafOperation(
            leaf_id="placeholder",
            operation_type=LeafOperationType.DELETE,
            checksum="",
            file_path="",
            section_path=prev_section.section_path
        )

        leaf_analysis = LeafOperationAnalysis(
            leaf_operation=placeholder_leaf,
            replace_semantics=None,
            is_compliant=True,
            violation_reason=None
        )

        return SectionUpdateAnalysis(
            section_identifier=prev_section.identifier,
            sequence_number=self.current_index.sequence_number,
            previous_sequence_number=prev_section.sequence_number,
            metadata_changes=metadata_changes,
            has_metadata_change=True,
            leaf_analysis=leaf_analysis,
            is_deleted_section=True
        )

    def _compare_attributes(
        self,
        old_attrs: Dict[str, str],
        new_attrs: Dict[str, str]
    ) -> List[MetadataChange]:
        """
        比对属性变更

        参数:
            old_attrs: 前序列属性字典
            new_attrs: 当前序列属性字典

        返回:
            MetadataChange对象列表
        """
        changes = []
        all_attrs = set(old_attrs.keys()) | set(new_attrs.keys())

        for attr in sorted(all_attrs):
            old_val = old_attrs.get(attr)
            new_val = new_attrs.get(attr)

            if old_val is None and new_val is not None:
                change_type = "added"
            elif old_val is not None and new_val is None:
                change_type = "removed"
            elif old_val != new_val:
                change_type = "modified"
            else:
                change_type = "unchanged"

            changes.append(MetadataChange(
                attribute_name=attr,
                old_value=old_val,
                new_value=new_val,
                change_type=change_type
            ))

        return changes

    def _leaf_to_dict(self, leaf: LeafMetadata) -> Dict[str, any]:
        """将LeafMetadata转换为字典格式（供LeafOperationAnalyzer使用）"""
        return {
            "ID": leaf.leaf_id,
            "operation": leaf.operation,
            "title": leaf.title,
            "xlink:href": leaf.xlink_href,
            "modified-file": leaf.modified_file,
            "checksum": leaf.checksum,
            "checksum-type": leaf.checksum_type
        }

    def get_violations(self) -> List[SectionUpdateAnalysis]:
        """
        获取所有违规的section更新

        返回:
            违规的SectionUpdateAnalysis对象列表
        """
        all_analyses = self.detect_metadata_changes()
        return [analysis for analysis in all_analyses if not analysis.is_compliant()]

    def get_compliant_updates(self) -> List[SectionUpdateAnalysis]:
        """
        获取所有合规的section更新

        返回:
            合规的SectionUpdateAnalysis对象列表
        """
        all_analyses = self.detect_metadata_changes()
        return [analysis for analysis in all_analyses if analysis.is_compliant()]

    def generate_summary_report(self) -> Dict[str, any]:
        """
        生成摘要报告

        返回:
            包含统计信息的字典
        """
        all_analyses = self.detect_metadata_changes()
        violations = self.get_violations()

        return {
            "previous_sequence": self.previous_index.sequence_number,
            "current_sequence": self.current_index.sequence_number,
            "total_sections_analyzed": len(all_analyses),
            "new_sections": sum(1 for a in all_analyses if a.is_new_section),
            "deleted_sections": sum(1 for a in all_analyses if a.is_deleted_section),
            "updated_sections": sum(1 for a in all_analyses if a.is_updated_section),
            "sections_with_metadata_changes": sum(1 for a in all_analyses if a.has_metadata_change),
            "violations_found": len(violations),
            "compliant_updates": len(all_analyses) - len(violations),
            "compliance_rate": (len(all_analyses) - len(violations)) / len(all_analyses) * 100
                               if len(all_analyses) > 0 else 100.0
        }


def compare_sequence_metadata(
    previous_index: SequenceMetadataIndex,
    current_index: SequenceMetadataIndex
) -> List[SectionUpdateAnalysis]:
    """
    便捷函数：比对两个序列的元数据

    参数:
        previous_index: 前序列的元数据索引
        current_index: 当前序列的元数据索引

    返回:
        SectionUpdateAnalysis对象列表

    示例:
        prev_index = extract_sequence_metadata_index("/path/to/0004")
        curr_index = extract_sequence_metadata_index("/path/to/0005")
        analyses = compare_sequence_metadata(prev_index, curr_index)

        for analysis in analyses:
            if not analysis.is_compliant():
                print(f"违规: {analysis.section_identifier.to_display_path()}")
                print(f"原因: {analysis.get_violation_summary()}")
    """
    comparator = ECTDMetadataComparator(previous_index, current_index)
    return comparator.detect_metadata_changes()
