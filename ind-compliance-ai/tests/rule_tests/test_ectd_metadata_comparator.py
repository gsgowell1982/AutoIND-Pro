"""
测试eCTD元数据比对模块

测试覆盖：
1. ECTDMetadataComparator类
2. MetadataChange检测
3. SectionUpdateAnalysis生成
4. 跨序列比对
5. 便捷函数compare_sequence_metadata

版本: v2.0 (Stage 1)
创建日期: 2026-09-14
"""

import unittest
import tempfile
from pathlib import Path
from core.ectd_metadata_comparator import (
    ECTDMetadataComparator,
    MetadataChange,
    SectionUpdateAnalysis,
    compare_sequence_metadata
)
from core.ectd_metadata_extractor import (
    SequenceMetadataIndex,
    SectionMetadataSnapshot,
    LeafMetadata
)
from core.ectd_section_identifier import SectionIdentifier


class TestMetadataChange(unittest.TestCase):
    """测试MetadataChange数据结构"""

    def test_create_metadata_change_modified(self):
        """测试创建修改类型的变更"""
        change = MetadataChange(
            attribute_name="manufacturer",
            old_value="MFR-X",
            new_value="MFR-Y",
            change_type="modified"
        )

        self.assertEqual(change.attribute_name, "manufacturer")
        self.assertEqual(change.old_value, "MFR-X")
        self.assertEqual(change.new_value, "MFR-Y")
        self.assertEqual(change.change_type, "modified")

    def test_metadata_change_str_modified(self):
        """测试修改类型的字符串表示"""
        change = MetadataChange(
            attribute_name="manufacturer",
            old_value="MFR-X",
            new_value="MFR-Y",
            change_type="modified"
        )

        str_repr = str(change)
        self.assertIn("manufacturer", str_repr)
        self.assertIn("MFR-X", str_repr)
        self.assertIn("MFR-Y", str_repr)

    def test_metadata_change_str_added(self):
        """测试新增类型的字符串表示"""
        change = MetadataChange(
            attribute_name="indication",
            old_value=None,
            new_value="Treatment A",
            change_type="added"
        )

        str_repr = str(change)
        self.assertIn("新增", str_repr)
        self.assertIn("Treatment A", str_repr)

    def test_metadata_change_str_removed(self):
        """测试删除类型的字符串表示"""
        change = MetadataChange(
            attribute_name="old-attr",
            old_value="old-value",
            new_value=None,
            change_type="removed"
        )

        str_repr = str(change)
        self.assertIn("删除", str_repr)


class TestSectionUpdateAnalysis(unittest.TestCase):
    """测试SectionUpdateAnalysis数据结构"""

    def test_is_compliant_new_section(self):
        """测试新增section（总是合规）"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A"}
        )

        # Mock leaf analysis (新增section，无违规)
        from core.ectd_leaf_operation_analyzer import (
            LeafOperationAnalysis,
            LeafOperation,
            LeafOperationType
        )

        placeholder_leaf = LeafOperation(
            leaf_id="test-leaf",
            operation_type=LeafOperationType.NEW,
            checksum="",
            file_path="",
            section_path="m2/m2-3/s"
        )

        leaf_analysis = LeafOperationAnalysis(
            leaf_operation=placeholder_leaf,
            is_compliant=True
        )

        analysis = SectionUpdateAnalysis(
            section_identifier=identifier,
            sequence_number="0005",
            previous_sequence_number=None,
            metadata_changes=[],
            has_metadata_change=False,
            leaf_analysis=leaf_analysis,
            is_new_section=True,
            is_deleted_section=False,
            is_updated_section=False
        )

        self.assertTrue(analysis.is_compliant())
        self.assertIsNone(analysis.get_violation_summary())

    def test_is_compliant_deleted_section(self):
        """测试删除section（总是合规）"""
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A"}
        )

        from core.ectd_leaf_operation_analyzer import (
            LeafOperationAnalysis,
            LeafOperation,
            LeafOperationType
        )

        placeholder_leaf = LeafOperation(
            leaf_id="test-leaf",
            operation_type=LeafOperationType.DELETE,
            checksum="",
            file_path="",
            section_path="m2/m2-3/s"
        )

        leaf_analysis = LeafOperationAnalysis(
            leaf_operation=placeholder_leaf,
            is_compliant=True
        )

        analysis = SectionUpdateAnalysis(
            section_identifier=identifier,
            sequence_number="0005",
            previous_sequence_number="0004",
            metadata_changes=[],
            has_metadata_change=False,
            leaf_analysis=leaf_analysis,
            is_new_section=False,
            is_deleted_section=True,
            is_updated_section=False
        )

        self.assertTrue(analysis.is_compliant())


class TestECTDMetadataComparator(unittest.TestCase):
    """测试ECTDMetadataComparator类"""

    def setUp(self):
        """设置测试fixture"""
        # 创建前序列索引
        self.prev_index = SequenceMetadataIndex(
            sequence_number="0004",
            sequence_path="/path/to/0004"
        )

        # 创建当前序列索引
        self.curr_index = SequenceMetadataIndex(
            sequence_number="0005",
            sequence_path="/path/to/0005"
        )

    def test_comparator_init(self):
        """测试初始化"""
        comparator = ECTDMetadataComparator(self.prev_index, self.curr_index)

        self.assertEqual(comparator.previous_index, self.prev_index)
        self.assertEqual(comparator.current_index, self.curr_index)

    def test_detect_new_section(self):
        """测试检测新增section"""
        # 当前序列有新section
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A"}
        )

        snapshot = SectionMetadataSnapshot(
            sequence_number="0005",
            identifier=identifier,
            leaf_metadata=[
                LeafMetadata(leaf_id="l001", operation="new", title="Doc 1")
            ],
            section_path="m2/m2-3/s"
        )

        self.curr_index.add_section(snapshot)

        # 比对
        comparator = ECTDMetadataComparator(self.prev_index, self.curr_index)
        analyses = comparator.detect_metadata_changes()

        self.assertEqual(len(analyses), 1)
        self.assertTrue(analyses[0].is_new_section)
        # 新增section会有metadata_changes（所有属性都是"added"）
        self.assertTrue(analyses[0].has_metadata_change)

    def test_detect_deleted_section(self):
        """测试检测删除section"""
        # 前序列有section，当前序列删除
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A"}
        )

        snapshot = SectionMetadataSnapshot(
            sequence_number="0004",
            identifier=identifier,
            leaf_metadata=[
                LeafMetadata(leaf_id="l001", operation="new", title="Doc 1")
            ],
            section_path="m2/m2-3/s"
        )

        self.prev_index.add_section(snapshot)

        # 比对
        comparator = ECTDMetadataComparator(self.prev_index, self.curr_index)
        analyses = comparator.detect_metadata_changes()

        self.assertEqual(len(analyses), 1)
        self.assertTrue(analyses[0].is_deleted_section)

    def test_detect_metadata_change(self):
        """测试检测元数据变更"""
        # 前序列
        prev_identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-X"}
        )
        prev_snapshot = SectionMetadataSnapshot(
            sequence_number="0004",
            identifier=prev_identifier,
            leaf_metadata=[],
            section_path="m2/m2-3/s"
        )
        self.prev_index.add_section(prev_snapshot)

        # 当前序列（manufacturer变更）
        curr_identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A", "manufacturer": "MFR-Y"}
        )
        curr_snapshot = SectionMetadataSnapshot(
            sequence_number="0005",
            identifier=curr_identifier,
            leaf_metadata=[],
            section_path="m2/m2-3/s"
        )
        self.curr_index.add_section(curr_snapshot)

        # 比对
        comparator = ECTDMetadataComparator(self.prev_index, self.curr_index)
        analyses = comparator.detect_metadata_changes()

        # 应该只有1个section被分析
        # (之前也许分析了多个leaf，但section层面只有1个)
        section_count = len(set(a.section_identifier.get_matching_key() for a in analyses))
        self.assertEqual(section_count, 1)

        # 找到更新的section
        updated = [a for a in analyses if a.is_updated_section]
        self.assertTrue(len(updated) > 0)
        self.assertTrue(updated[0].has_metadata_change)

        # 检查变更详情
        changes = analyses[0].metadata_changes
        manufacturer_change = next(
            c for c in changes if c.attribute_name == "manufacturer"
        )
        self.assertEqual(manufacturer_change.change_type, "modified")
        self.assertEqual(manufacturer_change.old_value, "MFR-X")
        self.assertEqual(manufacturer_change.new_value, "MFR-Y")

    def test_get_violations(self):
        """测试获取违规列表"""
        comparator = ECTDMetadataComparator(self.prev_index, self.curr_index)
        violations = comparator.get_violations()

        # 空序列，无违规
        self.assertEqual(len(violations), 0)

    def test_get_compliant_updates(self):
        """测试获取合规更新"""
        comparator = ECTDMetadataComparator(self.prev_index, self.curr_index)
        compliant = comparator.get_compliant_updates()

        # 空序列，无更新
        self.assertEqual(len(compliant), 0)

    def test_generate_summary_report(self):
        """测试生成摘要报告"""
        # 添加一个新section
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A"}
        )
        snapshot = SectionMetadataSnapshot(
            sequence_number="0005",
            identifier=identifier,
            leaf_metadata=[
                LeafMetadata(leaf_id="l001", operation="new", title="Doc 1")
            ],
            section_path="m2/m2-3/s"
        )
        self.curr_index.add_section(snapshot)

        # 生成报告
        comparator = ECTDMetadataComparator(self.prev_index, self.curr_index)
        comparator.detect_metadata_changes()
        report = comparator.generate_summary_report()

        self.assertEqual(report["previous_sequence"], "0004")
        self.assertEqual(report["current_sequence"], "0005")
        self.assertEqual(report["total_sections_analyzed"], 1)
        self.assertEqual(report["new_sections"], 1)
        self.assertEqual(report["deleted_sections"], 0)
        self.assertEqual(report["updated_sections"], 0)


class TestConvenienceFunction(unittest.TestCase):
    """测试便捷函数compare_sequence_metadata"""

    def test_compare_sequence_metadata(self):
        """测试便捷函数"""
        prev_index = SequenceMetadataIndex(
            sequence_number="0004",
            sequence_path="/path/to/0004"
        )

        curr_index = SequenceMetadataIndex(
            sequence_number="0005",
            sequence_path="/path/to/0005"
        )

        # 添加一个新section
        identifier = SectionIdentifier(
            element_name="m2-3-s-drug-substance",
            attributes={"substance": "API-A"}
        )
        snapshot = SectionMetadataSnapshot(
            sequence_number="0005",
            identifier=identifier,
            leaf_metadata=[],
            section_path="m2/m2-3/s"
        )
        curr_index.add_section(snapshot)

        # 使用便捷函数比对
        analyses = compare_sequence_metadata(prev_index, curr_index)

        self.assertEqual(len(analyses), 1)
        self.assertTrue(analyses[0].is_new_section)


if __name__ == '__main__':
    unittest.main()
