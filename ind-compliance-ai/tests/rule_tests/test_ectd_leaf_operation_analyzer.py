"""
测试eCTD Leaf操作分析模块

测试覆盖：
1. LeafOperation和LeafOperationAnalysis数据结构
2. Replace语义判断逻辑
3. Replace操作验证逻辑
4. 批量操作分析
5. 工具函数（筛选、分组、统计）
6. 复杂场景和边界情况

版本: v1.0
创建日期: 2026-09-11
"""

import unittest
from core.ectd_leaf_operation_analyzer import (
    LeafOperation,
    LeafOperationType,
    LeafOperationAnalysis,
    ReplaceSemantics,
    determine_replace_semantics,
    validate_replace_operation,
    analyze_leaf_operations,
    filter_violations,
    group_by_section,
    get_operation_statistics
)


# ============================================================================
# 测试组1: 数据结构基本功能
# ============================================================================

class TestDataStructures(unittest.TestCase):
    """测试基本数据结构"""

    def test_leaf_operation_creation(self):
        """测试LeafOperation创建"""
        op = LeafOperation(
            leaf_id="l-001",
            operation_type=LeafOperationType.REPLACE,
            checksum="abc123",
            file_path="m2/file.pdf",
            section_path="m2-3-s[substance='API-A']",
            title="Substance Information"
        )

        assert op.leaf_id == "l-001"
        assert op.operation_type == LeafOperationType.REPLACE
        assert op.checksum == "abc123"
        assert op.file_path == "m2/file.pdf"
        assert op.section_path == "m2-3-s[substance='API-A']"
        assert op.title == "Substance Information"

    def test_leaf_operation_repr(self):
        """测试LeafOperation的字符串表示"""
        op = LeafOperation(
            leaf_id="l-001",
            operation_type=LeafOperationType.NEW,
            checksum="abc123",
            file_path="m2/file.pdf",
            section_path="m2-3-s[substance='API-A']"
        )

        repr_str = repr(op)
        assert "LeafOperation" in repr_str
        assert "l-001" in repr_str
        assert "new" in repr_str

    def test_leaf_operation_analysis_creation(self):
        """测试LeafOperationAnalysis创建"""
        op = LeafOperation(
            leaf_id="l-001",
            operation_type=LeafOperationType.REPLACE,
            checksum="new_hash",
            file_path="m2/file.pdf",
            section_path="m2-3-s[substance='API-A']"
        )

        analysis = LeafOperationAnalysis(
            leaf_operation=op,
            replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING,
            previous_leaf_id="l-001",
            previous_checksum="old_hash",
            metadata_changed=True,
            metadata_changes={"manufacturer": ("MFR-X", "MFR-Y")},
            requires_full_update=True,
            is_compliant=True,
            violation_reason=None
        )

        assert analysis.leaf_operation == op
        assert analysis.replace_semantics == ReplaceSemantics.METADATA_CONTENT_COUPLING
        assert analysis.metadata_changed is True
        assert analysis.is_compliant is True

    def test_get_operation_summary_replace_compliant(self):
        """测试符合规范的Replace操作摘要"""
        op = LeafOperation(
            leaf_id="l-001",
            operation_type=LeafOperationType.REPLACE,
            checksum="new_hash",
            file_path="m2/file.pdf",
            section_path="m2-3-s[substance='API-A']"
        )

        analysis = LeafOperationAnalysis(
            leaf_operation=op,
            replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING,
            metadata_changed=True,
            metadata_changes={"manufacturer": ("MFR-X", "MFR-Y")},
            is_compliant=True
        )

        summary = analysis.get_operation_summary()
        assert "REPLACE" in summary
        assert "metadata_content_coupling" in summary
        assert "manufacturer" in summary
        assert "compliant" in summary

    def test_get_operation_summary_replace_violation(self):
        """测试违规的Replace操作摘要"""
        op = LeafOperation(
            leaf_id="l-001",
            operation_type=LeafOperationType.REPLACE,
            checksum="same_hash",
            file_path="m2/file.pdf",
            section_path="m2-3-s[substance='API-A']"
        )

        analysis = LeafOperationAnalysis(
            leaf_operation=op,
            replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING,
            metadata_changed=True,
            is_compliant=False,
            violation_reason="Metadata changed but content not updated"
        )

        summary = analysis.get_operation_summary()
        assert "REPLACE" in summary
        assert "VIOLATION" in summary
        assert "Metadata changed" in summary

    def test_get_operation_summary_new(self):
        """测试NEW操作摘要"""
        op = LeafOperation(
            leaf_id="l-001",
            operation_type=LeafOperationType.NEW,
            checksum="abc123",
            file_path="m2/file.pdf",
            section_path="m2-3-s[substance='API-A']"
        )

        analysis = LeafOperationAnalysis(
            leaf_operation=op,
            is_compliant=True
        )

        summary = analysis.get_operation_summary()
        assert "NEW" in summary
        assert "m2/file.pdf" in summary


# ============================================================================
# 测试组2: Replace语义判断
# ============================================================================

class TestReplaceSemantics(unittest.TestCase):
    """测试Replace操作语义判断"""

    def test_metadata_content_coupling(self):
        """测试元数据-内容耦合语义"""
        semantics = determine_replace_semantics(
            current_checksum="new_hash",
            previous_checksum="old_hash",
            metadata_changed=True,
            section_requires_lifecycle_tracking=True
        )

        assert semantics == ReplaceSemantics.METADATA_CONTENT_COUPLING

    def test_full_content_update(self):
        """测试完整内容更新语义"""
        semantics = determine_replace_semantics(
            current_checksum="new_hash",
            previous_checksum="old_hash",
            metadata_changed=False,
            section_requires_lifecycle_tracking=True
        )

        assert semantics == ReplaceSemantics.FULL_CONTENT_UPDATE

    def test_minor_correction(self):
        """测试轻微修正语义"""
        semantics = determine_replace_semantics(
            current_checksum="same_hash",
            previous_checksum="same_hash",
            metadata_changed=False,
            section_requires_lifecycle_tracking=True
        )

        assert semantics == ReplaceSemantics.MINOR_CORRECTION

    def test_unknown_no_previous_checksum(self):
        """测试无法判断语义（缺少前序列数据）"""
        semantics = determine_replace_semantics(
            current_checksum="new_hash",
            previous_checksum=None,
            metadata_changed=False,
            section_requires_lifecycle_tracking=True
        )

        assert semantics == ReplaceSemantics.UNKNOWN

    def test_metadata_changed_no_lifecycle_tracking(self):
        """测试元数据变更但不需要生命周期追踪"""
        semantics = determine_replace_semantics(
            current_checksum="new_hash",
            previous_checksum="old_hash",
            metadata_changed=True,
            section_requires_lifecycle_tracking=False
        )

        # 不需要lifecycle tracking，判断为完整内容更新
        assert semantics == ReplaceSemantics.FULL_CONTENT_UPDATE


# ============================================================================
# 测试组3: Replace操作验证
# ============================================================================

class TestReplaceValidation(unittest.TestCase):
    """测试Replace操作验证逻辑"""

    def test_validate_compliant_metadata_content_coupling(self):
        """测试符合规范的元数据-内容耦合"""
        is_compliant, reason = validate_replace_operation(
            current_checksum="new_hash",
            previous_checksum="old_hash",
            metadata_changed=True,
            section_requires_lifecycle_tracking=True,
            replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING
        )

        assert is_compliant is True
        assert reason is None

    def test_validate_violation_metadata_changed_content_same(self):
        """测试违规：元数据变更但内容未变"""
        is_compliant, reason = validate_replace_operation(
            current_checksum="same_hash",
            previous_checksum="same_hash",
            metadata_changed=True,
            section_requires_lifecycle_tracking=True,
            replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING
        )

        assert is_compliant is False
        assert reason is not None
        assert "Metadata changed but content not updated" in reason

    def test_validate_violation_coupling_semantics_but_same_checksum(self):
        """测试违规：标记为耦合但校验和相同"""
        is_compliant, reason = validate_replace_operation(
            current_checksum="same_hash",
            previous_checksum="same_hash",
            metadata_changed=False,
            section_requires_lifecycle_tracking=False,
            replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING
        )

        assert is_compliant is False
        assert "metadata-content coupling" in reason
        assert "identical" in reason

    def test_validate_compliant_minor_correction(self):
        """测试符合规范的轻微修正"""
        is_compliant, reason = validate_replace_operation(
            current_checksum="same_hash",
            previous_checksum="same_hash",
            metadata_changed=False,
            section_requires_lifecycle_tracking=False,
            replace_semantics=ReplaceSemantics.MINOR_CORRECTION
        )

        assert is_compliant is True
        assert reason is None

    def test_validate_compliant_unknown_semantics(self):
        """测试未知语义但不标记为违规"""
        is_compliant, reason = validate_replace_operation(
            current_checksum="same_hash",
            previous_checksum="same_hash",
            metadata_changed=False,
            section_requires_lifecycle_tracking=False,
            replace_semantics=ReplaceSemantics.UNKNOWN
        )

        # UNKNOWN语义不标记为违规，但建议人工审核
        assert is_compliant is True
        assert reason is None

    def test_validate_no_previous_checksum(self):
        """测试没有前序列校验和时的验证"""
        is_compliant, reason = validate_replace_operation(
            current_checksum="new_hash",
            previous_checksum=None,
            metadata_changed=True,
            section_requires_lifecycle_tracking=True,
            replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING
        )

        # 没有前序列数据，无法验证，默认符合
        assert is_compliant is True
        assert reason is None


# ============================================================================
# 测试组4: 批量操作分析
# ============================================================================

class TestBatchAnalysis(unittest.TestCase):
    """测试批量操作分析"""

    def test_analyze_leaf_operations_single_replace_compliant(self):
        """测试单个符合规范的Replace操作"""
        current_ops = [
            LeafOperation(
                leaf_id="l-001",
                operation_type=LeafOperationType.REPLACE,
                checksum="new_hash",
                file_path="m2/file.pdf",
                section_path="m2-3-s[substance='API-A']"
            )
        ]
        previous_ops = [
            LeafOperation(
                leaf_id="l-001",
                operation_type=LeafOperationType.NEW,
                checksum="old_hash",
                file_path="m2/file.pdf",
                section_path="m2-3-s[substance='API-A']"
            )
        ]
        metadata_changes = {
            "m2-3-s[substance='API-A']": {"manufacturer": ("MFR-X", "MFR-Y")}
        }
        section_lifecycle = {
            "m2-3-s[substance='API-A']": True
        }

        analyses = analyze_leaf_operations(
            current_ops, previous_ops, metadata_changes, section_lifecycle
        )

        assert len(analyses) == 1
        analysis = analyses[0]
        assert analysis.replace_semantics == ReplaceSemantics.METADATA_CONTENT_COUPLING
        assert analysis.metadata_changed is True
        assert analysis.is_compliant is True

    def test_analyze_leaf_operations_single_replace_violation(self):
        """测试单个违规的Replace操作"""
        current_ops = [
            LeafOperation(
                leaf_id="l-001",
                operation_type=LeafOperationType.REPLACE,
                checksum="same_hash",  # 内容未变
                file_path="m2/file.pdf",
                section_path="m2-3-s[substance='API-A']"
            )
        ]
        previous_ops = [
            LeafOperation(
                leaf_id="l-001",
                operation_type=LeafOperationType.NEW,
                checksum="same_hash",  # 内容未变
                file_path="m2/file.pdf",
                section_path="m2-3-s[substance='API-A']"
            )
        ]
        metadata_changes = {
            "m2-3-s[substance='API-A']": {"manufacturer": ("MFR-X", "MFR-Y")}
        }
        section_lifecycle = {
            "m2-3-s[substance='API-A']": True
        }

        analyses = analyze_leaf_operations(
            current_ops, previous_ops, metadata_changes, section_lifecycle
        )

        assert len(analyses) == 1
        analysis = analyses[0]
        assert analysis.is_compliant is False
        assert "Metadata changed but content not updated" in analysis.violation_reason

    def test_analyze_leaf_operations_new_operation(self):
        """测试NEW操作（默认符合规范）"""
        current_ops = [
            LeafOperation(
                leaf_id="l-002",
                operation_type=LeafOperationType.NEW,
                checksum="new_hash",
                file_path="m2/new-file.pdf",
                section_path="m2-3-s[substance='API-B']"
            )
        ]
        previous_ops = []
        metadata_changes = {}
        section_lifecycle = {}

        analyses = analyze_leaf_operations(
            current_ops, previous_ops, metadata_changes, section_lifecycle
        )

        assert len(analyses) == 1
        analysis = analyses[0]
        assert analysis.leaf_operation.operation_type == LeafOperationType.NEW
        assert analysis.is_compliant is True
        assert analysis.replace_semantics is None

    def test_analyze_leaf_operations_delete_operation(self):
        """测试DELETE操作（默认符合规范）"""
        current_ops = [
            LeafOperation(
                leaf_id="l-003",
                operation_type=LeafOperationType.DELETE,
                checksum="",
                file_path="m2/deleted-file.pdf",
                section_path="m2-3-s[substance='API-C']"
            )
        ]
        previous_ops = [
            LeafOperation(
                leaf_id="l-003",
                operation_type=LeafOperationType.NEW,
                checksum="old_hash",
                file_path="m2/deleted-file.pdf",
                section_path="m2-3-s[substance='API-C']"
            )
        ]
        metadata_changes = {}
        section_lifecycle = {}

        analyses = analyze_leaf_operations(
            current_ops, previous_ops, metadata_changes, section_lifecycle
        )

        assert len(analyses) == 1
        analysis = analyses[0]
        assert analysis.leaf_operation.operation_type == LeafOperationType.DELETE
        assert analysis.is_compliant is True

    def test_analyze_leaf_operations_multiple_mixed(self):
        """测试混合多个操作"""
        current_ops = [
            LeafOperation(
                leaf_id="l-001",
                operation_type=LeafOperationType.REPLACE,
                checksum="new_hash1",
                file_path="m2/file1.pdf",
                section_path="m2-3-s[substance='API-A']"
            ),
            LeafOperation(
                leaf_id="l-002",
                operation_type=LeafOperationType.NEW,
                checksum="new_hash2",
                file_path="m2/file2.pdf",
                section_path="m2-3-s[substance='API-B']"
            ),
            LeafOperation(
                leaf_id="l-003",
                operation_type=LeafOperationType.REPLACE,
                checksum="same_hash",  # 违规：元数据变但内容未变
                file_path="m2/file3.pdf",
                section_path="m2-3-p[product='P1']"
            )
        ]
        previous_ops = [
            LeafOperation(
                leaf_id="l-001",
                operation_type=LeafOperationType.NEW,
                checksum="old_hash1",
                file_path="m2/file1.pdf",
                section_path="m2-3-s[substance='API-A']"
            ),
            LeafOperation(
                leaf_id="l-003",
                operation_type=LeafOperationType.NEW,
                checksum="same_hash",
                file_path="m2/file3.pdf",
                section_path="m2-3-p[product='P1']"
            )
        ]
        metadata_changes = {
            "m2-3-s[substance='API-A']": {"manufacturer": ("MFR-X", "MFR-Y")},
            "m2-3-p[product='P1']": {"dosageform": ("Tablet", "Capsule")}
        }
        section_lifecycle = {
            "m2-3-s[substance='API-A']": True,
            "m2-3-p[product='P1']": True
        }

        analyses = analyze_leaf_operations(
            current_ops, previous_ops, metadata_changes, section_lifecycle
        )

        assert len(analyses) == 3

        # l-001: REPLACE, 元数据变+内容变 = 符合规范
        assert analyses[0].is_compliant is True

        # l-002: NEW = 符合规范
        assert analyses[1].is_compliant is True

        # l-003: REPLACE, 元数据变+内容未变 = 违规
        assert analyses[2].is_compliant is False

    def test_analyze_leaf_operations_no_previous_sequence(self):
        """测试没有前序列时的分析（初始序列）"""
        current_ops = [
            LeafOperation(
                leaf_id="l-001",
                operation_type=LeafOperationType.NEW,
                checksum="hash1",
                file_path="m2/file1.pdf",
                section_path="m2-3-s[substance='API-A']"
            )
        ]
        previous_ops = None
        metadata_changes = {}
        section_lifecycle = {}

        analyses = analyze_leaf_operations(
            current_ops, previous_ops, metadata_changes, section_lifecycle
        )

        assert len(analyses) == 1
        analysis = analyses[0]
        assert analysis.previous_leaf_id is None
        assert analysis.previous_checksum is None
        assert analysis.is_compliant is True


# ============================================================================
# 测试组5: 工具函数
# ============================================================================

class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""

    def test_filter_violations(self):
        """测试筛选违规操作"""
        analyses = [
            LeafOperationAnalysis(
                leaf_operation=LeafOperation(
                    leaf_id="l-001",
                    operation_type=LeafOperationType.REPLACE,
                    checksum="hash1",
                    file_path="file1.pdf",
                    section_path="section1"
                ),
                is_compliant=True
            ),
            LeafOperationAnalysis(
                leaf_operation=LeafOperation(
                    leaf_id="l-002",
                    operation_type=LeafOperationType.REPLACE,
                    checksum="hash2",
                    file_path="file2.pdf",
                    section_path="section2"
                ),
                is_compliant=False,
                violation_reason="Test violation"
            ),
            LeafOperationAnalysis(
                leaf_operation=LeafOperation(
                    leaf_id="l-003",
                    operation_type=LeafOperationType.NEW,
                    checksum="hash3",
                    file_path="file3.pdf",
                    section_path="section3"
                ),
                is_compliant=True
            )
        ]

        violations = filter_violations(analyses)

        assert len(violations) == 1
        assert violations[0].leaf_operation.leaf_id == "l-002"
        assert violations[0].is_compliant is False

    def test_group_by_section(self):
        """测试按section分组"""
        analyses = [
            LeafOperationAnalysis(
                leaf_operation=LeafOperation(
                    leaf_id="l-001",
                    operation_type=LeafOperationType.REPLACE,
                    checksum="hash1",
                    file_path="file1.pdf",
                    section_path="section-A"
                ),
                is_compliant=True
            ),
            LeafOperationAnalysis(
                leaf_operation=LeafOperation(
                    leaf_id="l-002",
                    operation_type=LeafOperationType.NEW,
                    checksum="hash2",
                    file_path="file2.pdf",
                    section_path="section-A"
                ),
                is_compliant=True
            ),
            LeafOperationAnalysis(
                leaf_operation=LeafOperation(
                    leaf_id="l-003",
                    operation_type=LeafOperationType.REPLACE,
                    checksum="hash3",
                    file_path="file3.pdf",
                    section_path="section-B"
                ),
                is_compliant=True
            )
        ]

        grouped = group_by_section(analyses)

        assert len(grouped) == 2
        assert "section-A" in grouped
        assert "section-B" in grouped
        assert len(grouped["section-A"]) == 2
        assert len(grouped["section-B"]) == 1

    def test_get_operation_statistics(self):
        """测试操作统计"""
        analyses = [
            LeafOperationAnalysis(
                leaf_operation=LeafOperation(
                    leaf_id="l-001",
                    operation_type=LeafOperationType.REPLACE,
                    checksum="hash1",
                    file_path="file1.pdf",
                    section_path="section1"
                ),
                replace_semantics=ReplaceSemantics.METADATA_CONTENT_COUPLING,
                is_compliant=True
            ),
            LeafOperationAnalysis(
                leaf_operation=LeafOperation(
                    leaf_id="l-002",
                    operation_type=LeafOperationType.NEW,
                    checksum="hash2",
                    file_path="file2.pdf",
                    section_path="section2"
                ),
                is_compliant=True
            ),
            LeafOperationAnalysis(
                leaf_operation=LeafOperation(
                    leaf_id="l-003",
                    operation_type=LeafOperationType.REPLACE,
                    checksum="hash3",
                    file_path="file3.pdf",
                    section_path="section3"
                ),
                replace_semantics=ReplaceSemantics.FULL_CONTENT_UPDATE,
                is_compliant=False,
                violation_reason="Test"
            ),
            LeafOperationAnalysis(
                leaf_operation=LeafOperation(
                    leaf_id="l-004",
                    operation_type=LeafOperationType.DELETE,
                    checksum="",
                    file_path="file4.pdf",
                    section_path="section4"
                ),
                is_compliant=True
            )
        ]

        stats = get_operation_statistics(analyses)

        assert stats["total"] == 4
        assert stats["new"] == 1
        assert stats["delete"] == 1
        assert stats["replace"] == 2
        assert stats["append"] == 0
        assert stats["violations"] == 1
        assert stats["metadata_content_coupling"] == 1
        assert stats["full_content_update"] == 1


# ============================================================================
# 测试组6: 复杂场景
# ============================================================================

class TestComplexScenarios(unittest.TestCase):
    """测试复杂场景"""

    def test_scenario_metadata_only_change_violation(self):
        """测试场景：仅元数据变更（违规）"""
        current_ops = [
            LeafOperation(
                leaf_id="l-m2-substance-001",
                operation_type=LeafOperationType.REPLACE,
                checksum="abc123def456",  # 内容未变
                file_path="m2/23s/substance.pdf",
                section_path="m2-3-s[substance='API-A', manufacturer='MFR-Y']"
            )
        ]
        previous_ops = [
            LeafOperation(
                leaf_id="l-m2-substance-001",
                operation_type=LeafOperationType.NEW,
                checksum="abc123def456",  # 内容未变
                file_path="m2/23s/substance.pdf",
                section_path="m2-3-s[substance='API-A', manufacturer='MFR-X']"
            )
        ]
        metadata_changes = {
            "m2-3-s[substance='API-A', manufacturer='MFR-Y']": {
                "manufacturer": ("MFR-X", "MFR-Y")
            }
        }
        section_lifecycle = {
            "m2-3-s[substance='API-A', manufacturer='MFR-Y']": True
        }

        analyses = analyze_leaf_operations(
            current_ops, previous_ops, metadata_changes, section_lifecycle
        )

        assert len(analyses) == 1
        assert analyses[0].is_compliant is False
        assert "Metadata changed but content not updated" in analyses[0].violation_reason

    def test_scenario_metadata_and_content_change_compliant(self):
        """测试场景：元数据和内容同时变更（符合规范）"""
        current_ops = [
            LeafOperation(
                leaf_id="l-m2-substance-001",
                operation_type=LeafOperationType.REPLACE,
                checksum="new_checksum",  # 内容变更
                file_path="m2/23s/substance.pdf",
                section_path="m2-3-s[substance='API-A', manufacturer='MFR-Y']"
            )
        ]
        previous_ops = [
            LeafOperation(
                leaf_id="l-m2-substance-001",
                operation_type=LeafOperationType.NEW,
                checksum="old_checksum",  # 内容变更
                file_path="m2/23s/substance.pdf",
                section_path="m2-3-s[substance='API-A', manufacturer='MFR-X']"
            )
        ]
        metadata_changes = {
            "m2-3-s[substance='API-A', manufacturer='MFR-Y']": {
                "manufacturer": ("MFR-X", "MFR-Y")
            }
        }
        section_lifecycle = {
            "m2-3-s[substance='API-A', manufacturer='MFR-Y']": True
        }

        analyses = analyze_leaf_operations(
            current_ops, previous_ops, metadata_changes, section_lifecycle
        )

        assert len(analyses) == 1
        assert analyses[0].is_compliant is True
        assert analyses[0].replace_semantics == ReplaceSemantics.METADATA_CONTENT_COUPLING

    def test_scenario_no_lifecycle_tracking_no_violation(self):
        """测试场景：不需要生命周期追踪的section，元数据变更不违规"""
        current_ops = [
            LeafOperation(
                leaf_id="l-excipient-001",
                operation_type=LeafOperationType.REPLACE,
                checksum="same_checksum",
                file_path="m3/32p4/excipient.pdf",
                section_path="m3-2-p-4-control-of-excipients[excipient='Lactose']"
            )
        ]
        previous_ops = [
            LeafOperation(
                leaf_id="l-excipient-001",
                operation_type=LeafOperationType.NEW,
                checksum="same_checksum",
                file_path="m3/32p4/excipient.pdf",
                section_path="m3-2-p-4-control-of-excipients[excipient='Starch']"
            )
        ]
        metadata_changes = {
            "m3-2-p-4-control-of-excipients[excipient='Lactose']": {
                "excipient": ("Starch", "Lactose")
            }
        }
        section_lifecycle = {
            "m3-2-p-4-control-of-excipients[excipient='Lactose']": False  # 不需要追踪
        }

        analyses = analyze_leaf_operations(
            current_ops, previous_ops, metadata_changes, section_lifecycle
        )

        assert len(analyses) == 1
        # 不需要lifecycle tracking，所以即使内容未变也不违规
        assert analyses[0].is_compliant is True


# ============================================================================
# 测试组7: 边界情况
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_empty_current_ops(self):
        """测试空的当前操作列表"""
        analyses = analyze_leaf_operations([], [], {}, {})
        assert len(analyses) == 0

    def test_missing_metadata_changes(self):
        """测试缺少元数据变更信息"""
        current_ops = [
            LeafOperation(
                leaf_id="l-001",
                operation_type=LeafOperationType.REPLACE,
                checksum="hash1",
                file_path="file.pdf",
                section_path="section-unknown"
            )
        ]

        analyses = analyze_leaf_operations(
            current_ops, None, {}, {}
        )

        assert len(analyses) == 1
        assert analyses[0].metadata_changed is False

    def test_missing_lifecycle_tracking_info(self):
        """测试缺少生命周期追踪信息（默认False）"""
        current_ops = [
            LeafOperation(
                leaf_id="l-001",
                operation_type=LeafOperationType.REPLACE,
                checksum="new_hash",
                file_path="file.pdf",
                section_path="section-unknown"
            )
        ]
        previous_ops = [
            LeafOperation(
                leaf_id="l-001",
                operation_type=LeafOperationType.NEW,
                checksum="old_hash",
                file_path="file.pdf",
                section_path="section-unknown"
            )
        ]

        analyses = analyze_leaf_operations(
            current_ops, previous_ops, {}, {}
        )

        assert len(analyses) == 1
        assert analyses[0].requires_full_update is False


if __name__ == "__main__":
    unittest.main()
