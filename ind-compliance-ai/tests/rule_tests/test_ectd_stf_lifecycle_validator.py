"""
测试 eCTD STF生命周期管理验证器

测试目标:
1. STF操作类型验证 (new vs append)
2. Modified-file引用验证
3. Study-identifier一致性验证
4. 累积方式验证 (Accumulative Approach)
5. 完整序列对验证

版本: v1.0
创建日期: 2026-09-14
"""

import pytest
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from core.ectd_stf_lifecycle_validator import (
    STFLifecycleValidator,
    STFLifecycleSnapshot,
    ViolationDetail,
    ViolationSeverity,
    validate_stf_lifecycle_pair,
    validate_stf_lifecycle_application
)


# ============================================================================
# 测试夹具：创建Mock STF文件
# ============================================================================

@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_mock_stf_file(
    file_path: Path,
    study_id: str,
    operation: str,
    modified_file: str = None,
    leaf_ids: List[str] = None
) -> str:
    """
    创建Mock STF XML文件

    参数:
        file_path: STF文件路径
        study_id: 研究ID
        operation: 操作类型 (new/append/delete/replace)
        modified_file: 引用的前一个STF路径
        leaf_ids: 引用的leaf ID列表

    返回:
        文件路径字符串
    """
    # 注册命名空间
    ET.register_namespace('ectd', 'http://www.ich.org/ectd')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')

    # 构建XML
    root = ET.Element("{http://www.ich.org/ectd}study")
    root.set("operation", operation)

    if modified_file:
        root.set("modified-file", modified_file)

    # 添加study-identifier元素
    study_identifier = ET.SubElement(root, "{http://www.ich.org/ectd}study-identifier")
    study_title = ET.SubElement(study_identifier, "{http://www.ich.org/ectd}study-title")
    study_title.text = f"Study {study_id}"

    # 添加study-document元素
    study_document = ET.SubElement(root, "{http://www.ich.org/ectd}study-document")

    if leaf_ids:
        for leaf_id in leaf_ids:
            leaf_ref = ET.SubElement(study_document, "{http://www.ich.org/ectd}leaf-reference")
            leaf_ref.set("{http://www.w3.org/1999/xlink}href", f"#{leaf_id}")

    # 写入文件
    tree = ET.ElementTree(root)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(file_path), encoding="utf-8", xml_declaration=True)

    return str(file_path)


# ============================================================================
# 测试 STFLifecycleSnapshot 数据结构
# ============================================================================

class TestSTFLifecycleSnapshot:
    """测试 STFLifecycleSnapshot 数据结构"""

    def test_snapshot_creation(self):
        """测试创建快照"""
        snapshot = STFLifecycleSnapshot(
            sequence_number="0005",
            stf_file_path="/path/to/stf-abc123.xml",
            study_id="abc123",
            operation="new",
            modified_file=None,
            leaf_ids={"leaf-001", "leaf-002"}
        )

        assert snapshot.sequence_number == "0005"
        assert snapshot.study_id == "abc123"
        assert snapshot.operation == "new"
        assert len(snapshot.leaf_ids) == 2

    def test_snapshot_from_list(self):
        """测试从列表初始化leaf_ids"""
        snapshot = STFLifecycleSnapshot(
            sequence_number="0005",
            stf_file_path="/path/to/stf-abc123.xml",
            leaf_ids=["leaf-001", "leaf-002"]
        )

        assert isinstance(snapshot.leaf_ids, set)
        assert len(snapshot.leaf_ids) == 2


# ============================================================================
# 测试 STF操作类型验证
# ============================================================================

class TestSTFOperationTypeValidation:
    """测试 STF操作类型验证"""

    def test_first_submission_should_be_new(self, temp_dir):
        """测试首次提交应使用operation='new'"""
        validator = STFLifecycleValidator()

        # 创建首次提交的STF (operation='new')
        seq_0000 = temp_dir / "0000"
        stf_path = seq_0000 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(stf_path, "abc123", "new", leaf_ids=["leaf-001"])

        snapshot = validator.extract_stf_snapshot(str(seq_0000), str(stf_path))
        violations = validator.validate_stf_operation_type(snapshot, None)

        # 断言：首次提交使用'new'，无违规
        assert len(violations) == 0

    def test_first_submission_with_append_should_fail(self, temp_dir):
        """测试首次提交错误使用operation='append'"""
        validator = STFLifecycleValidator()

        # 创建首次提交的STF (operation='append' - 错误)
        seq_0000 = temp_dir / "0000"
        stf_path = seq_0000 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(stf_path, "abc123", "append", leaf_ids=["leaf-001"])

        snapshot = validator.extract_stf_snapshot(str(seq_0000), str(stf_path))
        violations = validator.validate_stf_operation_type(snapshot, None)

        # 断言：应检测到违规
        assert len(violations) == 1
        assert violations[0].rule_id == "STF-LC-002"
        assert violations[0].severity == ViolationSeverity.ERROR
        assert "首次提交" in violations[0].message

    def test_subsequent_submission_should_be_append(self, temp_dir):
        """测试后续提交应使用operation='append'"""
        validator = STFLifecycleValidator()

        # 创建前序列的STF
        seq_0000 = temp_dir / "0000"
        prev_stf_path = seq_0000 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(prev_stf_path, "abc123", "new", leaf_ids=["leaf-001"])
        prev_snapshot = validator.extract_stf_snapshot(str(seq_0000), str(prev_stf_path))

        # 创建当前序列的STF (operation='append')
        seq_0001 = temp_dir / "0001"
        curr_stf_path = seq_0001 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(
            curr_stf_path,
            "abc123",
            "append",
            modified_file="../0000/m5/stf-abc123.xml",
            leaf_ids=["leaf-002"]
        )
        curr_snapshot = validator.extract_stf_snapshot(str(seq_0001), str(curr_stf_path))

        violations = validator.validate_stf_operation_type(curr_snapshot, prev_snapshot)

        # 断言：后续提交使用'append'，无违规
        assert len(violations) == 0

    def test_subsequent_submission_with_new_should_fail(self, temp_dir):
        """测试后续提交错误使用operation='new'"""
        validator = STFLifecycleValidator()

        # 创建前序列的STF
        seq_0000 = temp_dir / "0000"
        prev_stf_path = seq_0000 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(prev_stf_path, "abc123", "new", leaf_ids=["leaf-001"])
        prev_snapshot = validator.extract_stf_snapshot(str(seq_0000), str(prev_stf_path))

        # 创建当前序列的STF (operation='new' - 错误)
        seq_0001 = temp_dir / "0001"
        curr_stf_path = seq_0001 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(curr_stf_path, "abc123", "new", leaf_ids=["leaf-002"])
        curr_snapshot = validator.extract_stf_snapshot(str(seq_0001), str(curr_stf_path))

        violations = validator.validate_stf_operation_type(curr_snapshot, prev_snapshot)

        # 断言：应检测到违规
        assert len(violations) == 1
        assert violations[0].rule_id == "STF-LC-003"
        assert violations[0].severity == ViolationSeverity.ERROR
        assert "后续提交" in violations[0].message

    def test_delete_operation_should_warn(self, temp_dir):
        """测试使用operation='delete'应产生警告"""
        validator = STFLifecycleValidator()

        # 创建使用delete操作的STF
        seq_0001 = temp_dir / "0001"
        stf_path = seq_0001 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(stf_path, "abc123", "delete", leaf_ids=[])

        # 模拟前序列快照
        prev_snapshot = STFLifecycleSnapshot(
            sequence_number="0000",
            stf_file_path=str(temp_dir / "0000" / "m5" / "stf-abc123.xml"),
            study_id="abc123",
            operation="new"
        )

        curr_snapshot = validator.extract_stf_snapshot(str(seq_0001), str(stf_path))
        violations = validator.validate_stf_operation_type(curr_snapshot, prev_snapshot)

        # 断言：应检测到警告
        assert any(v.rule_id == "STF-LC-004" for v in violations)
        warning_violation = next(v for v in violations if v.rule_id == "STF-LC-004")
        assert warning_violation.severity == ViolationSeverity.WARNING
        assert "不应使用" in warning_violation.message


# ============================================================================
# 测试 Modified-file引用验证
# ============================================================================

class TestModifiedFileValidation:
    """测试 Modified-file引用验证"""

    def test_append_without_modified_file_should_fail(self, temp_dir):
        """测试append操作缺少modified-file属性"""
        validator = STFLifecycleValidator()

        # 创建前序列
        prev_snapshot = STFLifecycleSnapshot(
            sequence_number="0000",
            stf_file_path=str(temp_dir / "0000" / "m5" / "stf-abc123.xml"),
            operation="new"
        )

        # 创建当前序列 (append但没有modified-file)
        seq_0001 = temp_dir / "0001"
        curr_stf_path = seq_0001 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(curr_stf_path, "abc123", "append", leaf_ids=["leaf-002"])
        curr_snapshot = validator.extract_stf_snapshot(str(seq_0001), str(curr_stf_path))

        violations = validator.validate_modified_file_reference(curr_snapshot, prev_snapshot)

        # 断言：应检测到违规
        assert len(violations) == 1
        assert violations[0].rule_id == "STF-LC-005"
        assert violations[0].severity == ViolationSeverity.ERROR
        assert "缺少modified-file" in violations[0].message

    def test_append_with_correct_modified_file(self, temp_dir):
        """测试append操作正确引用了modified-file"""
        validator = STFLifecycleValidator()

        # 创建前序列
        prev_snapshot = STFLifecycleSnapshot(
            sequence_number="0000",
            stf_file_path=str(temp_dir / "0000" / "m5" / "stf-abc123.xml"),
            operation="new"
        )

        # 创建当前序列 (正确引用modified-file)
        seq_0001 = temp_dir / "0001"
        curr_stf_path = seq_0001 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(
            curr_stf_path,
            "abc123",
            "append",
            modified_file="../0000/m5/stf-abc123.xml",
            leaf_ids=["leaf-002"]
        )
        curr_snapshot = validator.extract_stf_snapshot(str(seq_0001), str(curr_stf_path))

        violations = validator.validate_modified_file_reference(curr_snapshot, prev_snapshot)

        # 断言：无违规
        assert len(violations) == 0

    def test_append_with_wrong_modified_file_should_warn(self, temp_dir):
        """测试append操作引用了错误的modified-file"""
        validator = STFLifecycleValidator()

        # 创建前序列
        prev_snapshot = STFLifecycleSnapshot(
            sequence_number="0001",
            stf_file_path=str(temp_dir / "0001" / "m5" / "stf-abc123.xml"),
            operation="append"
        )

        # 创建当前序列 (引用了错误的STF)
        seq_0002 = temp_dir / "0002"
        curr_stf_path = seq_0002 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(
            curr_stf_path,
            "abc123",
            "append",
            modified_file="../0000/m5/stf-abc123.xml",  # 错误：应该是../0001/...
            leaf_ids=["leaf-003"]
        )
        curr_snapshot = validator.extract_stf_snapshot(str(seq_0002), str(curr_stf_path))

        violations = validator.validate_modified_file_reference(curr_snapshot, prev_snapshot)

        # 断言：应检测到警告
        assert len(violations) == 1
        assert violations[0].rule_id == "STF-LC-007"
        assert violations[0].severity == ViolationSeverity.WARNING
        assert "可能未引用最近一次" in violations[0].message


# ============================================================================
# 测试 Study-identifier一致性验证
# ============================================================================

class TestStudyIdentifierConsistency:
    """测试 Study-identifier一致性验证"""

    def test_consistent_study_id(self, temp_dir):
        """测试study-id保持一致"""
        validator = STFLifecycleValidator()

        prev_snapshot = STFLifecycleSnapshot(
            sequence_number="0000",
            stf_file_path=str(temp_dir / "0000" / "stf-abc123.xml"),
            study_id="abc123",
            operation="new"
        )

        curr_snapshot = STFLifecycleSnapshot(
            sequence_number="0001",
            stf_file_path=str(temp_dir / "0001" / "stf-abc123.xml"),
            study_id="abc123",
            operation="append"
        )

        violations = validator.validate_study_identifier_consistency(curr_snapshot, prev_snapshot)

        # 断言：无违规
        assert len(violations) == 0

    def test_changed_study_id_should_fail(self, temp_dir):
        """测试study-id发生变化"""
        validator = STFLifecycleValidator()

        prev_snapshot = STFLifecycleSnapshot(
            sequence_number="0000",
            stf_file_path=str(temp_dir / "0000" / "stf-abc123.xml"),
            study_id="abc123",
            operation="new"
        )

        curr_snapshot = STFLifecycleSnapshot(
            sequence_number="0001",
            stf_file_path=str(temp_dir / "0001" / "stf-xyz789.xml"),
            study_id="xyz789",  # 变化了
            operation="append"
        )

        violations = validator.validate_study_identifier_consistency(curr_snapshot, prev_snapshot)

        # 断言：应检测到违规
        assert len(violations) == 1
        assert violations[0].rule_id == "STF-LC-008"
        assert violations[0].severity == ViolationSeverity.ERROR
        assert "study-identifier在不同序列间发生了变化" in violations[0].message


# ============================================================================
# 测试累积方式验证
# ============================================================================

class TestCumulativeApproach:
    """测试累积方式验证"""

    def test_no_duplicate_leaves(self, temp_dir):
        """测试后续STF没有重复引用leaf"""
        validator = STFLifecycleValidator()

        prev_snapshot = STFLifecycleSnapshot(
            sequence_number="0000",
            stf_file_path=str(temp_dir / "0000" / "stf-abc123.xml"),
            operation="new",
            leaf_ids={"leaf-001", "leaf-002"}
        )

        curr_snapshot = STFLifecycleSnapshot(
            sequence_number="0001",
            stf_file_path=str(temp_dir / "0001" / "stf-abc123.xml"),
            operation="append",
            leaf_ids={"leaf-003", "leaf-004"}  # 新的leaf，无重复
        )

        violations = validator.validate_cumulative_approach(curr_snapshot, prev_snapshot)

        # 断言：无违规
        assert len(violations) == 0

    def test_duplicate_leaves_should_warn(self, temp_dir):
        """测试后续STF包含重复的leaf引用"""
        validator = STFLifecycleValidator()

        prev_snapshot = STFLifecycleSnapshot(
            sequence_number="0000",
            stf_file_path=str(temp_dir / "0000" / "stf-abc123.xml"),
            operation="new",
            leaf_ids={"leaf-001", "leaf-002"}
        )

        curr_snapshot = STFLifecycleSnapshot(
            sequence_number="0001",
            stf_file_path=str(temp_dir / "0001" / "stf-abc123.xml"),
            operation="append",
            leaf_ids={"leaf-002", "leaf-003"}  # leaf-002重复了
        )

        violations = validator.validate_cumulative_approach(curr_snapshot, prev_snapshot)

        # 断言：应检测到警告
        assert len(violations) == 1
        assert violations[0].rule_id == "STF-LC-009"
        assert violations[0].severity == ViolationSeverity.WARNING
        assert "包含前序列已存在的leaf引用" in violations[0].message


# ============================================================================
# 集成测试：完整序列对验证
# ============================================================================

class TestSequencePairValidation:
    """测试完整序列对验证"""

    def test_compliant_sequence_pair(self, temp_dir):
        """测试完全合规的序列对"""
        # 创建序列0000
        seq_0000 = temp_dir / "0000"
        stf_0000 = seq_0000 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(stf_0000, "abc123", "new", leaf_ids=["leaf-001"])

        # 创建序列0001
        seq_0001 = temp_dir / "0001"
        stf_0001 = seq_0001 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(
            stf_0001,
            "abc123",
            "append",
            modified_file="../0000/m5/stf-abc123.xml",
            leaf_ids=["leaf-002"]
        )

        # 验证
        violations = validate_stf_lifecycle_pair(str(seq_0001), str(seq_0000))

        # 断言：无违规
        assert len(violations) == 0

    def test_violation_sequence_pair(self, temp_dir):
        """测试有违规的序列对"""
        # 创建序列0000
        seq_0000 = temp_dir / "0000"
        stf_0000 = seq_0000 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(stf_0000, "abc123", "new", leaf_ids=["leaf-001"])

        # 创建序列0001 (多个违规：operation错误、缺少modified-file)
        seq_0001 = temp_dir / "0001"
        stf_0001 = seq_0001 / "m5" / "stf-abc123.xml"
        create_mock_stf_file(stf_0001, "abc123", "new", leaf_ids=["leaf-002"])  # 应该是append

        # 验证
        violations = validate_stf_lifecycle_pair(str(seq_0001), str(seq_0000))

        # 断言：应检测到至少1个违规
        assert len(violations) >= 1
        assert any(v.rule_id == "STF-LC-003" for v in violations)  # operation错误


# ============================================================================
# 测试便捷函数
# ============================================================================

class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_validate_application_sequences(self, temp_dir):
        """测试批量验证整个申请的所有序列"""
        # 创建多个序列
        for seq_num in ["0000", "0001", "0002"]:
            seq_dir = temp_dir / seq_num
            stf_path = seq_dir / "m5" / "stf-abc123.xml"

            if seq_num == "0000":
                create_mock_stf_file(stf_path, "abc123", "new", leaf_ids=["leaf-001"])
            else:
                prev_seq = f"{int(seq_num) - 1:04d}"
                create_mock_stf_file(
                    stf_path,
                    "abc123",
                    "append",
                    modified_file=f"../{prev_seq}/m5/stf-abc123.xml",
                    leaf_ids=[f"leaf-{seq_num}"]
                )

        # 验证
        results = validate_stf_lifecycle_application(str(temp_dir))

        # 断言：所有序列都合规（无违规记录）
        assert len(results) == 0  # 无违规的序列不会出现在results中
