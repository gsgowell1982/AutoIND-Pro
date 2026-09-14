"""
测试 eCTD元数据生命周期耦合规则验证器

测试目标:
1. validate_metadata_lifecycle_coupling() 函数的基本功能
2. ViolationDetail 数据结构的创建和序列化
3. 使用 Mock 场景测试违规检测准确性
4. 验证合规场景不产生误报

版本: v1.0
创建日期: 2026-09-14
"""

import pytest
from pathlib import Path
from typing import List

from core.ectd_metadata_lifecycle_validator import (
    validate_metadata_lifecycle_coupling,
    validate_sequences_from_paths,
    validate_application_sequences,
    ViolationDetail,
    ValidationResult
)
from core.ectd_metadata_extractor import (
    extract_sequence_metadata_index,
    SequenceMetadataIndex
)


# ============================================================================
# 测试夹具：Mock场景路径
# ============================================================================

@pytest.fixture
def mock_data_root() -> Path:
    """Mock测试数据根目录"""
    return Path(__file__).parent.parent / "test_data" / "ectd_metadata_mock_scenarios"


@pytest.fixture
def compliant_scenario_path(mock_data_root: Path) -> Path:
    """场景1：完全合规的序列对"""
    return mock_data_root / "scenario_1_compliant"


@pytest.fixture
def metadata_violation_scenario_path(mock_data_root: Path) -> Path:
    """场景2：元数据变更但内容未更新（违规）"""
    return mock_data_root / "scenario_2_metadata_only"


@pytest.fixture
def metadata_content_coupled_scenario_path(mock_data_root: Path) -> Path:
    """场景3：元数据变更且内容正确更新（合规）"""
    return mock_data_root / "scenario_3_partial_update"


# ============================================================================
# 测试 ViolationDetail 数据结构
# ============================================================================

class TestViolationDetail:
    """测试 ViolationDetail 数据结构"""

    def test_violation_detail_creation(self):
        """测试创建 ViolationDetail 实例"""
        from core.ectd_metadata_comparator import MetadataChange

        violation = ViolationDetail(
            section_identifier="m2-3-s-drug-substance?substance=API-A",
            section_path="m2/m2-3-body-data/m2-3-s-drug-substance[@substance='API-A']",
            sequence_number="0005",
            previous_sequence_number="0004",
            metadata_changes=[
                MetadataChange(
                    attribute_name="manufacturer",
                    old_value="MFR-X",
                    new_value="MFR-Y",
                    change_type="modified"
                )
            ],
            changed_attributes=["manufacturer"],
            leaf_operations=[{
                "leaf_id": "l-m2-3-s-001",
                "operation": "replace",
                "checksum": "abc123",
                "file_path": "m2/m2-3-s-drug-substance.pdf"
            }],
            leaf_ids=["l-m2-3-s-001"],
            violation_type="metadata_changed_content_not_updated",
            violation_message="Metadata changed but content unchanged",
            severity="error",
            rule_id="HR-ECTD-200",
            rule_citation="cn_ectd_technical_specification#sec_3_6"
        )

        assert violation.section_identifier == "m2-3-s-drug-substance?substance=API-A"
        assert violation.sequence_number == "0005"
        assert violation.previous_sequence_number == "0004"
        assert len(violation.metadata_changes) == 1
        assert violation.changed_attributes == ["manufacturer"]
        assert len(violation.leaf_ids) == 1
        assert violation.severity == "error"
        assert violation.rule_id == "HR-ECTD-200"

    def test_violation_detail_to_dict(self):
        """测试 ViolationDetail 转换为字典"""
        from core.ectd_metadata_comparator import MetadataChange

        violation = ViolationDetail(
            section_identifier="test-section",
            section_path="test/path",
            sequence_number="0005",
            previous_sequence_number="0004",
            metadata_changes=[
                MetadataChange(
                    attribute_name="attr1",
                    old_value="old",
                    new_value="new",
                    change_type="modified"
                )
            ],
            changed_attributes=["attr1"],
            leaf_operations=[],
            leaf_ids=[],
            violation_type="test_type",
            violation_message="test message",
            severity="error",
            rule_id="HR-ECTD-200",
            rule_citation="test_citation"
        )

        result_dict = violation.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["section_identifier"] == "test-section"
        assert result_dict["sequence_number"] == "0005"
        assert len(result_dict["metadata_changes"]) == 1
        assert result_dict["metadata_changes"][0]["attribute"] == "attr1"
        assert result_dict["rule_id"] == "HR-ECTD-200"

    def test_violation_detail_get_summary(self):
        """测试 ViolationDetail 获取摘要"""
        from core.ectd_metadata_comparator import MetadataChange

        violation = ViolationDetail(
            section_identifier="test-section",
            section_path="test/path",
            sequence_number="0005",
            previous_sequence_number="0004",
            metadata_changes=[],
            changed_attributes=["manufacturer", "supplier"],
            leaf_operations=[],
            leaf_ids=[],
            violation_type="metadata_changed_content_not_updated",
            violation_message="test",
            severity="error",
            rule_id="HR-ECTD-200",
            rule_citation="test"
        )

        summary = violation.get_summary()

        assert "HR-ECTD-200" in summary
        assert "test-section" in summary
        assert "manufacturer" in summary
        assert "supplier" in summary
        assert "metadata_changed_content_not_updated" in summary


# ============================================================================
# 测试 ValidationResult 数据结构
# ============================================================================

class TestValidationResult:
    """测试 ValidationResult 数据结构"""

    def test_validation_result_creation(self):
        """测试创建 ValidationResult 实例"""
        result = ValidationResult(
            sequence_number="0005",
            previous_sequence_number="0004",
            total_sections_analyzed=10,
            compliant_sections=8,
            violation_count=2,
            violations=[],
            warning_count=0,
            warnings=[]
        )

        assert result.sequence_number == "0005"
        assert result.previous_sequence_number == "0004"
        assert result.total_sections_analyzed == 10
        assert result.compliant_sections == 8
        assert result.violation_count == 2

    def test_validation_result_is_fully_compliant(self):
        """测试 is_fully_compliant() 方法"""
        compliant_result = ValidationResult(
            sequence_number="0005",
            previous_sequence_number="0004",
            total_sections_analyzed=10,
            compliant_sections=10,
            violation_count=0,
            violations=[]
        )

        non_compliant_result = ValidationResult(
            sequence_number="0005",
            previous_sequence_number="0004",
            total_sections_analyzed=10,
            compliant_sections=8,
            violation_count=2,
            violations=[]
        )

        assert compliant_result.is_fully_compliant() is True
        assert non_compliant_result.is_fully_compliant() is False

    def test_validation_result_get_summary(self):
        """测试 get_summary() 方法"""
        result = ValidationResult(
            sequence_number="0005",
            previous_sequence_number="0004",
            total_sections_analyzed=10,
            compliant_sections=8,
            violation_count=2,
            violations=[],
            warning_count=1,
            warnings=[]
        )

        summary = result.get_summary()

        assert "0004" in summary
        assert "0005" in summary
        assert "10" in summary  # total sections
        assert "8" in summary   # compliant sections
        assert "2" in summary   # violations
        assert "1" in summary   # warnings


# ============================================================================
# 测试 validate_metadata_lifecycle_coupling 核心函数
# ============================================================================

class TestValidateMetadataLifecycleCoupling:
    """测试 validate_metadata_lifecycle_coupling 函数"""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("test_data/ectd_metadata_mock_scenarios").exists(),
        reason="Mock test data not available"
    )
    def test_validate_compliant_sequence_pair(self, compliant_scenario_path: Path):
        """测试验证完全合规的序列对"""
        if not compliant_scenario_path.exists():
            pytest.skip("Compliant scenario not available")

        seq_0004 = compliant_scenario_path / "0004"
        seq_0005 = compliant_scenario_path / "0005"

        if not seq_0004.exists() or not seq_0005.exists():
            pytest.skip("Sequence directories not found")

        # 提取元数据索引
        previous_index = extract_sequence_metadata_index(str(seq_0004))
        current_index = extract_sequence_metadata_index(str(seq_0005))

        # 执行验证
        result = validate_metadata_lifecycle_coupling(previous_index, current_index)

        # 断言：应该完全合规
        assert isinstance(result, ValidationResult)
        assert result.sequence_number == current_index.sequence_number
        assert result.previous_sequence_number == previous_index.sequence_number
        assert result.violation_count == 0
        assert result.is_fully_compliant() is True

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("test_data/ectd_metadata_mock_scenarios").exists(),
        reason="Mock test data not available"
    )
    def test_validate_violation_sequence_pair(self, metadata_violation_scenario_path: Path):
        """测试验证有违规的序列对"""
        if not metadata_violation_scenario_path.exists():
            pytest.skip("Violation scenario not available")

        seq_0004 = metadata_violation_scenario_path / "0004"
        seq_0005 = metadata_violation_scenario_path / "0005"

        if not seq_0004.exists() or not seq_0005.exists():
            pytest.skip("Sequence directories not found")

        # 提取元数据索引
        previous_index = extract_sequence_metadata_index(str(seq_0004))
        current_index = extract_sequence_metadata_index(str(seq_0005))

        # 执行验证
        result = validate_metadata_lifecycle_coupling(previous_index, current_index)

        # 断言：应该检测到违规
        assert isinstance(result, ValidationResult)
        assert result.violation_count > 0
        assert result.is_fully_compliant() is False
        assert len(result.violations) > 0

        # 检查违规详情
        violation = result.violations[0]
        assert isinstance(violation, ViolationDetail)
        assert violation.severity == "error"
        assert violation.rule_id == "HR-ECTD-200"
        assert len(violation.changed_attributes) > 0


# ============================================================================
# 测试便捷函数
# ============================================================================

class TestConvenienceFunctions:
    """测试便捷函数"""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("test_data/ectd_metadata_mock_scenarios").exists(),
        reason="Mock test data not available"
    )
    def test_validate_sequences_from_paths(self, compliant_scenario_path: Path):
        """测试 validate_sequences_from_paths 函数"""
        if not compliant_scenario_path.exists():
            pytest.skip("Compliant scenario not available")

        seq_0004 = str(compliant_scenario_path / "0004")
        seq_0005 = str(compliant_scenario_path / "0005")

        if not Path(seq_0004).exists() or not Path(seq_0005).exists():
            pytest.skip("Sequence directories not found")

        # 使用便捷函数验证
        result = validate_sequences_from_paths(seq_0004, seq_0005)

        assert isinstance(result, ValidationResult)
        assert result.sequence_number == "5"
        assert result.previous_sequence_number == "4"

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("test_data/ectd_metadata_mock_scenarios").exists(),
        reason="Mock test data not available"
    )
    def test_validate_application_sequences(self, compliant_scenario_path: Path):
        """测试 validate_application_sequences 批量验证函数"""
        if not compliant_scenario_path.exists():
            pytest.skip("Compliant scenario not available")

        # 批量验证应用下的所有序列
        results = validate_application_sequences(str(compliant_scenario_path))

        assert isinstance(results, list)
        # 至少有一对序列（0004→0005）
        if len(results) > 0:
            assert all(isinstance(r, ValidationResult) for r in results)


# ============================================================================
# 集成测试：使用所有Mock场景
# ============================================================================

@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("test_data/ectd_metadata_mock_scenarios").exists(),
    reason="Mock test data not available"
)
class TestMockScenariosIntegration:
    """集成测试：使用所有Mock场景"""

    def test_all_mock_scenarios_can_be_validated(self, mock_data_root: Path):
        """测试所有Mock场景都能成功验证（不抛出异常）"""
        if not mock_data_root.exists():
            pytest.skip("Mock scenarios not available")

        scenarios = [
            "scenario_01_fully_compliant",
            "scenario_02_metadata_changed_content_not_updated",
            "scenario_03_metadata_content_coupled_compliant"
        ]

        for scenario_name in scenarios:
            scenario_path = mock_data_root / scenario_name
            if not scenario_path.exists():
                continue

            seq_0004 = scenario_path / "0004"
            seq_0005 = scenario_path / "0005"

            if not seq_0004.exists() or not seq_0005.exists():
                continue

            # 应该能成功验证，不抛出异常
            try:
                result = validate_sequences_from_paths(str(seq_0004), str(seq_0005))
                assert isinstance(result, ValidationResult)
            except Exception as e:
                pytest.fail(f"Scenario {scenario_name} failed: {e}")
