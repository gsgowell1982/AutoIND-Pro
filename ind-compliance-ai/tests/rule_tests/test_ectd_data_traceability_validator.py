"""
测试 eCTD数据可追溯性验证器

测试目标:
1. aCRF映射验证
2. 衍生变量可追溯性验证
3. 数据流程完整性验证
4. 违规检测准确性

版本: v1.0
创建日期: 2026-09-14
"""

import pytest
from typing import List, Dict

from core.ectd_data_traceability_validator import (
    DataTraceabilityValidator,
    ACRFAnnotation,
    DatasetVariable,
    DerivationMetadata,
    TraceabilityValidationResult,
    ViolationDetail,
    ViolationSeverity,
    validate_data_traceability
)


# ============================================================================
# 测试夹具：创建Mock数据
# ============================================================================

@pytest.fixture
def sample_raw_datasets() -> Dict[str, List[DatasetVariable]]:
    """样本原始数据集"""
    return {
        "DM": [
            DatasetVariable("DM", "STUDYID", "Study Identifier", "Char"),
            DatasetVariable("DM", "USUBJID", "Unique Subject Identifier", "Char"),
            DatasetVariable("DM", "AGE", "Age", "Num"),
            DatasetVariable("DM", "SEX", "Sex", "Char"),
            DatasetVariable("DM", "RACE", "Race", "Char"),
        ],
        "VS": [
            DatasetVariable("VS", "STUDYID", "Study Identifier", "Char"),
            DatasetVariable("VS", "USUBJID", "Unique Subject Identifier", "Char"),
            DatasetVariable("VS", "VSTESTCD", "Vital Signs Test Code", "Char"),
            DatasetVariable("VS", "VSORRES", "Result in Original Units", "Char"),
            DatasetVariable("VS", "VSORRESU", "Original Units", "Char"),
        ],
        "AE": [
            DatasetVariable("AE", "STUDYID", "Study Identifier", "Char"),
            DatasetVariable("AE", "USUBJID", "Unique Subject Identifier", "Char"),
            DatasetVariable("AE", "AETERM", "Reported Term for the AE", "Char"),
            DatasetVariable("AE", "AESTDTC", "Start Date/Time", "Char"),
        ]
    }


@pytest.fixture
def sample_analysis_datasets() -> Dict[str, List[DatasetVariable]]:
    """样本分析数据集"""
    return {
        "ADSL": [
            DatasetVariable("ADSL", "STUDYID", "Study Identifier", "Char"),
            DatasetVariable("ADSL", "USUBJID", "Unique Subject Identifier", "Char"),
            DatasetVariable("ADSL", "AGE", "Age", "Num"),
            DatasetVariable("ADSL", "AGEGR1", "Age Group 1", "Char",
                          is_derived=True, derivation_method="AGE categorized"),
            DatasetVariable("ADSL", "SEX", "Sex", "Char"),
            DatasetVariable("ADSL", "RACE", "Race", "Char"),
        ],
        "ADAE": [
            DatasetVariable("ADAE", "STUDYID", "Study Identifier", "Char"),
            DatasetVariable("ADAE", "USUBJID", "Unique Subject Identifier", "Char"),
            DatasetVariable("ADAE", "AEDECOD", "Dictionary-Derived Term", "Char",
                          is_derived=True, derivation_method="MedDRA coded"),
            DatasetVariable("ADAE", "AESEV", "Severity", "Char"),
        ]
    }


@pytest.fixture
def valid_acrf_annotations() -> List[ACRFAnnotation]:
    """有效的aCRF注释"""
    return [
        ACRFAnnotation("CRF-01", "AGE_FIELD", "DM", "AGE", "Subject age in years"),
        ACRFAnnotation("CRF-01", "SEX_FIELD", "DM", "SEX", "Subject sex"),
        ACRFAnnotation("CRF-02", "BP_SYS", "VS", "VSORRES", "Systolic BP"),
        ACRFAnnotation("CRF-03", "AE_TERM", "AE", "AETERM", "AE reported term"),
    ]


@pytest.fixture
def invalid_acrf_annotations() -> List[ACRFAnnotation]:
    """包含无效引用的aCRF注释"""
    return [
        ACRFAnnotation("CRF-01", "AGE_FIELD", "DM", "AGE"),
        ACRFAnnotation("CRF-01", "WEIGHT_FIELD", "DM", "WEIGHT"),  # 变量不存在
        ACRFAnnotation("CRF-02", "HEIGHT", "XX", "HEIGHT"),  # 数据集不存在
    ]


@pytest.fixture
def valid_derivation_metadata() -> List[DerivationMetadata]:
    """有效的衍生变量元数据"""
    return [
        DerivationMetadata(
            target_dataset="ADSL",
            target_variable="AGEGR1",
            source_datasets=["DM"],
            source_variables=["AGE"],
            derivation_algorithm="if AGE < 65 then AGEGR1='<65'; else AGEGR1='>=65'",
            program_file="adsl.sas",
            documentation="Age grouped into <65 and >=65"
        ),
        DerivationMetadata(
            target_dataset="ADAE",
            target_variable="AEDECOD",
            source_datasets=["AE"],
            source_variables=["AETERM"],
            derivation_algorithm="MedDRA coding of AETERM",
            program_file="adae.sas",
            documentation="AE terms coded using MedDRA dictionary"
        ),
    ]


@pytest.fixture
def incomplete_derivation_metadata() -> List[DerivationMetadata]:
    """不完整的衍生变量元数据"""
    return [
        DerivationMetadata(
            target_dataset="ADSL",
            target_variable="AGEGR1",
            source_datasets=["DM"],
            source_variables=["AGE"],
            # 缺少derivation_algorithm和documentation
        ),
        DerivationMetadata(
            target_dataset="ADAE",
            target_variable="AEDECOD",
            source_datasets=["AE"],
            source_variables=["NONEXIST"],  # 源变量不存在
            derivation_algorithm="Some algorithm",
        ),
    ]


# ============================================================================
# 测试 ACRFAnnotation 数据结构
# ============================================================================

class TestACRFAnnotation:
    """测试 ACRFAnnotation 数据结构"""

    def test_acrf_annotation_creation(self):
        """测试创建aCRF注释"""
        annotation = ACRFAnnotation(
            crf_page="CRF-01",
            crf_field="AGE_FIELD",
            dataset_name="DM",
            variable_name="AGE",
            annotation_text="Subject age"
        )

        assert annotation.crf_page == "CRF-01"
        assert annotation.crf_field == "AGE_FIELD"
        assert annotation.dataset_name == "DM"
        assert annotation.variable_name == "AGE"
        assert annotation.annotation_text == "Subject age"

    def test_get_mapping_key(self):
        """测试获取映射键"""
        annotation = ACRFAnnotation("CRF-01", "AGE", "DM", "AGE")
        mapping_key = annotation.get_mapping_key()

        assert mapping_key == "DM.AGE"


# ============================================================================
# 测试 aCRF映射验证
# ============================================================================

class TestACRFMappingValidation:
    """测试 aCRF映射验证"""

    def test_valid_acrf_mapping(self, valid_acrf_annotations, sample_raw_datasets):
        """测试有效的aCRF映射（无违规）"""
        validator = DataTraceabilityValidator()
        violations = validator.validate_acrf_mapping(
            valid_acrf_annotations,
            sample_raw_datasets
        )

        # 断言：无违规
        assert len(violations) == 0

    def test_acrf_references_nonexistent_dataset(self, sample_raw_datasets):
        """测试aCRF引用不存在的数据集"""
        validator = DataTraceabilityValidator()

        annotations = [
            ACRFAnnotation("CRF-01", "FIELD1", "NONEXIST", "VAR1")
        ]

        violations = validator.validate_acrf_mapping(annotations, sample_raw_datasets)

        # 断言：应检测到违规
        assert len(violations) == 1
        assert violations[0].rule_id == "TRACE-ACRF-001"
        assert violations[0].severity == ViolationSeverity.ERROR
        assert "不存在的数据集" in violations[0].message

    def test_acrf_references_nonexistent_variable(self, sample_raw_datasets):
        """测试aCRF引用不存在的变量"""
        validator = DataTraceabilityValidator()

        annotations = [
            ACRFAnnotation("CRF-01", "FIELD1", "DM", "NONEXIST")
        ]

        violations = validator.validate_acrf_mapping(annotations, sample_raw_datasets)

        # 断言：应检测到违规
        assert len(violations) == 1
        assert violations[0].rule_id == "TRACE-ACRF-002"
        assert violations[0].severity == ViolationSeverity.ERROR
        assert "不存在的变量" in violations[0].message

    def test_duplicate_crf_field_mapping(self, sample_raw_datasets):
        """测试重复的CRF字段映射（映射到多个变量）"""
        validator = DataTraceabilityValidator()

        annotations = [
            ACRFAnnotation("CRF-01", "AGE_FIELD", "DM", "AGE"),
            ACRFAnnotation("CRF-01", "AGE_FIELD", "ADSL", "AGE"),  # 同一字段映射到不同数据集
        ]

        # 需要提供ADSL数据集
        datasets = {
            **sample_raw_datasets,
            "ADSL": [DatasetVariable("ADSL", "AGE", "Age", "Num")]
        }

        violations = validator.validate_acrf_mapping(annotations, datasets)

        # 断言：应检测到警告
        assert len(violations) == 1
        assert violations[0].rule_id == "TRACE-ACRF-003"
        assert violations[0].severity == ViolationSeverity.WARNING
        assert "映射到多个不同的变量" in violations[0].message


# ============================================================================
# 测试 衍生变量可追溯性验证
# ============================================================================

class TestDerivationTraceabilityValidation:
    """测试 衍生变量可追溯性验证"""

    def test_valid_derivation_metadata(
        self,
        valid_derivation_metadata,
        sample_raw_datasets,
        sample_analysis_datasets
    ):
        """测试有效的衍生变量元数据（无违规）"""
        validator = DataTraceabilityValidator()
        violations = validator.validate_derivation_traceability(
            valid_derivation_metadata,
            sample_raw_datasets,
            sample_analysis_datasets
        )

        # 断言：无违规
        assert len(violations) == 0

    def test_derivation_missing_algorithm(
        self,
        sample_raw_datasets,
        sample_analysis_datasets
    ):
        """测试衍生变量缺少算法文档"""
        validator = DataTraceabilityValidator()

        metadata = [
            DerivationMetadata(
                target_dataset="ADSL",
                target_variable="AGEGR1",
                source_datasets=["DM"],
                source_variables=["AGE"],
                # 缺少derivation_algorithm和documentation
            )
        ]

        violations = validator.validate_derivation_traceability(
            metadata,
            sample_raw_datasets,
            sample_analysis_datasets
        )

        # 断言：应检测到违规
        assert len(violations) >= 1
        assert any(v.rule_id == "TRACE-DERIV-001" for v in violations)
        error_violation = next(v for v in violations if v.rule_id == "TRACE-DERIV-001")
        assert error_violation.severity == ViolationSeverity.ERROR
        assert "缺少算法文档" in error_violation.message

    def test_derivation_missing_program_code(
        self,
        sample_raw_datasets,
        sample_analysis_datasets
    ):
        """测试衍生变量缺少程序代码"""
        validator = DataTraceabilityValidator()

        metadata = [
            DerivationMetadata(
                target_dataset="ADSL",
                target_variable="AGEGR1",
                source_datasets=["DM"],
                source_variables=["AGE"],
                derivation_algorithm="AGE categorized",
                # 缺少program_file
            )
        ]

        violations = validator.validate_derivation_traceability(
            metadata,
            sample_raw_datasets,
            sample_analysis_datasets
        )

        # 断言：应检测到警告
        assert len(violations) >= 1
        assert any(v.rule_id == "TRACE-DERIV-002" for v in violations)
        warning_violation = next(v for v in violations if v.rule_id == "TRACE-DERIV-002")
        assert warning_violation.severity == ViolationSeverity.WARNING
        assert "缺少关联的程序代码" in warning_violation.message

    def test_derivation_references_nonexistent_source_variable(
        self,
        sample_raw_datasets,
        sample_analysis_datasets
    ):
        """测试衍生变量引用不存在的源变量"""
        validator = DataTraceabilityValidator()

        metadata = [
            DerivationMetadata(
                target_dataset="ADSL",
                target_variable="AGEGR1",
                source_datasets=["DM"],
                source_variables=["NONEXIST"],  # 不存在的变量
                derivation_algorithm="Some algorithm",
            )
        ]

        violations = validator.validate_derivation_traceability(
            metadata,
            sample_raw_datasets,
            sample_analysis_datasets
        )

        # 断言：应检测到违规
        assert len(violations) >= 1
        assert any(v.rule_id == "TRACE-DERIV-003" for v in violations)
        error_violation = next(v for v in violations if v.rule_id == "TRACE-DERIV-003")
        assert error_violation.severity == ViolationSeverity.ERROR
        assert "不存在的源变量" in error_violation.message

    def test_derivation_references_nonexistent_source_dataset(
        self,
        sample_raw_datasets,
        sample_analysis_datasets
    ):
        """测试衍生变量引用不存在的源数据集"""
        validator = DataTraceabilityValidator()

        metadata = [
            DerivationMetadata(
                target_dataset="ADSL",
                target_variable="AGEGR1",
                source_datasets=["NONEXIST"],  # 不存在的数据集
                source_variables=["AGE"],
                derivation_algorithm="Some algorithm",
            )
        ]

        violations = validator.validate_derivation_traceability(
            metadata,
            sample_raw_datasets,
            sample_analysis_datasets
        )

        # 断言：应检测到违规
        assert len(violations) >= 1
        assert any(v.rule_id == "TRACE-DERIV-004" for v in violations)
        error_violation = next(v for v in violations if v.rule_id == "TRACE-DERIV-004")
        assert error_violation.severity == ViolationSeverity.ERROR
        assert "不存在的源数据集" in error_violation.message


# ============================================================================
# 测试 综合验证
# ============================================================================

class TestDataFlowCompletenessValidation:
    """测试 数据流程完整性验证"""

    def test_fully_compliant_data_flow(
        self,
        valid_acrf_annotations,
        valid_derivation_metadata,
        sample_raw_datasets,
        sample_analysis_datasets
    ):
        """测试完全合规的数据流程"""
        validator = DataTraceabilityValidator()
        result = validator.validate_data_flow_completeness(
            valid_acrf_annotations,
            valid_derivation_metadata,
            sample_raw_datasets,
            sample_analysis_datasets
        )

        # 断言：完全合规
        assert result.is_fully_compliant() is True
        assert len(result.violations) == 0
        assert result.get_acrf_mapping_rate() == 100.0
        assert result.get_derivation_traceability_rate() == 100.0

    def test_partial_compliant_data_flow(
        self,
        invalid_acrf_annotations,
        incomplete_derivation_metadata,
        sample_raw_datasets,
        sample_analysis_datasets
    ):
        """测试部分合规的数据流程"""
        validator = DataTraceabilityValidator()
        result = validator.validate_data_flow_completeness(
            invalid_acrf_annotations,
            incomplete_derivation_metadata,
            sample_raw_datasets,
            sample_analysis_datasets
        )

        # 断言：存在违规
        assert result.is_fully_compliant() is False
        assert len(result.violations) > 0
        assert result.get_acrf_mapping_rate() < 100.0

    def test_traceability_validation_result_summary(
        self,
        valid_acrf_annotations,
        valid_derivation_metadata,
        sample_raw_datasets,
        sample_analysis_datasets
    ):
        """测试验证结果摘要"""
        result = validate_data_traceability(
            valid_acrf_annotations,
            valid_derivation_metadata,
            sample_raw_datasets,
            sample_analysis_datasets
        )

        summary = result.get_summary()

        # 断言：摘要包含关键信息
        assert "数据可追溯性验证结果" in summary
        assert "aCRF映射" in summary
        assert "衍生可追溯性" in summary
        assert "违规数" in summary


# ============================================================================
# 测试 便捷函数
# ============================================================================

class TestConvenienceFunctions:
    """测试 便捷函数"""

    def test_validate_data_traceability_function(
        self,
        valid_acrf_annotations,
        valid_derivation_metadata,
        sample_raw_datasets,
        sample_analysis_datasets
    ):
        """测试便捷验证函数"""
        result = validate_data_traceability(
            valid_acrf_annotations,
            valid_derivation_metadata,
            sample_raw_datasets,
            sample_analysis_datasets
        )

        # 断言：返回正确的结果类型
        assert isinstance(result, TraceabilityValidationResult)
        assert result.total_acrf_annotations == len(valid_acrf_annotations)
        assert result.total_derived_variables == len(valid_derivation_metadata)


# ============================================================================
# 测试 边界情况
# ============================================================================

class TestEdgeCases:
    """测试 边界情况"""

    def test_empty_acrf_annotations(self, sample_raw_datasets, sample_analysis_datasets):
        """测试空aCRF注释列表"""
        validator = DataTraceabilityValidator()
        result = validator.validate_data_flow_completeness(
            [],  # 空列表
            [],
            sample_raw_datasets,
            sample_analysis_datasets
        )

        # 断言：100%映射率（无注释即无未映射）
        assert result.get_acrf_mapping_rate() == 100.0
        assert result.total_acrf_annotations == 0

    def test_empty_derivation_metadata(
        self,
        valid_acrf_annotations,
        sample_raw_datasets,
        sample_analysis_datasets
    ):
        """测试空衍生变量元数据列表"""
        validator = DataTraceabilityValidator()
        result = validator.validate_data_flow_completeness(
            valid_acrf_annotations,
            [],  # 空列表
            sample_raw_datasets,
            sample_analysis_datasets
        )

        # 断言：100%可追溯率（无衍生变量即无不可追溯）
        assert result.get_derivation_traceability_rate() == 100.0
        assert result.total_derived_variables == 0

    def test_empty_datasets(self):
        """测试空数据集"""
        validator = DataTraceabilityValidator()

        annotations = [
            ACRFAnnotation("CRF-01", "FIELD1", "DM", "AGE")
        ]

        violations = validator.validate_acrf_mapping(annotations, {})

        # 断言：应检测到数据集不存在
        assert len(violations) == 1
        assert violations[0].rule_id == "TRACE-ACRF-001"
