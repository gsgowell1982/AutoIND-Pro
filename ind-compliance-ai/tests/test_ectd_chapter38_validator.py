"""
eCTD 3.8章节验证器单元测试
"""

import pytest
from core.ectd_chapter38_validator import (
    ECTDChapter38Validator,
    ViolationSeverity,
    ValidationResult
)
from core.ectd_stf_format_validator import STFValidationContext
from core.ectd_china_data_validator import DatasetMetadata
from core.ectd_section_identifier import SectionIdentifier


class TestSTFFormatValidation:
    """STF格式验证测试"""

    def test_valid_stf_naming(self):
        """测试有效的STF文件命名"""
        validator = ECTDChapter38Validator()
        context = STFValidationContext(
            file_path="stf-abc123.xml",
            module_path="m5/m5-3/m5-3-5/",
            study_id="abc123"
        )

        violations = validator.stf_validator.validate_stf_naming(context)
        assert len(violations) == 0

    def test_invalid_stf_naming_no_prefix(self):
        """测试无效的STF命名（缺少stf-前缀）"""
        validator = ECTDChapter38Validator()
        context = STFValidationContext(
            file_path="abc123.xml",
            module_path="m5/m5-3/m5-3-5/"
        )

        violations = validator.stf_validator.validate_stf_naming(context)
        assert len(violations) > 0
        assert violations[0].severity == ViolationSeverity.ERROR
        assert "stf-" in violations[0].details.lower()

    def test_stf_naming_mismatch_study_id(self):
        """测试STF文件名与study-id不匹配"""
        validator = ECTDChapter38Validator()
        context = STFValidationContext(
            file_path="stf-abc123.xml",
            module_path="m5/m5-3/m5-3-5/",
            study_id="xyz789"
        )

        violations = validator.stf_validator.validate_stf_naming(context)
        assert len(violations) > 0
        assert any("不匹配" in v.message for v in violations)

    def test_stf_structure_valid(self):
        """测试有效的STF结构"""
        validator = ECTDChapter38Validator()
        stf_content = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:study xmlns:ectd="http://www.ich.org/ectd" dtd-version="2.2">
    <study-identifier>
        <title>Test Study</title>
        <study-id>abc123</study-id>
        <category name="species" info-type="ich">mouse</category>
    </study-identifier>
    <study-document>
        <doc-content xlink:href="index.xml#a101">
            <file-tag name="synopsis" info-type="ich"/>
        </doc-content>
    </study-document>
</ectd:study>"""

        context = STFValidationContext(
            file_path="stf-abc123.xml",
            module_path="m4/m4-2/",
            stf_content=stf_content
        )

        violations = validator.stf_validator.validate_stf_structure(context)
        # 应该没有CRITICAL级别的错误
        critical = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        assert len(critical) == 0

    def test_stf_missing_study_identifier(self):
        """测试缺少study-identifier元素"""
        validator = ECTDChapter38Validator()
        stf_content = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:study xmlns:ectd="http://www.ich.org/ectd">
    <study-document/>
</ectd:study>"""

        context = STFValidationContext(
            file_path="stf-test.xml",
            module_path="m4/m4-2/",
            stf_content=stf_content
        )

        violations = validator.stf_validator.validate_stf_structure(context)
        assert any("study-identifier" in v.message for v in violations)
        assert any(v.severity == ViolationSeverity.CRITICAL for v in violations)

    def test_category_validation_valid_species(self):
        """测试有效的species category值"""
        validator = ECTDChapter38Validator()
        categories = [
            {'name': 'species', 'info-type': 'ich', 'value': 'mouse'},
            {'name': 'route-of-admin', 'info-type': 'ich', 'value': 'oral'}
        ]

        context = STFValidationContext(
            file_path="stf-test.xml",
            module_path="m4/m4-2/m4-2-3/m4-2-3-1/"
        )

        violations = validator.stf_validator.validate_category_elements(categories, context)
        # 有效值不应产生ERROR级别违规
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) == 0

    def test_category_validation_invalid_species(self):
        """测试无效的species值"""
        validator = ECTDChapter38Validator()
        categories = [
            {'name': 'species', 'info-type': 'ich', 'value': 'elephant'}  # 无效值
        ]

        context = STFValidationContext(
            file_path="stf-test.xml",
            module_path="m4/m4-2/m4-2-3/m4-2-3-1/"
        )

        violations = validator.stf_validator.validate_category_elements(categories, context)
        assert len(violations) > 0
        assert any("species" in v.message.lower() for v in violations)

    def test_file_tag_validation_valid_ich(self):
        """测试有效的ICH file-tag"""
        validator = ECTDChapter38Validator()
        file_tags = [
            {'name': 'synopsis', 'info-type': 'ich'},
            {'name': 'study-report-body', 'info-type': 'ich'}
        ]

        context = STFValidationContext(
            file_path="stf-test.xml",
            module_path="m5/m5-3/m5-3-5/"
        )

        violations = validator.stf_validator.validate_file_tags(file_tags, context)
        # 标准ICH标签不应产生ERROR
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) == 0

    def test_file_tag_validation_invalid(self):
        """测试无效的file-tag"""
        validator = ECTDChapter38Validator()
        file_tags = [
            {'name': 'invalid-tag-name', 'info-type': 'ich'}
        ]

        context = STFValidationContext(
            file_path="stf-test.xml",
            module_path="m5/m5-3/m5-3-5/"
        )

        violations = validator.stf_validator.validate_file_tags(file_tags, context)
        assert len(violations) > 0
        assert any("不在标准列表" in v.message for v in violations)

    def test_property_validation_crf_needs_site_identifier(self):
        """测试CRF需要site-identifier property"""
        validator = ECTDChapter38Validator()
        properties = []  # 缺少site-identifier

        context = STFValidationContext(
            file_path="stf-test.xml",
            module_path="m5/m5-3/m5-3-5/"
        )

        violations = validator.stf_validator.validate_property_elements(
            'case-report-forms',
            properties,
            context
        )

        assert len(violations) > 0
        assert any("site-identifier" in v.message.lower() for v in violations)


class TestModuleExemptionValidation:
    """模块豁免规则测试"""

    def test_exempted_module_without_stf(self):
        """测试豁免模块不使用STF（正常）"""
        validator = ECTDChapter38Validator()

        # 创建一个5.2模块的section
        section_id = SectionIdentifier(
            element_name="m5-2-clinical-overview",
            attributes={}
        )

        result = validator.validate_module_stf_usage(section_id, has_stf=False)
        # 豁免模块不使用STF是允许的，不应有ERROR
        errors = [v for v in result.violations if v.severity in [ViolationSeverity.ERROR, ViolationSeverity.CRITICAL]]
        assert len(errors) == 0

    def test_required_module_without_stf(self):
        """测试必需STF的模块缺少STF"""
        validator = ECTDChapter38Validator()

        # 模拟一个5.3.5模块的研究报告章节
        section_id = SectionIdentifier(
            element_name="m5-3-5-1-study-report",
            attributes={'tag': 'clinical-study-report'}
        )

        result = validator.validate_module_stf_usage(section_id, has_stf=False)
        # 应该产生ERROR
        assert any(v.severity == ViolationSeverity.ERROR for v in result.violations)


class TestChinaDataValidation:
    """中国数据递交规范测试"""

    def test_valid_dataset_naming(self):
        """测试有效的数据集命名"""
        validator = ECTDChapter38Validator()

        violations = validator.china_validator.validate_dataset_naming(
            "ae",
            "ae.xpt"
        )

        assert len(violations) == 0

    def test_invalid_dataset_naming_uppercase(self):
        """测试无效的数据集命名（大写字母）"""
        validator = ECTDChapter38Validator()

        violations = validator.china_validator.validate_dataset_naming(
            "AE",  # 应为小写
            "AE.xpt"
        )

        assert len(violations) > 0
        assert any(v.severity == ViolationSeverity.ERROR for v in violations)

    def test_invalid_dataset_naming_too_long(self):
        """测试数据集名称超过8字节"""
        validator = ECTDChapter38Validator()

        violations = validator.china_validator.validate_dataset_naming(
            "verylongname",  # 超过8字节
            "verylongname.xpt"
        )

        assert len(violations) > 0
        assert any("8个字节" in v.details for v in violations)

    def test_valid_variable_naming(self):
        """测试有效的变量命名"""
        validator = ECTDChapter38Validator()

        violations = validator.china_validator.validate_variable_naming(
            "STUDYID",
            "dm",
            "dm.xpt"
        )

        assert len(violations) == 0

    def test_invalid_variable_naming_lowercase(self):
        """测试无效的变量命名（小写）"""
        validator = ECTDChapter38Validator()

        violations = validator.china_validator.validate_variable_naming(
            "studyid",  # 应为大写
            "dm",
            "dm.xpt"
        )

        assert len(violations) > 0

    def test_required_identifiers_dm_dataset(self):
        """测试dm数据集必需标识符"""
        validator = ECTDChapter38Validator()

        dataset = DatasetMetadata(
            name="dm",
            label="人口学",
            variables=[
                {'name': 'STUDYID', 'label': '研究标识符', 'type': 'char'},
                {'name': 'USUBJID', 'label': '受试者唯一标识符', 'type': 'char'},
                # 缺少SUBJID
            ]
        )

        violations = validator.china_validator.validate_required_identifiers(
            dataset,
            "dm.xpt"
        )

        assert len(violations) > 0
        assert any("SUBJID" in v.message for v in violations)

    def test_required_identifiers_complete(self):
        """测试完整的必需标识符"""
        validator = ECTDChapter38Validator()

        dataset = DatasetMetadata(
            name="dm",
            label="人口学",
            variables=[
                {'name': 'STUDYID', 'label': '研究标识符', 'type': 'char'},
                {'name': 'USUBJID', 'label': '受试者唯一标识符', 'type': 'char'},
                {'name': 'SUBJID', 'label': '受试者标识符', 'type': 'char'},
            ]
        )

        violations = validator.china_validator.validate_required_identifiers(
            dataset,
            "dm.xpt"
        )

        # 不应有CRITICAL或ERROR级别的违规
        critical_errors = [v for v in violations if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.ERROR]]
        assert len(critical_errors) == 0

    def test_labels_in_chinese(self):
        """测试中文标签"""
        validator = ECTDChapter38Validator()

        dataset = DatasetMetadata(
            name="dm",
            label="人口学数据",  # 中文标签
            variables=[
                {'name': 'STUDYID', 'label': '研究标识符', 'type': 'char'},
                {'name': 'AGE', 'label': '年龄', 'type': 'num'},
            ]
        )

        violations = validator.china_validator.validate_labels_in_chinese(
            dataset,
            "dm.xpt"
        )

        # 中文标签应该通过，不应有WARNING（除非格式有问题）
        warnings = [v for v in violations if v.severity == ViolationSeverity.WARNING and "标签格式" in v.message]
        assert len(warnings) == 0

    def test_labels_without_chinese(self):
        """测试缺少中文标签"""
        validator = ECTDChapter38Validator()

        dataset = DatasetMetadata(
            name="dm",
            label="Demographics",  # 英文标签
            variables=[
                {'name': 'STUDYID', 'label': 'Study ID', 'type': 'char'},
            ]
        )

        violations = validator.china_validator.validate_labels_in_chinese(
            dataset,
            "dm.xpt"
        )

        assert len(violations) > 0
        assert any("中文" in v.details for v in violations)

    def test_xpt_format_valid(self):
        """测试有效的XPT格式"""
        validator = ECTDChapter38Validator()

        violations = validator.china_validator.validate_xpt_format(
            "ae.xpt",
            "ae",
            "utf-8"
        )

        assert len(violations) == 0

    def test_xpt_format_wrong_extension(self):
        """测试错误的文件扩展名"""
        validator = ECTDChapter38Validator()

        violations = validator.china_validator.validate_xpt_format(
            "ae.sas7bdat",  # 错误的扩展名
            "ae"
        )

        assert len(violations) > 0
        assert any(".xpt" in v.details for v in violations)

    def test_xpt_filename_mismatch(self):
        """测试文件名与数据集名称不一致"""
        validator = ECTDChapter38Validator()

        violations = validator.china_validator.validate_xpt_format(
            "adverse_events.xpt",  # 文件名不匹配
            "ae"  # 数据集名称
        )

        assert len(violations) > 0
        assert any("不一致" in v.message for v in violations)


class TestIntegratedValidation:
    """集成验证测试"""

    def test_validation_result_from_violations(self):
        """测试从违规列表创建验证结果"""
        from core.ectd_chapter38_validator import ViolationDetail

        violations = [
            ViolationDetail(
                rule_id="3.8.1",
                severity=ViolationSeverity.CRITICAL,
                message="Critical issue",
                location="test",
                details="Details",
                suggestion="Fix it"
            ),
            ViolationDetail(
                rule_id="3.8.2",
                severity=ViolationSeverity.ERROR,
                message="Error issue",
                location="test",
                details="Details",
                suggestion="Fix it"
            ),
            ViolationDetail(
                rule_id="3.8.3",
                severity=ViolationSeverity.WARNING,
                message="Warning issue",
                location="test",
                details="Details",
                suggestion="Consider fixing"
            ),
        ]

        result = ValidationResult.from_violations(violations)

        assert result.total_violations == 3
        assert result.critical_count == 1
        assert result.error_count == 1
        assert result.warning_count == 1
        assert result.passed is False  # 有CRITICAL和ERROR

    def test_validation_result_passed(self):
        """测试通过验证的结果"""
        from core.ectd_chapter38_validator import ViolationDetail

        violations = [
            ViolationDetail(
                rule_id="3.8.1",
                severity=ViolationSeverity.INFO,
                message="Info",
                location="test",
                details="Details",
                suggestion="Optional"
            ),
        ]

        result = ValidationResult.from_violations(violations)

        assert result.passed is True  # 只有INFO，视为通过
        assert result.info_count == 1

    def test_report_generation_text(self):
        """测试文本格式报告生成"""
        validator = ECTDChapter38Validator()

        from core.ectd_chapter38_validator import ViolationDetail

        violations = [
            ViolationDetail(
                rule_id="3.8.1",
                severity=ViolationSeverity.ERROR,
                message="Test error",
                location="test.xml",
                details="Error details",
                suggestion="Fix suggestion"
            ),
        ]

        result = ValidationResult.from_violations(violations)
        report = validator.generate_validation_report(result, "text")

        assert "eCTD 3.8章节验证报告" in report
        assert "总违规数: 1" in report
        assert "ERROR" in report
        assert "test.xml" in report

    def test_report_generation_markdown(self):
        """测试Markdown格式报告生成"""
        validator = ECTDChapter38Validator()

        from core.ectd_chapter38_validator import ViolationDetail

        violations = [
            ViolationDetail(
                rule_id="3.8.1",
                severity=ViolationSeverity.WARNING,
                message="Test warning",
                location="test.xml",
                details="Warning details",
                suggestion="Consider fixing"
            ),
        ]

        result = ValidationResult.from_violations(violations)
        report = validator.generate_validation_report(result, "markdown")

        assert "# eCTD 3.8章节验证报告" in report
        assert "## 验证摘要" in report
        assert "⚠️" in report  # Warning icon
        assert "`test.xml`" in report

    def test_report_generation_json(self):
        """测试JSON格式报告生成"""
        import json
        validator = ECTDChapter38Validator()

        from core.ectd_chapter38_validator import ViolationDetail

        violations = [
            ViolationDetail(
                rule_id="3.8.1",
                severity=ViolationSeverity.INFO,
                message="Test info",
                location="test.xml",
                details="Info details",
                suggestion="Optional action"
            ),
        ]

        result = ValidationResult.from_violations(violations)
        report = validator.generate_validation_report(result, "json")

        # 验证JSON可解析
        data = json.loads(report)
        assert data["summary"]["total_violations"] == 1
        assert data["summary"]["info_count"] == 1
        assert len(data["violations"]) == 1
        assert data["violations"][0]["rule_id"] == "3.8.1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
