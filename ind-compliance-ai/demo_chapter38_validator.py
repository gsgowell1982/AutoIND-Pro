"""
eCTD 3.8章节验证器演示脚本

展示验证器的主要功能
"""

import sys
import io
from pathlib import Path

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ectd_chapter38_validator import (
    ECTDChapter38Validator,
    ViolationSeverity,
    ValidationResult
)
from core.ectd_stf_format_validator import STFValidationContext
from core.ectd_china_data_validator import DatasetMetadata
from core.ectd_section_identifier import SectionIdentifier


def test_stf_naming():
    """测试STF文件命名验证"""
    print("\n=== 测试1: STF文件命名验证 ===")
    validator = ECTDChapter38Validator()

    # 测试有效命名
    context1 = STFValidationContext(
        file_path="stf-abc123.xml",
        module_path="m5/m5-3/m5-3-5/",
        study_id="abc123"
    )
    violations1 = validator.stf_validator.validate_stf_naming(context1)
    print(f"✓ 有效命名 'stf-abc123.xml': {len(violations1)} 个违规")

    # 测试无效命名
    context2 = STFValidationContext(
        file_path="abc123.xml",
        module_path="m5/m5-3/m5-3-5/"
    )
    violations2 = validator.stf_validator.validate_stf_naming(context2)
    print(f"✗ 无效命名 'abc123.xml': {len(violations2)} 个违规")
    if violations2:
        print(f"  - {violations2[0].message}")


def test_stf_structure():
    """测试STF结构验证"""
    print("\n=== 测试2: STF结构验证 ===")
    validator = ECTDChapter38Validator()

    # 有效结构
    valid_stf = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:study xmlns:ectd="http://www.ich.org/ectd" dtd-version="2.2">
    <study-identifier>
        <title>临床试验研究报告</title>
        <study-id>abc123</study-id>
        <category name="species" info-type="ich">mouse</category>
    </study-identifier>
    <study-document>
        <doc-content>
            <file-tag name="synopsis" info-type="ich"/>
        </doc-content>
    </study-document>
</ectd:study>"""

    context = STFValidationContext(
        file_path="stf-abc123.xml",
        module_path="m4/m4-2/",
        stf_content=valid_stf
    )

    violations = validator.stf_validator.validate_stf_structure(context)
    critical = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
    print(f"✓ 有效STF结构: {len(critical)} 个CRITICAL违规")

    # 无效结构（缺少study-identifier）
    invalid_stf = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:study xmlns:ectd="http://www.ich.org/ectd">
    <study-document/>
</ectd:study>"""

    context2 = STFValidationContext(
        file_path="stf-test.xml",
        module_path="m4/m4-2/",
        stf_content=invalid_stf
    )

    violations2 = validator.stf_validator.validate_stf_structure(context2)
    print(f"✗ 无效STF结构（缺少study-identifier）: {len(violations2)} 个违规")
    if violations2:
        print(f"  - {violations2[0].message}")


def test_category_validation():
    """测试category元素验证"""
    print("\n=== 测试3: Category元素验证 ===")
    validator = ECTDChapter38Validator()

    # 有效category
    valid_categories = [
        {'name': 'species', 'info-type': 'ich', 'value': 'mouse'},
        {'name': 'route-of-admin', 'info-type': 'ich', 'value': 'oral'}
    ]

    context = STFValidationContext(
        file_path="stf-test.xml",
        module_path="m4/m4-2/m4-2-3/m4-2-3-1/"
    )

    violations = validator.stf_validator.validate_category_elements(valid_categories, context)
    errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
    print(f"✓ 有效category值: {len(errors)} 个ERROR")

    # 无效category
    invalid_categories = [
        {'name': 'species', 'info-type': 'ich', 'value': 'elephant'}
    ]

    violations2 = validator.stf_validator.validate_category_elements(invalid_categories, context)
    print(f"✗ 无效species值 'elephant': {len(violations2)} 个违规")
    if violations2:
        print(f"  - {violations2[0].message}")


def test_file_tag_validation():
    """测试file-tag验证"""
    print("\n=== 测试4: File-tag验证 ===")
    validator = ECTDChapter38Validator()

    # 有效ICH标签
    valid_tags = [
        {'name': 'synopsis', 'info-type': 'ich'},
        {'name': 'study-report-body', 'info-type': 'ich'}
    ]

    context = STFValidationContext(
        file_path="stf-test.xml",
        module_path="m5/m5-3/m5-3-5/"
    )

    violations = validator.stf_validator.validate_file_tags(valid_tags, context)
    errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
    print(f"✓ 有效ICH file-tag: {len(errors)} 个ERROR")

    # 无效标签
    invalid_tags = [
        {'name': 'unknown-tag', 'info-type': 'ich'}
    ]

    violations2 = validator.stf_validator.validate_file_tags(invalid_tags, context)
    print(f"✗ 无效file-tag 'unknown-tag': {len(violations2)} 个违规")
    if violations2:
        print(f"  - {violations2[0].message}")


def test_dataset_naming():
    """测试数据集命名验证"""
    print("\n=== 测试5: 数据集命名验证（中国规范）===")
    validator = ECTDChapter38Validator()

    # 有效命名
    violations1 = validator.china_validator.validate_dataset_naming("ae", "ae.xpt")
    print(f"✓ 有效数据集名称 'ae': {len(violations1)} 个违规")

    # 无效命名（大写）
    violations2 = validator.china_validator.validate_dataset_naming("AE", "AE.xpt")
    print(f"✗ 无效数据集名称 'AE' (大写): {len(violations2)} 个违规")
    if violations2:
        print(f"  - {violations2[0].message}")

    # 超长命名
    violations3 = validator.china_validator.validate_dataset_naming("verylongname", "verylongname.xpt")
    print(f"✗ 超长数据集名称 'verylongname': {len(violations3)} 个违规")
    if violations3:
        print(f"  - {violations3[0].details}")


def test_variable_naming():
    """测试变量命名验证"""
    print("\n=== 测试6: 变量命名验证（中国规范）===")
    validator = ECTDChapter38Validator()

    # 有效命名
    violations1 = validator.china_validator.validate_variable_naming("STUDYID", "dm", "dm.xpt")
    print(f"✓ 有效变量名称 'STUDYID': {len(violations1)} 个违规")

    # 无效命名（小写）
    violations2 = validator.china_validator.validate_variable_naming("studyid", "dm", "dm.xpt")
    print(f"✗ 无效变量名称 'studyid' (小写): {len(violations2)} 个违规")
    if violations2:
        print(f"  - {violations2[0].message}")


def test_required_identifiers():
    """测试必需标识符验证"""
    print("\n=== 测试7: 必需标识符验证（中国规范）===")
    validator = ECTDChapter38Validator()

    # 完整标识符
    complete_dataset = DatasetMetadata(
        name="dm",
        label="人口学",
        variables=[
            {'name': 'STUDYID', 'label': '研究标识符', 'type': 'char'},
            {'name': 'USUBJID', 'label': '受试者唯一标识符', 'type': 'char'},
            {'name': 'SUBJID', 'label': '受试者标识符', 'type': 'char'},
        ]
    )

    violations1 = validator.china_validator.validate_required_identifiers(complete_dataset, "dm.xpt")
    critical_errors = [v for v in violations1 if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.ERROR]]
    print(f"✓ 完整的必需标识符: {len(critical_errors)} 个CRITICAL/ERROR")

    # 缺少SUBJID
    incomplete_dataset = DatasetMetadata(
        name="dm",
        label="人口学",
        variables=[
            {'name': 'STUDYID', 'label': '研究标识符', 'type': 'char'},
            {'name': 'USUBJID', 'label': '受试者唯一标识符', 'type': 'char'},
        ]
    )

    violations2 = validator.china_validator.validate_required_identifiers(incomplete_dataset, "dm.xpt")
    print(f"✗ 缺少SUBJID: {len(violations2)} 个违规")
    if violations2:
        print(f"  - {violations2[0].message}")


def test_chinese_labels():
    """测试中文标签验证"""
    print("\n=== 测试8: 中文标签验证（中国规范）===")
    validator = ECTDChapter38Validator()

    # 中文标签
    chinese_dataset = DatasetMetadata(
        name="dm",
        label="人口学数据",
        variables=[
            {'name': 'STUDYID', 'label': '研究标识符', 'type': 'char'},
            {'name': 'AGE', 'label': '年龄', 'type': 'num'},
        ]
    )

    violations1 = validator.china_validator.validate_labels_in_chinese(chinese_dataset, "dm.xpt")
    format_issues = [v for v in violations1 if v.severity == ViolationSeverity.WARNING and "格式" in v.message]
    print(f"✓ 中文标签: {len(format_issues)} 个格式问题")

    # 英文标签
    english_dataset = DatasetMetadata(
        name="dm",
        label="Demographics",
        variables=[
            {'name': 'STUDYID', 'label': 'Study ID', 'type': 'char'},
        ]
    )

    violations2 = validator.china_validator.validate_labels_in_chinese(english_dataset, "dm.xpt")
    print(f"✗ 英文标签: {len(violations2)} 个违规")
    if violations2:
        chinese_related = [v for v in violations2 if "中文" in v.details]
        print(f"  - {len(chinese_related)} 个标签缺少中文")


def test_report_generation():
    """测试报告生成"""
    print("\n=== 测试9: 报告生成 ===")
    validator = ECTDChapter38Validator()

    from core.ectd_chapter38_validator import ViolationDetail

    violations = [
        ViolationDetail(
            rule_id="3.8.1",
            severity=ViolationSeverity.CRITICAL,
            message="STF文件命名不符合规范",
            location="test.xml",
            details="文件名必须以stf-开头",
            suggestion="重命名为stf-{study-id}.xml"
        ),
        ViolationDetail(
            rule_id="3.8.10",
            severity=ViolationSeverity.ERROR,
            message="数据集名称使用大写",
            location="AE.xpt",
            details="数据集名称应使用小写字母",
            suggestion="改为ae.xpt"
        ),
        ViolationDetail(
            rule_id="3.8.13",
            severity=ViolationSeverity.WARNING,
            message="变量标签缺少中文",
            location="dm.xpt:STUDYID",
            details="变量标签应使用中文",
            suggestion="添加中文标签"
        ),
    ]

    result = ValidationResult.from_violations(violations)

    print(f"\n验证结果:")
    print(f"  - 总违规数: {result.total_violations}")
    print(f"  - CRITICAL: {result.critical_count}")
    print(f"  - ERROR: {result.error_count}")
    print(f"  - WARNING: {result.warning_count}")
    print(f"  - 通过: {result.passed}")

    # 生成文本报告
    print("\n--- 文本报告预览 ---")
    text_report = validator.generate_validation_report(result, "text")
    print(text_report[:500] + "...\n")


def main():
    """运行所有测试"""
    print("=" * 80)
    print("eCTD 3.8章节验证器功能演示")
    print("=" * 80)

    try:
        test_stf_naming()
        test_stf_structure()
        test_category_validation()
        test_file_tag_validation()
        test_dataset_naming()
        test_variable_naming()
        test_required_identifiers()
        test_chinese_labels()
        test_report_generation()

        print("\n" + "=" * 80)
        print("✓ 所有演示测试完成")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
