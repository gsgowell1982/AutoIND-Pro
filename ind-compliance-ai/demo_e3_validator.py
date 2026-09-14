#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICH E3结构验证器演示脚本

展示ICH E3临床研究报告结构验证的各种场景
"""

import sys
import io

# 设置UTF-8输出（Windows兼容）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.ectd_chapter38_validator import ECTDChapter38Validator, DatasetMetadata


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(passed: bool, message: str):
    """打印测试结果"""
    status = "✓ 通过" if passed else "✗ 失败"
    print(f"{status}: {message}")


def demo_e3_complete_structure():
    """演示场景1: 完整的E3结构验证"""
    print_section("场景1: 完整的E3结构 - 应该只有轻微警告")

    validator = ECTDChapter38Validator()

    # 模拟一个相对完整的E3报告TOC
    complete_toc = {
        "1": "Title Page",
        "2": "Synopsis",
        "3": "Table of Contents",
        "4": "List of Abbreviations and Definition of Terms",
        "5": "Ethics",
        "5.1": "IEC/IRB",
        "5.2": "Ethical Conduct of the Study",
        "5.3": "Patient Information and Consent",
        "6": "Investigators and Study Administrative Structure",
        "7": "Introduction",
        "8": "Study Objectives",
        "9": "Investigational Plan",
        "9.1": "Overall Study Design and Plan",
        "9.2": "Discussion of Study Design",
        "9.3": "Selection of Study Population",
        "9.4": "Treatment",
        "9.5": "Efficacy and Safety Variables",
        "9.6": "Data Quality Assurance",
        "9.7": "Statistical Methods",
        "10": "Study Patients",
        "10.1": "Disposition of Patients",
        "10.2": "Protocol Deviations",
        "11": "Efficacy Evaluation",
        "11.1": "Datasets Analysed",
        "11.2": "Demographic and Other Baseline Characteristics",
        "11.3": "Measurements of Treatment Compliance",
        "11.4": "Efficacy Results and Tabulations of Individual Subject Data",
        "12": "Safety Evaluation",
        "12.1": "Extent of Exposure",
        "12.2": "Adverse Events",
        "12.3": "Deaths, Other Serious Adverse Events, and Other Significant Adverse Events",
        "12.4": "Clinical Laboratory Evaluations",
        "12.5": "Vital Signs, Physical Findings, and Other Observations",
        "12.6": "Safety Conclusions",
        "13": "Discussion and Overall Conclusions",
        "14": "Tables, Figures and Graphs",
        "14.1": "Demographic Data",
        "14.2": "Efficacy Data",
        "14.3": "Safety Data",
        "15": "Reference List",
        "16": "Appendices",
        "16.1": "Study Information",
        "16.1.1": "Protocol and Protocol Amendments",
        "16.1.2": "Sample Case Report Form",
        "16.1.3": "List of IECs/IRBs and Informed Consent Forms",
        "16.1.4": "List of Investigators",
        "16.1.5": "Signature of Responsible Parties",
        "16.1.9": "Documentation of Statistical Methods",
        "16.2": "Patient Data Listings",
        "16.2.1": "Listing of Patients Who Discontinued",
        "16.2.2": "Protocol Deviations",
        "16.2.3": "Patients Excluded from Efficacy Analysis",
        "16.2.4": "Demographic Data",
        "16.2.6": "Individual Efficacy Response Data",
        "16.2.7": "Listing of Adverse Events by Patient",
    }

    result = validator.validate_e3_clinical_report(
        report_file_path="study-001-csr.pdf",
        toc_data=complete_toc
    )

    print(f"\n总违规数: {result.total_violations}")
    print(f"  - CRITICAL: {result.critical_count}")
    print(f"  - ERROR: {result.error_count}")
    print(f"  - WARNING: {result.warning_count}")
    print(f"  - INFO: {result.info_count}")

    print_result(result.passed, f"完整E3结构验证 (无CRITICAL/ERROR)")

    if result.violations:
        print("\n发现的问题:")
        for v in result.violations[:3]:  # 只显示前3个
            print(f"  - {v.severity.value}: {v.message}")


def demo_e3_missing_sections():
    """演示场景2: 缺少必需章节"""
    print_section("场景2: 缺少必需E3章节 - 应该报告ERROR")

    validator = ECTDChapter38Validator()

    # 模拟一个不完整的E3报告TOC（缺少多个必需章节）
    incomplete_toc = {
        "1": "Title Page",
        "2": "Synopsis",
        "3": "Table of Contents",
        # 缺少章节4-8
        "9": "Investigational Plan",
        "10": "Study Patients",
        # 缺少章节11-12（疗效和安全性评价）
        "13": "Discussion and Overall Conclusions",
        "16": "Appendices",
    }

    result = validator.validate_e3_clinical_report(
        report_file_path="study-002-incomplete.pdf",
        toc_data=incomplete_toc
    )

    print(f"\n总违规数: {result.total_violations}")
    print(f"  - ERROR: {result.error_count}")

    print_result(not result.passed, f"检测到缺少必需章节 (有ERROR)")

    if result.violations:
        print("\n发现的关键问题:")
        error_violations = [v for v in result.violations if v.severity.value == "ERROR"]
        for v in error_violations[:5]:  # 显示前5个ERROR
            print(f"  - ERROR: {v.message}")


def demo_e3_page_length_validation():
    """演示场景3: 章节长度验证"""
    print_section("场景3: Synopsis和Introduction长度验证")

    validator = ECTDChapter38Validator()

    minimal_toc = {
        "1": "Title Page",
        "2": "Synopsis",
        "7": "Introduction",
    }

    # 测试Synopsis过长（>3页）
    result1 = validator.validate_e3_clinical_report(
        report_file_path="study-003-long-synopsis.pdf",
        toc_data=minimal_toc,
        synopsis_pages=5
    )

    synopsis_violation = [v for v in result1.violations if "Synopsis" in v.message]
    if synopsis_violation:
        print(f"\nSynopsis长度检查:")
        print(f"  - {synopsis_violation[0].severity.value}: {synopsis_violation[0].message}")
        print_result(True, "检测到Synopsis过长（5页 > 3页）")

    # 测试Introduction过长（>1页）
    result2 = validator.validate_e3_clinical_report(
        report_file_path="study-004-long-intro.pdf",
        toc_data=minimal_toc,
        introduction_pages=3
    )

    intro_violation = [v for v in result2.violations if "Introduction" in v.message]
    if intro_violation:
        print(f"\nIntroduction长度检查:")
        print(f"  - {intro_violation[0].severity.value}: {intro_violation[0].message}")
        print_result(True, "检测到Introduction过长（3页 > 1页）")


def demo_e3_stf_consistency():
    """演示场景4: STF file-tags与E3结构的一致性"""
    print_section("场景4: STF file-tags与E3章节一致性验证")

    validator = ECTDChapter38Validator()

    # 完整的TOC
    complete_toc = {
        "16": "Appendices",
        "16.1": "Study Information",
        "16.1.1": "Protocol and Protocol Amendments",
        "16.1.2": "Sample Case Report Form",
        "16.1.3": "List of IECs/IRBs and Informed Consent Forms",
        "16.1.4": "List of Investigators",
        "16.1.9": "Documentation of Statistical Methods",
        "16.2": "Patient Data Listings",
        "16.2.7": "Listing of Adverse Events by Patient",
    }

    # STF声明了这些file-tags
    stf_file_tags = [
        "protocol",
        "sample-crf",
        "informed-consent-form",
        "investigator-list",
        "statistical-analysis-plan",
        "adverse-event-listings"
    ]

    result = validator.validate_e3_clinical_report(
        report_file_path="study-005-with-stf.pdf",
        toc_data=complete_toc,
        stf_file_tags=stf_file_tags
    )

    print(f"\n总违规数: {result.total_violations}")
    print_result(result.passed, "STF file-tags与E3章节一致")

    # 测试不一致情况：STF声明但E3章节缺失
    incomplete_toc = {
        "16": "Appendices",
        "16.1": "Study Information",
        # 缺少16.1.1 (protocol)
        # 缺少16.1.2 (sample-crf)
    }

    result2 = validator.validate_e3_clinical_report(
        report_file_path="study-006-inconsistent.pdf",
        toc_data=incomplete_toc,
        stf_file_tags=["protocol", "sample-crf"]
    )

    if result2.violations:
        print(f"\n不一致检测:")
        print(f"  - 总违规数: {result2.total_violations}")
        for v in result2.violations[:2]:
            print(f"  - {v.severity.value}: {v.message}")
        print_result(True, "检测到STF与E3结构不一致")


def demo_e3_section_numbering():
    """演示场景5: 章节编号规范性验证"""
    print_section("场景5: 章节编号规范性验证")

    validator = ECTDChapter38Validator()

    # 包含无效编号的TOC
    invalid_toc = {
        "1": "Title Page",
        "2": "Synopsis",
        "3": "Table of Contents",
        "3a": "Additional Contents",  # 无效格式
        "17": "Extra Section",  # 超出1-16范围
        "9.1.a": "Invalid Subsection",  # 无效子章节格式
    }

    result = validator.validate_e3_clinical_report(
        report_file_path="study-007-invalid-numbering.pdf",
        toc_data=invalid_toc
    )

    numbering_violations = [v for v in result.violations if "number" in v.rule_id.lower() or "NUMBER" in v.rule_id]

    print(f"\n编号问题数: {len(numbering_violations)}")
    if numbering_violations:
        print("\n发现的编号问题:")
        for v in numbering_violations:
            print(f"  - {v.severity.value}: {v.message}")
        print_result(True, "检测到无效的章节编号")


def demo_comprehensive_validation():
    """演示场景6: 综合验证（STF + E3 + 数据集）"""
    print_section("场景6: 综合验证 - STF + E3报告 + 数据集")

    validator = ECTDChapter38Validator()

    # 准备E3报告TOC
    report_toc = {
        "1": "Title Page",
        "2": "Synopsis",
        "3": "Table of Contents",
        "11": "Efficacy Evaluation",
        "12": "Safety Evaluation",
        "16": "Appendices",
        "16.1": "Study Information",
        "16.2": "Patient Data Listings",
    }

    # 准备数据集元数据
    dm_dataset = DatasetMetadata(
        name="dm",
        label="人口学数据",
        is_analysis=False,
        variables=[
            {"name": "STUDYID", "label": "研究标识符", "type": "char"},
            {"name": "USUBJID", "label": "受试者唯一标识符", "type": "char"},
            {"name": "SUBJID", "label": "受试者标识符", "type": "char"},
            {"name": "AGE", "label": "年龄", "type": "num"},
        ]
    )

    ae_dataset = DatasetMetadata(
        name="ae",
        label="不良事件",
        is_analysis=False,
        variables=[
            {"name": "STUDYID", "label": "研究标识符", "type": "char"},
            {"name": "USUBJID", "label": "受试者唯一标识符", "type": "char"},
            {"name": "AETERM", "label": "不良事件术语", "type": "char"},
        ]
    )

    adsl_dataset = DatasetMetadata(
        name="adsl",
        label="受试者水平分析数据",
        is_analysis=True,
        variables=[
            {"name": "STUDYID", "label": "研究标识符", "type": "char"},
            {"name": "USUBJID", "label": "受试者唯一标识符", "type": "char"},
            {"name": "SAFFL", "label": "安全性分析集标志", "type": "char"},
        ]
    )

    # 执行综合验证
    result = validator.validate_comprehensive(
        stf_file_path=None,  # 这里我们不测试实际文件
        report_file_path="study-008-comprehensive.pdf",
        report_toc=report_toc,
        dataset_files=["dm.xpt", "ae.xpt", "adsl.xpt"],
        dataset_metadata_list=[dm_dataset, ae_dataset, adsl_dataset],
        module_path="m5/m5-3/m5-3-5/m5-3-5-1",
        validate_china_rules=True
    )

    print(f"\n综合验证结果:")
    print(f"  - 总违规数: {result.total_violations}")
    print(f"  - CRITICAL: {result.critical_count}")
    print(f"  - ERROR: {result.error_count}")
    print(f"  - WARNING: {result.warning_count}")
    print(f"  - INFO: {result.info_count}")
    print(f"  - 验证状态: {'通过' if result.passed else '未通过'}")

    print_result(True, "综合验证执行完成")


def demo_markdown_report():
    """演示场景7: Markdown格式报告生成"""
    print_section("场景7: Markdown格式报告生成")

    validator = ECTDChapter38Validator()

    # 创建一些违规
    incomplete_toc = {
        "1": "Title Page",
        "2": "Synopsis",
    }

    result = validator.validate_e3_clinical_report(
        report_file_path="study-009-report.pdf",
        toc_data=incomplete_toc,
        synopsis_pages=4
    )

    # 生成Markdown报告
    markdown_report = validator.generate_validation_report(result, output_format="markdown")

    print("\n生成的Markdown报告预览（前500字符）:")
    print("-" * 80)
    print(markdown_report[:500])
    print("...")
    print("-" * 80)

    print_result(True, "Markdown报告生成成功")


def main():
    """主函数"""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  ICH E3结构验证器 - 功能演示".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    try:
        demo_e3_complete_structure()
        demo_e3_missing_sections()
        demo_e3_page_length_validation()
        demo_e3_stf_consistency()
        demo_e3_section_numbering()
        demo_comprehensive_validation()
        demo_markdown_report()

        print("\n" + "=" * 80)
        print("  ✓ 所有E3验证演示完成")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ 演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
