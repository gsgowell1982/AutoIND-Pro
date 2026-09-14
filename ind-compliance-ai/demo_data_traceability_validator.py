"""
eCTD数据可追溯性验证器 - 演示脚本

演示如何使用数据可追溯性验证器检查临床研究数据的可追溯性。

Phase 2.12 - Data Traceability Validation
创建日期: 2026-09-14
"""

import sys
import io
from pathlib import Path
from typing import List, Dict

# 设置stdout为UTF-8编码（Windows兼容）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.ectd_data_traceability_validator import (
    DataTraceabilityValidator,
    ACRFAnnotation,
    DatasetVariable,
    DerivationMetadata,
    ViolationDetail,
    ViolationSeverity,
    validate_data_traceability
)


def print_section_header(title: str):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_violations(violations: List[ViolationDetail]):
    """打印违规详情"""
    if not violations:
        print("✅ 未发现违规")
        return

    print(f"❌ 发现 {len(violations)} 个违规:")
    for i, violation in enumerate(violations, 1):
        severity_icon = {
            ViolationSeverity.CRITICAL: "🔴",
            ViolationSeverity.ERROR: "🔴",
            ViolationSeverity.WARNING: "🟡",
            ViolationSeverity.INFO: "🔵"
        }.get(violation.severity, "⚪")

        print(f"\n  {i}. {severity_icon} [{violation.rule_id}] {violation.severity.value}")
        print(f"     消息: {violation.message}")
        print(f"     位置: {violation.location}")
        print(f"     详情: {violation.details}")
        if violation.suggestion:
            print(f"     建议: {violation.suggestion}")


def create_sample_datasets() -> tuple:
    """创建样本数据集"""
    raw_datasets = {
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
        ],
        "AE": [
            DatasetVariable("AE", "STUDYID", "Study Identifier", "Char"),
            DatasetVariable("AE", "USUBJID", "Unique Subject Identifier", "Char"),
            DatasetVariable("AE", "AETERM", "Reported Term for the AE", "Char"),
        ]
    }

    analysis_datasets = {
        "ADSL": [
            DatasetVariable("ADSL", "STUDYID", "Study Identifier", "Char"),
            DatasetVariable("ADSL", "USUBJID", "Unique Subject Identifier", "Char"),
            DatasetVariable("ADSL", "AGE", "Age", "Num"),
            DatasetVariable("ADSL", "AGEGR1", "Age Group 1", "Char",
                          is_derived=True, derivation_method="AGE categorized"),
            DatasetVariable("ADSL", "SEX", "Sex", "Char"),
        ],
        "ADAE": [
            DatasetVariable("ADAE", "STUDYID", "Study Identifier", "Char"),
            DatasetVariable("ADAE", "USUBJID", "Unique Subject Identifier", "Char"),
            DatasetVariable("ADAE", "AEDECOD", "Dictionary-Derived Term", "Char",
                          is_derived=True, derivation_method="MedDRA coded"),
        ]
    }

    return raw_datasets, analysis_datasets


def demo_scenario_1_valid_acrf_mapping():
    """场景1: 有效的aCRF映射"""
    print_section_header("场景1: 有效的aCRF映射（完全合规）")

    raw_datasets, _ = create_sample_datasets()

    # 创建有效的aCRF注释
    acrf_annotations = [
        ACRFAnnotation("CRF-01", "AGE_FIELD", "DM", "AGE", "受试者年龄"),
        ACRFAnnotation("CRF-01", "SEX_FIELD", "DM", "SEX", "受试者性别"),
        ACRFAnnotation("CRF-02", "BP_SYSTOLIC", "VS", "VSORRES", "收缩压"),
        ACRFAnnotation("CRF-03", "AE_TERM", "AE", "AETERM", "不良事件术语"),
    ]

    print("\naCRF注释:")
    for ann in acrf_annotations:
        print(f"  {ann.crf_page}:{ann.crf_field} → {ann.dataset_name}.{ann.variable_name}")

    # 执行验证
    print("\n执行验证...")
    validator = DataTraceabilityValidator()
    violations = validator.validate_acrf_mapping(acrf_annotations, raw_datasets)

    print_violations(violations)


def demo_scenario_2_invalid_acrf_mapping():
    """场景2: 无效的aCRF映射"""
    print_section_header("场景2: aCRF映射违规（引用不存在的数据集/变量）")

    raw_datasets, _ = create_sample_datasets()

    # 创建包含错误的aCRF注释
    acrf_annotations = [
        ACRFAnnotation("CRF-01", "AGE_FIELD", "DM", "AGE", "受试者年龄"),
        ACRFAnnotation("CRF-01", "WEIGHT_FIELD", "DM", "WEIGHT", "体重"),  # ❌ 变量不存在
        ACRFAnnotation("CRF-02", "HEIGHT_FIELD", "XX", "HEIGHT", "身高"),  # ❌ 数据集不存在
    ]

    print("\naCRF注释:")
    for ann in acrf_annotations:
        status = "✅" if (ann.dataset_name in raw_datasets and
                         ann.variable_name in [v.variable_name for v in raw_datasets.get(ann.dataset_name, [])]) else "❌"
        print(f"  {status} {ann.crf_page}:{ann.crf_field} → {ann.dataset_name}.{ann.variable_name}")

    # 执行验证
    print("\n执行验证...")
    validator = DataTraceabilityValidator()
    violations = validator.validate_acrf_mapping(acrf_annotations, raw_datasets)

    print_violations(violations)


def demo_scenario_3_duplicate_mapping():
    """场景3: 重复的CRF字段映射"""
    print_section_header("场景3: CRF字段映射到多个变量（警告）")

    raw_datasets, analysis_datasets = create_sample_datasets()

    # 同一个CRF字段映射到多个变量
    acrf_annotations = [
        ACRFAnnotation("CRF-01", "AGE_FIELD", "DM", "AGE", "原始年龄"),
        ACRFAnnotation("CRF-01", "AGE_FIELD", "ADSL", "AGE", "分析年龄"),  # ⚠️ 重复映射
    ]

    # 合并数据集
    all_datasets = {**raw_datasets, **analysis_datasets}

    print("\naCRF注释:")
    for ann in acrf_annotations:
        print(f"  {ann.crf_page}:{ann.crf_field} → {ann.dataset_name}.{ann.variable_name}")
    print("\n⚠️ 注意: CRF-01:AGE_FIELD被映射到2个不同的变量")

    # 执行验证
    print("\n执行验证...")
    validator = DataTraceabilityValidator()
    violations = validator.validate_acrf_mapping(acrf_annotations, all_datasets)

    print_violations(violations)


def demo_scenario_4_valid_derivation():
    """场景4: 有效的衍生变量可追溯性"""
    print_section_header("场景4: 有效的衍生变量可追溯性（完全合规）")

    raw_datasets, analysis_datasets = create_sample_datasets()

    # 创建有效的衍生变量元数据
    derivation_metadata = [
        DerivationMetadata(
            target_dataset="ADSL",
            target_variable="AGEGR1",
            source_datasets=["DM"],
            source_variables=["AGE"],
            derivation_algorithm="if AGE < 65 then AGEGR1='<65'; else AGEGR1='>=65'",
            program_file="adsl.sas",
            documentation="将年龄分为<65和>=65两组"
        ),
        DerivationMetadata(
            target_dataset="ADAE",
            target_variable="AEDECOD",
            source_datasets=["AE"],
            source_variables=["AETERM"],
            derivation_algorithm="使用MedDRA字典对AETERM进行编码",
            program_file="adae.sas",
            documentation="不良事件术语的MedDRA编码"
        ),
    ]

    print("\n衍生变量:")
    for deriv in derivation_metadata:
        print(f"  {deriv.target_dataset}.{deriv.target_variable}")
        print(f"    源: {deriv.source_datasets} → {deriv.source_variables}")
        print(f"    算法: {deriv.derivation_algorithm}")
        print(f"    程序: {deriv.program_file}")

    # 执行验证
    print("\n执行验证...")
    validator = DataTraceabilityValidator()
    violations = validator.validate_derivation_traceability(
        derivation_metadata, raw_datasets, analysis_datasets
    )

    print_violations(violations)


def demo_scenario_5_missing_algorithm():
    """场景5: 缺少衍生算法文档"""
    print_section_header("场景5: 衍生变量缺少算法文档（违规）")

    raw_datasets, analysis_datasets = create_sample_datasets()

    # 创建缺少文档的衍生变量元数据
    derivation_metadata = [
        DerivationMetadata(
            target_dataset="ADSL",
            target_variable="AGEGR1",
            source_datasets=["DM"],
            source_variables=["AGE"],
            # ❌ 缺少derivation_algorithm和documentation
            program_file="adsl.sas"
        ),
    ]

    print("\n衍生变量:")
    for deriv in derivation_metadata:
        print(f"  {deriv.target_dataset}.{deriv.target_variable}")
        print(f"    源: {deriv.source_datasets} → {deriv.source_variables}")
        print(f"    算法: {deriv.derivation_algorithm or '❌ 缺失'}")
        print(f"    文档: {deriv.documentation or '❌ 缺失'}")

    # 执行验证
    print("\n执行验证...")
    validator = DataTraceabilityValidator()
    violations = validator.validate_derivation_traceability(
        derivation_metadata, raw_datasets, analysis_datasets
    )

    print_violations(violations)


def demo_scenario_6_invalid_source_variable():
    """场景6: 引用不存在的源变量"""
    print_section_header("场景6: 衍生变量引用不存在的源变量（违规）")

    raw_datasets, analysis_datasets = create_sample_datasets()

    # 创建引用不存在源变量的元数据
    derivation_metadata = [
        DerivationMetadata(
            target_dataset="ADSL",
            target_variable="AGEGR1",
            source_datasets=["DM"],
            source_variables=["NONEXIST"],  # ❌ 源变量不存在
            derivation_algorithm="基于NONEXIST计算",
            program_file="adsl.sas"
        ),
    ]

    print("\n衍生变量:")
    for deriv in derivation_metadata:
        print(f"  {deriv.target_dataset}.{deriv.target_variable}")
        print(f"    源数据集: {deriv.source_datasets}")
        print(f"    源变量: {deriv.source_variables} ❌")

    # 执行验证
    print("\n执行验证...")
    validator = DataTraceabilityValidator()
    violations = validator.validate_derivation_traceability(
        derivation_metadata, raw_datasets, analysis_datasets
    )

    print_violations(violations)


def demo_scenario_7_comprehensive_validation():
    """场景7: 综合验证（完整数据流）"""
    print_section_header("场景7: 综合数据可追溯性验证")

    raw_datasets, analysis_datasets = create_sample_datasets()

    # 创建完整的可追溯性场景
    acrf_annotations = [
        ACRFAnnotation("CRF-01", "AGE", "DM", "AGE"),
        ACRFAnnotation("CRF-01", "SEX", "DM", "SEX"),
        ACRFAnnotation("CRF-03", "AE_TERM", "AE", "AETERM"),
    ]

    derivation_metadata = [
        DerivationMetadata(
            target_dataset="ADSL",
            target_variable="AGEGR1",
            source_datasets=["DM"],
            source_variables=["AGE"],
            derivation_algorithm="if AGE < 65 then '<65'; else '>=65'",
            program_file="adsl.sas"
        ),
        DerivationMetadata(
            target_dataset="ADAE",
            target_variable="AEDECOD",
            source_datasets=["AE"],
            source_variables=["AETERM"],
            derivation_algorithm="MedDRA coding",
            program_file="adae.sas"
        ),
    ]

    print("\n数据流概览:")
    print("  1. CRF数据收集 → 原始数据集 (DM, VS, AE)")
    print("  2. 原始数据集 → 分析数据集 (ADSL, ADAE)")
    print("  3. 衍生变量生成 (AGEGR1, AEDECOD)")

    print(f"\n  aCRF注释: {len(acrf_annotations)}个")
    print(f"  衍生变量: {len(derivation_metadata)}个")

    # 执行综合验证
    print("\n执行综合验证...")
    result = validate_data_traceability(
        acrf_annotations,
        derivation_metadata,
        raw_datasets,
        analysis_datasets
    )

    # 打印验证结果
    print("\n" + "=" * 80)
    print(result.get_summary())
    print("=" * 80)

    if result.violations:
        print("\n违规详情:")
        print_violations(result.violations)

    if result.warnings:
        print("\n警告:")
        print_violations(result.warnings)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" eCTD数据可追溯性验证器 - 演示脚本")
    print(" Phase 2.12 - Data Traceability Validation")
    print("=" * 80)

    print("\n本演示展示数据可追溯性验证的主要功能:")
    print("  1. aCRF与原始数据集的映射完整性验证")
    print("  2. 衍生变量可追溯性验证")
    print("  3. 数据流程完整性综合验证")

    try:
        demo_scenario_1_valid_acrf_mapping()
        demo_scenario_2_invalid_acrf_mapping()
        demo_scenario_3_duplicate_mapping()
        demo_scenario_4_valid_derivation()
        demo_scenario_5_missing_algorithm()
        demo_scenario_6_invalid_source_variable()
        demo_scenario_7_comprehensive_validation()

        print("\n" + "=" * 80)
        print(" 演示完成")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
