"""
eCTD 3.8章节综合验证器

整合STF格式、模块豁免、中国数据递交规范等所有3.8章节相关的验证规则。
"""

from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .ectd_stf_format_validator import STFFormatValidator, STFValidationContext
from .ectd_module_exemption_validator import ModuleExemptionValidator
from .ectd_china_data_validator import ChinaDataSubmissionValidator, DatasetMetadata
from .ectd_section_identifier import SectionIdentifier
from .ectd_e3_structure_validator import E3StructureValidator


class ViolationSeverity(Enum):
    """违规严重程度"""
    CRITICAL = "CRITICAL"  # 阻塞性错误，必须修复
    ERROR = "ERROR"        # 严重错误，应该修复
    WARNING = "WARNING"    # 警告，建议修复
    INFO = "INFO"          # 信息提示


@dataclass
class ViolationDetail:
    """违规详情"""
    rule_id: str                    # 规则ID (如 "3.8.1")
    severity: ViolationSeverity     # 严重程度
    message: str                    # 简短消息
    location: str                   # 位置（文件路径或章节标识）
    details: str                    # 详细说明
    suggestion: str                 # 修复建议
    timestamp: str = field(default_factory=lambda: "")


@dataclass
class ValidationResult:
    """验证结果"""
    total_violations: int
    critical_count: int
    error_count: int
    warning_count: int
    info_count: int
    violations: List[ViolationDetail]
    passed: bool  # 是否通过验证（无CRITICAL和ERROR）

    @classmethod
    def from_violations(cls, violations: List[ViolationDetail]) -> "ValidationResult":
        """从违规列表创建验证结果"""
        critical = sum(1 for v in violations if v.severity == ViolationSeverity.CRITICAL)
        error = sum(1 for v in violations if v.severity == ViolationSeverity.ERROR)
        warning = sum(1 for v in violations if v.severity == ViolationSeverity.WARNING)
        info = sum(1 for v in violations if v.severity == ViolationSeverity.INFO)

        return cls(
            total_violations=len(violations),
            critical_count=critical,
            error_count=error,
            warning_count=warning,
            info_count=info,
            violations=violations,
            passed=(critical == 0 and error == 0)
        )


class ECTDChapter38Validator:
    """
    eCTD 3.8章节综合验证器

    整合以下验证功能:
    1. STF文件格式验证（命名、结构、category、file-tag、property）
    2. 模块豁免规则验证（5.2/5.3.6/5.4可不使用STF）
    3. 数据集位置验证（数据集应在研究报告之后）
    4. 中国数据递交规范验证（命名、标识符、标签、格式）
    """

    def __init__(self):
        self.stf_validator = STFFormatValidator()
        self.exemption_validator = ModuleExemptionValidator()
        self.china_validator = ChinaDataSubmissionValidator()
        self.e3_validator = E3StructureValidator()

    def validate_stf_file(
        self,
        stf_file_path: str,
        module_path: str,
        stf_content: Optional[str] = None
    ) -> ValidationResult:
        """
        验证单个STF文件

        参数:
            stf_file_path: STF文件路径
            module_path: 所在模块路径
            stf_content: STF文件内容（可选，如不提供则尝试读取）

        返回:
            ValidationResult对象
        """
        violations = []

        # 如果没有提供内容，尝试读取
        if not stf_content:
            try:
                with open(stf_file_path, 'r', encoding='utf-8') as f:
                    stf_content = f.read()
            except Exception as e:
                violations.append(ViolationDetail(
                    rule_id="3.8.0",
                    severity=ViolationSeverity.CRITICAL,
                    message="无法读取STF文件",
                    location=stf_file_path,
                    details=f"读取文件时发生错误: {str(e)}",
                    suggestion="确保文件存在且有读取权限"
                ))
                return ValidationResult.from_violations(violations)

        # 创建验证上下文
        context = STFValidationContext(
            file_path=stf_file_path,
            module_path=module_path,
            stf_content=stf_content
        )

        # 1. 验证STF命名
        violations.extend(self.stf_validator.validate_stf_naming(context))

        # 2. 验证STF结构
        violations.extend(self.stf_validator.validate_stf_structure(context))

        # 3. 如果结构验证通过，进行更深入的验证
        if context.stf_content and not any(v.severity == ViolationSeverity.CRITICAL for v in violations):
            # 解析STF内容获取category和file-tag
            categories, file_tags, properties_by_tag = self._parse_stf_content(stf_content)

            # 验证category元素
            if categories:
                violations.extend(
                    self.stf_validator.validate_category_elements(categories, context)
                )

            # 验证file-tag元素
            if file_tags:
                violations.extend(
                    self.stf_validator.validate_file_tags(file_tags, context)
                )

            # 验证property元素
            for tag_name, properties in properties_by_tag.items():
                violations.extend(
                    self.stf_validator.validate_property_elements(tag_name, properties, context)
                )

        return ValidationResult.from_violations(violations)

    def validate_module_stf_usage(
        self,
        section_id: SectionIdentifier,
        has_stf: bool
    ) -> ValidationResult:
        """
        验证模块的STF使用是否符合豁免规则

        参数:
            section_id: 章节标识符
            has_stf: 该章节是否使用了STF

        返回:
            ValidationResult对象
        """
        violations = []

        violation = self.exemption_validator.validate_stf_exemption_usage(section_id, has_stf)
        if violation:
            violations.append(violation)

        return ValidationResult.from_violations(violations)

    def validate_dataset_positions(
        self,
        study_report_sections: List[SectionIdentifier],
        dataset_sections: List[SectionIdentifier]
    ) -> ValidationResult:
        """
        验证数据集位置（应在对应研究报告之后）

        参数:
            study_report_sections: 研究报告章节列表
            dataset_sections: 数据集章节列表

        返回:
            ValidationResult对象
        """
        violations = self.exemption_validator.validate_dataset_position(
            study_report_sections,
            dataset_sections
        )

        return ValidationResult.from_violations(violations)

    def validate_dataset(
        self,
        dataset_file_path: str,
        dataset_metadata: DatasetMetadata,
        encoding: Optional[str] = None,
        validate_china_rules: bool = True
    ) -> ValidationResult:
        """
        验证数据集文件

        参数:
            dataset_file_path: 数据集文件路径（.xpt）
            dataset_metadata: 数据集元数据
            encoding: 字符编码
            validate_china_rules: 是否应用中国数据递交规范

        返回:
            ValidationResult对象
        """
        violations = []

        if not validate_china_rules:
            return ValidationResult.from_violations(violations)

        # 1. 验证数据集命名
        violations.extend(
            self.china_validator.validate_dataset_naming(
                dataset_metadata.name,
                dataset_file_path
            )
        )

        # 2. 验证变量命名
        for var in dataset_metadata.variables:
            var_name = var.get('name', '')
            if var_name:
                violations.extend(
                    self.china_validator.validate_variable_naming(
                        var_name,
                        dataset_metadata.name,
                        dataset_file_path
                    )
                )

        # 3. 验证必需标识符
        violations.extend(
            self.china_validator.validate_required_identifiers(
                dataset_metadata,
                dataset_file_path
            )
        )

        # 4. 验证标签中文化
        violations.extend(
            self.china_validator.validate_labels_in_chinese(
                dataset_metadata,
                dataset_file_path
            )
        )

        # 5. 验证XPT格式
        violations.extend(
            self.china_validator.validate_xpt_format(
                dataset_file_path,
                dataset_metadata.name,
                encoding
            )
        )

        return ValidationResult.from_violations(violations)

    def validate_dataset_collection(
        self,
        datasets: List[DatasetMetadata],
        file_paths: List[str]
    ) -> ValidationResult:
        """
        验证数据集集合（检查ADSL等必需数据集）

        参数:
            datasets: 数据集元数据列表
            file_paths: 对应的文件路径列表

        返回:
            ValidationResult对象
        """
        violations = []

        # 检查是否有分析数据集
        has_analysis = any(ds.is_analysis or ds.name.startswith('ad') for ds in datasets)

        if has_analysis:
            # 验证ADSL数据集
            violations.extend(
                self.china_validator.validate_adsl_dataset(datasets, file_paths)
            )

        return ValidationResult.from_violations(violations)

    def validate_stf_tag_for_china(
        self,
        stf_tag: str,
        file_path: str
    ) -> ValidationResult:
        """
        验证STF标签是否符合中国标准

        参数:
            stf_tag: STF标签（file-tag的name属性）
            file_path: 文件路径

        返回:
            ValidationResult对象
        """
        violations = []

        violation = self.china_validator.validate_stf_tags_china(stf_tag, file_path)
        if violation:
            violations.append(violation)

        return ValidationResult.from_violations(violations)

    def _parse_stf_content(self, stf_content: str) -> tuple:
        """
        解析STF内容，提取category、file-tag和property信息

        返回:
            (categories, file_tags, properties_by_tag) 元组
        """
        import xml.etree.ElementTree as ET

        categories = []
        file_tags = []
        properties_by_tag = {}

        try:
            root = ET.fromstring(stf_content)

            # 提取category元素
            for category in root.findall('.//*[@name]'):
                if category.tag.endswith('category'):
                    categories.append({
                        'name': category.get('name', ''),
                        'info-type': category.get('info-type', ''),
                        'value': category.text or ''
                    })

            # 提取file-tag和property元素
            for doc_content in root.findall('.//*'):
                if doc_content.tag.endswith('doc-content'):
                    # 找到file-tag
                    for file_tag in doc_content.findall('.//*'):
                        if file_tag.tag.endswith('file-tag'):
                            tag_name = file_tag.get('name', '')
                            file_tags.append({
                                'name': tag_name,
                                'info-type': file_tag.get('info-type', '')
                            })

                            # 找到对应的property
                            properties = []
                            for prop in doc_content.findall('.//*'):
                                if prop.tag.endswith('property'):
                                    properties.append({
                                        'name': prop.get('name', ''),
                                        'info-type': prop.get('info-type', ''),
                                        'value': prop.text or ''
                                    })

                            if properties:
                                properties_by_tag[tag_name] = properties

        except ET.ParseError:
            # XML解析错误，在结构验证阶段已经报告
            pass

        return categories, file_tags, properties_by_tag

    def generate_validation_report(
        self,
        result: ValidationResult,
        output_format: str = "text"
    ) -> str:
        """
        生成验证报告

        参数:
            result: 验证结果
            output_format: 输出格式 ("text", "json", "markdown")

        返回:
            格式化的报告字符串
        """
        if output_format == "json":
            return self._generate_json_report(result)
        elif output_format == "markdown":
            return self._generate_markdown_report(result)
        else:
            return self._generate_text_report(result)

    def _generate_text_report(self, result: ValidationResult) -> str:
        """生成文本格式报告"""
        lines = []
        lines.append("=" * 80)
        lines.append("eCTD 3.8章节验证报告")
        lines.append("=" * 80)
        lines.append(f"总违规数: {result.total_violations}")
        lines.append(f"  - CRITICAL: {result.critical_count}")
        lines.append(f"  - ERROR: {result.error_count}")
        lines.append(f"  - WARNING: {result.warning_count}")
        lines.append(f"  - INFO: {result.info_count}")
        lines.append(f"验证状态: {'通过' if result.passed else '未通过'}")
        lines.append("")

        if result.violations:
            lines.append("违规详情:")
            lines.append("-" * 80)

            for idx, v in enumerate(result.violations, 1):
                lines.append(f"\n[{idx}] {v.severity.value} - {v.rule_id}")
                lines.append(f"    位置: {v.location}")
                lines.append(f"    消息: {v.message}")
                lines.append(f"    详情: {v.details}")
                lines.append(f"    建议: {v.suggestion}")

        return "\n".join(lines)

    def _generate_markdown_report(self, result: ValidationResult) -> str:
        """生成Markdown格式报告"""
        lines = []
        lines.append("# eCTD 3.8章节验证报告\n")
        lines.append("## 验证摘要\n")
        lines.append(f"- **总违规数**: {result.total_violations}")
        lines.append(f"- **CRITICAL**: {result.critical_count}")
        lines.append(f"- **ERROR**: {result.error_count}")
        lines.append(f"- **WARNING**: {result.warning_count}")
        lines.append(f"- **INFO**: {result.info_count}")
        lines.append(f"- **验证状态**: {'✅ 通过' if result.passed else '❌ 未通过'}\n")

        if result.violations:
            lines.append("## 违规详情\n")

            severity_icons = {
                ViolationSeverity.CRITICAL: "🚫",
                ViolationSeverity.ERROR: "❌",
                ViolationSeverity.WARNING: "⚠️",
                ViolationSeverity.INFO: "ℹ️"
            }

            for idx, v in enumerate(result.violations, 1):
                icon = severity_icons.get(v.severity, "•")
                lines.append(f"### {icon} [{idx}] {v.severity.value} - {v.rule_id}\n")
                lines.append(f"**位置**: `{v.location}`\n")
                lines.append(f"**消息**: {v.message}\n")
                lines.append(f"**详情**: {v.details}\n")
                lines.append(f"**建议**: {v.suggestion}\n")

        return "\n".join(lines)

    def _generate_json_report(self, result: ValidationResult) -> str:
        """生成JSON格式报告"""
        import json

        report = {
            "summary": {
                "total_violations": result.total_violations,
                "critical_count": result.critical_count,
                "error_count": result.error_count,
                "warning_count": result.warning_count,
                "info_count": result.info_count,
                "passed": result.passed
            },
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "severity": v.severity.value,
                    "message": v.message,
                    "location": v.location,
                    "details": v.details,
                    "suggestion": v.suggestion
                }
                for v in result.violations
            ]
        }

        return json.dumps(report, ensure_ascii=False, indent=2)

    def validate_e3_clinical_report(
        self,
        report_file_path: str,
        toc_data: Dict[str, str],
        synopsis_pages: Optional[int] = None,
        introduction_pages: Optional[int] = None,
        stf_file_tags: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        验证临床研究报告的ICH E3结构合规性

        参数:
            report_file_path: 报告文件路径
            toc_data: 目录数据，格式为 {章节号: 章节标题}
            synopsis_pages: Synopsis章节的页数（可选）
            introduction_pages: Introduction章节的页数（可选）
            stf_file_tags: STF文件中的file-tag列表（可选，用于一致性检查）

        返回:
            ValidationResult对象
        """
        violations = []

        # 1. 验证E3基本结构
        violations.extend(
            self.e3_validator.validate_e3_structure(
                report_file_path,
                toc_data,
                report_file_path
            )
        )

        # 2. 验证章节编号规范性
        violations.extend(
            self.e3_validator.validate_section_numbering(
                toc_data,
                report_file_path
            )
        )

        # 3. 验证Synopsis长度（如果提供了页数）
        if synopsis_pages is not None:
            violation = self.e3_validator.validate_synopsis_length(
                synopsis_pages,
                report_file_path
            )
            if violation:
                violations.append(violation)

        # 4. 验证Introduction长度（如果提供了页数）
        if introduction_pages is not None:
            violation = self.e3_validator.validate_introduction_length(
                introduction_pages,
                report_file_path
            )
            if violation:
                violations.append(violation)

        # 5. 验证STF file-tags与E3结构的一致性（如果提供了file-tags）
        if stf_file_tags:
            violations.extend(
                self.e3_validator.validate_stf_e3_consistency(
                    stf_file_tags,
                    toc_data,
                    report_file_path
                )
            )

        return ValidationResult.from_violations(violations)

    def validate_comprehensive(
        self,
        stf_file_path: Optional[str] = None,
        report_file_path: Optional[str] = None,
        report_toc: Optional[Dict[str, str]] = None,
        dataset_files: Optional[List[str]] = None,
        dataset_metadata_list: Optional[List[DatasetMetadata]] = None,
        module_path: str = "",
        validate_china_rules: bool = True
    ) -> ValidationResult:
        """
        综合验证：STF + E3报告 + 数据集

        参数:
            stf_file_path: STF文件路径（可选）
            report_file_path: 临床研究报告路径（可选）
            report_toc: 报告目录数据（可选）
            dataset_files: 数据集文件路径列表（可选）
            dataset_metadata_list: 数据集元数据列表（可选）
            module_path: 模块路径
            validate_china_rules: 是否验证中国规范

        返回:
            ValidationResult对象（合并所有验证结果）
        """
        all_violations = []

        # 1. 验证STF文件
        if stf_file_path:
            stf_result = self.validate_stf_file(stf_file_path, module_path)
            all_violations.extend(stf_result.violations)

        # 2. 验证E3报告结构
        if report_file_path and report_toc:
            # 从STF中提取file-tags（如果有STF）
            stf_file_tags = None
            if stf_file_path:
                try:
                    with open(stf_file_path, 'r', encoding='utf-8') as f:
                        stf_content = f.read()
                    _, file_tags_data, _ = self._parse_stf_content(stf_content)
                    stf_file_tags = [ft['name'] for ft in file_tags_data]
                except:
                    pass

            e3_result = self.validate_e3_clinical_report(
                report_file_path,
                report_toc,
                stf_file_tags=stf_file_tags
            )
            all_violations.extend(e3_result.violations)

        # 3. 验证数据集
        if dataset_files and dataset_metadata_list:
            for dataset_file, dataset_meta in zip(dataset_files, dataset_metadata_list):
                dataset_result = self.validate_dataset(
                    dataset_file,
                    dataset_meta,
                    encoding='utf-8',
                    validate_china_rules=validate_china_rules
                )
                all_violations.extend(dataset_result.violations)

            # 验证数据集集合
            collection_result = self.validate_dataset_collection(
                dataset_metadata_list,
                dataset_files
            )
            all_violations.extend(collection_result.violations)

        return ValidationResult.from_violations(all_violations)
