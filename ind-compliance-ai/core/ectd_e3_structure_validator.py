#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eCTD ICH E3临床研究报告结构验证器

根据ICH E3 Guideline验证临床研究报告的结构完整性和合规性。

主要验证规则：
1. 必需章节存在性验证（1-16章）
2. 章节编号和标题规范性
3. 附录16.1和16.2子章节完整性
4. Synopsis长度限制（≤3页）
5. Introduction长度限制（≤1页）
6. STF file-tags与E3结构的一致性

标准来源: ICH E3 Guideline - Structure and Content of Clinical Study Reports
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Set


class ViolationSeverity(Enum):
    """违规严重程度"""
    CRITICAL = "CRITICAL"  # 阻塞性错误，必须修复
    ERROR = "ERROR"        # 严重错误，强烈建议修复
    WARNING = "WARNING"    # 警告，建议修复
    INFO = "INFO"          # 信息提示


@dataclass
class ViolationDetail:
    """违规详情"""
    rule_id: str           # 规则ID
    severity: ViolationSeverity  # 严重程度
    message: str           # 违规消息
    location: str          # 违规位置
    details: str           # 详细说明
    suggestion: str        # 修复建议


class E3StructureValidator:
    """ICH E3临床研究报告结构验证器"""

    # ICH E3必需章节定义（章节号 -> 标题）
    REQUIRED_SECTIONS = {
        "1": "Title Page",
        "2": "Synopsis",
        "3": "Table of Contents",
        "4": "List of Abbreviations and Definition of Terms",
        "5": "Ethics",
        "6": "Investigators and Study Administrative Structure",
        "7": "Introduction",
        "8": "Study Objectives",
        "9": "Investigational Plan",
        "10": "Study Patients",
        "11": "Efficacy Evaluation",
        "12": "Safety Evaluation",
        "13": "Discussion and Overall Conclusions",
        "14": "Tables, Figures and Graphs",
        "15": "Reference List",
        "16": "Appendices",
    }

    # 章节5的必需子章节
    SECTION_5_SUBSECTIONS = {
        "5.1": "IEC/IRB",
        "5.2": "Ethical Conduct of the Study",
        "5.3": "Patient Information and Consent",
    }

    # 章节9的必需子章节
    SECTION_9_SUBSECTIONS = {
        "9.1": "Overall Study Design and Plan",
        "9.2": "Discussion of Study Design",
        "9.3": "Selection of Study Population",
        "9.4": "Treatment",
        "9.5": "Efficacy and Safety Variables",
        "9.6": "Data Quality Assurance",
        "9.7": "Statistical Methods",
    }

    # 章节10的必需子章节
    SECTION_10_SUBSECTIONS = {
        "10.1": "Disposition of Patients",
        "10.2": "Protocol Deviations",
    }

    # 章节11的必需子章节
    SECTION_11_SUBSECTIONS = {
        "11.1": "Datasets Analysed",
        "11.2": "Demographic and Other Baseline Characteristics",
        "11.3": "Measurements of Treatment Compliance",
        "11.4": "Efficacy Results and Tabulations of Individual Subject Data",
    }

    # 章节12的必需子章节
    SECTION_12_SUBSECTIONS = {
        "12.1": "Extent of Exposure",
        "12.2": "Adverse Events",
        "12.3": "Deaths, Other Serious Adverse Events, and Other Significant Adverse Events",
        "12.4": "Clinical Laboratory Evaluations",
        "12.5": "Vital Signs, Physical Findings, and Other Observations",
        "12.6": "Safety Conclusions",
    }

    # 章节14的必需子章节
    SECTION_14_SUBSECTIONS = {
        "14.1": "Demographic Data",
        "14.2": "Efficacy Data",
        "14.3": "Safety Data",
    }

    # 章节16.1的必需子章节（Study Information）
    SECTION_16_1_SUBSECTIONS = {
        "16.1.1": "Protocol and Protocol Amendments",
        "16.1.2": "Sample Case Report Form",
        "16.1.3": "List of IECs/IRBs and Informed Consent Forms",
        "16.1.4": "List of Investigators and Other Important Participants",
        "16.1.5": "Signature of Responsible Parties",
        "16.1.9": "Documentation of Statistical Methods",
    }

    # 章节16.2的必需子章节（Patient Data Listings）
    SECTION_16_2_SUBSECTIONS = {
        "16.2.1": "Listing of Patients Who Discontinued",
        "16.2.2": "Protocol Deviations",
        "16.2.3": "Patients Excluded from Efficacy Analysis",
        "16.2.4": "Demographic Data",
        "16.2.6": "Individual Efficacy Response Data",
        "16.2.7": "Listing of Adverse Events by Patient",
    }

    # STF file-tags与E3章节的映射
    E3_STF_TAG_MAPPING = {
        "protocol": ["16.1.1"],
        "informed-consent-form": ["16.1.3"],
        "sample-crf": ["16.1.2"],
        "investigator-list": ["16.1.4"],
        "statistical-analysis-plan": ["16.1.9"],
        "subject-profiles": ["16.2"],
        "adverse-event-listings": ["16.2.7"],
    }

    def __init__(self):
        """初始化验证器"""
        pass

    def validate_e3_structure(
        self,
        report_path: str,
        toc_data: Dict[str, str],
        file_path: str = ""
    ) -> List[ViolationDetail]:
        """
        验证临床研究报告的ICH E3结构合规性

        Args:
            report_path: 报告文件路径
            toc_data: 目录数据，格式为 {章节号: 章节标题}
            file_path: 用于错误报告的文件路径

        Returns:
            违规详情列表
        """
        violations = []

        if not file_path:
            file_path = report_path

        # 验证必需章节
        violations.extend(self._validate_required_sections(toc_data, file_path))

        # 验证子章节
        violations.extend(self._validate_subsections(toc_data, file_path))

        return violations

    def _validate_required_sections(
        self,
        toc_data: Dict[str, str],
        file_path: str
    ) -> List[ViolationDetail]:
        """验证必需章节是否存在"""
        violations = []

        for section_num, section_title in self.REQUIRED_SECTIONS.items():
            if section_num not in toc_data:
                violations.append(ViolationDetail(
                    rule_id="E3.STRUCT.001",
                    severity=ViolationSeverity.ERROR,
                    message=f"Missing required ICH E3 section: {section_num} {section_title}",
                    location=f"{file_path}:section-{section_num}",
                    details=f"ICH E3 requires section {section_num} '{section_title}' to be present in all clinical study reports.",
                    suggestion=f"Add section {section_num} '{section_title}' to the clinical study report according to ICH E3 guidelines."
                ))

        return violations

    def _validate_subsections(
        self,
        toc_data: Dict[str, str],
        file_path: str
    ) -> List[ViolationDetail]:
        """验证各主要章节的必需子章节"""
        violations = []

        # 验证章节5子章节
        violations.extend(self._check_subsections(
            toc_data, self.SECTION_5_SUBSECTIONS, "5", file_path, "E3.STRUCT.005"
        ))

        # 验证章节9子章节
        violations.extend(self._check_subsections(
            toc_data, self.SECTION_9_SUBSECTIONS, "9", file_path, "E3.STRUCT.009"
        ))

        # 验证章节10子章节
        violations.extend(self._check_subsections(
            toc_data, self.SECTION_10_SUBSECTIONS, "10", file_path, "E3.STRUCT.010"
        ))

        # 验证章节11子章节
        violations.extend(self._check_subsections(
            toc_data, self.SECTION_11_SUBSECTIONS, "11", file_path, "E3.STRUCT.011"
        ))

        # 验证章节12子章节
        violations.extend(self._check_subsections(
            toc_data, self.SECTION_12_SUBSECTIONS, "12", file_path, "E3.STRUCT.012"
        ))

        # 验证章节14子章节
        violations.extend(self._check_subsections(
            toc_data, self.SECTION_14_SUBSECTIONS, "14", file_path, "E3.STRUCT.014"
        ))

        # 验证章节16.1子章节
        violations.extend(self._check_subsections(
            toc_data, self.SECTION_16_1_SUBSECTIONS, "16.1", file_path, "E3.STRUCT.016.1"
        ))

        # 验证章节16.2子章节
        violations.extend(self._check_subsections(
            toc_data, self.SECTION_16_2_SUBSECTIONS, "16.2", file_path, "E3.STRUCT.016.2"
        ))

        return violations

    def _check_subsections(
        self,
        toc_data: Dict[str, str],
        required_subsections: Dict[str, str],
        parent_section: str,
        file_path: str,
        rule_id_prefix: str
    ) -> List[ViolationDetail]:
        """检查特定章节的必需子章节"""
        violations = []

        # 首先检查父章节是否存在
        if parent_section not in toc_data:
            # 父章节不存在会在上层被捕获，这里跳过子章节检查
            return violations

        for subsection_num, subsection_title in required_subsections.items():
            if subsection_num not in toc_data:
                violations.append(ViolationDetail(
                    rule_id=rule_id_prefix,
                    severity=ViolationSeverity.WARNING,
                    message=f"Missing recommended subsection: {subsection_num} {subsection_title}",
                    location=f"{file_path}:section-{subsection_num}",
                    details=f"ICH E3 recommends section {subsection_num} '{subsection_title}' under section {parent_section}.",
                    suggestion=f"Add subsection {subsection_num} '{subsection_title}' to section {parent_section}."
                ))

        return violations

    def validate_synopsis_length(
        self,
        synopsis_pages: int,
        file_path: str
    ) -> Optional[ViolationDetail]:
        """
        验证Synopsis章节长度（应≤3页）

        Args:
            synopsis_pages: Synopsis的页数
            file_path: 文件路径

        Returns:
            如果超过3页，返回违规详情；否则返回None
        """
        if synopsis_pages > 3:
            return ViolationDetail(
                rule_id="E3.LENGTH.002",
                severity=ViolationSeverity.WARNING,
                message=f"Synopsis exceeds recommended length: {synopsis_pages} pages (recommended: ≤3 pages)",
                location=f"{file_path}:section-2",
                details=f"ICH E3 guideline recommends that the Synopsis (section 2) should normally not exceed 3 pages. Current length: {synopsis_pages} pages.",
                suggestion="Condense the Synopsis to 3 pages or less by focusing on key information."
            )
        return None

    def validate_introduction_length(
        self,
        intro_pages: int,
        file_path: str
    ) -> Optional[ViolationDetail]:
        """
        验证Introduction章节长度（应≤1页）

        Args:
            intro_pages: Introduction的页数
            file_path: 文件路径

        Returns:
            如果超过1页，返回违规详情；否则返回None
        """
        if intro_pages > 1:
            return ViolationDetail(
                rule_id="E3.LENGTH.007",
                severity=ViolationSeverity.WARNING,
                message=f"Introduction exceeds recommended length: {intro_pages} pages (recommended: ≤1 page)",
                location=f"{file_path}:section-7",
                details=f"ICH E3 guideline recommends that the Introduction (section 7) should normally not exceed 1 page. Current length: {intro_pages} pages.",
                suggestion="Condense the Introduction to 1 page by providing only essential background information."
            )
        return None

    def validate_stf_e3_consistency(
        self,
        stf_file_tags: List[str],
        toc_data: Dict[str, str],
        file_path: str
    ) -> List[ViolationDetail]:
        """
        验证STF file-tags与E3结构的一致性

        检查STF中声明的file-tags是否与报告中对应的E3章节匹配

        Args:
            stf_file_tags: STF文件中的file-tag列表
            toc_data: 报告的目录数据
            file_path: 文件路径

        Returns:
            违规详情列表
        """
        violations = []

        for file_tag in stf_file_tags:
            if file_tag in self.E3_STF_TAG_MAPPING:
                expected_sections = self.E3_STF_TAG_MAPPING[file_tag]

                # 检查对应的E3章节是否存在
                for section_num in expected_sections:
                    if section_num not in toc_data:
                        violations.append(ViolationDetail(
                            rule_id="E3.STF.001",
                            severity=ViolationSeverity.WARNING,
                            message=f"STF declares file-tag '{file_tag}' but corresponding E3 section {section_num} is missing",
                            location=f"{file_path}:file-tag-{file_tag}",
                            details=f"The STF file contains file-tag '{file_tag}' which maps to E3 section {section_num}, but this section is not found in the report structure.",
                            suggestion=f"Either add E3 section {section_num} to the report or remove the '{file_tag}' file-tag from the STF."
                        ))

        return violations

    def validate_section_numbering(
        self,
        toc_data: Dict[str, str],
        file_path: str
    ) -> List[ViolationDetail]:
        """
        验证章节编号的规范性

        检查章节编号是否符合ICH E3规范（1-16主章节，子章节使用点分格式）

        Args:
            toc_data: 目录数据
            file_path: 文件路径

        Returns:
            违规详情列表
        """
        violations = []

        # 章节编号格式正则：主章节(1-16)或子章节(如9.1, 16.1.2)
        valid_section_pattern = re.compile(r'^(\d+)(\.\d+)*$')

        for section_num in toc_data.keys():
            if not valid_section_pattern.match(section_num):
                violations.append(ViolationDetail(
                    rule_id="E3.NUMBER.001",
                    severity=ViolationSeverity.WARNING,
                    message=f"Invalid section numbering format: {section_num}",
                    location=f"{file_path}:section-{section_num}",
                    details=f"Section number '{section_num}' does not follow ICH E3 numbering convention (e.g., 1, 9.1, 16.1.2).",
                    suggestion=f"Renumber section '{section_num}' to follow ICH E3 hierarchical numbering format."
                ))
            else:
                # 检查主章节编号是否在1-16范围内
                main_section = int(section_num.split('.')[0])
                if main_section < 1 or main_section > 16:
                    violations.append(ViolationDetail(
                        rule_id="E3.NUMBER.002",
                        severity=ViolationSeverity.WARNING,
                        message=f"Section number out of E3 range: {section_num}",
                        location=f"{file_path}:section-{section_num}",
                        details=f"Main section number {main_section} is outside the ICH E3 range (1-16).",
                        suggestion=f"ICH E3 defines sections 1-16. Consider organizing content within these sections."
                    ))

        return violations

    def get_validation_summary(self, violations: List[ViolationDetail]) -> Dict[str, int]:
        """
        生成验证摘要统计

        Args:
            violations: 违规列表

        Returns:
            按严重程度分类的统计字典
        """
        summary = {
            "CRITICAL": 0,
            "ERROR": 0,
            "WARNING": 0,
            "INFO": 0,
            "TOTAL": len(violations)
        }

        for violation in violations:
            summary[violation.severity.value] += 1

        return summary


if __name__ == "__main__":
    # 演示用法
    print("ICH E3 Structure Validator")
    print("=" * 60)

    validator = E3StructureValidator()

    # 示例1: 检查完整的TOC
    print("\n示例1: 验证完整的E3结构")
    complete_toc = {
        "1": "Title Page",
        "2": "Synopsis",
        "3": "Table of Contents",
        "4": "Abbreviations",
        "5": "Ethics",
        "5.1": "IEC/IRB",
        "5.2": "Ethical Conduct",
        "5.3": "Patient Consent",
        "9": "Study Plan",
        "9.1": "Design",
        "16": "Appendices",
        "16.1": "Study Information",
        "16.1.1": "Protocol",
    }

    violations = validator.validate_e3_structure("study-001.pdf", complete_toc)
    print(f"发现 {len(violations)} 个问题")

    # 示例2: 检查不完整的TOC
    print("\n示例2: 验证不完整的E3结构")
    incomplete_toc = {
        "1": "Title Page",
        "2": "Synopsis",
    }

    violations = validator.validate_e3_structure("study-002.pdf", incomplete_toc)
    print(f"发现 {len(violations)} 个问题")
    for v in violations[:3]:
        print(f"  - {v.severity.value}: {v.message}")

    # 示例3: 验证Synopsis长度
    print("\n示例3: 验证Synopsis长度")
    violation = validator.validate_synopsis_length(5, "study-003.pdf")
    if violation:
        print(f"  - {violation.severity.value}: {violation.message}")

    print("\n验证器初始化完成！")
