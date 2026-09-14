"""
eCTD模块豁免验证器

验证某些特定模块的STF使用豁免规则和数据集位置要求。

根据eCTD技术规范3.8章节:
- 模块5.2 (所有临床研究列表)、5.3.6 (上市后报告) 和 5.4 (参考文献) 可以不使用STF
- 数据集应在eCTD骨架文件中位于相应的临床研究报告之后
"""

import re
from typing import List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .ectd_section_identifier import SectionIdentifier


class ViolationSeverity(Enum):
    """违规严重程度"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ViolationDetail:
    """违规详情"""
    rule_id: str
    severity: ViolationSeverity
    message: str
    location: str
    details: str
    suggestion: str


class ModuleExemptionValidator:
    """模块豁免和位置验证器"""

    # STF豁免模块模式
    STF_EXEMPTED_MODULES = [
        r'm5[/\\]m5-2[/\\]',           # 5.2 所有临床研究列表
        r'm5[/\\]m5-3[/\\]m5-3-6[/\\]', # 5.3.6 上市后报告
        r'm5[/\\]m5-4[/\\]',           # 5.4 参考文献
    ]

    # 需要STF的临床模块模式
    STF_REQUIRED_CLINICAL_MODULES = [
        r'm5[/\\]m5-3[/\\]m5-3-1[/\\]',  # 5.3.1 生物利用度研究报告
        r'm5[/\\]m5-3[/\\]m5-3-2[/\\]',  # 5.3.2 体内等效性研究报告
        r'm5[/\\]m5-3[/\\]m5-3-3[/\\]',  # 5.3.3 体内和体外比较研究报告
        r'm5[/\\]m5-3[/\\]m5-3-4[/\\]',  # 5.3.4 其他临床药理学研究报告
        r'm5[/\\]m5-3[/\\]m5-3-5[/\\]',  # 5.3.5 临床疗效和安全性研究报告
    ]

    # 需要STF的非临床模块模式
    STF_REQUIRED_NONCLINICAL_MODULES = [
        r'm4[/\\]m4-2[/\\]',  # 4.2 非临床研究报告
    ]

    def is_stf_exempted_module(self, module_path: str) -> bool:
        """
        检查模块是否豁免STF要求

        参数:
            module_path: 模块路径

        返回:
            True如果该模块可以不使用STF
        """
        for pattern in self.STF_EXEMPTED_MODULES:
            if re.search(pattern, module_path, re.IGNORECASE):
                return True
        return False

    def is_stf_required_module(self, module_path: str) -> bool:
        """
        检查模块是否必须使用STF

        参数:
            module_path: 模块路径

        返回:
            True如果该模块必须使用STF
        """
        all_required = self.STF_REQUIRED_CLINICAL_MODULES + self.STF_REQUIRED_NONCLINICAL_MODULES

        for pattern in all_required:
            if re.search(pattern, module_path, re.IGNORECASE):
                return True
        return False

    def validate_stf_exemption_usage(
        self,
        section_id: SectionIdentifier,
        has_stf: bool
    ) -> Optional[ViolationDetail]:
        """
        验证STF在豁免模块中的使用

        参数:
            section_id: 章节标识符
            has_stf: 是否使用了STF

        返回:
            如果发现违规则返回ViolationDetail，否则返回None
        """
        # 构建模块路径
        module_path = self._extract_module_path(section_id)

        # 检查是否在豁免模块中
        if self.is_stf_exempted_module(module_path):
            # 豁免模块可以使用STF，也可以不使用，这是INFO级别的提示
            if has_stf:
                return ViolationDetail(
                    rule_id="3.8.7",
                    severity=ViolationSeverity.INFO,
                    message=f"豁免模块使用了STF",
                    location=str(section_id),
                    details=f"模块 {module_path} (5.2/5.3.6/5.4) 可以不使用STF，但当前使用了STF。这是允许的。",
                    suggestion="无需修改，此为信息提示"
                )
        # 检查是否在必需STF的模块中
        elif self.is_stf_required_module(module_path):
            if not has_stf:
                # 判断是否为数据集（数据集不需要STF）
                if self._is_dataset_section(section_id):
                    return None

                # 判断是否为研究报告章节
                if self._is_study_report_section(section_id):
                    return ViolationDetail(
                        rule_id="3.8.7",
                        severity=ViolationSeverity.ERROR,
                        message=f"必需STF的模块缺少STF",
                        location=str(section_id),
                        details=f"模块 {module_path} 中的研究报告必须使用STF",
                        suggestion="为此研究报告创建相应的STF文件"
                    )

        return None

    def validate_dataset_position(
        self,
        study_report_sections: List[SectionIdentifier],
        dataset_sections: List[SectionIdentifier]
    ) -> List[ViolationDetail]:
        """
        验证数据集在骨架文件中应位于对应研究报告之后

        参数:
            study_report_sections: 研究报告章节列表（已按文档顺序排序）
            dataset_sections: 数据集章节列表（已按文档顺序排序）

        返回:
            违规列表
        """
        violations = []

        # 为每个数据集找到对应的研究报告
        for dataset in dataset_sections:
            dataset_study_id = self._extract_study_id_from_path(dataset)

            if not dataset_study_id:
                continue  # 无法确定研究ID，跳过

            # 查找对应的研究报告
            corresponding_report = None
            report_index = -1

            for idx, report in enumerate(study_report_sections):
                report_study_id = self._extract_study_id_from_path(report)
                if report_study_id and report_study_id == dataset_study_id:
                    corresponding_report = report
                    report_index = idx
                    break

            if corresponding_report:
                # 检查数据集的索引是否在研究报告之后
                dataset_index = self._get_section_index(dataset)
                report_full_index = self._get_section_index(corresponding_report)

                if dataset_index <= report_full_index:
                    violations.append(ViolationDetail(
                        rule_id="3.8.8",
                        severity=ViolationSeverity.ERROR,
                        message="数据集位置不正确",
                        location=str(dataset),
                        details=f"数据集应在对应的研究报告之后。数据集: {dataset.element_name}，研究报告: {corresponding_report.element_name}",
                        suggestion=f"在eCTD骨架文件中，将数据集相关的leaf元素移动到研究报告leaf元素之后"
                    ))

        return violations

    def validate_dataset_stf_tag(
        self,
        dataset_section: SectionIdentifier,
        stf_tag: Optional[str]
    ) -> Optional[ViolationDetail]:
        """
        验证数据集的STF标签

        参数:
            dataset_section: 数据集章节标识符
            stf_tag: STF标签值

        返回:
            如果发现违规则返回ViolationDetail
        """
        # 数据集应该有特定的STF标签
        valid_dataset_tags = {
            'data-tabulation-dataset',
            'data-tabulation-dataset-sdtm',
            'data-tabulation-dataset-legacy',
            'data-listing-dataset',
            'analysis-dataset',
            'analysis-dataset-adam',
            'analysis-dataset-legacy',
            'data-tabulation-data-definition',
            'data-listing-data-definition',
            'analysis-data-definition',
        }

        if stf_tag and stf_tag not in valid_dataset_tags:
            return ViolationDetail(
                rule_id="3.8.9",
                severity=ViolationSeverity.WARNING,
                message="数据集STF标签不标准",
                location=str(dataset_section),
                details=f"数据集使用了标签 '{stf_tag}'，建议使用标准数据集标签",
                suggestion=f"使用标准标签: {', '.join(sorted(valid_dataset_tags))}"
            )

        return None

    def _extract_module_path(self, section_id: SectionIdentifier) -> str:
        """从章节标识符中提取模块路径"""
        # 假设section_id有parent_path属性或类似机制
        # 这里简化处理，实际需要根据SectionIdentifier的实现调整
        element_name = section_id.element_name.lower()

        # 从element_name推断模块路径
        if 'm5-3-6' in element_name or '5.3.6' in element_name:
            return 'm5/m5-3/m5-3-6/'
        elif 'm5-2' in element_name or '5.2' in element_name:
            return 'm5/m5-2/'
        elif 'm5-4' in element_name or '5.4' in element_name:
            return 'm5/m5-4/'
        elif 'm5-3-1' in element_name or '5.3.1' in element_name:
            return 'm5/m5-3/m5-3-1/'
        elif 'm5-3-2' in element_name or '5.3.2' in element_name:
            return 'm5/m5-3/m5-3-2/'
        elif 'm5-3-3' in element_name or '5.3.3' in element_name:
            return 'm5/m5-3/m5-3-3/'
        elif 'm5-3-4' in element_name or '5.3.4' in element_name:
            return 'm5/m5-3/m5-3-4/'
        elif 'm5-3-5' in element_name or '5.3.5' in element_name:
            return 'm5/m5-3/m5-3-5/'
        elif 'm4-2' in element_name or '4.2' in element_name:
            return 'm4/m4-2/'

        return ''

    def _is_dataset_section(self, section_id: SectionIdentifier) -> bool:
        """判断是否为数据集章节"""
        element_name_lower = section_id.element_name.lower()
        return 'dataset' in element_name_lower or 'data-set' in element_name_lower

    def _is_study_report_section(self, section_id: SectionIdentifier) -> bool:
        """判断是否为研究报告章节"""
        # 检查是否有study-report相关标签
        if hasattr(section_id, 'attributes') and section_id.attributes:
            tag = section_id.attributes.get('tag') or section_id.attributes.get('type')
            if tag:
                return 'study-report' in tag.lower() or 'clinical-study-report' in tag.lower()

        # 检查element_name
        element_name_lower = section_id.element_name.lower()
        return 'study-report' in element_name_lower or 'clinical-study-report' in element_name_lower

    def _extract_study_id_from_path(self, section_id: SectionIdentifier) -> Optional[str]:
        """从章节路径中提取研究ID"""
        # 尝试从路径中提取研究ID
        # 假设路径格式类似: .../study-abc123/...
        if hasattr(section_id, 'path'):
            path = section_id.path
            match = re.search(r'study[-_]([a-zA-Z0-9\-_]+)', path, re.IGNORECASE)
            if match:
                return match.group(1)

        # 尝试从element_name中提取
        element_name = section_id.element_name
        match = re.search(r'study[-_]([a-zA-Z0-9\-_]+)', element_name, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _get_section_index(self, section_id: SectionIdentifier) -> int:
        """
        获取章节在文档中的索引位置

        这是一个简化实现，实际应该从index.xml的解析结果中获取
        """
        # 假设section_id有一个index或position属性
        if hasattr(section_id, 'index'):
            return section_id.index

        # 否则返回一个默认值（这种情况下位置验证可能不准确）
        return 0
