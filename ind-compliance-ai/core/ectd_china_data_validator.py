"""
中国数据递交规范验证器

根据《药物临床试验数据递交指导原则（试行）》验证数据集的格式和内容。

验证内容:
1. 数据集命名规范（小写字母开头，≤8字节）
2. 变量命名规范（大写字母开头，≤8字节）
3. 必需标识符（STUDYID, USUBJID, SUBJID）
4. 中文标签要求
5. 中国特定STF标签
6. XPT格式规范
"""

import re
from typing import List, Optional, Set, Dict
from dataclasses import dataclass
from enum import Enum


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


@dataclass
class DatasetMetadata:
    """数据集元数据"""
    name: str
    label: str
    variables: List[Dict[str, str]]  # [{'name': '...', 'label': '...', 'type': '...', 'length': ...}]
    is_analysis: bool = False  # True表示分析数据集，False表示原始数据集


class ChinaDataSubmissionValidator:
    """中国数据递交规范验证器"""

    # 必需的标识符变量
    REQUIRED_IDENTIFIERS = {
        'STUDYID': '研究标识符',
        'USUBJID': '受试者唯一标识符',
    }

    # dm数据集必需的额外标识符
    DM_REQUIRED_IDENTIFIERS = {
        'STUDYID': '研究标识符',
        'USUBJID': '受试者唯一标识符',
        'SUBJID': '受试者标识符',
    }

    # 推荐的时间变量
    RECOMMENDED_TIME_VARS = ['VISIT', 'VISITNUM']

    # 常用原始数据集（必须递交或如适用递交）
    COMMON_RAW_DATASETS = {
        'dm': ('人口学', True),  # (中文名称, 是否必须)
        'mh': ('病史', False),
        'ae': ('不良事件', False),
        'cm': ('既往与合并用药', False),
        'ex': ('暴露', False),
        'ds': ('受试者分布', False),
        'dv': ('方案偏离', False),
        'lb': ('实验室检查', False),
        'eg': ('心电图', False),
        'vs': ('生命体征', False),
        'pe': ('体格检查', False),
        'qs': ('问卷与量表', False),
        'ce': ('临床事件', False),
    }

    # 分析数据集命名前缀
    ANALYSIS_DATASET_PREFIX = 'ad'

    # 必需的分析数据集
    REQUIRED_ANALYSIS_DATASETS = {
        'adsl': '受试者水平分析数据集',
    }

    # 中国特定STF标签
    CHINA_STF_TAGS = {
        'data-tabulation-dataset-legacy': '原始数据库（非CDISC标准）',
        'data-tabulation-dataset-sdtm': '原始数据库（CDISC标准）',
        'data-tabulation-data-definition': '原始数据库数据说明文件、数据审阅说明',
        'analysis-dataset-legacy': '分析数据库（非CDISC标准）',
        'analysis-dataset-adam': '分析数据库（CDISC标准）',
        'analysis-data-definition': '分析数据库数据说明文件、数据审阅说明',
        'annotated-crf': '注释CRF',
        'analysis-program': '编程程序代码',
    }

    # 数据集名称模式（小写字母开头，仅包含小写字母和数字，≤8字节）
    DATASET_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9]{0,7}$')

    # 变量名称模式（大写字母开头，仅包含大写字母、数字和下划线，≤8字节）
    VARIABLE_NAME_PATTERN = re.compile(r'^[A-Z][A-Z0-9_]{0,7}$')

    def validate_dataset_naming(self, dataset_name: str, file_path: str) -> List[ViolationDetail]:
        """
        验证数据集命名规范

        规则:
        - 仅包含小写英文字母和数字
        - 必须以小写字母开头
        - 最大长度8个字节
        """
        violations = []

        if not dataset_name:
            violations.append(ViolationDetail(
                rule_id="3.8.10",
                severity=ViolationSeverity.ERROR,
                message="数据集名称为空",
                location=file_path,
                details="数据集必须有有效的名称",
                suggestion="为数据集提供符合规范的名称"
            ))
            return violations

        # 检查长度
        if len(dataset_name) > 8:
            violations.append(ViolationDetail(
                rule_id="3.8.10",
                severity=ViolationSeverity.ERROR,
                message=f"数据集名称超过8个字节: {dataset_name}",
                location=file_path,
                details=f"数据集名称 '{dataset_name}' 长度为 {len(dataset_name)} 字节，超过8字节限制",
                suggestion="缩短数据集名称至8个字节以内"
            ))

        # 检查格式
        if not self.DATASET_NAME_PATTERN.match(dataset_name):
            violations.append(ViolationDetail(
                rule_id="3.8.10",
                severity=ViolationSeverity.ERROR,
                message=f"数据集名称不符合规范: {dataset_name}",
                location=file_path,
                details="数据集名称只能包含小写英文字母和数字，且必须以小写字母开头",
                suggestion="修改数据集名称，例如: ae, dm, adsl等"
            ))

        # 检查是否为保留的数据集名称（如果需要）
        # 可以添加对标准数据集名称的建议

        return violations

    def validate_variable_naming(
        self,
        variable_name: str,
        dataset_name: str,
        file_path: str
    ) -> List[ViolationDetail]:
        """
        验证变量命名规范

        规则:
        - 仅包含大写英文字母、下划线和数字
        - 必须以大写字母开头
        - 最大长度8个字节
        """
        violations = []

        if not variable_name:
            violations.append(ViolationDetail(
                rule_id="3.8.11",
                severity=ViolationSeverity.ERROR,
                message=f"数据集 {dataset_name} 中存在空变量名",
                location=file_path,
                details="所有变量必须有有效的名称",
                suggestion="为变量提供符合规范的名称"
            ))
            return violations

        # 检查长度
        if len(variable_name) > 8:
            violations.append(ViolationDetail(
                rule_id="3.8.11",
                severity=ViolationSeverity.ERROR,
                message=f"变量名称超过8个字节: {variable_name}",
                location=f"{file_path}:{dataset_name}",
                details=f"变量 '{variable_name}' 长度为 {len(variable_name)} 字节，超过8字节限制",
                suggestion="缩短变量名称至8个字节以内"
            ))

        # 检查格式
        if not self.VARIABLE_NAME_PATTERN.match(variable_name):
            violations.append(ViolationDetail(
                rule_id="3.8.11",
                severity=ViolationSeverity.ERROR,
                message=f"变量名称不符合规范: {variable_name}",
                location=f"{file_path}:{dataset_name}",
                details="变量名称只能包含大写英文字母、下划线和数字，且必须以大写字母开头",
                suggestion="修改变量名称，例如: STUDYID, USUBJID, AETERM等"
            ))

        return violations

    def validate_required_identifiers(
        self,
        dataset: DatasetMetadata,
        file_path: str
    ) -> List[ViolationDetail]:
        """
        验证必需标识符

        规则:
        - 所有原始数据集必须包含STUDYID
        - 反映受试者观测结果的数据集必须包含USUBJID
        - dm数据集必须包含SUBJID
        """
        violations = []

        variable_names = {var['name'] for var in dataset.variables}

        # 确定需要哪些标识符
        required_ids = self.REQUIRED_IDENTIFIERS.copy()

        if dataset.name == 'dm':
            required_ids = self.DM_REQUIRED_IDENTIFIERS.copy()

        # 检查必需标识符是否存在
        for id_var, id_desc in required_ids.items():
            if id_var not in variable_names:
                severity = ViolationSeverity.CRITICAL if id_var == 'STUDYID' else ViolationSeverity.ERROR

                violations.append(ViolationDetail(
                    rule_id="3.8.12",
                    severity=severity,
                    message=f"数据集 {dataset.name} 缺少必需标识符: {id_var}",
                    location=f"{file_path}:{dataset.name}",
                    details=f"{id_desc} ({id_var}) 是必需变量",
                    suggestion=f"在数据集中添加 {id_var} 变量"
                ))

        # 检查推荐的时间变量（仅警告）
        if not dataset.is_analysis:  # 仅对原始数据集检查
            has_visit = any(var in variable_names for var in self.RECOMMENDED_TIME_VARS)
            if not has_visit and dataset.name != 'dm':  # dm数据集可能不需要访视变量
                violations.append(ViolationDetail(
                    rule_id="3.8.12",
                    severity=ViolationSeverity.INFO,
                    message=f"数据集 {dataset.name} 缺少推荐的时间变量",
                    location=f"{file_path}:{dataset.name}",
                    details=f"建议包含 VISIT (访视名称) 和/或 VISITNUM (访视编号) 变量",
                    suggestion="添加 VISIT 和 VISITNUM 变量以标识时间点"
                ))

        return violations

    def validate_labels_in_chinese(
        self,
        dataset: DatasetMetadata,
        file_path: str
    ) -> List[ViolationDetail]:
        """
        验证数据集标签和变量标签是否为中文

        规则:
        - 数据集标签应使用中文
        - 变量标签应使用中文
        - 标签长度不超过40字节
        - 不能包含不成对的引号、括号或特殊字符
        """
        violations = []

        # 检查数据集标签
        if dataset.label:
            label_issues = self._check_label_format(dataset.label, is_chinese_required=True)
            if label_issues:
                violations.append(ViolationDetail(
                    rule_id="3.8.13",
                    severity=ViolationSeverity.WARNING,
                    message=f"数据集 {dataset.name} 的标签格式问题",
                    location=f"{file_path}:{dataset.name}",
                    details=f"标签: '{dataset.label}' - {label_issues}",
                    suggestion="使用中文标签，长度不超过40字节，避免特殊字符"
                ))
        else:
            violations.append(ViolationDetail(
                rule_id="3.8.13",
                severity=ViolationSeverity.WARNING,
                message=f"数据集 {dataset.name} 缺少标签",
                location=f"{file_path}:{dataset.name}",
                details="数据集应包含中文标签以便审阅",
                suggestion="为数据集添加描述性的中文标签"
            ))

        # 检查变量标签
        for var in dataset.variables:
            var_name = var.get('name', '')
            var_label = var.get('label', '')

            if var_label:
                label_issues = self._check_label_format(var_label, is_chinese_required=True)
                if label_issues:
                    violations.append(ViolationDetail(
                        rule_id="3.8.13",
                        severity=ViolationSeverity.WARNING,
                        message=f"变量 {var_name} 的标签格式问题",
                        location=f"{file_path}:{dataset.name}.{var_name}",
                        details=f"标签: '{var_label}' - {label_issues}",
                        suggestion="使用中文标签，长度不超过40字节"
                    ))
            else:
                # 某些标准变量可能使用英文标签，仅给出INFO级别提示
                violations.append(ViolationDetail(
                    rule_id="3.8.13",
                    severity=ViolationSeverity.INFO,
                    message=f"变量 {var_name} 缺少标签",
                    location=f"{file_path}:{dataset.name}.{var_name}",
                    details="变量应包含中文标签以便审阅",
                    suggestion="为变量添加描述性的中文标签"
                ))

        return violations

    def validate_stf_tags_china(
        self,
        stf_tag: str,
        file_path: str
    ) -> Optional[ViolationDetail]:
        """
        验证中国特定的STF标签

        参数:
            stf_tag: STF file-tag的name属性值
            file_path: 文件路径

        返回:
            如果标签不在中国标准列表中，返回INFO级别的提示
        """
        if stf_tag and stf_tag not in self.CHINA_STF_TAGS:
            # 检查是否为ICH标准标签（允许）
            ich_tags = {
                'data-tabulation-dataset', 'analysis-dataset',
                'data-listing-dataset', 'annotated-crf', 'analysis-program'
            }
            if stf_tag in ich_tags:
                return None  # ICH标签也是可接受的

            return ViolationDetail(
                rule_id="3.8.14",
                severity=ViolationSeverity.INFO,
                message=f"使用了非中国标准的STF标签: {stf_tag}",
                location=file_path,
                details=f"建议使用中国药物临床试验数据递交指导原则中定义的标准标签",
                suggestion=f"参考标准标签: {', '.join(sorted(self.CHINA_STF_TAGS.keys()))}"
            )

        return None

    def validate_xpt_format(
        self,
        file_path: str,
        dataset_name: str,
        encoding: Optional[str] = None
    ) -> List[ViolationDetail]:
        """
        验证XPT格式规范

        规则:
        - 文件扩展名必须为.xpt
        - 数据集名称必须与XPT文件名一致
        - 应说明所用编码（utf-8, euc-cn等）
        """
        violations = []

        # 检查文件扩展名
        if not file_path.lower().endswith('.xpt'):
            violations.append(ViolationDetail(
                rule_id="3.8.15",
                severity=ViolationSeverity.ERROR,
                message="数据集文件扩展名不正确",
                location=file_path,
                details=f"数据集文件必须使用.xpt扩展名，当前为: {file_path}",
                suggestion="将文件扩展名修改为.xpt"
            ))

        # 检查文件名与数据集名称是否一致
        import os
        filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
        if filename_without_ext.lower() != dataset_name.lower():
            violations.append(ViolationDetail(
                rule_id="3.8.15",
                severity=ViolationSeverity.ERROR,
                message="XPT文件名与数据集名称不一致",
                location=file_path,
                details=f"文件名为 '{filename_without_ext}.xpt'，但数据集名称为 '{dataset_name}'",
                suggestion=f"将文件重命名为 {dataset_name}.xpt"
            ))

        # 检查编码说明
        if not encoding:
            violations.append(ViolationDetail(
                rule_id="3.8.15",
                severity=ViolationSeverity.WARNING,
                message="未说明数据集编码",
                location=file_path,
                details="应在数据审阅说明中说明所用编码（如utf-8、euc-cn等）",
                suggestion="在数据审阅说明文档中明确说明所有数据集使用的字符编码"
            ))

        return violations

    def validate_adsl_dataset(
        self,
        datasets: List[DatasetMetadata],
        file_paths: List[str]
    ) -> List[ViolationDetail]:
        """
        验证受试者水平分析数据集（ADSL）的存在和内容

        规则:
        - 分析数据库必须包含ADSL数据集
        - ADSL中每个受试者仅有一条记录
        - ADSL应包含人口学、基线特征、治疗组、分析人群划分等信息
        """
        violations = []

        # 查找ADSL数据集
        adsl_dataset = None
        adsl_path = None

        for idx, dataset in enumerate(datasets):
            if dataset.name.lower() == 'adsl':
                adsl_dataset = dataset
                adsl_path = file_paths[idx] if idx < len(file_paths) else 'unknown'
                break

        if not adsl_dataset:
            violations.append(ViolationDetail(
                rule_id="3.8.16",
                severity=ViolationSeverity.ERROR,
                message="缺少受试者水平分析数据集（ADSL）",
                location="分析数据库",
                details="ADSL是必需的分析数据集，应包含每个受试者的关键信息",
                suggestion="创建ADSL数据集，包含人口学、基线特征、治疗组等信息"
            ))
            return violations

        # 检查ADSL的必需变量（示例，实际可能需要更多）
        adsl_required_vars = ['STUDYID', 'USUBJID', 'SUBJID', 'AGE', 'SEX', 'RACE']
        variable_names = {var['name'] for var in adsl_dataset.variables}

        missing_vars = [var for var in adsl_required_vars if var not in variable_names]
        if missing_vars:
            violations.append(ViolationDetail(
                rule_id="3.8.16",
                severity=ViolationSeverity.WARNING,
                message=f"ADSL缺少推荐变量",
                location=adsl_path,
                details=f"缺少: {', '.join(missing_vars)}",
                suggestion="ADSL应包含基本的人口学变量（AGE, SEX, RACE等）"
            ))

        return violations

    def _check_label_format(self, label: str, is_chinese_required: bool = True) -> Optional[str]:
        """
        检查标签格式

        返回:
            如果有问题返回问题描述，否则返回None
        """
        issues = []

        # 检查长度
        label_bytes = label.encode('utf-8')
        if len(label_bytes) > 40:
            issues.append(f"长度 {len(label_bytes)} 字节超过40字节限制")

        # 检查是否包含中文
        if is_chinese_required:
            has_chinese = any('一' <= c <= '鿿' for c in label)
            if not has_chinese:
                issues.append("应使用中文标签")

        # 检查特殊字符
        if label.count("'") % 2 != 0 or label.count('"') % 2 != 0:
            issues.append("包含不成对的引号")

        if label.count('(') != label.count(')') or label.count('（') != label.count('）'):
            issues.append("包含不成对的括号")

        if any(c in label for c in ['<', '>']):
            issues.append("包含特殊字符 < 或 >")

        # 检查是否以数字开头
        if label and label[0].isdigit():
            issues.append("标签不应以数字开头")

        return '; '.join(issues) if issues else None

    def _contains_chinese(self, text: str) -> bool:
        """检查文本是否包含中文字符"""
        return any('一' <= c <= '鿿' for c in text)
