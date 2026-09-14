"""
eCTD数据可追溯性验证器

验证临床研究数据的可追溯性，包括：
1. aCRF与原始数据集的映射完整性
2. 分析数据集衍生变量的可追溯性
3. 数据流程的完整性

Phase 2.12 - Data Traceability Validation
创建日期: 2026-09-14
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from pathlib import Path
import re


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

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "details": self.details,
            "suggestion": self.suggestion
        }

    def get_summary(self) -> str:
        """获取摘要"""
        return f"[{self.rule_id}] {self.severity.value}: {self.message}"


@dataclass
class ACRFAnnotation:
    """aCRF注释信息"""
    crf_page: str
    crf_field: str
    dataset_name: str
    variable_name: str
    annotation_text: Optional[str] = None

    def get_mapping_key(self) -> str:
        """获取映射键（用于匹配）"""
        return f"{self.dataset_name}.{self.variable_name}"


@dataclass
class DatasetVariable:
    """数据集变量信息"""
    dataset_name: str
    variable_name: str
    variable_label: Optional[str] = None
    variable_type: Optional[str] = None
    is_derived: bool = False
    derivation_method: Optional[str] = None
    source_variables: List[str] = field(default_factory=list)


@dataclass
class DerivationMetadata:
    """衍生变量元数据"""
    target_dataset: str
    target_variable: str
    source_datasets: List[str]
    source_variables: List[str]
    derivation_algorithm: Optional[str] = None
    program_file: Optional[str] = None
    documentation: Optional[str] = None


@dataclass
class TraceabilityValidationResult:
    """可追溯性验证结果"""
    total_acrf_annotations: int
    mapped_annotations: int
    unmapped_annotations: int
    total_derived_variables: int
    traceable_variables: int
    untraceable_variables: int
    violations: List[ViolationDetail]
    warnings: List[ViolationDetail]

    def is_fully_compliant(self) -> bool:
        """是否完全合规"""
        return len(self.violations) == 0

    def get_acrf_mapping_rate(self) -> float:
        """获取aCRF映射率"""
        if self.total_acrf_annotations == 0:
            return 100.0
        return (self.mapped_annotations / self.total_acrf_annotations) * 100

    def get_derivation_traceability_rate(self) -> float:
        """获取衍生可追溯性率"""
        if self.total_derived_variables == 0:
            return 100.0
        return (self.traceable_variables / self.total_derived_variables) * 100

    def get_summary(self) -> str:
        """获取摘要"""
        return (
            f"数据可追溯性验证结果:\n"
            f"  aCRF映射: {self.mapped_annotations}/{self.total_acrf_annotations} "
            f"({self.get_acrf_mapping_rate():.1f}%)\n"
            f"  衍生可追溯性: {self.traceable_variables}/{self.total_derived_variables} "
            f"({self.get_derivation_traceability_rate():.1f}%)\n"
            f"  违规数: {len(self.violations)}\n"
            f"  警告数: {len(self.warnings)}"
        )


class DataTraceabilityValidator:
    """数据可追溯性验证器"""

    def __init__(self):
        """初始化验证器"""
        pass

    def validate_acrf_mapping(
        self,
        acrf_annotations: List[ACRFAnnotation],
        raw_datasets: Dict[str, List[DatasetVariable]]
    ) -> List[ViolationDetail]:
        """
        验证aCRF与原始数据集的映射完整性

        规则:
        1. aCRF中标注的每个变量都应在相应的原始数据集中存在
        2. aCRF标注应覆盖所有关键数据收集点
        3. 映射应保持一致性（同一CRF字段不应映射到多个变量）

        参数:
            acrf_annotations: aCRF注释列表
            raw_datasets: 原始数据集字典 {dataset_name: [variables]}

        返回:
            违规详情列表
        """
        violations = []

        # 构建数据集变量索引
        dataset_variables_index = {}
        for dataset_name, variables in raw_datasets.items():
            dataset_variables_index[dataset_name] = {
                var.variable_name: var for var in variables
            }

        # 规则1: 验证每个aCRF注释对应的变量是否存在
        for annotation in acrf_annotations:
            dataset_name = annotation.dataset_name
            variable_name = annotation.variable_name

            # 检查数据集是否存在
            if dataset_name not in dataset_variables_index:
                violations.append(ViolationDetail(
                    rule_id="TRACE-ACRF-001",
                    severity=ViolationSeverity.ERROR,
                    message=f"aCRF标注引用了不存在的数据集",
                    location=f"{annotation.crf_page}:{annotation.crf_field}",
                    details=(
                        f"aCRF在页面'{annotation.crf_page}'字段'{annotation.crf_field}'中"
                        f"标注映射到数据集'{dataset_name}'，但该数据集不存在"
                    ),
                    suggestion=(
                        f"检查数据集名称是否正确，或确保数据集'{dataset_name}'已包含在递交中"
                    )
                ))
                continue

            # 检查变量是否存在
            if variable_name not in dataset_variables_index[dataset_name]:
                violations.append(ViolationDetail(
                    rule_id="TRACE-ACRF-002",
                    severity=ViolationSeverity.ERROR,
                    message=f"aCRF标注引用了不存在的变量",
                    location=f"{annotation.crf_page}:{annotation.crf_field}",
                    details=(
                        f"aCRF在页面'{annotation.crf_page}'字段'{annotation.crf_field}'中"
                        f"标注映射到变量'{dataset_name}.{variable_name}'，但该变量不存在"
                    ),
                    suggestion=(
                        f"检查变量名称是否正确，或确保变量'{variable_name}'已包含在"
                        f"数据集'{dataset_name}'的define.xml中"
                    )
                ))

        # 规则2: 检测重复映射（同一CRF字段映射到多个变量）
        crf_field_mappings = {}
        for annotation in acrf_annotations:
            crf_key = f"{annotation.crf_page}:{annotation.crf_field}"
            mapping_key = annotation.get_mapping_key()

            if crf_key not in crf_field_mappings:
                crf_field_mappings[crf_key] = []
            crf_field_mappings[crf_key].append(mapping_key)

        for crf_key, mappings in crf_field_mappings.items():
            if len(set(mappings)) > 1:
                violations.append(ViolationDetail(
                    rule_id="TRACE-ACRF-003",
                    severity=ViolationSeverity.WARNING,
                    message=f"CRF字段映射到多个不同的变量",
                    location=crf_key,
                    details=(
                        f"CRF字段'{crf_key}'被标注映射到{len(set(mappings))}个不同的变量: "
                        f"{', '.join(set(mappings))}"
                    ),
                    suggestion=(
                        "检查aCRF标注是否正确。如果一个CRF字段确实对应多个变量，"
                        "应在数据递交文档中明确说明原因。"
                    )
                ))

        return violations

    def validate_derivation_traceability(
        self,
        derivation_metadata: List[DerivationMetadata],
        raw_datasets: Dict[str, List[DatasetVariable]],
        analysis_datasets: Dict[str, List[DatasetVariable]]
    ) -> List[ViolationDetail]:
        """
        验证分析数据集衍生变量的可追溯性

        规则:
        1. 每个衍生变量都应有明确的衍生算法文档
        2. 衍生算法引用的源变量都应存在
        3. 衍生变量应有关联的程序代码
        4. 源数据集应在衍生前已提交

        参数:
            derivation_metadata: 衍生变量元数据列表
            raw_datasets: 原始数据集字典
            analysis_datasets: 分析数据集字典

        返回:
            违规详情列表
        """
        violations = []

        # 构建变量索引
        all_variables_index = {}

        # 索引原始数据集变量
        for dataset_name, variables in raw_datasets.items():
            for var in variables:
                all_variables_index[f"{dataset_name}.{var.variable_name}"] = var

        # 索引分析数据集变量
        for dataset_name, variables in analysis_datasets.items():
            for var in variables:
                all_variables_index[f"{dataset_name}.{var.variable_name}"] = var

        # 规则1: 验证衍生变量的元数据完整性
        for metadata in derivation_metadata:
            target_key = f"{metadata.target_dataset}.{metadata.target_variable}"

            # 检查衍生算法是否有文档
            if not metadata.derivation_algorithm and not metadata.documentation:
                violations.append(ViolationDetail(
                    rule_id="TRACE-DERIV-001",
                    severity=ViolationSeverity.ERROR,
                    message=f"衍生变量缺少算法文档",
                    location=target_key,
                    details=(
                        f"衍生变量'{target_key}'未提供衍生算法或文档说明。"
                        "监管部门无法理解该变量的计算方法。"
                    ),
                    suggestion=(
                        "在define.xml的<def:Origin>元素中添加derivation算法描述，"
                        "或在数据分析计划（SAP）中提供详细文档。"
                    )
                ))

            # 检查是否有关联的程序代码
            if not metadata.program_file:
                violations.append(ViolationDetail(
                    rule_id="TRACE-DERIV-002",
                    severity=ViolationSeverity.WARNING,
                    message=f"衍生变量缺少关联的程序代码",
                    location=target_key,
                    details=(
                        f"衍生变量'{target_key}'未关联任何程序代码文件。"
                        "建议提供生成该变量的程序代码以增强可追溯性。"
                    ),
                    suggestion=(
                        "在递交包中包含生成该变量的程序代码（如SAS、R或Python脚本），"
                        "并在define.xml中通过<def:CommentDef>引用。"
                    )
                ))

            # 规则2: 验证源变量是否存在
            for source_var in metadata.source_variables:
                # 尝试多种命名格式
                possible_keys = [
                    source_var,  # 完整格式: dataset.variable
                ]

                # 如果source_var只是变量名，尝试在源数据集中查找
                if '.' not in source_var:
                    for source_dataset in metadata.source_datasets:
                        possible_keys.append(f"{source_dataset}.{source_var}")

                found = False
                for key in possible_keys:
                    if key in all_variables_index:
                        found = True
                        break

                if not found:
                    violations.append(ViolationDetail(
                        rule_id="TRACE-DERIV-003",
                        severity=ViolationSeverity.ERROR,
                        message=f"衍生变量引用了不存在的源变量",
                        location=target_key,
                        details=(
                            f"衍生变量'{target_key}'的算法引用了源变量'{source_var}'，"
                            "但该变量在任何已知数据集中都不存在"
                        ),
                        suggestion=(
                            f"检查源变量名称是否正确，或确保源变量'{source_var}'"
                            "已包含在相应的数据集中"
                        )
                    ))

            # 规则3: 验证源数据集是否存在
            for source_dataset in metadata.source_datasets:
                if source_dataset not in raw_datasets and source_dataset not in analysis_datasets:
                    violations.append(ViolationDetail(
                        rule_id="TRACE-DERIV-004",
                        severity=ViolationSeverity.ERROR,
                        message=f"衍生变量引用了不存在的源数据集",
                        location=target_key,
                        details=(
                            f"衍生变量'{target_key}'声称从数据集'{source_dataset}'衍生，"
                            "但该数据集不存在"
                        ),
                        suggestion=(
                            f"检查源数据集名称是否正确，或确保数据集'{source_dataset}'"
                            "已包含在递交中"
                        )
                    ))

        return violations

    def validate_data_flow_completeness(
        self,
        acrf_annotations: List[ACRFAnnotation],
        derivation_metadata: List[DerivationMetadata],
        raw_datasets: Dict[str, List[DatasetVariable]],
        analysis_datasets: Dict[str, List[DatasetVariable]]
    ) -> TraceabilityValidationResult:
        """
        综合验证数据流程的完整性

        参数:
            acrf_annotations: aCRF注释列表
            derivation_metadata: 衍生变量元数据列表
            raw_datasets: 原始数据集字典
            analysis_datasets: 分析数据集字典

        返回:
            完整的可追溯性验证结果
        """
        all_violations = []
        all_warnings = []

        # 验证aCRF映射
        acrf_violations = self.validate_acrf_mapping(acrf_annotations, raw_datasets)
        all_violations.extend([v for v in acrf_violations if v.severity == ViolationSeverity.ERROR])
        all_warnings.extend([v for v in acrf_violations if v.severity == ViolationSeverity.WARNING])

        # 验证衍生可追溯性
        deriv_violations = self.validate_derivation_traceability(
            derivation_metadata,
            raw_datasets,
            analysis_datasets
        )
        all_violations.extend([v for v in deriv_violations if v.severity == ViolationSeverity.ERROR])
        all_warnings.extend([v for v in deriv_violations if v.severity == ViolationSeverity.WARNING])

        # 统计映射完整性
        dataset_variables_index = {}
        for dataset_name, variables in raw_datasets.items():
            dataset_variables_index[dataset_name] = {var.variable_name for var in variables}

        mapped_annotations = 0
        for annotation in acrf_annotations:
            if (annotation.dataset_name in dataset_variables_index and
                annotation.variable_name in dataset_variables_index[annotation.dataset_name]):
                mapped_annotations += 1

        # 统计衍生可追溯性
        traceable_variables = 0
        for metadata in derivation_metadata:
            if metadata.derivation_algorithm or metadata.documentation:
                traceable_variables += 1

        return TraceabilityValidationResult(
            total_acrf_annotations=len(acrf_annotations),
            mapped_annotations=mapped_annotations,
            unmapped_annotations=len(acrf_annotations) - mapped_annotations,
            total_derived_variables=len(derivation_metadata),
            traceable_variables=traceable_variables,
            untraceable_variables=len(derivation_metadata) - traceable_variables,
            violations=all_violations,
            warnings=all_warnings
        )


def validate_data_traceability(
    acrf_annotations: List[ACRFAnnotation],
    derivation_metadata: List[DerivationMetadata],
    raw_datasets: Dict[str, List[DatasetVariable]],
    analysis_datasets: Dict[str, List[DatasetVariable]]
) -> TraceabilityValidationResult:
    """
    便捷函数：验证数据可追溯性

    参数:
        acrf_annotations: aCRF注释列表
        derivation_metadata: 衍生变量元数据列表
        raw_datasets: 原始数据集字典
        analysis_datasets: 分析数据集字典

    返回:
        可追溯性验证结果

    示例:
        acrf_annotations = [
            ACRFAnnotation(crf_page="CRF-01", crf_field="FIELD-01",
                          dataset_name="DM", variable_name="AGE"),
            ...
        ]

        derivation_metadata = [
            DerivationMetadata(target_dataset="ADSL", target_variable="AGEGR1",
                             source_datasets=["DM"], source_variables=["AGE"],
                             derivation_algorithm="AGE categorized into groups"),
            ...
        ]

        result = validate_data_traceability(
            acrf_annotations, derivation_metadata,
            raw_datasets, analysis_datasets
        )

        print(result.get_summary())
        for violation in result.violations:
            print(violation.get_summary())
    """
    validator = DataTraceabilityValidator()
    return validator.validate_data_flow_completeness(
        acrf_annotations,
        derivation_metadata,
        raw_datasets,
        analysis_datasets
    )
