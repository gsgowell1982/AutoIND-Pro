"""
eCTD验证器集成模块

将Phase 2.10和Phase 2.12的验证器集成到material_assessment规则引擎中。

功能:
1. 将STF生命周期验证器转换为Rule对象
2. 将数据可追溯性验证器转换为Rule对象
3. 提供统一的规则注册接口

创建日期: 2026-09-14
"""

from typing import List, Dict, Optional, Any
from pathlib import Path

from core.rule_engine import Rule, RuleResult
from core.ectd_stf_lifecycle_validator import (
    STFLifecycleValidator,
    validate_stf_lifecycle_pair,
    ViolationSeverity as STFViolationSeverity
)
from core.ectd_data_traceability_validator import (
    DataTraceabilityValidator,
    ViolationSeverity as TraceViolationSeverity
)


class STFLifecycleRule(Rule):
    """STF生命周期验证规则"""

    def __init__(self):
        super().__init__(
            rule_id="STF-LIFECYCLE",
            requirement_id="ich_stf_specification:req_lifecycle_management",
            requirement_citation="ich_stf_specification#sec_1_8"
        )
        self.validator = STFLifecycleValidator()

    def evaluate(self, context: Dict[str, Any]) -> RuleResult:
        """
        评估STF生命周期规则

        context参数:
            current_sequence_path: 当前序列路径
            previous_sequence_path: 前序列路径（可选）
        """
        current_path = context.get('current_sequence_path')
        previous_path = context.get('previous_sequence_path')

        if not current_path:
            return RuleResult(
                rule_id=self.rule_id,
                passed=True,
                status="NOT_APPLICABLE",
                message="未提供序列路径"
            )

        try:
            violations = validate_stf_lifecycle_pair(current_path, previous_path)

            if not violations:
                return RuleResult(
                    rule_id=self.rule_id,
                    passed=True,
                    status="PASS",
                    message="STF生命周期验证通过"
                )

            # 转换违规为规则结果
            findings = []
            for violation in violations:
                severity_map = {
                    STFViolationSeverity.CRITICAL: "CRITICAL",
                    STFViolationSeverity.ERROR: "ERROR",
                    STFViolationSeverity.WARNING: "WARNING",
                    STFViolationSeverity.INFO: "INFO"
                }

                findings.append({
                    "rule_id": violation.rule_id,
                    "severity": severity_map.get(violation.severity, "ERROR"),
                    "message": violation.message,
                    "location": violation.location,
                    "details": violation.details,
                    "suggestion": violation.suggestion
                })

            return RuleResult(
                rule_id=self.rule_id,
                passed=False,
                status="FAIL",
                message=f"发现 {len(violations)} 个STF生命周期违规",
                findings=findings
            )

        except Exception as e:
            return RuleResult(
                rule_id=self.rule_id,
                passed=False,
                status="ERROR",
                message=f"STF生命周期验证失败: {str(e)}"
            )


class DataTraceabilityRule(Rule):
    """数据可追溯性验证规则"""

    def __init__(self):
        super().__init__(
            rule_id="DATA-TRACEABILITY",
            requirement_id="cdisc_data_submission:req_traceability",
            requirement_citation="cdisc_data_submission#sec_3_9"
        )
        self.validator = DataTraceabilityValidator()

    def evaluate(self, context: Dict[str, Any]) -> RuleResult:
        """
        评估数据可追溯性规则

        context参数:
            acrf_annotations: aCRF注释列表
            derivation_metadata: 衍生变量元数据列表
            raw_datasets: 原始数据集字典
            analysis_datasets: 分析数据集字典
        """
        acrf_annotations = context.get('acrf_annotations', [])
        derivation_metadata = context.get('derivation_metadata', [])
        raw_datasets = context.get('raw_datasets', {})
        analysis_datasets = context.get('analysis_datasets', {})

        # 如果没有提供任何数据，标记为不适用
        if not any([acrf_annotations, derivation_metadata]):
            return RuleResult(
                rule_id=self.rule_id,
                passed=True,
                status="NOT_APPLICABLE",
                message="未提供可追溯性验证数据"
            )

        try:
            # aCRF映射验证
            acrf_violations = []
            if acrf_annotations and raw_datasets:
                acrf_violations = self.validator.validate_acrf_mapping(
                    acrf_annotations,
                    raw_datasets
                )

            # 衍生变量可追溯性验证
            deriv_violations = []
            if derivation_metadata:
                deriv_violations = self.validator.validate_derivation_traceability(
                    derivation_metadata,
                    raw_datasets,
                    analysis_datasets
                )

            all_violations = acrf_violations + deriv_violations

            if not all_violations:
                return RuleResult(
                    rule_id=self.rule_id,
                    passed=True,
                    status="PASS",
                    message="数据可追溯性验证通过"
                )

            # 转换违规为规则结果
            findings = []
            for violation in all_violations:
                severity_map = {
                    TraceViolationSeverity.CRITICAL: "CRITICAL",
                    TraceViolationSeverity.ERROR: "ERROR",
                    TraceViolationSeverity.WARNING: "WARNING",
                    TraceViolationSeverity.INFO: "INFO"
                }

                findings.append({
                    "rule_id": violation.rule_id,
                    "severity": severity_map.get(violation.severity, "ERROR"),
                    "message": violation.message,
                    "location": violation.location,
                    "details": violation.details,
                    "suggestion": violation.suggestion
                })

            return RuleResult(
                rule_id=self.rule_id,
                passed=False,
                status="FAIL",
                message=f"发现 {len(all_violations)} 个数据可追溯性违规",
                findings=findings
            )

        except Exception as e:
            return RuleResult(
                rule_id=self.rule_id,
                passed=False,
                status="ERROR",
                message=f"数据可追溯性验证失败: {str(e)}"
            )


def register_advanced_validators(rule_engine) -> List[Rule]:
    """
    注册高级验证器到规则引擎

    参数:
        rule_engine: RuleEngine实例

    返回:
        注册的规则列表

    示例:
        from core.rule_engine import RuleEngine
        from core.ectd_validator_integration import register_advanced_validators

        engine = RuleEngine()
        rules = register_advanced_validators(engine)
        print(f"已注册 {len(rules)} 个高级验证规则")
    """
    rules = [
        STFLifecycleRule(),
        DataTraceabilityRule()
    ]

    for rule in rules:
        rule_engine.register_rule(rule)

    return rules


def create_stf_lifecycle_context(
    current_sequence_path: str,
    previous_sequence_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建STF生命周期验证上下文

    参数:
        current_sequence_path: 当前序列路径
        previous_sequence_path: 前序列路径（可选）

    返回:
        验证上下文字典
    """
    return {
        'current_sequence_path': current_sequence_path,
        'previous_sequence_path': previous_sequence_path
    }


def create_data_traceability_context(
    acrf_annotations: List = None,
    derivation_metadata: List = None,
    raw_datasets: Dict = None,
    analysis_datasets: Dict = None
) -> Dict[str, Any]:
    """
    创建数据可追溯性验证上下文

    参数:
        acrf_annotations: aCRF注释列表
        derivation_metadata: 衍生变量元数据列表
        raw_datasets: 原始数据集字典
        analysis_datasets: 分析数据集字典

    返回:
        验证上下文字典
    """
    return {
        'acrf_annotations': acrf_annotations or [],
        'derivation_metadata': derivation_metadata or [],
        'raw_datasets': raw_datasets or {},
        'analysis_datasets': analysis_datasets or {}
    }


# 便捷函数
def validate_with_rule_engine(
    rule_engine,
    current_sequence_path: str,
    previous_sequence_path: Optional[str] = None,
    acrf_annotations: List = None,
    derivation_metadata: List = None,
    raw_datasets: Dict = None,
    analysis_datasets: Dict = None
) -> Dict[str, RuleResult]:
    """
    使用规则引擎执行所有高级验证

    参数:
        rule_engine: RuleEngine实例
        current_sequence_path: 当前序列路径
        previous_sequence_path: 前序列路径
        acrf_annotations: aCRF注释列表
        derivation_metadata: 衍生变量元数据列表
        raw_datasets: 原始数据集字典
        analysis_datasets: 分析数据集字典

    返回:
        规则ID到RuleResult的映射

    示例:
        from core.rule_engine import RuleEngine
        from core.ectd_validator_integration import (
            register_advanced_validators,
            validate_with_rule_engine
        )

        engine = RuleEngine()
        register_advanced_validators(engine)

        results = validate_with_rule_engine(
            engine,
            current_sequence_path="./0001",
            previous_sequence_path="./0000"
        )

        for rule_id, result in results.items():
            print(f"{rule_id}: {result.status}")
    """
    results = {}

    # STF生命周期验证
    stf_context = create_stf_lifecycle_context(
        current_sequence_path,
        previous_sequence_path
    )
    stf_rule = STFLifecycleRule()
    results['STF-LIFECYCLE'] = stf_rule.evaluate(stf_context)

    # 数据可追溯性验证
    trace_context = create_data_traceability_context(
        acrf_annotations,
        derivation_metadata,
        raw_datasets,
        analysis_datasets
    )
    trace_rule = DataTraceabilityRule()
    results['DATA-TRACEABILITY'] = trace_rule.evaluate(trace_context)

    return results
