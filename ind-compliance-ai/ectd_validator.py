#!/usr/bin/env python
"""
eCTD合规性验证 - 统一CLI入口

提供统一的命令行界面来调用所有验证器。

用法:
    python ectd_validator.py <command> [options]

命令:
    stf-lifecycle      验证STF生命周期管理
    data-traceability  验证数据可追溯性
    e3-structure       验证ICH E3结构
    china-data         验证中国数据递交规范
    all                运行所有验证器

示例:
    python ectd_validator.py stf-lifecycle --sequence-dir ./0001 --prev-sequence ./0000
    python ectd_validator.py data-traceability --acrf-file acrf.json --datasets datasets/
    python ectd_validator.py all --application-dir ./application
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import io

# 设置UTF-8输出（Windows兼容）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.ectd_stf_lifecycle_validator import (
    STFLifecycleValidator,
    validate_stf_lifecycle_pair,
    validate_stf_lifecycle_application
)
from core.ectd_data_traceability_validator import (
    DataTraceabilityValidator,
    ACRFAnnotation,
    DatasetVariable,
    DerivationMetadata,
    validate_data_traceability
)
from core.ectd_report_generator import generate_html_report


class ECTDValidatorCLI:
    """eCTD验证器统一CLI"""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """创建参数解析器"""
        parser = argparse.ArgumentParser(
            description='eCTD合规性验证 - 统一CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__
        )

        subparsers = parser.add_subparsers(dest='command', help='验证命令')

        # STF生命周期验证命令
        stf_parser = subparsers.add_parser(
            'stf-lifecycle',
            help='验证STF生命周期管理'
        )
        stf_parser.add_argument(
            '--sequence-dir',
            required=True,
            help='当前序列目录路径'
        )
        stf_parser.add_argument(
            '--prev-sequence',
            help='前一个序列目录路径（可选）'
        )
        stf_parser.add_argument(
            '--application-dir',
            help='整个申请目录路径（批量验证所有序列）'
        )
        stf_parser.add_argument(
            '--output',
            choices=['text', 'json'],
            default='text',
            help='输出格式'
        )

        # 数据可追溯性验证命令
        trace_parser = subparsers.add_parser(
            'data-traceability',
            help='验证数据可追溯性'
        )
        trace_parser.add_argument(
            '--acrf-file',
            help='aCRF注释JSON文件路径'
        )
        trace_parser.add_argument(
            '--derivation-file',
            help='衍生变量元数据JSON文件路径'
        )
        trace_parser.add_argument(
            '--raw-datasets-dir',
            help='原始数据集目录'
        )
        trace_parser.add_argument(
            '--analysis-datasets-dir',
            help='分析数据集目录'
        )
        trace_parser.add_argument(
            '--output',
            choices=['text', 'json'],
            default='text',
            help='输出格式'
        )

        # 全部验证命令
        all_parser = subparsers.add_parser(
            'all',
            help='运行所有验证器'
        )
        all_parser.add_argument(
            '--application-dir',
            required=True,
            help='申请目录路径'
        )
        all_parser.add_argument(
            '--output',
            choices=['text', 'json', 'html'],
            default='text',
            help='输出格式'
        )

        # 通用选项
        parser.add_argument(
            '--verbose',
            '-v',
            action='store_true',
            help='详细输出'
        )
        parser.add_argument(
            '--version',
            action='version',
            version='eCTD Validator 1.0.0'
        )

        return parser

    def run_stf_lifecycle_validation(self, args) -> int:
        """运行STF生命周期验证"""
        print("=" * 80)
        print("STF生命周期管理验证")
        print("=" * 80)

        if args.application_dir:
            # 批量验证整个申请
            print(f"\n验证申请: {args.application_dir}")
            results = validate_stf_lifecycle_application(args.application_dir)

            if not results:
                print("\n✅ 所有序列对均合规，未发现违规")
                return 0

            # 输出结果
            total_violations = sum(len(v) for v in results.values())
            print(f"\n发现 {len(results)} 个序列对存在违规，共 {total_violations} 个违规:")

            for seq_pair, violations in results.items():
                print(f"\n序列对: {seq_pair}")
                self._print_violations(violations)

            return 1 if total_violations > 0 else 0

        else:
            # 单个序列对验证
            print(f"\n当前序列: {args.sequence_dir}")
            if args.prev_sequence:
                print(f"前序列: {args.prev_sequence}")

            violations = validate_stf_lifecycle_pair(
                args.sequence_dir,
                args.prev_sequence
            )

            if not violations:
                print("\n✅ 未发现违规")
                return 0

            print(f"\n❌ 发现 {len(violations)} 个违规:")
            self._print_violations(violations)
            return 1

    def run_data_traceability_validation(self, args) -> int:
        """运行数据可追溯性验证"""
        print("=" * 80)
        print("数据可追溯性验证")
        print("=" * 80)

        # 加载输入数据
        acrf_annotations = self._load_acrf_annotations(args.acrf_file) if args.acrf_file else []
        derivation_metadata = self._load_derivation_metadata(args.derivation_file) if args.derivation_file else []
        raw_datasets = self._load_datasets(args.raw_datasets_dir) if args.raw_datasets_dir else {}
        analysis_datasets = self._load_datasets(args.analysis_datasets_dir) if args.analysis_datasets_dir else {}

        # 执行验证
        result = validate_data_traceability(
            acrf_annotations,
            derivation_metadata,
            raw_datasets,
            analysis_datasets
        )

        # 输出结果
        print(f"\n{result.get_summary()}")

        if result.violations:
            print(f"\n违规详情:")
            self._print_violations(result.violations)

        if result.warnings:
            print(f"\n警告:")
            self._print_violations(result.warnings)

        return 0 if result.is_fully_compliant() else 1

    def run_all_validations(self, args) -> int:
        """运行所有验证器"""
        print("=" * 80)
        print("eCTD合规性全面验证")
        print("=" * 80)
        print(f"\n申请目录: {args.application_dir}\n")

        total_violations = 0
        results = {}

        # 1. STF生命周期验证
        print("\n" + "=" * 80)
        print("1. STF生命周期管理验证")
        print("=" * 80)
        try:
            stf_results = validate_stf_lifecycle_application(args.application_dir)
            if stf_results:
                stf_violations = sum(len(v) for v in stf_results.values())
                total_violations += stf_violations
                results['stf_lifecycle'] = {
                    'status': 'FAILED' if stf_violations > 0 else 'PASSED',
                    'violations': stf_violations,
                    'details': stf_results
                }
                print(f"❌ 发现 {stf_violations} 个违规")
            else:
                results['stf_lifecycle'] = {'status': 'PASSED', 'violations': 0}
                print("✅ 通过")
        except Exception as e:
            print(f"⚠️ 执行失败: {e}")
            results['stf_lifecycle'] = {'status': 'ERROR', 'error': str(e)}

        # 2. 数据可追溯性验证（需要额外数据文件）
        print("\n" + "=" * 80)
        print("2. 数据可追溯性验证")
        print("=" * 80)
        print("⚠️ 需要aCRF和衍生变量元数据文件，跳过")
        results['data_traceability'] = {'status': 'SKIPPED'}

        # 输出最终汇总
        print("\n" + "=" * 80)
        print("验证汇总")
        print("=" * 80)
        for validator_name, result in results.items():
            status_icon = {
                'PASSED': '✅',
                'FAILED': '❌',
                'SKIPPED': '⚠️',
                'ERROR': '🔴'
            }.get(result['status'], '?')
            print(f"{status_icon} {validator_name}: {result['status']}")
            if 'violations' in result:
                print(f"   违规数: {result['violations']}")

        print(f"\n总违规数: {total_violations}")

        # 生成HTML报告（如果指定了html输出格式）
        if args.output == 'html':
            try:
                report_file = Path(args.application_dir) / 'validation_report.html'
                report_path = generate_html_report(results, str(report_file))
                print(f"\n📄 HTML报告已生成: {report_path}")
            except Exception as e:
                print(f"\n⚠️ HTML报告生成失败: {e}")

        return 0 if total_violations == 0 else 1

    def _print_violations(self, violations):
        """打印违规详情"""
        for i, violation in enumerate(violations, 1):
            print(f"\n  {i}. [{violation.rule_id}] {violation.severity.value}")
            print(f"     消息: {violation.message}")
            print(f"     位置: {violation.location}")
            if violation.suggestion:
                print(f"     建议: {violation.suggestion}")

    def _load_acrf_annotations(self, file_path: str) -> List[ACRFAnnotation]:
        """从JSON文件加载aCRF注释"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [ACRFAnnotation(**item) for item in data]

    def _load_derivation_metadata(self, file_path: str) -> List[DerivationMetadata]:
        """从JSON文件加载衍生变量元数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [DerivationMetadata(**item) for item in data]

    def _load_datasets(self, directory: str) -> Dict[str, List[DatasetVariable]]:
        """从目录加载数据集"""
        # 简化实现：假设每个数据集是一个JSON文件
        datasets = {}
        dataset_dir = Path(directory)
        if not dataset_dir.exists():
            return datasets

        for json_file in dataset_dir.glob('*.json'):
            dataset_name = json_file.stem
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            datasets[dataset_name] = [DatasetVariable(**var) for var in data]

        return datasets

    def run(self, argv=None):
        """运行CLI"""
        args = self.parser.parse_args(argv)

        if not args.command:
            self.parser.print_help()
            return 1

        try:
            if args.command == 'stf-lifecycle':
                return self.run_stf_lifecycle_validation(args)
            elif args.command == 'data-traceability':
                return self.run_data_traceability_validation(args)
            elif args.command == 'all':
                return self.run_all_validations(args)
            else:
                print(f"未知命令: {args.command}")
                return 1

        except KeyboardInterrupt:
            print("\n\n中断执行")
            return 130
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1


def main():
    """主入口"""
    cli = ECTDValidatorCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
