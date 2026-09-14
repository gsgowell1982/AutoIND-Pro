"""
eCTD验证报告生成器

生成HTML格式的验证报告，包括：
1. 验证结果汇总
2. 违规详情列表
3. 统计图表
4. 建议和修复指南

创建日期: 2026-09-14
"""

from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json


class HTMLReportGenerator:
    """HTML报告生成器"""

    def __init__(self):
        self.report_template = self._get_template()

    def _get_template(self) -> str:
        """获取HTML模板"""
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>eCTD验证报告 - {timestamp}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .summary {{
            padding: 40px;
            background: #f8f9fa;
        }}

        .summary h2 {{
            margin-bottom: 20px;
            color: #495057;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}

        .stat-card .label {{
            color: #6c757d;
            font-size: 0.9em;
        }}

        .stat-card.passed .number {{ color: #28a745; }}
        .stat-card.failed .number {{ color: #dc3545; }}
        .stat-card.warning .number {{ color: #ffc107; }}
        .stat-card.info .number {{ color: #17a2b8; }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            color: #495057;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #dee2e6;
        }}

        .validator-result {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}

        .validator-header {{
            padding: 15px 20px;
            background: #f8f9fa;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .validator-header h3 {{
            margin: 0;
            color: #495057;
        }}

        .badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}

        .badge.passed {{
            background: #d4edda;
            color: #155724;
        }}

        .badge.failed {{
            background: #f8d7da;
            color: #721c24;
        }}

        .badge.skipped {{
            background: #fff3cd;
            color: #856404;
        }}

        .violations-list {{
            padding: 20px;
        }}

        .violation {{
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #dc3545;
            background: #fff5f5;
            border-radius: 4px;
        }}

        .violation.warning {{
            border-left-color: #ffc107;
            background: #fffbf0;
        }}

        .violation.info {{
            border-left-color: #17a2b8;
            background: #f0f9ff;
        }}

        .violation-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 10px;
        }}

        .violation-id {{
            font-weight: bold;
            color: #495057;
        }}

        .severity {{
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
        }}

        .severity.ERROR {{ background: #f8d7da; color: #721c24; }}
        .severity.WARNING {{ background: #fff3cd; color: #856404; }}
        .severity.INFO {{ background: #d1ecf1; color: #0c5460; }}

        .violation-message {{
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #212529;
        }}

        .violation-details {{
            color: #6c757d;
            margin-bottom: 10px;
        }}

        .violation-location {{
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 5px 10px;
            border-radius: 3px;
            margin-bottom: 10px;
            font-size: 0.9em;
        }}

        .violation-suggestion {{
            background: #d4edda;
            border-left: 3px solid #28a745;
            padding: 10px;
            margin-top: 10px;
            border-radius: 3px;
        }}

        .violation-suggestion strong {{
            color: #155724;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}

        .no-violations {{
            text-align: center;
            padding: 40px;
            color: #28a745;
        }}

        .no-violations .icon {{
            font-size: 4em;
            margin-bottom: 20px;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}

            .container {{
                box-shadow: none;
            }}

            .violation {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📋 eCTD验证报告</h1>
            <div class="subtitle">生成时间: {timestamp}</div>
        </div>

        <div class="summary">
            <h2>验证结果汇总</h2>
            <div class="stats-grid">
                {stats_cards}
            </div>
        </div>

        <div class="content">
            {validators_section}
        </div>

        <div class="footer">
            <p>eCTD合规性验证系统 v1.0.0</p>
            <p>报告生成器 © 2026</p>
        </div>
    </div>
</body>
</html>"""

    def generate_report(
        self,
        results: Dict,
        output_file: str,
        title: Optional[str] = None
    ):
        """
        生成HTML报告

        参数:
            results: 验证结果字典
            output_file: 输出文件路径
            title: 报告标题（可选）
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 统计数据
        total_violations = 0
        total_warnings = 0
        total_errors = 0
        passed_validators = 0
        failed_validators = 0

        for validator_name, result in results.items():
            if result.get('status') == 'PASSED':
                passed_validators += 1
            elif result.get('status') == 'FAILED':
                failed_validators += 1
                if 'violations' in result:
                    total_violations += result['violations']
                    # 统计ERROR和WARNING
                    if 'details' in result:
                        for violations_list in result['details'].values():
                            for v in violations_list:
                                if hasattr(v, 'severity'):
                                    if 'ERROR' in str(v.severity.value):
                                        total_errors += 1
                                    elif 'WARNING' in str(v.severity.value):
                                        total_warnings += 1

        # 生成统计卡片
        stats_cards = f"""
            <div class="stat-card {'passed' if passed_validators > 0 else ''}">
                <div class="number">{passed_validators}</div>
                <div class="label">验证通过</div>
            </div>
            <div class="stat-card {'failed' if failed_validators > 0 else ''}">
                <div class="number">{failed_validators}</div>
                <div class="label">验证失败</div>
            </div>
            <div class="stat-card failed">
                <div class="number">{total_errors}</div>
                <div class="label">错误</div>
            </div>
            <div class="stat-card warning">
                <div class="number">{total_warnings}</div>
                <div class="label">警告</div>
            </div>
        """

        # 生成验证器结果部分
        validators_html = ""
        for validator_name, result in results.items():
            validators_html += self._generate_validator_section(validator_name, result)

        # 渲染模板
        html_content = self.report_template.format(
            timestamp=timestamp,
            stats_cards=stats_cards,
            validators_section=validators_html
        )

        # 写入文件
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(output_path)

    def _generate_validator_section(self, validator_name: str, result: Dict) -> str:
        """生成单个验证器的HTML部分"""
        status = result.get('status', 'UNKNOWN')
        status_badge = f'<span class="badge {status.lower()}">{status}</span>'

        # 验证器标题映射
        validator_titles = {
            'stf_lifecycle': 'STF生命周期管理验证',
            'data_traceability': '数据可追溯性验证',
            'e3_structure': 'ICH E3结构验证',
            'china_data': '中国数据递交规范验证'
        }
        title = validator_titles.get(validator_name, validator_name)

        html = f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="validator-result">
                <div class="validator-header">
                    <h3>{title}</h3>
                    {status_badge}
                </div>
        """

        if status == 'PASSED':
            html += """
                <div class="no-violations">
                    <div class="icon">✅</div>
                    <p>所有检查通过，未发现违规</p>
                </div>
            """
        elif status == 'FAILED' and 'details' in result:
            html += '<div class="violations-list">'
            details = result['details']

            for seq_pair, violations in details.items():
                if violations:
                    html += f'<h4>序列对: {seq_pair}</h4>'
                    for violation in violations:
                        html += self._generate_violation_html(violation)

            html += '</div>'
        elif status == 'SKIPPED':
            html += """
                <div class="no-violations">
                    <div class="icon">⚠️</div>
                    <p>此验证被跳过</p>
                </div>
            """

        html += """
            </div>
        </div>
        """

        return html

    def _generate_violation_html(self, violation) -> str:
        """生成单个违规的HTML"""
        severity = violation.severity.value if hasattr(violation, 'severity') else 'INFO'
        severity_class = severity.lower()

        html = f"""
        <div class="violation {severity_class}">
            <div class="violation-header">
                <span class="violation-id">[{violation.rule_id}]</span>
                <span class="severity {severity}">{severity}</span>
            </div>
            <div class="violation-message">{violation.message}</div>
            <div class="violation-location">📍 位置: {violation.location}</div>
            <div class="violation-details">{violation.details}</div>
        """

        if violation.suggestion:
            html += f"""
            <div class="violation-suggestion">
                <strong>💡 建议:</strong> {violation.suggestion}
            </div>
            """

        html += "</div>"
        return html


def generate_html_report(results: Dict, output_file: str) -> str:
    """
    便捷函数：生成HTML报告

    参数:
        results: 验证结果字典
        output_file: 输出文件路径

    返回:
        生成的报告文件路径

    示例:
        results = {
            'stf_lifecycle': {
                'status': 'FAILED',
                'violations': 2,
                'details': {...}
            }
        }
        report_path = generate_html_report(results, 'reports/validation_report.html')
        print(f"报告已生成: {report_path}")
    """
    generator = HTMLReportGenerator()
    return generator.generate_report(results, output_file)
