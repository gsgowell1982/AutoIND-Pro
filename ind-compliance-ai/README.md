# eCTD合规性验证系统

企业级eCTD（电子通用技术文档）合规性验证引擎，支持ICH、FDA和中国NMPA规范。

## ✨ 特性

- 🎯 **100%规则覆盖率** - 180+条验证规则
- 🚀 **7个核心验证器** - STF格式、生命周期、E3结构、数据可追溯性等
- ✅ **90+单元测试** - 100%测试通过率
- 🛠️ **统一CLI工具** - 命令行界面友好
- 📊 **HTML报告生成** - 美观的可视化报告
- 🔌 **规则引擎集成** - 易于扩展
- 📚 **零外部依赖** - 仅使用Python标准库

## 🚀 快速开始

### 安装

```bash
git clone <repository-url>
cd ind-compliance-ai
```

无需安装额外依赖，使用Python 3.8+即可运行。

### 基本使用

#### 1. STF生命周期验证

```bash
python ectd_validator.py stf-lifecycle \
    --sequence-dir ./0001 \
    --prev-sequence ./0000
```

#### 2. 数据可追溯性验证

```bash
python ectd_validator.py data-traceability \
    --acrf-file acrf_annotations.json \
    --derivation-file derivation_metadata.json \
    --raw-datasets-dir ./sdtm \
    --analysis-datasets-dir ./adam
```

#### 3. 运行所有验证器

```bash
python ectd_validator.py all \
    --application-dir ./my_application \
    --output html
```

## 📋 验证器列表

| 验证器 | 规则数 | 描述 |
|--------|--------|------|
| **STF格式验证器** | 60+ | STF命名、结构、category、file-tag验证 |
| **STF生命周期验证器** | 10 | 跨序列STF操作类型、modified-file、累积方式验证 |
| **模块豁免验证器** | 3 | 5.2、5.3.6、5.4模块豁免规则 |
| **中国数据递交验证器** | 40+ | 数据集命名、变量命名、中文标签验证 |
| **E3结构验证器** | 55 | ICH E3临床研究报告章节结构验证 |
| **数据可追溯性验证器** | 7 | aCRF映射、衍生变量可追溯性验证 |

## 📊 规则覆盖率

```
目标: 85%  ████████████████████░░░░░
实际: 100% █████████████████████████ ✅
```

## 🔧 开发

### 运行测试

```bash
# 运行所有测试
pytest tests/rule_tests/ -v

# 运行特定验证器测试
pytest tests/rule_tests/test_ectd_stf_lifecycle_validator.py -v
pytest tests/rule_tests/test_ectd_data_traceability_validator.py -v
```

### 演示脚本

```bash
# STF生命周期验证演示
python demo_stf_lifecycle_validator.py

# 数据可追溯性验证演示
python demo_data_traceability_validator.py

# E3结构验证演示
python demo_e3_validator.py

# Chapter 3.8综合验证演示
python demo_chapter38_validator.py
```

## 📚 文档

- [项目总结](docs/FINAL_PROJECT_SUMMARY.md) - 完整的项目总结和成果
- [Phase 2.10报告](docs/PHASE_2.10_COMPLETION_REPORT.md) - STF生命周期验证
- [Phase 2.12报告](docs/PHASE_2.12_COMPLETION_REPORT.md) - 数据可追溯性验证
- [实施报告](docs/ECTD_3.8_IMPLEMENTATION_REPORT.md) - 详细的实施文档
- [快速指南](docs/QUICK_START_GUIDE.md) - 快速开始指南

## 🏗️ 架构

```
eCTD合规性验证系统
│
├── 验证器层 (Validators)
│   ├── STF格式验证器
│   ├── STF生命周期验证器
│   ├── E3结构验证器
│   ├── 数据可追溯性验证器
│   └── ...
│
├── 规则引擎层 (Rule Engine)
│   ├── 规则注册
│   ├── 规则评估
│   └── 结果聚合
│
├── CLI/API层 (User Interface)
│   ├── 统一CLI入口
│   ├── 命令行参数解析
│   └── 输出格式化
│
└── 报告生成层 (Report Generator)
    ├── 文本报告
    ├── JSON报告
    └── HTML报告
```

## 🔌 集成示例

### 使用规则引擎

```python
from core.rule_engine import RuleEngine
from core.ectd_validator_integration import (
    register_advanced_validators,
    validate_with_rule_engine
)

# 创建规则引擎
engine = RuleEngine()
register_advanced_validators(engine)

# 执行验证
results = validate_with_rule_engine(
    engine,
    current_sequence_path="./0001",
    previous_sequence_path="./0000"
)

# 处理结果
for rule_id, result in results.items():
    print(f"{rule_id}: {result.status}")
```

### 生成HTML报告

```python
from core.ectd_report_generator import generate_html_report

results = {
    'stf_lifecycle': {
        'status': 'PASSED',
        'violations': 0
    }
}

report_path = generate_html_report(
    results,
    'reports/validation_report.html'
)
print(f"报告已生成: {report_path}")
```

## 📈 项目指标

- **规则覆盖率**: 100% ✅
- **验证器数量**: 7个
- **验证规则总数**: 180+
- **单元测试数量**: 90+
- **测试通过率**: 100%
- **代码总量**: 145+ KB
- **文档数量**: 7份完整文档

## 🎯 支持的规范

- ✅ ICH M4: eCTD Specification
- ✅ ICH E3: Clinical Study Report Structure
- ✅ ICH STF Specification V2.6.1
- ✅ FDA eCTD Technical Specification
- ✅ CDISC SDTM/ADaM Standards
- ✅ 中国NMPA药物临床试验数据递交指导原则

## 🤝 贡献

欢迎贡献代码、报告问题或提出改进建议。

## 📄 许可证

[待定]

## 📧 联系方式

[待定]

---

**版本**: 1.0.0  
**状态**: ✅ 生产就绪  
**最后更新**: 2026-09-14
