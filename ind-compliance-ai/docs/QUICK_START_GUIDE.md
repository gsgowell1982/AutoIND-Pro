# eCTD 3.8章节验证器 - 快速开始指南

## 快速开始

### 1. 运行演示

```bash
# 在项目根目录运行
python demo_chapter38_validator.py
```

预期输出：
```
================================================================================
eCTD 3.8章节验证器功能演示
================================================================================

=== 测试1: STF文件命名验证 ===
✓ 有效命名 'stf-abc123.xml': 0 个违规
✗ 无效命名 'abc123.xml': 1 个违规

=== 测试2: STF结构验证 ===
✓ 有效STF结构: 0 个CRITICAL违规
...

================================================================================
✓ 所有演示测试完成
================================================================================
```

### 2. 基本用法

#### 验证STF文件

```python
from core.ectd_chapter38_validator import ECTDChapter38Validator

# 创建验证器
validator = ECTDChapter38Validator()

# 验证STF文件
result = validator.validate_stf_file(
    stf_file_path="path/to/stf-study001.xml",
    module_path="m5/m5-3/m5-3-5/"
)

# 检查结果
if result.passed:
    print("✓ 验证通过")
else:
    print(f"✗ 发现 {result.total_violations} 个违规")
    print(f"  - CRITICAL: {result.critical_count}")
    print(f"  - ERROR: {result.error_count}")
    
    # 生成报告
    report = validator.generate_validation_report(result, "markdown")
    with open("stf_validation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
```

#### 验证数据集

```python
from core.ectd_china_data_validator import DatasetMetadata

# 构建数据集元数据（通常从XPT文件读取）
dataset = DatasetMetadata(
    name="dm",
    label="人口学数据",
    variables=[
        {'name': 'STUDYID', 'label': '研究标识符', 'type': 'char'},
        {'name': 'USUBJID', 'label': '受试者唯一标识符', 'type': 'char'},
        {'name': 'SUBJID', 'label': '受试者标识符', 'type': 'char'},
    ]
)

# 验证数据集
result = validator.validate_dataset(
    dataset_file_path="data/dm.xpt",
    dataset_metadata=dataset,
    validate_china_rules=True
)

if not result.passed:
    print("数据集验证失败：")
    for violation in result.violations:
        print(f"  [{violation.severity.value}] {violation.message}")
```

### 3. 报告格式

#### Markdown格式（推荐用于文档）

```python
report = validator.generate_validation_report(result, "markdown")
```

输出示例：
```markdown
# eCTD 3.8章节验证报告

## 验证摘要

- **总违规数**: 3
- **CRITICAL**: 1
- **ERROR**: 1
- **WARNING**: 1
- **验证状态**: ❌ 未通过

## 违规详情

### 🚫 [1] CRITICAL - 3.8.1

**位置**: `stf-test.xml`

**消息**: STF文件命名不符合规范

**详情**: 文件名必须以stf-开头

**建议**: 重命名为stf-{study-id}.xml
```

#### JSON格式（推荐用于集成）

```python
report = validator.generate_validation_report(result, "json")
```

输出示例：
```json
{
  "summary": {
    "total_violations": 3,
    "critical_count": 1,
    "error_count": 1,
    "warning_count": 1,
    "passed": false
  },
  "violations": [
    {
      "rule_id": "3.8.1",
      "severity": "CRITICAL",
      "message": "STF文件命名不符合规范",
      "location": "stf-test.xml",
      "details": "...",
      "suggestion": "..."
    }
  ]
}
```

#### 文本格式（推荐用于控制台）

```python
report = validator.generate_validation_report(result, "text")
print(report)
```

### 4. 常见验证场景

#### 场景1: 批量验证STF文件

```python
import os
from pathlib import Path

validator = ECTDChapter38Validator()
all_violations = []

# 遍历m5模块下所有STF文件
for root, dirs, files in os.walk("m5"):
    for file in files:
        if file.startswith("stf-") and file.endswith(".xml"):
            stf_path = os.path.join(root, file)
            
            with open(stf_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            result = validator.validate_stf_file(
                stf_file_path=stf_path,
                module_path=root,
                stf_content=content
            )
            
            all_violations.extend(result.violations)

# 生成综合报告
from core.ectd_chapter38_validator import ValidationResult
final_result = ValidationResult.from_violations(all_violations)

print(f"批量验证完成:")
print(f"  总文件数: {len([f for f in files if f.startswith('stf-')])}")
print(f"  总违规数: {final_result.total_violations}")
print(f"  通过状态: {'✓' if final_result.passed else '✗'}")
```

#### 场景2: 验证数据集集合

```python
# 假设已经读取了多个数据集
datasets = [dm_dataset, ae_dataset, lb_dataset, adsl_dataset]
file_paths = ["dm.xpt", "ae.xpt", "lb.xpt", "adsl.xpt"]

# 逐个验证
for dataset, path in zip(datasets, file_paths):
    result = validator.validate_dataset(path, dataset)
    if not result.passed:
        print(f"{dataset.name}: {result.total_violations} 个违规")

# 验证集合（检查ADSL等）
collection_result = validator.validate_dataset_collection(datasets, file_paths)
if collection_result.total_violations > 0:
    print("数据集集合验证问题：")
    for v in collection_result.violations:
        print(f"  - {v.message}")
```

#### 场景3: 集成到CI/CD流程

```python
#!/usr/bin/env python
"""
CI/CD验证脚本
返回值: 0=通过, 1=失败
"""
import sys
from core.ectd_chapter38_validator import ECTDChapter38Validator

def main():
    validator = ECTDChapter38Validator()
    
    # 验证所有STF文件
    results = []
    for stf_file in get_all_stf_files():
        result = validator.validate_stf_file(stf_file, ...)
        results.append(result)
    
    # 汇总
    total_critical = sum(r.critical_count for r in results)
    total_error = sum(r.error_count for r in results)
    
    if total_critical > 0 or total_error > 0:
        print(f"❌ 验证失败: {total_critical} CRITICAL, {total_error} ERROR")
        return 1
    else:
        print("✅ 验证通过")
        return 0

if __name__ == "__main__":
    sys.exit(main())
```

### 5. 常见问题排查

#### Q1: 如何修复"STF文件命名不符合规范"？

**违规信息**:
```
[ERROR] 3.8.1 - STF文件命名不符合规范: abc123.xml
详情: STF文件名必须遵循 'stf-{study-id}.xml' 格式
```

**解决方案**:
```bash
# 重命名文件
mv abc123.xml stf-abc123.xml
```

#### Q2: 如何修复"缺少必需标识符"？

**违规信息**:
```
[ERROR] 3.8.12 - 数据集 dm 缺少必需标识符: SUBJID
详情: 受试者标识符 (SUBJID) 是必需变量
```

**解决方案**:
在数据集中添加SUBJID变量，确保每个受试者都有唯一的标识符。

#### Q3: 如何修复"变量标签缺少中文"？

**违规信息**:
```
[WARNING] 3.8.13 - 变量 STUDYID 的标签格式问题
详情: 标签: 'Study ID' - 应使用中文标签
```

**解决方案**:
```python
# 修改变量标签为中文
variables = [
    {'name': 'STUDYID', 'label': '研究标识符', 'type': 'char'},  # ✓
    # 而非
    # {'name': 'STUDYID', 'label': 'Study ID', 'type': 'char'}  # ✗
]
```

#### Q4: 如何修复"无效的species值"？

**违规信息**:
```
[ERROR] 3.8.3 - species的category值无效: elephant
有效的species值: mouse, rat, hamster, other-rodent, rabbit, dog, ...
```

**解决方案**:
在STF文件中使用标准species值：
```xml
<category name="species" info-type="ich">mouse</category>
<!-- 而非 -->
<!-- <category name="species" info-type="ich">elephant</category> -->
```

#### Q5: 如何处理"数据集名称使用大写"？

**违规信息**:
```
[ERROR] 3.8.10 - 数据集名称不符合规范: AE
详情: 数据集名称只能包含小写英文字母和数字
```

**解决方案**:
```bash
# 重命名数据集文件
mv AE.xpt ae.xpt

# 同时修改数据集内部的名称属性
# 在SAS中: data ae; set AE; run;
```

### 6. 验证规则参考

#### STF相关规则 (3.8.1 - 3.8.6)
- ✅ 3.8.1: STF文件命名 `stf-{study-id}.xml`
- ✅ 3.8.2: STF XML结构完整性
- ✅ 3.8.3: Category元素有效值
- ✅ 3.8.4: File-tag标准符合性
- ✅ 3.8.5: Property元素要求
- ✅ 3.8.6: STF版本属性

#### 模块豁免规则 (3.8.7 - 3.8.9)
- ✅ 3.8.7: 5.2/5.3.6/5.4可不使用STF
- ✅ 3.8.8: 数据集位置验证
- ✅ 3.8.9: 数据集STF标签

#### 中国数据递交规范 (3.8.10 - 3.8.16)
- ✅ 3.8.10: 数据集命名规范
- ✅ 3.8.11: 变量命名规范
- ✅ 3.8.12: 必需标识符
- ✅ 3.8.13: 中文标签
- ✅ 3.8.14: 中国STF标签
- ✅ 3.8.15: XPT格式
- ✅ 3.8.16: ADSL数据集

### 7. 进阶使用

#### 自定义违规处理

```python
from core.ectd_chapter38_validator import ViolationSeverity

result = validator.validate_stf_file(...)

# 按严重程度分组
critical_violations = [v for v in result.violations if v.severity == ViolationSeverity.CRITICAL]
error_violations = [v for v in result.violations if v.severity == ViolationSeverity.ERROR]

# 只处理特定规则的违规
stf_naming_violations = [v for v in result.violations if v.rule_id == "3.8.1"]

# 按位置分组
from collections import defaultdict
by_location = defaultdict(list)
for v in result.violations:
    by_location[v.location].append(v)
```

#### 扩展验证器

```python
# 继承并添加自定义规则
class CustomValidator(ECTDChapter38Validator):
    def validate_custom_rule(self, data):
        violations = []
        
        # 自定义验证逻辑
        if not self._check_custom_condition(data):
            violations.append(ViolationDetail(
                rule_id="CUSTOM.1",
                severity=ViolationSeverity.WARNING,
                message="自定义规则违规",
                location="custom_location",
                details="详细说明",
                suggestion="修复建议"
            ))
        
        return ValidationResult.from_violations(violations)
```

### 8. 性能优化建议

#### 大批量验证优化

```python
from concurrent.futures import ThreadPoolExecutor

def validate_file(file_path):
    validator = ECTDChapter38Validator()
    return validator.validate_stf_file(file_path, ...)

# 并行验证多个文件
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(validate_file, stf_files))

# 汇总结果
all_violations = []
for result in results:
    all_violations.extend(result.violations)
```

### 9. 获取帮助

- 📖 **详细文档**: `docs/ECTD_3.8_PDF_PARSING_SUMMARY.md`
- 📊 **实施报告**: `docs/ECTD_3.8_IMPLEMENTATION_REPORT.md`
- 🧪 **测试用例**: `tests/test_ectd_chapter38_validator.py`
- 🎬 **演示脚本**: `demo_chapter38_validator.py`

### 10. 下一步

1. ✅ 运行演示脚本熟悉功能
2. ✅ 在测试数据上试用验证器
3. ✅ 集成到现有工作流
4. ✅ 根据实际需求调整报告格式
5. 📝 收集反馈并报告问题

---

**版本**: 1.0
**更新日期**: 2024年9月11日
**状态**: ✅ 生产就绪
