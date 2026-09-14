# eCTD 3.8章节 - PDF文件解析总结

## 文档概览

本文档汇总了从3个外部PDF文件中提取的关键规则，用于补全eCTD 3.8章节（研究报告和STF）的验证逻辑。

### 已解析的PDF文件

1. **STFV2-6-1_0.pdf** - ICH eCTD STF Specification V2.6.1 (22页)
2. **E3_Guideline.pdf** - ICH E3 临床研究报告结构和内容 (49页)
3. **《药物临床试验数据递交指导原则（试行）》.pdf** - 中国药监局临床试验数据递交指南 (21页)

---

## 一、STFV2-6-1_0.pdf 关键规则提取

### 1.1 STF文件命名规范
- **规则**: STF文件名必须以 "stf-" 开头，后跟研究编号(study-id)，以 ".xml" 结尾
- **示例**: `stf-abc123xyz789.xml`, `stf-jm-12-345.xml`
- **验证点**: 文件路径匹配模式 `^stf-[a-zA-Z0-9_-]+\.xml$`

### 1.2 STF位置要求
- **规则**: STF应与对应的研究文件放在同一模块文件夹内
- **适用模块**: 4.2.X (非临床) 和 5.3.1.X-5.3.5.X (临床)
- **验证点**: STF文件路径应在同一父目录下

### 1.3 STF操作类型限制
- **首次提交**: operation="new"
- **后续提交**: operation="append" (必须引用最近一次的STF)
- **禁止操作**: 不应使用 "replace" 或 "delete" 操作于STF文件本身
- **验证点**: 检查index.xml中STF leaf节点的operation属性

### 1.4 STF版本属性
- **规则**: STF的leaf元素必须包含version属性，引用DTD版本
- **示例**: `version="STF version 2.2"`
- **验证点**: 检查version属性是否存在且格式正确

### 1.5 STF必需元素结构

#### 1.5.1 study-identifier元素
- **title**: 研究完整标题（非单个文档标题）
- **study-id**: 申办方用于唯一标识研究的内部代码
- **category**: 额外的研究分类（仅适用于特定章节）
  - 4.2.3.1 单次给药毒性: species + route-of-admin
  - 4.2.3.2 重复给药毒性: species + route-of-admin + duration
  - 4.2.3.4.1 长期致癌性: species
  - 5.3.5.1 对照临床研究: type-of-control

#### 1.5.2 category元素有效值

**species (物种):**
- mouse, rat, hamster, other-rodent
- rabbit, dog, non-human-primate
- other-non-rodent-mammal, non-mammals

**route-of-admin (给药途径):**
- oral, intravenous, intramuscular, intraperitoneal
- subcutaneous, inhalation, topical, other

**duration (持续时间) - US特定:**
- short, medium, long

**type-of-control (对照类型):**
- placebo, no-treatment, dose-response-without-placebo
- active-control-without-placebo, external

### 1.6 file-tag元素有效值（ICH标准）

#### 1.6.1 非临床研究
- **pre-clinical-study-report** (info-type="ich"): 非临床研究报告

#### 1.6.2 临床研究 - 核心文档
- **legacy-clinical-study-report**: 未按ICH E3编制的单文件临床研究报告
- **synopsis**: 研究摘要
- **study-report-body**: 研究报告正文

#### 1.6.3 临床研究 - 支持性文档 (16.1.X)
- **protocol-or-amendment**: 方案和/或修订
- **sample-case-report-form**: 样本CRF
- **iec-irb-consent-form-list**: 伦理委员会和知情同意书清单
- **list-description-investigator-site**: 研究者和研究中心描述
- **signatures-investigators**: 主要研究者或申办方负责人签名
- **list-patients-with-batches**: 接受特定批次试验药物的受试者清单
- **randomisation-scheme**: 随机化方案
- **audit-certificates-report**: 审计证书
- **statistical-methods-interim-analysis-plan**: 统计方法和中期分析计划文档
- **inter-laboratory-standardisation-methods-quality-assurance**: 实验室间标准化方法和质量保证文档
- **publications-based-on-study**: 基于研究的发表
- **publications-referenced-in-report**: 报告中引用的发表

#### 1.6.4 临床研究 - 受试者数据清单 (16.2.X)
- **discontinued-patients**: 中止受试者清单
- **protocol-deviations**: 方案偏离清单
- **patients-excluded-from-efficacy-analysis**: 排除在疗效分析外的受试者
- **demographic-data**: 人口学数据清单
- **compliance-and-drug-concentration-data**: 依从性和/或药物浓度数据
- **individual-efficacy-response-data**: 个体疗效反应数据
- **adverse-event-listings**: 不良事件清单
- **listing-individual-laboratory-measurements-by-patient**: 按受试者列出的个体实验室测量值
- **case-report-forms**: 个体受试者的CRF（需要site-identifier属性）

#### 1.6.5 US特定标签
- **data-tabulation-dataset**: 数据表格数据集
- **data-tabulation-data-definition**: 数据表格数据集定义
- **data-listing-dataset**: 数据清单数据集
- **data-listing-data-definition**: 数据清单数据集定义
- **analysis-dataset**: 分析数据集
- **analysis-program**: 分析数据集的程序文件
- **analysis-data-definition**: 分析数据集的数据定义
- **annotated-crf**: 数据集的注释CRF
- **ecg**: 注释的ECG波形数据集
- **image**: 图像文件
- **subject-profiles**: 受试者概况（需要site-identifier属性）
- **safety-report**: IND安全性报告
- **antibacterial**, **special-pathogen**, **antiviral**: 微生物学报告类型
- **iss**: 综合安全性总结报告
- **ise**: 综合疗效总结报告
- **pm-description**: 上市后定期不良事件报告描述

### 1.7 property元素（site-identifier）
- **适用场景**: case-report-forms 和 subject-profiles 标签在美国递交时
- **属性**: name="site-identifier", info-type="us"
- **内容**: 研究中心的标识文本

### 1.8 STF生命周期管理规则

#### 1.8.1 累积方式（Accumulative Approach）- 唯一支持的方式
- 后续STF仅包含新增或修改的文档信息
- 使用 modified-file 属性引用最近一次提交的STF
- 不需要重复提交完整的file-tag枚举

#### 1.8.2 修改study-identifier信息
- 提交新的STF，operation="append"
- 包含完整的study-identifier块（所有category值）
- 包含空的 `<study-document/>` 元素（如果无新文档）

#### 1.8.3 添加新文件
- 提交新STF，operation="append"
- 仅包含新增leaf元素的引用

#### 1.8.4 删除文件
- **不提交新STF**
- 在index.xml中将leaf元素operation设为"delete"

#### 1.8.5 纠正file-tag值
- 在index.xml中删除错误的leaf（operation="delete"）
- 添加新leaf，operation="new"，xlink:href指向原始文件位置
- 提交新STF，operation="append"，包含正确的file-tag

---

## 二、E3_Guideline.pdf 关键规则提取

### 2.1 ICH E3临床研究报告标准结构

#### 2.1.1 必需章节（Mandatory Sections）
1. **Title Page** (标题页)
2. **Synopsis** (摘要 - 通常限3页)
3. **Table of Contents** (目录)
4. **List of Abbreviations and Definition of Terms** (缩写和术语定义列表)
5. **Ethics** (伦理)
   - 5.1 IEC/IRB
   - 5.2 伦理行为
   - 5.3 受试者信息和同意
6. **Investigators and Study Administrative Structure** (研究者和管理结构)
7. **Introduction** (引言 - 最多1页)
8. **Study Objectives** (研究目的)
9. **Investigational Plan** (研究计划)
   - 9.1 总体设计
   - 9.2 设计讨论
   - 9.3 受试者选择
   - 9.4 治疗
   - 9.5 疗效和安全性变量
   - 9.6 数据质量保证
   - 9.7 统计方法
   - 9.8 研究实施变更
10. **Study Patients** (研究受试者)
    - 10.1 受试者分布
    - 10.2 方案偏离
11. **Efficacy Evaluation** (疗效评价)
    - 11.1 分析数据集
    - 11.2 人口学和基线特征
    - 11.3 治疗依从性测量
    - 11.4 疗效结果和个体受试者数据表格
12. **Safety Evaluation** (安全性评价)
    - 12.1 暴露程度
    - 12.2 不良事件
    - 12.3 死亡、其他严重不良事件和其他重要不良事件
    - 12.4 临床实验室评估
    - 12.5 生命体征、体格检查发现和其他安全性观察
    - 12.6 安全性结论
13. **Discussion and Overall Conclusions** (讨论和总体结论)
14. **Tables, Figures and Graphs** (表格、图形和图表)
    - 14.1 人口学数据
    - 14.2 疗效数据
    - 14.3 安全性数据
15. **Reference List** (参考文献列表)
16. **Appendices** (附录)
    - 16.1 研究信息
    - 16.2 受试者数据清单
    - 16.3 病例报告表
    - 16.4 个体受试者数据清单（US档案清单）

#### 2.1.2 章节16.1详细内容（Study Information）
- 16.1.1 方案和方案修订
- 16.1.2 样本病例报告表（仅唯一页面）
- 16.1.3 IEC/IRB清单和知情同意书样本
- 16.1.4 研究者清单和简历
- 16.1.5 主要研究者或申办方负责医学官签名
- 16.1.6 接受特定批次试验药物的受试者清单
- 16.1.7 随机化方案和代码
- 16.1.8 审计证书（如有）
- 16.1.9 统计方法文档
- 16.1.10 实验室间标准化方法和质量保证程序文档
- 16.1.11 基于研究的发表
- 16.1.12 报告中引用的重要发表

#### 2.1.3 章节16.2详细内容（Patient Data Listings）
- 16.2.1 中止受试者
- 16.2.2 方案偏离
- 16.2.3 排除在疗效分析外的受试者
- 16.2.4 人口学数据
- 16.2.5 依从性和/或药物浓度数据（如有）
- 16.2.6 个体疗效反应数据
- 16.2.7 不良事件清单（每个受试者）
- 16.2.8 按受试者列出的个体实验室测量值（监管机构要求时）

### 2.2 E3合规性验证要点

#### 2.2.1 结构完整性
- 报告必须包含所有必需章节（1-16）
- 章节编号和标题应遵循E3标准
- 附录16.1和16.2的子章节应完整

#### 2.2.2 内容要求
- **Synopsis**: 应限制在3页以内，包含数值数据（不仅是p值）
- **Introduction**: 最多1页
- **Table标识**: 所有分析、表格和图形必须清楚标识数据来源的受试者集合

#### 2.2.3 特殊场景
- **简化报告**: 可用于非对照研究、设计严重缺陷的研究、或与申报适应症无关的对照研究
- **安全性**: 即使是简化报告，也必须包含完整的安全性描述
- **大型试验**: 对于非常大的试验，某些条款可能不切实际，应与监管机构讨论

---

## 三、《药物临床试验数据递交指导原则（试行）》关键规则提取

### 3.1 数据库递交要求

#### 3.1.1 原始数据库（Raw/Tabulation Database）
- **定义**: 包含直接从CRF和外部文件收集的原始数据
- **标准化**: 可进行必要的标准化（数据集名称/标签/结构、变量名称/标签、变量值编码）
- **编码**: 应使用MedDRA等标准医学词典
- **CDISC等价物**: SDTM数据库视为原始数据库
- **缺失值**: 不应进行填补

#### 3.1.2 必需标识符
所有原始数据集必须包含:
- **STUDYID**: 研究标识符（研究编号）
- **USUBJID**: 受试者唯一标识符（在整个试验申请中保持一致）
- **SUBJID**: 受试者标识符（必须在dm数据集中）
- **VISIT/VISITNUM**: 访视名称和编号（适用的数据集）

#### 3.1.3 分析数据库（Analysis Database）
- **定义**: 为统计分析衍生的数据库，包含原始数据和衍生数据
- **CDISC等价物**: ADaM数据库视为分析数据库
- **命名约定**: 以"ad"开头（如adcm, adae, adlb）
- **ADSL**: 受试者水平分析数据集（每个受试者仅一条记录）- 必需
- **可追溯性**: 所有衍生变量必须能从原始数据库生成

#### 3.1.4 常用原始数据集（见附录1）
| 数据集 | 命名 | 递交要求 |
|--------|------|----------|
| 人口学 | dm | 必须递交 |
| 病史 | mh | 如适用 |
| 不良事件 | ae | 如适用 |
| 既往与合并用药 | cm | 如适用 |
| 暴露 | ex | 如适用 |
| 受试者分布 | ds | 如适用 |
| 问卷与量表 | qs | 如适用 |
| 方案偏离 | dv | 如适用 |
| 实验室检查 | lb | 如适用 |
| 心电图 | eg | 如适用 |
| 生命体征 | vs | 如适用 |
| 临床事件 | ce | 如适用 |
| 体格检查 | pe | 如适用 |

### 3.2 数据说明文件要求

#### 3.2.1 必需内容
- 数据集名称、标签、基本结构描述
- 每个变量的名称、标签、类型
- 变量来源或衍生过程
- 编码列表和来源的清晰定义
- 外部词典名称和版本（如适用）

#### 3.2.2 可追溯性
- 原始数据集与CRF之间的映射
- 分析数据集与原始数据集之间的映射
- 衍生变量的详细说明（必要时使用程序代码辅助）

#### 3.2.3 格式
- XML格式或PDF格式
- 如递交XML，必须同时递交XSL文件

### 3.3 数据审阅说明（Data Reviewer's Guide）

#### 3.3.1 推荐内容
- 研究数据使用说明
- 临床总结报告与数据之间的关系
- 研究文档中的关键信息（方案、统计分析计划、CSR）
- 程序代码使用说明
- 数据集编码（utf-8, euc-cn等）
- 其他特殊情形说明

#### 3.3.2 格式
- 必须为PDF文件

### 3.4 注释病例报告表（aCRF）

#### 3.4.1 内容
- 空白CRF基础上标注CRF字段与数据集变量之间的映射关系
- 未递交数据标记为"NOT SUBMITTED"并说明理由

#### 3.4.2 格式
- 必须为PDF文件

### 3.5 程序代码要求

#### 3.5.1 必需递交的代码
- 分析数据集衍生变量的衍生过程
- 疗效指标分析结果的生成过程

#### 3.5.2 质量要求
- 易懂、可读性强
- 提供充分注释
- 避免外部(宏)程序调用

#### 3.5.3 格式
- TXT文件

### 3.6 数据格式规范

#### 3.6.1 XPT格式（研究数据传输格式）
- 一个XPT文件对应一个数据集
- 数据集名称必须与XPT文件名一致
- 文件扩展名统一为.xpt
- 推荐使用XPT V5或以上版本
- 必须说明所用编码（utf-8, euc-cn等）

#### 3.6.2 命名规范
**数据集名称:**
- 仅包含小写英文字母和数字
- 必须以小写字母开头
- 最大长度8个字节

**变量名称:**
- 仅包含大写英文字母、下划线和数字
- 必须以字母开头
- 最大长度8个字节

**字符型变量长度:**
- 设置为该变量在所有数据集中的最大实际值长度

#### 3.6.3 标签规范
**数据集标签和变量标签:**
- 应使用中文
- 长度不超过40字节
- 可包含英文字符、下划线或数字（不能以数字开头）
- 不能包含:
  - 不成对的半角或全角单引号、双引号
  - 不成对的半角或全角括号
  - 特殊字符（如'>'、'<'）

### 3.7 eCTD下的STF标签（附录2）

| name属性值 | 说明 |
|-----------|------|
| data-tabulation-dataset-legacy | 原始数据库（非CDISC标准）|
| data-tabulation-dataset-sdtm | 原始数据库（CDISC标准）|
| data-tabulation-data-definition | 原始数据库数据说明文件、数据审阅说明 |
| analysis-dataset-legacy | 分析数据库（非CDISC标准）|
| analysis-dataset-adam | 分析数据库（CDISC标准）|
| analysis-data-definition | 分析数据库数据说明文件、数据审阅说明 |
| annotated-crf | 注释CRF |
| analysis-program | 编程程序代码 |

### 3.8 中文翻译最低要求

#### 3.8.1 数据库
至少以下内容应为中文:
- 数据集标签和变量标签
- 临床总结报告中出现的不良事件名称
- 合并用药名称
- 病史名称

#### 3.8.2 数据说明文件
至少以下内容应为中文:
- 数据集描述/标签和说明
- 变量描述/标签和衍生过程
- 涉及疗效指标的取值或编码列表

#### 3.8.3 aCRF
至少以下内容应为中文:
- 为收集数据设计的问题描述
- 涉及疗效指标问题的取值或编码

#### 3.8.4 数据审阅说明
- 应为中文

### 3.9 数据可追溯性要求

#### 3.9.1 可追溯性定义
审评人员能够:
- 理解分析数据集的构建
- 确定用于衍生变量的观测记录和算法
- 理解统计结果的计算方法
- 建立从原始数据到报表之间的关联

#### 3.9.2 验证要求
- 监管部门能够利用原始数据库衍生出与申办方一致的分析数据库
- 利用分析数据库能够直接重现与申办方一致的统计分析结果
- 建议提供数据从收集到递交的详细流程图

---

## 四、当前实现状态与增强建议

### 4.1 已实现的验证规则

#### 4.1.1 StudyReportValidator (core/ectd_study_report_validator.py)
✅ **已实现:**
- 识别4.2.X和5.3.1.X-5.3.5.X模块中的研究报告章节
- 验证STF标签有效性（pre-clinical-study-report, legacy-clinical-study-report）
- 检查研究报告是否有关联的数据集
- 排除dataset章节避免误判

❌ **缺失:**
- STF文件命名验证（stf-{study-id}.xml格式）
- STF版本属性验证
- STF操作类型验证（首次new，后续append）
- Category元素验证（species, route-of-admin, duration, type-of-control）
- File-tag完整性验证（对照ICH标准tag列表）
- Property元素验证（site-identifier for CRF/subject-profiles）
- ICH E3结构合规性验证
- 中国数据递交规范验证（STUDYID, USUBJID, SUBJID等必需标识符）

#### 4.1.2 STFOperationValidator (core/ectd_stf_operation_validator.py)
✅ **已实现:**
- 识别STF文件（.xml后缀且包含study-tagging-file相关标签）
- 提供操作建议（NEW=preferred, REPLACE/DELETE=discouraged, APPEND=acceptable）

❌ **缺失:**
- STF文件路径验证（应与研究文件在同一目录）
- Modified-file属性验证（append操作时必须引用最近STF）
- STF生命周期管理验证（累积方式）
- Study-identifier修改验证

### 4.2 待实现的增强功能

#### 4.2.1 高优先级（立即实施）

**A. STF文件格式验证器 (新建)**
```python
# core/ectd_stf_format_validator.py
class STFFormatValidator:
    def validate_stf_naming(self, file_path: str) -> List[ViolationDetail]:
        """验证STF文件命名: stf-{study-id}.xml"""
        
    def validate_stf_structure(self, stf_content: str) -> List[ViolationDetail]:
        """验证STF XML结构（study-identifier, study-document元素）"""
        
    def validate_category_elements(self, categories: List[dict], module: str) -> List[ViolationDetail]:
        """验证category元素及其值的有效性"""
        
    def validate_file_tags(self, file_tags: List[str]) -> List[ViolationDetail]:
        """验证file-tag name属性值是否在ICH标准列表中"""
        
    def validate_property_elements(self, file_tag: str, properties: List[dict]) -> List[ViolationDetail]:
        """验证property元素（site-identifier for CRF/subject-profiles）"""
```

**B. 模块5.2/5.3.6/5.4豁免规则 (增强现有)**
```python
# 在StudyReportValidator中添加
EXEMPTED_MODULES = [
    r"m5[/\\]m5-2[/\\].*",      # 5.2 所有临床研究列表
    r"m5[/\\]m5-3[/\\]m5-3-6[/\\].*",  # 5.3.6 上市后报告
    r"m5[/\\]m5-4[/\\].*",      # 5.4 参考文献
]

def validate_stf_exemptions(self, section_id: SectionIdentifier) -> Optional[ViolationDetail]:
    """验证5.2/5.3.6/5.4模块可以不使用STF"""
```

**C. 数据集位置验证 (增强现有)**
```python
def validate_dataset_position(
    self, 
    study_report_sections: List[SectionIdentifier],
    dataset_sections: List[SectionIdentifier]
) -> List[ViolationDetail]:
    """验证数据集应在对应研究报告之后"""
```

#### 4.2.2 中优先级（本周内）

**D. ICH E3结构验证器 (新建)**
```python
# core/ectd_e3_structure_validator.py
class E3StructureValidator:
    # E3必需章节列表
    REQUIRED_SECTIONS = {
        "1": "Title Page",
        "2": "Synopsis",
        "3": "Table of Contents",
        # ... 完整的1-16章节
        "16.1": "Study Information",
        "16.2": "Patient Data Listings",
    }
    
    def validate_e3_compliance(
        self, 
        report_file_path: str,
        stf_file_tags: List[str]
    ) -> List[ViolationDetail]:
        """验证临床研究报告是否符合ICH E3结构要求"""
        
    def check_required_sections(self, toc: dict) -> List[ViolationDetail]:
        """检查必需章节是否存在"""
        
    def validate_appendix_161_content(self, file_tags: List[str]) -> List[ViolationDetail]:
        """验证16.1附录内容与STF file-tags的一致性"""
        
    def validate_synopsis_length(self, synopsis_content: str) -> Optional[ViolationDetail]:
        """验证摘要长度（应≤3页）"""
```

**E. 中国数据递交规范验证器 (新建)**
```python
# core/ectd_china_data_validator.py
class ChinaDataSubmissionValidator:
    REQUIRED_IDENTIFIERS = ["STUDYID", "USUBJID", "SUBJID"]
    REQUIRED_TIME_VARS = ["VISIT", "VISITNUM"]
    
    def validate_dataset_naming(self, dataset_name: str) -> List[ViolationDetail]:
        """验证数据集命名规范（小写字母开头，≤8字节）"""
        
    def validate_variable_naming(self, var_name: str) -> List[ViolationDetail]:
        """验证变量命名规范（大写字母开头，≤8字节）"""
        
    def validate_required_identifiers(self, dataset: dict) -> List[ViolationDetail]:
        """验证必需标识符（STUDYID, USUBJID, SUBJID）"""
        
    def validate_labels_in_chinese(self, dataset: dict) -> List[ViolationDetail]:
        """验证数据集标签和变量标签是否为中文"""
        
    def validate_stf_tags_china(self, stf_tags: List[str]) -> List[ViolationDetail]:
        """验证中国特定的STF标签（附录2）"""
```

#### 4.2.3 低优先级（下周）

**F. 数据可追溯性验证器 (新建)**
```python
# core/ectd_data_traceability_validator.py
class DataTraceabilityValidator:
    def validate_aCRF_mapping(
        self, 
        acrf_annotations: dict,
        raw_datasets: List[str]
    ) -> List[ViolationDetail]:
        """验证aCRF与原始数据集的映射完整性"""
        
    def validate_derivation_traceability(
        self, 
        analysis_dataset: dict,
        raw_datasets: List[dict],
        program_code: str
    ) -> List[ViolationDetail]:
        """验证分析数据集衍生变量的可追溯性"""
```

### 4.3 测试用例增强

#### 4.3.1 新增测试文件
- `test_ectd_stf_format_validator.py` - STF格式验证测试
- `test_ectd_e3_structure_validator.py` - E3结构合规性测试
- `test_ectd_china_data_validator.py` - 中国数据递交规范测试
- `test_ectd_data_traceability_validator.py` - 数据可追溯性测试

#### 4.3.2 现有测试增强
- `test_ectd_study_report_validator.py` - 添加5.2/5.3.6/5.4豁免测试
- `test_ectd_stf_operation_validator.py` - 添加STF生命周期管理测试

---

## 五、实施路线图

### Phase 2.7: STF格式验证（2-3天）
1. 创建 `STFFormatValidator` 类
2. 实现STF命名、结构、category、file-tag、property验证
3. 编写20个单元测试
4. 集成到 `ECTDLifecycleValidator`

### Phase 2.8: 模块豁免和数据集位置（1天）
1. 增强 `StudyReportValidator` 添加豁免规则
2. 实现数据集位置验证
3. 添加8个测试用例

### Phase 2.9: ICH E3结构验证（3-4天）
1. 创建 `E3StructureValidator` 类
2. 实现E3章节完整性检查
3. 实现附录16.1/16.2验证
4. 编写15个单元测试

### Phase 2.10: 中国数据递交规范（3-4天）
1. 创建 `ChinaDataSubmissionValidator` 类
2. 实现命名规范、标识符、中文标签验证
3. 实现中国特定STF标签验证
4. 编写18个单元测试

### Phase 2.11: 数据可追溯性（2-3天）
1. 创建 `DataTraceabilityValidator` 类
2. 实现aCRF映射和衍生可追溯性验证
3. 编写12个单元测试

### Phase 2.12: 集成测试和文档（1-2天）
1. 端到端集成测试
2. 更新 `ECTD_3.8_RULE_COVERAGE_ANALYSIS.md`
3. 生成最终验证报告

---

## 六、预期成果

### 6.1 规则覆盖率提升
- **当前**: 65% (5/12 fully + 4/12 partially)
- **Phase 2.7完成后**: 75%
- **Phase 2.10完成后**: 85%
- **Phase 2.12完成后**: 95%

### 6.2 新增违规类型
- STF_INVALID_NAMING
- STF_MISSING_VERSION
- STF_INVALID_OPERATION
- STF_CATEGORY_INVALID
- STF_FILE_TAG_UNKNOWN
- STF_MISSING_PROPERTY
- E3_MISSING_SECTION
- E3_SYNOPSIS_TOO_LONG
- CHINA_INVALID_DATASET_NAME
- CHINA_INVALID_VARIABLE_NAME
- CHINA_MISSING_IDENTIFIER
- CHINA_LABEL_NOT_CHINESE
- TRACEABILITY_ACRF_UNMAPPED
- TRACEABILITY_DERIVATION_UNDOCUMENTED

### 6.3 文档交付物
1. ✅ 本文档 (ECTD_3.8_PDF_PARSING_SUMMARY.md)
2. 更新的规则覆盖分析
3. 新增验证器代码（5个新类）
4. 新增测试用例（73个新测试）
5. 集成报告示例

---

## 附录A: STF File-tag完整映射表

### ICH标准File-tags (info-type="ich")

| File-tag Name | 中文名称 | E3章节 | 适用模块 |
|--------------|---------|--------|---------|
| pre-clinical-study-report | 非临床研究报告 | - | 4.2.X |
| legacy-clinical-study-report | 遗留临床研究报告 | - | 5.3.X |
| synopsis | 摘要 | 2 | 5.3.X |
| study-report-body | 研究报告正文 | 1,3-15 | 5.3.X |
| protocol-or-amendment | 方案或修订 | 16.1.1 | 5.3.X |
| sample-case-report-form | 样本CRF | 16.1.2 | 5.3.X |
| iec-irb-consent-form-list | 伦理委员会和知情同意书清单 | 16.1.3 | 5.3.X |
| list-description-investigator-site | 研究者和研究中心描述 | 16.1.4 | 5.3.X |
| signatures-investigators | 研究者签名 | 16.1.5 | 5.3.X |
| list-patients-with-batches | 批次受试者清单 | 16.1.6 | 5.3.X |
| randomisation-scheme | 随机化方案 | 16.1.7 | 5.3.X |
| audit-certificates-report | 审计证书 | 16.1.8 | 5.3.X |
| statistical-methods-interim-analysis-plan | 统计方法和中期分析计划 | 16.1.9 | 5.3.X |
| inter-laboratory-standardisation-methods-quality-assurance | 实验室间标准化和质量保证 | 16.1.10 | 5.3.X |
| publications-based-on-study | 基于研究的发表 | 16.1.11 | 5.3.X |
| publications-referenced-in-report | 报告中引用的发表 | 16.1.12 | 5.3.X |
| discontinued-patients | 中止受试者 | 16.2.1 | 5.3.X |
| protocol-deviations | 方案偏离 | 16.2.2 | 5.3.X |
| patients-excluded-from-efficacy-analysis | 排除在疗效分析外的受试者 | 16.2.3 | 5.3.X |
| demographic-data | 人口学数据 | 16.2.4 | 5.3.X |
| compliance-and-drug-concentration-data | 依从性和药物浓度数据 | 16.2.5 | 5.3.X |
| individual-efficacy-response-data | 个体疗效反应数据 | 16.2.6 | 5.3.X |
| adverse-event-listings | 不良事件清单 | 16.2.7 | 5.3.X |
| listing-individual-laboratory-measurements-by-patient | 按受试者的实验室测量值 | 16.2.8 | 5.3.X |
| case-report-forms | 病例报告表 | 16.3 | 5.3.X |
| available-on-request | 可应要求提供的文档 | - | Regional |

### 美国特定File-tags (info-type="us")

| File-tag Name | 中文名称 | 用途 |
|--------------|---------|------|
| data-tabulation-dataset | 数据表格数据集 | 原始数据集 |
| data-tabulation-data-definition | 数据表格数据定义 | 数据说明文件 |
| data-listing-dataset | 数据清单数据集 | 清单数据集 |
| data-listing-data-definition | 数据清单数据定义 | 清单说明文件 |
| analysis-dataset | 分析数据集 | 分析用数据 |
| analysis-program | 分析程序 | SAS/R代码 |
| analysis-data-definition | 分析数据定义 | 分析数据说明 |
| annotated-crf | 注释CRF | aCRF |
| ecg | ECG波形数据集 | 心电图 |
| image | 图像文件 | 影像学 |
| subject-profiles | 受试者概况 | 个体资料 |
| safety-report | 安全性报告 | IND安全报告 |
| antibacterial | 抗菌微生物学报告 | 微生物 |
| special-pathogen | 特殊病原体报告 | 微生物 |
| antiviral | 抗病毒微生物学报告 | 微生物 |
| iss | 综合安全性总结 | ISS |
| ise | 综合疗效总结 | ISE |
| pm-description | 上市后定期报告描述 | 药物警戒 |

### 日本特定File-tags (info-type="jp")

| File-tag Name | 中文名称 |
|--------------|---------|
| complete-patient-list | 完整受试者清单 |
| serious-adverse-event-patient-list | 严重不良事件受试者清单 |
| adverse-event-patient-list | 不良事件受试者清单 |
| abnormal-lab-values-patient-list | 异常实验室值受试者清单 |

---

## 附录B: Category元素完整编码表

### Species (物种) - info-type="ich"
| 编码 | 中文名称 |
|------|---------|
| mouse | 小鼠 |
| rat | 大鼠 |
| hamster | 仓鼠 |
| other-rodent | 其他啮齿类 |
| rabbit | 兔 |
| dog | 犬 |
| non-human-primate | 非人灵长类 |
| other-non-rodent-mammal | 其他非啮齿类哺乳动物 |
| non-mammals | 非哺乳动物 |

### Route of Administration (给药途径) - info-type="ich"
| 编码 | 中文名称 |
|------|---------|
| oral | 口服 |
| intravenous | 静脉注射 |
| intramuscular | 肌肉注射 |
| intraperitoneal | 腹腔注射 |
| subcutaneous | 皮下注射 |
| inhalation | 吸入 |
| topical | 外用 |
| other | 其他（咨询监管机构后使用）|

### Duration (持续时间) - info-type="us"
| 编码 | 中文名称 |
|------|---------|
| short | 短期 |
| medium | 中期 |
| long | 长期 |

### Type of Control (对照类型) - info-type="ich"
| 编码 | 中文名称 |
|------|---------|
| placebo | 安慰剂对照 |
| no-treatment | 无治疗对照 |
| dose-response-without-placebo | 剂量反应（无安慰剂）|
| active-control-without-placebo | 阳性药对照（无安慰剂）|
| external | 外部对照 |

---

## 总结

通过解析3个关键PDF文件，我们提取了约**150+条具体验证规则**，覆盖：
- STF文件格式和生命周期管理（STFV2-6-1）
- ICH E3临床研究报告结构要求（E3_Guideline）
- 中国药监局数据递交规范（药物临床试验数据递交指导原则）

当前实现覆盖了基础的STF标签验证和操作建议，但缺失大量细节验证。通过Phase 2.7-2.12的实施，预计可将3.8章节规则覆盖率从65%提升至95%，实现**真正的企业级eCTD合规性验证引擎**。
