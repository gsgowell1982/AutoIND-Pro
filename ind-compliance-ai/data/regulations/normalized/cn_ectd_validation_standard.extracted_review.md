# eCTD验证标准 提取校对稿

- 来源文件: `eCTD验证标准.pdf`
- regulation_id: `cn_ectd_validation_standard`
- 提取章节数: `6`
- 提取验证项数: `149`

## 章节目录

- 1. 基础识别
- 2. 文件/文件夹
- 3. ICH 骨架文件
- 4. 区域性管理信息
- 5. 研究标签文件（STF）
- 6. PDF分析

## 1. 基础识别

### 1.1 文件数量统计

- `clause_id`: `cn_ectd_validation_standard:sec_1_1`
- `chapter_no`: `1`
- `source_heading`: `1. 基础识别`
- `严重程度`: `提示信息`
- `原始说明行`:
```text
显示当前序列中包含的所有文件的数量。
```
- `规范化全文`: 1.1 文件数量统计 显示当前序列中包含的所有文件的数量。 提示信息

### 1.2 文件大小统计

- `clause_id`: `cn_ectd_validation_standard:sec_1_2`
- `chapter_no`: `1`
- `source_heading`: `1. 基础识别`
- `严重程度`: `提示信息`
- `原始说明行`:
```text
显示当前序列所有文件的总容量大小。
```
- `规范化全文`: 1.2 文件大小统计 显示当前序列所有文件的总容量大小。 提示信息

### 1.3 空缺的章节统计

- `clause_id`: `cn_ectd_validation_standard:sec_1_3`
- `chapter_no`: `1`
- `source_heading`: `1. 基础识别`
- `严重程度`: `提示信息`
- `原始说明行`:
```text
显示当前序列中空缺的目录结构清单。
```
- `规范化全文`: 1.3 空缺的章节统计 显示当前序列中空缺的目录结构清单。 提示信息

## 2. 文件/文件夹

### 2.1 文件夹不能为空

- `clause_id`: `cn_ectd_validation_standard:sec_2_1`
- `chapter_no`: `2`
- `source_heading`: `2. 文件/文件夹`
- `严重程度`: `错误`
- `原始说明行`:
```text
序列文件夹目录结构中不允许存在空文件夹（文件夹没有文件或子文件夹）。
```
- `规范化全文`: 2.1 文件夹不能为空 序列文件夹目录结构中不允许存在空文件夹（文件夹没有文件或子文件夹）。 错误

### 2.2 不能超出文件大小限制

- `clause_id`: `cn_ectd_validation_standard:sec_2_2`
- `chapter_no`: `2`
- `source_heading`: `2. 文件/文件夹`
- `严重程度`: `警告`
- `原始说明行`:
```text
超出允许大小的文件会提示警告信息。
普通单个文件最大允许500MB，单个SAS XPT文件可支持4GB。
```
- `规范化全文`: 2.2 不能超出文件大小限制 超出允许大小的文件会提示警告信息。 普通单个文件最大允许500MB，单个SAS XPT文件可支持4GB。 警告

### 2.3 不允许未被引用的文件

- `clause_id`: `cn_ectd_validation_standard:sec_2_3`
- `chapter_no`: `2`
- `source_heading`: `2. 文件/文件夹`
- `严重程度`: `错误`
- `原始说明行`:
```text
m1至m5文件夹目录下的所有文件必须被骨架文件（index.xml或cn-regional.xml）引用，否则提示
错误信息。
```
- `规范化全文`: 2.3 不允许未被引用的文件 m1至m5文件夹目录下的所有文件必须被骨架文件（index.xml或cn-regional.xml）引用，否则提示 错误信息。 错误

### 2.4 文件类型（文件扩展名检查）

- `clause_id`: `cn_ectd_validation_standard:sec_2_4`
- `chapter_no`: `2`
- `source_heading`: `2. 文件/文件夹`
- `严重程度`: `错误`
- `原始说明行`:
```text
所有被引用文件必须有且仅有一个文件扩展名，文件扩展名必须在《eCTD技术规范V1.0》规定的可接
受的文件类型列表内。
```
- `规范化全文`: 2.4 文件类型（文件扩展名检查） 所有被引用文件必须有且仅有一个文件扩展名，文件扩展名必须在《eCTD技术规范V1.0》规定的可接 受的文件类型列表内。 错误

### 2.5 文件和文件夹命名规范必须正确

- `clause_id`: `cn_ectd_validation_standard:sec_2_5`
- `chapter_no`: `2`
- `source_heading`: `2. 文件/文件夹`
- `严重程度`: `错误`
- `原始说明行`:
```text
文件和文件夹的命名规则必须符合《eCTD技术规范V1.0》第3.3.2章节的规定。
```
- `规范化全文`: 2.5 文件和文件夹命名规范必须正确 文件和文件夹的命名规则必须符合《eCTD技术规范V1.0》第3.3.2章节的规定。 错误

### 2.6 m1文件夹存在但是不存放单个文件

- `clause_id`: `cn_ectd_validation_standard:sec_2_6`
- `chapter_no`: `2`
- `source_heading`: `2. 文件/文件夹`
- `严重程度`: `错误`
- `原始说明行`:
```text
m1文件夹必须存在并且必须仅包含子文件夹而不是单个文件。
```
- `规范化全文`: 2.6 m1文件夹存在但是不存放单个文件 m1文件夹必须存在并且必须仅包含子文件夹而不是单个文件。 错误

### 2.7 util文件夹必须存在且包含的文件必须正确

- `clause_id`: `cn_ectd_validation_standard:sec_2_7`
- `chapter_no`: `2`
- `source_heading`: `2. 文件/文件夹`
- `严重程度`: `错误`
- `原始说明行`:
```text
util文件夹必须存在，并检查该文件夹中的下列文件：
- ich-ectd-3-2.dtd (checksum必须符合ICH发布的值)
- cn-regional-1-0.xsd (checksum必须符合NMPA发布的值)
- xml.xsd (checksum必须符合NMPA发布的值)
- xlink.xsd (checksum必须符合NMPA发布的值)
- ich-stf-v2-2.dtd (checksum必须符合ICH发布的值)
- ectd-2-0.xsl (checksum必须符合ICH发布的值)
- cn-regional-1-0.xsl (checksum必须符合NMPA发布的值)
- ich-stf-stylesheet-2-3.xsl (checksum必须符合ICH发布的值)
- ich-stf-stylesheet-2-2a.xsl (checksum必须符合ICH发布的值)
- valid-values.xml (checksum必须符合ICH发布的值)
请注意，只有在当前序列使用到STF时才需要检查ich-stf-v2-2.dtd、stf样式表和valid-values.xml。
另外，在STF中，ich-stf-stylesheet-2-3.xsl和ich-stf-stylesheet-2-2a.xsl样式表均可以被引用，此项
验证标准仅检查在STF中被引用到的样式表是否在util\style文件夹中存在以及其checksum是否符合规
范。
```
- `规范化全文`: 2.7 util文件夹必须存在且包含的文件必须正确 util文件夹必须存在，并检查该文件夹中的下列文件： - ich-ectd-3-2.dtd (checksum必须符合ICH发布的值) - cn-regional-1-0.xsd (checksum必须符合NMPA发布的值) - xml.xsd (checksum必须符合NMPA发布的值) - xlink.xsd (checksum必须符合NMPA发布的值) - ich-stf-v2-2.dtd (checksum必须符合ICH发布的值) - ectd-2-0.xsl (checksum必须符合ICH发布的值) - cn-regional-1-0.xsl (checksum必须符合NMPA发布的值) - ich-stf-stylesheet-2-3.xsl (checksum必须符合ICH发布的值) - ich-stf-stylesheet-2-2a.xsl (checksum必须符合ICH发布的值) - valid-values.xml (checksum必须符合ICH发布的值) 请注意，只有在当前序列使用到STF时才需要检查ich-stf-v2-2.dtd、stf样式表和valid-values.xml。 另外，在STF中，ich-stf-stylesheet-2-3.xsl和ich-stf-stylesheet-2-2a.xsl样式表均可以被引用，此项 验证标准仅检查在STF中被引用到的样式表是否在util\style文件夹中存在以及其checksum是否符合规 范。 错误

### 2.8 序列号根文件夹不允许其他文件

- `clause_id`: `cn_ectd_validation_standard:sec_2_8`
- `chapter_no`: `2`
- `source_heading`: `2. 文件/文件夹`
- `严重程度`: `错误`
- `原始说明行`:
```text
根文件夹（申请序列文件夹）除《eCTD技术规范V1.0》明确允许的文件外，不能有其他文件。
```
- `规范化全文`: 2.8 序列号根文件夹不允许其他文件 根文件夹（申请序列文件夹）除《eCTD技术规范V1.0》明确允许的文件外，不能有其他文件。 错误

### 2.9 序列文件夹要求

- `clause_id`: `cn_ectd_validation_standard:sec_2_9`
- `chapter_no`: `2`
- `source_heading`: `2. 文件/文件夹`
- `严重程度`: `错误`
- `原始说明行`:
```text
序列文件夹名称必须仅包含4个数字。
```
- `规范化全文`: 2.9 序列文件夹要求 序列文件夹名称必须仅包含4个数字。 错误

### 2.10 序列编号

- `clause_id`: `cn_ectd_validation_standard:sec_2_10`
- `chapter_no`: `2`
- `source_heading`: `2. 文件/文件夹`
- `严重程度`: `错误`
- `原始说明行`:
```text
在注册文件生命周期内的每次提交都必须有一个序列号，初始序列号必须从0000开始，依次增加1
（例如，0000，0001，0002，0003，依次类推）。序列号不允许跳号。如果序列号0003不存在，即
便0004中不引用0003的内容，验证序列号为0004的提交时依然会提示错误信息。
```
- `规范化全文`: 2.10 序列编号 在注册文件生命周期内的每次提交都必须有一个序列号，初始序列号必须从0000开始，依次增加1 （例如，0000，0001，0002，0003，依次类推）。序列号不允许跳号。如果序列号0003不存在，即 便0004中不引用0003的内容，验证序列号为0004的提交时依然会提示错误信息。 错误

## 3. ICH 骨架文件

### 3.1 index.xml文件必须存在

- `clause_id`: `cn_ectd_validation_standard:sec_3_1`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
序列的根文件夹必须包含index.xml文件。
```
- `规范化全文`: 3.1 index.xml文件必须存在 序列的根文件夹必须包含index.xml文件。 错误

### 3.2 index.xml中DTD的引用必须指向util文件夹中提供的DTD

- `clause_id`: `cn_ectd_validation_standard:sec_3_2`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
如何创建有效的引用请参考http://www.w3.org/TR/xml/ 和 http://www.ietf.org/rfc/rfc3986.txt
（2005版本 第22页，第3.3章节）。
```
- `规范化全文`: 3.2 index.xml中DTD的引用必须指向util文件夹中提供的DTD 如何创建有效的引用请参考http://www.w3.org/TR/xml/ 和 http://www.ietf.org/rfc/rfc3986.txt （2005版本 第22页，第3.3章节）。 错误

### 3.3 index.xml文件必须有效。

- `clause_id`: `cn_ectd_validation_standard:sec_3_3`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
index.xml文件必须根据DTD文件ich-ectd-3-2.dtd给定的规则创建并且是有效的。
```
- `规范化全文`: 3.3 index.xml文件必须有效。 index.xml文件必须根据DTD文件ich-ectd-3-2.dtd给定的规则创建并且是有效的。 错误

### 3.4 文件引用（xlink:href）属性指向的文件必须存在

- `clause_id`: `cn_ectd_validation_standard:sec_3_4`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
XML叶元素中的文件引用链接必须有效，即引用的目标文件必须存在。
```
- `规范化全文`: 3.4 文件引用（xlink:href）属性指向的文件必须存在 XML叶元素中的文件引用链接必须有效，即引用的目标文件必须存在。 错误

### 3.5 文件在一个序列中不允许对应多个操作

- `clause_id`: `cn_ectd_validation_standard:sec_3_5`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
ICH骨架文件中的叶元素文件在同一个序列中不能作为“被修改文件对象（modified-file）”被多次
引用。
```
- `规范化全文`: 3.5 文件在一个序列中不允许对应多个操作 ICH骨架文件中的叶元素文件在同一个序列中不能作为“被修改文件对象（modified-file）”被多次 引用。 错误

### 3.6 替换或增补的内容不能与之前文件相同

- `clause_id`: `cn_ectd_validation_standard:sec_3_6`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
当执行“替换（replace）”或“增补（append）”操作时，新内容必须与之前的内容不同（即
checksum不能相同）。
```
- `规范化全文`: 3.6 替换或增补的内容不能与之前文件相同 当执行“替换（replace）”或“增补（append）”操作时，新内容必须与之前的内容不同（即 checksum不能相同）。 错误

### 3.7 叶元素：新建、替换或增补的叶元素，必须有“文件引用（xlink:href）”值

- `clause_id`: `cn_ectd_validation_standard:sec_3_7`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
（xlink:href）”值
“操作（operation）”属性值为“新建（new）”、“替换（replace）”或“增补（append）”的
所有叶元素，必须有“文件引用（xlink:href）”值。
```
- `规范化全文`: 3.7 叶元素：新建、替换或增补的叶元素，必须有“文件引用（xlink:href）”值 错误

### 3.8 叶元素：删除的叶元素不能包含“文件引用（xlink:href）”值

- `clause_id`: `cn_ectd_validation_standard:sec_3_8`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
”值
“操作（operation）”属性值为“删除（delete）”的所有叶元素，不能有“文件引用
（xlink:href）”值。
```
- `规范化全文`: 3.8 叶元素：删除的叶元素不能包含“文件引用（xlink:href）”值 错误

### 3.9 叶元素：对替换、删除和增补的叶元素，必须有对应的文件“操作（operation）”属性值为“替换（replace）”、“删除（delete）”或“增补（append）”的所有叶元素，对应的“被修改文件对象（modified-file）”必须有值。

- `clause_id`: `cn_ectd_validation_standard:sec_3_9`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
”的所有叶元素，对应的“被修改文件对象（modified-file）”必须有值。
```
- `规范化全文`: 3.9 叶元素：对替换、删除和增补的叶元素，必须有对应的文件“操作（operation）”属性值为“替换（replace）”、“删除（delete）”或“增补（append）”的所有叶元素，对应的“被修改文件对象（modified-file）”必须有值。 错误

### 3.10 叶元素：初始序列中所有文件必须为新建

- `clause_id`: `cn_ectd_validation_standard:sec_3_10`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
初始序列（即序列号为0000的序列）中所有内容文件的“操作（operation）”属性必须为“新建
（new）”。
```
- `规范化全文`: 3.10 叶元素：初始序列中所有文件必须为新建 初始序列（即序列号为0000的序列）中所有内容文件的“操作（operation）”属性必须为“新建 （new）”。 错误

### 3.11 被修改文件对象必须存在

- `clause_id`: `cn_ectd_validation_standard:sec_3_11`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
XML叶元素中的被修改文件的引用链接必须是有效的，即“被修改文件对象（modified-file）”对应
的文件确实存在。
```
- `规范化全文`: 3.11 被修改文件对象必须存在 XML叶元素中的被修改文件的引用链接必须是有效的，即“被修改文件对象（modified-file）”对应 的文件确实存在。 错误

### 3.12 只允许使用相对路径引用

- `clause_id`: `cn_ectd_validation_standard:sec_3_12`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
ICH骨架文件中引用文件使用的是相对路径，不允许使用绝对路径。
在“文件引用（ xlink:href ）”和“被修改文件对象（modified-file）”属性中只允许使用相对路径
引用，并且路径中只允许使用正斜杠"/"，不允许使用反斜杠"\"。
```
- `规范化全文`: 3.12 只允许使用相对路径引用 ICH骨架文件中引用文件使用的是相对路径，不允许使用绝对路径。 在“文件引用（ xlink:href ）”和“被修改文件对象（modified-file）”属性中只允许使用相对路径 引用，并且路径中只允许使用正斜杠"/"，不允许使用反斜杠"\"。 错误

### 3.13 “checksum-type”属性

- `clause_id`: `cn_ectd_validation_standard:sec_3_13`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
“checksum-type”属性必须设置为“md5”或“MD5”。
```
- `规范化全文`: 3.13 “checksum-type”属性 “checksum-type”属性必须设置为“md5”或“MD5”。 错误

### 3.14 MD5值校验

- `clause_id`: `cn_ectd_validation_standard:sec_3_14`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
所有文件的MD5 checksum必须和骨架文件中提供的checksum值保持一致。
```
- `规范化全文`: 3.14 MD5值校验 所有文件的MD5 checksum必须和骨架文件中提供的checksum值保持一致。 错误

### 3.15 骨架文件的MD5值校验

- `clause_id`: `cn_ectd_validation_standard:sec_3_15`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
骨架文件的MD5 checksum必须和MD5文本文件（index-md5.txt）中提供的值保持一致。
```
- `规范化全文`: 3.15 骨架文件的MD5值校验 骨架文件的MD5 checksum必须和MD5文本文件（index-md5.txt）中提供的值保持一致。 错误

### 3.16 扩展节点的使用要求

- `clause_id`: `cn_ectd_validation_standard:sec_3_16`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
扩展节点仅允许在产品类型为生物制品的序列的3.2.R章节使用。
如产品类型为生物制品的序列存在3.2.R章节，则必须使用扩展节点。
```
- `规范化全文`: 3.16 扩展节点的使用要求 扩展节点仅允许在产品类型为生物制品的序列的3.2.R章节使用。 如产品类型为生物制品的序列存在3.2.R章节，则必须使用扩展节点。 错误

### 3.17 扩展节点标题的命名规范

- `clause_id`: `cn_ectd_validation_standard:sec_3_17`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `警告`
- `原始说明行`:
```text
对于生物制品，3.2.R章节扩展节点标题的构建和名称定义必须遵循《eCTD技术规范V1.0》第3.2章节
中的规定。
```
- `规范化全文`: 3.17 扩展节点标题的命名规范 对于生物制品，3.2.R章节扩展节点标题的构建和名称定义必须遵循《eCTD技术规范V1.0》第3.2章节 中的规定。 警告

### 3.18 叶标题不能为空

- `clause_id`: `cn_ectd_validation_standard:sec_3_18`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
所有叶元素必须有子元素<title>，叶标题不能为空。
```
- `规范化全文`: 3.18 叶标题不能为空 所有叶元素必须有子元素<title>，叶标题不能为空。 错误

### 3.19 删除的叶标题必须和被删除的叶标题保持一致

- `clause_id`: `cn_ectd_validation_standard:sec_3_19`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
“操作（operation）”属性为“删除（delete）”的叶元素，其叶标题必须和被删除的叶标题保持一
致。
```
- `规范化全文`: 3.19 删除的叶标题必须和被删除的叶标题保持一致 “操作（operation）”属性为“删除（delete）”的叶元素，其叶标题必须和被删除的叶标题保持一 致。 错误

### 3.20 叶标题开头和结尾不能为空格

- `clause_id`: `cn_ectd_validation_standard:sec_3_20`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `警告`
- `原始说明行`:
```text
叶标题开头和结尾不能为空格。
```
- `规范化全文`: 3.20 叶标题开头和结尾不能为空格 叶标题开头和结尾不能为空格。 警告

### 3.21 元素下必须有叶元素

- `clause_id`: `cn_ectd_validation_standard:sec_3_21`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
名称以“m”开始的元素必须有叶元素。
```
- `规范化全文`: 3.21 元素下必须有叶元素 名称以“m”开始的元素必须有叶元素。 错误

### 3.22 m1行政文件和药品信息元素必须存在。

- `clause_id`: `cn_ectd_validation_standard:sec_3_22`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
元素“m1-administrative-information-and-prescribing-information”必须存在。
```
- `规范化全文`: 3.22 m1行政文件和药品信息元素必须存在。 元素“m1-administrative-information-and-prescribing-information”必须存在。 错误

### 3.23 增补（append）的使用

- `clause_id`: `cn_ectd_validation_standard:sec_3_23`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `警告`
- `原始说明行`:
```text
不建议在STF定义范围外使用“增补（append）”。在STF定义范围外使用“增补（append）”操
作，需要在说明函中进行说明。
```
- `规范化全文`: 3.23 增补（append）的使用 不建议在STF定义范围外使用“增补（append）”。在STF定义范围外使用“增补（append）”操 作，需要在说明函中进行说明。 警告

### 3.24 不能重新定位文件位置

- `clause_id`: `cn_ectd_validation_standard:sec_3_24`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
该规则适用于ICH模块二至模块五的叶元素。
当对现有文件进行修订时，新的叶元素应在骨架文件中的相同位置进行提交，并以增补、替换或删除
的方式体现。
文件位置指的是文件在目录结构中的位置。该位置由CTD目录结构以及eCTD中的部分属性来定义。例
如，申请人不能使用说明函章节修改的内容替换申请表章节中的内容。另外，申请人可使用eCTD属性
创建自定义章节。 例如，m3-2-s-drug-substance中的每个“活性成分（substance）”或“生产商
（manufacturer）”属性，或m3-2-p-drug-product中的“产品名称（product-name）”属性均
可创建一个新的CTD章节，申请人同样不能使用其中一个章节的内容替换其他章节的内容。
```
- `规范化全文`: 3.24 不能重新定位文件位置 该规则适用于ICH模块二至模块五的叶元素。 当对现有文件进行修订时，新的叶元素应在骨架文件中的相同位置进行提交，并以增补、替换或删除 的方式体现。 文件位置指的是文件在目录结构中的位置。该位置由CTD目录结构以及eCTD中的部分属性来定义。例 如，申请人不能使用说明函章节修改的内容替换申请表章节中的内容。另外，申请人可使用eCTD属性 创建自定义章节。 例如，m3-2-s-drug-substance中的每个“活性成分（substance）”或“生产商 （manufacturer）”属性，或m3-2-p-drug-product中的“产品名称（product-name）”属性均 可创建一个新的CTD章节，申请人同样不能使用其中一个章节的内容替换其他章节的内容。 错误

### 3.25 属性-适应症（indication）

- `clause_id`: `cn_ectd_validation_standard:sec_3_25`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
适应症属性在2.7.3和5.3.5章节中使用时为必填项，值不能为空。
```
- `规范化全文`: 3.25 属性-适应症（indication） 适应症属性在2.7.3和5.3.5章节中使用时为必填项，值不能为空。 错误

### 3.26 属性-生产商（manufacturer）

- `clause_id`: `cn_ectd_validation_standard:sec_3_26`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
生产商属性在2.3.S和3.2.S章节中使用时为必填项，值不能为空。
```
- `规范化全文`: 3.26 属性-生产商（manufacturer） 生产商属性在2.3.S和3.2.S章节中使用时为必填项，值不能为空。 错误

### 3.27 属性-活性成分（substance）

- `clause_id`: `cn_ectd_validation_standard:sec_3_27`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
活性成分属性在2.3.S和3.2.S章节中使用时为必填项，值不能为空。
```
- `规范化全文`: 3.27 属性-活性成分（substance） 活性成分属性在2.3.S和3.2.S章节中使用时为必填项，值不能为空。 错误

### 3.28 属性值开头和结尾不能为空格

- `clause_id`: `cn_ectd_validation_standard:sec_3_28`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `警告`
- `原始说明行`:
```text
属性值开头和结尾不能为空格。
```
- `规范化全文`: 3.28 属性值开头和结尾不能为空格 属性值开头和结尾不能为空格。 警告

### 3.29 内容的引用

- `clause_id`: `cn_ectd_validation_standard:sec_3_29`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
骨架文件（index.xml）中不允许包含跨申请引用。当在骨架文件中引用另一个序列的内容时，只能引
用同一申请中早先已提交序列的内容。
```
- `规范化全文`: 3.29 内容的引用 骨架文件（index.xml）中不允许包含跨申请引用。当在骨架文件中引用另一个序列的内容时，只能引 用同一申请中早先已提交序列的内容。 错误

### 3.30 检测无效的生命周期模式：增补操作造成分支

- `clause_id`: `cn_ectd_validation_standard:sec_3_30`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
已经被一个叶元素替换的叶元素不能再进行增补操作。
```
- `规范化全文`: 3.30 检测无效的生命周期模式：增补操作造成分支 已经被一个叶元素替换的叶元素不能再进行增补操作。 错误

### 3.31 检测无效的生命周期模式：删除操作造成分支

- `clause_id`: `cn_ectd_validation_standard:sec_3_31`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
已经被一个叶元素替换的叶元素不能再进行删除操作。
```
- `规范化全文`: 3.31 检测无效的生命周期模式：删除操作造成分支 已经被一个叶元素替换的叶元素不能再进行删除操作。 错误

### 3.32 检测无效的生命周期模式：替换操作造成分支

- `clause_id`: `cn_ectd_validation_standard:sec_3_32`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
已经被一个叶元素替换的叶元素不能再进行第二次替换操作。
```
- `规范化全文`: 3.32 检测无效的生命周期模式：替换操作造成分支 已经被一个叶元素替换的叶元素不能再进行第二次替换操作。 错误

### 3.33 检测无效的生命周期模式：对已删除叶元素的操作

- `clause_id`: `cn_ectd_validation_standard:sec_3_33`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
已经被删除的叶元素不能再做其他任何操作。
```
- `规范化全文`: 3.33 检测无效的生命周期模式：对已删除叶元素的操作 已经被删除的叶元素不能再做其他任何操作。 错误

### 3.34 检测无效的生命周期模式：对增补的叶元素进行增补操作

- `clause_id`: `cn_ectd_validation_standard:sec_3_34`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
不允许对一个操作属性为增补的叶元素使用增补操作。该规则不适用于STF的情况。
```
- `规范化全文`: 3.34 检测无效的生命周期模式：对增补的叶元素进行增补操作 不允许对一个操作属性为增补的叶元素使用增补操作。该规则不适用于STF的情况。 错误

### 3.35 检测无效的生命周期模式：对不是最新的STF叶元素进行增

- `clause_id`: `cn_ectd_validation_standard:sec_3_35`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `错误`
- `原始说明行`:
```text
补操作
对于STF叶元素，增补操作必须针对该文件最新版本来进行。
```
- `规范化全文`: 3.35 检测无效的生命周期模式：对不是最新的STF叶元素进行增 补操作 对于STF叶元素，增补操作必须针对该文件最新版本来进行。 错误

### 3.36 替换操作时语言属性不得变更

- `clause_id`: `cn_ectd_validation_standard:sec_3_36`
- `chapter_no`: `3`
- `source_heading`: `3. ICH 骨架文件`
- `严重程度`: `警告`
- `原始说明行`:
```text
对ICH骨架文件叶元素进行替换操作时，被替换的文件和替换后的文件应具有相同的语言属性。
```
- `规范化全文`: 3.36 替换操作时语言属性不得变更 对ICH骨架文件叶元素进行替换操作时，被替换的文件和替换后的文件应具有相同的语言属性。 警告

## 4. 区域性管理信息

### 4.1.1 模块一的区域骨架文件必须存在

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_1`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
m1\cn文件夹中必须包含cn-regional.xml文件。
```
- `规范化全文`: 4.1.1 模块一的区域骨架文件必须存在 m1\cn文件夹中必须包含cn-regional.xml文件。 错误

### 4.1.2 cn-regional.xml中对Schema的引用必须指向util文件夹中

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_2`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
提供的Schema
如何创建有效的引用请参考http://www.w3.org/TR/xml/ 和 http://www.ietf.org/rfc/rfc3986.txt
（2005版本 第22页，第3.3章节）。
```
- `规范化全文`: 4.1.2 cn-regional.xml中对Schema的引用必须指向util文件夹中 提供的Schema 如何创建有效的引用请参考http://www.w3.org/TR/xml/ 和 http://www.ietf.org/rfc/rfc3986.txt （2005版本 第22页，第3.3章节）。 错误

### 4.1.3 区域骨架文件必须有效

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_3`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
使用申请文件夹中（util/dtd目录）给定的Schema对区域骨架文件进行验证。
```
- `规范化全文`: 4.1.3 区域骨架文件必须有效 使用申请文件夹中（util/dtd目录）给定的Schema对区域骨架文件进行验证。 错误

### 4.1.4 当前序列必须使用一个不低于先前序列所用的schema版本

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_4`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
当前申请已使用较新版本Schema时，不允许返回早期版本。
```
- `规范化全文`: 4.1.4 当前序列必须使用一个不低于先前序列所用的schema版本 当前申请已使用较新版本Schema时，不允许返回早期版本。 错误

### 4.1.5 文件引用（xlink:href）属性指向的文件必须存在

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_5`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
XML叶元素中的文件引用链接必须有效，即引用的目标文件必须存在。
```
- `规范化全文`: 4.1.5 文件引用（xlink:href）属性指向的文件必须存在 XML叶元素中的文件引用链接必须有效，即引用的目标文件必须存在。 错误

### 4.1.6 文件在一个序列中不允许对应多个操作

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_6`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
区域骨架文件中的叶元素文件在同一个序列中不能作为“被修改文件对象（modified-file）”被多次
引用。
```
- `规范化全文`: 4.1.6 文件在一个序列中不允许对应多个操作 区域骨架文件中的叶元素文件在同一个序列中不能作为“被修改文件对象（modified-file）”被多次 引用。 错误

### 4.1.7 替换或增补的内容不能与之前文件相同

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_7`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
当执行“替换（replace）”或“增补（append）”操作时，新内容必须与之前的内容不同（即
checksum不能相同）。
```
- `规范化全文`: 4.1.7 替换或增补的内容不能与之前文件相同 当执行“替换（replace）”或“增补（append）”操作时，新内容必须与之前的内容不同（即 checksum不能相同）。 错误

### 4.1.8 叶元素：新建、替换或增补的叶元素，必须有“文件引用（xlink:href）”值

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_8`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
（xlink:href）”值
“操作（operation）”属性值为“新建（new）”、“替换（replace）”或“增补（append）”的
所有叶元素，必须有“文件引用（xlink:href）”值。
```
- `规范化全文`: 4.1.8 叶元素：新建、替换或增补的叶元素，必须有“文件引用（xlink:href）”值 错误

### 4.1.9 叶元素：删除的叶元素不能包含“文件引用（xlink:href）”值

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_9`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
”值
“操作（operation）”属性值为“删除（delete）”的所有叶元素，不能有“文件引用
（xlink:href）”值。
```
- `规范化全文`: 4.1.9 叶元素：删除的叶元素不能包含“文件引用（xlink:href）”值 错误

### 4.1.10 叶元素：对替换、删除和增补的叶元素，必须有对应的文件“操作（operation）”属性值为“替换（replace）”、“删除（delete）”或“增补（append）”的所有叶元素，对应的“被修改文件对象（modified-file）”必须有值。

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_10`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
”的所有叶元素，对应的“被修改文件对象（modified-file）”必须有值。
```
- `规范化全文`: 4.1.10 叶元素：对替换、删除和增补的叶元素，必须有对应的文件“操作（operation）”属性值为“替换（replace）”、“删除（delete）”或“增补（append）”的所有叶元素，对应的“被修改文件对象（modified-file）”必须有值。 错误

### 4.1.11 叶元素：初始序列中所有文件必须为新建

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_11`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
初始序列（即序列号为0000的序列）中所有内容文件的“操作（operation）”属性必须为“新建
（new）”。
```
- `规范化全文`: 4.1.11 叶元素：初始序列中所有文件必须为新建 初始序列（即序列号为0000的序列）中所有内容文件的“操作（operation）”属性必须为“新建 （new）”。 错误

### 4.1.12 “checksum-type”属性

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_12`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
“checksum-type”属性必须设置为“md5”或“MD5”。
```
- `规范化全文`: 4.1.12 “checksum-type”属性 “checksum-type”属性必须设置为“md5”或“MD5”。 错误

### 4.1.13 MD5值校验

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_13`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
所有文件的MD5 checksum必须和区域骨架文件中提供的checksum值保持一致。
```
- `规范化全文`: 4.1.13 MD5值校验 所有文件的MD5 checksum必须和区域骨架文件中提供的checksum值保持一致。 错误

### 4.1.14 说明函的“操作（operations）”属性

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_14`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
所有说明函的“操作（operation）”属性值必须为“新建（new）”。
```
- `规范化全文`: 4.1.14 说明函的“操作（operations）”属性 所有说明函的“操作（operation）”属性值必须为“新建（new）”。 错误

### 4.1.15 申请表的“操作（operations）”属性

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_15`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
序列类型为“首次提交”，且提交序列中包含申请表文件时，其“操作（operation）”属性值必须为
“新建（new）”。
```
- `规范化全文`: 4.1.15 申请表的“操作（operations）”属性 序列类型为“首次提交”，且提交序列中包含申请表文件时，其“操作（operation）”属性值必须为 “新建（new）”。 错误

### 4.1.16 不允许使用“扩展节点（Node Extension）”

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_16`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
不允许在区域骨架文件结构中使用“扩展节点（Node Extension）”。
```
- `规范化全文`: 4.1.16 不允许使用“扩展节点（Node Extension）” 不允许在区域骨架文件结构中使用“扩展节点（Node Extension）”。 错误

### 4.1.17 叶标题不能为空

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_17`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
所有叶元素必须有子元素<title>，叶标题不能为空。
```
- `规范化全文`: 4.1.17 叶标题不能为空 所有叶元素必须有子元素<title>，叶标题不能为空。 错误

### 4.1.18 删除的叶标题必须和被删除的叶标题保持一致

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_18`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
“操作（operation）”属性为“删除（delete）”的叶元素，其叶标题必须和被删除的叶标题保持一
致。
```
- `规范化全文`: 4.1.18 删除的叶标题必须和被删除的叶标题保持一致 “操作（operation）”属性为“删除（delete）”的叶元素，其叶标题必须和被删除的叶标题保持一 致。 错误

### 4.1.19 叶标题开头和结尾不能为空格

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_19`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `警告`
- `原始说明行`:
```text
叶标题开头和结尾不能为空格。
```
- `规范化全文`: 4.1.19 叶标题开头和结尾不能为空格 叶标题开头和结尾不能为空格。 警告

### 4.1.20 元素下必须有叶元素

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_20`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
名称以“cn-”开始的元素必须有叶元素。
```
- `规范化全文`: 4.1.20 元素下必须有叶元素 名称以“cn-”开始的元素必须有叶元素。 错误

### 4.1.21 申请文件夹名称

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_21`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请文件夹名称必须和信封信息中的申请编号保持一致。
```
- `规范化全文`: 4.1.21 申请文件夹名称 申请文件夹名称必须和信封信息中的申请编号保持一致。 错误

### 4.1.22 序列文件夹名称

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_22`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
序列文件夹名称必须和信封信息中的序列号保持一致。
```
- `规范化全文`: 4.1.22 序列文件夹名称 序列文件夹名称必须和信封信息中的序列号保持一致。 错误

### 4.1.23 m1文件夹目录结构

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_23`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
模块一输出文件夹基本结构必须遵循《eCTD技术规范V1.0》第4章节中的规定。
```
- `规范化全文`: 4.1.23 m1文件夹目录结构 模块一输出文件夹基本结构必须遵循《eCTD技术规范V1.0》第4章节中的规定。 错误

### 4.1.24 内容的引用

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_24`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
区域骨架文件（cn-regional.xml）中不允许包含跨申请引用。当在区域骨架文件中引用另一个序列的
内容时，只能引用同一申请中早先已提交序列的内容。
```
- `规范化全文`: 4.1.24 内容的引用 区域骨架文件（cn-regional.xml）中不允许包含跨申请引用。当在区域骨架文件中引用另一个序列的 内容时，只能引用同一申请中早先已提交序列的内容。 错误

### 4.1.25 检测无效的生命周期模式：增补操作造成分支

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_25`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
已经被一个叶元素替换的叶元素不能再进行增补操作。
```
- `规范化全文`: 4.1.25 检测无效的生命周期模式：增补操作造成分支 已经被一个叶元素替换的叶元素不能再进行增补操作。 错误

### 4.1.26 检测无效的生命周期模式：删除操作造成分支

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_26`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
已经被一个叶元素替换的叶元素不能再进行删除操作。
```
- `规范化全文`: 4.1.26 检测无效的生命周期模式：删除操作造成分支 已经被一个叶元素替换的叶元素不能再进行删除操作。 错误

### 4.1.27 检测无效的生命周期模式：替换操作造成分支

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_27`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
已经被一个叶元素替换的叶元素不能再进行第二次替换操作。
```
- `规范化全文`: 4.1.27 检测无效的生命周期模式：替换操作造成分支 已经被一个叶元素替换的叶元素不能再进行第二次替换操作。 错误

### 4.1.28 检测无效的生命周期模式：对已删除叶元素的操作

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_28`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
已经被删除的叶元素不能再做其他任何操作。
```
- `规范化全文`: 4.1.28 检测无效的生命周期模式：对已删除叶元素的操作 已经被删除的叶元素不能再做其他任何操作。 错误

### 4.1.29 检测无效的生命周期模式：对增补的叶元素进行增补操作

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_29`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
不允许对一个操作属性为增补的叶元素使用增补操作。该规则不适用于STF的情况。
```
- `规范化全文`: 4.1.29 检测无效的生命周期模式：对增补的叶元素进行增补操作 不允许对一个操作属性为增补的叶元素使用增补操作。该规则不适用于STF的情况。 错误

### 4.1.30 增补（append）的使用

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_30`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `警告`
- `原始说明行`:
```text
不建议在区域骨架文件中使用“增补（append）”操作属性。在区域骨架文件中使用“增补
（append）”操作，需要在说明函中进行说明。
```
- `规范化全文`: 4.1.30 增补（append）的使用 不建议在区域骨架文件中使用“增补（append）”操作属性。在区域骨架文件中使用“增补 （append）”操作，需要在说明函中进行说明。 警告

### 4.1.31 替换操作时语言属性不得变更

- `clause_id`: `cn_ectd_validation_standard:sec_4_1_31`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `警告`
- `原始说明行`:
```text
对区域骨架文件叶元素进行替换操作时，被替换的文件和替换后的文件应具有相同的语言属性。
```
- `规范化全文`: 4.1.31 替换操作时语言属性不得变更 对区域骨架文件叶元素进行替换操作时，被替换的文件和替换后的文件应具有相同的语言属性。 警告

### 4.2.1 信封元素：申请编号

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_1`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请编号必须符合《eCTD技术规范V1.0》中的的编码规则。
```
- `规范化全文`: 4.2.1 信封元素：申请编号 申请编号必须符合《eCTD技术规范V1.0》中的的编码规则。 错误

### 4.2.2 信封元素：申请类型

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_2`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请类型必须参考“cv-application-type.xml”中的定义。
```
- `规范化全文`: 4.2.2 信封元素：申请类型 申请类型必须参考“cv-application-type.xml”中的定义。 错误

### 4.2.3 信封元素：产品类型

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_3`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
产品类型必须参考“cv-product-type.xml”中的定义。
```
- `规范化全文`: 4.2.3 信封元素：产品类型 产品类型必须参考“cv-product-type.xml”中的定义。 错误

### 4.2.4 信封元素：原始编号

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_4`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
原始编号不能为空。
```
- `规范化全文`: 4.2.4 信封元素：原始编号 原始编号不能为空。 错误

### 4.2.5 信封元素：相关序列

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_5`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
相关序列必须是4位数字，且小于或等于当前序列的序列号。
```
- `规范化全文`: 4.2.5 信封元素：相关序列 相关序列必须是4位数字，且小于或等于当前序列的序列号。 错误

### 4.2.6 信封元素：注册行为类型

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_6`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
注册行为类型必须参考“cv-regulatory-activity-type.xml”中的定义。
```
- `规范化全文`: 4.2.6 信封元素：注册行为类型 注册行为类型必须参考“cv-regulatory-activity-type.xml”中的定义。 错误

### 4.2.7 信封元素：序列号

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_7`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
序列号必须由四位数字组成。
```
- `规范化全文`: 4.2.7 信封元素：序列号 序列号必须由四位数字组成。 错误

### 4.2.8 信封元素：序列类型

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_8`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
序列类型必须参考“cv-sequence-type.xml”中的定义。
```
- `规范化全文`: 4.2.8 信封元素：序列类型 序列类型必须参考“cv-sequence-type.xml”中的定义。 错误

### 4.2.9 信封元素：序列描述

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_9`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
序列描述不能为空，且总长度不能超过120个中文字符。
```
- `规范化全文`: 4.2.9 信封元素：序列描述 序列描述不能为空，且总长度不能超过120个中文字符。 错误

### 4.2.10 信封元素：序列相关信息

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_10`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
提交序列的申请类型、注册行为类型、序列类型之间的关联关系必须参考“depend-apt-rat-sqt.xml
”中的定义。
```
- `规范化全文`: 4.2.10 信封元素：序列相关信息 提交序列的申请类型、注册行为类型、序列类型之间的关联关系必须参考“depend-apt-rat-sqt.xml ”中的定义。 错误

### 4.2.11 相关序列的值

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_11`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
如果序列类型不是首次提交或格式转换，则相关序列不应与当前序列号相同。
```
- `规范化全文`: 4.2.11 相关序列的值 如果序列类型不是首次提交或格式转换，则相关序列不应与当前序列号相同。 错误

### 4.2.12 相关序列的值

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_12`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
如果序列类型是首次提交或格式转换，则相关序列应与当前序列号相同。
```
- `规范化全文`: 4.2.12 相关序列的值 如果序列类型是首次提交或格式转换，则相关序列应与当前序列号相同。 错误

### 4.2.13 申请级别的信封元素必须保持不变

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_13`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
初始序列中使用的申请级别的信封元素，即申请编号、申请类型、产品类型、原始编号的值在整个生
命周期中不能更改。
```
- `规范化全文`: 4.2.13 申请级别的信封元素必须保持不变 初始序列中使用的申请级别的信封元素，即申请编号、申请类型、产品类型、原始编号的值在整个生 命周期中不能更改。 错误

### 4.2.14 注册行为类型必须保持不变

- `clause_id`: `cn_ectd_validation_standard:sec_4_2_14`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
对于属于同一个注册行为的所有序列，其注册行为类型的值必须相同。
```
- `规范化全文`: 4.2.14 注册行为类型必须保持不变 对于属于同一个注册行为的所有序列，其注册行为类型的值必须相同。 错误

### 4.3.1 模块一完整性验证：

- `clause_id`: `cn_ectd_validation_standard:sec_4_3_1`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请类型：新药申请
注册行为类型：首次申请或新适应症
序列类型：首次提交
针对上述类型，提交序列必须包含以下元素
cn-1-0，cn-1-2，cn-1-3，cn-1-3-1，cn-1-3-1-2，cn-1-3-2，cn-1-3-2-2，cn-1-3-3，cn-1-3-
8，cn-1-3-8-1，cn-1-3-8-2，cn-1-11
```
- `规范化全文`: 4.3.1 模块一完整性验证： 申请类型：新药申请 注册行为类型：首次申请或新适应症 序列类型：首次提交 针对上述类型，提交序列必须包含以下元素 cn-1-0，cn-1-2，cn-1-3，cn-1-3-1，cn-1-3-1-2，cn-1-3-2，cn-1-3-2-2，cn-1-3-3，cn-1-3- 8，cn-1-3-8-1，cn-1-3-8-2，cn-1-11 错误

### 4.3.2 模块一完整性验证：

- `clause_id`: `cn_ectd_validation_standard:sec_4_3_2`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请类型：新药申请
注册行为类型：首次申请或新适应症
序列类型：首次提交
针对上述类型，提交序列不能包含以下元素
cn-1-3-1-1，cn-1-3-2-1，cn-1-3-4，cn-1-3-4-1，cn-1-3-4-2，cn-1-3-4-3
```
- `规范化全文`: 4.3.2 模块一完整性验证： 申请类型：新药申请 注册行为类型：首次申请或新适应症 序列类型：首次提交 针对上述类型，提交序列不能包含以下元素 cn-1-3-1-1，cn-1-3-2-1，cn-1-3-4，cn-1-3-4-1，cn-1-3-4-2，cn-1-3-4-3 错误

### 4.3.3 模块一完整性验证：

- `clause_id`: `cn_ectd_validation_standard:sec_4_3_3`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请类型：仿制药申请
注册行为类型：首次申请或新适应症
序列类型：首次提交
针对上述类型，提交序列必须包含以下元素
cn-1-0，cn-1-2，cn-1-3，cn-1-3-1，cn-1-3-1-2，cn-1-3-2，cn-1-3-2-2，cn-1-3-3，cn-1-3-
8，cn-1-3-8-1，cn-1-3-8-2，cn-1-11
```
- `规范化全文`: 4.3.3 模块一完整性验证： 申请类型：仿制药申请 注册行为类型：首次申请或新适应症 序列类型：首次提交 针对上述类型，提交序列必须包含以下元素 cn-1-0，cn-1-2，cn-1-3，cn-1-3-1，cn-1-3-1-2，cn-1-3-2，cn-1-3-2-2，cn-1-3-3，cn-1-3- 8，cn-1-3-8-1，cn-1-3-8-2，cn-1-11 错误

### 4.3.4 模块一完整性验证：

- `clause_id`: `cn_ectd_validation_standard:sec_4_3_4`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请类型：仿制药申请
注册行为类型：首次申请或新适应症
序列类型：首次提交
针对上述类型，提交序列不能包含以下元素
cn-1-3-1-1，cn-1-3-2-1，cn-1-3-4，cn-1-3-4-1，cn-1-3-4-2，cn-1-3-4-3，cn-1-12
```
- `规范化全文`: 4.3.4 模块一完整性验证： 申请类型：仿制药申请 注册行为类型：首次申请或新适应症 序列类型：首次提交 针对上述类型，提交序列不能包含以下元素 cn-1-3-1-1，cn-1-3-2-1，cn-1-3-4，cn-1-3-4-1，cn-1-3-4-2，cn-1-3-4-3，cn-1-12 错误

### 4.3.5 模块一完整性验证：

- `clause_id`: `cn_ectd_validation_standard:sec_4_3_5`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请类型：临床试验申请
注册行为类型：首次申请或新适应症和联合用药
序列类型：首次提交
针对上述类型，提交序列必须包含以下元素
cn-1-0，cn-1-2，cn-1-3，cn-1-3-1，cn-1-3-1-1，cn-1-3-2，cn-1-3-2-1，cn-1-3-3，cn-1-3-
4，cn-1-3-4-1，cn-1-3-4-2，cn-1-3-4-3，cn-1-3-8，cn-1-3-8-1，cn-1-3-8-2，cn-1-11
```
- `规范化全文`: 4.3.5 模块一完整性验证： 申请类型：临床试验申请 注册行为类型：首次申请或新适应症和联合用药 序列类型：首次提交 针对上述类型，提交序列必须包含以下元素 cn-1-0，cn-1-2，cn-1-3，cn-1-3-1，cn-1-3-1-1，cn-1-3-2，cn-1-3-2-1，cn-1-3-3，cn-1-3- 4，cn-1-3-4-1，cn-1-3-4-2，cn-1-3-4-3，cn-1-3-8，cn-1-3-8-1，cn-1-3-8-2，cn-1-11 错误

### 4.3.6 模块一完整性验证：

- `clause_id`: `cn_ectd_validation_standard:sec_4_3_6`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请类型：临床试验申请
注册行为类型：首次申请或新适应症和联合用药
序列类型：首次提交
针对上述类型，提交序列不能包含以下元素
cn-1-3-1-2，cn-1-3-2-2，cn-1-3-5，cn-1-3-6
```
- `规范化全文`: 4.3.6 模块一完整性验证： 申请类型：临床试验申请 注册行为类型：首次申请或新适应症和联合用药 序列类型：首次提交 针对上述类型，提交序列不能包含以下元素 cn-1-3-1-2，cn-1-3-2-2，cn-1-3-5，cn-1-3-6 错误

### 4.3.7 模块一完整性验证：

- `clause_id`: `cn_ectd_validation_standard:sec_4_3_7`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请类型：临床试验申请
注册行为类型：研发期间安全性报告
序列类型：首次提交
针对上述类型，提交序列必须包含以下两个元素之一
cn-1-8-1，cn-1-8-2
```
- `规范化全文`: 4.3.7 模块一完整性验证： 申请类型：临床试验申请 注册行为类型：研发期间安全性报告 序列类型：首次提交 针对上述类型，提交序列必须包含以下两个元素之一 cn-1-8-1，cn-1-8-2 错误

### 4.3.8 模块一完整性验证：

- `clause_id`: `cn_ectd_validation_standard:sec_4_3_8`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `错误`
- `原始说明行`:
```text
申请类型：临床试验申请
注册行为类型：研发期间安全性报告
序列类型：首次提交
针对上述类型，提交序列不能同时包含以下两个元素
cn-1-8-1，cn-1-8-2
```
- `规范化全文`: 4.3.8 模块一完整性验证： 申请类型：临床试验申请 注册行为类型：研发期间安全性报告 序列类型：首次提交 针对上述类型，提交序列不能同时包含以下两个元素 cn-1-8-1，cn-1-8-2 错误

### 4.3.9 元素包含的叶元素数量

- `clause_id`: `cn_ectd_validation_standard:sec_4_3_9`
- `chapter_no`: `4`
- `source_heading`: `4. 区域性管理信息`
- `严重程度`: `提示信息`
- `原始说明行`:
```text
针对以下元素，其包含的叶元素数量建议不超过一个，如该元素中包含多个PDF文件，建议合并为一
个文件提交
cn-1-1，cn-1-3-2-1，cn-1-3-2-2，cn-1-3-8-8，cn-1-4-1，cn-1-4-2，cn-1-4-3，cn-1-4-4，
cn-1-4-5，cn-1-4-6，cn-1-4-7，cn-1-6-1，cn-1-6-3，cn-1-10-1，cn-1-10-2，cn-1-10-3，cn-
```
- `规范化全文`: 4.3.9 元素包含的叶元素数量 针对以下元素，其包含的叶元素数量建议不超过一个，如该元素中包含多个PDF文件，建议合并为一 个文件提交 cn-1-1，cn-1-3-2-1，cn-1-3-2-2，cn-1-3-8-8，cn-1-4-1，cn-1-4-2，cn-1-4-3，cn-1-4-4， cn-1-4-5，cn-1-4-6，cn-1-4-7，cn-1-6-1，cn-1-6-3，cn-1-10-1，cn-1-10-2，cn-1-10-3，cn- 提示信息

## 5. 研究标签文件（STF）

### 5.1 STF文件必须有效

- `clause_id`: `cn_ectd_validation_standard:sec_5_1`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `错误`
- `原始说明行`:
```text
STF文件必须根据ich-stf-v2-2.dtd创建并保证有效性。
```
- `规范化全文`: 5.1 STF文件必须有效 STF文件必须根据ich-stf-v2-2.dtd创建并保证有效性。 错误

### 5.2 检查索引引用

- `clause_id`: `cn_ectd_validation_standard:sec_5_2`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `错误`
- `原始说明行`:
```text
“文件引用（xlink:href）”对应的被引用文件必须存在。
```
- `规范化全文`: 5.2 检查索引引用 “文件引用（xlink:href）”对应的被引用文件必须存在。 错误

### 5.3 不建议使用“content-block”元素

- `clause_id`: `cn_ectd_validation_standard:sec_5_3`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
不建议使用“content-block”元素。
```
- `规范化全文`: 5.3 不建议使用“content-block”元素 不建议使用“content-block”元素。 警告

### 5.4 文件引用值不能使用反斜杠

- `clause_id`: `cn_ectd_validation_standard:sec_5_4`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `错误`
- `原始说明行`:
```text
文件引用（xlink:href）的值不能包含反斜杠（"\"）。
```
- `规范化全文`: 5.4 文件引用值不能使用反斜杠 文件引用（xlink:href）的值不能包含反斜杠（"\"）。 错误

### 5.5 研究标识的类别不能空

- `clause_id`: `cn_ectd_validation_standard:sec_5_5`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
研究标识/类别（study-identifier>>category）值不能是空的。
```
- `规范化全文`: 5.5 研究标识的类别不能空 研究标识/类别（study-identifier>>category）值不能是空的。 警告

### 5.6 研究标识的研究ID不能为空

- `clause_id`: `cn_ectd_validation_standard:sec_5_6`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
研究标识/研究ID（study-identifier>>study-id）值不能是空的。
```
- `规范化全文`: 5.6 研究标识的研究ID不能为空 研究标识/研究ID（study-identifier>>study-id）值不能是空的。 警告

### 5.7 研究标识的标题必须与叶元素的叶标题匹配

- `clause_id`: `cn_ectd_validation_standard:sec_5_7`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
研究标识/标题（study-identifier>>title）的值必须与骨架文件（index.xml）中相应STF叶标题保持
一致。
```
- `规范化全文`: 5.7 研究标识的标题必须与叶元素的叶标题匹配 研究标识/标题（study-identifier>>title）的值必须与骨架文件（index.xml）中相应STF叶标题保持 一致。 警告

### 5.8 标签属性和类别元素的值

- `clause_id`: `cn_ectd_validation_standard:sec_5_8`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
根据“valid-values.xml”文件（版本5）定义的内容检查STF文件的标签属性（tag）值和类别属性
（category）值是否有效。
```
- `规范化全文`: 5.8 标签属性和类别元素的值 根据“valid-values.xml”文件（版本5）定义的内容检查STF文件的标签属性（tag）值和类别属性 （category）值是否有效。 警告

### 5.9 STF的增补操作

- `clause_id`: `cn_ectd_validation_standard:sec_5_9`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
对STF叶元素进行增补操作时，其“被修改文件对象（modified-file）”必须是另一个STF叶元素。
```
- `规范化全文`: 5.9 STF的增补操作 对STF叶元素进行增补操作时，其“被修改文件对象（modified-file）”必须是另一个STF叶元素。 警告

### 5.10 STF必须提供类别元素（category）信息

- `clause_id`: `cn_ectd_validation_standard:sec_5_10`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
见《ICH eCTD STF标准文件 V2.6.1》
类别（category）为研究组织提供了一个额外的级别，该级别是eCTD DTD目前不能提供的。该元素
仅与下列CTD章节的研究有关：4.2.3.1，4.2.3.2，4.2.3.4.1，5.3.5.1。在上述章节，需提供类别
（category）信息。
```
- `规范化全文`: 5.10 STF必须提供类别元素（category）信息 见《ICH eCTD STF标准文件 V2.6.1》 类别（category）为研究组织提供了一个额外的级别，该级别是eCTD DTD目前不能提供的。该元素 仅与下列CTD章节的研究有关：4.2.3.1，4.2.3.2，4.2.3.4.1，5.3.5.1。在上述章节，需提供类别 （category）信息。 警告

### 5.11 STF不能引用另一个STF

- `clause_id`: `cn_ectd_validation_standard:sec_5_11`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
STF中的引用对象必须是有目标内容的文件，而不是另一个STF。
```
- `规范化全文`: 5.11 STF不能引用另一个STF STF中的引用对象必须是有目标内容的文件，而不是另一个STF。 警告

### 5.12 STF文件必须至少引用一个叶元素

- `clause_id`: `cn_ectd_validation_standard:sec_5_12`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
不关联任何叶元素的STF会提示警告信息。
```
- `规范化全文`: 5.12 STF文件必须至少引用一个叶元素 不关联任何叶元素的STF会提示警告信息。 警告

### 5.13 STF的“study-id”值必须保持一致

- `clause_id`: `cn_ectd_validation_standard:sec_5_13`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
在申请的全生命周期内，同一个STF的“ study-id”值必须保持不变。
```
- `规范化全文`: 5.13 STF的“study-id”值必须保持一致 在申请的全生命周期内，同一个STF的“ study-id”值必须保持不变。 警告

### 5.14 无效的STF目录位置

- `clause_id`: `cn_ectd_validation_standard:sec_5_14`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
STF只存在于模块四（4.2.x章节）和模块五（5.3.1.x-5.3.5.x章节）中。
```
- `规范化全文`: 5.14 无效的STF目录位置 STF只存在于模块四（4.2.x章节）和模块五（5.3.1.x-5.3.5.x章节）中。 警告

### 5.15 STF“doc-content”的标签（file-tag）数量

- `clause_id`: `cn_ectd_validation_standard:sec_5_15`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
每个“doc-content”元素有且仅有1个“文件标签（file-tag）”。
```
- `规范化全文`: 5.15 STF“doc-content”的标签（file-tag）数量 每个“doc-content”元素有且仅有1个“文件标签（file-tag）”。 警告

### 5.16 5.3.7章节病例报告表结构

- `clause_id`: `cn_ectd_validation_standard:sec_5_16`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `错误`
- `原始说明行`:
```text
如果当前序列使用了STF，5.3.7章节将禁止被使用。病例报告表必须在STF中被引用和展现。
```
- `规范化全文`: 5.16 5.3.7章节病例报告表结构 如果当前序列使用了STF，5.3.7章节将禁止被使用。病例报告表必须在STF中被引用和展现。 错误

### 5.17 使用STF

- `clause_id`: `cn_ectd_validation_standard:sec_5_17`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `错误`
- `原始说明行`:
```text
第4.2章节中的叶元素必须使用STF引用。
第5.3.1至5.3.5章节中的叶元素必须使用STF引用。
```
- `规范化全文`: 5.17 使用STF 第4.2章节中的叶元素必须使用STF引用。 第5.3.1至5.3.5章节中的叶元素必须使用STF引用。 错误

### 5.18 临床试验数据集相关文件必须使用正确的STF文件标签

- `clause_id`: `cn_ectd_validation_standard:sec_5_18`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
模块五（5.3章节）中xpt格式的临床数据集文件的有效文件标签有：data-tabulation-dataset-
legacy；data-tabulation-dataset-sdtm；analysis-dataset-adam；analysis-dataset-legacy
XML格式的数据说明文件（define.xml）的有效文件标签有：data-tabulation-data-definition；
analysis-data-definition
```
- `规范化全文`: 5.18 临床试验数据集相关文件必须使用正确的STF文件标签 模块五（5.3章节）中xpt格式的临床数据集文件的有效文件标签有：data-tabulation-dataset- legacy；data-tabulation-dataset-sdtm；analysis-dataset-adam；analysis-dataset-legacy XML格式的数据说明文件（define.xml）的有效文件标签有：data-tabulation-data-definition； analysis-data-definition 警告

### 5.19 申请人递交的药物临床试验数据必须有正确的数据集文件和

- `clause_id`: `cn_ectd_validation_standard:sec_5_19`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
数据说明文件
申请人提交的原始数据库中都应包含dm数据集和对应的数据说明文件，分析数据库中都应包含adsl数
据集和对应的数据说明文件。
```
- `规范化全文`: 5.19 申请人递交的药物临床试验数据必须有正确的数据集文件和 数据说明文件 申请人提交的原始数据库中都应包含dm数据集和对应的数据说明文件，分析数据库中都应包含adsl数 据集和对应的数据说明文件。 警告

### 5.20 同一个研究下提交的数据集不能重名

- `clause_id`: `cn_ectd_validation_standard:sec_5_20`
- `chapter_no`: `5`
- `source_heading`: `5. 研究标签文件（STF）`
- `严重程度`: `警告`
- `原始说明行`:
```text
对于模块五（5.3章节）中的同一研究，不应存在相同名称的数据集。
```
- `规范化全文`: 5.20 同一个研究下提交的数据集不能重名 对于模块五（5.3章节）中的同一研究，不应存在相同名称的数据集。 警告

## 6. PDF分析

### 6.1 PDF文件必须可读。

- `clause_id`: `cn_ectd_validation_standard:sec_6_1`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `错误`
- `原始说明行`:
```text
针对被破坏或不可读的PDF文件（因内容无效，或页码数为0）进行检查，并提示错误信息。
```
- `规范化全文`: 6.1 PDF文件必须可读。 针对被破坏或不可读的PDF文件（因内容无效，或页码数为0）进行检查，并提示错误信息。 错误

### 6.2 书签必须指向相对路径

- `clause_id`: `cn_ectd_validation_standard:sec_6_2`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
PDF文件中必须使用指向相对路径的书签，不允许使用指向绝对路径的书签。参考文献（2.7.5章节、
3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.2 书签必须指向相对路径 PDF文件中必须使用指向相对路径的书签，不允许使用指向绝对路径的书签。参考文献（2.7.5章节、 3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.3 有网页、邮箱地址或其他外部链接的书签

- `clause_id`: `cn_ectd_validation_standard:sec_6_3`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `错误`
- `原始说明行`:
```text
PDF文件中不允许使用包含网页链接、电子邮箱地址或其他外部链接的书签。参考文献（2.7.5章节、
3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.3 有网页、邮箱地址或其他外部链接的书签 PDF文件中不允许使用包含网页链接、电子邮箱地址或其他外部链接的书签。参考文献（2.7.5章节、 3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 错误

### 6.4 不允许有未知动作的书签

- `clause_id`: `cn_ectd_validation_standard:sec_6_4`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `错误`
- `原始说明行`:
```text
不允许使用GoTo，GoToR和Launch以外的操作类型的书签。参考文献（2.7.5章节、3.3章节、4.3章
节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.4 不允许有未知动作的书签 不允许使用GoTo，GoToR和Launch以外的操作类型的书签。参考文献（2.7.5章节、3.3章节、4.3章 节、5.4章节）和外文参考资料不做此要求。 错误

### 6.5 书签不能失效

- `clause_id`: `cn_ectd_validation_standard:sec_6_5`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
不允许在PDF文件中使用失效的书签（例如：未分配任何操作的书签）。参考文献（2.7.5章节、3.3章
节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.5 书签不能失效 不允许在PDF文件中使用失效的书签（例如：未分配任何操作的书签）。参考文献（2.7.5章节、3.3章 节、4.3章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.6 书签不能被损坏

- `clause_id`: `cn_ectd_validation_standard:sec_6_6`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
PDF文件中包含损坏的书签（例如：书签指向地址不存在）时将提示警告信息。参考文献（2.7.5章节
、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.6 书签不能被损坏 PDF文件中包含损坏的书签（例如：书签指向地址不存在）时将提示警告信息。参考文献（2.7.5章节 、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.7 书签不能有多个动作

- `clause_id`: `cn_ectd_validation_standard:sec_6_7`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
不允许有指定多个动作的书签（例如，打开两个不同的页面）。参考文献（2.7.5章节、3.3章节、4.3
章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.7 书签不能有多个动作 不允许有指定多个动作的书签（例如，打开两个不同的页面）。参考文献（2.7.5章节、3.3章节、4.3 章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.8 书签必须承前缩放（Inherit Zoom）

- `clause_id`: `cn_ectd_validation_standard:sec_6_8`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
所有的书签的放大率设置应为承前缩放（Inherit Zoom）。参考文献（2.7.5章节、3.3章节、4.3章节
、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.8 书签必须承前缩放（Inherit Zoom） 所有的书签的放大率设置应为承前缩放（Inherit Zoom）。参考文献（2.7.5章节、3.3章节、4.3章节 、5.4章节）和外文参考资料不做此要求。 警告

### 6.9 超文本链接必须使用相对路径

- `clause_id`: `cn_ectd_validation_standard:sec_6_9`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
PDF文件中的超文本链接必须使用相对路径，不允许使用绝对路径。参考文献（2.7.5章节、3.3章节、
4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.9 超文本链接必须使用相对路径 PDF文件中的超文本链接必须使用相对路径，不允许使用绝对路径。参考文献（2.7.5章节、3.3章节、 4.3章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.10 有网页、邮箱地址或其他外部链接的超文本链接

- `clause_id`: `cn_ectd_validation_standard:sec_6_10`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `错误`
- `原始说明行`:
```text
PDF文件中不允许使用包含网页链接、电子邮箱地址或其他外部链接的超文本链接。参考文献（2.7.5
章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.10 有网页、邮箱地址或其他外部链接的超文本链接 PDF文件中不允许使用包含网页链接、电子邮箱地址或其他外部链接的超文本链接。参考文献（2.7.5 章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 错误

### 6.11 不允许有未知动作的超文本链接

- `clause_id`: `cn_ectd_validation_standard:sec_6_11`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `错误`
- `原始说明行`:
```text
不允许在PDF文件使用除GoTo, GoToR和Launch以外动作类型的超文本链接。参考文献（2.7.5章节
、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.11 不允许有未知动作的超文本链接 不允许在PDF文件使用除GoTo, GoToR和Launch以外动作类型的超文本链接。参考文献（2.7.5章节 、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 错误

### 6.12 超文本链接不能处于失效状态

- `clause_id`: `cn_ectd_validation_standard:sec_6_12`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
不允许在PDF文件中使用失效的超文本链接（例如：未指定任何操作的超文本链接）。参考文献
（2.7.5章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.12 超文本链接不能处于失效状态 不允许在PDF文件中使用失效的超文本链接（例如：未指定任何操作的超文本链接）。参考文献 （2.7.5章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.13 超文本链接不能被损坏

- `clause_id`: `cn_ectd_validation_standard:sec_6_13`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
PDF文件中包含损坏的超文本链接（例如：超文本链接指向地址不存在）时将提示警告信息。参考文
献（2.7.5章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.13 超文本链接不能被损坏 PDF文件中包含损坏的超文本链接（例如：超文本链接指向地址不存在）时将提示警告信息。参考文 献（2.7.5章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.14 超文本链接不能有多个动作

- `clause_id`: `cn_ectd_validation_standard:sec_6_14`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `错误`
- `原始说明行`:
```text
不允许设置有多个指定动作(例如，打开两个不同的页面)的超文本链接。参考文献（2.7.5章节、3.3章
节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.14 超文本链接不能有多个动作 不允许设置有多个指定动作(例如，打开两个不同的页面)的超文本链接。参考文献（2.7.5章节、3.3章 节、4.3章节、5.4章节）和外文参考资料不做此要求。 错误

### 6.15 超文本链接必须承前缩放（Inherit Zoom）

- `clause_id`: `cn_ectd_validation_standard:sec_6_15`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
所有超文本链接的放大率设置应为承前缩放（Inherit Zoom）。参考文献（2.7.5章节、3.3章节、4.3
章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.15 超文本链接必须承前缩放（Inherit Zoom） 所有超文本链接的放大率设置应为承前缩放（Inherit Zoom）。参考文献（2.7.5章节、3.3章节、4.3 章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.16 PDF版本必须正确

- `clause_id`: `cn_ectd_validation_standard:sec_6_16`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
允许的PDF文件版本为1.4，1.5，1.6，1.7，PDF/A-1，PDF/A-2。参考文献（2.7.5章节、3.3章节、
4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.16 PDF版本必须正确 允许的PDF文件版本为1.4，1.5，1.6，1.7，PDF/A-1，PDF/A-2。参考文献（2.7.5章节、3.3章节、 4.3章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.17 不允许带附件的PDF文件

- `clause_id`: `cn_ectd_validation_standard:sec_6_17`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `错误`
- `原始说明行`:
```text
PDF文件中不能嵌入任何附件。参考文献（2.7.5章节、3.3章节、4.3章节、5.4章节）和外文参考资料
不做此要求。
```
- `规范化全文`: 6.17 不允许带附件的PDF文件 PDF文件中不能嵌入任何附件。参考文献（2.7.5章节、3.3章节、4.3章节、5.4章节）和外文参考资料 不做此要求。 错误

### 6.18 有注释的PDF文件

- `clause_id`: `cn_ectd_validation_standard:sec_6_18`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `提示信息`
- `原始说明行`:
```text
除了超文本链接以外，PDF文件中不能包含其他注释。参考文献（2.7.5章节、3.3章节、4.3章节、5.4
章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.18 有注释的PDF文件 除了超文本链接以外，PDF文件中不能包含其他注释。参考文献（2.7.5章节、3.3章节、4.3章节、5.4 章节）和外文参考资料不做此要求。 提示信息

### 6.19 PDF文件不能有任何安全设置

- `clause_id`: `cn_ectd_validation_standard:sec_6_19`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `错误`
- `原始说明行`:
```text
不能提交有安全设置的PDF文件，例如限制选择文本或图形等。参考文献（2.7.5章节、3.3章节、4.3
章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.19 PDF文件不能有任何安全设置 不能提交有安全设置的PDF文件，例如限制选择文本或图形等。参考文献（2.7.5章节、3.3章节、4.3 章节、5.4章节）和外文参考资料不做此要求。 错误

### 6.20 PDF初始视图正确

- `clause_id`: `cn_ectd_validation_standard:sec_6_20`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
根据《ICH eCTD技术规范V3.2.2》：有书签的文件在初始视图中应显示书签。放大率和页面布局应设
置为默认。参考文献（2.7.5章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.20 PDF初始视图正确 根据《ICH eCTD技术规范V3.2.2》：有书签的文件在初始视图中应显示书签。放大率和页面布局应设 置为默认。参考文献（2.7.5章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.21 PDF文件不能有密码保护

- `clause_id`: `cn_ectd_validation_standard:sec_6_21`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `错误`
- `原始说明行`:
```text
不能提交使用密码保护且无法打开的文件。
```
- `规范化全文`: 6.21 PDF文件不能有密码保护 不能提交使用密码保护且无法打开的文件。 错误

### 6.22 PDF应该设置启用“快速Web访问（Fast Web Access）”

- `clause_id`: `cn_ectd_validation_standard:sec_6_22`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `提示信息`
- `原始说明行`:
```text
不能提交未启用“快速Web访问（Fast Web Access）”情况下创建的PDF文件。参考文献（2.7.5章
节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.22 PDF应该设置启用“快速Web访问（Fast Web Access）” 不能提交未启用“快速Web访问（Fast Web Access）”情况下创建的PDF文件。参考文献（2.7.5章 节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 提示信息

### 6.23 大于5页的文件必须有书签

- `clause_id`: `cn_ectd_validation_standard:sec_6_23`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
除外文参考资料、参考文献（2.7.5章节、3.3章节、4.3章节、5.4章节）和申请表之外，大于5页的文
件必须有书签。
```
- `规范化全文`: 6.23 大于5页的文件必须有书签 除外文参考资料、参考文献（2.7.5章节、3.3章节、4.3章节、5.4章节）和申请表之外，大于5页的文 件必须有书签。 警告

### 6.24 PDF内容限制

- `clause_id`: `cn_ectd_validation_standard:sec_6_24`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
PDF文件不能包含JavaScript，3D内容或动态内容（音频/视频）。参考文献（2.7.5章节、3.3章节、
4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.24 PDF内容限制 PDF文件不能包含JavaScript，3D内容或动态内容（音频/视频）。参考文献（2.7.5章节、3.3章节、 4.3章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.25 PDF内容可搜索

- `clause_id`: `cn_ectd_validation_standard:sec_6_25`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
PDF文件中的文本必须可搜索。 如果是扫描页面，则应使用OCR提供可搜索的文本。参考文献（2.7.5
章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `规范化全文`: 6.25 PDF内容可搜索 PDF文件中的文本必须可搜索。 如果是扫描页面，则应使用OCR提供可搜索的文本。参考文献（2.7.5 章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 警告

### 6.26 如使用非标准字体，需嵌入在PDF文件中

- `clause_id`: `cn_ectd_validation_standard:sec_6_26`
- `chapter_no`: `6`
- `source_heading`: `6. PDF分析`
- `严重程度`: `警告`
- `原始说明行`:
```text
PDF文件应尽量使用标准字体。如果包含非标准字体，则需在文件中嵌入该非标准字体。标准字体列
表如下：
宋体
Times New Roman
Times New Roman Italic
Times New Roman Bold
Times New Roman Bold Italic
Arial
Arial Italic
Arial Bold
Arial Bold Italic
Courier New
Courier New Italic
Courier New Bold
Courier New Bold Italic
Symbol
Zapf Dingbats
参考文献（2.7.5章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。
```
- `结构化列表项`:
```text
宋体
Times New Roman
Times New Roman Italic
Times New Roman Bold
Times New Roman Bold Italic
Arial
Arial Italic
Arial Bold
Arial Bold Italic
Courier New
Courier New Italic
Courier New Bold
Courier New Bold Italic
Symbol
Zapf Dingbats
```
- `结构化单行比对串`: 宋体 | Times New Roman | Times New Roman Italic | Times New Roman Bold | Times New Roman Bold Italic | Arial | Arial Italic | Arial Bold | Arial Bold Italic | Courier New | Courier New Italic | Courier New Bold | Courier New Bold Italic | Symbol | Zapf Dingbats
- `规范化全文`: 6.26 如使用非标准字体，需嵌入在PDF文件中 PDF文件应尽量使用标准字体。如果包含非标准字体，则需在文件中嵌入该非标准字体。标准字体列 表如下： 宋体 Times New Roman Times New Roman Italic Times New Roman Bold Times New Roman Bold Italic Arial Arial Italic Arial Bold Arial Bold Italic Courier New Courier New Italic Courier New Bold Courier New Bold Italic Symbol Zapf Dingbats 参考文献（2.7.5章节、3.3章节、4.3章节、5.4章节）和外文参考资料不做此要求。 警告
