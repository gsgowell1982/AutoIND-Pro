"""
eCTD STF格式验证器

根据ICH STF Specification V2.6.1规范，验证STF文件的格式和内容是否符合标准。

验证内容:
1. STF文件命名规范 (stf-{study-id}.xml)
2. STF XML结构完整性 (study-identifier, study-document)
3. Category元素及其值的有效性
4. File-tag元素的标准符合性
5. Property元素的正确使用
6. STF版本属性验证
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Set, Dict
from dataclasses import dataclass
from pathlib import Path
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
class STFValidationContext:
    """STF验证上下文"""
    file_path: str
    module_path: str  # 所在模块路径 (如 m4/m4-2/...)
    stf_content: Optional[str] = None
    study_id: Optional[str] = None


class STFFormatValidator:
    """STF格式验证器"""

    # STF文件命名模式
    STF_FILENAME_PATTERN = re.compile(r'^stf-[a-zA-Z0-9_-]+\.xml$', re.IGNORECASE)

    # 有效的species值
    VALID_SPECIES = {
        'mouse', 'rat', 'hamster', 'other-rodent',
        'rabbit', 'dog', 'non-human-primate',
        'other-non-rodent-mammal', 'non-mammals'
    }

    # 有效的route-of-admin值
    VALID_ROUTE_OF_ADMIN = {
        'oral', 'intravenous', 'intramuscular', 'intraperitoneal',
        'subcutaneous', 'inhalation', 'topical', 'other'
    }

    # 有效的duration值 (US特定)
    VALID_DURATION = {'short', 'medium', 'long'}

    # 有效的type-of-control值
    VALID_TYPE_OF_CONTROL = {
        'placebo', 'no-treatment', 'dose-response-without-placebo',
        'active-control-without-placebo', 'external'
    }

    # 有效的file-tag name属性值 (ICH标准)
    VALID_FILE_TAGS_ICH = {
        # 非临床
        'pre-clinical-study-report',
        # 临床核心文档
        'legacy-clinical-study-report',
        'synopsis',
        'study-report-body',
        # 16.1 研究信息
        'protocol-or-amendment',
        'sample-case-report-form',
        'iec-irb-consent-form-list',
        'list-description-investigator-site',
        'signatures-investigators',
        'list-patients-with-batches',
        'randomisation-scheme',
        'audit-certificates-report',
        'statistical-methods-interim-analysis-plan',
        'inter-laboratory-standardisation-methods-quality-assurance',
        'publications-based-on-study',
        'publications-referenced-in-report',
        # 16.2 受试者数据清单
        'discontinued-patients',
        'protocol-deviations',
        'patients-excluded-from-efficacy-analysis',
        'demographic-data',
        'compliance-and-drug-concentration-data',
        'individual-efficacy-response-data',
        'adverse-event-listings',
        'listing-individual-laboratory-measurements-by-patient',
        'case-report-forms',
        # 其他
        'available-on-request',
    }

    # 有效的file-tag name属性值 (US特定)
    VALID_FILE_TAGS_US = {
        'data-tabulation-dataset',
        'data-tabulation-data-definition',
        'data-listing-dataset',
        'data-listing-data-definition',
        'analysis-dataset',
        'analysis-program',
        'analysis-data-definition',
        'annotated-crf',
        'ecg',
        'image',
        'subject-profiles',
        'safety-report',
        'antibacterial',
        'special-pathogen',
        'antiviral',
        'iss',
        'ise',
        'pm-description',
    }

    # 有效的file-tag name属性值 (JP特定)
    VALID_FILE_TAGS_JP = {
        'complete-patient-list',
        'serious-adverse-event-patient-list',
        'adverse-event-patient-list',
        'abnormal-lab-values-patient-list',
    }

    # 需要category的模块模式
    CATEGORY_REQUIRED_MODULES = {
        r'm4[/\\]m4-2[/\\]m4-2-3[/\\]m4-2-3-1[/\\]': ['species', 'route-of-admin'],  # 4.2.3.1 单次给药毒性
        r'm4[/\\]m4-2[/\\]m4-2-3[/\\]m4-2-3-2[/\\]': ['species', 'route-of-admin'],  # 4.2.3.2 重复给药毒性
        r'm4[/\\]m4-2[/\\]m4-2-3[/\\]m4-2-3-4[/\\]m4-2-3-4-1[/\\]': ['species'],  # 4.2.3.4.1 长期致癌性
        r'm5[/\\]m5-3[/\\]m5-3-5[/\\]m5-3-5-1[/\\]': ['type-of-control'],  # 5.3.5.1 对照临床研究
    }

    # 需要property元素的file-tag
    PROPERTY_REQUIRED_FILE_TAGS = {
        'case-report-forms': 'site-identifier',
        'subject-profiles': 'site-identifier',
    }

    def validate_stf_naming(self, context: STFValidationContext) -> List[ViolationDetail]:
        """
        验证STF文件命名规范

        规则: stf-{study-id}.xml
        """
        violations = []

        filename = Path(context.file_path).name

        # 检查文件名模式
        if not self.STF_FILENAME_PATTERN.match(filename):
            violations.append(ViolationDetail(
                rule_id="3.8.1",
                severity=ViolationSeverity.ERROR,
                message=f"STF文件命名不符合规范: {filename}",
                location=context.file_path,
                details=f"STF文件名必须遵循 'stf-{{study-id}}.xml' 格式，其中study-id为字母、数字、下划线或连字符的组合",
                suggestion="重命名文件为 'stf-<研究编号>.xml' 格式，例如: stf-abc123xyz789.xml"
            ))

        # 如果有study-id，检查文件名是否匹配
        if context.study_id:
            expected_filename = f"stf-{context.study_id}.xml"
            if filename.lower() != expected_filename.lower():
                violations.append(ViolationDetail(
                    rule_id="3.8.1",
                    severity=ViolationSeverity.ERROR,
                    message=f"STF文件名与study-id不匹配",
                    location=context.file_path,
                    details=f"文件名为 '{filename}'，但study-id为 '{context.study_id}'，期望文件名为 '{expected_filename}'",
                    suggestion=f"将文件重命名为 {expected_filename}"
                ))

        return violations

    def validate_stf_structure(self, context: STFValidationContext) -> List[ViolationDetail]:
        """
        验证STF XML结构完整性

        必需元素:
        - ectd:study (根元素)
        - study-identifier (包含title, study-id)
        - study-document
        """
        violations = []

        if not context.stf_content:
            violations.append(ViolationDetail(
                rule_id="3.8.2",
                severity=ViolationSeverity.CRITICAL,
                message="无法读取STF文件内容",
                location=context.file_path,
                details="STF文件为空或无法访问",
                suggestion="确保STF文件存在且可读"
            ))
            return violations

        try:
            # 解析XML
            root = ET.fromstring(context.stf_content)

            # 检查根元素
            if not root.tag.endswith('study'):
                violations.append(ViolationDetail(
                    rule_id="3.8.2",
                    severity=ViolationSeverity.CRITICAL,
                    message="STF根元素不正确",
                    location=context.file_path,
                    details=f"根元素为 '{root.tag}'，期望为 'ectd:study'",
                    suggestion="确保XML根元素为 <ectd:study>"
                ))

            # 检查DTD版本属性
            dtd_version = root.get('dtd-version')
            if not dtd_version:
                violations.append(ViolationDetail(
                    rule_id="3.8.2",
                    severity=ViolationSeverity.WARNING,
                    message="STF缺少dtd-version属性",
                    location=context.file_path,
                    details="ectd:study元素应包含dtd-version属性",
                    suggestion="添加 dtd-version='2.2' 属性到根元素"
                ))

            # 查找study-identifier元素
            study_identifier = None
            for child in root:
                if child.tag.endswith('study-identifier'):
                    study_identifier = child
                    break

            if study_identifier is None:
                violations.append(ViolationDetail(
                    rule_id="3.8.2",
                    severity=ViolationSeverity.CRITICAL,
                    message="STF缺少study-identifier元素",
                    location=context.file_path,
                    details="STF必须包含study-identifier元素",
                    suggestion="添加 <study-identifier> 元素，包含 <title>, <study-id> 和必要的 <category> 子元素"
                ))
            else:
                # 检查study-identifier的必需子元素
                title = None
                study_id = None
                for child in study_identifier:
                    if child.tag.endswith('title'):
                        title = child
                    elif child.tag.endswith('study-id'):
                        study_id = child

                if title is None:
                    violations.append(ViolationDetail(
                        rule_id="3.8.2",
                        severity=ViolationSeverity.ERROR,
                        message="study-identifier缺少title元素",
                        location=context.file_path,
                        details="study-identifier必须包含title元素（研究完整标题）",
                        suggestion="在study-identifier中添加 <title>研究标题</title>"
                    ))

                if study_id is None:
                    violations.append(ViolationDetail(
                        rule_id="3.8.2",
                        severity=ViolationSeverity.ERROR,
                        message="study-identifier缺少study-id元素",
                        location=context.file_path,
                        details="study-identifier必须包含study-id元素（申办方内部研究编号）",
                        suggestion="在study-identifier中添加 <study-id>研究编号</study-id>"
                    ))
                else:
                    # 提取study-id用于后续验证
                    context.study_id = study_id.text

            # 查找study-document元素
            study_document = None
            for child in root:
                if child.tag.endswith('study-document'):
                    study_document = child
                    break

            if study_document is None:
                violations.append(ViolationDetail(
                    rule_id="3.8.2",
                    severity=ViolationSeverity.CRITICAL,
                    message="STF缺少study-document元素",
                    location=context.file_path,
                    details="STF必须包含study-document元素（即使为空元素）",
                    suggestion="添加 <study-document/> 或 <study-document>...</study-document>"
                ))

        except ET.ParseError as e:
            violations.append(ViolationDetail(
                rule_id="3.8.2",
                severity=ViolationSeverity.CRITICAL,
                message="STF XML格式错误",
                location=context.file_path,
                details=f"无法解析XML: {str(e)}",
                suggestion="使用XML验证工具检查文件格式，确保符合XML语法规范"
            ))

        return violations

    def validate_category_elements(
        self,
        categories: List[Dict[str, str]],
        context: STFValidationContext
    ) -> List[ViolationDetail]:
        """
        验证category元素及其值的有效性

        参数:
            categories: category元素列表，每个元素为 {'name': '...', 'info-type': '...', 'value': '...'}
            context: STF验证上下文
        """
        violations = []

        # 检查模块是否需要category
        required_categories = []
        for pattern, cats in self.CATEGORY_REQUIRED_MODULES.items():
            if re.search(pattern, context.module_path, re.IGNORECASE):
                required_categories = cats
                break

        if required_categories:
            # 检查必需的category是否存在
            provided_categories = {cat['name'] for cat in categories}
            missing_categories = set(required_categories) - provided_categories

            if missing_categories:
                violations.append(ViolationDetail(
                    rule_id="3.8.3",
                    severity=ViolationSeverity.ERROR,
                    message=f"STF缺少必需的category元素",
                    location=context.file_path,
                    details=f"模块 {context.module_path} 需要category: {', '.join(required_categories)}，但缺少: {', '.join(missing_categories)}",
                    suggestion=f"在study-identifier中添加缺失的category元素"
                ))

        # 验证每个category的值
        for cat in categories:
            name = cat.get('name', '')
            info_type = cat.get('info-type', '')
            value = cat.get('value', '')

            # 检查info-type
            if info_type not in ['ich', 'us', 'jp', 'eu', 'ca']:
                violations.append(ViolationDetail(
                    rule_id="3.8.3",
                    severity=ViolationSeverity.WARNING,
                    message=f"category的info-type值不标准: {info_type}",
                    location=context.file_path,
                    details=f"category name='{name}' 的info-type='{info_type}' 不是标准区域代码",
                    suggestion="使用标准info-type值: 'ich', 'us', 'jp', 'eu', 'ca'"
                ))

            # 根据name验证value
            if name == 'species':
                if value not in self.VALID_SPECIES:
                    violations.append(ViolationDetail(
                        rule_id="3.8.3",
                        severity=ViolationSeverity.ERROR,
                        message=f"species的category值无效: {value}",
                        location=context.file_path,
                        details=f"有效的species值: {', '.join(sorted(self.VALID_SPECIES))}",
                        suggestion=f"使用标准species值或咨询监管机构"
                    ))

            elif name == 'route-of-admin':
                if value not in self.VALID_ROUTE_OF_ADMIN:
                    if value != 'other':  # 'other'需要监管机构批准
                        violations.append(ViolationDetail(
                            rule_id="3.8.3",
                            severity=ViolationSeverity.ERROR,
                            message=f"route-of-admin的category值无效: {value}",
                            location=context.file_path,
                            details=f"有效的route-of-admin值: {', '.join(sorted(self.VALID_ROUTE_OF_ADMIN))}",
                            suggestion=f"使用标准给药途径值"
                        ))
                    else:
                        violations.append(ViolationDetail(
                            rule_id="3.8.3",
                            severity=ViolationSeverity.INFO,
                            message="使用了route-of-admin='other'",
                            location=context.file_path,
                            details="使用'other'作为给药途径需要提前咨询监管机构",
                            suggestion="确认已获得监管机构对使用'other'的批准"
                        ))

            elif name == 'duration':
                if info_type == 'us':
                    if value not in self.VALID_DURATION:
                        violations.append(ViolationDetail(
                            rule_id="3.8.3",
                            severity=ViolationSeverity.ERROR,
                            message=f"duration的category值无效: {value}",
                            location=context.file_path,
                            details=f"有效的duration值 (US): {', '.join(sorted(self.VALID_DURATION))}",
                            suggestion="使用标准duration值: short, medium, long"
                        ))
                else:
                    violations.append(ViolationDetail(
                        rule_id="3.8.3",
                        severity=ViolationSeverity.WARNING,
                        message="duration是US特定的category",
                        location=context.file_path,
                        details=f"duration category的info-type应为'us'，当前为'{info_type}'",
                        suggestion="将info-type设置为'us'或删除此category"
                    ))

            elif name == 'type-of-control':
                if value not in self.VALID_TYPE_OF_CONTROL:
                    violations.append(ViolationDetail(
                        rule_id="3.8.3",
                        severity=ViolationSeverity.ERROR,
                        message=f"type-of-control的category值无效: {value}",
                        location=context.file_path,
                        details=f"有效的type-of-control值: {', '.join(sorted(self.VALID_TYPE_OF_CONTROL))}",
                        suggestion="使用标准对照类型值"
                    ))

        return violations

    def validate_file_tags(
        self,
        file_tags: List[Dict[str, str]],
        context: STFValidationContext
    ) -> List[ViolationDetail]:
        """
        验证file-tag元素的标准符合性

        参数:
            file_tags: file-tag元素列表，每个元素为 {'name': '...', 'info-type': '...'}
        """
        violations = []

        for tag in file_tags:
            name = tag.get('name', '')
            info_type = tag.get('info-type', '')

            # 检查name属性是否为空
            if not name:
                violations.append(ViolationDetail(
                    rule_id="3.8.4",
                    severity=ViolationSeverity.ERROR,
                    message="file-tag的name属性为空",
                    location=context.file_path,
                    details="每个file-tag元素必须有name属性",
                    suggestion="为file-tag添加有效的name属性值"
                ))
                continue

            # 检查info-type是否为空
            if not info_type:
                violations.append(ViolationDetail(
                    rule_id="3.8.4",
                    severity=ViolationSeverity.WARNING,
                    message=f"file-tag '{name}' 缺少info-type属性",
                    location=context.file_path,
                    details="file-tag应包含info-type属性以标识值的来源",
                    suggestion="添加info-type='ich', 'us', 'jp'等属性"
                ))
                continue

            # 根据info-type验证name值
            valid = False
            if info_type == 'ich':
                valid = name in self.VALID_FILE_TAGS_ICH
            elif info_type == 'us':
                valid = name in self.VALID_FILE_TAGS_US or name in self.VALID_FILE_TAGS_ICH
            elif info_type == 'jp':
                valid = name in self.VALID_FILE_TAGS_JP or name in self.VALID_FILE_TAGS_ICH
            else:
                # 其他区域代码，检查是否在ICH标准中
                valid = name in self.VALID_FILE_TAGS_ICH

            if not valid:
                violations.append(ViolationDetail(
                    rule_id="3.8.4",
                    severity=ViolationSeverity.ERROR,
                    message=f"file-tag name值不在标准列表中: {name}",
                    location=context.file_path,
                    details=f"file-tag name='{name}' (info-type='{info_type}') 不是ICH STF规范中定义的标准值",
                    suggestion="使用标准file-tag值，参考ICH STF Specification V2.6.1文档"
                ))

        return violations

    def validate_property_elements(
        self,
        file_tag_name: str,
        properties: List[Dict[str, str]],
        context: STFValidationContext
    ) -> List[ViolationDetail]:
        """
        验证property元素的正确使用

        参数:
            file_tag_name: 关联的file-tag name
            properties: property元素列表，每个元素为 {'name': '...', 'info-type': '...', 'value': '...'}
        """
        violations = []

        # 检查是否需要property元素
        required_property = self.PROPERTY_REQUIRED_FILE_TAGS.get(file_tag_name)

        if required_property:
            # 检查是否提供了必需的property
            provided_properties = {prop.get('name', '') for prop in properties}

            if required_property not in provided_properties:
                violations.append(ViolationDetail(
                    rule_id="3.8.5",
                    severity=ViolationSeverity.ERROR,
                    message=f"file-tag '{file_tag_name}' 缺少必需的property元素",
                    location=context.file_path,
                    details=f"使用file-tag='{file_tag_name}'时必须提供property name='{required_property}'",
                    suggestion=f"添加 <property name='{required_property}' info-type='us'>研究中心标识</property>"
                ))

        # 验证每个property的格式
        for prop in properties:
            name = prop.get('name', '')
            info_type = prop.get('info-type', '')
            value = prop.get('value', '')

            if name == 'site-identifier':
                if info_type != 'us':
                    violations.append(ViolationDetail(
                        rule_id="3.8.5",
                        severity=ViolationSeverity.WARNING,
                        message="site-identifier的info-type应为'us'",
                        location=context.file_path,
                        details=f"property name='site-identifier' 的info-type为'{info_type}'，通常应为'us'",
                        suggestion="将info-type设置为'us'"
                    ))

                if not value:
                    violations.append(ViolationDetail(
                        rule_id="3.8.5",
                        severity=ViolationSeverity.ERROR,
                        message="site-identifier的值为空",
                        location=context.file_path,
                        details="property name='site-identifier' 必须包含研究中心的标识值",
                        suggestion="提供有效的研究中心标识符"
                    ))

        return violations

    def validate_stf_version_attribute(self, leaf_element: dict, context: STFValidationContext) -> List[ViolationDetail]:
        """
        验证index.xml中STF leaf元素的version属性

        参数:
            leaf_element: leaf元素字典，包含 'version', 'operation' 等属性
        """
        violations = []

        version = leaf_element.get('version', '')

        if not version:
            violations.append(ViolationDetail(
                rule_id="3.8.6",
                severity=ViolationSeverity.ERROR,
                message="STF的leaf元素缺少version属性",
                location=context.file_path,
                details="在index.xml中，引用STF的leaf元素必须包含version属性",
                suggestion="添加version属性，例如: version='STF version 2.2'"
            ))
        elif not version.startswith('STF version') and not version.startswith('stf version'):
            violations.append(ViolationDetail(
                rule_id="3.8.6",
                severity=ViolationSeverity.WARNING,
                message="STF的version属性格式不标准",
                location=context.file_path,
                details=f"version属性值为 '{version}'，标准格式应为 'STF version X.X'",
                suggestion="使用标准格式: 'STF version 2.2'"
            ))

        return violations
