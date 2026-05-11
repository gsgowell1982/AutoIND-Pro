from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import html
import json
from pathlib import Path
import re
from typing import Any

import fitz

from parsers.parser_registry import parse_file


REGULATION_LIBRARY_VERSION = "regulation-library-v1"
REGULATION_RULE_DRAFT_VERSION = "regulation-rule-draft-v1"
REGULATION_REQUIREMENT_MATRIX_VERSION = "regulation-requirement-matrix-v1"
REGULATION_GLOSSARY_VERSION = "regulation-glossary-v1"
REGULATION_REFERENCE_MANIFEST_VERSION = "regulation-reference-manifest-v1"
REGULATION_COVERAGE_REPORT_VERSION = "regulation-coverage-report-v1"
REGULATION_CAPABILITY_CROSSWALK_VERSION = "regulation-capability-crosswalk-v1"
_ECTD_ATTACHMENT_12_BUNDLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "regulations"
    / "normalized"
    / "cn_ectd_attachment_1_2.controlled_vocabulary_bundle.json"
)
_REGULATIONS_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "data" / "regulations"

_CHAPTER_PATTERN = re.compile(r"^第([一二三四五六七八九十百零〇0-9]+)章\s*(.+)$")
_ARTICLE_PATTERN = re.compile(r"^第([一二三四五六七八九十百零〇0-9]+)条\s*(.*)$")
_SECTION_PATTERN = re.compile(r"^(?P<section_no>[一二三四五六七八九十]+)、(?P<title>.+)$")
_PAREN_ITEM_PATTERN = re.compile(r"^（(?P<item_no>[一二三四五六七八九十]+)）(?P<body>.+)$")
_CLASS_ITEM_PATTERN = re.compile(r"^(?P<class_no>\d(?:\.\d+)?)类[:：](?P<body>.+)$")
_NUMBERED_HEADING_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)*)(?:\.)?\s+(?P<title>\S.*)$")
_NUMBERED_TOC_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)*)(?:\.)?\s+(?P<title>.+?)\s+\.{2,}\s*\d+\s*$")
_DOT_LEADER_PATTERN = re.compile(r"\.{4,}")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_SCRIPT_PATTERN = re.compile(r"<script.*?</script>", re.IGNORECASE | re.DOTALL)
_HTML_STYLE_PATTERN = re.compile(r"<style.*?</style>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_PATTERN = re.compile(r"\s+")

_MANDATORY_HARD_KEYWORDS = (
    "应当提交",
    "应当报送",
    "应当申请",
    "应当使用",
    "应当遵守",
    "不得",
    "保证记录和数据真实、准确、完整和可追溯",
)
_DIRECT_MATERIAL_KEYWORDS = (
    "申请药品注册",
    "申请再注册",
    "提交",
    "报送",
    "药品注册标准",
    "国家药品标准品",
    "对照品",
    "样品",
    "参比制剂",
    "研究数据",
    "记录和数据真实、准确、完整和可追溯",
    "临床试验批准证明文件",
    "变更申请",
    "备案",
    "报告",
)
_PARTIAL_MATERIAL_KEYWORDS = (
    "临床试验机构",
    "申办者",
    "质量保证体系",
    "质量管理",
    "追溯体系",
    "药物警戒",
    "药品生产质量管理规范",
    "药品经营质量管理规范",
    "许可证",
    "资格证书",
    "药品注册证书",
    "批准证书",
    "审评审批",
)
_SOFT_REVIEW_KEYWORDS = (
    "科学选择",
    "综合评价",
    "体现中药的特点",
    "资源的可持续利用",
    "持续提升药品质量水平",
    "评估",
)

_REGULATION_METADATA_OVERRIDES: dict[str, dict[str, str]] = {
    "中华人民共和国药品管理法实施条例": {
        "regulation_id": "cn_drug_administration_law_implementation_regulation",
        "title": "中华人民共和国药品管理法实施条例",
        "issuer": "中华人民共和国国务院",
        "jurisdiction": "CN",
        "domain": "drug_regulation",
        "version_label": "2026-01-16-828",
    },
    "药品注册分类及申报资料要求": {
        "regulation_id": "cn_drug_registration_classification_and_dossier_requirements",
        "title": "药品注册分类及申报资料要求",
        "issuer": "国家药品监督管理局",
        "jurisdiction": "CN",
        "domain": "drug_registration",
        "version_label": "unknown",
    },
    "eCTD技术规范": {
        "regulation_id": "cn_ectd_technical_specification",
        "title": "eCTD技术规范",
        "issuer": "国家药品监督管理局",
        "jurisdiction": "CN",
        "domain": "ectd_specification",
        "version_label": "2021-09-v1.0",
    },
    "eCTD楠岃瘉鏍囧噯": {
        "regulation_id": "cn_ectd_validation_standard",
        "title": "eCTD楠岃瘉鏍囧噯",
        "issuer": "鍥藉鑽搧鐩戠潱绠＄悊灞€",
        "jurisdiction": "CN",
        "domain": "ectd_validation_standard",
        "version_label": "2021-09-v1.0",
    },
}

_ECTD_TECHNICAL_SPEC_GLOSSARY_TERMS: tuple[tuple[str, str], ...] = (
    (
        "电子通用技术文档（eCTD）",
        "电子通用技术文档是用于药品注册申报和审评的电子注册文档。通过可扩展标记语言（XML）将符合 CTD 规范的药品申报资料以电子化形式进行组织、传输和呈现。",
    ),
    (
        "申请",
        "申请是指为了一个特殊的监管目的（如临床试验申请）来整理和提交的申报资料的集合。",
    ),
    (
        "注册行为",
        "注册行为是针对某一特定注册目的从首次提交到获得批准的所有序列的申报资料集合，可以包含一个序列或多个序列。同一个注册行为中的多个序列可以是连续的序列，也可以是不连续的序列。",
    ),
    (
        "序列",
        "序列是在某一注册行为中单次提交的申报资料的集合。",
    ),
    (
        "申请编号",
        "申请编号是一个申请在其全生命周期内的唯一识别编号，由监管机构分配给申请人。",
    ),
    (
        "原始编号",
        "原始编号是对一个进入注册审批程序的药品所给予的基本的和永久的资料代号，是用于标识申请人、活性成分和剂型的唯一识别码，由监管机构分配。",
    ),
    (
        "相关序列",
        "一个注册行为中首次提交的序列被称为该注册行为中提交的所有序列的相关序列。",
    ),
    (
        "序列号",
        "序列号是申请中唯一的 4 位数字的字符串，是用于区分同一申请中不同提交序列的唯一标识。",
    ),
    (
        "叶元素（leaf element）",
        "叶元素是 eCTD 骨架文件的一部分，是在序列中提交的单个文件的引用地址、显示名称、校验和及生命周期操作等信息的集合。",
    ),
    (
        "信封信息",
        "信封信息是 eCTD 区域骨架文件的一部分，给电子资料管理系统提供处理和组织申报资料时使用的元数据。",
    ),
    (
        "受控词汇",
        "受控词汇是对特定概念规定术语的限定列表。",
    ),
    (
        "基线",
        "基线指申请人将已以纸质递交获批上市许可的药品从纸质递交格式转换为 eCTD 提交的注册行为。",
    ),
    (
        "扩展节点（node extension）",
        "扩展节点为申请人提供了自定义目录元素的途径，用以扩展技术规范中既定的 eCTD 目录元素结构，实现将多个叶元素在自定义目录元素下组合显示功能。",
    ),
    (
        "研究标签文件（Study Tagging Files, STF）",
        "研究标签文件用以提供在 eCTD 骨架文件中没有包含的关于研究主题和研究报告的信息，例如研究全称，研究 ID，研究使用的种属，给药途径，研究时长，对照类型等。",
    ),
    (
        "MD5",
        "MD5 消息摘要算法（MD5 Message Digest Algorithm），一种被广泛使用的密码散列函数，用于产生一个文件对应的数字指纹，即校验和。",
    ),
    (
        "校验和（checksum）",
        "使用 MD5 消息摘要算法产生的文件校验和，用以确保信息传输的完整性和一致性。",
    ),
    (
        "DTD",
        "文档类型定义（Document Type Definition）是一套为了进行程序间的数据交换而建立的关于标记符的语法规则，用于保证 eCTD 骨架文件的合法性，如元素和属性使用是否正确等。",
    ),
    (
        "验证",
        "验证指申请人和监管机构根据公开和统一的验证标准，对 eCTD 申报资料进行检查校验的过程。",
    ),
)

_ECTD_TECHNICAL_SPEC_SUPPLEMENTAL_GLOSSARY_TERMS: tuple[dict[str, str], ...] = (
    {
        "term": "OCR",
        "definition": "光学字符识别（Optical Character Recognition）指对扫描的PDF文件进行光学识别，使扫描文件中的文本可以被检索查找。",
        "source_clause_id": "cn_ectd_implementation_guide:sec_11",
        "source_heading": "11. 术语表",
        "source_filename": bytes("eCTD\\u5b9e\\u65bd\\u6307\\u5357.pdf", "ascii").decode("unicode_escape"),
        "citation_anchor": "cn_ectd_implementation_guide#sec_11_term_ocr",
        "source_note": "Supplemental glossary term from eCTD实施指南.pdf pages 38-39; the term row spans a page boundary.",
    },
)

_ECTD_TECHNICAL_SPEC_REFERENCE_MANIFEST: tuple[dict[str, str], ...] = (
    {
        "title": "ICH eCTD Specification and Related files",
        "authority": "ICH",
        "reference_type": "specification_bundle",
        "normative_role": "external_normative_dependency",
    },
    {
        "title": "ICH Electronic Common Technical Document Specification V3.2.2",
        "authority": "ICH",
        "reference_type": "technical_specification",
        "normative_role": "external_normative_dependency",
    },
    {
        "title": "ICH The eCTD Backbone File Specification for Study Tagging Files V2.6.1",
        "authority": "ICH",
        "reference_type": "stf_backbone_specification",
        "normative_role": "external_normative_dependency",
    },
    {
        "title": "ICH Specification for Submission Formats for eCTD V1.2",
        "authority": "ICH",
        "reference_type": "submission_format_specification",
        "normative_role": "external_normative_dependency",
    },
    {
        "title": "ICH eCTD IWG Question and Answer and Specification Change Request Document V1.31",
        "authority": "ICH",
        "reference_type": "qa_change_request_guidance",
        "normative_role": "supporting_guidance",
    },
    {
        "title": "ICH E3 Structure and Content of Clinical Study Reports",
        "authority": "ICH",
        "reference_type": "clinical_study_report_guidance",
        "normative_role": "supporting_guidance",
    },
    {
        "title": "《M4 模块一行政文件和药品信息》",
        "authority": "NMPA/CTD",
        "reference_type": "ctd_module_guidance",
        "normative_role": "external_normative_dependency",
    },
    {
        "title": "《药物临床试验数据递交指导原则（试行）》",
        "authority": "NMPA",
        "reference_type": "clinical_trial_data_guidance",
        "normative_role": "supporting_guidance",
    },
)

_ECTD_TECHNICAL_SPEC_SUPPORTING_ARTIFACT_COVERAGE: tuple[dict[str, Any], ...] = (
    {
        "attachment_id": "cn_ectd_attachment_1_1",
        "artifact_filename": "cn_ectd_attachment_1_1.region_schema_bundle.json",
        "coverage_status": "covered",
        "relevant_clause_ids": [
            "cn_ectd_technical_specification:sec_1_3",
            "cn_ectd_technical_specification:sec_4_2",
            "cn_ectd_technical_specification:sec_4_3",
            "cn_ectd_technical_specification:sec_4_4",
        ],
    },
    {
        "attachment_id": "cn_ectd_attachment_1_2",
        "artifact_filename": "cn_ectd_attachment_1_2.controlled_vocabulary_bundle.json",
        "coverage_status": "covered",
        "relevant_clause_ids": [
            "cn_ectd_technical_specification:sec_1_3",
            "cn_ectd_technical_specification:sec_2_1_2",
            "cn_ectd_technical_specification:sec_2_1_3",
            "cn_ectd_technical_specification:sec_2_2_1",
            "cn_ectd_technical_specification:sec_2_3_2",
            "cn_ectd_technical_specification:sec_2_4",
            "cn_ectd_technical_specification:sec_4_3",
        ],
    },
    {
        "attachment_id": "cn_ectd_attachment_1_3",
        "artifact_filename": "cn_ectd_attachment_1_3.region_style_bundle.json",
        "coverage_status": "covered",
        "relevant_clause_ids": [
            "cn_ectd_technical_specification:sec_1_3",
            "cn_ectd_technical_specification:sec_4_2",
            "cn_ectd_technical_specification:sec_4_3",
            "cn_ectd_technical_specification:sec_4_4",
        ],
    },
    {
        "attachment_id": "cn_ectd_attachment_1_4",
        "artifact_filename": "cn_ectd_attachment_1_4.module1_structure_bundle.json",
        "coverage_status": "covered",
        "relevant_clause_ids": [
            "cn_ectd_technical_specification:sec_1_3",
            "cn_ectd_technical_specification:sec_3_1",
            "cn_ectd_technical_specification:sec_4_1",
            "cn_ectd_technical_specification:sec_4_4",
        ],
    },
    {
        "attachment_id": "cn_ectd_attachment_2_6",
        "artifact_filename": "cn_ectd_attachment_2_6.stf_valid_values_bundle.json",
        "coverage_status": "covered",
        "relevant_clause_ids": [
            "cn_ectd_technical_specification:sec_1_3",
            "cn_ectd_technical_specification:sec_3_8",
        ],
    },
)

_ECTD_VALIDATION_STANDARD_TITLE_OVERRIDES: dict[str, str] = {
    "3.7": "叶元素：新建、替换或增补的叶元素，必须有“文件引用（xlink:href）”值",
    "3.8": "叶元素：删除的叶元素不能包含“文件引用（xlink:href）”值",
    "3.9": "叶元素：对替换、删除和增补的叶元素，必须有对应的文件“操作（operation）”属性值为“替换（replace）”、“删除（delete）”或“增补（append）”的所有叶元素，对应的“被修改文件对象（modified-file）”必须有值。",
    "4.1.8": "叶元素：新建、替换或增补的叶元素，必须有“文件引用（xlink:href）”值",
    "4.1.9": "叶元素：删除的叶元素不能包含“文件引用（xlink:href）”值",
    "4.1.10": "叶元素：对替换、删除和增补的叶元素，必须有对应的文件“操作（operation）”属性值为“替换（replace）”、“删除（delete）”或“增补（append）”的所有叶元素，对应的“被修改文件对象（modified-file）”必须有值。",
    "6.22": "PDF应该设置启用“快速Web访问（Fast Web Access）”",
}

_ECTD_VALIDATION_STANDARD_FULL_TITLE_ONLY_ARTICLES = {
    "3.7",
    "3.8",
    "3.9",
    "4.1.8",
    "4.1.9",
    "4.1.10",
}

_ECTD_VALIDATION_STANDARD_INLINE_TITLE_SPLIT_ARTICLES = {"6.22"}

_ECTD_VALIDATION_STANDARD_TAIL_STOP_MARKERS = {
    "必须遵守的关键验证标准",
    "建议遵守的验证标准",
    "用于收集信息的验证标准",
    "说明:",
}

_ECTD_TECHNICAL_SPEC_CLAUSE_COVERAGE_OVERRIDES: dict[str, dict[str, Any]] = {
    "sec_1_1": {
        "coverage_status": "citation_only_recorded",
        "implemented_rule_ids": [],
        "coverage_note": "Purpose/background clause. Recorded as context rather than an executable dossier rule.",
    },
    "sec_1_2": {
        "coverage_status": "citation_only_recorded",
        "implemented_rule_ids": [],
        "coverage_note": "Scope/application clause. Recorded as context and external dependency guidance.",
    },
    "sec_1_3": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["HR-ECTD-078", "HR-ECTD-079"],
        "coverage_note": "ICH index.xml DTD reference and DTD validity boundaries are executable through validation-standard rules, while the broader supporting-file chapter also covers schema, stylesheet, STF, and controlled-vocabulary dependencies.",
    },
    "sec_2_1": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["SR-ECTD-006"],
        "coverage_note": "Aggregate application-information coverage exists, but original-number exactness is still outside the executable subset.",
    },
    "sec_2_1_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-002"],
        "coverage_note": "Application-number format is executable.",
    },
    "sec_2_1_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-011", "HR-ECTD-037"],
        "coverage_note": "Application-type controlled-vocabulary validity is executable against Attachment 1-2.",
    },
    "sec_2_1_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-011", "HR-ECTD-038"],
        "coverage_note": "Product-type controlled-vocabulary validity is executable against Attachment 1-2.",
    },
    "sec_2_1_4": {
        "coverage_status": "deferred",
        "implemented_rule_ids": [],
        "coverage_note": "Original-number/product-number exactness is not yet modeled as a deterministic runtime rule.",
    },
    "sec_2_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-005"],
        "coverage_note": "Aggregate regulatory-activity information coverage exists.",
    },
    "sec_2_2_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-012", "HR-ECTD-011", "HR-ECTD-039"],
        "coverage_note": "Regulatory-activity-type presence and controlled-vocabulary exactness are executable.",
    },
    "sec_2_2_2": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["SR-ECTD-003"],
        "coverage_note": "Related-sequence reference integrity is covered at a bounded soft-rule level, not as a full history-hard exactness rule.",
    },
    "sec_2_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-004"],
        "coverage_note": "Aggregate sequence-information coverage exists.",
    },
    "sec_2_3_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-001"],
        "coverage_note": "Sequence-number progression and numbering constraints are executable.",
    },
    "sec_2_3_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-013", "HR-ECTD-011", "HR-ECTD-040"],
        "coverage_note": "Sequence-type presence and controlled-vocabulary exactness are executable.",
    },
    "sec_2_3_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-002"],
        "coverage_note": "Sequence-description length boundary is executable.",
    },
    "sec_2_3_4": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-014"],
        "coverage_note": "Sequence-contact presence is executable.",
    },
    "sec_2_4": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-017", "HR-ECTD-018"],
        "coverage_note": "Application/activity/sequence compatibility is executable against Attachment 1-2 support files.",
    },
    "sec_3_1": {
        "coverage_status": "deferred",
        "implemented_rule_ids": [],
        "coverage_note": "This clause points forward to chapter 4 and M4 module-1 content rather than defining an independent executable boundary.",
    },
    "sec_3_2": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["SR-ECTD-007", "SR-ECTD-026"],
        "coverage_note": "Bounded 3.2.R title/path structure is executable from both technical-spec and validation-standard-facing rule surfaces, but not the entire reviewer-facing semantics of regional information usage.",
    },
    "sec_3_3": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": [
            "HR-ECTD-019",
            "HR-ECTD-005",
            "HR-ECTD-020",
            "HR-ECTD-029",
            "HR-ECTD-021",
            "HR-ECTD-022",
            "SR-ECTD-011",
            "HR-ECTD-006",
            "HR-ECTD-027",
        ],
        "coverage_note": "Most executable packaging/file-system boundaries are covered, while softer reuse optimization guidance remains advisory.",
    },
    "sec_3_3_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-019"],
        "coverage_note": "Allowed content-file format set is executable.",
    },
    "sec_3_3_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-005", "HR-ECTD-020", "HR-ECTD-029"],
        "coverage_note": "Naming, path-character, path-length, and XML-covered directory boundaries are executable.",
    },
    "sec_3_3_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-021", "HR-ECTD-022"],
        "coverage_note": "No-empty-directory and no-placeholder-document boundaries are executable.",
    },
    "sec_3_3_4": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-011", "HR-ECTD-006", "HR-ECTD-027", "HR-ECTD-063", "HR-ECTD-064"],
        "coverage_note": "Same-application reuse legality, cross-application prohibition, and replace/append checksum difference against modified-file targets are covered at practical executable boundaries.",
    },
    "sec_3_4": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["SR-ECTD-001", "SR-ECTD-012", "SR-ECTD-013", "SR-ECTD-014", "HR-ECTD-028"],
        "coverage_note": "Navigation and hyperlink-integrity subsets are covered, but PDF typography/layout details remain outside the deterministic subset.",
    },
    "sec_3_5": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-024", "HR-ECTD-023", "HR-ECTD-026"],
        "coverage_note": "Chinese/foreign sibling structure, xml:lang classification, and replace-language consistency are executable.",
    },
    "sec_3_5_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-023"],
        "coverage_note": "xml:lang classification boundary is executable.",
    },
    "sec_3_5_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-026"],
        "coverage_note": "Replace-language class consistency is executable when a unique prior-sequence match exists.",
    },
    "sec_3_6": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["HR-ECTD-075", "HR-ECTD-076", "HR-ECTD-077", "SR-ECTD-027"],
        "coverage_note": "Validation-standard-facing ICH backbone attribute required/non-empty checks and attribute edge-whitespace warnings are executable, while broader metadata update lifecycle coupling remains prerequisite-dependent.",
    },
    "sec_3_7": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-008", "HR-ECTD-074"],
        "coverage_note": "Node-extension scope boundary is executable; validation-standard-facing coverage hard-fails objective misuse while preserving prerequisite review for missing product-type evidence.",
    },
    "sec_3_8": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["SR-ECTD-010"],
        "coverage_note": "STF-required-zone structure is covered, but full STF tagging semantics remain dependent on the external STF specification.",
    },
    "sec_3_9": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["HR-ECTD-025", "SR-ECTD-009"],
        "coverage_note": "Allowed lifecycle-operation values and non-STF append warning are covered, while broader lifecycle semantics remain bounded.",
    },
    "sec_3_10": {
        "coverage_status": "deferred",
        "implemented_rule_ids": [],
        "coverage_note": "Electronic-signature legality is an external legal/operational boundary rather than a dossier-structure rule.",
    },
    "sec_3_11": {
        "coverage_status": "deferred",
        "implemented_rule_ids": [],
        "coverage_note": "Physical media and password-protection constraints are not reliably decidable from the unpacked dossier artifact alone.",
    },
    "sec_4_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-015", "HR-ECTD-016", "HR-ECTD-030", "HR-ECTD-031", "HR-ECTD-032"],
        "coverage_note": "Core module-1 package/backbone composition is executable.",
    },
    "sec_4_2": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["HR-ECTD-030", "HR-ECTD-036"],
        "coverage_note": "Root element and schema-version exactness are covered; namespace declaration text was intentionally left outside hard enforcement.",
    },
    "sec_4_3": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["HR-ECTD-004", "HR-ECTD-031", "HR-ECTD-035", "HR-ECTD-011", "HR-ECTD-037", "HR-ECTD-038", "HR-ECTD-039", "HR-ECTD-040", "HR-ECTD-115", "HR-ECTD-116"],
        "coverage_note": "Envelope structure, controlled-vocabulary exactness, and bounded cross-sequence immutability checks are covered; display semantics and missing-history prerequisites remain outside hard enforcement.",
    },
    "sec_4_4": {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": ["HR-ECTD-003", "HR-ECTD-006", "HR-ECTD-007", "HR-ECTD-008", "HR-ECTD-009", "HR-ECTD-010", "HR-ECTD-029", "HR-ECTD-032", "HR-ECTD-033", "HR-ECTD-034", "HR-ECTD-050", "HR-ECTD-051", "HR-ECTD-052", "HR-ECTD-061", "HR-ECTD-062", "HR-ECTD-063", "HR-ECTD-064", "HR-ECTD-065", "HR-ECTD-066", "HR-ECTD-067", "HR-ECTD-068", "HR-ECTD-069", "HR-ECTD-070", "HR-ECTD-071", "HR-ECTD-072", "HR-ECTD-073", "SR-ECTD-025"],
        "coverage_note": "Backbone integrity, module-1 leaf checksum exactness, replace/append checksum difference against modified-file targets, index.xml lifecycle href/modified-file/delete/single-operation/initial-sequence/target-existence/path exactness, index.xml leaf-title non-empty/continuity/edge-whitespace exactness, regional cover-letter/application-form operation exactness, module-1 hierarchy exactness, index.xml m-element leaf-descendant presence, and cn-content element leaf-descendant presence are covered, while qualitative display semantics remain outside deterministic enforcement.",
    },
}

_ECTD_VALIDATION_STANDARD_CHAPTER2_CLAUSE_COVERAGE_OVERRIDES: dict[str, dict[str, Any]] = {
    "sec_2_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-021"],
        "coverage_note": "No-empty-directory boundary is executable.",
    },
    "sec_2_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-015"],
        "coverage_note": "File-size warning boundary is executable with the validation-standard 500MB / 4GB thresholds.",
    },
    "sec_2_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-010"],
        "coverage_note": "Actual package files must be declared by the XML backbone.",
    },
    "sec_2_4": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-019"],
        "coverage_note": "Allowed eCTD content-file format set is executable.",
    },
    "sec_2_5": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-005"],
        "coverage_note": "File/folder naming-character and path-constraint boundary is executable.",
    },
    "sec_2_6": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-015", "HR-ECTD-046"],
        "coverage_note": "m1-folder existence and the no-direct-file-under-m1 boundary are executable from bounded package-root structure checks.",
    },
    "sec_2_7": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-015", "HR-ECTD-017", "HR-ECTD-047"],
        "coverage_note": "util structure, support-file presence, and authoritative checksum exactness are executable from bounded package-root file checks plus STF leaf stylesheet-reference detection.",
    },
    "sec_2_8": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-015"],
        "coverage_note": "Unexpected root-level package entries are now rejected by the package-structure rule.",
    },
    "sec_2_9": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-015"],
        "coverage_note": "Sequence-folder 4-digit naming boundary is executable at package-structure scope.",
    },
    "sec_2_10": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-001"],
        "coverage_note": "Sequence-number format plus known-history continuity boundary is executable.",
    },
    "sec_3_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-015"],
        "coverage_note": "index.xml required-presence boundary is executable from bounded package-structure checks.",
    },
    "sec_3_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-078"],
        "coverage_note": "index.xml DOCTYPE SYSTEM reference is checked for the bounded util/dtd/ich-ectd-3-2.dtd target and local file availability.",
    },
    "sec_3_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-079"],
        "coverage_note": "index.xml well-formedness and local ich-ectd-3-2.dtd validation are executable when the DTD reference resolves; unresolved DTD validation prerequisites remain explicit guidance rather than false pass/fail.",
    },
    "sec_3_4": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-006"],
        "coverage_note": "XML leaf xlink:href target-existence boundary is executable from bounded sequence-package file-resolution checks.",
    },
    "sec_3_5": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-065"],
        "coverage_note": "index.xml lifecycle target files must not correspond to multiple operations within one sequence, which is executable from all-leaf lifecycle metadata and sequence-root path normalization.",
    },
    "sec_3_6": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-063"],
        "coverage_note": "index.xml replace/append leaf checksum values must differ from the resolved modified-file leaf checksum when a prior target leaf is available.",
    },
    "sec_3_7": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-066"],
        "coverage_note": "index.xml new, replace, and append leaf href-value exactness is executable from all-leaf lifecycle metadata extracted from index.xml.",
    },
    "sec_3_8": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-067"],
        "coverage_note": "index.xml delete leaf href-value prohibition is executable from all-leaf lifecycle metadata, including href-less leaf records.",
    },
    "sec_3_9": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-025", "SR-ECTD-009", "HR-ECTD-068"],
        "coverage_note": "Allowed lifecycle-operation values, append scope warning, and index.xml replace/delete/append modified-file value exactness are covered from lifecycle metadata.",
    },
    "sec_3_10": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-071"],
        "coverage_note": "Initial sequence index.xml leaf operation=new exactness is executable from sequence-package context plus index lifecycle metadata.",
    },
    "sec_3_11": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-072"],
        "coverage_note": "index.xml modified-file target existence is executable when the referenced sequence package is available; unavailable prior sequence evidence remains prerequisite-dependent.",
    },
    "sec_3_12": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-073"],
        "coverage_note": "index.xml href and modified-file references are checked for relative path usage and forward slashes from preserved lifecycle path strings.",
    },
    "sec_3_16": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-074"],
        "coverage_note": "Node-extension usage is executable for objective violations: non-biologic use, use outside 3.2.R, and biologic 3.2.R sections lacking required node-extension; missing product-type remains prerequisite guidance instead of false hard failure.",
    },
    "sec_3_17": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-026"],
        "coverage_note": "3.2.R node-extension title/path naming is executable as the validation-standard warning counterpart of the technical-spec 3.2 extension-title rule.",
    },
    "sec_3_13": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-003"],
        "coverage_note": "checksum-type must be MD5/md5 boundary is executable from XML leaf metadata.",
    },
    "sec_3_14": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-049"],
        "coverage_note": "index.xml leaf-declared checksum values must match actual target-file MD5 values, which is executable from bounded sequence-package file checks.",
    },
    "sec_3_15": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-048"],
        "coverage_note": "index.xml backbone-file MD5 must match the declared value in index-md5.txt, which is executable from bounded package-root file checks.",
    },
    "sec_3_18": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-069"],
        "coverage_note": "index.xml leaf title presence and non-empty value checks are executable from preserved leaf-title metadata.",
    },
    "sec_3_19": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-070"],
        "coverage_note": "index.xml delete leaf titles must match the prior modified-file leaf title when the modified-file target resolves to known prior lifecycle metadata.",
    },
    "sec_3_20": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-025"],
        "coverage_note": "index.xml leaf title leading/trailing whitespace is executable as a warning from preserved raw title text.",
    },
    "sec_3_21": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-062"],
        "coverage_note": "index.xml elements whose names begin with m[number] must have at least one descendant leaf, which is executable from index.xml structural metadata.",
    },
    "sec_3_22": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-086"],
        "coverage_note": "index.xml must contain the m1 administrative-information-and-prescribing-information element, which is executable from parsed XML element structure.",
    },
    "sec_3_23": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-028"],
        "coverage_note": "index.xml append operations outside an explicit STF scope are surfaced as validation-standard warnings from preserved lifecycle and ancestor-context metadata; explanation-letter adequacy remains reviewer-facing context.",
    },
    "sec_3_24": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-029"],
        "coverage_note": "index.xml replace/delete/append modified-file pairs are reviewed as a warning-level relocation rule: clear CTD parent/path or eCTD position-attribute mismatches are surfaced when target leaf context resolves; missing or ambiguous target context becomes prerequisite/human-review guidance.",
    },
    "sec_3_25": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-075"],
        "coverage_note": "index.xml indication attribute presence/non-empty exactness is executable for 2.7.3 and 5.3.5 section elements from preserved XML element and attribute records.",
    },
    "sec_3_26": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-076"],
        "coverage_note": "index.xml manufacturer attribute presence/non-empty exactness is executable for 2.3.S and 3.2.S section elements from preserved XML element and attribute records.",
    },
    "sec_3_27": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-077"],
        "coverage_note": "index.xml substance attribute presence/non-empty exactness is executable for 2.3.S and 3.2.S section elements from preserved XML element and attribute records.",
    },
    "sec_3_28": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-027"],
        "coverage_note": "index.xml attribute leading/trailing whitespace is executable as a warning from preserved raw XML attribute values.",
    },
    "sec_3_29": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-087"],
        "coverage_note": "index.xml leaf href and modified-file content references are checked against resolvable sequence/application context so cross-application references and later-sequence references fail; unresolved target context remains prerequisite-dependent.",
    },
    "sec_3_30": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-080"],
        "coverage_note": "index.xml append operations are checked against available prior lifecycle graph evidence so a leaf already replaced by another leaf cannot be appended; incomplete historical sequences remain explicit prerequisites.",
    },
    "sec_3_31": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-081"],
        "coverage_note": "index.xml delete operations are checked against available prior lifecycle graph evidence so a leaf already replaced by another leaf cannot be deleted; incomplete historical sequences remain explicit prerequisites.",
    },
    "sec_3_32": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-082"],
        "coverage_note": "index.xml replace operations are checked against available prior lifecycle graph evidence so a leaf already replaced by another leaf cannot be replaced a second time; incomplete historical sequences remain explicit prerequisites.",
    },
    "sec_3_33": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-083"],
        "coverage_note": "index.xml replace/delete/append operations are checked against available prior lifecycle graph evidence so an already deleted leaf cannot be operated on again; incomplete historical sequences remain explicit prerequisites.",
    },
    "sec_3_34": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-084"],
        "coverage_note": "Non-STF index.xml append operations are checked against available prior lifecycle graph evidence so they cannot target a leaf whose operation is append; STF-scoped append remains excluded from this rule.",
    },
    "sec_3_35": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-085"],
        "coverage_note": "STF-scoped index.xml append operations are checked against available prior lifecycle graph evidence so append targets must be the latest known lifecycle leaf; missing latest-version evidence remains prerequisite guidance.",
    },
    "sec_3_36": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-026"],
        "coverage_note": "Replace-operation language consistency is executable through the technical-spec lifecycle language rule when a unique prior-sequence leaf match is available.",
    },
    "sec_4_1_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-015"],
        "coverage_note": "cn-regional.xml required-presence boundary is executable from bounded package-structure checks.",
    },
    "sec_4_1_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-112"],
        "coverage_note": "cn-regional.xml schema-location references are checked from parser-preserved XML instance evidence: local schema references must resolve and point under util; absent schema-location evidence remains an explicit na/prerequisite boundary rather than an assumed pass/fail.",
    },
    "sec_4_1_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-113"],
        "coverage_note": "cn-regional.xml validity is checked from parser-preserved XML schema validation diagnostics: local schema compilation and validation must succeed for pass, validation errors fail, and missing schema/import prerequisites remain explicit na rather than assumed pass/fail.",
    },
    "sec_4_1_4": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-114"],
        "coverage_note": "Current-sequence cn-regional.xml schema-version ordering is checked only when prior sequence packages in the same application expose comparable numeric schema-version evidence; missing prior history, missing schema-version, or non-numeric versions return na/prerequisite guidance rather than assumed pass/fail.",
    },
    "sec_4_1_5": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-006"],
        "coverage_note": "Regional XML leaf xlink:href target-existence boundary is executable from bounded sequence-package file-resolution checks.",
    },
    "sec_4_1_6": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-059"],
        "coverage_note": "Regional lifecycle target files must not correspond to multiple operations within one sequence, which is executable from all-leaf lifecycle metadata and sequence-root path normalization.",
    },
    "sec_4_1_7": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-064"],
        "coverage_note": "cn-regional.xml replace/append leaf checksum values must differ from the resolved modified-file leaf checksum when a prior target leaf is available.",
    },
    "sec_4_1_8": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-055"],
        "coverage_note": "Regional new, replace, and append leaf href-value exactness is executable from all-leaf lifecycle metadata extracted from cn-regional.xml.",
    },
    "sec_4_1_9": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-056"],
        "coverage_note": "Regional delete leaf href-value prohibition is executable from all-leaf lifecycle metadata, including href-less leaf records.",
    },
    "sec_4_1_10": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-057"],
        "coverage_note": "Regional replace, delete, and append leaf modified-file value exactness is executable from additive lifecycle metadata.",
    },
    "sec_4_1_11": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-058"],
        "coverage_note": "Initial-sequence regional leaf operation=new exactness is executable from sequence-package context plus cn-regional.xml lifecycle metadata.",
    },
    "sec_4_1_12": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-003"],
        "coverage_note": "Regional XML checksum-type must be MD5/md5 boundary is executable from XML leaf metadata.",
    },
    "sec_4_1_13": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-050"],
        "coverage_note": "cn-regional.xml leaf-declared checksum values must match actual target-file MD5 values, which is executable from bounded sequence-package file checks.",
    },
    "sec_4_1_14": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-051"],
        "coverage_note": "Cover-letter leaf records under cn-1-0 must declare operation=new, which is executable from bounded regional-backbone leaf metadata.",
    },
    "sec_4_1_15": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-052"],
        "coverage_note": "When sequence-type is initial submission, application-form leaf records under cn-1-2 must declare operation=new, which is executable from bounded sequence metadata plus regional-backbone leaf metadata.",
    },
    "sec_4_1_16": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-053"],
        "coverage_note": "Regional backbone node-extension prohibition is executable directly from cn-regional.xml node-extension metadata.",
    },
    "sec_4_1_17": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-054"],
        "coverage_note": "Regional XML leaf title non-empty exactness is executable from additive parser leaf-title records that include href-less leaves.",
    },
    "sec_4_1_18": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-060"],
        "coverage_note": "Regional delete leaf title continuity is executable by resolving modified-file targets to prior lifecycle leaf hrefs and comparing preserved leaf titles.",
    },
    "sec_4_1_19": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-024"],
        "coverage_note": "Regional XML leaf-title leading/trailing whitespace exactness is executable from preserved raw title text in additive leaf-title records.",
    },
    "sec_4_1_20": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-061"],
        "coverage_note": "Regional cn-content elements whose names begin with cn-[number] must have at least one descendant leaf, which is executable from cn-regional.xml structural metadata.",
    },
    "sec_4_1_21": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-016"],
        "coverage_note": "Application-folder name equality with the envelope application-number is executable from the package identity-chain rule, which compares the resolved application root with cn-regional/index envelope metadata.",
    },
    "sec_4_1_22": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-016"],
        "coverage_note": "Sequence-folder name equality with the envelope sequence-number is executable from the package identity-chain rule, which compares the resolved sequence root with cn-regional/index envelope metadata.",
    },
    "sec_4_1_23": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-015", "HR-ECTD-046"],
        "coverage_note": "Module-1 output folder structure is executable from bounded package-structure checks plus the no-direct-file-under-m1 boundary.",
    },
    "sec_4_1_24": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-027", "HR-ECTD-093"],
        "coverage_note": "Regional cn-regional.xml content references are checked for cross-application targets and later-sequence targets from resolved leaf href/modified-file path context; unresolved target context remains prerequisite-dependent.",
    },
    "sec_4_1_25": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-094"],
        "coverage_note": "cn-regional.xml append operations are checked against available prior lifecycle graph evidence so a leaf already replaced by another leaf cannot be appended; incomplete historical sequences remain explicit prerequisites.",
    },
    "sec_4_1_26": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-095"],
        "coverage_note": "cn-regional.xml delete operations are checked against available prior lifecycle graph evidence so a leaf already replaced by another leaf cannot be deleted; incomplete historical sequences remain explicit prerequisites.",
    },
    "sec_4_1_27": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-096"],
        "coverage_note": "cn-regional.xml replace operations are checked against available prior lifecycle graph evidence so a leaf already replaced by another leaf cannot be replaced a second time; incomplete historical sequences remain explicit prerequisites.",
    },
    "sec_4_1_28": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-097"],
        "coverage_note": "cn-regional.xml replace/delete/append operations are checked against available prior lifecycle graph evidence so an already deleted leaf cannot be operated on again; incomplete historical sequences remain explicit prerequisites.",
    },
    "sec_4_1_29": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-098"],
        "coverage_note": "cn-regional.xml append operations are checked against available prior lifecycle graph evidence so they cannot target a leaf whose operation is append; incomplete historical sequences remain explicit prerequisites.",
    },
    "sec_4_1_30": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-009"],
        "coverage_note": "cn-regional.xml append operations outside explicit STF scope are surfaced by the existing technical-spec non-STF append warning rule; explanation-letter adequacy remains reviewer-facing context.",
    },
    "sec_4_1_31": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-026"],
        "coverage_note": "cn-regional.xml replace-operation language consistency is executable through the technical-spec lifecycle language rule when a unique prior-sequence leaf match is available.",
    },
    "sec_4_2_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-002"],
        "coverage_note": "Application-number coding-rule exactness is executable through the technical-spec application-number format rule, including x/y/l plus year plus serial format and available application-type prefix semantics.",
    },
    "sec_4_2_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-037"],
        "coverage_note": "Application-type controlled-vocabulary validity is executable against cv-application-type.xml or the authoritative Attachment 1-2 bundle.",
    },
    "sec_4_2_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-038"],
        "coverage_note": "Product-type controlled-vocabulary validity is executable against cv-product-type.xml or the authoritative Attachment 1-2 bundle.",
    },
    "sec_4_2_4": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-103"],
        "coverage_note": "The validation-standard non-empty original-number requirement is executable from cn-regional.xml cn-envelope product-number element evidence.",
    },
    "sec_4_2_5": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-102"],
        "coverage_note": "The validation-standard related-sequence four-digit format and not-after-current ordering rule is executable from cn-regional.xml envelope metadata.",
    },
    "sec_4_2_6": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-039"],
        "coverage_note": "Regulatory-activity-type controlled-vocabulary validity is executable against cv-regulatory-activity-type.xml or the authoritative Attachment 1-2 bundle.",
    },
    "sec_4_2_7": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-001"],
        "coverage_note": "Sequence-number four-digit exactness is executable through the technical-spec sequence-number rule, with package sequence-directory consistency and known-history continuity checked when evidence is available.",
    },
    "sec_4_2_8": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-040"],
        "coverage_note": "Sequence-type controlled-vocabulary validity is executable against cv-sequence-type.xml or the authoritative Attachment 1-2 bundle.",
    },
    "sec_4_2_9": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-099"],
        "coverage_note": "The validation-standard non-empty and 120-Chinese-character limit for cn-regional.xml sequence-description is executable from envelope metadata.",
    },
    "sec_4_2_10": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-018"],
        "coverage_note": "Application-type, regulatory-activity-type, and sequence-type compatibility is executable against depend-apt-rat-sqt.xml or the authoritative Attachment 1-2 dependency matrix.",
    },
    "sec_4_2_11": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-100"],
        "coverage_note": "The validation-standard sequence-type-specific related-sequence non-self-reference rule is executable from cn-regional.xml envelope metadata and cv-sequence-type.xml codes.",
    },
    "sec_4_2_12": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-101"],
        "coverage_note": "The validation-standard initial-submission/reformat related-sequence self-reference rule is executable from cn-regional.xml envelope metadata and cv-sequence-type.xml codes.",
    },
    "sec_4_2_13": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-115"],
        "coverage_note": "application-level envelope immutability is checked against local initial sequence history for application-number, application-type, product-type, and original-number; missing initial sequence or missing comparable envelope fields return na/prerequisite guidance rather than assumed pass/fail.",
    },
    "sec_4_2_14": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-116"],
        "coverage_note": "same regulatory activity regulatory-activity-type immutability is checked from local same-application sequence packages grouped by related-sequence; missing same-activity history, missing related-sequence, or missing regulatory-activity-type evidence return na/prerequisite guidance rather than assumed pass/fail.",
    },
    "sec_4_3_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-106"],
        "coverage_note": "The validation-standard new-drug initial/new-indication first-submission module-1 required-element checklist is executable from cn-regional.xml envelope codes and cn-content element structure.",
    },
    "sec_4_3_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-107"],
        "coverage_note": "The validation-standard prohibition on listed module-1 elements for new-drug initial/new-indication first submissions is executable from cn-regional.xml envelope codes and cn-content element structure.",
    },
    "sec_4_3_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-108"],
        "coverage_note": "The validation-standard generic-drug initial/new-indication first-submission module-1 required-element checklist is executable from cn-regional.xml envelope codes and cn-content element structure.",
    },
    "sec_4_3_4": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-109"],
        "coverage_note": "The validation-standard prohibition on listed module-1 elements for generic-drug initial/new-indication first submissions is executable from cn-regional.xml envelope codes and cn-content element structure.",
    },
    "sec_4_3_5": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-110"],
        "coverage_note": "The validation-standard clinical-trial initial/new-indication-and-combination first-submission module-1 required-element checklist is executable from cn-regional.xml envelope codes and cn-content element structure.",
    },
    "sec_4_3_6": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-111"],
        "coverage_note": "The validation-standard prohibition on listed module-1 elements for clinical-trial initial/new-indication-and-combination first submissions is executable from cn-regional.xml envelope codes and cn-content element structure.",
    },
    "sec_4_3_7": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-104"],
        "coverage_note": "The validation-standard development-safety-report module-1 completeness requirement is executable from cn-regional.xml envelope codes and cn-content element structure for cn-1-8-1/cn-1-8-2.",
    },
    "sec_4_3_8": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-105"],
        "coverage_note": "The validation-standard prohibition on submitting both development-safety-report module-1 safety elements is executable from cn-regional.xml envelope codes and cn-content element structure.",
    },
    "sec_4_3_9": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-055"],
        "coverage_note": "The validation-standard advisory leaf-count limit for the listed module-1 elements is executable from cn-regional.xml element structure and descendant leaf counts; multiple descendant leaves are surfaced as a warning.",
    },
    "sec_5_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-090"],
        "coverage_note": "STF XML well-formedness, DOCTYPE reference to local util/dtd/ich-stf-v2-2.dtd, and local DTD validation are executable when the DTD resolves; unresolved validation prerequisites remain explicit guidance.",
    },
    "sec_5_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-006"],
        "coverage_note": "STF XML leaf xlink:href target-existence boundary is covered by the same bounded package file-resolution check used for index.xml and regional XML references.",
    },
    "sec_5_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-030"],
        "coverage_note": "STF XML content-block usage is detected directly from preserved XML element records and surfaced as a validation-standard warning.",
    },
    "sec_5_4": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-088"],
        "coverage_note": "STF XML xlink:href values containing backslashes are detected from preserved XML attribute records as a hard validation error.",
    },
    "sec_5_5": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-031"],
        "coverage_note": "STF study-identifier category values are checked for missing or blank content from objective XML element text evidence.",
    },
    "sec_5_6": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-032"],
        "coverage_note": "STF study-identifier study-id values are checked for missing or blank content from objective XML element text evidence.",
    },
    "sec_5_7": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-037"],
        "coverage_note": "STF study-identifier title values are compared with uniquely resolved index.xml leaf titles via doc-content href to leaf ID evidence; unresolved or ambiguous references remain prerequisite guidance.",
    },
    "sec_5_8": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-036"],
        "coverage_note": "STF file-tag name attributes and named category values are checked against the local Attachment 2-6 valid-values.xml bundle when comparable values are available.",
    },
    "sec_5_18": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-040"],
        "coverage_note": "Module 5 STF dataset references are checked for the validation-standard 5.18 file-tag/file-type mapping when a doc-content has a directly comparable xpt dataset or define.xml leaf and a unique file-tag name; incomplete STF structure remains prerequisite guidance.",
    },
    "sec_5_19": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-041"],
        "coverage_note": "Module 5 STF dataset package families are checked for required dm/adsl xpt dataset and matching data-definition evidence when raw data-tabulation or analysis dataset family tags are locally comparable; broader clinical database completeness remains prerequisite/human-review guidance.",
    },
    "sec_5_20": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-042"],
        "coverage_note": "Module 5 STF dataset names are checked for duplicates within a single local STF study document using normalized xpt filename stems from directly comparable dataset doc-content entries; cross-document same-study grouping and broader package inventory completeness remain prerequisite/human-review guidance.",
    },
    "sec_5_9": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-043"],
        "coverage_note": "STF-scoped index.xml append operations are checked so their modified-file target must resolve to another locally classifiable STF leaf; missing historical target metadata remains prerequisite/human-review guidance.",
    },
    "sec_5_10": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-038"],
        "coverage_note": "STF study-identifier category presence is checked for the explicitly category-required CTD sections 4.2.3.1, 4.2.3.2, 4.2.3.4.1, and 5.3.5.1 using index.xml section context or clear numeric path evidence; unmapped STF files remain prerequisite/human-review guidance.",
    },
    "sec_5_11": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-044"],
        "coverage_note": "STF XML leaf references are checked for locally resolvable targets that are strongly classifiable as another STF XML document; unresolved XML targets and ambiguous classifications remain prerequisite/human-review guidance.",
    },
    "sec_5_13": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-039"],
        "coverage_note": "STF study-id lifecycle consistency is checked only when index.xml replace/append lifecycle metadata links a current STF XML file to a locally available modified-file STF XML file and both sides have unique non-empty study-id values; missing history or ambiguous STF identity remains prerequisite/human-review guidance.",
    },
    "sec_5_12": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-034"],
        "coverage_note": "STF XML files are checked for at least one leaf element reference from objective XML element structure evidence.",
    },
    "sec_5_14": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-035"],
        "coverage_note": "STF-like XML files are checked against bounded module directory classification for Module 4 4.2.x and Module 5 5.3.1.x-5.3.5.x, with unclassified paths kept as prerequisite guidance.",
    },
    "sec_5_15": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-033"],
        "coverage_note": "Each STF doc-content element is checked for exactly one direct file-tag child from objective XML element structure evidence.",
    },
    "sec_5_16": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-089"],
        "coverage_note": "When index.xml shows explicit STF usage in the current sequence, Module 5 section 5.3.7 case-report-form leafs are prohibited using local leaf structure evidence; absent STF usage remains not applicable/prerequisite guidance.",
    },
    "sec_5_17": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-010"],
        "coverage_note": "STF-required zones in Module 4 section 4.2 and Module 5 sections 5.3.1 through 5.3.5 are checked for explicit STF parent structure using index.xml leaf records; cross-document STF content completeness remains prerequisite/human-review guidance.",
    },
    "sec_6_17": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-041"],
        "coverage_note": "Embedded PDF attachment prohibition is executable from bounded parser metadata.",
    },
    "sec_6_18": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-017"],
        "coverage_note": "Non-link PDF annotation restriction is executable from bounded parser metadata.",
    },
    "sec_6_19": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-042"],
        "coverage_note": "PDF security-setting prohibition is executable from bounded encryption and permission metadata.",
    },
    "sec_6_1": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-044"],
        "coverage_note": "PDF readability boundary is executable from bounded parser openability and page-count metadata, including graceful unreadable-file fallback.",
    },
    "sec_6_21": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-043"],
        "coverage_note": "Password-protected PDF prohibition is executable from bounded parser preflight metadata.",
    },
    "sec_6_16": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-018"],
        "coverage_note": "Allowed PDF-version boundary is executable from bounded format metadata.",
    },
    "sec_6_20": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-019"],
        "coverage_note": "Initial-view default boundary is executable from bounded catalog metadata (`PageMode`, `PageLayout`, `OpenAction`).",
    },
    "sec_6_9": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-021"],
        "coverage_note": "PDF hyperlink relative-path boundary is executable from bounded external-file link target strings without broader viewer semantics.",
    },
    "sec_6_10": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-022"],
        "coverage_note": "PDF external hyperlink prohibition is executable from bounded external URI hyperlink counts without broader hyperlink-action semantics.",
    },
    "sec_6_11": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-045"],
        "coverage_note": "PDF hyperlink action whitelist boundary is executable from bounded link-annotation action names extracted from raw PDF link objects.",
    },
    "sec_6_4": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-091"],
        "coverage_note": "PDF bookmark action whitelist boundary is executable from raw outline/bookmark action dictionaries and their `/S` action names; missing metadata remains prerequisite/NA rather than assumed.",
    },
    "sec_6_3": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["HR-ECTD-092"],
        "coverage_note": "PDF bookmark external-link boundary is executable from raw outline/bookmark URI target strings such as web or mail links; broader target reachability remains outside this deterministic check.",
    },
    "sec_6_5": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-051"],
        "coverage_note": "PDF invalid-bookmark boundary is executable for bookmarks that expose neither an assigned action nor a destination in raw outline/bookmark metadata; broader damaged-target semantics remain separate.",
    },
    "sec_6_6": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-052"],
        "coverage_note": "PDF damaged-bookmark boundary is executable for raw internal bookmark destinations whose destination array is empty, malformed, or points to a non-page object; viewer-normalized navigation behavior is not used as evidence.",
    },
    "sec_6_8": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-053"],
        "coverage_note": "PDF bookmark inherited-zoom boundary is executable from raw destination arrays: `/XYZ` destinations with `null` or `0` zoom are accepted as inherited/current zoom, while Fit-style or explicit non-zero zoom destinations are surfaced as warnings.",
    },
    "sec_6_2": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-050"],
        "coverage_note": "PDF bookmark relative-path boundary is executable from raw outline/bookmark external-file `/F` target strings without broader target-existence or viewer semantics.",
    },
    "sec_6_12": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-023"],
        "coverage_note": "Broken hyperlink boundary is executable from bounded raw link objects that lack any valid action or destination.",
    },
    "sec_6_13": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-023"],
        "coverage_note": "PDF broken-hyperlink prohibition is executable from bounded raw link objects that lack any valid action or destination.",
    },
    "sec_6_7": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-049"],
        "coverage_note": "PDF bookmark single-action boundary is executable from raw outline/bookmark action dictionaries that contain explicit `/Next` action-chain markers; top-level outline sibling `/Next` pointers are not counted.",
    },
    "sec_6_14": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-048"],
        "coverage_note": "PDF hyperlink single-action boundary is executable from explicit raw link-annotation `/Next` action-chain markers; missing metadata remains prerequisite/NA rather than assumed.",
    },
    "sec_6_15": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-054"],
        "coverage_note": "PDF hyperlink inherited-zoom boundary is executable from raw link annotation destination arrays: `/XYZ` destinations with `null` or `0` zoom are accepted as inherited/current zoom, while Fit-style or explicit non-zero zoom destinations are surfaced as warnings.",
    },
    "sec_6_24": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-020"],
        "coverage_note": "Disallowed active-content boundary is executable from bounded object-marker detection (`JavaScript`, `3D`, `RichMedia`, `Movie`, `Sound`, `Screen`).",
    },
    "sec_6_25": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-045"],
        "coverage_note": "PDF searchable-text boundary is executable from existing parser text-layer evidence: readable non-exempt PDFs with pages must expose content evidence or content units; scanned pages without extractable text remain OCR/manual-review warnings.",
    },
    "sec_6_22": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-046"],
        "coverage_note": "PDF Fast Web Access boundary is executable from objective PDF linearization metadata detected near the file header; missing linearization evidence remains prerequisite/NA rather than assumed.",
    },
    "sec_6_26": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-047"],
        "coverage_note": "PDF non-standard font embedding is executable from objective page font records and embedded font-program markers; missing font metadata remains prerequisite/NA rather than assumed.",
    },
    "sec_6_23": {
        "coverage_status": "covered",
        "implemented_rule_ids": ["SR-ECTD-016"],
        "coverage_note": "Long-PDF bookmark requirement is executable from embedded outline metadata.",
    },
}

_ECTD_VALIDATION_STANDARD_CHAPTER_DEFAULTS: dict[int, dict[str, Any]] = {
    1: {
        "coverage_status": "citation_only_recorded",
        "implemented_rule_ids": [],
        "coverage_note": "Chapter 1 is validation-report informational output rather than a dossier-decidable rule surface.",
    },
    2: {
        "coverage_status": "covered",
        "implemented_rule_ids": [],
        "coverage_note": "Chapter 2 package/file-system rules are mostly covered; see clause-level overrides for the remaining bounded gaps.",
    },
    3: {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": [],
        "coverage_note": "Many ICH backbone integrity checks are already represented by existing eCTD runtime rules, but article-by-article validation-standard parity is not yet fully aligned.",
    },
    4: {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": [],
        "coverage_note": "Regional XML/package structure has strong runtime coverage, but the full validation-standard chapter-4 article surface is not yet explicitly mapped item by item.",
    },
    5: {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": [],
        "coverage_note": "STF structure and bounded lifecycle semantics are covered only at a practical subset level; many STF-specific validation items still need explicit mapping.",
    },
    6: {
        "coverage_status": "partially_covered",
        "implemented_rule_ids": [],
        "coverage_note": "PDF navigation and hyperlink subsets are covered, but most PDF-format validation checks remain only partially modeled.",
    },
}


@dataclass(slots=True)
class ClauseAccumulator:
    article_no_raw: str
    article_no: int
    chapter_no_raw: str
    chapter_no: int
    chapter_title: str
    heading: str
    line_index: int
    body_lines: list[str]


@dataclass(slots=True)
class SectionClauseAccumulator:
    clause_no_raw: str
    section_no_raw: str
    section_no: int
    section_title: str
    heading: str
    line_index: int
    body_lines: list[str]


@dataclass(slots=True)
class NumberedClauseAccumulator:
    clause_no_raw: str
    chapter_no_raw: str
    chapter_no: int
    chapter_title: str
    heading: str
    line_index: int
    body_lines: list[str]
    clause_ref_suffix: str


def _chinese_numeral_to_int(value: str) -> int:
    value = str(value or "").strip()
    if not value:
        return 0
    if value.isdigit():
        return int(value)

    digit_map = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    unit_map = {"十": 10, "百": 100}

    total = 0
    current = 0
    for char in value:
        if char in digit_map:
            current = digit_map[char]
            continue
        unit = unit_map.get(char)
        if unit is None:
            continue
        if current == 0:
            current = 1
        total += current * unit
        current = 0
    total += current
    return total


def _strip_word_html(text: str) -> str:
    cleaned = _HTML_SCRIPT_PATTERN.sub(" ", text)
    cleaned = _HTML_STYLE_PATTERN.sub(" ", cleaned)
    cleaned = _HTML_COMMENT_PATTERN.sub(" ", cleaned)
    cleaned = _HTML_TAG_PATTERN.sub("\n", cleaned)
    return html.unescape(cleaned).replace("\xa0", " ")


def normalize_regulation_source_text(text: str) -> str:
    normalized = str(text or "")
    lowered = normalized.lower()
    if "<html" in lowered or "<body" in lowered:
        normalized = _strip_word_html(normalized)
    normalized = normalized.replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{2,}", "\n", normalized)
    return normalized.strip()


def _tokenize_nonempty_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = " ".join(raw_line.split())
        if line:
            lines.append(line)
    return lines


def _normalize_heading_title(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", str(text or "")).strip()


def _build_numbered_clause_ref_suffix(number: str) -> str:
    normalized = str(number or "").strip().replace(".", "_")
    return f"sec_{normalized}"


def _format_numbered_heading(number: str, title: str) -> str:
    normalized_number = str(number or "").strip()
    normalized_title = _normalize_heading_title(title)
    if "." in normalized_number:
        return f"{normalized_number} {normalized_title}"
    return f"{normalized_number}. {normalized_title}"


def _extract_numbered_heading_parts(line: str) -> tuple[str, str] | None:
    match = _NUMBERED_HEADING_PATTERN.match(str(line or "").strip())
    if not match:
        return None
    return str(match.group("number") or "").strip(), _normalize_heading_title(match.group("title"))


def _collect_numbered_heading_candidates(lines: list[str]) -> list[tuple[int, str, str]]:
    candidates: list[tuple[int, str, str]] = []
    previous_line = ""
    for line_index, line in enumerate(lines, start=1):
        parts = _extract_numbered_heading_parts(line)
        if parts is None:
            previous_line = line
            continue
        number, title = parts
        if "." not in number and len(number) > 2:
            previous_line = line
            continue
        if _DOT_LEADER_PATTERN.search(line):
            previous_line = line
            continue
        if str(previous_line).rstrip().endswith(("：", ":", "；", ";")):
            previous_line = line
            continue
        if title.endswith(("。", "；", ";", "：", ":")):
            previous_line = line
            continue
        candidates.append((line_index, number, title))
        previous_line = line
    return candidates


def _find_numbered_body_start_index(lines: list[str]) -> int | None:
    candidates = _collect_numbered_heading_candidates(lines)
    toc_titles = _extract_numbered_toc_titles(lines, len(lines) + 1)
    if toc_titles:
        for line_index, number, title in candidates:
            if "." in number:
                continue
            if toc_titles.get(number) == title:
                return line_index
    top_level_positions = [line_index for line_index, number, _ in candidates if "." not in number and number == "1"]
    if len(top_level_positions) >= 2:
        return top_level_positions[1]
    if candidates:
        return candidates[0][0]
    return None


def _extract_numbered_toc_titles(lines: list[str], body_start_index: int) -> dict[str, str]:
    toc_titles: dict[str, str] = {}
    for line in lines[: max(body_start_index - 1, 0)]:
        match = _NUMBERED_TOC_PATTERN.match(line)
        if not match:
            continue
        number = str(match.group("number") or "").strip()
        if "." in number:
            continue
        toc_titles[number] = _normalize_heading_title(match.group("title"))
    return toc_titles


def _normalize_toc_outline_index(value: str) -> str:
    outline_index = str(value or "").strip()
    if outline_index.endswith(".0"):
        outline_index = outline_index[:-2]
    return outline_index


def _extract_numbered_toc_outline_titles(parsed_document: dict[str, Any]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for sequence in parsed_document.get("toc_sequences", []) or []:
        for entry in sequence.get("entries", []) or []:
            outline_index = _normalize_toc_outline_index(entry.get("outline_index"))
            if not outline_index:
                continue
            text = _normalize_heading_title(entry.get("text"))
            if not text:
                continue
            titles[outline_index] = text
    return titles


def _resolve_regulation_metadata(path: Path, lines: list[str]) -> dict[str, Any]:
    stem = path.stem
    override = _REGULATION_METADATA_OVERRIDES.get(stem, {})
    validation_standard_stem = Path(
        bytes("eCTD\\u9a8c\\u8bc1\\u6807\\u51c6.pdf", "ascii").decode("unicode_escape")
    ).stem
    if stem == validation_standard_stem:
        override = {
            "regulation_id": "cn_ectd_validation_standard",
            "title": bytes("eCTD\\u9a8c\\u8bc1\\u6807\\u51c6", "ascii").decode("unicode_escape"),
            "issuer": "鍥藉鑽搧鐩戠潱绠＄悊灞€",
            "jurisdiction": "CN",
            "domain": "ectd_validation_standard",
            "version_label": "2021-09-v1.0",
        }
    title = override.get("title") or next((line for line in lines if len(line) >= 4), stem)
    regulation_id = override.get("regulation_id") or f"reg_{sha1(title.encode('utf-8')).hexdigest()[:12]}"
    metadata: dict[str, Any] = {
        "regulation_id": regulation_id,
        "title": title,
        "issuer": override.get("issuer"),
        "jurisdiction": override.get("jurisdiction", "unknown"),
        "domain": override.get("domain", "unknown"),
        "version_label": override.get("version_label", "unknown"),
        "source_filename": path.name,
        "source_path": str(path),
    }
    return metadata


def _extract_chapters_and_clauses(
    lines: list[str],
    metadata: dict[str, Any],
    parsed_document: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if str(metadata.get("regulation_id") or "").strip() == "cn_ectd_validation_standard":
        validation_lines = _load_validation_standard_lines(Path(str(metadata.get("source_path") or "").strip()))
        validation_chapters, validation_clauses = _extract_ectd_validation_standard_units(validation_lines, metadata)
        if validation_chapters or validation_clauses:
            return validation_chapters, validation_clauses

    chapters: list[dict[str, Any]] = []
    clauses: list[dict[str, Any]] = []
    current_chapter_no = 0
    current_chapter_no_raw = ""
    current_chapter_title = ""
    active_clause: ClauseAccumulator | None = None

    def flush_active_clause() -> None:
        nonlocal active_clause
        if active_clause is None:
            return
        body_text = _normalize_clause_text(" ".join(active_clause.body_lines))
        clauses.append(
            _build_clause_record(
                metadata=metadata,
                chapter_no_raw=active_clause.chapter_no_raw,
                chapter_no=active_clause.chapter_no,
                chapter_title=active_clause.chapter_title,
                article_no_raw=active_clause.article_no_raw,
                article_no=active_clause.article_no,
                heading=active_clause.heading,
                normalized_text=body_text,
                line_index=active_clause.line_index,
            )
        )
        active_clause = None

    for line_index, line in enumerate(lines, start=1):
        chapter_match = _CHAPTER_PATTERN.match(line)
        if chapter_match:
            flush_active_clause()
            current_chapter_no_raw = chapter_match.group(1)
            current_chapter_no = _chinese_numeral_to_int(current_chapter_no_raw)
            current_chapter_title = chapter_match.group(2).strip()
            chapters.append(
                {
                    "chapter_id": f"{metadata['regulation_id']}:ch_{current_chapter_no:02d}",
                    "chapter_no_raw": current_chapter_no_raw,
                    "chapter_no": current_chapter_no,
                    "chapter_title": current_chapter_title,
                    "heading": line,
                    "line_index": line_index,
                }
            )
            continue

        article_match = _ARTICLE_PATTERN.match(line)
        if article_match:
            flush_active_clause()
            article_no_raw = article_match.group(1)
            article_no = _chinese_numeral_to_int(article_no_raw)
            active_clause = ClauseAccumulator(
                article_no_raw=article_no_raw,
                article_no=article_no,
                chapter_no_raw=current_chapter_no_raw,
                chapter_no=current_chapter_no,
                chapter_title=current_chapter_title,
                heading=f"第{article_no_raw}条",
                line_index=line_index,
                body_lines=[line],
            )
            continue

        if active_clause is not None:
            active_clause.body_lines.append(line)

    flush_active_clause()
    if clauses:
        return chapters, clauses
    sectioned_chapters, sectioned_clauses = _extract_sectioned_regulation_units(lines, metadata)
    if sectioned_chapters or sectioned_clauses:
        return sectioned_chapters, sectioned_clauses
    numbered_chapters, numbered_clauses = _extract_numbered_heading_regulation_units(
        lines,
        metadata,
        parsed_document=parsed_document or {},
    )
    if numbered_chapters or numbered_clauses:
        return numbered_chapters, numbered_clauses
    return chapters, clauses


def _load_validation_standard_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    doc = fitz.open(path)
    lines: list[str] = []
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            for raw_line in page.get_text("text").splitlines():
                line = _normalize_heading_title(raw_line)
                if not line:
                    continue
                lines.append(line)
    finally:
        doc.close()
    return lines


def _extract_ectd_validation_standard_units(
    lines: list[str],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    footer_title_pattern = re.compile(r"^(?P<number>\d)\s*-\s*(?P<title>.+)$")
    clause_number_pattern = re.compile(r"^(?P<number>\d+(?:\.\d+)+)$")
    severity_values = {"错误", "警告", "提示信息"}

    chapter_title_map: dict[str, str] = {}
    for line in lines:
        match = footer_title_pattern.match(str(line or "").strip())
        if not match:
            continue
        title = _normalize_heading_title(match.group("title"))
        if title.isdigit():
            continue
        chapter_title_map[str(match.group("number") or "").strip()] = title

    chapters: list[dict[str, Any]] = []
    clauses: list[dict[str, Any]] = []
    chapter_seen: set[str] = set()
    current_clause_number = ""
    current_clause_title = ""
    current_clause_lines: list[str] = []
    current_clause_line_index = 0
    clause_counter = 0

    def flush_clause() -> None:
        nonlocal current_clause_number, current_clause_title, current_clause_lines, current_clause_line_index, clause_counter
        if not current_clause_number:
            return
        top_level_no = current_clause_number.split(".", 1)[0]
        chapter_title = chapter_title_map.get(top_level_no, "")
        if top_level_no not in chapter_seen:
            chapter_seen.add(top_level_no)
            chapters.append(
                {
                    "chapter_id": f"{metadata['regulation_id']}:ch_{int(top_level_no):02d}",
                    "chapter_no_raw": top_level_no,
                    "chapter_no": int(top_level_no),
                    "chapter_title": chapter_title,
                    "heading": f"{top_level_no}. {chapter_title}" if chapter_title else top_level_no,
                    "line_index": current_clause_line_index,
                }
            )
        clause_counter += 1
        clauses.append(
            _build_clause_record(
                metadata=metadata,
                chapter_no_raw=top_level_no,
                chapter_no=int(top_level_no),
                chapter_title=chapter_title,
                article_no_raw=current_clause_number,
                article_no=clause_counter,
                heading=_format_numbered_heading(current_clause_number, current_clause_title),
                normalized_text=_normalize_clause_text(" ".join(current_clause_lines)),
                line_index=current_clause_line_index,
                clause_ref_suffix=_build_numbered_clause_ref_suffix(current_clause_number),
            )
        )
        current_clause_number = ""
        current_clause_title = ""
        current_clause_lines = []
        current_clause_line_index = 0

    def split_entry_title_and_body(
        number: str,
        entry_lines: list[str],
    ) -> tuple[str, list[str]]:
        if not entry_lines:
            return number, []

        title_override = _ECTD_VALIDATION_STANDARD_TITLE_OVERRIDES.get(number)
        if title_override:
            if number in _ECTD_VALIDATION_STANDARD_FULL_TITLE_ONLY_ARTICLES:
                return title_override, []
            if number in _ECTD_VALIDATION_STANDARD_INLINE_TITLE_SPLIT_ARTICLES:
                first_line = entry_lines[0]
                remainder = first_line[len(title_override):].strip()
                detail_lines = [remainder] if remainder else []
                detail_lines.extend(entry_lines[1:])
                return title_override, detail_lines
            return title_override, entry_lines

        return entry_lines[0], entry_lines[1:]

    index = 0
    while index < len(lines):
        line = str(lines[index] or "").strip()
        match = clause_number_pattern.match(line)
        if not match:
            index += 1
            continue

        number = str(match.group("number") or "").strip()
        flush_clause()
        current_clause_number = number
        current_clause_line_index = index + 1

        entry_lines: list[str] = []
        body_index = index + 1
        while body_index < len(lines):
            candidate = str(lines[body_index] or "").strip()
            if clause_number_pattern.match(candidate):
                break
            if candidate in {"序号", "描述", "说明", "严重程度"}:
                body_index += 1
                continue
            if footer_title_pattern.match(candidate) or candidate.startswith("第 ") or candidate.startswith("eCTD验证标准"):
                body_index += 1
                continue
            if candidate:
                if candidate in _ECTD_VALIDATION_STANDARD_TAIL_STOP_MARKERS:
                    break
                entry_lines.append(candidate)
                if candidate in severity_values:
                    break
            body_index += 1

        severity_index = next(
            (i for i, entry_line in enumerate(entry_lines) if entry_line in severity_values),
            len(entry_lines),
        )
        content_lines = entry_lines[:severity_index]
        trailing_lines = entry_lines[severity_index:]
        current_clause_title, detail_lines = split_entry_title_and_body(number, content_lines)
        current_clause_lines = [number, current_clause_title]
        current_clause_lines.extend(detail_lines)
        current_clause_lines.extend(trailing_lines)

        index = body_index

    flush_clause()
    return chapters, clauses


def _extract_sectioned_regulation_units(
    lines: list[str],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chapters: list[dict[str, Any]] = []
    clauses: list[dict[str, Any]] = []
    current_section_no = 0
    current_section_no_raw = ""
    current_section_title = ""
    active_clause: SectionClauseAccumulator | None = None
    clause_counter = 0

    def flush_active_clause() -> None:
        nonlocal active_clause, clause_counter
        if active_clause is None:
            return
        clause_counter += 1
        body_text = _normalize_clause_text(" ".join(active_clause.body_lines))
        clauses.append(
            _build_clause_record(
                metadata=metadata,
                chapter_no_raw=active_clause.section_no_raw,
                chapter_no=active_clause.section_no,
                chapter_title=active_clause.section_title,
                article_no_raw=active_clause.clause_no_raw,
                article_no=clause_counter,
                heading=active_clause.heading,
                normalized_text=body_text,
                line_index=active_clause.line_index,
            )
        )
        active_clause = None

    for line_index, line in enumerate(lines, start=1):
        section_match = _SECTION_PATTERN.match(line)
        if section_match:
            flush_active_clause()
            current_section_no_raw = str(section_match.group("section_no") or "").strip()
            current_section_no = _chinese_numeral_to_int(current_section_no_raw)
            current_section_title = str(section_match.group("title") or "").strip()
            chapters.append(
                {
                    "chapter_id": f"{metadata['regulation_id']}:sec_{current_section_no:02d}",
                    "chapter_no_raw": current_section_no_raw,
                    "chapter_no": current_section_no,
                    "chapter_title": current_section_title,
                    "heading": line,
                    "line_index": line_index,
                }
            )
            continue

        class_match = _CLASS_ITEM_PATTERN.match(line)
        if class_match and current_section_no > 0:
            flush_active_clause()
            class_no = str(class_match.group("class_no") or "").strip()
            active_clause = SectionClauseAccumulator(
                clause_no_raw=f"{class_no}类",
                section_no_raw=current_section_no_raw,
                section_no=current_section_no,
                section_title=current_section_title,
                heading=f"{class_no}类",
                line_index=line_index,
                body_lines=[line],
            )
            continue

        item_match = _PAREN_ITEM_PATTERN.match(line)
        if item_match and current_section_no > 0:
            flush_active_clause()
            item_no = str(item_match.group("item_no") or "").strip()
            active_clause = SectionClauseAccumulator(
                clause_no_raw=f"（{item_no}）",
                section_no_raw=current_section_no_raw,
                section_no=current_section_no,
                section_title=current_section_title,
                heading=f"（{item_no}）",
                line_index=line_index,
                body_lines=[line],
            )
            continue

        if current_section_no == 1 and current_section_title and (
            line.startswith("原研药品是")
            or line.startswith("参比制剂是")
        ):
            flush_active_clause()
            definition_heading = "原研药品定义" if line.startswith("原研药品是") else "参比制剂定义"
            active_clause = SectionClauseAccumulator(
                clause_no_raw=definition_heading,
                section_no_raw=current_section_no_raw,
                section_no=current_section_no,
                section_title=current_section_title,
                heading=definition_heading,
                line_index=line_index,
                body_lines=[line],
            )
            continue

        if active_clause is not None:
            active_clause.body_lines.append(line)

    flush_active_clause()
    return chapters, clauses


def _extract_numbered_heading_regulation_units(
    lines: list[str],
    metadata: dict[str, Any],
    *,
    parsed_document: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    body_start_index = _find_numbered_body_start_index(lines)
    if body_start_index is None:
        return [], []

    parsed_document = parsed_document or {}
    toc_outline_titles = _extract_numbered_toc_outline_titles(parsed_document)
    toc_titles = {
        number: title
        for number, title in toc_outline_titles.items()
        if "." not in number
    } or _extract_numbered_toc_titles(lines, body_start_index)
    allowed_numbered_titles = toc_outline_titles
    chapters: list[dict[str, Any]] = []
    clauses: list[dict[str, Any]] = []
    current_chapter_no = 0
    current_chapter_no_raw = ""
    current_chapter_title = ""
    active_clause: NumberedClauseAccumulator | None = None
    clause_counter = 0

    def flush_active_clause() -> None:
        nonlocal active_clause, clause_counter
        if active_clause is None:
            return
        clause_counter += 1
        body_text = _normalize_clause_text(" ".join(active_clause.body_lines))
        clauses.append(
            _build_clause_record(
                metadata=metadata,
                chapter_no_raw=active_clause.chapter_no_raw,
                chapter_no=active_clause.chapter_no,
                chapter_title=active_clause.chapter_title,
                article_no_raw=active_clause.clause_no_raw,
                article_no=clause_counter,
                heading=active_clause.heading,
                normalized_text=body_text,
                line_index=active_clause.line_index,
                clause_ref_suffix=active_clause.clause_ref_suffix,
            )
        )
        active_clause = None

    for line_index in range(body_start_index, len(lines) + 1):
        line = lines[line_index - 1]
        parts = _extract_numbered_heading_parts(line)
        if parts is None:
            if active_clause is not None:
                active_clause.body_lines.append(line)
            continue

        number, title = parts
        if _DOT_LEADER_PATTERN.search(line):
            if active_clause is not None:
                active_clause.body_lines.append(line)
            continue

        depth = number.count(".") + 1
        if depth == 1:
            expected_title = toc_titles.get(number)
            if toc_titles:
                title_matches = bool(expected_title) and (
                    title == expected_title or title.startswith(expected_title)
                )
                if not title_matches:
                    if active_clause is not None:
                        active_clause.body_lines.append(line)
                    continue
                title = expected_title
            flush_active_clause()
            current_chapter_no_raw = number
            current_chapter_no = int(number)
            current_chapter_title = title
            chapters.append(
                {
                    "chapter_id": f"{metadata['regulation_id']}:ch_{current_chapter_no:02d}",
                    "chapter_no_raw": current_chapter_no_raw,
                    "chapter_no": current_chapter_no,
                    "chapter_title": current_chapter_title,
                    "heading": _format_numbered_heading(number, title),
                    "line_index": line_index,
                }
            )
            continue

        expected_clause_title = allowed_numbered_titles.get(number)
        if allowed_numbered_titles and not expected_clause_title:
            if active_clause is not None:
                active_clause.body_lines.append(line)
            continue

        flush_active_clause()
        top_level_no = number.split(".", 1)[0]
        top_level_title = current_chapter_title
        if not top_level_title or current_chapter_no_raw != top_level_no:
            top_level_title = toc_titles.get(top_level_no, "")
        canonical_title = expected_clause_title or title
        active_clause = NumberedClauseAccumulator(
            clause_no_raw=number,
            chapter_no_raw=top_level_no,
            chapter_no=int(top_level_no),
            chapter_title=top_level_title,
            heading=_format_numbered_heading(number, canonical_title),
            line_index=line_index,
            body_lines=[line],
            clause_ref_suffix=_build_numbered_clause_ref_suffix(number),
        )

    flush_active_clause()
    return chapters, clauses


def _normalize_clause_text(text: str) -> str:
    normalized = _WHITESPACE_PATTERN.sub(" ", str(text or "")).strip()
    normalized = re.sub(
        r"\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        r" \1 年 \2 月 \3 日",
        normalized,
    )
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    normalized = re.sub(r"(\d)\s+个工作日", r"\1个工作日", normalized)
    normalized = re.sub(r"第\s+([一二三四五六七八九十百零〇0-9]+)\s+条", r"第\1条", normalized)
    return normalized


def _derive_expected_material_evidence(text: str) -> list[str]:
    evidence_types: list[str] = []
    if "申请" in text or "药品注册" in text:
        evidence_types.append("general_registration_dossier")
    if "临床试验" in text:
        evidence_types.append("clinical_trial_materials")
    if "非临床" in text:
        evidence_types.append("nonclinical_study_materials")
    if "药品注册标准" in text or "标准品" in text or "对照品" in text:
        evidence_types.append("quality_standard_materials")
    if "样品" in text or "参比制剂" in text:
        evidence_types.append("reference_sample_materials")
    if "注册证书" in text or "批准证书" in text or "许可证" in text:
        evidence_types.append("certificate_or_license_materials")
    if "中药" in text:
        evidence_types.append("tcm_supporting_materials")
    if "化学原料药" in text or "原料药" in text:
        evidence_types.append("api_supporting_materials")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in evidence_types:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def _classify_clause(chapter_no: int, article_no: int, text: str) -> dict[str, Any]:
    evidence_types = _derive_expected_material_evidence(text)
    direct_hit = any(keyword in text for keyword in _DIRECT_MATERIAL_KEYWORDS)
    partial_hit = any(keyword in text for keyword in _PARTIAL_MATERIAL_KEYWORDS)
    soft_hit = any(keyword in text for keyword in _SOFT_REVIEW_KEYWORDS)
    hard_hit = any(keyword in text for keyword in _MANDATORY_HARD_KEYWORDS)

    if chapter_no == 2 and direct_hit:
        material_checkability = "direct"
        ind_relevance = "high"
        recommended_rule_mode = "hard" if hard_hit and not soft_hit else "soft"
    elif chapter_no in {2, 3, 4} and (direct_hit or partial_hit):
        material_checkability = "partial"
        ind_relevance = "high" if chapter_no == 2 else "medium"
        recommended_rule_mode = "review_only" if not direct_hit else "soft"
    elif chapter_no in {1, 9}:
        material_checkability = "external_only"
        ind_relevance = "low"
        recommended_rule_mode = "citation_only"
    elif chapter_no in {5, 6, 7, 8}:
        material_checkability = "external_only"
        ind_relevance = "low"
        recommended_rule_mode = "citation_only"
    else:
        material_checkability = "partial"
        ind_relevance = "medium"
        recommended_rule_mode = "review_only"

    candidate_reason = (
        "Direct dossier-facing requirement or evidence-bearing registration condition."
        if material_checkability == "direct"
        else (
            "Partially automatable from dossier structure/evidence, but likely needs reviewer interpretation or external context."
            if material_checkability == "partial"
            else "Regulation clause is better used as citation/background because it is not reliably decidable from dossier content alone."
        )
    )

    return {
        "ind_relevance": ind_relevance,
        "material_checkability": material_checkability,
        "recommended_rule_mode": recommended_rule_mode,
        "automation_ready": material_checkability == "direct",
        "candidate_reason": candidate_reason,
        "expected_material_evidence": evidence_types,
    }


def _build_clause_record(
    *,
    metadata: dict[str, Any],
    chapter_no_raw: str,
    chapter_no: int,
    chapter_title: str,
    article_no_raw: str,
    article_no: int,
    heading: str,
    normalized_text: str,
    line_index: int,
    clause_ref_suffix: str | None = None,
) -> dict[str, Any]:
    classification = _classify_clause(chapter_no, article_no, normalized_text)
    anchor_suffix = clause_ref_suffix or f"art_{article_no:03d}"
    clause_id = f"{metadata['regulation_id']}:{anchor_suffix}"
    severity = ""
    if metadata["regulation_id"] == "cn_ectd_validation_standard":
        for candidate in ("错误", "警告", "提示信息"):
            if str(normalized_text or "").strip().endswith(candidate):
                severity = candidate
                break
    return {
        "clause_id": clause_id,
        "regulation_id": metadata["regulation_id"],
        "regulation_title": metadata["title"],
        "version_label": metadata["version_label"],
        "chapter_no_raw": chapter_no_raw,
        "chapter_no": chapter_no,
        "chapter_title": chapter_title,
        "article_no_raw": article_no_raw,
        "article_no": article_no,
        "heading": heading,
        "original_text": normalized_text,
        "normalized_text": normalized_text,
        "clause_type": _infer_clause_type(normalized_text, classification["recommended_rule_mode"]),
        "source_path": metadata["source_path"],
        "source_filename": metadata["source_filename"],
        "source_locator": {
            "line_index": line_index,
            "citation_anchor": f"{metadata['regulation_id']}#{anchor_suffix}",
        },
        "severity": severity,
        "classification": classification,
    }


def _infer_clause_type(text: str, recommended_rule_mode: str) -> str:
    if "罚" in text or "法律责任" in text:
        return "penalty"
    if "不得" in text:
        return "prohibition"
    if "应当" in text:
        return "obligation"
    if "可以" in text and recommended_rule_mode == "citation_only":
        return "scope"
    if "定义" in text:
        return "definition"
    return "procedure"


def build_rule_candidates(clauses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for clause in clauses:
        classification = dict(clause.get("classification", {}) or {})
        article_no = int(clause.get("article_no", 0) or 0)
        citation_anchor = str(((clause.get("source_locator") or {}).get("citation_anchor")) or "")
        anchor_suffix = citation_anchor.split("#", 1)[1] if "#" in citation_anchor else f"art_{article_no:03d}"
        candidates.append(
            {
                "rule_candidate_id": (
                    f"{clause['regulation_id']}:rule_{article_no:03d}"
                    if anchor_suffix.startswith("art_")
                    else f"{clause['regulation_id']}:rule_{anchor_suffix}"
                ),
                "regulation_id": clause["regulation_id"],
                "clause_id": clause["clause_id"],
                "article_no": article_no,
                "title": clause["heading"],
                "recommended_rule_mode": classification.get("recommended_rule_mode"),
                "material_checkability": classification.get("material_checkability"),
                "ind_relevance": classification.get("ind_relevance"),
                "automation_ready": bool(classification.get("automation_ready", False)),
                "candidate_reason": classification.get("candidate_reason"),
                "expected_material_evidence": list(classification.get("expected_material_evidence", []) or []),
                "citation_anchor": citation_anchor,
                "source_path": clause["source_path"],
            }
        )
    return candidates


def build_direct_rule_drafts(
    regulation: dict[str, Any],
    rule_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    direct_candidates = [
        candidate
        for candidate in rule_candidates
        if str(candidate.get("material_checkability") or "") == "direct"
    ]
    draft_rules: list[dict[str, Any]] = []
    for candidate in direct_candidates:
        category = str(candidate.get("recommended_rule_mode") or "soft")
        article_no = int(candidate.get("article_no", 0) or 0)
        rule_id = f"DRAFT-{'HR' if category == 'hard' else 'SR'}-{regulation['regulation_id']}-{article_no:03d}"
        draft_rules.append(
            {
                "rule_id": rule_id,
                "status": "draft",
                "implementation_state": "planned",
                "category": category,
                "title": candidate.get("title"),
                "regulation_id": candidate.get("regulation_id"),
                "clause_id": candidate.get("clause_id"),
                "article_no": article_no,
                "material_checkability": candidate.get("material_checkability"),
                "ind_relevance": candidate.get("ind_relevance"),
                "expected_material_evidence": list(candidate.get("expected_material_evidence", []) or []),
                "candidate_reason": candidate.get("candidate_reason"),
                "citation": {
                    "anchor": candidate.get("citation_anchor"),
                    "source_path": candidate.get("source_path"),
                    "source_regulation": regulation.get("regulation_id"),
                    "source_version": regulation.get("version_label"),
                },
                "implementation_notes": (
                    "Promoted from direct dossier-checkable regulation clause. "
                    "Still needs explicit evaluator logic over material_review_contract evidence."
                ),
            }
        )

    mode_counts = {"hard": 0, "soft": 0}
    for rule in draft_rules:
        mode_counts[str(rule.get("category"))] += 1

    return {
        "schema_version": REGULATION_RULE_DRAFT_VERSION,
        "regulation_id": regulation.get("regulation_id"),
        "regulation_title": regulation.get("title"),
        "source_version": regulation.get("version_label"),
        "draft_rule_count": len(draft_rules),
        "category_counts": mode_counts,
        "rules": draft_rules,
    }


def _build_classification_summary(clauses: list[dict[str, Any]]) -> dict[str, Any]:
    checkability_counts = {"direct": 0, "partial": 0, "external_only": 0}
    mode_counts = {"hard": 0, "soft": 0, "review_only": 0, "citation_only": 0}
    for clause in clauses:
        classification = dict(clause.get("classification", {}) or {})
        checkability = str(classification.get("material_checkability") or "")
        mode = str(classification.get("recommended_rule_mode") or "")
        if checkability in checkability_counts:
            checkability_counts[checkability] += 1
        if mode in mode_counts:
            mode_counts[mode] += 1
    return {
        "material_checkability_counts": checkability_counts,
        "recommended_rule_mode_counts": mode_counts,
        "automation_ready_clause_count": checkability_counts["direct"],
    }


def _find_clause_with_all_phrases(
    clauses: list[dict[str, Any]],
    *phrases: str,
) -> dict[str, Any] | None:
    for clause in clauses:
        text = str(clause.get("normalized_text") or "")
        if all(phrase in text for phrase in phrases):
            return clause
    return None


def _find_clause_by_heading(
    clauses: list[dict[str, Any]],
    heading: str,
) -> dict[str, Any] | None:
    expected = str(heading or "").strip()
    for clause in clauses:
        if str(clause.get("heading") or "").strip() == expected:
            return clause
    return None


def _load_attachment_12_bundle() -> dict[str, Any]:
    if not _ECTD_ATTACHMENT_12_BUNDLE_PATH.exists():
        return {}
    try:
        payload = json.loads(_ECTD_ATTACHMENT_12_BUNDLE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if str(payload.get("bundle_id") or "").strip() != "cn_ectd_attachment_1_2":
        return {}
    return payload


def _build_requirement_record(
    *,
    regulation_id: str,
    clause: dict[str, Any],
    requirement_suffix: str,
    requirement_type: str,
    applicable_stage: str,
    requirement_level: str = "required",
    registration_classes: list[str] | None = None,
    requirement_text: str | None = None,
    expected_material_evidence: list[str] | None = None,
    review_focus: str | None = None,
) -> dict[str, Any]:
    clause_classification = dict(clause.get("classification", {}) or {})
    clause_evidence = list(clause_classification.get("expected_material_evidence", []) or [])
    if expected_material_evidence is None:
        expected_material_evidence = list(clause_evidence)
        if not expected_material_evidence:
            expected_material_evidence = ["general_registration_dossier"]
            if requirement_type == "reference_consistency":
                expected_material_evidence.append("reference_sample_materials")
            if requirement_type == "clinical_data_submission":
                expected_material_evidence.append("clinical_trial_materials")
    return {
        "requirement_id": f"{regulation_id}:{requirement_suffix}",
        "regulation_id": regulation_id,
        "source_clause_id": clause.get("clause_id"),
        "source_article_no": clause.get("article_no"),
        "section_no": clause.get("chapter_no"),
        "section_title": clause.get("chapter_title"),
        "source_heading": clause.get("heading"),
        "registration_classes": list(registration_classes or []),
        "applicable_stage": applicable_stage,
        "requirement_type": requirement_type,
        "requirement_level": requirement_level,
        "requirement_text": requirement_text or clause.get("normalized_text"),
        "expected_material_evidence": list(expected_material_evidence),
        "review_focus": review_focus,
        "citation_anchor": ((clause.get("source_locator") or {}).get("citation_anchor")),
        "source_path": clause.get("source_path"),
        "source_filename": clause.get("source_filename"),
    }


def build_requirement_matrix(
    regulation: dict[str, Any],
    clauses: list[dict[str, Any]],
) -> dict[str, Any]:
    regulation_id = str(regulation.get("regulation_id") or "")
    requirements: list[dict[str, Any]] = []

    if regulation_id == "cn_drug_registration_classification_and_dossier_requirements":
        class_1_clause = _find_clause_with_all_phrases(clauses, "化学药品1类", "创新药")
        class_2_clause = _find_clause_with_all_phrases(clauses, "化学药品2类", "明显临床优势")
        class_3_clause = _find_clause_with_all_phrases(clauses, "化学药品3类", "参比制剂")
        class_4_clause = _find_clause_with_all_phrases(clauses, "化学药品4类", "参比制剂")
        class_5_clause = _find_clause_with_all_phrases(clauses, "化学药品5类", "5.1类", "5.2类")
        ctd_clause = _find_clause_with_all_phrases(clauses, "CTD", "提交申报资料")
        electronic_db_clause = _find_clause_with_all_phrases(clauses, "电子临床试验数据库")

        if class_1_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=class_1_clause,
                    requirement_suffix="req_class_1_innovation",
                    registration_classes=["1类"],
                    applicable_stage="registration_classification",
                    requirement_type="classification_eligibility",
                    review_focus="确认申报材料能证明新结构明确、具有药理作用且具有临床价值。",
                )
            )
        if class_2_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=class_2_clause,
                    requirement_suffix="req_class_2_clinical_advantage",
                    registration_classes=["2类"],
                    applicable_stage="registration_classification",
                    requirement_type="classification_eligibility",
                    review_focus="确认优化依据充分，并能证明相对改良前具有明显临床优势。",
                )
            )
        if class_3_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=class_3_clause,
                    requirement_suffix="req_class_3_reference_consistency",
                    registration_classes=["3类"],
                    applicable_stage="marketing_registration",
                    requirement_type="reference_consistency",
                    review_focus="确认活性成份、剂型、规格、适应症、给药途径和用法用量与参比制剂一致，或已给出充分合理性研究说明。",
                )
            )
        if class_4_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=class_4_clause,
                    requirement_suffix="req_class_4_reference_consistency",
                    registration_classes=["4类"],
                    applicable_stage="marketing_registration",
                    requirement_type="reference_consistency",
                    review_focus="确认申报材料能证明与境内已上市原研药参比制剂质量和疗效一致。",
                )
            )
        if class_5_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=class_5_clause,
                    requirement_suffix="req_class_5_1_clinical_advantage",
                    registration_classes=["5.1类"],
                    applicable_stage="marketing_registration",
                    requirement_type="classification_eligibility",
                    requirement_text=(
                        "化学药品5.1类为原研药品和改良型药品，改良型药品在已知活性成份基础上进行优化，应比改良前具有明显临床优势。"
                    ),
                    review_focus="确认 5.1 类境外上市药品在境内申报时，改良型药品具备明显临床优势证明。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=class_5_clause,
                    requirement_suffix="req_class_5_2_reference_consistency",
                    registration_classes=["5.2类"],
                    applicable_stage="marketing_registration",
                    requirement_type="reference_consistency",
                    requirement_text=(
                        "化学药品5.2类为仿制药，应证明与参比制剂质量和疗效一致，技术要求与化学药品3类、4类相同。"
                    ),
                    review_focus="确认 5.2 类境外上市仿制药在境内申报时满足与参比制剂一致性的技术要求。",
                )
            )
        if ctd_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=ctd_clause,
                    requirement_suffix="req_ctd_base_submission",
                    registration_classes=["1类", "2类", "3类", "4类", "5.1类", "5.2类", "化学原料药"],
                    applicable_stage="clinical_and_marketing",
                    requirement_type="submission_structure",
                    review_focus="确认药物临床试验、上市注册和化学原料药申请均按 CTD 格式编号和项目顺序提交。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=ctd_clause,
                    requirement_suffix="req_research_follows_technical_guidance",
                    registration_classes=["1类", "2类", "3类", "4类", "5.1类", "5.2类", "化学原料药"],
                    applicable_stage="clinical_and_marketing",
                    requirement_type="research_governance",
                    review_focus="确认研究工作依据国家药品监管部门公布的相关技术指导原则开展。",
                )
            )
        if electronic_db_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=electronic_db_clause,
                    requirement_suffix="req_electronic_clinical_trial_database",
                    registration_classes=["1类", "2类", "3类", "4类", "5.1类", "5.2类"],
                    applicable_stage="marketing_registration",
                    requirement_type="clinical_data_submission",
                    review_focus="确认完成临床试验后的上市注册申请在 CTD 基础上补充电子临床试验数据库。",
                )
            )

    if regulation_id == "cn_ectd_technical_specification":
        application_number_clause = _find_clause_by_heading(clauses, "2.1.1 申请编号")
        application_type_clause = _find_clause_by_heading(clauses, "2.1.2 申请类型")
        product_type_clause = _find_clause_by_heading(clauses, "2.1.3 产品类型")
        regulatory_activity_type_clause = _find_clause_by_heading(clauses, "2.2.1 注册行为类型")
        sequence_number_clause = _find_clause_by_heading(clauses, "2.3.1 序列号")
        sequence_type_clause = _find_clause_by_heading(clauses, "2.3.2 序列类型")
        sequence_description_clause = _find_clause_by_heading(clauses, "2.3.3 序列描述")
        type_compatibility_clause = _find_clause_by_heading(clauses, "2.4 申请、注册行为和序列的关系")
        filename_rule_clause = _find_clause_by_heading(clauses, "3.3.2 文件和文件夹命名规则")
        pdf_navigation_clause = _find_clause_by_heading(clauses, "3.4 PDF 电子提交标准")
        envelope_clause = _find_clause_by_heading(clauses, "4.3 信封元素")
        checksum_clause = _find_clause_by_heading(clauses, "4.4 目录元素")
        application_information_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_2_1")
            ),
            None,
        )
        supporting_file_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_1_3")
            ),
            None,
        )
        cn_regional_root_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_4_2")
            ),
            None,
        )
        module1_backbone_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_4_1")
            ),
            None,
        )
        regulatory_activity_information_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_2_2")
            ),
            None,
        )
        related_sequence_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_2_2_2")
            ),
            None,
        )
        sequence_contact_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_2_3_4")
            ),
            None,
        )
        sequence_information_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_2_3")
            ),
            None,
        )
        node_extension_32r_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_2")
            ),
            None,
        )
        node_extension_scope_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_7")
            ),
            None,
        )
        file_and_folder_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_3")
            ),
            None,
        )
        leaf_xml_lang_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_5_1")
            ),
            None,
        )
        foreign_reference_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_5")
            ),
            None,
        )
        replace_language_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_5_2")
            ),
            None,
        )
        content_file_format_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_3_1")
            ),
            None,
        )
        empty_placeholder_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_3_3")
            ),
            None,
        )
        file_reuse_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_3_4")
            ),
            None,
        )
        stf_required_zone_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_8")
            ),
            None,
        )
        lifecycle_operation_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_9")
            ),
            None,
        )
        attachment_12_bundle = _load_attachment_12_bundle()
        attachment_12_supporting_files = {
            str(item.get("filename") or "").strip()
            for item in attachment_12_bundle.get("controlled_vocabularies", []) or []
            if str(item.get("filename") or "").strip()
        }
        dependency_filename = str(
            ((attachment_12_bundle.get("dependency_matrix") or {}).get("filename")) or ""
        ).strip()
        if dependency_filename:
            attachment_12_supporting_files.add(dependency_filename)

        if application_number_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=application_number_clause,
                    requirement_suffix="req_application_number_format",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="identifier_format",
                    requirement_text=(
                        "首次提交临床试验申请、新药申请或仿制药申请的 eCTD 序列时，应获取申请编号；"
                        "申请编号编码规则为字母（x=新药申请；y=仿制药申请；l=临床试验申请）+4 位年份+5 位流水号。"
                    ),
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对信封信息中的申请编号是否存在、是否符合规范格式，并且首字母是否与申请类型语义一致。",
                )
            )

        if application_information_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=application_information_clause,
                    requirement_suffix="req_application_information_core_completeness",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="application_information_core_completeness",
                    requirement_level="warning",
                    requirement_text=(
                        "eCTD application information should provide complete core application-number, "
                        "application-type, and product-type metadata when local application project "
                        "context is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "cn_regional_xml_envelope_metadata",
                        "application_project_context",
                    ],
                    review_focus=(
                        "Bounded executable subset: surface missing, invalid, or inconsistent "
                        "application-number, application-type, and product-type values from local "
                        "application project context as warning-level evidence; 2.1.4 original-number "
                        "semantics remain review-only/prerequisite-dependent and are not hard-enforced "
                        "by this aggregate requirement."
                    ),
                )
            )

        if sequence_number_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=sequence_number_clause,
                    requirement_suffix="req_sequence_number_progression",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="sequence_numbering",
                    requirement_text=(
                        "序列号应为申请中唯一的 4 位数字字符串，并从 0000 开始，每次提交加 1，按先后次序提交，不得跳号。"
                    ),
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对序列号格式、起始值与连续性。",
                )
            )

        if application_type_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=application_type_clause,
                    requirement_suffix="req_application_type_controlled_vocabulary_validity",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="controlled_vocabulary",
                    requirement_text="申请类型应使用受控词汇文件 cv-application-type.xml 中定义的有效代码名称。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对 application-type 是否取自 cv-application-type.xml 的有效代码集合。",
                )
            )

        if product_type_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=product_type_clause,
                    requirement_suffix="req_product_type_controlled_vocabulary_validity",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="controlled_vocabulary",
                    requirement_text="产品类型应使用受控词汇文件 cv-product-type.xml 中定义的有效代码名称。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对 product-type 是否取自 cv-product-type.xml 的有效代码集合。",
                )
            )

        if regulatory_activity_type_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=regulatory_activity_type_clause,
                    requirement_suffix="req_regulatory_activity_type_controlled_vocabulary_validity",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="controlled_vocabulary",
                    requirement_text="注册行为类型应使用受控词汇文件 cv-regulatory-activity-type.xml 中定义的有效代码名称。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对 regulatory-activity-type 是否取自 cv-regulatory-activity-type.xml 的有效代码集合。",
                )
            )

        if regulatory_activity_information_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=regulatory_activity_information_clause,
                    requirement_suffix="req_regulatory_activity_information_completeness",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="regulatory_activity_information_completeness",
                    requirement_level="warning",
                    requirement_text=(
                        "eCTD regulatory activity information should provide "
                        "regulatory-activity-type and related-sequence values when local "
                        "regulatory activity metadata is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "regulatory_activity_context",
                        "cn_regional_xml_envelope_metadata",
                        "same_regulatory_activity_sequence_grouping",
                    ],
                    review_focus=(
                        "Bounded executable subset: aggregate local regulatory activity context "
                        "and cn-regional.xml envelope metadata for regulatory-activity-type and "
                        "related-sequence presence/format/order evidence within the same regulatory "
                        "activity; detailed child requirements remain separately traced to "
                        "sec_2_2_1 and sec_2_2_2, related-sequence is same regulatory activity "
                        "grouping evidence and not a generic previous-sequence pointer, and missing "
                        "activity metadata remains na/prerequisite guidance rather than assumed "
                        "pass/fail."
                    ),
                )
            )

        if sequence_type_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=sequence_type_clause,
                    requirement_suffix="req_sequence_type_controlled_vocabulary_validity",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="controlled_vocabulary",
                    requirement_text="序列类型应使用受控词汇文件 cv-sequence-type.xml 中定义的有效代码名称。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对 sequence-type 是否取自 cv-sequence-type.xml 的有效代码集合。",
                )
            )

        if sequence_description_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=sequence_description_clause,
                    requirement_suffix="req_sequence_description_length",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="sequence_metadata",
                    requirement_text="序列描述应为区分提交目的的简要描述，长度应在 120 个中文字符以内。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对序列描述长度是否超限。",
                )
            )

        if sequence_contact_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=sequence_contact_clause,
                    requirement_suffix="req_sequence_contact_information_presence",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="sequence_contact_information_presence",
                    requirement_text=(
                        "The cn-regional.xml sequence-contact record should provide complete contact "
                        "name, phone, and email values when eCTD envelope metadata is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "cn_regional_xml_envelope_metadata",
                        "cn_regional_xml_sequence_contact",
                    ],
                    review_focus=(
                        "Bounded executable subset: verify local cn-regional.xml sequence-contact "
                        "name/phone/email presence from parsed envelope metadata; envelope metadata "
                        "prerequisite gaps remain na/guidance, and internal cn-contact structure "
                        "alignment is traced separately from this presence requirement."
                    ),
                )
            )

        if type_compatibility_clause is not None:
            review_focus = "核对申请类型、注册行为类型和序列类型的组合是否符合 depend-apt-rat-sqt.xml 中定义的兼容矩阵。"
            if attachment_12_supporting_files:
                review_focus = (
                    review_focus
                    + " 当前项目已具备 supporting files: "
                    + ", ".join(sorted(attachment_12_supporting_files))
                    + "。"
                )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=type_compatibility_clause,
                    requirement_suffix="req_application_registration_sequence_type_compatibility",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="type_compatibility",
                    requirement_text="申请类型、注册行为类型和序列类型的组合应符合 depend-apt-rat-sqt.xml 定义的对应关系。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus=review_focus,
                )
            )

        if filename_rule_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=filename_rule_clause,
                    requirement_suffix="req_file_name_character_constraints",
                    applicable_stage="ectd_packaging",
                    requirement_type="file_naming",
                    requirement_text=(
                        "eCTD 文件及文件夹命名仅允许使用小写字母 a-z、数字 0-9、中划线 - 和下划线 _；"
                        "从序列文件夹开始的完整路径长度不应超过 180 个字符，单个文件或文件夹名称长度不应超过 64 个字符。"
                    ),
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对文件名字符集与路径长度是否符合命名规则。",
                )
            )

        if pdf_navigation_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=pdf_navigation_clause,
                    requirement_suffix="req_long_pdf_navigation_aids",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="pdf_navigation",
                    requirement_text=(
                        "除外文参考资料、参考文献和申请表外，单个 PDF 文件内容超过 5 页时，应提供目录和书签辅助导航。"
                    ),
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对长 PDF 是否包含目录/书签导航结构。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=pdf_navigation_clause,
                    requirement_suffix="req_hyperlink_navigation_support",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="pdf_hyperlink_navigation",
                    requirement_text="在不能使用目录和书签进行文档导航的情况下，可使用超文本链接来辅助定位。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对长 PDF 在无显式目录/书签时是否还保留可消费的文档内或跨文件超文本导航链接。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=pdf_navigation_clause,
                    requirement_suffix="req_no_cross_application_pdf_hyperlinks",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="pdf_hyperlink_integrity",
                    requirement_text="申报资料中不应创建跨申请超文本链接。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对 PDF 外部文件超链接在可解析时是否指向同一申请内的文件，而非另一申请根目录下的文件。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=pdf_navigation_clause,
                    requirement_suffix="req_pdf_hyperlink_target_file_presence",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="pdf_hyperlink_target_integrity",
                    requirement_text="同一申请内可解析的 PDF 外部文件超链接目标应指向真实存在的文件，以避免链接失效。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对同一申请内、可解析到文件级路径的 PDF 外部文件超链接目标是否真实存在。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=pdf_navigation_clause,
                    requirement_suffix="req_pdf_hyperlink_target_resolvability",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="pdf_hyperlink_target_resolvability",
                    requirement_text="PDF 外部文件超链接目标应保持可解析，避免因目标路径无法解析而导致链接失效或定位错误。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对 PDF 外部文件超链接目标是否能从当前包内路径上下文被稳定解析。",
                )
            )

        metadata_attribute_clause = next(
            (
                clause
                for clause in clauses
                if str(clause.get("clause_id") or "").strip().endswith(":sec_3_6")
            ),
            None,
        )
        if supporting_file_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=supporting_file_clause,
                    requirement_suffix="req_index_dtd_reference_points_to_util_dtd",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="index_dtd_reference",
                    requirement_text=(
                        "The index.xml DOCTYPE SYSTEM reference must point to the local "
                        "util/dtd/ich-ectd-3-2.dtd file when index.xml package evidence is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "index_xml_doctype_metadata",
                        "package_supporting_file_inventory",
                    ],
                    review_focus=(
                        "Bounded executable subset: verify the local package evidence for index.xml "
                        "DOCTYPE SYSTEM reference and resolved util/dtd/ich-ectd-3-2.dtd availability; "
                        "broader supporting-file dependencies such as schema, stylesheet, STF, and "
                        "controlled-vocabulary resources remain separately traced or prerequisite scope."
                    ),
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=supporting_file_clause,
                    requirement_suffix="req_index_xml_valid_against_ich_dtd",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="index_xml_dtd_validity",
                    requirement_level="warning",
                    requirement_text=(
                        "The index.xml document should be well-formed and valid against the resolved "
                        "local ich-ectd-3-2.dtd when the required DTD validation prerequisite is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "index_xml_parse_diagnostics",
                        "index_xml_dtd_validation_diagnostics",
                    ],
                    review_focus=(
                        "Bounded executable subset: surface index.xml well-formedness and local DTD "
                        "validation diagnostics; a resolvable local DTD is a prerequisite for "
                        "deterministic validity judgment, and missing DTD validation prerequisites "
                        "should remain na/human-review guidance rather than assumed pass/fail."
                    ),
                )
            )

        if node_extension_32r_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=node_extension_32r_clause,
                    requirement_suffix="req_32r_node_extension_structure_and_title_compliance",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="node_extension_structure_and_title",
                    requirement_level="warning",
                    requirement_text=(
                        "3.2.R node-extension records in local index.xml should use allowed 3.2.R.1 "
                        "through 3.2.R.6 titles, remain under the regional information parent structure, "
                        "and reference leaf files with the expected m3/32-body-data/32r-* path pattern."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "index_xml_node_extension_records",
                        "index_xml_leaf_records",
                    ],
                    review_focus=(
                        "Bounded executable subset: surface local index.xml 3.2.R node-extension parent "
                        "placement, allowed title, and leaf href path-pattern issues as warning-level "
                        "evidence; do not infer full reviewer-facing content-category adequacy from "
                        "title/path evidence alone."
                    ),
                )
            )

        if node_extension_scope_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=node_extension_scope_clause,
                    requirement_suffix="req_node_extension_scope_boundary",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="node_extension_scope_boundary",
                    requirement_level="warning",
                    requirement_text=(
                        "eCTD node-extension usage should remain within the biologic-only 3.2.R "
                        "regional-information scope when local node-extension and product-type evidence "
                        "are available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "index_xml_node_extension_records",
                        "cn_regional_xml_node_extension_records",
                        "cn_regional_xml_envelope_metadata",
                    ],
                    review_focus=(
                        "Bounded executable subset: surface node-extension usage outside biologic "
                        "product-type evidence or outside 3.2.R scope from local XML metadata; missing "
                        "product-type evidence remains prerequisite guidance rather than false hard "
                        "failure, and validation-standard HR-ECTD-074 hard-fail behavior remains traced "
                        "to validation-standard sec_3_16."
                    ),
                )
            )

        if sequence_information_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=sequence_information_clause,
                    requirement_suffix="req_sequence_information_completeness",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="sequence_information_completeness",
                    requirement_level="warning",
                    requirement_text=(
                        "eCTD sequence information should provide complete sequence-number, "
                        "sequence-type, sequence-description, and sequence-contact name, phone, and "
                        "email values when local sequence package metadata is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "sequence_package_context",
                        "cn_regional_xml_envelope_metadata",
                        "cn_regional_xml_sequence_contact",
                    ],
                    review_focus=(
                        "Bounded executable subset: aggregate local sequence package context and "
                        "cn-regional.xml envelope metadata for sequence-number, sequence-type, "
                        "sequence-description, and sequence-contact presence; detailed child "
                        "requirements remain separately traced to sec_2_3_1, sec_2_3_2, "
                        "sec_2_3_3, and sec_2_3_4, while missing sequence metadata remains "
                        "na/prerequisite guidance rather than assumed pass/fail."
                    ),
                )
            )

        if leaf_xml_lang_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=leaf_xml_lang_clause,
                    requirement_suffix="req_leaf_xml_lang_attribute_classification",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="leaf_xml_lang_attribute_classification",
                    requirement_text=(
                        "eCTD leaf xml:lang values should support the 3.5.1 language classification "
                        "boundary: zh and missing/empty xml:lang are treated as Chinese dossier leafs, "
                        "and non-zh values should be valid ISO639-1 language codes for foreign reference "
                        "leaf classification."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "index_xml_leaf_records",
                        "cn_regional_xml_leaf_records",
                    ],
                    review_focus=(
                        "Bounded executable subset: classify local leaf metadata by xml:lang value for "
                        "language classification and surface invalid non-zh ISO639-1 codes; do not infer "
                        "substantive content-language adequacy from filename, path, or document content "
                        "evidence."
                    ),
                )
            )

        if foreign_reference_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=foreign_reference_clause,
                    requirement_suffix="req_foreign_reference_leaf_sibling_structure",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="foreign_reference_leaf_sibling_structure",
                    requirement_text=(
                        "foreign-reference leafs should be colocated with a same-parent Chinese dossier leaf "
                        "and ordered after that Chinese dossier leaf when local eCTD leaf structure metadata "
                        "is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "index_xml_leaf_records",
                        "cn_regional_xml_leaf_records",
                    ],
                    review_focus=(
                        "Bounded executable subset: use local leaf metadata to verify same-parent Chinese "
                        "dossier sibling presence and ordering for foreign-reference leafs; do not infer "
                        "substantive content-language adequacy from filename, path, or document text, and "
                        "leave missing leaf structure metadata as na/prerequisite guidance."
                    ),
                )
            )

        if replace_language_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=replace_language_clause,
                    requirement_suffix="req_replace_leaf_language_class_consistency",
                    applicable_stage="ectd_lifecycle_management",
                    requirement_type="replace_leaf_language_class_consistency",
                    requirement_text=(
                        "A replace-operation eCTD leaf should preserve the Chinese-dossier or foreign-reference "
                        "language class of the uniquely matched prior-sequence leaf in the same application."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "current_sequence_index_xml_leaf_records",
                        "prior_sequence_index_xml_leaf_records",
                        "current_sequence_cn_regional_xml_leaf_records",
                        "prior_sequence_cn_regional_xml_leaf_records",
                    ],
                    review_focus=(
                        "Bounded executable subset: compare replace leaf language class only when a unique "
                        "prior-sequence leaf match exists in the same application; missing history, ambiguous "
                        "matches, or unavailable leaf metadata remain na/prerequisite guidance, and this does "
                        "not infer substantive content-language adequacy from filename, path, or document text."
                    ),
                )
            )

        if file_and_folder_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=file_and_folder_clause,
                    requirement_suffix="req_file_and_folder_packaging_boundaries",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="file_and_folder_packaging_rollup",
                    requirement_level="warning",
                    requirement_text=(
                        "Section 3.3 file and folder packaging is traced through bounded child "
                        "requirements for allowed content-file formats, file and folder naming "
                        "constraints, empty directories, placeholder documents, same-application "
                        "file reuse guidance, and cross-application reference prohibition."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "index_xml_leaf_records",
                        "cn_regional_xml_leaf_records",
                        "sequence_package_path",
                        "local_directory_inventory",
                        "local_parsed_document_text",
                    ],
                    review_focus=(
                        "This parent traceability rollup links section 3.3 to already mapped child "
                        "requirements and does not create new runtime verdicts; broad packaging "
                        "adequacy, substantive content adequacy, and advisory reuse optimization "
                        "remain bounded by child-rule prerequisites, na guidance, and human review."
                    ),
                )
            )

        if content_file_format_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=content_file_format_clause,
                    requirement_suffix="req_content_file_format_allowed",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="content_file_format",
                    requirement_text=(
                        "eCTD content files referenced by local leaf href values should use allowed "
                        "PDF, XML, XPT, TXT, or XSL file formats."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "index_xml_leaf_records",
                        "cn_regional_xml_leaf_records",
                    ],
                    review_focus=(
                        "Bounded executable subset: verify local leaf href filename extensions against "
                        "the allowed PDF/XML/XPT/TXT/XSL set; file readability, substantive content "
                        "adequacy, and whether a file belongs in a dossier section remain separate "
                        "parser/review concerns."
                    ),
                )
            )

        if empty_placeholder_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=empty_placeholder_clause,
                    requirement_suffix="req_no_empty_directories",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="empty_directory_policy",
                    requirement_text=(
                        "eCTD sequence packages should not contain empty directories except explicit "
                        "utility or scaffold contexts that are exempt from local empty-directory checks."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "sequence_package_path",
                        "local_directory_inventory",
                    ],
                    review_focus=(
                        "Bounded executable subset: enumerate the local sequence package directory "
                        "inventory and report non-scaffold empty directories; do not infer missing "
                        "dossier-section adequacy or substantive submission completeness from directory "
                        "absence alone."
                    ),
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=empty_placeholder_clause,
                    requirement_suffix="req_no_placeholder_documents",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="placeholder_document_policy",
                    requirement_text=(
                        "Declared PDF content targets should not be placeholder documents that only "
                        "indicate not applicable, N/A, or similar marker text instead of actual content."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "index_xml_leaf_records",
                        "cn_regional_xml_leaf_records",
                        "local_parsed_document_text",
                    ],
                    review_focus=(
                        "Bounded executable subset: use local parsed document text and declared leaf "
                        "href evidence to identify obvious placeholder PDFs; when cn-regional.xml, "
                        "index.xml, declared PDF targets, or parsed text prerequisites are unavailable, "
                        "return prerequisite guidance instead of judging legitimate short documents as "
                        "placeholders."
                    ),
                )
            )

        if file_reuse_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=file_reuse_clause,
                    requirement_suffix="req_duplicate_entity_file_submission_within_sequence_warning",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="duplicate_entity_file_reuse_warning",
                    requirement_level="warning",
                    requirement_text=(
                        "When the same eCTD application needs to submit files with the same checksum, "
                        "the package should prefer leaf-based file reuse instead of resubmitting "
                        "duplicate entity files."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "sequence_package_context",
                        "index_xml_leaf_records",
                        "cn_regional_xml_leaf_records",
                    ],
                    review_focus=(
                        "Bounded advisory subset: compare local leaf checksum and href evidence within "
                        "the current sequence and available same-application prior sequences to surface "
                        "obvious duplicate entity-file submissions as reuse guidance; this is advisory "
                        "and not a hard invalidation of the dossier."
                    ),
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=file_reuse_clause,
                    requirement_suffix="req_no_cross_application_leaf_reference",
                    applicable_stage="ectd_document_packaging",
                    requirement_type="cross_application_reference_prohibition",
                    requirement_text=(
                        "eCTD leaf href values must not use cross-application references to files "
                        "submitted under another application."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "sequence_package_path",
                        "index_xml_leaf_records",
                        "cn_regional_xml_leaf_records",
                    ],
                    review_focus=(
                        "Bounded executable subset: resolve local leaf href paths against the current "
                        "sequence package and compare the resolved target application root with the "
                        "current local application root; missing href metadata or sequence-root "
                        "context remains prerequisite guidance."
                    ),
                )
            )

        if metadata_attribute_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=metadata_attribute_clause,
                    requirement_suffix="req_ich_indication_attribute_required",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="ich_backbone_attribute_metadata",
                    requirement_text="ICH eCTD backbone section elements that require an indication attribute must provide a non-empty indication value when the relevant section metadata is locally available.",
                    expected_material_evidence=["general_registration_dossier", "index_xml_attribute_records"],
                    review_focus="Bounded executable subset: validate required/non-empty indication attributes for locally parsed index.xml section metadata; broader metadata update lifecycle coupling remains prerequisite-dependent.",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=metadata_attribute_clause,
                    requirement_suffix="req_ich_manufacturer_attribute_required",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="ich_backbone_attribute_metadata",
                    requirement_text="ICH eCTD backbone section elements that require a manufacturer attribute must provide a non-empty manufacturer value when the relevant section metadata is locally available.",
                    expected_material_evidence=["general_registration_dossier", "index_xml_attribute_records"],
                    review_focus="Bounded executable subset: validate required/non-empty manufacturer attributes for locally parsed index.xml section metadata; broader metadata update lifecycle coupling remains prerequisite-dependent.",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=metadata_attribute_clause,
                    requirement_suffix="req_ich_substance_attribute_required",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="ich_backbone_attribute_metadata",
                    requirement_text="ICH eCTD backbone section elements that require a substance attribute must provide a non-empty substance value when the relevant section metadata is locally available.",
                    expected_material_evidence=["general_registration_dossier", "index_xml_attribute_records"],
                    review_focus="Bounded executable subset: validate required/non-empty substance attributes for locally parsed index.xml section metadata; broader metadata update lifecycle coupling remains prerequisite-dependent.",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=metadata_attribute_clause,
                    requirement_suffix="req_ich_attribute_edge_whitespace_warning",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="ich_backbone_attribute_metadata_warning",
                    requirement_level="warning",
                    requirement_text="ICH eCTD backbone metadata attribute values should not contain leading or trailing whitespace when raw XML attribute values are locally available.",
                    expected_material_evidence=["general_registration_dossier", "index_xml_attribute_records"],
                    review_focus="Bounded executable subset: surface leading/trailing whitespace in locally parsed XML attribute values as warning-level evidence; do not infer broader metadata lifecycle adequacy from whitespace alone.",
                )
            )

        if stf_required_zone_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=stf_required_zone_clause,
                    requirement_suffix="req_stf_required_zone_structure",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="stf_required_zone_structure",
                    requirement_text=(
                        "Leaves in Module 4 section 4.2 and Module 5 sections 5.3.1 through 5.3.5 "
                        "should be organized under explicit STF parent structure when locally parsed "
                        "index.xml leaf and ancestor context is available."
                    ),
                    expected_material_evidence=["general_registration_dossier", "index_xml_leaf_records"],
                    review_focus=(
                        "Bounded executable subset: verify STF-required-zone parent structure from local "
                        "index.xml leaf records; full tag-level adequacy remains dependent on the external "
                        "STF specification and should stay prerequisite/human-review guidance."
                    ),
                )
            )

        if lifecycle_operation_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=lifecycle_operation_clause,
                    requirement_suffix="req_allowed_lifecycle_operation_values",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="lifecycle_operation_values",
                    requirement_text=(
                        "Locally parsed eCTD leaf lifecycle operation attributes must use one of the "
                        "recognized values: new, replace, delete, or append."
                    ),
                    expected_material_evidence=["general_registration_dossier", "index_xml_leaf_records"],
                    review_focus=(
                        "Bounded executable subset: validate allowed operation values from local leaf "
                        "metadata; broader lifecycle graph semantics still depend on comparable prior "
                        "sequence evidence and must remain prerequisite-guided."
                    ),
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=lifecycle_operation_clause,
                    requirement_suffix="req_non_stf_append_warning",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="non_stf_append_warning",
                    requirement_level="warning",
                    requirement_text=(
                        "Append lifecycle operations outside an explicit STF scope should be surfaced "
                        "as warning-level reviewer guidance when local leaf context is available."
                    ),
                    expected_material_evidence=["general_registration_dossier", "index_xml_leaf_records"],
                    review_focus=(
                        "Bounded executable subset: surface non-STF append usage from local leaf and "
                        "ancestor-context metadata; explanation-letter adequacy and broader lifecycle "
                        "history remain reviewer-facing or prerequisite-dependent."
                    ),
                )
            )

        if cn_regional_root_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=cn_regional_root_clause,
                    requirement_suffix="req_cn_regional_xml_root_element",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="xml_root_element",
                    requirement_text=(
                        "The cn-regional.xml root element must be the expected cn_ectd root element "
                        "when locally parsed XML evidence is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "cn_regional_xml_root_element",
                    ],
                    review_focus=(
                        "Bounded executable subset: verify root element exactness from local parsed "
                        "cn-regional.xml evidence; do not infer broader namespace declaration adequacy "
                        "beyond objective XML evidence."
                    ),
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=cn_regional_root_clause,
                    requirement_suffix="req_cn_regional_xml_schema_version",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="xml_schema_version",
                    requirement_text=(
                        "The cn-regional.xml root schema-version attribute must match the supported "
                        "regional schema version when local root attribute evidence is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "cn_regional_xml_root_attributes",
                        "attachment_1_1_schema_metadata",
                    ],
                    review_focus=(
                        "Bounded executable subset: verify schema-version exactness from local parsed "
                        "cn-regional.xml root attributes and known Attachment 1-1 schema metadata; "
                        "namespace declaration prose remains outside hard enforcement."
                    ),
                )
            )

        if module1_backbone_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=module1_backbone_clause,
                    requirement_suffix="req_module1_package_backbone_composition",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="module1_package_backbone_composition",
                    requirement_level="warning",
                    requirement_text=(
                        "module-1 package backbone evidence should show the expected application and "
                        "sequence roots, index.xml and cn-regional.xml backbone files, and cn-regional "
                        "root, envelope, and content composition when local package evidence is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "sequence_package_inventory",
                        "index_xml_parse_diagnostics",
                        "cn_regional_xml_parse_diagnostics",
                        "cn_regional_xml_envelope_metadata",
                        "cn_regional_xml_content_structure",
                    ],
                    review_focus=(
                        "Bounded executable subset: aggregate existing local package evidence for core "
                        "directory/backbone presence, application/sequence identity, and cn-regional root, "
                        "envelope, and content structure; detailed XML root, envelope, and leaf integrity "
                        "requirements remain separately traced to sec_4_2, sec_4_3, and sec_4_4, while "
                        "missing XML/package prerequisites remain na/prerequisite guidance rather than "
                        "assumed pass/fail."
                    ),
                )
            )

        if related_sequence_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=related_sequence_clause,
                    requirement_suffix="req_related_sequence_reference",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="related_sequence_reference",
                    requirement_level="warning",
                    requirement_text=(
                        "The cn-regional.xml related-sequence value should be a valid four-digit "
                        "same-regulatory-activity reference, not after the current sequence, "
                        "and should self-reference for initial sequence submissions when local envelope "
                        "metadata is available."
                    ),
                    expected_material_evidence=[
                        "general_registration_dossier",
                        "cn_regional_xml_envelope_metadata",
                    ],
                    review_focus=(
                        "Bounded executable subset: surface local related-sequence format/order and "
                        "initial-sequence self-reference issues as warning-level evidence for same "
                        "regulatory activity grouping; related-sequence is not a generic "
                        "previous-sequence pointer and full history-hard exactness remains "
                        "prerequisite/human-review scope."
                    ),
                )
            )

        if envelope_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=envelope_clause,
                    requirement_suffix="req_envelope_attributes_required",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="envelope_metadata",
                    requirement_text="申请人提交的每个序列中，信封元素所有属性均为必填项，且有且仅有一个值。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对信封元素是否缺少必填属性或存在多值。",
                )
            )

            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=envelope_clause,
                    requirement_suffix="req_application_level_envelope_immutability",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="cross_sequence_envelope_immutability",
                    requirement_text="同一申请中，申请编号、申请类型、产品类型和原始编号等申请级别信封元素属性信息不应更新。",
                    expected_material_evidence=["general_registration_dossier", "prior_sequence_history"],
                    review_focus="仅在本地同一申请序列历史可比较时，核对非初始序列是否保持 application-number、application-type、product-type 和 original-number 不变；缺少历史或字段证据时应返回前置条件提示而非硬判。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=envelope_clause,
                    requirement_suffix="req_regulatory_activity_level_envelope_immutability",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="cross_sequence_envelope_immutability",
                    requirement_text="同一注册行为中，相关序列和注册行为类型等注册行为级别信封元素属性信息不应更新。",
                    expected_material_evidence=["general_registration_dossier", "same_regulatory_activity_sequence_history"],
                    review_focus="仅在 related-sequence 能作为同一注册行为分组证据且本地序列历史可比较时，核对 regulatory-activity-type 是否保持不变；related-sequence 不得被解释为通用前序序列。",
                )
            )

        if checksum_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=checksum_clause,
                    requirement_suffix="req_leaf_checksum_md5",
                    applicable_stage="ectd_packaging",
                    requirement_type="checksum_rule",
                    requirement_text="目录叶元素中的 checksum-type 属性值必须设置为 MD5 或 md5。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对叶元素校验和算法是否为 MD5。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=checksum_clause,
                    requirement_suffix="req_leaf_href_resolves_to_present_file",
                    applicable_stage="ectd_packaging",
                    requirement_type="file_reference_integrity",
                    requirement_text="目录叶元素中的 xlink:href 应能解析到当前 eCTD 提交包中真实存在的文件。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对叶元素引用地址是否能解析到已提交文件。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=checksum_clause,
                    requirement_suffix="req_package_xml_envelope_consistency",
                    applicable_stage="ectd_packaging",
                    requirement_type="xml_backbone_consistency",
                    requirement_text="cn-regional.xml 与 index.xml 的核心 envelope 字段应保持一致。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对包内多个 XML 主干文件的申请号、序列号和相关序列是否一致。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=checksum_clause,
                    requirement_suffix="req_package_xml_leaf_set_consistency",
                    applicable_stage="ectd_packaging",
                    requirement_type="xml_backbone_consistency",
                    requirement_text="cn-regional.xml 与 index.xml 的 leaf 引用集合应保持一致。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对包内多个 XML 主干文件引用的叶节点文件集合是否一致。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=checksum_clause,
                    requirement_suffix="req_package_xml_leaf_checksum_consistency",
                    applicable_stage="ectd_packaging",
                    requirement_type="xml_backbone_consistency",
                    requirement_text="cn-regional.xml 与 index.xml 中相同 leaf href 的 checksum 与 checksum-type 应保持一致。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对包内多个 XML 主干文件对同一 leaf 的校验值描述是否一致。",
                )
            )
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=checksum_clause,
                    requirement_suffix="req_package_declared_file_coverage",
                    applicable_stage="ectd_packaging",
                    requirement_type="xml_backbone_completeness",
                    requirement_text="当前 eCTD 提交包中的实际文件应被 XML 主干文件的 leaf 引用完整覆盖。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对包内已提交文件是否都存在对应的 XML leaf 声明。",
                )
            )
        envelope_clause = _find_clause_by_heading(clauses, "4.3 信封元素")
        if envelope_clause is not None:
            requirements.append(
                _build_requirement_record(
                    regulation_id=regulation_id,
                    clause=envelope_clause,
                    requirement_suffix="req_envelope_controlled_vocabulary_validity",
                    applicable_stage="ectd_sequence_submission",
                    requirement_type="controlled_vocabulary",
                    requirement_text="信封元素中使用受控词汇的属性值应为受控词汇文件中定义的代码名称。",
                    expected_material_evidence=["general_registration_dossier"],
                    review_focus="核对 application-type、product-type 等信封属性值是否出自对应受控词汇文件。",
                )
            )

    return {
        "schema_version": REGULATION_LIBRARY_VERSION,
        "matrix_schema_version": REGULATION_REQUIREMENT_MATRIX_VERSION,
        "regulation_id": regulation_id,
        "regulation_title": regulation.get("title"),
        "requirement_count": len(requirements),
        "requirements": requirements,
    }


def build_glossary(
    regulation: dict[str, Any],
    chapters: list[dict[str, Any]],
) -> dict[str, Any]:
    regulation_id = str(regulation.get("regulation_id") or "").strip()
    glossary_chapter = next(
        (
            chapter
            for chapter in chapters
            if str(chapter.get("chapter_title") or "").strip() == "术语表"
        ),
        None,
    )

    if regulation_id != "cn_ectd_technical_specification" or glossary_chapter is None:
        return {
            "schema_version": REGULATION_LIBRARY_VERSION,
            "glossary_schema_version": REGULATION_GLOSSARY_VERSION,
            "regulation_id": regulation_id,
            "regulation_title": regulation.get("title"),
            "term_count": 0,
            "terms": [],
        }

    chapter_id = str(glossary_chapter.get("chapter_id") or "").strip()
    source_heading = str(glossary_chapter.get("heading") or "").strip()
    source_path = str(regulation.get("source_path") or "").strip()
    source_filename = str(regulation.get("source_filename") or "").strip()
    terms: list[dict[str, Any]] = []
    for index, (term, definition) in enumerate(_ECTD_TECHNICAL_SPEC_GLOSSARY_TERMS, start=1):
        term_id = f"{regulation_id}:glossary_{index:02d}"
        terms.append(
            {
                "term_id": term_id,
                "term_no": index,
                "term": term,
                "definition": definition,
                "source_clause_id": chapter_id,
                "source_heading": source_heading,
                "citation_anchor": f"{regulation_id}#ch_06_term_{index:02d}",
                "source_path": source_path,
                "source_filename": source_filename,
            }
        )
    for offset, supplemental_term in enumerate(_ECTD_TECHNICAL_SPEC_SUPPLEMENTAL_GLOSSARY_TERMS, start=1):
        index = len(_ECTD_TECHNICAL_SPEC_GLOSSARY_TERMS) + offset
        supplemental_source_filename = str(supplemental_term.get("source_filename") or "").strip()
        supplemental_source_path = (
            str(_REGULATIONS_SOURCE_ROOT / supplemental_source_filename)
            if supplemental_source_filename
            else source_path
        )
        term_payload = {
            "term_id": f"{regulation_id}:glossary_{index:02d}",
            "term_no": index,
            "term": str(supplemental_term.get("term") or "").strip(),
            "definition": str(supplemental_term.get("definition") or "").strip(),
            "source_clause_id": str(supplemental_term.get("source_clause_id") or chapter_id).strip(),
            "source_heading": str(supplemental_term.get("source_heading") or source_heading).strip(),
            "citation_anchor": str(
                supplemental_term.get("citation_anchor") or f"{regulation_id}#ch_06_term_{index:02d}"
            ).strip(),
            "source_path": supplemental_source_path,
            "source_filename": supplemental_source_filename or source_filename,
        }
        source_note = str(supplemental_term.get("source_note") or "").strip()
        if source_note:
            term_payload["source_note"] = source_note
        terms.append(term_payload)

    return {
        "schema_version": REGULATION_LIBRARY_VERSION,
        "glossary_schema_version": REGULATION_GLOSSARY_VERSION,
        "regulation_id": regulation_id,
        "regulation_title": regulation.get("title"),
        "source_clause_id": chapter_id,
        "source_heading": source_heading,
        "term_count": len(terms),
        "terms": terms,
    }


def build_reference_manifest(
    regulation: dict[str, Any],
    chapters: list[dict[str, Any]],
) -> dict[str, Any]:
    regulation_id = str(regulation.get("regulation_id") or "").strip()
    reference_chapter = next(
        (
            chapter
            for chapter in chapters
            if str(chapter.get("chapter_title") or "").strip() == "参考"
        ),
        None,
    )

    if regulation_id != "cn_ectd_technical_specification" or reference_chapter is None:
        return {
            "schema_version": REGULATION_LIBRARY_VERSION,
            "reference_manifest_schema_version": REGULATION_REFERENCE_MANIFEST_VERSION,
            "regulation_id": regulation_id,
            "regulation_title": regulation.get("title"),
            "reference_count": 0,
            "references": [],
        }

    chapter_id = str(reference_chapter.get("chapter_id") or "").strip()
    source_heading = str(reference_chapter.get("heading") or "").strip()
    source_path = str(regulation.get("source_path") or "").strip()
    source_filename = str(regulation.get("source_filename") or "").strip()
    references: list[dict[str, Any]] = []
    for index, reference in enumerate(_ECTD_TECHNICAL_SPEC_REFERENCE_MANIFEST, start=1):
        references.append(
            {
                "reference_id": f"{regulation_id}:reference_{index:02d}",
                "ref_no": index,
                "title": reference["title"],
                "authority": reference["authority"],
                "reference_type": reference["reference_type"],
                "normative_role": reference["normative_role"],
                "source_clause_id": chapter_id,
                "source_heading": source_heading,
                "citation_anchor": f"{regulation_id}#ch_05_ref_{index:02d}",
                "source_path": source_path,
                "source_filename": source_filename,
            }
        )

    return {
        "schema_version": REGULATION_LIBRARY_VERSION,
        "reference_manifest_schema_version": REGULATION_REFERENCE_MANIFEST_VERSION,
        "regulation_id": regulation_id,
        "regulation_title": regulation.get("title"),
        "source_clause_id": chapter_id,
        "source_heading": source_heading,
        "reference_count": len(references),
        "references": references,
    }


def build_coverage_report(
    regulation: dict[str, Any],
    clauses: list[dict[str, Any]],
    requirement_matrix: dict[str, Any],
) -> dict[str, Any]:
    regulation_id = str(regulation.get("regulation_id") or "").strip()
    if regulation_id == "cn_ectd_technical_specification":
        relevant_clauses = [
            clause
            for clause in clauses
            if int(clause.get("chapter_no") or 0) in {1, 2, 3, 4}
        ]
        scope = "chapter_1_to_4"
        coverage_overrides = _ECTD_TECHNICAL_SPEC_CLAUSE_COVERAGE_OVERRIDES
        supporting_artifact_coverage = list(_ECTD_TECHNICAL_SPEC_SUPPORTING_ARTIFACT_COVERAGE)
        recommended_next_direction = {
            "primary": "run_technical_specification_closure_qa",
            "primary_reason": (
                "Technical-specification traceability gaps are closed; inspect partially_covered, deferred, and "
                "citation-only statuses for accurate evidence-boundary notes, and do not inflate prerequisite-dependent "
                "or review-only clauses into inaccurate executable rules."
            ),
            "secondary": "keep_validation_standard_closed_unless_artifacts_change",
            "secondary_reason": (
                "cn_ectd_validation_standard remains closed at 146 covered / 0 partially_covered / 3 citation-only; "
                "reopen it only if source artifacts or coverage requirements materially change."
            ),
        }
        use_requirement_traceability = True
    elif regulation_id == "cn_ectd_validation_standard":
        relevant_clauses = [
            clause
            for clause in clauses
            if int(clause.get("chapter_no") or 0) in {1, 2, 3, 4, 5, 6}
        ]
        scope = "chapter_1_to_6"
        coverage_overrides = _ECTD_VALIDATION_STANDARD_CHAPTER2_CLAUSE_COVERAGE_OVERRIDES
        supporting_artifact_coverage = []
        recommended_next_direction = {
            "primary": "resolve_validation_standard_chapter4_prerequisite_evidence_before_more_automation",
            "primary_reason": (
                "The remaining validation-standard Chapter 4 partials are intentionally held at a human-review and "
                "prerequisite boundary: schema reference/schema validation require local schema resources and validation "
                "diagnostics, while schema-version and envelope immutability checks require historical sequence evidence."
            ),
            "secondary": "implement_schema_or_history_evidence_path_before_promoting_remaining_partials",
            "secondary_reason": (
                "Further automation should start only after a local schema-validation path or a trusted multi-sequence "
                "history comparison path exists; otherwise keep the clauses as prerequisite guidance rather than false coverage."
            ),
        }
        use_requirement_traceability = False
    else:
        relevant_clauses = []
        scope = "chapter_1_to_4"
        coverage_overrides = {}
        supporting_artifact_coverage = []
        recommended_next_direction = None
        use_requirement_traceability = False

    if not relevant_clauses:
        return {
            "schema_version": REGULATION_LIBRARY_VERSION,
            "coverage_schema_version": REGULATION_COVERAGE_REPORT_VERSION,
            "regulation_id": regulation_id,
            "regulation_title": regulation.get("title"),
            "scope": scope,
            "clause_count": 0,
            "chapter_count": 0,
            "chapter_summaries": [],
            "clause_coverages": [],
            "supporting_artifact_coverage": [],
            "recommended_next_direction": None,
        }

    requirement_ids_by_clause_id: dict[str, list[str]] = {}
    for requirement in requirement_matrix.get("requirements", []) or []:
        clause_id = str(requirement.get("source_clause_id") or "").strip()
        requirement_id = str(requirement.get("requirement_id") or "").strip()
        if not clause_id or not requirement_id:
            continue
        requirement_ids_by_clause_id.setdefault(clause_id, []).append(requirement_id)

    clause_coverages: list[dict[str, Any]] = []
    chapter_summary_index: dict[int, dict[str, Any]] = {}
    for clause in relevant_clauses:
        clause_id = str(clause.get("clause_id") or "").strip()
        clause_ref = clause_id.split(":")[-1] if ":" in clause_id else clause_id
        chapter_no = int(clause.get("chapter_no") or 0)
        chapter_default = (
            _ECTD_VALIDATION_STANDARD_CHAPTER_DEFAULTS.get(chapter_no, {})
            if regulation_id == "cn_ectd_validation_standard"
            else {}
        )
        override = {**chapter_default, **coverage_overrides.get(clause_ref, {})}
        implemented_rule_ids = list(override.get("implemented_rule_ids", []) or [])
        requirement_ids = list(requirement_ids_by_clause_id.get(clause_id, []) or []) if use_requirement_traceability else []
        coverage_status = str(override.get("coverage_status") or "deferred")
        coverage_note = str(override.get("coverage_note") or "").strip()
        traceability_gap = bool(implemented_rule_ids) and not requirement_ids if use_requirement_traceability else False

        clause_coverages.append(
            {
                "clause_id": clause_id,
                "chapter_no": chapter_no,
                "heading": clause.get("heading"),
                "recommended_rule_mode": (clause.get("classification") or {}).get("recommended_rule_mode"),
                "automation_ready": bool((clause.get("classification") or {}).get("automation_ready", False)),
                "coverage_status": coverage_status,
                "implemented_rule_ids": implemented_rule_ids,
                "requirement_ids": requirement_ids,
                "traceability_gap": traceability_gap,
                "coverage_note": coverage_note,
            }
        )

        chapter_summary = chapter_summary_index.setdefault(
            chapter_no,
            {
                "chapter_no": chapter_no,
                "chapter_title": clause.get("chapter_title"),
                "clause_count": 0,
                "covered_clause_count": 0,
                "partially_covered_clause_count": 0,
                "deferred_clause_count": 0,
                "citation_only_clause_count": 0,
                "traceability_gap_clause_count": 0,
            },
        )
        chapter_summary["clause_count"] += 1
        if coverage_status == "covered":
            chapter_summary["covered_clause_count"] += 1
        elif coverage_status == "partially_covered":
            chapter_summary["partially_covered_clause_count"] += 1
        elif coverage_status == "citation_only_recorded":
            chapter_summary["citation_only_clause_count"] += 1
        else:
            chapter_summary["deferred_clause_count"] += 1
        if traceability_gap:
            chapter_summary["traceability_gap_clause_count"] += 1

    chapter_summaries = [
        chapter_summary_index[key]
        for key in sorted(chapter_summary_index)
    ]
    coverage_counts: dict[str, int] = {}
    traceability_gap_clause_ids: list[str] = []
    for clause_coverage in clause_coverages:
        coverage_status = str(clause_coverage.get("coverage_status") or "").strip()
        if coverage_status:
            coverage_counts[coverage_status] = coverage_counts.get(coverage_status, 0) + 1
        if bool(clause_coverage.get("traceability_gap")):
            traceability_gap_clause_ids.append(str(clause_coverage.get("clause_id") or "").strip())

    return {
        "schema_version": REGULATION_LIBRARY_VERSION,
        "coverage_schema_version": REGULATION_COVERAGE_REPORT_VERSION,
        "regulation_id": regulation_id,
        "regulation_title": regulation.get("title"),
        "scope": scope,
        "clause_count": len(clause_coverages),
        "chapter_count": len(chapter_summaries),
        "coverage_counts": coverage_counts,
        "traceability_gap_clause_count": len(traceability_gap_clause_ids),
        "traceability_gap_clause_ids": traceability_gap_clause_ids,
        "chapter_summaries": chapter_summaries,
        "clause_coverages": clause_coverages,
        "supporting_artifact_coverage": supporting_artifact_coverage,
        "recommended_next_direction": recommended_next_direction,
    }


def build_capability_crosswalk(
    regulation: dict[str, Any],
    clauses: list[dict[str, Any]],
    coverage_report: dict[str, Any],
) -> dict[str, Any]:
    source_regulation_id = str(regulation.get("regulation_id") or "").strip()
    target_regulation_id = "cn_ectd_technical_specification"
    empty_payload = {
        "schema_version": REGULATION_LIBRARY_VERSION,
        "capability_crosswalk_schema_version": REGULATION_CAPABILITY_CROSSWALK_VERSION,
        "source_regulation_id": source_regulation_id,
        "target_regulation_id": target_regulation_id,
        "relation_basis": "shared_runtime_rule_ids",
        "relation_note": (
            "Shared runtime rules indicate common executable capability reuse across regulations; "
            "they do not by themselves claim strict one-to-one normative equivalence."
        ),
        "relation_count": 0,
        "source_clause_count_with_links": 0,
        "source_clause_count_without_links": 0,
        "relations": [],
        "source_clauses_without_technical_spec_links": [],
    }
    if source_regulation_id != "cn_ectd_validation_standard":
        return empty_payload

    clause_heading_by_id = {
        str(clause.get("clause_id") or "").strip(): str(clause.get("heading") or "").strip()
        for clause in clauses
        if str(clause.get("clause_id") or "").strip()
    }
    target_rule_to_clause_ids: dict[str, set[str]] = {}
    for clause_ref, override in _ECTD_TECHNICAL_SPEC_CLAUSE_COVERAGE_OVERRIDES.items():
        target_clause_id = f"{target_regulation_id}:{clause_ref}"
        for rule_id in list(override.get("implemented_rule_ids", []) or []):
            normalized_rule_id = str(rule_id or "").strip()
            if not normalized_rule_id:
                continue
            target_rule_to_clause_ids.setdefault(normalized_rule_id, set()).add(target_clause_id)

    relations: list[dict[str, Any]] = []
    source_clauses_without_links: list[dict[str, Any]] = []
    linked_source_clause_ids: set[str] = set()
    for clause_coverage in list(coverage_report.get("clause_coverages", []) or []):
        source_clause_id = str(clause_coverage.get("clause_id") or "").strip()
        implemented_rule_ids = [
            str(rule_id or "").strip()
            for rule_id in list(clause_coverage.get("implemented_rule_ids", []) or [])
            if str(rule_id or "").strip()
        ]
        if not source_clause_id or not implemented_rule_ids:
            continue

        shared_rules_by_target_clause_id: dict[str, set[str]] = {}
        for rule_id in implemented_rule_ids:
            for target_clause_id in sorted(target_rule_to_clause_ids.get(rule_id, set())):
                shared_rules_by_target_clause_id.setdefault(target_clause_id, set()).add(rule_id)

        if not shared_rules_by_target_clause_id:
            source_clauses_without_links.append(
                {
                    "source_clause_id": source_clause_id,
                    "source_heading": clause_heading_by_id.get(source_clause_id, ""),
                    "implemented_rule_ids": implemented_rule_ids,
                }
            )
            continue

        linked_source_clause_ids.add(source_clause_id)
        for target_clause_id, shared_rule_ids in sorted(shared_rules_by_target_clause_id.items()):
            relations.append(
                {
                    "relation_id": f"{source_clause_id}::{target_clause_id}",
                    "source_clause_id": source_clause_id,
                    "source_heading": clause_heading_by_id.get(source_clause_id, ""),
                    "target_clause_id": target_clause_id,
                    "shared_rule_ids": sorted(shared_rule_ids),
                    "relation_type": "shared_runtime_rule_capability",
                }
            )

    return {
        **empty_payload,
        "relation_count": len(relations),
        "source_clause_count_with_links": len(linked_source_clause_ids),
        "source_clause_count_without_links": len(source_clauses_without_links),
        "relations": relations,
        "source_clauses_without_technical_spec_links": source_clauses_without_links,
    }


def build_regulation_corpus_entry(path: Path) -> dict[str, Any]:
    parsed_document = parse_file(path)
    normalized_text = normalize_regulation_source_text(str(parsed_document.get("text") or ""))
    lines = _tokenize_nonempty_lines(normalized_text)
    metadata = _resolve_regulation_metadata(path, lines)
    chapters, clauses = _extract_chapters_and_clauses(lines, metadata, parsed_document)
    rule_candidates = build_rule_candidates(clauses)
    classification_summary = _build_classification_summary(clauses)
    direct_rule_drafts = build_direct_rule_drafts(
        {
            **metadata,
            "title": metadata["title"],
            "version_label": metadata["version_label"],
        },
        rule_candidates,
    )
    requirement_matrix = build_requirement_matrix(
        {
            **metadata,
            "title": metadata["title"],
            "version_label": metadata["version_label"],
        },
        clauses,
    )
    glossary = build_glossary(
        {
            **metadata,
            "title": metadata["title"],
            "source_path": str(path),
            "source_filename": path.name,
        },
        chapters,
    )
    reference_manifest = build_reference_manifest(
        {
            **metadata,
            "title": metadata["title"],
            "source_path": str(path),
            "source_filename": path.name,
        },
        chapters,
    )
    coverage_report = build_coverage_report(
        {
            **metadata,
            "title": metadata["title"],
            "source_path": str(path),
            "source_filename": path.name,
        },
        clauses,
        requirement_matrix,
    )
    capability_crosswalk = build_capability_crosswalk(
        {
            **metadata,
            "title": metadata["title"],
            "source_path": str(path),
            "source_filename": path.name,
        },
        clauses,
        coverage_report,
    )

    return {
        "schema_version": REGULATION_LIBRARY_VERSION,
        "regulation": {
            **metadata,
            "source_type": parsed_document.get("source_type"),
            "parser_hint": (parsed_document.get("metadata") or {}).get("parser_hint"),
            "paragraph_count": (parsed_document.get("metadata") or {}).get("paragraph_count"),
            "chapter_count": len(chapters),
            "article_count": len(clauses),
            "chapter_refs": [chapter["chapter_id"] for chapter in chapters],
            "clause_refs": [clause["clause_id"] for clause in clauses],
            "rule_candidate_refs": [candidate["rule_candidate_id"] for candidate in rule_candidates],
            "classification_summary": classification_summary,
        },
        "chapters": chapters,
        "clauses": clauses,
        "rule_candidates": rule_candidates,
        "direct_rule_drafts": direct_rule_drafts,
        "requirement_matrix": requirement_matrix,
        "glossary": glossary,
        "reference_manifest": reference_manifest,
        "coverage_report": coverage_report,
        "capability_crosswalk": capability_crosswalk,
    }


def write_regulation_corpus_entry(
    path: Path,
    *,
    output_root: Path,
    draft_rules_root: Path | None = None,
) -> dict[str, Path]:
    payload = build_regulation_corpus_entry(path)
    regulation = payload["regulation"]
    regulation_id = str(regulation["regulation_id"])
    output_root.mkdir(parents=True, exist_ok=True)

    document_path = output_root / f"{regulation_id}.document.json"
    clauses_path = output_root / f"{regulation_id}.clauses.json"
    candidates_path = output_root / f"{regulation_id}.rule_candidates.json"
    requirement_matrix_path = output_root / f"{regulation_id}.requirement_matrix.json"
    glossary_path = output_root / f"{regulation_id}.glossary.json"
    reference_manifest_path = output_root / f"{regulation_id}.reference_manifest.json"
    coverage_report_path = output_root / f"{regulation_id}.coverage_report.json"
    capability_crosswalk_path = output_root / f"{regulation_id}.capability_crosswalk.json"
    draft_rules_path = (
        (draft_rules_root or output_root) / f"{regulation_id}.direct_rule_drafts.json"
    )

    document_payload = {
        "schema_version": REGULATION_LIBRARY_VERSION,
        "regulation": regulation,
        "chapters": payload["chapters"],
    }
    clauses_payload = {
        "schema_version": REGULATION_LIBRARY_VERSION,
        "regulation_id": regulation_id,
        "clause_count": len(payload["clauses"]),
        "clauses": payload["clauses"],
    }
    candidates_payload = {
        "schema_version": REGULATION_LIBRARY_VERSION,
        "regulation_id": regulation_id,
        "rule_candidate_count": len(payload["rule_candidates"]),
        "rule_candidates": payload["rule_candidates"],
    }
    draft_rules_payload = payload["direct_rule_drafts"]
    requirement_matrix_payload = payload["requirement_matrix"]
    glossary_payload = payload["glossary"]
    reference_manifest_payload = payload["reference_manifest"]
    coverage_report_payload = payload["coverage_report"]
    capability_crosswalk_payload = payload["capability_crosswalk"]

    document_path.write_text(json.dumps(document_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    clauses_path.write_text(json.dumps(clauses_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates_path.write_text(json.dumps(candidates_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if int(requirement_matrix_payload.get("requirement_count", 0) or 0) > 0:
        requirement_matrix_path.write_text(
            json.dumps(requirement_matrix_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if int(glossary_payload.get("term_count", 0) or 0) > 0:
        glossary_path.write_text(
            json.dumps(glossary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if int(reference_manifest_payload.get("reference_count", 0) or 0) > 0:
        reference_manifest_path.write_text(
            json.dumps(reference_manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if int(coverage_report_payload.get("clause_count", 0) or 0) > 0:
        coverage_report_path.write_text(
            json.dumps(coverage_report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if (
        int(capability_crosswalk_payload.get("relation_count", 0) or 0) > 0
        or int(capability_crosswalk_payload.get("source_clause_count_without_links", 0) or 0) > 0
    ):
        capability_crosswalk_path.write_text(
            json.dumps(capability_crosswalk_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    draft_rules_path.parent.mkdir(parents=True, exist_ok=True)
    draft_rules_path.write_text(json.dumps(draft_rules_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    outputs = {
        "document": document_path,
        "clauses": clauses_path,
        "rule_candidates": candidates_path,
        "direct_rule_drafts": draft_rules_path,
    }
    if requirement_matrix_path.exists():
        outputs["requirement_matrix"] = requirement_matrix_path
    if glossary_path.exists():
        outputs["glossary"] = glossary_path
    if reference_manifest_path.exists():
        outputs["reference_manifest"] = reference_manifest_path
    if coverage_report_path.exists():
        outputs["coverage_report"] = coverage_report_path
    if capability_crosswalk_path.exists():
        outputs["capability_crosswalk"] = capability_crosswalk_path
    return outputs
