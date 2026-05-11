import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildRuleBasisDisplay,
  buildRuleLocationLabelParts,
  buildRuleHeaderPresentation,
  getCitationLabel,
  getNavigationReasonLabel,
  getNavigationStatusLabel,
  getRuleBasisPresentation,
  getRuleCategoryLabel,
  getRuleStatusLabel,
  getRuleSummary,
  getRuleTitle,
  getVerificationStatusLabel,
  shouldDisplayNavigationDiagnostic,
} from '../src/rulePresentation.js'

test('returns Chinese title for known rule ids', () => {
  assert.equal(getRuleTitle('SR-CTD-002'), '当前材料集具备 CTD 基础申报结构信号')
  assert.equal(getRuleTitle('SR-REG-006'), '材料证据链支撑《实施条例》第六条的数据可追溯要求')
  assert.equal(getRuleTitle('SR-TOC-001'), '目录与正文章节树主干存在可验证重合')
  assert.equal(getRuleTitle('HR-ECTD-045'), 'PDF 超链接动作符合验证标准白名单')
  assert.equal(getRuleTitle('HR-ECTD-001'), 'eCTD 序列号递增校验')
  assert.equal(getRuleTitle('HR-ECTD-041'), 'PDF 文件不得包含嵌入附件')
  assert.equal(getRuleTitle('SR-ECTD-017'), 'PDF 文件不得包含非链接注释')
})

test('returns Chinese summary for known rule status combinations', () => {
  assert.equal(
    getRuleSummary('HR-CTD-001', 'fail', 'fallback'),
    '已识别到模块 3 / CMC 材料，但未检测到质量综述/QOS 文件。',
  )
  assert.equal(
    getRuleSummary(
      'HR-ECTD-045',
      'fail',
      'Detected PDF hyperlink actions outside the validation-standard whitelist (`GoTo`, `GoToR`, `Launch`).',
    ),
    '检测到 PDF 超链接动作不在验证标准允许范围内。',
  )
  assert.equal(
    getRuleSummary(
      'HR-ECTD-001',
      'na',
      'No eCTD sequence-number metadata was available, so sequence-number progression checks are not applicable.',
    ),
    '未检测到 eCTD 序列号元数据，当前不适用序列号递增检查。',
  )
  assert.equal(
    getRuleSummary('SR-TOC-001', 'warn', 'fallback'),
    '目录根章节与正文章节树主干缺少可验证重合，建议先复核目录或结构锚点。',
  )
  assert.equal(
    getRuleSummary(
      'HR-ECTD-041',
      'pass',
      'Validated that 1 applicable PDF document(s) do not contain embedded attachments.',
    ),
    '已检查 1 个适用 PDF 文件，未发现嵌入附件。',
  )
  assert.equal(
    getRuleSummary(
      'SR-ECTD-017',
      'pass',
      'Validated that 1 applicable PDF document(s) contain no non-link annotations.',
    ),
    '已检查 1 个适用 PDF 文件，未发现非链接注释。',
  )
})

test('returns fallback summary when no localized summary exists', () => {
  assert.equal(getRuleSummary('UNKNOWN-RULE', 'pass', '已完成规则检查。'), '已完成规则检查。')
})

test('does not expose English backend fallback messages in customer-facing rule summaries', () => {
  assert.equal(
    getRuleSummary('UNKNOWN-RULE', 'pass', 'Validated that 2 applicable PDF document(s) expose no security settings.'),
    '已完成该规则检查，未发现不符合项。',
  )
  assert.equal(
    getRuleSummary(
      'UNKNOWN-RULE',
      'fail',
      'Detected PDF documents with embedded attachments under the validation-standard attachment prohibition.',
    ),
    '该规则检查发现不符合项，请结合原文件定位与规则溯源复核。',
  )
  assert.equal(
    getRuleSummary(
      'UNKNOWN-RULE',
      'na',
      'No applicable PDF documents were available, so embedded-attachment checks are not applicable.',
    ),
    '当前缺少适用证据或前置条件，本规则不适用。',
  )
})

test('hides duplicate secondary rule id when title falls back to the same id', () => {
  assert.deepEqual(buildRuleHeaderPresentation('HR-LAW-017'), {
    title: 'HR-LAW-017',
    secondaryRuleId: null,
  })
})

test('keeps secondary rule id when a customer-readable title exists', () => {
  assert.deepEqual(buildRuleHeaderPresentation('HR-ECTD-045'), {
    title: 'PDF 超链接动作符合验证标准白名单',
    secondaryRuleId: 'HR-ECTD-045',
  })
})

test('returns Chinese labels for status/category/navigation/verification values', () => {
  assert.equal(getRuleStatusLabel('warn'), '预警')
  assert.equal(getRuleCategoryLabel('hard'), '硬规则')
  assert.equal(getNavigationStatusLabel('resolved_page'), '页级目标已解析')
  assert.equal(getNavigationReasonLabel('fallback_to_section_anchor'), '通过章节锚点回退到目标页')
  assert.equal(getVerificationStatusLabel('verification_failed'), '端到端验证失败')
})

test('hides unresolved navigation diagnostics from customer-facing rule details', () => {
  assert.equal(shouldDisplayNavigationDiagnostic('unresolved', 'no_navigation_target', undefined), false)
  assert.equal(shouldDisplayNavigationDiagnostic('unresolved', 'no_navigation_target', 12), false)
  assert.equal(shouldDisplayNavigationDiagnostic('resolved_page', 'fallback_to_section_anchor', 12), true)
  assert.equal(shouldDisplayNavigationDiagnostic('resolved_structural', 'resolved_from_structural_evidence', 12), true)
})

test('labels DOC module values as document classification instead of page evidence', () => {
  assert.deepEqual(
    buildRuleLocationLabelParts({
      filename: 'eCTD实施指南.pdf',
      module_label: 'DOC-1',
      section_refs: [],
    }),
    ['eCTD实施指南.pdf', '文档分类: DOC-1'],
  )
})

test('returns customer-readable citation labels for known internal anchors', () => {
  assert.equal(
    getCitationLabel('material-review-contract-v1#documents.summary'),
    '材料审阅契约 v1 / 文档摘要',
  )
  assert.equal(
    getCitationLabel('material-review-contract-v1#section_tree'),
    '材料审阅契约 v1 / 章节树',
  )
  assert.equal(
    getCitationLabel('cn_drug_registration_classification_and_dossier_requirements#art_015'),
    '《药品注册分类及申报资料要求》 第三部分（申报资料要求）第（一）项',
  )
})

test('separates regulation basis from system basis for rule presentation', () => {
  assert.deepEqual(
    getRuleBasisPresentation('SR-CTD-002', 'cn_drug_registration_classification_and_dossier_requirements#art_015'),
    {
      basisKind: 'regulation',
      basisLabel: '《药品注册分类及申报资料要求》第三部分（申报资料要求）第（一）项',
    },
  )
  assert.deepEqual(
    getRuleBasisPresentation('SR-TOC-001', 'material-review-contract-v1#section_tree'),
    {
      basisKind: 'system',
      basisLabel: '材料审阅契约 v1 / 章节树',
    },
  )
  assert.deepEqual(
    getRuleBasisPresentation('HR-PARSE-001', 'material-review-contract-v1#documents.summary'),
    {
      basisKind: 'system',
      basisLabel: '材料审阅契约 v1 / 文档摘要',
    },
  )
})

test('builds customer-facing rule basis display without leaking raw anchors', () => {
  assert.deepEqual(
    buildRuleBasisDisplay(
      'SR-CTD-002',
      'cn_drug_registration_classification_and_dossier_requirements#art_015',
      {
        basis_kind: 'regulation',
        basis_label: '《药品注册分类及申报资料要求》第三部分（申报资料要求）第（一）项',
        basis_detail: '申请资料应按 CTD 模块组织提交。',
      },
    ),
    {
      labelPrefix: '规则溯源',
      basisKind: 'regulation',
      basisLabel: '《药品注册分类及申报资料要求》第三部分（申报资料要求）第（一）项',
      basisDetail: '申请资料应按 CTD 模块组织提交。',
    },
  )
})

test('omits raw internal anchor when no customer-readable basis exists', () => {
  assert.equal(
    buildRuleBasisDisplay('UNKNOWN-RULE', 'cn_ectd_validation_standard:req_pdf_hyperlink_action_whitelist', null),
    null,
  )
  assert.equal(
    buildRuleBasisDisplay(
      'UNKNOWN-RULE',
      'cn_ectd_validation_standard:req_pdf_hyperlink_action_whitelist',
      {
        basis_kind: 'generic',
        basis_label: '',
        basis_detail: '',
      },
    ),
    null,
  )
})

test('returns a readable basis display for regulation-backed rules', () => {
  assert.deepEqual(
    buildRuleBasisDisplay(
      'SR-REG-006',
      'cn_drug_administration_law_implementation_regulation#art_006',
      {
        basis_kind: 'regulation',
        basis_label: '《中华人民共和国药品管理法实施条例》第六条',
        basis_detail: '药品证据链需可追溯。',
      },
    ),
    {
      labelPrefix: '规则溯源',
      basisKind: 'regulation',
      basisLabel: '《中华人民共和国药品管理法实施条例》第六条',
      basisDetail: '药品证据链需可追溯。',
    },
  )
})

test('uses the Chinese citation label for eCTD sequence-number basis presentation', () => {
  assert.deepEqual(
    buildRuleBasisDisplay('HR-ECTD-001', 'cn_ectd_technical_specification#sec_2_3_1', null),
    {
      labelPrefix: '规则溯源',
      basisKind: 'regulation',
      basisLabel: 'eCTD 技术规范 2.3.1 序列号',
      basisDetail: null,
    },
  )
  assert.deepEqual(
    buildRuleBasisDisplay(
      'HR-ECTD-001',
      'cn_ectd_technical_specification#sec_2_3_1',
      {
        basis_kind: 'generic',
        basis_label: '',
        basis_detail: '',
      },
    ),
    {
      labelPrefix: '依据说明',
      basisKind: 'generic',
      basisLabel: 'eCTD 技术规范 2.3.1 序列号',
      basisDetail: null,
    },
  )
})
