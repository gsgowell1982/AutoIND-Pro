import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildStructureAuditRecordSignature,
  buildStructureAuditPathFocusKey,
  buildStructureAuditPathFocusLabel,
  buildStructureAuditNavigationTargets,
  buildStructureAuditIssueSummary,
  getStructureAuditMissingPathRows,
  getStructureAuditPathAlignmentRows,
  resolveStructureAuditPathNavigationTarget,
  buildStructureAuditRuleFocusList,
  buildStructureAuditGroups,
  filterStructureAuditRecords,
  getStructureAuditRootPageConflictRows,
  getStructureAuditFilenameOptions,
  getStructureAuditIssueLabel,
  getStructureAuditIssueOptions,
  getStructureAuditIssues,
  getStructureAuditSeverity,
  toggleStructureAuditFilterValue,
} from '../src/structureAudit.js'

const SAMPLE_RECORDS = [
  {
    audit_record_id: 'audit-1',
    rule_id: 'SR-TOC-001',
    filename: 'b-body.pdf',
    alignment_ready: false,
    root_outline_coverage_ratio: 0.5,
    root_page_order_ready: false,
    root_page_offset_ready: true,
    root_page_span_ready: false,
    direct_child_coverage_ratio: 0.75,
    bounded_subtree_coverage_ratio: 1,
  },
  {
    audit_record_id: 'audit-2',
    rule_id: 'SR-TOC-001',
    filename: 'a-outline.pdf',
    alignment_ready: true,
    root_outline_coverage_ratio: 1,
    root_page_order_ready: true,
    root_page_offset_ready: true,
    root_page_span_ready: true,
    direct_child_coverage_ratio: 1,
    bounded_subtree_coverage_ratio: 1,
  },
  {
    audit_record_id: 'audit-3',
    rule_id: 'SR-TOC-001',
    filename: 'b-body.pdf',
    alignment_ready: false,
    root_outline_coverage_ratio: 1,
    root_page_order_ready: true,
    root_page_offset_ready: false,
    root_page_span_ready: true,
    direct_child_coverage_ratio: 1,
    bounded_subtree_coverage_ratio: 0.5,
    root_page_alignment_rows: [
      {
        outline_index: '1.0',
        toc_navigation_page: 2,
        toc_page_locator_value: 3,
        body_page_start: 3,
        body_page_end: 5,
        projected_page: 3,
        offset: 0,
        order_conflict: false,
        offset_conflict: false,
        span_conflict: false,
      },
      {
        outline_index: '2.0',
        toc_navigation_page: 2,
        toc_page_locator_value: 5,
        body_page_start: 8,
        body_page_end: 40,
        projected_page: 8,
        offset: 3,
        order_conflict: false,
        offset_conflict: true,
        span_conflict: false,
      },
      {
        outline_index: '3.0',
        toc_navigation_page: 3,
        toc_page_locator_value: 18,
        body_page_start: 8,
        body_page_end: 39,
        projected_page: 8,
        offset: -10,
        order_conflict: true,
        offset_conflict: true,
        span_conflict: true,
      },
    ],
    missing_body_direct_child_path_rows: [
      {
        root_outline_index: '1.0',
        root_normalized_outline_index: '1',
        parent_outline_index: '1.0',
        parent_normalized_outline_index: '1',
        outline_index: '1.2',
        normalized_outline_index: '1.2',
        outline_path: '1.0 > 1.2',
        text_path: 'Overview > Composition',
        page_locator_value: 5,
        navigation_page: 2,
        level: 2,
        nearest_body_anchor_outline_index: '1',
        nearest_body_anchor_page: 3,
      },
    ],
    missing_body_bounded_subtree_path_rows: [
      {
        root_outline_index: '1.0',
        root_normalized_outline_index: '1',
        parent_outline_index: '1.1.1',
        parent_normalized_outline_index: '1.1.1',
        outline_index: '1.1.1.2',
        normalized_outline_index: '1.1.1.2',
        outline_path: '1.0 > 1.1 > 1.1.1 > 1.1.1.2',
        text_path: 'Overview > Scope > Dosage > Administration',
        page_locator_value: 7,
        navigation_page: 2,
        level: 4,
        nearest_body_anchor_outline_index: '1.1.1',
        nearest_body_anchor_page: 5,
      },
    ],
    toc_body_alignment_path_rows: [
      {
        root_outline_index: '1.0',
        root_normalized_outline_index: '1',
        parent_outline_index: null,
        parent_normalized_outline_index: null,
        outline_index: '1.0',
        normalized_outline_index: '1',
        outline_path: '1.0',
        text_path: 'Overview',
        page_locator_value: 3,
        navigation_page: 2,
        level: 1,
        body_anchor_outline_index: '1',
        body_anchor_page: 3,
        body_anchor_kind: 'heading',
        alignment_status: 'matched',
        page_offset: 0,
        has_body_match: true,
      },
      {
        root_outline_index: '1.0',
        root_normalized_outline_index: '1',
        parent_outline_index: '1.1.1',
        parent_normalized_outline_index: '1.1.1',
        outline_index: '1.1.1.2',
        normalized_outline_index: '1.1.1.2',
        outline_path: '1.0 > 1.1 > 1.1.1 > 1.1.1.2',
        text_path: 'Overview > Scope > Dosage > Administration',
        page_locator_value: 7,
        navigation_page: 2,
        level: 4,
        body_anchor_outline_index: '1.1.1',
        body_anchor_page: 5,
        body_anchor_kind: 'nearest_parent',
        alignment_status: 'nearest_parent',
        page_offset: null,
        has_body_match: false,
      },
    ],
  },
]

const SAMPLE_RULE_ITEMS = [
  { rule_id: 'HR-CTD-001', status: 'fail' },
  { rule_id: 'SR-TOC-001', status: 'warn' },
  { rule_id: 'SR-REG-006', status: 'pass' },
]

const SAMPLE_TOC_SEQUENCES = [
  {
    toc_sequence_id: 'toc-seq-001',
    pages: [2, 3],
    page_span: [2, 3],
  },
]

const SAMPLE_TOC_BLOCKS = [
  {
    toc_id: 'toc_001',
    page: 2,
    toc_sequence_id: 'toc-seq-001',
  },
]

test('collects issue keys from a structure audit record', () => {
  assert.deepEqual(getStructureAuditIssues(SAMPLE_RECORDS[0]), [
    'root_coverage_gap',
    'root_page_order_conflict',
    'root_page_span_conflict',
    'direct_child_coverage_gap',
  ])
})

test('derives warn or pass severity from alignment readiness', () => {
  assert.equal(getStructureAuditSeverity(SAMPLE_RECORDS[0]), 'warn')
  assert.equal(getStructureAuditSeverity(SAMPLE_RECORDS[1]), 'pass')
})

test('builds a structure audit result signature from file and path-row counts', () => {
  const firstSignature = buildStructureAuditRecordSignature(SAMPLE_RECORDS, { pdfFileId: 'file-a' })
  const sameSignature = buildStructureAuditRecordSignature(SAMPLE_RECORDS, { pdfFileId: 'file-a' })
  const nextFileSignature = buildStructureAuditRecordSignature(SAMPLE_RECORDS, { pdfFileId: 'file-b' })
  const nextPathSignature = buildStructureAuditRecordSignature(
    [
      SAMPLE_RECORDS[0],
      SAMPLE_RECORDS[1],
      {
        ...SAMPLE_RECORDS[2],
        toc_body_alignment_path_rows: [
          ...SAMPLE_RECORDS[2].toc_body_alignment_path_rows,
          {
            outline_index: '2.1',
            text_path: 'Next > Child',
          },
        ],
      },
    ],
    { pdfFileId: 'file-a' },
  )

  assert.equal(firstSignature, sameSignature)
  assert.notEqual(firstSignature, nextFileSignature)
  assert.notEqual(firstSignature, nextPathSignature)
})

test('filters structure audit records to all records by default', () => {
  const result = filterStructureAuditRecords(SAMPLE_RECORDS)

  assert.deepEqual(
    result.map((record) => record.audit_record_id),
    ['audit-1', 'audit-2', 'audit-3'],
  )
})

test('filters structure audit records by severity and issue key', () => {
  const passes = filterStructureAuditRecords(SAMPLE_RECORDS, { severityFilter: 'passes' })
  assert.deepEqual(
    passes.map((record) => record.audit_record_id),
    ['audit-2'],
  )

  const offsetWarnings = filterStructureAuditRecords(SAMPLE_RECORDS, {
    severityFilter: 'warnings',
    issueFilter: 'root_page_offset_conflict',
  })
  assert.deepEqual(
    offsetWarnings.map((record) => record.audit_record_id),
    ['audit-3'],
  )

  const singleDocument = filterStructureAuditRecords(SAMPLE_RECORDS, {
    severityFilter: 'all',
    filenameFilter: 'b-body.pdf',
  })
  assert.deepEqual(
    singleDocument.map((record) => record.audit_record_id),
    ['audit-1', 'audit-3'],
  )
})

test('groups filtered records by filename in stable order', () => {
  const result = buildStructureAuditGroups(SAMPLE_RECORDS, { severityFilter: 'all' })

  assert.deepEqual(result, [
    {
      filename: 'a-outline.pdf',
      warnCount: 0,
      passCount: 1,
      issueKeys: [],
      records: [SAMPLE_RECORDS[1]],
    },
    {
      filename: 'b-body.pdf',
      warnCount: 2,
      passCount: 0,
      issueKeys: [
        'root_coverage_gap',
        'root_page_order_conflict',
        'root_page_offset_conflict',
        'root_page_span_conflict',
        'direct_child_coverage_gap',
        'bounded_subtree_coverage_gap',
      ],
      records: [SAMPLE_RECORDS[0], SAMPLE_RECORDS[2]],
    },
  ])
})

test('returns issue filter options with all option first', () => {
  const options = getStructureAuditIssueOptions()

  assert.equal(options[0].value, 'all')
  assert.equal(options[0].label.length > 0, true)
  assert.equal(options.some((option) => option.value === 'root_page_order_conflict'), true)
})

test('returns filename filter options with all option first and document counts', () => {
  const options = getStructureAuditFilenameOptions(SAMPLE_RECORDS)

  assert.deepEqual(options, [
    { value: 'all', label: '全部文件' },
    { value: 'a-outline.pdf', label: 'a-outline.pdf (1)' },
    { value: 'b-body.pdf', label: 'b-body.pdf (2)' },
  ])
})

test('builds issue summary counts after applying filters', () => {
  const summary = buildStructureAuditIssueSummary(SAMPLE_RECORDS, {
    severityFilter: 'warnings',
    filenameFilter: 'b-body.pdf',
  })

  assert.deepEqual(summary, [
    { issueKey: 'root_coverage_gap', label: getStructureAuditIssueLabel('root_coverage_gap'), count: 1 },
    {
      issueKey: 'root_page_order_conflict',
      label: getStructureAuditIssueLabel('root_page_order_conflict'),
      count: 1,
    },
    {
      issueKey: 'root_page_offset_conflict',
      label: getStructureAuditIssueLabel('root_page_offset_conflict'),
      count: 1,
    },
    {
      issueKey: 'root_page_span_conflict',
      label: getStructureAuditIssueLabel('root_page_span_conflict'),
      count: 1,
    },
    {
      issueKey: 'direct_child_coverage_gap',
      label: getStructureAuditIssueLabel('direct_child_coverage_gap'),
      count: 1,
    },
    {
      issueKey: 'bounded_subtree_coverage_gap',
      label: getStructureAuditIssueLabel('bounded_subtree_coverage_gap'),
      count: 1,
    },
  ])
})

test('moves the focused rule item to the front without losing other rule items', () => {
  const result = buildStructureAuditRuleFocusList(SAMPLE_RULE_ITEMS, 'SR-TOC-001')

  assert.deepEqual(
    result.map((item) => item.rule_id),
    ['SR-TOC-001', 'HR-CTD-001', 'SR-REG-006'],
  )
})

test('keeps original rule order when there is no matching focused rule id', () => {
  const result = buildStructureAuditRuleFocusList(SAMPLE_RULE_ITEMS, 'SR-UNKNOWN-999')

  assert.deepEqual(
    result.map((item) => item.rule_id),
    ['HR-CTD-001', 'SR-TOC-001', 'SR-REG-006'],
  )
})

test('toggles a structure audit filter value back to all when clicking the active option', () => {
  assert.equal(toggleStructureAuditFilterValue('root_page_order_conflict', 'root_page_order_conflict'), 'all')
  assert.equal(toggleStructureAuditFilterValue('all', 'root_page_order_conflict'), 'root_page_order_conflict')
  assert.equal(toggleStructureAuditFilterValue('b-body.pdf', 'a-outline.pdf'), 'a-outline.pdf')
})

test('builds structure audit navigation targets for the active pdf document', () => {
  const targets = buildStructureAuditNavigationTargets(SAMPLE_RECORDS[0], {
    activeFilename: 'b-body.pdf',
    tocSequences: SAMPLE_TOC_SEQUENCES,
  })

  assert.deepEqual(targets, [
    {
      key: 'toc',
      label: '定位目录页',
      page: 2,
      tocSequenceId: 'toc-seq-001',
    },
  ])

  const bodyTargets = buildStructureAuditNavigationTargets(
    {
      ...SAMPLE_RECORDS[0],
      projected_root_page_values: [9, 12],
    },
    {
      activeFilename: 'b-body.pdf',
      tocSequences: SAMPLE_TOC_SEQUENCES,
    },
  )

  assert.deepEqual(bodyTargets, [
    {
      key: 'toc',
      label: '定位目录页',
      page: 2,
      tocSequenceId: 'toc-seq-001',
    },
    {
      key: 'body',
      label: '定位正文根章节页',
      page: 9,
      tocSequenceId: null,
    },
  ])
})

test('returns only conflicting root-page alignment rows for a structure audit record', () => {
  const conflicts = getStructureAuditRootPageConflictRows(SAMPLE_RECORDS[2])

  assert.deepEqual(
    conflicts.map((row) => row.outline_index),
    ['2.0', '3.0'],
  )
})

test('returns normalized path rows for missing direct-child and subtree issues', () => {
  const directChildRows = getStructureAuditMissingPathRows(SAMPLE_RECORDS[2], 'direct_child')
  const subtreeRows = getStructureAuditMissingPathRows(SAMPLE_RECORDS[2], 'bounded_subtree')

  assert.deepEqual(directChildRows, [
    {
      rootOutlineIndex: '1.0',
      parentOutlineIndex: '1.0',
      outlineIndex: '1.2',
      outlinePath: '1.0 > 1.2',
      textPath: 'Overview > Composition',
      pageLocatorValue: 5,
      navigationPage: 2,
      level: 2,
      nearestBodyAnchorOutlineIndex: '1',
      nearestBodyAnchorPage: 3,
    },
  ])
  assert.deepEqual(subtreeRows, [
    {
      rootOutlineIndex: '1.0',
      parentOutlineIndex: '1.1.1',
      outlineIndex: '1.1.1.2',
      outlinePath: '1.0 > 1.1 > 1.1.1 > 1.1.1.2',
      textPath: 'Overview > Scope > Dosage > Administration',
      pageLocatorValue: 7,
      navigationPage: 2,
      level: 4,
      nearestBodyAnchorOutlineIndex: '1.1.1',
      nearestBodyAnchorPage: 5,
    },
  ])
})

test('returns normalized full TOC-body path alignment rows', () => {
  const rows = getStructureAuditPathAlignmentRows(SAMPLE_RECORDS[2])

  assert.deepEqual(rows, [
    {
      rootOutlineIndex: '1.0',
      parentOutlineIndex: null,
      outlineIndex: '1.0',
      outlinePath: '1.0',
      textPath: 'Overview',
      pageLocatorValue: 3,
      navigationPage: 2,
      level: 1,
      bodyAnchorOutlineIndex: '1',
      bodyAnchorPage: 3,
      bodyAnchorKind: 'heading',
      alignmentStatus: 'matched',
      pageOffset: 0,
      hasBodyMatch: true,
    },
    {
      rootOutlineIndex: '1.0',
      parentOutlineIndex: '1.1.1',
      outlineIndex: '1.1.1.2',
      outlinePath: '1.0 > 1.1 > 1.1.1 > 1.1.1.2',
      textPath: 'Overview > Scope > Dosage > Administration',
      pageLocatorValue: 7,
      navigationPage: 2,
      level: 4,
      bodyAnchorOutlineIndex: '1.1.1',
      bodyAnchorPage: 5,
      bodyAnchorKind: 'nearest_parent',
      alignmentStatus: 'nearest_parent',
      pageOffset: null,
      hasBodyMatch: false,
    },
  ])
})

test('resolves structure-audit path navigation targets for toc and body actions', () => {
  const [directChildRow] = getStructureAuditMissingPathRows(SAMPLE_RECORDS[2], 'direct_child')
  const [subtreeRow] = getStructureAuditMissingPathRows(SAMPLE_RECORDS[2], 'bounded_subtree')

  assert.deepEqual(
    resolveStructureAuditPathNavigationTarget(directChildRow, 'toc', {
      tocBlocks: SAMPLE_TOC_BLOCKS,
      preferredTocSequenceId: 'toc-seq-001',
    }),
    {
      page: 2,
      structuralId: 'toc_001',
      tocSequenceId: 'toc-seq-001',
      navigationStatus: 'resolved_structural',
      navigationReason: 'resolved_from_structure_audit_toc_path',
    },
  )

  assert.deepEqual(
    resolveStructureAuditPathNavigationTarget(subtreeRow, 'body'),
    {
      page: 5,
      structuralId: null,
      tocSequenceId: null,
      navigationStatus: 'resolved_page',
      navigationReason: 'resolved_from_structure_audit_body_anchor',
    },
  )
})

test('builds stable focus keys and readable focus labels for structure-audit paths', () => {
  const [directChildRow] = getStructureAuditMissingPathRows(SAMPLE_RECORDS[2], 'direct_child')
  const [subtreeRow] = getStructureAuditMissingPathRows(SAMPLE_RECORDS[2], 'bounded_subtree')

  assert.equal(
    buildStructureAuditPathFocusKey(directChildRow, 'direct_child'),
    'direct_child:1.0 > 1.2:5',
  )
  assert.equal(
    buildStructureAuditPathFocusLabel(directChildRow, 'toc'),
    'Overview > Composition -> 目录路径',
  )
  assert.equal(
    buildStructureAuditPathFocusLabel(subtreeRow, 'body'),
    'Overview > Scope > Dosage > Administration -> 正文锚点 1.1.1',
  )
})

test('does not build navigation targets for non-active documents', () => {
  const targets = buildStructureAuditNavigationTargets(SAMPLE_RECORDS[0], {
    activeFilename: 'other-document.pdf',
    tocSequences: SAMPLE_TOC_SEQUENCES,
  })

  assert.deepEqual(targets, [])
})

test('returns readable issue labels and falls back to the raw key', () => {
  assert.equal(getStructureAuditIssueLabel('root_coverage_gap').length > 0, true)
  assert.equal(getStructureAuditIssueLabel('custom_issue_key'), 'custom_issue_key')
})
