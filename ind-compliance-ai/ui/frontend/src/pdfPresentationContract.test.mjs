import assert from 'node:assert/strict'
import test from 'node:test'

import { buildPdfPresentationContractDisplay } from './pdfPresentationContract.js'

test('builds a compact source and verification summary', () => {
  const display = buildPdfPresentationContractDisplay({
    schema_version: 'ectd-pdf-presentation-contract-v1',
    source_references: {
      cn_technical_specification: {
        source_filename: 'eCTD技术规范.pdf',
        section: '3.4',
        pdf_page: 23,
      },
      ich_submission_formats: {
        source_filename: 'Specification_for_Submission_Formats_for_eCTD_v1_2.pdf',
        sections: ['2.1', '2.18'],
        pdf_page_start: 4,
        pdf_page_end: 9,
      },
    },
    deterministic_checks: { file_size: {}, navigation: {} },
    automation_boundary: { manual_review: ['Chinese font size by semantic text role'] },
  })

  assert.equal(display.schemaVersion, 'ectd-pdf-presentation-contract-v1')
  assert.equal(display.sourceRows[0].section, '3.4')
  assert.equal(display.sourceRows[1].pageLabel, 'PDF 4-9 页')
  assert.deepEqual(display.deterministicLabels, ['文件大小', '长文档导航'])
  assert.equal(display.hasManualReview, true)
})

test('does not create a misleading section for missing contract evidence', () => {
  assert.equal(buildPdfPresentationContractDisplay(null), null)
  assert.equal(buildPdfPresentationContractDisplay([]), null)
})
