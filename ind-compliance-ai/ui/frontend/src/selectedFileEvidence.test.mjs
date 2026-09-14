import test from 'node:test'
import assert from 'node:assert/strict'

import { buildSelectedFileEvidence } from './selectedFileEvidence.js'

test('builds parsed document evidence for the selected relative path', () => {
  const evidence = buildSelectedFileEvidence({
    selectedPath: 'x202112345/0000/m5/report.pdf',
    documents: [
      {
        file_id: 'f1',
        filename: 'report.pdf',
        relative_path: 'x202112345/0000/m5/report.pdf',
        source_type: 'pdf',
        status: 'parsed',
        page_count: 12,
        character_count: 4200,
        table_count: 2,
        image_count: 1,
        parse_available: true,
        text_preview: 'Study summary',
        file_url: '/api/v1/files/f1',
      },
    ],
  })

  assert.equal(evidence.kind, 'parsed')
  assert.equal(evidence.document?.file_id, 'f1')
  assert.equal(evidence.metrics.pageCount, 12)
  assert.equal(evidence.metrics.characterCount, 4200)
  assert.equal(evidence.preview, 'Study summary')
  assert.equal(evidence.fileUrl, '/api/v1/files/f1')
})

test('distinguishes package support files from missing document evidence', () => {
  const support = buildSelectedFileEvidence({
    selectedPath: 'x202112345/0000/util/dtd/ich-ectd-3-2.dtd',
    documents: [],
    files: [
      {
        file_id: 'f2',
        filename: 'ich-ectd-3-2.dtd',
        relative_path: 'x202112345/0000/util/dtd/ich-ectd-3-2.dtd',
        status: 'completed',
        message: 'Package support file received; content parsing skipped',
      },
    ],
  })

  assert.equal(support.kind, 'support')
  assert.equal(support.document, null)
  assert.match(support.message, /支持文件/)

  const missing = buildSelectedFileEvidence({
    selectedPath: 'x202112345/0000/m3',
    documents: [],
    files: [],
    directoryPaths: ['x202112345', 'x202112345/0000', 'x202112345/0000/m3'],
  })
  assert.equal(missing.kind, 'directory')
  assert.equal(missing.document, null)
})
