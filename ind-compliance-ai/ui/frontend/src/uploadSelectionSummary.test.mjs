import assert from 'node:assert/strict'
import test from 'node:test'

import { buildUploadSelectionSummary, filterUploadSelectionFiles } from './uploadSelectionSummary.js'

test('summarizes project selection with unique directories and one root label', () => {
  const summary = buildUploadSelectionSummary({
    mode: 'directory',
    files: [
      { path: 'x202112345\\0000\\m1\\index.xml', size: 10 },
      { path: 'x202112345/0000/m1/cn/cn-regional.xml', size: 20 },
      { path: 'x202112345/0000/index.xml', size: 0 },
    ],
    directoryPaths: ['x202112345', 'x202112345/0000', 'x202112345/0000', 'x202112345/0000/m1'],
  })

  assert.deepEqual(summary, {
    mode: 'directory',
    fileCount: 3,
    directoryCount: 4,
    rootCount: 1,
    rootLabel: 'x202112345',
    totalBytes: 30,
  })
})

test('filters selected files by normalized relative path case-insensitively', () => {
  const files = [
    { path: 'x202112345\\0000\\m1\\index.xml', size: 10 },
    { path: 'x202112345/0000/m2/overview.pdf', size: 20 },
  ]

  assert.deepEqual(filterUploadSelectionFiles(files, 'M1/INDEX'), [files[0]])
  assert.deepEqual(filterUploadSelectionFiles(files, ''), files)
})
