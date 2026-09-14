import test from 'node:test'
import assert from 'node:assert/strict'

import { resolveWorkbenchReviewScope } from './workbenchReviewScope.js'

test('package inventory forces project review mode even when only one PDF is parsed', () => {
  const scope = resolveWorkbenchReviewScope({
    pdf_document: { filename: 'm1.pdf' },
    package_inventory: {
      source_kind: 'folder',
      file_paths: ['x202112345/0000/m1.pdf', 'x202112345/0000/index.xml'],
      directory_paths: ['x202112345', 'x202112345/0000'],
    },
    rule_checks: { items: [] },
  })

  assert.equal(scope.mode, 'project')
  assert.equal(scope.isProjectReview, true)
  assert.equal(scope.isSingleFile, false)
})

test('a plain PDF upload remains a single-file review', () => {
  const scope = resolveWorkbenchReviewScope({
    pdf_document: { filename: 'report.pdf' },
    rule_checks: { items: [] },
  })

  assert.equal(scope.mode, 'single_file')
  assert.equal(scope.isSingleFile, true)
  assert.equal(scope.isProjectReview, false)
})
