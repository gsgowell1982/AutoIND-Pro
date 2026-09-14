import assert from 'node:assert/strict'
import test from 'node:test'

import {
  resolveCurrentPage,
  resolveRequestedStructuralId,
  resolveSelectedTocSequenceId,
} from './reviewStateResolution.js'

test('resolves the first available page when the current page is no longer present', () => {
  assert.equal(resolveCurrentPage([3, 5], 2), 3)
  assert.equal(resolveCurrentPage([3, 5], 5), 5)
  assert.equal(resolveCurrentPage([], 2), 1)
})

test('keeps a valid TOC selection and otherwise prefers a sequence on the active page', () => {
  assert.equal(
    resolveSelectedTocSequenceId({
      sequenceIds: ['toc-1', 'toc-2'],
      activePageSequenceIds: ['toc-2'],
      selectedId: 'toc-1',
    }),
    'toc-1',
  )
  assert.equal(
    resolveSelectedTocSequenceId({
      sequenceIds: ['toc-1', 'toc-2'],
      activePageSequenceIds: ['toc-2'],
      selectedId: 'toc-missing',
    }),
    'toc-2',
  )
  assert.equal(
    resolveSelectedTocSequenceId({ sequenceIds: [], activePageSequenceIds: [], selectedId: 'toc-1' }),
    null,
  )
})

test('normalizes a requested structural target for same-render page validation', () => {
  assert.equal(resolveRequestedStructuralId(' table-42 '), 'table-42')
  assert.equal(resolveRequestedStructuralId('  '), null)
  assert.equal(resolveRequestedStructuralId(undefined), null)
})
