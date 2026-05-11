import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildEndToEndNavigationAuditChain,
  buildRuleNavigationAuditRecord,
  resolveEndToEndRuleNavigationVerification,
  resolveRuleNavigationLanding,
  resolveStructuralSelection,
} from '../src/ruleNavigation.js'

test('keeps an existing selected structural id when it is still on the active page', () => {
  const result = resolveStructuralSelection({
    activePageStructuralSelectionOrder: ['tbl_001', 'img_001'],
    selectedStructuralId: 'img_001',
    pendingStructuralId: null,
  })

  assert.deepEqual(result, {
    selectedStructuralId: 'img_001',
    pendingStructuralId: null,
  })
})

test('promotes a pending structural target when the new page contains it', () => {
  const result = resolveStructuralSelection({
    activePageStructuralSelectionOrder: ['tbl_001', 'eq_003', 'img_001'],
    selectedStructuralId: 'tbl_001',
    pendingStructuralId: 'eq_003',
  })

  assert.deepEqual(result, {
    selectedStructuralId: 'eq_003',
    pendingStructuralId: null,
  })
})

test('clears selection instead of falling back to the first structural item when a pending target is missing', () => {
  const result = resolveStructuralSelection({
    activePageStructuralSelectionOrder: ['tbl_001', 'img_001'],
    selectedStructuralId: 'tbl_001',
    pendingStructuralId: 'eq_999',
  })

  assert.deepEqual(result, {
    selectedStructuralId: null,
    pendingStructuralId: null,
  })
})

test('falls back to the first structural item only when there is no pending target', () => {
  const result = resolveStructuralSelection({
    activePageStructuralSelectionOrder: ['tbl_001', 'img_001'],
    selectedStructuralId: null,
    pendingStructuralId: null,
  })

  assert.deepEqual(result, {
    selectedStructuralId: 'tbl_001',
    pendingStructuralId: null,
  })
})

test('marks navigation as waiting while the target page has not been reached yet', () => {
  const result = resolveRuleNavigationLanding({
    currentPage: 1,
    activePageStructuralSelectionOrder: ['tbl_001'],
    selectedStructuralId: null,
    targetPage: 3,
    targetStructuralId: 'tbl_010',
  })

  assert.deepEqual(result, {
    status: 'waiting_for_page',
    reason: 'awaiting_target_page',
    isFinal: false,
  })
})

test('confirms structural landing when the selected structure matches the requested target', () => {
  const result = resolveRuleNavigationLanding({
    currentPage: 3,
    activePageStructuralSelectionOrder: ['tbl_010', 'img_001'],
    activePageStructuralBoundingBoxIds: ['tbl_010', 'img_001'],
    selectedStructuralId: 'tbl_010',
    targetPage: 3,
    targetStructuralId: 'tbl_010',
  })

  assert.deepEqual(result, {
    status: 'confirmed_structural',
    reason: 'selected_structural_target_confirmed',
    isFinal: true,
  })
})

test('confirms page-only landing when a rule jump has no structural target', () => {
  const result = resolveRuleNavigationLanding({
    currentPage: 5,
    activePageStructuralSelectionOrder: ['eq_001'],
    activePageStructuralBoundingBoxIds: ['eq_001'],
    selectedStructuralId: 'eq_001',
    targetPage: 5,
    targetStructuralId: null,
  })

  assert.deepEqual(result, {
    status: 'confirmed_page',
    reason: 'page_navigation_confirmed',
    isFinal: true,
  })
})

test('fails structural landing when the target page is reached but the requested structure does not exist', () => {
  const result = resolveRuleNavigationLanding({
    currentPage: 4,
    activePageStructuralSelectionOrder: ['tbl_001', 'img_001'],
    activePageStructuralBoundingBoxIds: ['tbl_001', 'img_001'],
    selectedStructuralId: 'tbl_001',
    targetPage: 4,
    targetStructuralId: 'eq_999',
  })

  assert.deepEqual(result, {
    status: 'failed_missing_structural',
    reason: 'target_structural_id_not_found_on_page',
    isFinal: true,
  })
})

test('fails structural landing when the target structure is selected but no bbox-bearing object exists on the page', () => {
  const result = resolveRuleNavigationLanding({
    currentPage: 6,
    activePageStructuralSelectionOrder: ['tbl_010', 'img_001'],
    activePageStructuralBoundingBoxIds: ['img_001'],
    selectedStructuralId: 'tbl_010',
    targetPage: 6,
    targetStructuralId: 'tbl_010',
  })

  assert.deepEqual(result, {
    status: 'failed_structural_bbox_missing',
    reason: 'target_structural_bbox_not_visible',
    isFinal: true,
  })
})

test('verifies structural navigation when backend and frontend both confirm the same structural landing', () => {
  const result = resolveEndToEndRuleNavigationVerification({
    backendNavigationStatus: 'resolved_structural',
    backendNavigationReason: 'resolved_from_structural_evidence',
    landingStatus: 'confirmed_structural',
    landingReason: 'selected_structural_target_confirmed',
  })

  assert.deepEqual(result, {
    status: 'verified_structural',
    reason: 'backend_and_frontend_confirmed_structural_target',
    isFinal: true,
  })
})

test('verifies page navigation when backend and frontend agree on a page-level resolution', () => {
  const result = resolveEndToEndRuleNavigationVerification({
    backendNavigationStatus: 'resolved_page',
    backendNavigationReason: 'resolved_from_text_evidence',
    landingStatus: 'confirmed_page',
    landingReason: 'page_navigation_confirmed',
  })

  assert.deepEqual(result, {
    status: 'verified_page',
    reason: 'backend_and_frontend_confirmed_page_target',
    isFinal: true,
  })
})

test('keeps verification in waiting state while frontend landing is not final', () => {
  const result = resolveEndToEndRuleNavigationVerification({
    backendNavigationStatus: 'resolved_structural',
    backendNavigationReason: 'resolved_from_structural_evidence',
    landingStatus: 'waiting_for_selection',
    landingReason: 'awaiting_structural_selection_confirmation',
  })

  assert.deepEqual(result, {
    status: 'verification_waiting',
    reason: 'frontend_landing_not_final',
    isFinal: false,
  })
})

test('marks verification failed when backend promised structural navigation but frontend landing fails', () => {
  const result = resolveEndToEndRuleNavigationVerification({
    backendNavigationStatus: 'resolved_structural',
    backendNavigationReason: 'resolved_from_structural_evidence',
    landingStatus: 'failed_structural_bbox_missing',
    landingReason: 'target_structural_bbox_not_visible',
  })

  assert.deepEqual(result, {
    status: 'verification_failed',
    reason: 'frontend_failed_after_backend_structural_resolution',
    isFinal: true,
  })
})

test('builds a structured audit record for a verified structural navigation', () => {
  const record = buildRuleNavigationAuditRecord({
    ruleId: 'SR-CTD-002',
    requirementId:
      'cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission',
    citationAnchor: 'cn_drug_registration_classification_and_dossier_requirements#art_015',
    label: 'rule:SR-CTD-002:32-body-data.pdf',
    targetPage: 1,
    targetStructuralId: 'tbl_001',
    backendNavigationStatus: 'resolved_structural',
    backendNavigationReason: 'resolved_from_structural_evidence',
    landingStatus: 'confirmed_structural',
    landingReason: 'selected_structural_target_confirmed',
    verificationStatus: 'verified_structural',
    verificationReason: 'backend_and_frontend_confirmed_structural_target',
  })

  assert.deepEqual(record, {
    ruleId: 'SR-CTD-002',
    requirementId:
      'cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission',
    citationAnchor: 'cn_drug_registration_classification_and_dossier_requirements#art_015',
    label: 'rule:SR-CTD-002:32-body-data.pdf',
    targetPage: 1,
    targetStructuralId: 'tbl_001',
    backendNavigationStatus: 'resolved_structural',
    backendNavigationReason: 'resolved_from_structural_evidence',
    landingStatus: 'confirmed_structural',
    landingReason: 'selected_structural_target_confirmed',
    verificationStatus: 'verified_structural',
    verificationReason: 'backend_and_frontend_confirmed_structural_target',
    severity: 'success',
    isTerminal: true,
  })
})

test('classifies degraded audit records as warning severity', () => {
  const record = buildRuleNavigationAuditRecord({
    ruleId: 'SR-CTD-002',
    requirementId:
      'cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission',
    citationAnchor: 'cn_drug_registration_classification_and_dossier_requirements#art_015',
    label: 'rule:SR-CTD-002:weak-snippet',
    targetPage: 7,
    targetStructuralId: null,
    backendNavigationStatus: 'resolved_structural',
    backendNavigationReason: 'resolved_from_structural_evidence',
    landingStatus: 'confirmed_page',
    landingReason: 'page_navigation_confirmed',
    verificationStatus: 'verification_degraded',
    verificationReason: 'resolved_from_structural_evidence -> page_navigation_confirmed',
  })

  assert.equal(record.severity, 'warning')
  assert.equal(record.isTerminal, true)
})

test('classifies waiting audit records as non-terminal info severity', () => {
  const record = buildRuleNavigationAuditRecord({
    ruleId: 'SR-CTD-002',
    requirementId:
      'cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission',
    citationAnchor: 'cn_drug_registration_classification_and_dossier_requirements#art_015',
    label: 'rule:SR-CTD-002:pending-target',
    targetPage: 3,
    targetStructuralId: 'eq_003',
    backendNavigationStatus: 'resolved_structural',
    backendNavigationReason: 'resolved_from_structural_evidence',
    landingStatus: 'waiting_for_selection',
    landingReason: 'awaiting_structural_selection_confirmation',
    verificationStatus: 'verification_waiting',
    verificationReason: 'frontend_landing_not_final',
  })

  assert.equal(record.severity, 'info')
  assert.equal(record.isTerminal, false)
})

test('builds an end-to-end audit chain by merging backend seed with frontend verification', () => {
  const chain = buildEndToEndNavigationAuditChain({
    backendAuditRecord: {
      audit_record_id: 'SR-CTD-002:matched_document:01',
      rule_id: 'SR-CTD-002',
      requirement_id:
        'cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission',
      source_clause_id: 'cn_drug_registration_classification_and_dossier_requirements:art_015',
      citation_anchor: 'cn_drug_registration_classification_and_dossier_requirements#art_015',
      target_kind: 'matched_document',
      target_label: '32-body-data.pdf',
      filename: '32-body-data.pdf',
      target_page: 1,
      target_structural_id: 'tbl_001',
      backend_navigation_status: 'resolved_structural',
      backend_navigation_reason: 'resolved_from_structural_evidence',
      evidence_refs: ['ce_table_tbl_001'],
      section_outline_indices: ['3.2.S.1'],
    },
    landingStatus: 'confirmed_structural',
    landingReason: 'selected_structural_target_confirmed',
    verificationStatus: 'verified_structural',
    verificationReason: 'backend_and_frontend_confirmed_structural_target',
  })

  assert.deepEqual(chain, {
    auditRecordId: 'SR-CTD-002:matched_document:01',
    ruleId: 'SR-CTD-002',
    requirementId:
      'cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission',
    sourceClauseId: 'cn_drug_registration_classification_and_dossier_requirements:art_015',
    citationAnchor: 'cn_drug_registration_classification_and_dossier_requirements#art_015',
    targetKind: 'matched_document',
    targetLabel: '32-body-data.pdf',
    filename: '32-body-data.pdf',
    targetPage: 1,
    targetStructuralId: 'tbl_001',
    backendNavigationStatus: 'resolved_structural',
    backendNavigationReason: 'resolved_from_structural_evidence',
    landingStatus: 'confirmed_structural',
    landingReason: 'selected_structural_target_confirmed',
    verificationStatus: 'verified_structural',
    verificationReason: 'backend_and_frontend_confirmed_structural_target',
    evidenceRefs: ['ce_table_tbl_001'],
    sectionOutlineIndices: ['3.2.S.1'],
    severity: 'success',
    isTerminal: true,
  })
})

test('keeps end-to-end audit chain as warning severity when frontend verification degrades or fails', () => {
  const chain = buildEndToEndNavigationAuditChain({
    backendAuditRecord: {
      audit_record_id: 'SR-CTD-002:weak_signal_snippet:01',
      rule_id: 'SR-CTD-002',
      requirement_id:
        'cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission',
      citation_anchor: 'cn_drug_registration_classification_and_dossier_requirements#art_015',
      target_kind: 'weak_signal_snippet',
      target_label: 'submission-summary.pdf',
      filename: 'submission-summary.pdf',
      target_page: 7,
      target_structural_id: null,
      backend_navigation_status: 'resolved_page',
      backend_navigation_reason: 'resolved_from_text_evidence',
    },
    landingStatus: 'confirmed_page',
    landingReason: 'page_navigation_confirmed',
    verificationStatus: 'verification_degraded',
    verificationReason: 'resolved_from_text_evidence -> page_navigation_confirmed',
  })

  assert.equal(chain.severity, 'warning')
  assert.equal(chain.isTerminal, true)
})
