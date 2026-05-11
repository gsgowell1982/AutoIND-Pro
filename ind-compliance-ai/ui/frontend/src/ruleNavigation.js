export function resolveStructuralSelection({
  activePageStructuralSelectionOrder,
  selectedStructuralId,
  pendingStructuralId,
}) {
  const orderedIds = Array.isArray(activePageStructuralSelectionOrder)
    ? activePageStructuralSelectionOrder.filter((item) => typeof item === 'string' && item.trim())
    : []
  const currentSelectedId =
    typeof selectedStructuralId === 'string' && selectedStructuralId.trim()
      ? selectedStructuralId
      : null
  const currentPendingId =
    typeof pendingStructuralId === 'string' && pendingStructuralId.trim()
      ? pendingStructuralId
      : null

  if (orderedIds.length === 0) {
    return {
      selectedStructuralId: null,
      pendingStructuralId: currentPendingId,
    }
  }

  if (currentPendingId) {
    if (orderedIds.includes(currentPendingId)) {
      return {
        selectedStructuralId: currentPendingId,
        pendingStructuralId: null,
      }
    }
    return {
      selectedStructuralId: null,
      pendingStructuralId: null,
    }
  }

  if (currentSelectedId && orderedIds.includes(currentSelectedId)) {
    return {
      selectedStructuralId: currentSelectedId,
      pendingStructuralId: null,
    }
  }

  return {
    selectedStructuralId: orderedIds[0],
    pendingStructuralId: null,
  }
}

export function resolveRuleNavigationLanding({
  currentPage,
  activePageStructuralSelectionOrder,
  activePageStructuralBoundingBoxIds,
  selectedStructuralId,
  targetPage,
  targetStructuralId,
}) {
  const normalizedTargetPage =
    typeof targetPage === 'number' && Number.isFinite(targetPage) && targetPage > 0
      ? targetPage
      : null
  const orderedIds = Array.isArray(activePageStructuralSelectionOrder)
    ? activePageStructuralSelectionOrder.filter((item) => typeof item === 'string' && item.trim())
    : []
  const bboxIds = Array.isArray(activePageStructuralBoundingBoxIds)
    ? activePageStructuralBoundingBoxIds.filter((item) => typeof item === 'string' && item.trim())
    : []
  const normalizedTargetStructuralId =
    typeof targetStructuralId === 'string' && targetStructuralId.trim()
      ? targetStructuralId
      : null
  const normalizedSelectedStructuralId =
    typeof selectedStructuralId === 'string' && selectedStructuralId.trim()
      ? selectedStructuralId
      : null

  if (!normalizedTargetPage) {
    return {
      status: 'idle',
      reason: 'no_active_navigation_target',
      isFinal: true,
    }
  }

  if (currentPage !== normalizedTargetPage) {
    return {
      status: 'waiting_for_page',
      reason: 'awaiting_target_page',
      isFinal: false,
    }
  }

  if (!normalizedTargetStructuralId) {
    return {
      status: 'confirmed_page',
      reason: 'page_navigation_confirmed',
      isFinal: true,
    }
  }

  if (
    normalizedSelectedStructuralId === normalizedTargetStructuralId &&
    !bboxIds.includes(normalizedTargetStructuralId)
  ) {
    return {
      status: 'failed_structural_bbox_missing',
      reason: 'target_structural_bbox_not_visible',
      isFinal: true,
    }
  }

  if (normalizedSelectedStructuralId === normalizedTargetStructuralId) {
    return {
      status: 'confirmed_structural',
      reason: 'selected_structural_target_confirmed',
      isFinal: true,
    }
  }

  if (!orderedIds.includes(normalizedTargetStructuralId)) {
    return {
      status: 'failed_missing_structural',
      reason: 'target_structural_id_not_found_on_page',
      isFinal: true,
    }
  }

  return {
    status: 'waiting_for_selection',
    reason: 'awaiting_structural_selection_confirmation',
    isFinal: false,
  }
}

export function resolveEndToEndRuleNavigationVerification({
  backendNavigationStatus,
  backendNavigationReason,
  landingStatus,
  landingReason,
}) {
  const normalizedBackendStatus =
    typeof backendNavigationStatus === 'string' && backendNavigationStatus.trim()
      ? backendNavigationStatus
      : null
  const normalizedLandingStatus =
    typeof landingStatus === 'string' && landingStatus.trim()
      ? landingStatus
      : null

  if (!normalizedBackendStatus || !normalizedLandingStatus) {
    return {
      status: 'verification_unavailable',
      reason: 'missing_backend_or_frontend_navigation_state',
      isFinal: true,
    }
  }

  if (normalizedLandingStatus.startsWith('waiting_')) {
    return {
      status: 'verification_waiting',
      reason: 'frontend_landing_not_final',
      isFinal: false,
    }
  }

  if (
    normalizedBackendStatus === 'resolved_structural' &&
    normalizedLandingStatus === 'confirmed_structural'
  ) {
    return {
      status: 'verified_structural',
      reason: 'backend_and_frontend_confirmed_structural_target',
      isFinal: true,
    }
  }

  if (
    normalizedBackendStatus === 'resolved_page' &&
    normalizedLandingStatus === 'confirmed_page'
  ) {
    return {
      status: 'verified_page',
      reason: 'backend_and_frontend_confirmed_page_target',
      isFinal: true,
    }
  }

  if (
    normalizedBackendStatus === 'resolved_structural' &&
    normalizedLandingStatus.startsWith('failed_')
  ) {
    return {
      status: 'verification_failed',
      reason: 'frontend_failed_after_backend_structural_resolution',
      isFinal: true,
    }
  }

  if (
    normalizedBackendStatus === 'resolved_page' &&
    normalizedLandingStatus.startsWith('failed_')
  ) {
    return {
      status: 'verification_failed',
      reason: 'frontend_failed_after_backend_page_resolution',
      isFinal: true,
    }
  }

  return {
    status: 'verification_degraded',
    reason:
      backendNavigationReason && landingReason
        ? `${backendNavigationReason} -> ${landingReason}`
        : 'backend_frontend_navigation_states_do_not_match',
    isFinal: true,
  }
}

export function buildRuleNavigationAuditRecord({
  ruleId,
  requirementId,
  citationAnchor,
  label,
  targetPage,
  targetStructuralId,
  backendNavigationStatus,
  backendNavigationReason,
  landingStatus,
  landingReason,
  verificationStatus,
  verificationReason,
}) {
  let severity = 'info'
  if (verificationStatus === 'verified_structural' || verificationStatus === 'verified_page') {
    severity = 'success'
  } else if (
    verificationStatus === 'verification_failed' ||
    verificationStatus === 'verification_degraded'
  ) {
    severity = 'warning'
  }

  return {
    ruleId: ruleId?.trim() || null,
    requirementId: requirementId?.trim() || null,
    citationAnchor: citationAnchor?.trim() || null,
    label: label?.trim() || 'rule-navigation',
    targetPage: typeof targetPage === 'number' && Number.isFinite(targetPage) ? targetPage : null,
    targetStructuralId: targetStructuralId?.trim() || null,
    backendNavigationStatus: backendNavigationStatus?.trim() || null,
    backendNavigationReason: backendNavigationReason?.trim() || null,
    landingStatus: landingStatus?.trim() || null,
    landingReason: landingReason?.trim() || null,
    verificationStatus: verificationStatus?.trim() || null,
    verificationReason: verificationReason?.trim() || null,
    severity,
    isTerminal: verificationStatus !== 'verification_waiting',
  }
}

export function buildEndToEndNavigationAuditChain({
  backendAuditRecord,
  landingStatus,
  landingReason,
  verificationStatus,
  verificationReason,
}) {
  const record = backendAuditRecord ?? {}
  let severity = 'info'
  if (verificationStatus === 'verified_structural' || verificationStatus === 'verified_page') {
    severity = 'success'
  } else if (
    verificationStatus === 'verification_failed' ||
    verificationStatus === 'verification_degraded'
  ) {
    severity = 'warning'
  }

  return {
    auditRecordId: record.audit_record_id ?? null,
    ruleId: record.rule_id ?? null,
    requirementId: record.requirement_id ?? null,
    sourceClauseId: record.source_clause_id ?? null,
    citationAnchor: record.citation_anchor ?? null,
    targetKind: record.target_kind ?? null,
    targetLabel: record.target_label ?? null,
    filename: record.filename ?? null,
    targetPage: record.target_page ?? null,
    targetStructuralId: record.target_structural_id ?? null,
    backendNavigationStatus: record.backend_navigation_status ?? null,
    backendNavigationReason: record.backend_navigation_reason ?? null,
    landingStatus: landingStatus?.trim() || null,
    landingReason: landingReason?.trim() || null,
    verificationStatus: verificationStatus?.trim() || null,
    verificationReason: verificationReason?.trim() || null,
    evidenceRefs: Array.isArray(record.evidence_refs) ? record.evidence_refs : [],
    sectionOutlineIndices: Array.isArray(record.section_outline_indices) ? record.section_outline_indices : [],
    severity,
    isTerminal: verificationStatus !== 'verification_waiting',
  }
}
