export interface StructuralSelectionResolutionInput {
  activePageStructuralSelectionOrder: string[]
  selectedStructuralId: string | null
  pendingStructuralId: string | null
}

export interface StructuralSelectionResolution {
  selectedStructuralId: string | null
  pendingStructuralId: string | null
}

export interface RuleNavigationLandingInput {
  currentPage: number
  activePageStructuralSelectionOrder: string[]
  activePageStructuralBoundingBoxIds: string[]
  selectedStructuralId: string | null
  targetPage: number | null
  targetStructuralId: string | null
}

export interface RuleNavigationLandingResolution {
  status:
    | 'idle'
    | 'waiting_for_page'
    | 'waiting_for_selection'
    | 'confirmed_structural'
    | 'confirmed_page'
    | 'failed_missing_structural'
    | 'failed_structural_bbox_missing'
  reason: string
  isFinal: boolean
}

export interface EndToEndRuleNavigationVerificationInput {
  backendNavigationStatus: string | null
  backendNavigationReason: string | null
  landingStatus: string | null
  landingReason: string | null
}

export interface EndToEndRuleNavigationVerificationResolution {
  status:
    | 'verification_unavailable'
    | 'verification_waiting'
    | 'verified_structural'
    | 'verified_page'
    | 'verification_failed'
    | 'verification_degraded'
  reason: string
  isFinal: boolean
}

export interface RuntimeRuleNavigationAuditRecordInput {
  ruleId: string | null
  requirementId: string | null
  citationAnchor: string | null
  label: string | null
  targetPage: number | null
  targetStructuralId: string | null
  backendNavigationStatus: string | null
  backendNavigationReason: string | null
  landingStatus: string | null
  landingReason: string | null
  verificationStatus: string | null
  verificationReason: string | null
}

export interface RuntimeRuleNavigationAuditRecord {
  ruleId: string | null
  requirementId: string | null
  citationAnchor: string | null
  label: string
  targetPage: number | null
  targetStructuralId: string | null
  backendNavigationStatus: string | null
  backendNavigationReason: string | null
  landingStatus: string | null
  landingReason: string | null
  verificationStatus: string | null
  verificationReason: string | null
  severity: 'info' | 'success' | 'warning'
  isTerminal: boolean
}

export interface EndToEndNavigationAuditChainInput {
  backendAuditRecord: import('./types').RuleNavigationAuditRecord | null
  landingStatus: string | null
  landingReason: string | null
  verificationStatus: string | null
  verificationReason: string | null
}

export interface EndToEndNavigationAuditChain {
  auditRecordId: string | null
  ruleId: string | null
  requirementId: string | null
  sourceClauseId: string | null
  citationAnchor: string | null
  targetKind: string | null
  targetLabel: string | null
  filename: string | null
  targetPage: number | null
  targetStructuralId: string | null
  backendNavigationStatus: string | null
  backendNavigationReason: string | null
  landingStatus: string | null
  landingReason: string | null
  verificationStatus: string | null
  verificationReason: string | null
  evidenceRefs: string[]
  sectionOutlineIndices: string[]
  severity: 'info' | 'success' | 'warning'
  isTerminal: boolean
}

export interface EndToEndRuleNavigationVerificationInput {
  backendNavigationStatus: string | null
  backendNavigationReason: string | null
  landingStatus: string | null
  landingReason: string | null
}

export interface EndToEndRuleNavigationVerificationResolution {
  status:
    | 'verification_unavailable'
    | 'verification_waiting'
    | 'verified_structural'
    | 'verified_page'
    | 'verification_failed'
    | 'verification_degraded'
  reason: string
  isFinal: boolean
}

export function resolveStructuralSelection(
  input: StructuralSelectionResolutionInput,
): StructuralSelectionResolution

export function resolveRuleNavigationLanding(
  input: RuleNavigationLandingInput,
): RuleNavigationLandingResolution

export function resolveEndToEndRuleNavigationVerification(
  input: EndToEndRuleNavigationVerificationInput,
): EndToEndRuleNavigationVerificationResolution

export function buildRuleNavigationAuditRecord(
  input: RuntimeRuleNavigationAuditRecordInput,
): RuntimeRuleNavigationAuditRecord

export function buildEndToEndNavigationAuditChain(
  input: EndToEndNavigationAuditChainInput,
): EndToEndNavigationAuditChain

export function resolveEndToEndRuleNavigationVerification(
  input: EndToEndRuleNavigationVerificationInput,
): EndToEndRuleNavigationVerificationResolution
