export interface FileStatus {
  file_id: string
  filename: string
  relative_path?: string
  status: string
  message?: string
  progress: number
}

export interface UploadScopeOverview {
  upload_mode: string
  upload_mode_label: string
  likely_scopes: string[]
  likely_scope_labels: string[]
  file_count: number
  sequence_root_count: number
  application_root_count: number
  signal_filenames: string[]
  detection_basis: string
  scope_notice: string
}

export interface DemoSampleMetadata {
  sample_id: string
  synthetic: boolean
  sample_kind: string
  evidence_boundary: string
}

export interface UploadJobResponse {
  job_id: string
  status: string
  progress: number
  created_at: string
  updated_at: string
  files: FileStatus[]
  upload_scope_overview?: UploadScopeOverview
  demo_sample?: DemoSampleMetadata | null
}

export interface JobStatusResponse {
  job_id: string
  status: string
  progress: number
  created_at: string
  updated_at: string
  files: FileStatus[]
  upload_scope_overview?: UploadScopeOverview
  demo_sample?: DemoSampleMetadata | null
}

export interface BoundingBox {
  id: string
  page_number: number
  text: string
  block_type?: string
  semantic_role?: string
  bbox: {
    x0: number
    y0: number
    x1: number
    y1: number
  }
}

export interface PdfPageMeta {
  page_number: number
  width: number
  height: number
  text: string
  block_count: number
  image_count?: number
  algorithm_count?: number
  equation_count?: number
  table_count?: number
  toc_count?: number
  layout_mode?: string
  layout_confidence?: number
}

export interface PdfImageBlock {
  image_id: string
  page: number
  bbox: [number, number, number, number]
  title?: string
  figure_ref?: string
  caption_text?: string
  image_kind_guess?: string
}

export interface PdfEquationBlock {
  equation_id: string
  page: number
  bbox: [number, number, number, number]
  text: string
  equation_label?: string
  source?: string
  semantic_role?: string
  source_block_ids?: string[]
}

export interface PdfAlgorithmBlock {
  algorithm_id: string
  page: number
  bbox: [number, number, number, number]
  algorithm_ref?: string
  title?: string
  content_text?: string
  line_count?: number
  lines?: string[]
  continued_from_previous_page?: boolean
  continues_to_next_page?: boolean
  semantic_role?: string
}

export interface PdfTableCell {
  row: number
  col: number
  text: string
  bbox: [number, number, number, number]
  rowspan?: number
  colspan?: number
}

export interface PdfTableMergedRow {
  row: number
  raw_row?: number
  kind: string
  text: string
  colspan?: number
  source?: string
}

export interface PdfTableSemanticCompactionGroup {
  anchor_text: string
  anchor_data_row?: number
  source_data_row_start?: number
  source_data_row_end?: number
  source_data_row_count?: number
}

export interface PdfTableSemanticCompaction {
  applied: boolean
  strategy: string
  source_data_row_count?: number
  compacted_data_row_count?: number
  anchor_column?: number
  groups?: PdfTableSemanticCompactionGroup[]
}

export interface PdfTableAst {
  table_id: string
  page: number
  bbox: [number, number, number, number]
  title?: string
  detection_method?: string
  col_count?: number
  row_count?: number
  raw_row_count?: number
  display_row_count?: number
  data_row_count?: number
  logical_row_count?: number
  header: Array<{ text: string; col: number }>
  cells: PdfTableCell[]
  merged_rows?: PdfTableMergedRow[]
  column_hash?: string
  continued_from?: string
  continued_to?: string[]
  semantic_compaction?: PdfTableSemanticCompaction
  continuation_source?: {
    source_table_id: string
    strategy: string
    similarity: number
    hint_table_id?: string
    hint_similarity?: number
    inherited_fields?: string[]
  }
}

export interface PdfFigureRef {
  figure_ref: string
  image_id: string
  title?: string
  page: number
  bbox: [number, number, number, number]
}

export interface PdfTocBlock {
  toc_id: string
  page: number
  bbox: [number, number, number, number]
  title?: string
  entry_count?: number
  wrapped_entry_count?: number
  missing_page_locator_count?: number
  title_inferred?: boolean
  review_required?: boolean
  review_item_count?: number
  toc_sequence_id?: string
  toc_sequence_page_index?: number
  toc_sequence_length?: number
}

export interface PdfTocRootSection {
  outline_index?: string
  text?: string
  page?: number
  entry_count?: number
  cross_page_parent_entry_count?: number
}

export interface PdfTocPageEntrySpan {
  page: number
  toc_ids?: string[]
  entry_count?: number
  first_sequence_entry_index?: number
  last_sequence_entry_index?: number
  cross_page_parent_entry_count?: number
}

export interface PdfTocNavigationSummary {
  cross_page_parent_link_count?: number
  root_sections?: PdfTocRootSection[]
  page_entry_spans?: PdfTocPageEntrySpan[]
}

export interface PdfTocSequence {
  toc_sequence_id: string
  title?: string
  pages?: number[]
  page_span?: [number, number]
  entry_count?: number
  root_entry_count?: number
  max_branching_factor?: number
  leaf_entry_count?: number
  navigation_summary?: PdfTocNavigationSummary
  root_nodes?: Array<Record<string, unknown>>
}

export interface RuleCheckSectionRef {
  module_label?: string | null
  outline_index?: string | null
  section_title?: string | null
  page_span?: Array<number | null>
  jump_page?: number
}

export interface RuleCheckMatchedDocument {
  document_id?: string
  filename?: string
  module_label?: string
  signal_hits?: string[]
  evidence_refs?: string[]
  section_refs?: RuleCheckSectionRef[]
  jump_page?: number
  structural_id?: string
  navigation_status?: string
  navigation_reason?: string
  non_embedded_non_standard_fonts?: string[]
  non_embedded_non_standard_font_records?: Array<{
    font_label?: string
    base_font?: string
    resource_name?: string
    font_type?: string
    xref?: number
    is_embedded?: boolean
    font_file_kind?: string
    page_numbers?: number[]
  }>
}

export interface RuleCheckWeakSignalSnippet {
  document_id?: string
  filename?: string
  evidence_id?: string
  page?: number
  preview?: string
  jump_page?: number
  structural_id?: string
  navigation_status?: string
  navigation_reason?: string
}

export interface RuleCheckLlmDecision {
  decision?: string
  reason?: string
  provider?: string
}

export interface GuidanceTargetDetail {
  target_type: string
  label: string
  description: string
  source_filename?: string
  source_document_id?: string
  source_extension_title?: string
  source_parent_pointer?: string
  source_leaf_titles?: string[]
  source_leaf_hrefs?: string[]
}

export interface RuleRemediationGuidanceItem {
  guidance_code: string
  guidance_title: string
  guidance_summary: string
  guidance_priority: string
  guidance_steps: string[]
  guidance_targets: string[]
  guidance_target_details: GuidanceTargetDetail[]
}

export interface RuleExtensionIssueBundle {
  bundle_id: string
  source_filename?: string
  source_document_id?: string
  extension_title?: string
  leaf_titles?: string[]
  leaf_hrefs?: string[]
  invalid_leaf_hrefs?: string[]
  parent_pointer?: string
  parent_local_tag?: string
  issue_codes?: string[]
  issue_labels?: string[]
  primary_issue_code?: string
  primary_issue_label?: string
  recommended_fix_sequence?: string[]
  recommended_fix_labels?: string[]
  issue_diff_rows?: {
    issue_code: string
    issue_label: string
    field_path?: string
    current_value?: string
    target_value?: string
    recommended_target_value?: string
    target_value_candidates?: string[]
    recommendation_basis?: string
    has_recommendation_conflict?: boolean
    recommendation_conflict_summary?: string
    tie_break_guidance_title?: string
    tie_break_guidance_steps?: string[]
    tie_break_focus_items?: {
      label: string
      value: string
    }[]
    tie_break_candidate_comparisons?: {
      candidate_title: string
      supports?: string[]
      concerns?: string[]
    }[]
    note?: string
    suggested_action_title?: string
    suggested_action_steps?: string[]
    verification_checks?: string[]
    suggested_snippet?: string
  }[]
}

export interface RuleStructureAuditRow {
  document_id?: string | null
  filename?: string | null
  toc_outline_count?: number
  toc_root_outline_count?: number
  matched_outline_count?: number
  matched_root_outline_count?: number
  root_outline_coverage_ratio?: number | null
  root_page_order_ready?: boolean
  root_page_offset_values?: number[]
  root_page_offset_ready?: boolean
  projected_root_page_values?: number[]
  root_page_span_ready?: boolean
  direct_child_coverage_ratio?: number | null
  direct_child_coverage_ready?: boolean
  bounded_subtree_coverage_ratio?: number | null
  bounded_subtree_coverage_ready?: boolean
  missing_body_root_outline_indices?: string[]
  missing_body_direct_child_outline_indices?: string[]
  missing_body_direct_child_path_rows?: RuleStructureAuditMissingPathRow[]
  missing_body_bounded_subtree_outline_indices?: string[]
  missing_body_bounded_subtree_path_rows?: RuleStructureAuditMissingPathRow[]
  root_page_alignment_rows?: RuleStructureAuditRootPageAlignmentRow[]
  toc_body_alignment_path_rows?: RuleStructureAuditPathAlignmentRow[]
  navigation_targets?: RuleStructureAuditNavigationTarget[]
  alignment_ready?: boolean
}

export interface RuleStructureAuditNavigationTarget {
  target_kind: 'toc' | 'body_root'
  label: string
  page: number
  toc_sequence_id?: string | null
  outline_indices?: string[]
}

export interface RuleStructureAuditRootPageAlignmentRow {
  outline_index: string
  normalized_outline_index?: string
  toc_navigation_page?: number | null
  toc_page_locator_value: number
  body_page_start: number
  body_page_end: number
  projected_page?: number | null
  offset?: number | null
  toc_order_index?: number | null
  body_order_index?: number | null
  order_conflict?: boolean
  offset_conflict?: boolean
  span_conflict?: boolean
}

export interface RuleStructureAuditMissingPathRow {
  root_outline_index?: string | null
  root_normalized_outline_index?: string | null
  parent_outline_index?: string | null
  parent_normalized_outline_index?: string | null
  outline_index?: string | null
  normalized_outline_index?: string | null
  outline_path?: string | null
  text_path?: string | null
  page_locator_value?: number | null
  navigation_page?: number | null
  level?: number | null
  nearest_body_anchor_outline_index?: string | null
  nearest_body_anchor_page?: number | null
}

export interface RuleStructureAuditPathAlignmentRow {
  root_outline_index?: string | null
  root_normalized_outline_index?: string | null
  parent_outline_index?: string | null
  parent_normalized_outline_index?: string | null
  outline_index?: string | null
  normalized_outline_index?: string | null
  outline_path?: string | null
  text_path?: string | null
  page_locator_value?: number | null
  navigation_page?: number | null
  level?: number | null
  body_anchor_outline_index?: string | null
  body_anchor_page?: number | null
  body_anchor_kind?: string | null
  alignment_status?: string | null
  page_offset?: number | null
  has_body_match?: boolean
}

export interface RuleCheckDetails {
  requirement_id?: string
  source_clause_id?: string
  citation_anchor?: string
  regulation_severity?: string | null
  match_strength?: string
  matched_documents?: RuleCheckMatchedDocument[]
  weak_signal_snippets?: RuleCheckWeakSignalSnippet[]
  llm_decision?: RuleCheckLlmDecision
  structure_audit_rows?: RuleStructureAuditRow[]
  matched_extension_titles?: string[]
  invalid_extension_titles?: string[]
  invalid_leaf_hrefs?: string[]
  misplaced_extension_titles?: string[]
  allowed_extension_titles?: string[]
  expected_leaf_href_pattern?: string
  expected_leaf_href_note?: string
  extension_issue_bundles?: RuleExtensionIssueBundle[]
  remediation_guidance?: RuleRemediationGuidanceItem[]
}

export interface RuleCheckItem {
  rule_id: string
  category: string
  status: string
  message: string
  citation?: string
  scope?: string
  scope_status?: string
  basis?: {
    basis_kind: 'regulation' | 'system' | 'generic'
    basis_label: string
    basis_detail: string
    internal_anchor?: string | null
  }
  details?: RuleCheckDetails
}

export interface RuleCheckGroupBasisRef {
  citation: string
  basis_kind: 'regulation' | 'system' | 'generic'
  basis_label: string
}

export interface RuleCheckGroupSummary {
  group_id: string
  title: string
  description: string
  overall_status: string
  rule_ids: string[]
  focus_rule_ids: string[]
  total_rules: number
  applicable_rules: number
  fail_count: number
  warn_count: number
  pass_count: number
  na_count: number
  basis_refs?: RuleCheckGroupBasisRef[]
}

export interface RuleNavigationAuditRecord {
  audit_record_id: string
  rule_id: string
  requirement_id?: string | null
  source_clause_id?: string | null
  citation_anchor?: string | null
  target_kind: string
  target_label?: string | null
  filename?: string | null
  evidence_id?: string | null
  target_page?: number | null
  target_structural_id?: string | null
  backend_navigation_status?: string | null
  backend_navigation_reason?: string | null
  evidence_refs?: string[]
  section_outline_indices?: string[]
}

export interface RuleStructureAuditRecord extends RuleStructureAuditRow {
  audit_record_id: string
  rule_id: string
  citation_anchor?: string | null
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

export interface RuleChecksSummary {
  hard_failures?: number
  soft_risks?: number
  parsed_documents?: number
  rule_count?: number
  pass_rules?: number
  warn_rules?: number
  na_rules?: number
  upload_scope_overview?: UploadScopeOverview
  submission_scope?: Record<string, unknown>
  submission_scope_overview?: SubmissionScopeOverview
  scope_transition_overview?: ScopeTransitionOverview
}

export interface SubmissionScopeOverview {
  upload_mode: string
  upload_mode_label: string
  available_scopes: string[]
  available_scope_labels: string[]
  blocked_scopes: string[]
  blocked_scope_labels: string[]
  sequence_package_count: number
  regulatory_activity_count: number
  application_project_count: number
  scope_notice: string
}

export interface ScopeTransitionOverview {
  transition_status: string
  transition_status_label: string
  primary_reason_code: string
  primary_reason_label: string
  recommended_action_code: string
  recommended_action_label: string
  guidance_title: string
  guidance_summary: string
  guidance_priority: string
  guidance_steps: string[]
  guidance_targets: string[]
  guidance_target_details: GuidanceTargetDetail[]
  guidance_rule_groups: string[]
  guidance_rule_group_labels: string[]
  upload_mode: string
  upload_mode_label: string
  final_mode: string
  final_mode_label: string
  upload_scopes: string[]
  upload_scope_labels: string[]
  final_scopes: string[]
  final_scope_labels: string[]
  added_scopes: string[]
  added_scope_labels: string[]
  removed_scopes: string[]
  removed_scope_labels: string[]
  transition_notice: string
}

export interface RegulatoryReadinessSource {
  regulation_id: string
  source_file: string
  coverage_status: string
  clause_count: number
  covered_count: number
  partial_count: number
  deferred_count: number
  citation_only_count: number
  rule_candidate_count: number
  direct_ish_candidate_count: number
  requirement_count: number
  direct_rule_draft_count: number
  recommended_product_role: string
  default_triage: 'deterministic' | 'prerequisite_required' | 'human_review' | 'out_of_scope_low_roi'
  automation_boundary: string
  traceability_gap_count?: number
}

export interface RegulatoryReadinessProjection {
  schema_version: string
  phase: string
  phase_status: string
  recommended_next_action_code: string
  recommended_next_action_label: string
  rule_triage_categories: Array<'deterministic' | 'prerequisite_required' | 'human_review' | 'out_of_scope_low_roi'>
  sources: RegulatoryReadinessSource[]
  summary: {
    source_count: number
    closed_source_count: number
    demo_value_statement: string
  }
}

export interface DossierChecklistPrerequisiteFact {
  fact_key: string
  label: string
  description: string
  status: 'present' | 'missing'
  value?: unknown
  source?: string | null
  source_label?: string
  evidence_status?: 'local_evidence_present' | 'missing_required_prerequisite'
  confidence?: 'high' | 'medium' | 'none'
  review_action?: string
}

export interface DossierChecklistItem {
  requirement_id: string
  source_clause_id: string
  source_article_no: number
  section_no: number
  section_title: string
  source_heading: string
  registration_classes: string[]
  applicable_stage: string
  requirement_type: string
  requirement_level: string
  citation_anchor: string
  expected_material_evidence: string[]
  review_focus: string
  requirement_text_preview: string
  applicability_status: 'prerequisite_required' | 'human_review'
  default_triage: 'prerequisite_required' | 'human_review'
  blocking_prerequisite_fact_keys: string[]
  automation_boundary: string
}

export interface DossierChecklistProjection {
  schema_version: string
  phase: string
  regulation_id: string
  regulation_title: string
  source_file: string
  applicability_mode: 'prerequisite_required' | 'human_review'
  requirement_count: number
  prerequisite_facts: DossierChecklistPrerequisiteFact[]
  missing_prerequisite_fact_keys: string[]
  items: DossierChecklistItem[]
  summary: {
    requirement_count: number
    deterministic_decision_count: number
    prerequisite_required_count: number
    human_review_count: number
    present_prerequisite_fact_count?: number
    missing_prerequisite_fact_count?: number
    missing_prerequisite_fact_keys?: string[]
    evidence_boundary: string
  }
}

export interface DemoSummaryProjection {
  schema_version: string
  phase: string
  phase_status: string
  verdict_policy: string
  demo_capability_codes: string[]
  remaining_work_codes: string[]
  recommended_next_action_code: string
  recommended_next_action_label: string
  customer_demo_message: string
  evidence_boundary: string
  summary: {
    source_count: number
    closed_source_count: number
    source_readiness_ratio: number
    dossier_requirement_count: number
    deterministic_dossier_decision_count: number
    prerequisite_required_count: number
    human_review_count: number
    missing_prerequisite_fact_keys: string[]
    present_prerequisite_fact_count: number
    rule_check_count: number
    pass_rule_count: number
    warn_rule_count: number
    na_rule_count: number
    risk_count: number
    content_consistency_check_count?: number
    content_consistency_issue_count?: number
  }
}

export interface ContentConsistencyIssue {
  issue_code: string
  field_name: string
  sequence_package_id: string
  expected_value: string
  observed_value: string
  expected_source: string
  observed_source: string
  message: string
  review_recommendation: string
}

export interface ContentConsistencyCheck {
  check_id: string
  title: string
  description: string
  status: 'consistent' | 'review_required' | 'prerequisite_required'
  field_name: string
  comparable_package_count: number
  issue_count: number
  issues: ContentConsistencyIssue[]
  automation_boundary: string
}

export interface ContentConsistencyProjection {
  schema_version: string
  phase: string
  verdict_policy: string
  checks: ContentConsistencyCheck[]
  summary: {
    check_count: number
    comparable_check_count: number
    issue_count: number
    review_required_count: number
    prerequisite_required_count: number
    deterministic_rule_verdict_count: number
  }
  evidence_boundary: string
}

export interface DemoFlowStep {
  step_id: string
  title: string
  status: 'ready_for_demo' | 'prerequisite_prompt' | 'review_required' | 'not_available'
  talk_track: string
  focus_items: string[]
  asset_url?: string | null
}

export interface DemoFlowProjection {
  schema_version: string
  phase: string
  verdict_policy: string
  summary: {
    recommended_demo_mode: string
    walkthrough_step_count: number
    source_count: number
    closed_source_count: number
    missing_prerequisite_count: number
    issue_count: number
    deterministic_rule_verdict_count: number
  }
  walkthrough_steps: DemoFlowStep[]
  evidence_boundary: string
}

export interface DemoScenarioProjection {
  schema_version: string
  phase: string
  scenario_id: string
  scenario_status: string
  verdict_policy: string
  recommended_upload_mode: string
  demo_goal: string
  recommended_sample_profile: {
    sample_kind: string
    must_have_local_evidence: string[]
    intentionally_missing_prerequisites: string[]
    expected_review_focus: string[]
  }
  summary: {
    walkthrough_step_count: number
    expected_issue_count: number
    expected_missing_prerequisite_count: number
    deterministic_rule_verdict_count: number
  }
  demo_success_criteria: string[]
  primary_asset_urls: {
    demo_report_markdown?: string | null
  }
  do_not_claim: string[]
  evidence_boundary: string
}

export interface DemoRunStep {
  step_id: string
  title: string
  status: 'ready_for_demo' | 'prerequisite_prompt' | 'review_required' | 'not_available'
  talk_track: string
  focus_items: string[]
  asset_urls?: {
    demo_report_markdown?: string | null
    demo_script_markdown?: string | null
  }
}

export interface DemoRunProjection {
  schema_version: string
  phase: string
  run_mode: string
  run_status: 'ready' | 'ready_with_review_items' | 'asset_missing'
  verdict_policy: string
  scenario_id: string
  summary: {
    run_step_count: number
    ready_step_count: number
    review_required_step_count: number
    prerequisite_prompt_step_count: number
    not_available_step_count: number
    closed_source_count: number
    source_count: number
    missing_prerequisite_fact_count: number
    content_consistency_issue_count: number
    deterministic_rule_verdict_count: number
  }
  asset_urls: {
    demo_report_markdown?: string | null
    demo_script_markdown?: string | null
  }
  run_steps: DemoRunStep[]
  evidence_boundary: string
}

export interface WorkbenchPayload {
  demo_sample?: DemoSampleMetadata | null
  pdf_document: {
    file_id: string
    filename: string
    file_url: string
    pages: PdfPageMeta[]
    bounding_boxes: BoundingBox[]
    image_blocks?: PdfImageBlock[]
    algorithm_blocks?: PdfAlgorithmBlock[]
    equation_blocks?: PdfEquationBlock[]
    table_asts?: PdfTableAst[]
    figures?: PdfFigureRef[]
    toc_blocks?: PdfTocBlock[]
    toc_sequences?: PdfTocSequence[]
  } | null
  markdown: string
  full_markdown_download_url?: string
  structure_audit_download_url?: string
  structure_audit_markdown_download_url?: string
  regulatory_readiness?: RegulatoryReadinessProjection
  dossier_checklist?: DossierChecklistProjection
  demo_summary?: DemoSummaryProjection
  demo_scenario?: DemoScenarioProjection
  demo_flow?: DemoFlowProjection
  demo_run?: DemoRunProjection
  demo_report_markdown?: string
  demo_report_markdown_download_url?: string
  demo_script_markdown?: string
  demo_script_markdown_download_url?: string
  content_consistency?: ContentConsistencyProjection
  rule_checks: {
    enabled: boolean
    items: RuleCheckItem[]
    group_summaries?: RuleCheckGroupSummary[]
    navigation_audit_records?: RuleNavigationAuditRecord[]
    structure_audit_records?: RuleStructureAuditRecord[]
    summary?: RuleChecksSummary
    risk_count?: number
    message: string
  }
}

export interface ConsistencyRow {
  fact: string
  module_values: Array<{ module: string; value: string; document_id?: string; filename?: string }>
  is_consistent: boolean
}

export interface ConsistencyResponse {
  rows: ConsistencyRow[]
}
