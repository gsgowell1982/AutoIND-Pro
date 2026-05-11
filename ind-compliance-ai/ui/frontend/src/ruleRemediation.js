const PRIORITY_ORDER = {
  priority: 0,
  attention: 1,
  info: 2,
}

function getPriorityRank(priority) {
  return PRIORITY_ORDER[priority] ?? PRIORITY_ORDER.info
}

function normalizeTargetDetails(targetDetails) {
  const items = Array.isArray(targetDetails) ? targetDetails : []
  return items
    .map((detail) => ({
      target_type: typeof detail?.target_type === 'string' ? detail.target_type.trim() : '',
      label: typeof detail?.label === 'string' ? detail.label.trim() : '',
      description: typeof detail?.description === 'string' ? detail.description.trim() : '',
    }))
    .filter((detail) => detail.target_type || detail.label || detail.description)
}

export function getRuleRemediationGuidanceItems(remediationGuidance) {
  const items = Array.isArray(remediationGuidance) ? remediationGuidance : []

  return items
    .map((guidance, index) => ({
      _index: index,
      guidance_code: typeof guidance?.guidance_code === 'string' ? guidance.guidance_code.trim() : '',
      guidance_title: typeof guidance?.guidance_title === 'string' ? guidance.guidance_title.trim() : '',
      guidance_summary: typeof guidance?.guidance_summary === 'string' ? guidance.guidance_summary.trim() : '',
      guidance_priority:
        typeof guidance?.guidance_priority === 'string' ? guidance.guidance_priority.trim() : 'info',
      guidance_steps: (Array.isArray(guidance?.guidance_steps) ? guidance.guidance_steps : [])
        .map((step) => (typeof step === 'string' ? step.trim() : ''))
        .filter(Boolean),
      guidance_targets: (Array.isArray(guidance?.guidance_targets) ? guidance.guidance_targets : [])
        .map((target) => (typeof target === 'string' ? target.trim() : ''))
        .filter(Boolean),
      guidance_target_details: normalizeTargetDetails(guidance?.guidance_target_details),
    }))
    .filter(
      (guidance) =>
        guidance.guidance_title ||
        guidance.guidance_summary ||
        guidance.guidance_steps.length > 0 ||
        guidance.guidance_targets.length > 0 ||
        guidance.guidance_target_details.length > 0,
    )
    .sort((left, right) => {
      const priorityDiff = getPriorityRank(left.guidance_priority) - getPriorityRank(right.guidance_priority)
      return priorityDiff !== 0 ? priorityDiff : left._index - right._index
    })
    .map(({ _index, ...guidance }) => guidance)
}
