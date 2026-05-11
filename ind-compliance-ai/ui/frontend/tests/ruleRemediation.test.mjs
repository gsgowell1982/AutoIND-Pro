import test from 'node:test'
import assert from 'node:assert/strict'

import { getRuleRemediationGuidanceItems } from '../src/ruleRemediation.js'

test('filters empty remediation items and preserves actionable guidance content', () => {
  const items = getRuleRemediationGuidanceItems([
    {
      guidance_code: 'empty-item',
      guidance_title: '   ',
      guidance_summary: '',
      guidance_priority: 'info',
      guidance_steps: [],
      guidance_targets: [],
      guidance_target_details: [],
    },
    {
      guidance_code: 'fix_32r_extension_titles',
      guidance_title: '将 3.2.R 扩展节点标题改为表 4 允许值',
      guidance_summary: '当前扩展节点标题不合法。',
      guidance_priority: 'priority',
      guidance_steps: ['改为合法标题'],
      guidance_targets: ['3.2.R.7原辅料说明'],
      guidance_target_details: [
        {
          target_type: 'node_extension_title',
          label: '3.2.R.7原辅料说明',
          description: '当前标题不在允许集合内。',
        },
      ],
    },
  ])

  assert.equal(items.length, 1)
  assert.equal(items[0].guidance_code, 'fix_32r_extension_titles')
  assert.deepEqual(items[0].guidance_steps, ['改为合法标题'])
})

test('sorts higher-priority remediation items ahead of lower-priority items', () => {
  const items = getRuleRemediationGuidanceItems([
    {
      guidance_code: 'info-item',
      guidance_title: '低优先级提示',
      guidance_summary: '',
      guidance_priority: 'info',
      guidance_steps: ['稍后处理'],
      guidance_targets: [],
      guidance_target_details: [],
    },
    {
      guidance_code: 'priority-item',
      guidance_title: '高优先级整改',
      guidance_summary: '',
      guidance_priority: 'priority',
      guidance_steps: ['立即处理'],
      guidance_targets: [],
      guidance_target_details: [],
    },
  ])

  assert.deepEqual(
    items.map((item) => item.guidance_code),
    ['priority-item', 'info-item'],
  )
})
