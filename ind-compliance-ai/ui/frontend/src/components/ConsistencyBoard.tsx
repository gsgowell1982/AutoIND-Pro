import { ProCard } from '@ant-design/pro-components'
import { Empty, Space, Table, Tag, Typography } from 'antd'

import type { ConsistencyRow } from '../types'

interface ConsistencyBoardProps {
  rows: ConsistencyRow[]
}

export function ConsistencyBoard({ rows }: ConsistencyBoardProps) {
  return (
    <ProCard
      title="3) 一致性检查看板"
      subTitle="呈现检测项目的内容一致性，包含跨模块、跨资料或跨序列字段比对"
      bordered
      headerBordered
    >
      {rows.length === 0 ? (
        <Empty description="当前无可比对字段；需要多个模块、资料或序列中出现可比检测项目后再展示一致性结果" />
      ) : (
        <Table<ConsistencyRow>
          rowKey={(row) => row.fact}
          pagination={false}
          size="small"
          dataSource={rows}
          columns={[
            {
              title: '检测项目',
              dataIndex: 'fact',
              key: 'fact',
              width: 180,
              render: (value: string) => <Typography.Text code>{value}</Typography.Text>,
            },
            {
              title: '模块取值',
              dataIndex: 'module_values',
              key: 'module_values',
              render: (
                moduleValues: Array<{ module: string; value: string; document_id?: string; filename?: string }>,
              ) => (
                <Space direction="vertical" size={4}>
                  {moduleValues.map((item, index) => (
                    <Space key={`${item.document_id ?? item.module}-${item.value}-${index}`}>
                      <Tag>{item.module}</Tag>
                      <Typography.Text>{item.value || 'N/A'}</Typography.Text>
                    </Space>
                  ))}
                </Space>
              ),
            },
            {
              title: '一致性',
              dataIndex: 'is_consistent',
              key: 'is_consistent',
              width: 130,
              render: (isConsistent: boolean) =>
                isConsistent ? <Tag color="success">一致</Tag> : <Tag color="error">不一致</Tag>,
            },
          ]}
        />
      )}
    </ProCard>
  )
}
