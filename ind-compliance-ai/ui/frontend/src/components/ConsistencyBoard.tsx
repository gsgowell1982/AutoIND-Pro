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
      subTitle="展示跨模块数据比对（例如 M3 批号与 M5 临床样本批号）"
      bordered
      headerBordered
    >
      {rows.length === 0 ? (
        <Empty description="当前无可比对原子事实，请上传包含关键字段的文档" />
      ) : (
        <Table<ConsistencyRow>
          rowKey={(row) => row.fact}
          pagination={false}
          size="small"
          dataSource={rows}
          columns={[
            {
              title: '原子事实',
              dataIndex: 'fact',
              key: 'fact',
              width: 180,
              render: (value: string) => <Typography.Text code>{value}</Typography.Text>,
            },
            {
              title: '模块取值',
              dataIndex: 'module_values',
              key: 'module_values',
              render: (moduleValues: Array<{ module: string; value: string }>) => (
                <Space direction="vertical" size={4}>
                  {moduleValues.map((item) => (
                    <Space key={`${item.module}-${item.value}`}>
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
