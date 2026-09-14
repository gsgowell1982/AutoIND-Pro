import { FileOutlined, FolderOpenOutlined } from '@ant-design/icons'
import { Space, Tag, Tree, Typography } from 'antd'
import type { DataNode } from 'antd/es/tree'

import type { PackageInventory } from '../types'

interface ProjectPackageTreeProps {
  inventory?: PackageInventory | null
  fallbackPaths?: string[]
  title?: string
  selectedPath?: string | null
  onPathSelect?: (path: string, kind: 'file' | 'directory') => void
  fillHeight?: boolean
}

function buildTree(paths: string[]): DataNode[] {
  const roots: DataNode[] = []
  const nodes = new Map<string, DataNode>()
  const normalizedPaths = [...new Set(paths.map((path) => path.replaceAll('\\', '/').trim()).filter(Boolean))]

  const ensureDirectory = (path: string): DataNode => {
    const existing = nodes.get(path)
    if (existing) return existing
    const cleanPath = isDirectoryPath(path) ? path.slice(0, -1) : path
    const parts = cleanPath.split('/')
    const parentPath = parts.slice(0, -1).join('/')
    const node: DataNode = {
      key: `dir:${path}`,
      title: parts[parts.length - 1],
      icon: <FolderOpenOutlined />,
      children: [],
    }
    nodes.set(cleanPath, node)
    if (parentPath) {
      const parent = ensureDirectory(parentPath)
      ;(parent.children as DataNode[]).push(node)
    } else {
      roots.push(node)
    }
    return node
  }

  normalizedPaths.forEach((path) => {
    const isDirectory = isDirectoryPath(path)
    const cleanPath = isDirectory ? path.slice(0, -1) : path
    const parts = cleanPath.split('/')
    const directoryPath = isDirectory ? cleanPath : parts.slice(0, -1).join('/')
    if (isDirectory) {
      ensureDirectory(cleanPath)
      return
    }
    const fileName = parts[parts.length - 1]
    const fileNode: DataNode = {
      key: `file:${cleanPath}`,
      title: <Typography.Text ellipsis={{ tooltip: path }}>{fileName}</Typography.Text>,
      icon: <FileOutlined />,
      isLeaf: true,
    }
    if (directoryPath) {
      const parent = ensureDirectory(directoryPath)
      ;(parent.children as DataNode[]).push(fileNode)
    } else {
      roots.push(fileNode)
    }
  })

  const sortNodes = (items: DataNode[]): void => {
    items.sort((left, right) => {
      const leftLeaf = left.isLeaf ? 1 : 0
      const rightLeaf = right.isLeaf ? 1 : 0
      return leftLeaf - rightLeaf || String(left.title).localeCompare(String(right.title), 'zh-CN')
    })
    items.forEach((item) => {
      if (item.children) sortNodes(item.children as DataNode[])
    })
  }
  sortNodes(roots)
  return roots
}

function isDirectoryPath(path: string): boolean {
  return path.endsWith('/')
}

export function ProjectPackageTree({
  inventory,
  fallbackPaths = [],
  title = '项目文件树',
  selectedPath = null,
  onPathSelect,
  fillHeight = false,
}: ProjectPackageTreeProps) {
  const fallbackDirectoryPaths = fallbackPaths
    .filter(isDirectoryPath)
    .map((path) => path.slice(0, -1))
  const fallbackFilePaths = fallbackPaths.filter((path) => !isDirectoryPath(path))
  const directoryPaths = inventory?.directory_paths ?? fallbackDirectoryPaths
  const filePaths = inventory?.file_paths ?? fallbackFilePaths
  const treeData = buildTree([...directoryPaths.map((path) => `${path}/`), ...filePaths])
  if (treeData.length === 0) return null

  return (
    <div className={fillHeight ? 'project-package-tree project-package-tree-fill' : 'project-package-tree'}>
      <Space style={{ marginBottom: 8 }} wrap>
        <Typography.Text strong>{title}</Typography.Text>
        <Tag color="blue">目录 {directoryPaths.length}</Tag>
        <Tag color="cyan">文件 {filePaths.length}</Tag>
        {inventory?.source_kind ? <Tag>{inventory.source_kind === 'zip' ? 'ZIP 项目包' : '文件夹项目'}</Tag> : null}
      </Space>
      <div className="project-package-tree-scroll">
        <Tree
          blockNode
          showIcon
          selectable={Boolean(onPathSelect)}
          selectedKeys={selectedPath ? [`file:${selectedPath}`, `dir:${selectedPath}`] : []}
          onSelect={(keys) => {
            if (!onPathSelect) return
            const key = String(keys[0] ?? '')
            if (key.startsWith('file:')) {
              onPathSelect(key.slice('file:'.length), 'file')
            } else if (key.startsWith('dir:')) {
              onPathSelect(key.slice('dir:'.length), 'directory')
            }
          }}
          defaultExpandAll
          height={fillHeight ? undefined : 320}
          virtual={!fillHeight}
          treeData={treeData}
        />
      </div>
    </div>
  )
}
