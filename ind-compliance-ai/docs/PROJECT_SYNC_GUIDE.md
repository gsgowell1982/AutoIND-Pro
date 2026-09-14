# AutoIND-Pro 双项目同步指南

## 📍 项目位置

### 工作目录
```
D:\AutoIND-Pro\ind-compliance-ai
```
- 主要开发目录
- 推荐在此进行日常开发

### 镜像目录
```
D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai
```
- 备份/镜像目录
- 与工作目录保持同步

## 🔄 同步策略

### 方案一：使用同步脚本（推荐）

**位置：** `sync_projects.bat`

**使用方法：**
1. 双击运行 `sync_projects.bat`
2. 根据提示选择同步方向
3. 确认操作

**功能选项：**
- **选项1**：工作目录 → 镜像目录
- **选项2**：镜像目录 → 工作目录
- **选项3**：推送两个目录到 GitHub
- **选项4**：从 GitHub 拉取到两个目录（推荐定期使用）
- **选项5**：检查两个目录差异
- **选项6**：退出

### 方案二：手动Git命令

#### 从工作目录同步到镜像

```bash
# 1. 在工作目录提交更改
cd "D:\AutoIND-Pro\ind-compliance-ai"
git add -A
git commit -m "Your commit message"
git push origin main

# 2. 在镜像目录拉取更新
cd "D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai"
git pull origin main
```

#### 从镜像目录同步到工作

```bash
# 1. 在镜像目录提交更改
cd "D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai"
git add -A
git commit -m "Your commit message"
git push origin main

# 2. 在工作目录拉取更新
cd "D:\AutoIND-Pro\ind-compliance-ai"
git pull origin main
```

## 📋 日常工作流程

### 推荐工作流

**每天开始工作前：**
```bash
# 运行同步脚本，选择选项 4
# 或手动执行：
cd "D:\AutoIND-Pro\ind-compliance-ai"
git pull origin main

cd "D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai"
git pull origin main
```

**开发过程中（在工作目录）：**
```bash
cd "D:\AutoIND-Pro\ind-compliance-ai"

# 定期提交
git add -A
git commit -m "描述你的更改"

# 推送到远程
git push origin main
```

**每天结束工作后：**
```bash
# 运行同步脚本，选择选项 4
# 确保两个目录都是最新版本
```

## ⚠️ 注意事项

### 避免冲突

1. **单一开发位置**
   - 推荐只在工作目录开发
   - 镜像目录作为只读备份

2. **定期同步**
   - 每天开始和结束工作时同步
   - 重要更改后立即同步

3. **遇到冲突时**
   - 使用同步脚本选项 4（强制从远程拉取）
   - 或手动使用 `git reset --hard origin/main`

### 保存本地更改

如果某个目录有未提交的更改需要保留：

```bash
# 暂存更改
git stash save "描述你的更改"

# 同步后恢复
git stash pop
```

## 🛠️ 故障排除

### 问题1：两个目录不同步

**解决方法：**
```bash
# 使用同步脚本选项 5 检查差异
# 然后使用选项 4 强制同步
```

### 问题2：推送被拒绝

**原因：** 远程有新提交
**解决方法：**
```bash
cd "你的项目目录"
git fetch origin
git reset --hard origin/main
```

### 问题3：合并冲突

**解决方法：**
```bash
# 中止合并
git merge --abort

# 强制同步到远程版本
git fetch origin
git reset --hard origin/main
```

### 问题4：忘记在哪个目录工作

**检查方法：**
```bash
# 在两个目录分别执行
git status
git log -1

# 或使用同步脚本选项 5
```

## 📊 同步状态检查

### 快速检查命令

```bash
# 检查工作目录
cd "D:\AutoIND-Pro\ind-compliance-ai"
git status
git log -1 --oneline

# 检查镜像目录
cd "D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai"
git status
git log -1 --oneline
```

### 验证两个目录一致

```bash
# 使用同步脚本选项 5
# 或手动比较：
cd "D:\AutoIND-Pro\ind-compliance-ai"
git rev-parse HEAD

cd "D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai"
git rev-parse HEAD

# 两个哈希值应该相同
```

## 🎯 最佳实践

1. **每天早上**：运行同步脚本选项 4
2. **开发时**：只在工作目录进行
3. **提交时**：使用清晰的提交信息
4. **每天晚上**：推送更改并运行同步脚本选项 4
5. **每周**：检查两个目录是否同步（选项 5）

## 📝 提交信息规范

建议使用以下格式：

```
类型: 简短描述

详细说明（可选）

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

**类型示例：**
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `test`: 测试相关
- `refactor`: 重构
- `chore`: 构建/配置更改

## 🔐 安全提示

1. **不要提交敏感信息**
   - API密钥
   - 密码
   - 个人数据

2. **定期备份**
   - GitHub 是主要备份
   - 两个本地目录是双重保障

3. **检查 .gitignore**
   - 确保临时文件不被提交
   - 检查大文件是否被忽略

## 📞 需要帮助？

遇到问题时：
1. 首先尝试使用同步脚本的选项 5 检查差异
2. 使用选项 4 强制从远程同步
3. 查看本文档的故障排除部分
4. 保存好未提交的更改（使用 git stash）

## ✅ 当前同步状态

截至本文档创建时：
- 两个目录都已同步到提交：`879edfe`
- 提交信息：`Complete Phase 2.12 and fix test suite`
- 测试状态：577 passed (100%)
- 远程分支：origin/main
