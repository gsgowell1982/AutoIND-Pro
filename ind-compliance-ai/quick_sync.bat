@echo off
chcp 65001 >nul
echo ============================================
echo    AutoIND-Pro 快速同步工具
echo    确保两个目录与 GitHub 保持一致
echo ============================================
echo.

set WORK_DIR=D:\AutoIND-Pro\ind-compliance-ai
set MIRROR_DIR=D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai

echo 📥 正在从 GitHub 同步最新代码到两个目录...
echo.

echo [1/2] 同步工作目录...
cd /d "%WORK_DIR%"
git fetch origin
git reset --hard origin/main
if errorlevel 1 (
    echo ❌ 工作目录同步失败
    pause
    exit /b 1
)
echo ✅ 工作目录已更新

echo.
echo [2/2] 同步镜像目录...
cd /d "%MIRROR_DIR%"
git fetch origin
git reset --hard origin/main
if errorlevel 1 (
    echo ❌ 镜像目录同步失败
    pause
    exit /b 1
)
echo ✅ 镜像目录已更新

echo.
echo ============================================
echo ✅ 同步完成！
echo ============================================
echo.
echo 两个目录现在都是最新版本
echo 当前提交:
git log -1 --oneline
echo.
pause
