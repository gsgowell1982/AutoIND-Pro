@echo off
chcp 65001 >nul
echo ============================================
echo    AutoIND-Pro 项目同步工具
echo ============================================
echo.

set WORK_DIR=D:\AutoIND-Pro\ind-compliance-ai
set MIRROR_DIR=D:\d\funding\nation\new code\AutoIND-Pro\ind-compliance-ai

echo 🔍 检查项目状态...
echo.

REM 检查工作目录状态
echo [工作目录] %WORK_DIR%
cd /d "%WORK_DIR%"
git status --short
if errorlevel 1 (
    echo ❌ 错误：无法访问工作目录
    pause
    exit /b 1
)
echo.

REM 检查镜像目录状态
echo [镜像目录] %MIRROR_DIR%
cd /d "%MIRROR_DIR%"
git status --short
if errorlevel 1 (
    echo ❌ 错误：无法访问镜像目录
    pause
    exit /b 1
)
echo.

echo ============================================
echo 请选择同步方向：
echo ============================================
echo 1. 工作目录 → 镜像目录（从工作目录同步到镜像）
echo 2. 镜像目录 → 工作目录（从镜像同步到工作目录）
echo 3. 两个目录 → GitHub（推送到远程）
echo 4. GitHub → 两个目录（从远程拉取到两个目录）
echo 5. 检查两个目录差异
echo 6. 退出
echo ============================================
echo.

set /p choice="请输入选项 (1-6): "

if "%choice%"=="1" goto sync_work_to_mirror
if "%choice%"=="2" goto sync_mirror_to_work
if "%choice%"=="3" goto push_to_remote
if "%choice%"=="4" goto pull_from_remote
if "%choice%"=="5" goto check_diff
if "%choice%"=="6" goto end

echo ❌ 无效选项
pause
exit /b 1

:sync_work_to_mirror
echo.
echo 📤 正在从工作目录同步到镜像目录...
echo.

cd /d "%WORK_DIR%"
echo [工作目录] 检查状态...
git status

echo.
set /p confirm="是否继续同步？(y/n): "
if /i not "%confirm%"=="y" goto end

echo.
echo [工作目录] 提交更改...
git add -A
git commit -m "Sync: Update from work directory"
if errorlevel 1 (
    echo ℹ️  没有新的更改需要提交
)

echo.
echo [工作目录] 推送到远程...
git push origin main
if errorlevel 1 (
    echo ❌ 推送失败
    pause
    exit /b 1
)

echo.
echo [镜像目录] 拉取最新代码...
cd /d "%MIRROR_DIR%"
git fetch origin
git reset --hard origin/main

echo.
echo ✅ 同步完成！工作目录 → 镜像目录
goto end

:sync_mirror_to_work
echo.
echo 📤 正在从镜像目录同步到工作目录...
echo.

cd /d "%MIRROR_DIR%"
echo [镜像目录] 检查状态...
git status

echo.
set /p confirm="是否继续同步？(y/n): "
if /i not "%confirm%"=="y" goto end

echo.
echo [镜像目录] 提交更改...
git add -A
git commit -m "Sync: Update from mirror directory"
if errorlevel 1 (
    echo ℹ️  没有新的更改需要提交
)

echo.
echo [镜像目录] 推送到远程...
git push origin main
if errorlevel 1 (
    echo ❌ 推送失败
    pause
    exit /b 1
)

echo.
echo [工作目录] 拉取最新代码...
cd /d "%WORK_DIR%"
git fetch origin
git reset --hard origin/main

echo.
echo ✅ 同步完成！镜像目录 → 工作目录
goto end

:push_to_remote
echo.
echo 📤 正在推送两个目录到 GitHub...
echo.

echo [工作目录] 提交并推送...
cd /d "%WORK_DIR%"
git add -A
git commit -m "Sync: Update from work directory"
git push origin main

echo.
echo [镜像目录] 提交并推送...
cd /d "%MIRROR_DIR%"
git add -A
git commit -m "Sync: Update from mirror directory"
git push origin main

echo.
echo ✅ 推送完成！
goto end

:pull_from_remote
echo.
echo 📥 正在从 GitHub 拉取到两个目录...
echo.

echo [工作目录] 拉取最新代码...
cd /d "%WORK_DIR%"
git fetch origin
git reset --hard origin/main

echo.
echo [镜像目录] 拉取最新代码...
cd /d "%MIRROR_DIR%"
git fetch origin
git reset --hard origin/main

echo.
echo ✅ 拉取完成！两个目录已同步到最新版本
goto end

:check_diff
echo.
echo 🔍 检查两个目录的差异...
echo.

cd /d "%WORK_DIR%"
echo [工作目录] 当前提交:
git log -1 --oneline
for /f "tokens=*" %%a in ('git rev-parse HEAD') do set WORK_COMMIT=%%a

echo.
cd /d "%MIRROR_DIR%"
echo [镜像目录] 当前提交:
git log -1 --oneline
for /f "tokens=*" %%a in ('git rev-parse HEAD') do set MIRROR_COMMIT=%%a

echo.
if "%WORK_COMMIT%"=="%MIRROR_COMMIT%" (
    echo ✅ 两个目录完全同步！
    echo    提交哈希: %WORK_COMMIT%
) else (
    echo ⚠️  两个目录不同步！
    echo    工作目录: %WORK_COMMIT%
    echo    镜像目录: %MIRROR_COMMIT%
    echo.
    echo    建议执行选项 4 从 GitHub 拉取最新版本
)
goto end

:end
echo.
echo ============================================
pause
