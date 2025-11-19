@echo off
chcp 65001 >nul
echo ==========================================
echo   AutoDL VEC 实验一键部署
echo ==========================================
echo.
echo 服务器: region-41.seetacloud.com:38597
echo 实验: RSU计算资源对比 (1200轮)
echo.
echo 请确认：
echo   1. 已安装Git Bash（推荐）或SSH客户端
echo   2. AutoDL实例已启动（需40+小时运行时长）
echo   3. 本地网络正常
echo.
pause

echo.
echo [检测] 检查部署工具...

REM 检查是否有Git Bash
where bash >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 找到Git Bash
    echo.
    echo 请选择部署方式：
    echo   1. 自动部署（需要手动输入密码5次）
    echo   2. 手动部署（查看详细命令，逐步执行）
    echo.
    set /p deploy_choice=请输入选择 (1/2): 
    
    if "!deploy_choice!"=="1" (
        echo.
        echo 开始自动部署...
        bash autodl_deploy_simple.sh
    ) else (
        echo.
        call AutoDL快速部署助手.bat
    )
    goto :end
)

REM 检查SSH
where ssh >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 找到SSH客户端
    echo.
    echo 将使用交互式部署助手
    echo.
    call AutoDL快速部署助手.bat
    goto :end
)

echo ❌ 未找到Git Bash或SSH客户端！
echo.
echo 请选择以下方案之一：
echo   1. 安装Git for Windows（推荐）: https://git-scm.com/download/win
echo   2. 使用Windows内置SSH（Win10+）
echo   3. 查看手动部署步骤
echo.
set /p install_choice=请输入选择 (1/2/3): 

if "%install_choice%"=="3" (
    call AutoDL快速部署助手.bat
) else (
    echo.
    echo 请安装后重新运行此脚本
)

:end
echo.
echo ==========================================
echo 部署流程完成！
echo ==========================================
pause
