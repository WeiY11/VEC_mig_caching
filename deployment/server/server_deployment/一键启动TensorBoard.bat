@echo off
chcp 65001 > nul
echo ================================================================
echo   一键启动 TensorBoard 监控系统
echo ================================================================
echo.

set SERVER_HOST=region-9.autodl.pro
set SERVER_PORT=47042
set SERVER_USER=root
set REMOTE_DIR=/root/VEC_mig_caching

echo [步骤 1/3] 上传监控脚本到服务器...
scp -P %SERVER_PORT% server_deployment/log_to_tensorboard.py %SERVER_USER%@%SERVER_HOST%:%REMOTE_DIR%/server_deployment/
scp -P %SERVER_PORT% server_deployment/start_monitoring.sh %SERVER_USER%@%SERVER_HOST%:%REMOTE_DIR%/server_deployment/

if errorlevel 1 (
    echo [错误] 文件上传失败！
    echo 请确保：
    echo 1. 已安装 Git for Windows（包含 scp 命令）
    echo 2. 网络连接正常
    pause
    exit /b 1
)

echo [成功] 文件上传完成
echo.

echo [步骤 2/3] 在服务器上启动监控...
ssh -p %SERVER_PORT% %SERVER_USER%@%SERVER_HOST% "cd %REMOTE_DIR% && bash server_deployment/start_monitoring.sh"

if errorlevel 1 (
    echo [错误] 启动监控失败！
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   启动成功！
echo ================================================================
echo.
echo [步骤 3/3] 访问 TensorBoard：
echo.
echo 方式1（推荐）：
echo   1. 打开 https://www.autodl.com/console
echo   2. 找到您的实例
echo   3. 点击"自定义服务"
echo   4. 端口填：6006
echo   5. 点击生成的链接
echo.
echo 方式2（本地隧道）：
echo   运行以下命令建立 SSH 隧道：
echo   ssh -p %SERVER_PORT% -L 6006:localhost:6006 %SERVER_USER%@%SERVER_HOST%
echo   然后访问: http://localhost:6006
echo.
echo 监控指标：
echo   - 平均奖励 (Average Reward)
echo   - 平均时延 (Average Delay)
echo   - 总能耗 (Total Energy)
echo   - 任务完成率 (Completion Rate)
echo.
pause

