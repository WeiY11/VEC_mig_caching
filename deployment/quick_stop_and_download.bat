@echo off
chcp 65001 >nul
echo ==========================================
echo 停止训练并下载结果
echo ==========================================
echo.

set SERVER=root@connect.westc.gpuhub.com
set PORT=21960
set PASSWORD=B9iXNm5Ee0l4

echo 步骤1: 停止训练...
echo ----------------------------------------
echo 请在SSH窗口中输入密码: %PASSWORD%
echo.
pause

ssh -p %PORT% %SERVER% "pkill -f run_batch_experiments.py; shutdown -c; echo '✅ 训练已停止'; ps aux | grep python | grep -v grep"

echo.
echo ==========================================
echo 步骤2: 下载结果到本地
echo ==========================================
echo.

cd /d D:\VEC_mig_caching

if not exist "results_from_server" mkdir results_from_server

echo 正在下载实验结果...
echo 密码: %PASSWORD%
echo.

scp -P %PORT% -r %SERVER%:/root/VEC_mig_caching/results/camtd3_strategy_suite ./results_from_server/

echo.
echo 正在下载训练日志...
scp -P %PORT% -r %SERVER%:/root/VEC_mig_caching/logs ./results_from_server/logs

echo.
echo ==========================================
echo ✅ 下载完成！
echo ==========================================
echo.
echo 结果保存在: D:\VEC_mig_caching\results_from_server\
echo.

explorer results_from_server

echo.
echo ==========================================
echo 是否关闭服务器？（节省费用）
echo 关闭请按 Y，保留请按 N
echo ==========================================
set /p SHUTDOWN="您的选择 (Y/N): "

if /i "%SHUTDOWN%"=="Y" (
    echo.
    echo 正在关闭服务器...
    ssh -p %PORT% %SERVER% "shutdown -h now"
    echo ✅ 服务器已关机
) else (
    echo.
    echo ℹ️ 服务器保持运行
)

echo.
pause















