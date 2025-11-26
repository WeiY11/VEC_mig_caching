@echo off
echo ========================================
echo 开始运行双实验训练
echo ========================================
echo.

echo [1/2] 运行 RSU 计算资源实验...
echo 命令: python run_optimized_td3_parameter_sweep.py --experiments rsu_compute --episodes 800 --seed 42
python run_optimized_td3_parameter_sweep.py --experiments rsu_compute --episodes 800 --seed 42

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] RSU 计算资源实验失败，退出码: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [1/2] RSU 计算资源实验完成！
echo.
echo ========================================
echo [2/2] 运行带宽实验...
echo 命令: python run_optimized_td3_parameter_sweep.py --experiments bandwidth --episodes 800 --seed 42
python run_optimized_td3_parameter_sweep.py --experiments bandwidth --episodes 800 --seed 42

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 带宽实验失败，退出码: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo 双实验训练全部完成！
echo ========================================
pause
