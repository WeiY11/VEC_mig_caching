@echo off
REM ================================================================
REM TD3 四个核心参数敏感性实验批量运行脚本
REM ================================================================
REM 实验列表：
REM   1. 带宽成本对比 (10-50MHz)
REM   2. 任务到达率对比 (1.0-2.5 tasks/s)
REM   3. 数据大小对比 (100-600KB)
REM   4. 本地计算资源对比 (1.2-2.8GHz)
REM
REM 参数：
REM   - 每个实验运行 400 episodes
REM   - 静默模式（不显示详细训练日志）
REM   - 自动生成时间戳标识
REM ================================================================

echo ================================================================
echo TD3 四个核心参数敏感性实验
echo ================================================================
echo.
echo 实验轮数: 400 episodes/配置
echo 运行模式: 静默模式
echo 开始时间: %date% %time%
echo.
echo ================================================================

REM 设置统一的suite-id（使用时间戳）
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo [1/4] 运行带宽成本对比实验...
echo ----------------------------------------------------------------
python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py --bandwidths 10,20,30,40,50 --episodes 400 --silent --suite-id bw_%TIMESTAMP%
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: 带宽成本实验失败！
    pause
    exit /b 1
)
echo [1/4] 完成！
echo.

echo [2/4] 运行任务到达率对比实验...
echo ----------------------------------------------------------------
python experiments/td3_strategy_suite/run_task_arrival_comparison.py --episodes 400 --silent --suite-id arrival_%TIMESTAMP%
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: 任务到达率实验失败！
    pause
    exit /b 1
)
echo [2/4] 完成！
echo.

echo [3/4] 运行数据大小对比实验...
echo ----------------------------------------------------------------
python experiments/td3_strategy_suite/run_data_size_comparison.py --episodes 400 --silent --suite-id datasize_%TIMESTAMP%
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: 数据大小实验失败！
    pause
    exit /b 1
)
echo [3/4] 完成！
echo.

echo [4/4] 运行本地计算资源对比实验...
echo ----------------------------------------------------------------
python experiments/td3_strategy_suite/run_local_compute_resource_comparison.py --episodes 400 --silent --suite-id local_%TIMESTAMP%
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: 本地计算资源实验失败！
    pause
    exit /b 1
)
echo [4/4] 完成！
echo.

echo ================================================================
echo 所有实验完成！
echo ================================================================
echo 结束时间: %date% %time%
echo.
echo 结果保存在: results/parameter_sensitivity/
echo   - bw_%TIMESTAMP%/
echo   - arrival_%TIMESTAMP%/
echo   - datasize_%TIMESTAMP%/
echo   - local_%TIMESTAMP%/
echo.
echo ================================================================

pause

