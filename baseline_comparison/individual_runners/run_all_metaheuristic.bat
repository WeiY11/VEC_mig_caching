@echo off
REM 批量运行所有元启发式算法
REM 使用方法: run_all_metaheuristic.bat [episodes] [seed]

setlocal

set EPISODES=%1
set SEED=%2

if "%EPISODES%"=="" set EPISODES=100
if "%SEED%"=="" set SEED=42

echo ================================================================================
echo 批量运行所有元启发式算法
echo ================================================================================
echo 运行轮次: %EPISODES%
echo 随机种子: %SEED%
echo 注意: 元启发式算法计算量较大，请耐心等待
echo ================================================================================
echo.

echo [1/2] 运行GA (遗传算法)...
echo 开始时间: %date% %time%
python baseline_comparison/individual_runners/metaheuristic/run_ga.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo GA运行失败！
    goto :error
)
echo 完成时间: %date% %time%
echo.

echo [2/2] 运行PSO (粒子群优化)...
echo 开始时间: %date% %time%
python baseline_comparison/individual_runners/metaheuristic/run_pso.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo PSO运行失败！
    goto :error
)
echo 完成时间: %date% %time%
echo.

echo ================================================================================
echo ✓ 所有元启发式算法运行完成！
echo ================================================================================
echo 结果保存在: baseline_comparison/results/目录下
echo 包含算法: GA (遗传算法), PSO (粒子群优化)
echo ================================================================================
pause
exit /b 0

:error
echo.
echo ================================================================================
echo ✗ 运行过程中出现错误！
echo ================================================================================
pause
exit /b 1









