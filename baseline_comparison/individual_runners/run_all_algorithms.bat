@echo off
REM 批量运行所有算法（启发式 + 元启发式 + DRL）
REM 使用方法: run_all_algorithms.bat [episodes] [seed] [mode]
REM mode: all (默认), heuristic, metaheuristic, drl

setlocal

set EPISODES=%1
set SEED=%2
set MODE=%3

if "%EPISODES%"=="" set EPISODES=100
if "%SEED%"=="" set SEED=42
if "%MODE%"=="" set MODE=all

echo ================================================================================
echo 批量运行算法对比实验
echo ================================================================================
echo 运行轮次: %EPISODES%
echo 随机种子: %SEED%
echo 运行模式: %MODE%
echo ================================================================================
echo.

set TOTAL_TIME_START=%time%

REM 运行启发式算法
if "%MODE%"=="all" goto :run_heuristic
if "%MODE%"=="heuristic" goto :run_heuristic
goto :skip_heuristic

:run_heuristic
echo ===== 第一部分：启发式算法 (9个) =====
echo.
call run_all_heuristic.bat %EPISODES% %SEED%
if errorlevel 1 goto :error
echo.

:skip_heuristic

REM 运行元启发式算法
if "%MODE%"=="all" goto :run_metaheuristic
if "%MODE%"=="metaheuristic" goto :run_metaheuristic
goto :skip_metaheuristic

:run_metaheuristic
echo ===== 第二部分：元启发式算法 (2个) =====
echo.
call run_all_metaheuristic.bat %EPISODES% %SEED%
if errorlevel 1 goto :error
echo.

:skip_metaheuristic

REM 运行DRL算法
if "%MODE%"=="all" goto :run_drl
if "%MODE%"=="drl" goto :run_drl
goto :skip_drl

:run_drl
echo ===== 第三部分：深度强化学习算法 (5个) =====
echo.
call run_all_drl.bat %EPISODES% %SEED%
if errorlevel 1 goto :error
echo.

:skip_drl

echo ================================================================================
echo ✓ 所有算法运行完成！
echo ================================================================================
echo 开始时间: %TOTAL_TIME_START%
echo 结束时间: %time%
echo 结果保存在: baseline_comparison/results/目录下
echo.
echo 算法类别统计:
if "%MODE%"=="all" (
    echo   - 启发式算法 (9个): Random, Greedy, RoundRobin, LocalFirst, NearestNode,
    echo                       MinDelay, MinEnergy, LoadBalance, HybridGreedy
    echo   - 元启发式算法 (2个): GA (遗传算法), PSO (粒子群优化)
    echo   - 深度强化学习 (5个): DQN, DDPG, TD3, SAC, PPO
    echo   - 总计: 16个算法
)
if "%MODE%"=="heuristic" echo   - 启发式算法: 9个
if "%MODE%"=="metaheuristic" echo   - 元启发式算法: 2个
if "%MODE%"=="drl" echo   - 深度强化学习: 5个
echo ================================================================================
echo.
echo 后续步骤:
echo 1. 运行结果分析: python baseline_comparison/individual_runners/analyze_all_results.py
echo 2. 生成对比图表: python baseline_comparison/individual_runners/generate_comparison_charts.py
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











