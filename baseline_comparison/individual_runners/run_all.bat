@echo off
REM 批量运行所有算法（DRL + 启发式）
REM 使用方法: run_all.bat [episodes] [seed]

setlocal

set EPISODES=%1
set SEED=%2

if "%EPISODES%"=="" set EPISODES=200
if "%SEED%"=="" set SEED=42

echo ================================================================================
echo 批量运行所有算法（10个）
echo ================================================================================
echo 训练轮次: %EPISODES%
echo 随机种子: %SEED%
echo ================================================================================
echo.

echo 阶段1：运行所有DRL算法（5个）
echo --------------------------------------------------------------------------------
call run_all_drl.bat %EPISODES% %SEED%
if errorlevel 1 (
    echo DRL算法运行失败！
    goto :error
)
echo.

echo 阶段2：运行所有启发式算法（5个）
echo --------------------------------------------------------------------------------
call run_all_heuristic.bat %EPISODES% %SEED%
if errorlevel 1 (
    echo 启发式算法运行失败！
    goto :error
)
echo.

echo ================================================================================
echo ✓ 所有10个算法运行完成！
echo ================================================================================
echo.
echo 结果保存在: baseline_comparison/results/
echo   DRL算法: td3, ddpg, sac, ppo, dqn
echo   启发式: random, greedy, roundrobin, localfirst, nearestnode
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








