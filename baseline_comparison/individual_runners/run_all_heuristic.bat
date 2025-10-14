@echo off
REM 批量运行所有启发式算法
REM 使用方法: run_all_heuristic.bat [episodes] [seed]

setlocal

set EPISODES=%1
set SEED=%2

if "%EPISODES%"=="" set EPISODES=200
if "%SEED%"=="" set SEED=42

echo ================================================================================
echo 批量运行所有启发式算法
echo ================================================================================
echo 运行轮次: %EPISODES%
echo 随机种子: %SEED%
echo ================================================================================
echo.

echo [1/9] 运行Random...
python baseline_comparison/individual_runners/heuristic/run_random.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo Random运行失败！
    goto :error
)
echo.

echo [2/9] 运行Greedy...
python baseline_comparison/individual_runners/heuristic/run_greedy.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo Greedy运行失败！
    goto :error
)
echo.

echo [3/9] 运行RoundRobin...
python baseline_comparison/individual_runners/heuristic/run_roundrobin.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo RoundRobin运行失败！
    goto :error
)
echo.

echo [4/9] 运行LocalFirst...
python baseline_comparison/individual_runners/heuristic/run_localfirst.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo LocalFirst运行失败！
    goto :error
)
echo.

echo [5/9] 运行NearestNode...
python baseline_comparison/individual_runners/heuristic/run_nearestnode.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo NearestNode运行失败！
    goto :error
)
echo.

echo [6/9] 运行MinDelay（新增）...
python baseline_comparison/individual_runners/heuristic/run_mindelay.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo MinDelay运行失败！
    goto :error
)
echo.

echo [7/9] 运行MinEnergy（新增）...
python baseline_comparison/individual_runners/heuristic/run_minenergy.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo MinEnergy运行失败！
    goto :error
)
echo.

echo [8/9] 运行LoadBalance（新增）...
python baseline_comparison/individual_runners/heuristic/run_loadbalance.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo LoadBalance运行失败！
    goto :error
)
echo.

echo [9/9] 运行HybridGreedy（新增）...
python baseline_comparison/individual_runners/heuristic/run_hybridgreedy.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo HybridGreedy运行失败！
    goto :error
)
echo.

echo ================================================================================
echo ✓ 所有启发式算法运行完成！
echo ================================================================================
echo 结果保存在: baseline_comparison/results/目录下
echo 包含算法: random, greedy, roundrobin, localfirst, nearestnode,
echo          mindelay, minenergy, loadbalance, hybridgreedy
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

