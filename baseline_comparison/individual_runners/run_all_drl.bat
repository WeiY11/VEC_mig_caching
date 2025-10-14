@echo off
REM 批量运行所有DRL算法
REM 使用方法: run_all_drl.bat [episodes] [seed]

setlocal

set EPISODES=%1
set SEED=%2

if "%EPISODES%"=="" set EPISODES=200
if "%SEED%"=="" set SEED=42

echo ================================================================================
echo 批量运行所有DRL算法
echo ================================================================================
echo 训练轮次: %EPISODES%
echo 随机种子: %SEED%
echo ================================================================================
echo.

echo [1/5] 运行TD3...
python baseline_comparison/individual_runners/drl/run_td3_xuance.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo TD3运行失败！
    goto :error
)
echo.

echo [2/5] 运行DDPG...
python baseline_comparison/individual_runners/drl/run_ddpg_xuance.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo DDPG运行失败！
    goto :error
)
echo.

echo [3/5] 运行SAC...
python baseline_comparison/individual_runners/drl/run_sac_xuance.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo SAC运行失败！
    goto :error
)
echo.

echo [4/5] 运行PPO...
python baseline_comparison/individual_runners/drl/run_ppo_xuance.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo PPO运行失败！
    goto :error
)
echo.

echo [5/5] 运行DQN...
python baseline_comparison/individual_runners/drl/run_dqn_xuance.py --episodes %EPISODES% --seed %SEED%
if errorlevel 1 (
    echo DQN运行失败！
    goto :error
)
echo.

echo ================================================================================
echo ✓ 所有DRL算法运行完成！
echo ================================================================================
echo 结果保存在: baseline_comparison/results/{td3,ddpg,sac,ppo,dqn}/
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








