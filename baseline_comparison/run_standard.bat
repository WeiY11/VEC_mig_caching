@echo off
REM 标准对比实验（200轮）
echo ========================================
echo Baseline对比 - 标准实验 (200轮)
echo 预计耗时: 约4-5小时
echo ========================================
echo.
echo 将训练以下算法:
echo   DRL: TD3, DDPG, SAC, PPO, DQN
echo   Baseline: Random, Greedy, RoundRobin, LocalFirst, NearestNode
echo.
echo 按任意键开始...
pause

python run_baseline_comparison.py --episodes 200

pause


