@echo off
REM 只运行DRL算法对比（跳过启发式算法）
REM 节省时间，快速获取DRL对比结果

echo ========================================
echo DRL算法对比实验
echo ========================================
echo 模式: 只运行DRL算法
echo 算法: TD3, DDPG, SAC, PPO, DQN
echo 轮次: 200
echo 拓扑: 12车 + 4RSU + 2UAV
echo 预计时间: 约25-30分钟
echo ========================================
echo.

cd /d "%~dp0"
python run_baseline_comparison.py --episodes 200 --only-drl

echo.
echo ========================================
echo DRL算法对比完成！
echo 查看 analysis/convergence_curves.png
echo ========================================
pause


