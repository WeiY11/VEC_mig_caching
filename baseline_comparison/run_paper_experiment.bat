@echo off
REM 论文级Baseline对比实验（200轮）
REM 生成高质量数据和图表

echo ========================================
echo 论文级Baseline对比实验
echo ========================================
echo 模式: 标准对比
echo 轮次: 200
echo 拓扑: 12车 + 4RSU + 2UAV (固定)
echo 预计时间: 约40-50分钟
echo ========================================
echo.
echo 将运行:
echo   - 5个DRL算法 (TD3, DDPG, SAC, PPO, DQN)
echo   - 5个启发式算法 (Random, Greedy, RoundRobin, LocalFirst, NearestNode)
echo.
echo 生成内容:
echo   - 性能对比图 (3指标)
echo   - 目标函数对比图 (复合指标)
echo   - 收敛曲线图 (DRL算法)
echo   - 统计显著性分析 (t-test)
echo ========================================
echo.

cd /d "%~dp0"
python run_baseline_comparison.py --episodes 200 --seed 42

echo.
echo ========================================
echo 实验完成！
echo.
echo 查看结果:
echo   - results/ - 所有算法的详细数据
echo   - analysis/performance_comparison.png - 性能对比
echo   - analysis/objective_comparison.png - 目标函数对比
echo   - analysis/convergence_curves.png - 收敛曲线
echo ========================================
pause


