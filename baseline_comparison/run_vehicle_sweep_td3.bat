@echo off
REM TD3车辆数扫描实验（论文核心）
REM 展示TD3在不同负载下的扩展性

echo ========================================
echo TD3车辆数扫描实验
echo ========================================
echo 配置:
echo   - 车辆数: 8, 12, 16, 20, 24（5个配置）
echo   - 算法: TD3
echo   - 每配置轮次: 200
echo   - 固定拓扑: 4 RSU + 2 UAV
echo.
echo 预计时间: 约25分钟
echo.
echo 生成图表: 参数对比折线图
echo   X轴: 车辆数
echo   Y轴: 时延/能耗/完成率/目标函数
echo ========================================
echo.

cd /d "%~dp0"
python run_parameter_sweep.py --param vehicles --values 8 12 16 20 24 --algorithms TD3 --episodes 200 --seed 42

echo.
echo ========================================
echo 实验完成！
echo.
echo 查看结果:
echo   parameter_sweep_results/analysis/parameter_comparison_lines.png
echo.
echo 论文用途:
echo   - 展示TD3的扩展性
echo   - 证明在固定拓扑下算法性能稳定
echo ========================================
pause


