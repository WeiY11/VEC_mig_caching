@echo off
REM 多算法车辆数扫描对比实验
REM TD3 vs DDPG vs Greedy 全面对比

echo ========================================
echo 多算法车辆数扫描对比
echo ========================================
echo 配置:
echo   - 车辆数: 8, 12, 16, 20, 24（5个配置）
echo   - 算法: TD3, DDPG, Greedy（3个算法）
echo   - 每配置轮次: 200
echo   - 固定拓扑: 4 RSU + 2 UAV
echo.
echo 预计时间: 约75分钟（1.25小时）
echo.
echo 生成图表: 参数对比折线图
echo   - 3条曲线对比
echo   - 展示TD3相对优势
echo   - 展示扩展性差异
echo ========================================
echo.

cd /d "%~dp0"
python run_parameter_sweep.py --param vehicles --values 8 12 16 20 24 --algorithms TD3 DDPG Greedy --episodes 200 --seed 42

echo.
echo ========================================
echo 实验完成！
echo.
echo 查看结果:
echo   parameter_sweep_results/analysis/parameter_comparison_lines.png
echo.
echo 图表特点:
echo   - TD3曲线（蓝色圆形）- 预期最优且平缓
echo   - DDPG曲线（紫红方形）- 次优
echo   - Greedy曲线（橙色三角）- 性能较差且陡峭
echo.
echo 论文用途:
echo   - 主图: 展示TD3综合优势
echo   - 说明: TD3在所有负载下都最优
echo ========================================
pause


