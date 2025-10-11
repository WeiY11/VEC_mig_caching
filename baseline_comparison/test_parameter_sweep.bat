@echo off
REM 快速测试参数扫描功能
REM 测试3种车辆数 x 2种算法 x 30轮 = 约5分钟

echo ========================================
echo 参数扫描快速测试
echo ========================================
echo 配置: 8/12/16辆车
echo 算法: TD3 + Greedy
echo 轮次: 30
echo 预计时间: 约5分钟
echo ========================================
echo.

cd /d "%~dp0"
python run_parameter_sweep.py --param vehicles --values 8 12 16 --algorithms TD3 Greedy --episodes 30

echo.
echo ========================================
echo 测试完成！
echo 查看图表:
echo   parameter_sweep_results/analysis/parameter_comparison_lines.png
echo ========================================
pause


