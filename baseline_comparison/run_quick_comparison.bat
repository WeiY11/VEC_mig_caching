@echo off
REM 快速Baseline对比（50轮，约10分钟）
REM 用于验证系统功能

echo ========================================
echo 快速Baseline对比实验
echo ========================================
echo 模式: 快速测试
echo 轮次: 50
echo 拓扑: 12车 + 4RSU + 2UAV
echo 预计时间: 约10分钟
echo ========================================
echo.

cd /d "%~dp0"
python run_baseline_comparison.py --quick

echo.
echo ========================================
echo 实验完成！
echo 结果目录: results/
echo 分析目录: analysis/
echo ========================================
pause


