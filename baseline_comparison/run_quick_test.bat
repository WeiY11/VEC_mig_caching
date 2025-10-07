@echo off
REM 快速测试（50轮）
echo ========================================
echo Baseline对比 - 快速测试 (50轮)
echo 预计耗时: 约1小时
echo ========================================
echo.

python run_baseline_comparison.py --episodes 50 --quick

pause


