@echo off
REM 快速消融实验 (30轮测试)
echo ========================================
echo 运行快速消融实验 (30轮)
echo ========================================
echo.

python run_ablation_td3.py --episodes 30 --quick

pause

