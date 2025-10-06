@echo off
REM 标准消融实验 (200轮)
echo ========================================
echo 运行标准消融实验 (200轮)
echo 预计耗时: 2-3小时
echo ========================================
echo.

python run_ablation_td3.py --episodes 200

pause

