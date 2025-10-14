@echo off
chcp 65001 >nul
REM 快速启动TD3训练和对比实验

echo ========================================
echo 🚀 快速启动TD3对比实验
echo ========================================
echo.
echo 正在运行标准实验...
echo - 训练轮次: 400轮（确保收敛）
echo - 评估轮次: 50轮
echo.

cd /d "%~dp0"
python run_full_comparison_with_td3.py --train-episodes 400 --eval-episodes 50

echo.
echo ========================================
echo ✅ 实验完成！
echo.
echo 查看结果：
echo - 图表: academic_figures\vehicle_comparison\
echo - 数据: results\offloading_comparison\
echo ========================================
pause

