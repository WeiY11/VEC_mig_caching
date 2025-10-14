@echo off
chcp 65001 >nul
REM 一键运行TD3对比实验（改进版）

echo ========================================
echo 一键运行TD3对比实验
echo ========================================
echo.
echo 功能特点：
echo 1. 自动检测TD3模型
echo 2. 必要时自动训练
echo 3. 确保模型在所有正确位置
echo 4. 运行完整对比实验
echo 5. 生成学术图表
echo.
echo 策略包含：
echo - LocalOnly, RSUOnly, LoadBalance, Random
echo - TD3（完整版）
echo - TD3-NoMig（无迁移版）
echo.
echo ========================================
echo.

REM 询问用户选择
echo 请选择运行模式：
echo 1. 快速测试（5-10分钟，少量数据）
echo 2. 标准实验（30-60分钟，正常数据）  
echo 3. 完整实验（1-2小时，完整数据）
echo 4. 仅生成图表（使用现有数据）
echo.
set /p mode="请输入选项 (1/2/3/4): "

if "%mode%"=="1" (
    echo.
    echo 运行快速测试...
    python run_full_comparison_with_td3.py --quick --train-episodes 50 --eval-episodes 10
) else if "%mode%"=="2" (
    echo.
    echo 运行标准实验...
    python run_full_comparison_with_td3.py --train-episodes 200 --eval-episodes 50
) else if "%mode%"=="3" (
    echo.
    echo 运行完整实验...
    python run_full_comparison_with_td3.py --train-episodes 500 --eval-episodes 100
) else if "%mode%"=="4" (
    echo.
    echo 仅生成可视化图表...
    python visualize_vehicle_comparison.py
    echo.
    echo 图表生成完成！
) else (
    echo.
    echo 无效选项，使用默认快速测试模式...
    python run_full_comparison_with_td3.py --quick
)

echo.
echo ========================================
echo 实验结束！
echo.
echo 输出文件位置：
echo - 结果数据: results\offloading_comparison\
echo - 图表文件: academic_figures\vehicle_comparison\
echo ========================================
echo.
pause
