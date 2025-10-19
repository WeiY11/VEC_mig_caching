@echo off
REM 运行算法结果可视化
REM 使用方法: run_visualization.bat [save_format]

setlocal

set FORMAT=%1
if "%FORMAT%"=="" set FORMAT=png

echo ================================================================================
echo 算法结果可视化工具
echo ================================================================================
echo 图片格式: %FORMAT%
echo 保存目录: baseline_comparison/individual_runners/figures/
echo ================================================================================
echo.

REM 创建图表保存目录
if not exist figures mkdir figures

echo 开始生成可视化图表...
echo.

python visualize_results.py --save-dir ./figures --format %FORMAT%

if errorlevel 1 (
    echo.
    echo ❌ 可视化过程出现错误！
    echo 请确保已运行算法并生成结果文件。
    goto :error
)

echo.
echo ================================================================================
echo ✅ 可视化完成！
echo ================================================================================
echo 生成的图表：
echo   1. performance_comparison - 性能对比柱状图
echo   2. learning_curves - 学习曲线图
echo   3. type_comparison - 算法类型对比箱线图
echo   4. radar_chart - 多维度性能雷达图
echo   5. results_table.tex - LaTeX表格（论文用）
echo.
echo 图表保存在: baseline_comparison/individual_runners/figures/
echo ================================================================================

REM 打开图表目录
start "" "%cd%\figures"

pause
exit /b 0

:error
pause
exit /b 1










