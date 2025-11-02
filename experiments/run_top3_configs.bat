@echo off
REM Top 3配置快速验证脚本
REM 生成时间: 2025-11-02
REM 每个配置训练 500 轮，预计6-9小时

echo.
echo ============================================================
echo 权重对比实验 - Top 3配置快速验证
echo ============================================================
echo.
echo 将依次运行3个预测最优配置：
echo   1. balanced      - 预计总成本 3.40 ⭐⭐⭐⭐⭐
echo   2. min_cost      - 预计总成本 4.25 ⭐⭐⭐⭐
echo   3. balanced_v2   - 预计总成本 4.51 ⭐⭐⭐⭐
echo.
echo 预计完成时间: 6-9小时
echo.
pause

REM ============================================================
REM 实验 1/3: balanced (energy_target=3500J)
REM ============================================================

echo.
echo ============================================================
echo 实验 1/3: balanced
echo 时延能耗平衡配置（energy_target=3500J）
echo ============================================================
echo.

python experiments/weight_comparison.py --mode full --config balanced --episodes 500

if errorlevel 1 (
    echo 实验 balanced 失败！
    pause
    exit /b 1
)

echo.
echo ✅ 实验 balanced 完成！
echo.

REM ============================================================
REM 实验 2/3: min_cost (energy_target=2500J)
REM ============================================================

echo.
echo ============================================================
echo 实验 2/3: min_cost
echo 最小成本优化配置
echo ============================================================
echo.

python experiments/weight_comparison.py --mode full --config min_cost --episodes 500

if errorlevel 1 (
    echo 实验 min_cost 失败！
    pause
    exit /b 1
)

echo.
echo ✅ 实验 min_cost 完成！
echo.

REM ============================================================
REM 实验 3/3: balanced_v2 (energy_target=2000J)
REM ============================================================

echo.
echo ============================================================
echo 实验 3/3: balanced_v2
echo 平衡配置V2
echo ============================================================
echo.

python experiments/weight_comparison.py --mode full --config balanced_v2 --episodes 500

if errorlevel 1 (
    echo 实验 balanced_v2 失败！
    pause
    exit /b 1
)

echo.
echo ✅ 实验 balanced_v2 完成！
echo.

REM ============================================================
REM 所有实验完成，生成对比图表
REM ============================================================

echo.
echo ============================================================
echo 所有实验完成！开始生成对比图表...
echo ============================================================
echo.

python experiments/visualize_weight_comparison.py

if errorlevel 1 (
    echo 图表生成失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 🎉 Top 3配置验证完成！
echo ============================================================
echo.
echo 查看结果：
echo   📊 图表: results\weight_comparison\comparison_*\
echo   📝 数据: results\weight_comparison\*\training_results_*.json
echo.
echo 下一步：
echo   python experiments/weight_comparison.py --mode analyze
echo.
pause

