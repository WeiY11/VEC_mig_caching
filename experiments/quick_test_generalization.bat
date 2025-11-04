@echo off
chcp 65001 >nul
REM ================================================================
REM 模型泛化性快速测试脚本（Windows）
REM 
REM 功能：快速验证模型泛化性能（约20-30分钟）
REM 包含：5个维度的泛化性测试
REM ================================================================

echo.
echo ================================================================
echo 🧪 模型泛化性快速测试
echo ================================================================
echo.
echo 📋 测试配置：
echo    - 算法: TD3
echo    - 模式: quick (30轮)
echo    - 预计时间: 20-30分钟
echo    - 输出目录: results/generalization_test/
echo.
echo ================================================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 找不到Python
    echo 请确保Python已安装并添加到PATH
    pause
    exit /b 1
)

echo ▶️  开始测试...
echo.

REM 运行泛化性测试
python experiments/test_generalization.py --algorithm TD3 --mode quick

if errorlevel 1 (
    echo.
    echo ❌ 测试失败！
    echo 请检查错误信息
    pause
    exit /b 1
)

echo.
echo ================================================================
echo ✅ 测试完成！
echo ================================================================
echo.
echo 📁 查看结果：
echo    - 报告: results\generalization_test\generalization_report_*.md
echo    - 图表: results\generalization_test\generalization_visualization_*.png
echo    - 数据: results\generalization_test\generalization_results_*.json
echo.
echo ================================================================

pause

