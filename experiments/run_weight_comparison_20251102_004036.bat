@echo off
REM 权重对比实验批处理脚本
REM 生成时间: 2025-11-02 00:40:36
REM 每个配置训练 200 轮

echo.
echo ============================================================
echo 实验 1/8: current
echo 当前配置 - 能耗主导（能耗归一化值大）
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=2.0
set WEIGHT_REWARD_WEIGHT_ENERGY=1.2
set WEIGHT_REWARD_WEIGHT_CACHE=0.15
set WEIGHT_REWARD_PENALTY_DROPPED=0.05
set WEIGHT_ENERGY_TARGET=1200.0
set WEIGHT_LATENCY_TARGET=0.4
set EXPERIMENT_NAME=current

python train_single_agent.py --algorithm TD3 --episodes 200 --num-vehicles 12

if errorlevel 1 (
    echo 实验 current 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 2/8: delay_priority
echo 时延优先 - 时延权重加倍
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=3.0
set WEIGHT_REWARD_WEIGHT_ENERGY=1.0
set WEIGHT_REWARD_WEIGHT_CACHE=0.15
set WEIGHT_REWARD_PENALTY_DROPPED=0.05
set WEIGHT_ENERGY_TARGET=1200.0
set WEIGHT_LATENCY_TARGET=0.4
set EXPERIMENT_NAME=delay_priority

python train_single_agent.py --algorithm TD3 --episodes 200 --num-vehicles 12

if errorlevel 1 (
    echo 实验 delay_priority 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 3/8: energy_priority
echo 能耗优先 - 能耗权重加倍
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=1.5
set WEIGHT_REWARD_WEIGHT_ENERGY=2.0
set WEIGHT_REWARD_WEIGHT_CACHE=0.15
set WEIGHT_REWARD_PENALTY_DROPPED=0.05
set WEIGHT_ENERGY_TARGET=1200.0
set WEIGHT_LATENCY_TARGET=0.4
set EXPERIMENT_NAME=energy_priority

python train_single_agent.py --algorithm TD3 --episodes 200 --num-vehicles 12

if errorlevel 1 (
    echo 实验 energy_priority 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 4/8: balanced
echo 平衡配置 - 时延能耗归一化后等权重
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=2.0
set WEIGHT_REWARD_WEIGHT_ENERGY=1.2
set WEIGHT_REWARD_WEIGHT_CACHE=0.15
set WEIGHT_REWARD_PENALTY_DROPPED=0.05
set WEIGHT_ENERGY_TARGET=3500.0
set WEIGHT_LATENCY_TARGET=0.4
set EXPERIMENT_NAME=balanced

python train_single_agent.py --algorithm TD3 --episodes 200 --num-vehicles 12

if errorlevel 1 (
    echo 实验 balanced 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 5/8: cache_enhanced
echo 缓存增强 - 提高缓存权重
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=2.0
set WEIGHT_REWARD_WEIGHT_ENERGY=1.2
set WEIGHT_REWARD_WEIGHT_CACHE=0.35
set WEIGHT_REWARD_PENALTY_DROPPED=0.05
set WEIGHT_ENERGY_TARGET=1200.0
set WEIGHT_LATENCY_TARGET=0.4
set EXPERIMENT_NAME=cache_enhanced

python train_single_agent.py --algorithm TD3 --episodes 200 --num-vehicles 12

if errorlevel 1 (
    echo 实验 cache_enhanced 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 6/8: high_reliability
echo 高可靠性 - 强调任务完成率
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=2.0
set WEIGHT_REWARD_WEIGHT_ENERGY=1.2
set WEIGHT_REWARD_WEIGHT_CACHE=0.15
set WEIGHT_REWARD_PENALTY_DROPPED=0.1
set WEIGHT_ENERGY_TARGET=1200.0
set WEIGHT_LATENCY_TARGET=0.4
set EXPERIMENT_NAME=high_reliability

python train_single_agent.py --algorithm TD3 --episodes 200 --num-vehicles 12

if errorlevel 1 (
    echo 实验 high_reliability 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 7/8: aggressive
echo 激进配置 - 同时优化所有目标
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=3.0
set WEIGHT_REWARD_WEIGHT_ENERGY=2.0
set WEIGHT_REWARD_WEIGHT_CACHE=0.25
set WEIGHT_REWARD_PENALTY_DROPPED=0.08
set WEIGHT_ENERGY_TARGET=1200.0
set WEIGHT_LATENCY_TARGET=0.35
set EXPERIMENT_NAME=aggressive

python train_single_agent.py --algorithm TD3 --episodes 200 --num-vehicles 12

if errorlevel 1 (
    echo 实验 aggressive 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 8/8: conservative
echo 保守配置 - 平滑权重，易于收敛
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=1.5
set WEIGHT_REWARD_WEIGHT_ENERGY=1.0
set WEIGHT_REWARD_WEIGHT_CACHE=0.1
set WEIGHT_REWARD_PENALTY_DROPPED=0.03
set WEIGHT_ENERGY_TARGET=1200.0
set WEIGHT_LATENCY_TARGET=0.4
set EXPERIMENT_NAME=conservative

python train_single_agent.py --algorithm TD3 --episodes 200 --num-vehicles 12

if errorlevel 1 (
    echo 实验 conservative 失败！
    pause
    exit /b 1
)

echo.
echo 所有实验完成！
echo.
pause
