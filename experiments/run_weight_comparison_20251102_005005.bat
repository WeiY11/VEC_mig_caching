@echo off
REM 权重对比实验批处理脚本
REM 生成时间: 2025-11-02 00:50:05
REM 每个配置训练 500 轮

echo.
echo ============================================================
echo 实验 1/14: current
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

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 current 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 2/14: delay_priority
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

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 delay_priority 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 3/14: energy_priority
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

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 energy_priority 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 4/14: balanced
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

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 balanced 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 5/14: cache_enhanced
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

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 cache_enhanced 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 6/14: high_reliability
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

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 high_reliability 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 7/14: aggressive
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

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 aggressive 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 8/14: conservative
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

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 conservative 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 9/14: balanced_v2
echo 平衡v2 - 通过目标值平衡时延能耗权重
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=2.0
set WEIGHT_REWARD_WEIGHT_ENERGY=1.2
set WEIGHT_REWARD_WEIGHT_CACHE=0.15
set WEIGHT_REWARD_PENALTY_DROPPED=0.05
set WEIGHT_ENERGY_TARGET=2000.0
set WEIGHT_LATENCY_TARGET=0.4
set EXPERIMENT_NAME=balanced_v2

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 balanced_v2 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 10/14: cache_aggressive
echo 缓存激进 - 大幅提高缓存权重
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=2.0
set WEIGHT_REWARD_WEIGHT_ENERGY=1.2
set WEIGHT_REWARD_WEIGHT_CACHE=0.5
set WEIGHT_REWARD_PENALTY_DROPPED=0.05
set WEIGHT_ENERGY_TARGET=1200.0
set WEIGHT_LATENCY_TARGET=0.4
set EXPERIMENT_NAME=cache_aggressive

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 cache_aggressive 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 11/14: min_cost
echo 最小成本 - 平衡权重+合理目标
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=1.8
set WEIGHT_REWARD_WEIGHT_ENERGY=1.5
set WEIGHT_REWARD_WEIGHT_CACHE=0.12
set WEIGHT_REWARD_PENALTY_DROPPED=0.04
set WEIGHT_ENERGY_TARGET=2500.0
set WEIGHT_LATENCY_TARGET=0.38
set EXPERIMENT_NAME=min_cost

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 min_cost 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 12/14: strict_latency
echo 严格时延 - 更严格的时延目标
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=3.5
set WEIGHT_REWARD_WEIGHT_ENERGY=1.0
set WEIGHT_REWARD_WEIGHT_CACHE=0.15
set WEIGHT_REWARD_PENALTY_DROPPED=0.05
set WEIGHT_ENERGY_TARGET=1200.0
set WEIGHT_LATENCY_TARGET=0.35
set EXPERIMENT_NAME=strict_latency

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 strict_latency 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 13/14: energy_saver
echo 节能优先v2 - 极低能耗目标
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=1.5
set WEIGHT_REWARD_WEIGHT_ENERGY=2.5
set WEIGHT_REWARD_WEIGHT_CACHE=0.15
set WEIGHT_REWARD_PENALTY_DROPPED=0.05
set WEIGHT_ENERGY_TARGET=800.0
set WEIGHT_LATENCY_TARGET=0.4
set EXPERIMENT_NAME=energy_saver

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 energy_saver 失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 实验 14/14: comprehensive
echo 综合最优 - 基于经验的综合配置
echo ============================================================
echo.

set WEIGHT_REWARD_WEIGHT_DELAY=2.2
set WEIGHT_REWARD_WEIGHT_ENERGY=1.5
set WEIGHT_REWARD_WEIGHT_CACHE=0.2
set WEIGHT_REWARD_PENALTY_DROPPED=0.06
set WEIGHT_ENERGY_TARGET=1800.0
set WEIGHT_LATENCY_TARGET=0.38
set EXPERIMENT_NAME=comprehensive

python train_single_agent.py --algorithm TD3 --episodes 500 --num-vehicles 12 --silent-mode

if errorlevel 1 (
    echo 实验 comprehensive 失败！
    pause
    exit /b 1
)

echo.
echo 所有实验完成！
echo.
pause
