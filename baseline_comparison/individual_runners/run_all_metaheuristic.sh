#!/bin/bash
# 批量运行所有元启发式算法
# 使用方法: ./run_all_metaheuristic.sh [episodes] [seed]

EPISODES=${1:-100}
SEED=${2:-42}

echo "================================================================================"
echo "批量运行所有元启发式算法"
echo "================================================================================"
echo "运行轮次: $EPISODES"
echo "随机种子: $SEED"
echo "注意: 元启发式算法计算量较大，请耐心等待"
echo "================================================================================"
echo ""

echo "[1/2] 运行GA (遗传算法)..."
echo "开始时间: $(date)"
python baseline_comparison/individual_runners/metaheuristic/run_ga.py --episodes $EPISODES --seed $SEED
if [ $? -ne 0 ]; then
    echo "GA运行失败！"
    exit 1
fi
echo "完成时间: $(date)"
echo ""

echo "[2/2] 运行PSO (粒子群优化)..."
echo "开始时间: $(date)"
python baseline_comparison/individual_runners/metaheuristic/run_pso.py --episodes $EPISODES --seed $SEED
if [ $? -ne 0 ]; then
    echo "PSO运行失败！"
    exit 1
fi
echo "完成时间: $(date)"
echo ""

echo "================================================================================"
echo "✓ 所有元启发式算法运行完成！"
echo "================================================================================"
echo "结果保存在: baseline_comparison/results/目录下"
echo "包含算法: GA (遗传算法), PSO (粒子群优化)"
echo "================================================================================"








