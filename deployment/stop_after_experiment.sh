#!/bin/bash
# 在指定实验完成后自动停止训练
# 用法: bash stop_after_experiment.sh <实验序号>

TARGET_EXP=${1:-5}  # 默认第5个实验后停止

echo "=========================================="
echo "自动停止监控脚本"
echo "将在第 $TARGET_EXP 个实验完成后停止训练"
echo "=========================================="
echo ""

# 实验列表
declare -a EXPERIMENTS=(
    "run_data_size_comparison.py"
    "run_vehicle_count_comparison.py"
    "run_local_resource_offload_comparison.py"
    "run_local_resource_cost_comparison.py"
    "run_bandwidth_cost_comparison.py"
    "run_edge_node_comparison.py"
    "run_task_arrival_comparison.py"
    "run_mobility_speed_comparison.py"
    "run_strategy_context_window.py"
    "run_full_suite.py"
)

COMPLETED=0
RESULT_DIR="/root/VEC_mig_caching/results/camtd3_strategy_suite"

while true; do
    # 统计已完成的实验
    COMPLETED=0
    for exp in "${EXPERIMENTS[@]}"; do
        EXP_BASE="${exp%.py}"
        RESULT_COUNT=$(find "$RESULT_DIR" -name "*${EXP_BASE}*" 2>/dev/null | wc -l)
        if [ $RESULT_COUNT -gt 0 ]; then
            COMPLETED=$((COMPLETED + 1))
        fi
    done
    
    echo "[$(date '+%H:%M:%S')] 已完成: $COMPLETED/$TARGET_EXP 个实验"
    
    # 如果达到目标数量，停止训练
    if [ $COMPLETED -ge $TARGET_EXP ]; then
        echo ""
        echo "=========================================="
        echo "✅ 已完成 $COMPLETED 个实验，开始停止训练..."
        echo "=========================================="
        
        # 停止训练进程
        pkill -f "run_batch_experiments.py"
        
        echo "✅ 训练已停止"
        echo "✅ 取消自动关机"
        shutdown -c 2>/dev/null
        
        echo ""
        echo "结果保存在: $RESULT_DIR"
        echo ""
        exit 0
    fi
    
    # 每30秒检查一次
    sleep 30
done

