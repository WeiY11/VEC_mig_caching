#!/bin/bash
# 实验进度详细检查脚本
# 显示当前正在运行的实验、已完成的实验和剩余实验

clear
echo "=========================================="
echo "VEC 批量实验进度监控"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 定义实验列表
declare -a EXPERIMENTS=(
    "数据大小对比:run_data_size_comparison.py"
    "车辆数量对比:run_vehicle_count_comparison.py"
    "本地资源对卸载影响:run_local_resource_offload_comparison.py"
    "本地资源对成本影响:run_local_resource_cost_comparison.py"
    "带宽对成本影响:run_bandwidth_cost_comparison.py"
    "边缘节点配置对比:run_edge_node_comparison.py"
    "任务到达率影响:run_task_arrival_comparison.py"
    "移动速度影响:run_mobility_speed_comparison.py"
    "策略上下文窗口:run_strategy_context_window.py"
    "完整策略套件:run_full_suite.py"
)

LOG_DIR="/root/VEC_mig_caching/logs"
RESULT_DIR="/root/VEC_mig_caching/results/camtd3_strategy_suite"

# 1. 当前运行的实验
echo "【当前实验】"
CURRENT_EXP=$(ps aux | grep "experiments/camtd3_strategy_suite" | grep -v grep | grep -v "run_batch_experiments" | head -1)
if [ -n "$CURRENT_EXP" ]; then
    SCRIPT_NAME=$(echo "$CURRENT_EXP" | grep -oP 'run_\w+\.py')
    for exp in "${EXPERIMENTS[@]}"; do
        EXP_NAME="${exp%%:*}"
        EXP_SCRIPT="${exp##*:}"
        if [[ "$SCRIPT_NAME" == "$EXP_SCRIPT" ]]; then
            echo "  🔄 正在运行: $EXP_NAME"
            echo "     脚本: $EXP_SCRIPT"
            break
        fi
    done
else
    echo "  ⏸️  暂无实验运行（可能在准备阶段）"
fi
echo ""

# 2. 从日志中提取进度
echo "【训练进度】"
LATEST_LOG=$(ls -t $LOG_DIR/training_*.log 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    echo "  最新日志: $(basename $LATEST_LOG)"
    echo ""
    
    # 提取Episode进度
    EPISODE_INFO=$(grep -oP "Episode \d+/\d+" "$LATEST_LOG" | tail -1)
    if [ -n "$EPISODE_INFO" ]; then
        CURRENT_EP=$(echo "$EPISODE_INFO" | grep -oP "\d+" | head -1)
        TOTAL_EP=$(echo "$EPISODE_INFO" | grep -oP "\d+" | tail -1)
        PROGRESS=$((CURRENT_EP * 100 / TOTAL_EP))
        echo "  📊 Episode进度: $CURRENT_EP/$TOTAL_EP ($PROGRESS%)"
        
        # 绘制进度条
        FILLED=$((PROGRESS / 2))
        BAR=$(printf "█%.0s" $(seq 1 $FILLED))
        EMPTY=$(printf "░%.0s" $(seq 1 $((50 - FILLED))))
        echo "     [$BAR$EMPTY] $PROGRESS%"
    fi
    
    # 提取奖励信息
    REWARD_INFO=$(grep -oP "Reward: [-+]?[0-9]*\.?[0-9]+" "$LATEST_LOG" | tail -1)
    if [ -n "$REWARD_INFO" ]; then
        echo "  🎯 $REWARD_INFO"
    fi
    
    # 提取最新指标
    DELAY_INFO=$(grep -oP "平均时延: [0-9]*\.?[0-9]+" "$LATEST_LOG" | tail -1)
    ENERGY_INFO=$(grep -oP "平均能耗: [0-9]*\.?[0-9]+" "$LATEST_LOG" | tail -1)
    if [ -n "$DELAY_INFO" ] || [ -n "$ENERGY_INFO" ]; then
        echo "  📈 $DELAY_INFO"
        echo "     $ENERGY_INFO"
    fi
else
    echo "  ⚠️  未找到训练日志"
fi
echo ""

# 3. 已完成的实验
echo "【已完成实验】"
COMPLETED_COUNT=0
if [ -d "$RESULT_DIR" ]; then
    for exp in "${EXPERIMENTS[@]}"; do
        EXP_NAME="${exp%%:*}"
        EXP_SCRIPT="${exp##*:}"
        EXP_BASE="${EXP_SCRIPT%.py}"
        
        # 检查是否有结果文件
        RESULT_FILES=$(find "$RESULT_DIR" -name "*${EXP_BASE}*" 2>/dev/null | wc -l)
        if [ $RESULT_FILES -gt 0 ]; then
            echo "  ✅ $EXP_NAME ($RESULT_FILES 个文件)"
            COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        fi
    done
    
    if [ $COMPLETED_COUNT -eq 0 ]; then
        echo "  📝 暂无已完成实验"
    fi
else
    echo "  ⚠️  结果目录不存在"
fi
echo ""

# 4. 总体进度
echo "【总体进度】"
TOTAL_EXP=${#EXPERIMENTS[@]}
OVERALL_PROGRESS=$((COMPLETED_COUNT * 100 / TOTAL_EXP))
echo "  完成: $COMPLETED_COUNT/$TOTAL_EXP 个实验 ($OVERALL_PROGRESS%)"

# 总进度条
FILLED=$((OVERALL_PROGRESS / 2))
BAR=$(printf "█%.0s" $(seq 1 $FILLED))
EMPTY=$(printf "░%.0s" $(seq 1 $((50 - FILLED))))
echo "  [$BAR$EMPTY] $OVERALL_PROGRESS%"
echo ""

# 5. 预估剩余时间
if [ $COMPLETED_COUNT -gt 0 ]; then
    REMAINING=$((TOTAL_EXP - COMPLETED_COUNT))
    
    # 从日志中提取开始时间
    START_TIME=$(head -1 "$LATEST_LOG" 2>/dev/null | grep -oP "\d{8}_\d{6}")
    if [ -n "$START_TIME" ]; then
        START_EPOCH=$(date -d "${START_TIME:0:8} ${START_TIME:9:2}:${START_TIME:11:2}:${START_TIME:13:2}" +%s 2>/dev/null)
        NOW_EPOCH=$(date +%s)
        ELAPSED=$((NOW_EPOCH - START_EPOCH))
        
        if [ $ELAPSED -gt 0 ] && [ $COMPLETED_COUNT -gt 0 ]; then
            AVG_TIME=$((ELAPSED / COMPLETED_COUNT))
            REMAINING_TIME=$((AVG_TIME * REMAINING))
            
            HOURS=$((REMAINING_TIME / 3600))
            MINUTES=$(((REMAINING_TIME % 3600) / 60))
            
            echo "  ⏱️  预计剩余时间: ${HOURS}小时${MINUTES}分钟"
        fi
    fi
fi
echo ""

# 6. GPU状态
echo "【GPU状态】"
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1)
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
echo "  GPU利用率: ${GPU_UTIL}%"
echo "  显存使用: ${GPU_MEM}"
echo "  温度: ${GPU_TEMP}°C"
echo ""

# 7. 最新日志输出
echo "【最新日志】(最后5行)"
if [ -f "$LATEST_LOG" ]; then
    tail -5 "$LATEST_LOG" | sed 's/^/  /'
else
    echo "  暂无日志"
fi
echo ""

echo "=========================================="
echo "💡 快捷命令:"
echo "  查看实时日志: tail -f $LATEST_LOG"
echo "  进入训练会话: tmux attach -t vec_training"
echo "  再次检查: ./deployment/check_experiment_progress.sh"
echo "=========================================="

