#!/bin/bash
# 训练状态检查脚本
# 快速查看训练进度、GPU使用情况和日志

clear
echo "=========================================="
echo "VEC训练状态监控"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 1. GPU状态
echo "【GPU状态】"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits | \
    awk -F, '{printf "  GPU%s: %s\n  利用率: %s%% | 显存: %s/%s MB | 温度: %s°C\n\n", $1, $2, $3, $4, $5, $6}'

# 2. 训练进程
echo "【训练进程】"
TRAIN_PID=$(ps aux | grep "[r]un_batch_experiments.py" | awk '{print $2}')
if [ -n "$TRAIN_PID" ]; then
    echo "  ✓ 训练运行中 (PID: $TRAIN_PID)"
    ps aux | grep "[r]un_batch_experiments.py" | awk '{printf "  CPU: %s%% | 内存: %s%%\n", $3, $4}'
else
    echo "  ✗ 训练未运行"
fi
echo ""

# 3. Python进程GPU使用
echo "【Python GPU使用】"
if command -v gpustat &> /dev/null; then
    gpustat --no-header | grep python
else
    nvidia-smi | grep python | head -3
fi
echo ""

# 4. 最新日志
echo "【最新训练日志】"
LATEST_LOG=$(ls -t /root/VEC_mig_caching/logs/training_*.log 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    echo "  日志文件: $(basename $LATEST_LOG)"
    echo "  最后3行:"
    tail -3 "$LATEST_LOG" | sed 's/^/    /'
else
    echo "  暂无日志"
fi
echo ""

# 5. 结果文件
echo "【实验进度】"
RESULT_DIR="/root/VEC_mig_caching/results"
if [ -d "$RESULT_DIR" ]; then
    RESULT_COUNT=$(find $RESULT_DIR -name "*.json" -o -name "*.png" 2>/dev/null | wc -l)
    echo "  已生成结果文件: $RESULT_COUNT 个"
    
    # 显示最新的5个结果
    echo "  最新结果:"
    find $RESULT_DIR -type f \( -name "*.json" -o -name "*.png" \) -printf "%T@ %p\n" 2>/dev/null | \
        sort -rn | head -5 | awk '{print "    " $2}' | sed 's|/root/VEC_mig_caching/results/||'
fi
echo ""

# 6. 磁盘空间
echo "【磁盘空间】"
df -h /root | tail -1 | awk '{printf "  使用: %s / %s (剩余: %s)\n", $3, $2, $4}'
echo ""

echo "=========================================="
echo "💡 提示:"
echo "  - GPU利用率 4-15% 对强化学习是正常的"
echo "  - 重新进入训练: tmux attach -t vec_training"
echo "  - 取消自动关机: shutdown -c"
echo "=========================================="

