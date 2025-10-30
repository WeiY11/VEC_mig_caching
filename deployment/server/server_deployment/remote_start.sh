#!/bin/bash
# 在服务器上启动批量实验

cd /root/VEC_mig_caching

echo "=========================================="
echo "启动VEC批量参数敏感性实验"
echo "时间: $(date)"
echo "=========================================="
echo ""

# 启动后台实验
nohup python experiments/camtd3_strategy_suite/run_batch_experiments.py \
    --mode full \
    --all \
    --non-interactive \
    > batch_experiments.log 2>&1 &

PID=$!
echo $PID > batch_experiments.pid

echo "✅ 实验已启动！"
echo "   进程ID: $PID"
echo "   日志文件: batch_experiments.log"
echo ""
echo "监控命令:"
echo "  tail -f batch_experiments.log"
echo "  nvidia-smi"
echo ""
echo "检查进程:"
echo "  ps -p $PID"
echo ""

# 等待几秒确认启动
sleep 3

# 检查进程是否还在运行
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ 确认：实验正在运行"
else
    echo "❌ 警告：进程可能已终止，请检查日志"
    tail -20 batch_experiments.log
fi

