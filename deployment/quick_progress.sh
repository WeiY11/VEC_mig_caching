#!/bin/bash
# 快速进度查看（一行命令版本）

echo "========== VEC训练进度 =========="
echo "当前: $(ps aux | grep 'experiments/camtd3_strategy_suite' | grep -v grep | grep -v run_batch | grep -oP 'run_\w+\.py' | head -1 || echo '准备中...')"
echo "Episode: $(grep -oP 'Episode \d+/\d+' /root/VEC_mig_caching/logs/training_*.log 2>/dev/null | tail -1 || echo '未知')"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)% | 显存: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) / $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "完成实验: $(find /root/VEC_mig_caching/results/camtd3_strategy_suite -name '*.png' 2>/dev/null | wc -l) 个结果文件"
echo "================================="

