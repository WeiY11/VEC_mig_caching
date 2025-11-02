
ssh -p 21960 root@connect.westc.gpuhub.com



cd /root/VEC_mig_caching

cd /root/VEC_mig_caching
# 创建快速进度脚本
cat > deployment/quick_progress.sh << 'EOF'
#!/bin/bash
echo "========== VEC训练进度 =========="
echo "当前: $(ps aux | grep 'experiments/camtd3_strategy_suite' | grep -v grep | grep -v run_batch | grep -oP 'run_\w+\.py' | head -1 || echo '准备中...')"
echo "Episode: $(grep -oP 'Episode \d+/\d+' /root/VEC_mig_caching/logs/training_*.log 2>/dev/null | tail -1 || echo '未知')"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)% | 显存: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) / $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "完成实验: $(find /root/VEC_mig_caching/results/camtd3_strategy_suite -name '*.png' 2>/dev/null | wc -l) 个结果文件"
echo "================================="
EOF
chmod +x deployment/quick_progress.sh
# 测试一下
bash deployment/quick_progress.sh