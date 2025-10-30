#!/bin/bash
# 监控服务器上的批量实验

cd /root/VEC_mig_caching

echo "=========================================="
echo "VEC批量实验监控"
echo "当前时间: $(date)"
echo "=========================================="
echo ""

# 检查进程
echo "【进程状态】"
if [ -f batch_experiments.pid ]; then
    PID=$(cat batch_experiments.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 实验正在运行 (PID: $PID)"
        echo ""
        ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd --no-headers
    else
        echo "❌ 实验进程已停止 (PID: $PID)"
    fi
else
    echo "⚠️  未找到PID文件"
fi

echo ""
echo "【GPU使用情况】"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "⚠️ 无法获取GPU信息"

echo ""
echo "【最新日志】(最后50行)"
if [ -f batch_experiments.log ]; then
    echo "日志文件: batch_experiments.log"
    echo "文件大小: $(du -h batch_experiments.log | cut -f1)"
    echo "----------------------------------------"
    tail -50 batch_experiments.log
else
    echo "⚠️ 未找到日志文件"
fi

echo ""
echo "【结果目录】"
if [ -d results/parameter_sensitivity ]; then
    echo "results/parameter_sensitivity/"
    ls -lth results/parameter_sensitivity/ 2>/dev/null | head -10
else
    echo "⚠️ 结果目录尚未创建"
fi

echo ""
echo "=========================================="
echo "刷新: watch -n 10 ./remote_monitor.sh"
echo "实时日志: tail -f batch_experiments.log"
echo "停止实验: kill $(cat batch_experiments.pid 2>/dev/null)"
echo "=========================================="

