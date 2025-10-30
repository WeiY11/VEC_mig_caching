#!/bin/bash
# 启动TensorBoard监控批量实验

cd /root/VEC_mig_caching

echo "=========================================="
echo "启动TensorBoard监控"
echo "时间: $(date)"
echo "=========================================="
echo ""

# 检查是否安装tensorboard
if ! python -c "import tensorboard" 2>/dev/null; then
    echo "安装TensorBoard..."
    pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# 创建日志目录
mkdir -p runs/batch_experiments

# 启动TensorBoard
echo "启动TensorBoard服务..."
nohup tensorboard --logdir=runs/batch_experiments --port=6006 --bind_all > tensorboard.log 2>&1 &

TB_PID=$!
echo $TB_PID > tensorboard.pid

sleep 2

# 检查是否启动成功
if ps -p $TB_PID > /dev/null 2>&1; then
    echo "✅ TensorBoard已启动！"
    echo ""
    echo "进程ID: $TB_PID"
    echo "端口: 6006"
    echo "日志: tensorboard.log"
    echo ""
    echo "【本地访问方法】"
    echo "1. 在本地计算机新开一个PowerShell/终端"
    echo "2. 运行以下命令建立SSH隧道:"
    echo "   ssh -p 47042 -L 6006:localhost:6006 root@region-9.autodl.pro"
    echo "3. 在浏览器中访问: http://localhost:6006"
    echo ""
    echo "或者直接访问（如果服务器有公网IP）:"
    echo "   http://region-9.autodl.pro:6006"
    echo ""
    echo "停止TensorBoard:"
    echo "   kill $TB_PID"
else
    echo "❌ TensorBoard启动失败，请查看日志"
    tail -20 tensorboard.log
fi

