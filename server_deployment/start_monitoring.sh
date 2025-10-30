#!/bin/bash
# ================================================================
# 在服务器上启动 TensorBoard 和日志监控
# ================================================================

cd /root/VEC_mig_caching

echo "========================================="
echo "  启动 TensorBoard 监控系统"
echo "========================================="
echo ""

# ========== 步骤1: 创建日志目录 ==========
echo "[步骤 1/3] 创建 TensorBoard 日志目录..."
mkdir -p runs/batch_experiments
echo "[OK] 目录已创建"
echo ""

# ========== 步骤2: 启动日志监控器 ==========
echo "[步骤 2/3] 启动日志监控器（后台运行）..."

# 查找Python路径
PYTHON_CMD=""
if command -v /root/miniconda3/bin/python &> /dev/null; then
    PYTHON_CMD="/root/miniconda3/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] 找不到 Python！"
    exit 1
fi

echo "使用 Python: $PYTHON_CMD"

nohup $PYTHON_CMD server_deployment/log_to_tensorboard.py \
    --log-file batch_experiments.log \
    --tensorboard-dir runs/batch_experiments \
    > log_monitor.log 2>&1 &

MONITOR_PID=$!
echo "[OK] 日志监控器已启动 (PID: $MONITOR_PID)"
echo "     日志输出: log_monitor.log"
echo ""

# ========== 步骤3: 启动 TensorBoard ==========
echo "[步骤 3/3] 启动 TensorBoard 服务..."
pkill -f tensorboard  # 先停止旧的
sleep 2

nohup tensorboard \
    --logdir=runs/batch_experiments \
    --port=6006 \
    --bind_all \
    > tensorboard.log 2>&1 &

TB_PID=$!
echo "[OK] TensorBoard 已启动 (PID: $TB_PID)"
echo "     端口: 6006"
echo "     日志输出: tensorboard.log"
echo ""

# ========== 显示状态 ==========
echo "========================================="
echo "  启动完成！"
echo "========================================="
echo ""
echo "访问 TensorBoard："
echo "  1. AutoDL 控制台 -> 自定义服务 -> 端口 6006"
echo "  2. 或使用 SSH 隧道："
echo "     ssh -p 47042 -L 6006:localhost:6006 root@region-9.autodl.pro"
echo "     然后访问: http://localhost:6006"
echo ""
echo "监控进程："
echo "  日志监控器 PID: $MONITOR_PID"
echo "  TensorBoard PID: $TB_PID"
echo ""
echo "查看日志："
echo "  tail -f log_monitor.log     # 监控器日志"
echo "  tail -f tensorboard.log     # TensorBoard日志"
echo "  tail -f batch_experiments.log  # 实验日志"
echo ""

