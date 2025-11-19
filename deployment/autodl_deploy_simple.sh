#!/bin/bash
# AutoDL服务器简化部署脚本 - 不依赖sshpass
# 用途：手动输入密码进行部署

# ========== 服务器配置 ==========
SERVER_HOST="region-41.seetacloud.com"
SERVER_PORT="38597"
SERVER_USER="root"
REMOTE_DIR="/root/VEC_mig_caching"

echo "=========================================="
echo "AutoDL VEC项目部署 - 带宽成本对比实验"
echo "目标服务器: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}"
echo "=========================================="
echo ""
echo "⚠️  需要手动输入密码: dXI7ldI+vPec"
echo ""

# ========== 步骤1：测试连接 ==========
echo "[1/5] 测试服务器连接..."
echo "请输入密码..."
ssh -p ${SERVER_PORT} -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_HOST} "echo '✅ 连接成功！'" || {
    echo "❌ 连接失败！请检查："
    echo "   1. AutoDL实例是否启动"
    echo "   2. 端口和密码是否正确"
    echo "   3. 网络连接是否正常"
    exit 1
}

# ========== 步骤2：创建远程目录并上传文件 ==========
echo ""
echo "[2/5] 创建远程目录并上传项目文件..."
echo "请输入密码..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${REMOTE_DIR}"

echo ""
echo "正在上传项目文件（需要输入密码，这可能需要几分钟）..."
rsync -avz --progress \
    -e "ssh -p ${SERVER_PORT} -o StrictHostKeyChecking=no" \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'results/' \
    --exclude 'models/' \
    --exclude 'academic_figures/' \
    --exclude '*.png' \
    --exclude '*.pdf' \
    --exclude 'test_results/' \
    --exclude '.vscode' \
    --exclude 'venv' \
    ./ ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/ || {
    echo "❌ 文件上传失败！"
    exit 1
}

# ========== 步骤3：配置环境并创建脚本 ==========
echo ""
echo "[3/5] 配置环境并创建管理脚本..."
echo "请输入密码..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} bash << 'ENDSSH'
cd /root/VEC_mig_caching

echo "检查Python和CUDA环境..."
python --version
nvcc --version 2>/dev/null || echo "⚠️  CUDA未安装"

echo ""
echo "检查GPU..."
nvidia-smi || echo "⚠️  无法检测GPU"

echo ""
echo "安装Python依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || {
    echo "⚠️  清华镜像失败，使用默认源..."
    pip install -r requirements.txt
}

echo ""
echo "验证PyTorch和CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())];" || echo "⚠️  PyTorch验证失败"

echo ""
echo "创建管理脚本..."

# 创建启动脚本
cat > start_bandwidth_experiment.sh << 'EOF'
#!/bin/bash
echo "=========================================="
echo "VEC项目 - RSU计算资源对比实验"
echo "时间: $(date)"
echo "=========================================="
echo ""
echo "实验配置:"
echo "  实验类型: rsu_compute"
echo "  RSU计算资源档位: default (5档)"
echo "  训练轮次: 1200 (TD3), 300 (启发式)"
echo "  随机种子: 42"
echo "  优化启发式: 是"
echo ""

LOG_FILE="bandwidth_experiment_$(date +%Y%m%d_%H%M%S).log"

echo "启动实验（后台运行）..."
nohup python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \
    --experiment-types rsu_compute \
    --rsu-compute-levels default \
    --episodes 1200 \
    --seed 42 \
    --optimize-heuristic \
    > ${LOG_FILE} 2>&1 &

PID=$!
echo $PID > bandwidth_experiment.pid

echo ""
echo "✅ 实验已在后台启动！"
echo "   进程ID: ${PID}"
echo "   日志文件: ${LOG_FILE}"
echo ""
echo "监控命令:"
echo "  tail -f ${LOG_FILE}"
echo "  watch -n 5 nvidia-smi"
echo ""
EOF

chmod +x start_bandwidth_experiment.sh

# 创建监控脚本
cat > monitor_experiment.sh << 'EOF'
#!/bin/bash
echo "=========================================="
echo "VEC实验监控 - $(date)"
echo "=========================================="
echo ""
echo "GPU使用情况:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""

if [ -f bandwidth_experiment.pid ]; then
    PID=$(cat bandwidth_experiment.pid)
    echo "实验进程ID: ${PID}"
    if ps -p ${PID} > /dev/null 2>&1; then
        echo "状态: ✅ 正在运行"
    else
        echo "状态: ❌ 已停止"
    fi
else
    echo "⚠️  未找到进程ID文件"
fi

echo ""
echo "最新日志（最后30行）:"
LATEST_LOG=$(ls -t bandwidth_experiment_*.log 2>/dev/null | head -1)
if [ -n "${LATEST_LOG}" ]; then
    echo "日志文件: ${LATEST_LOG}"
    echo "----------------------------------------"
    tail -30 ${LATEST_LOG}
else
    echo "未找到训练日志"
fi
EOF

chmod +x monitor_experiment.sh

# 创建停止脚本
cat > stop_experiment.sh << 'EOF'
#!/bin/bash
echo "=========================================="
echo "停止VEC实验"
echo "=========================================="

if [ -f bandwidth_experiment.pid ]; then
    PID=$(cat bandwidth_experiment.pid)
    echo "尝试停止进程: ${PID}"
    
    if ps -p ${PID} > /dev/null 2>&1; then
        kill ${PID}
        echo "✅ 已发送停止信号"
        sleep 2
        
        if ps -p ${PID} > /dev/null 2>&1; then
            echo "⚠️  进程仍在运行，强制停止..."
            kill -9 ${PID}
        fi
        echo "✅ 实验已停止"
    else
        echo "⚠️  进程未运行"
    fi
else
    echo "⚠️  未找到进程ID文件"
fi
EOF

chmod +x stop_experiment.sh

echo "✅ 环境配置完成！"
ENDSSH

# ========== 步骤4：启动实验 ==========
echo ""
echo "[4/5] 启动实验..."
echo "请输入密码..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} bash << 'ENDSSH'
cd /root/VEC_mig_caching
./start_bandwidth_experiment.sh
ENDSSH

# ========== 步骤5：检查状态 ==========
echo ""
echo "[5/5] 检查实验状态..."
sleep 3
echo "请输入密码..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} bash << 'ENDSSH'
cd /root/VEC_mig_caching

if [ -f bandwidth_experiment.pid ]; then
    PID=$(cat bandwidth_experiment.pid)
    if ps -p ${PID} > /dev/null 2>&1; then
        echo ""
        echo "✅ 实验正在运行！"
        echo ""
        echo "最新日志（最后20行）:"
        LATEST_LOG=$(ls -t bandwidth_experiment_*.log 2>/dev/null | head -1)
        if [ -n "${LATEST_LOG}" ]; then
            tail -20 ${LATEST_LOG}
        fi
    else
        echo ""
        echo "❌ 警告：进程可能已终止"
        echo ""
        LATEST_LOG=$(ls -t bandwidth_experiment_*.log 2>/dev/null | head -1)
        if [ -n "${LATEST_LOG}" ]; then
            echo "检查日志："
            tail -30 ${LATEST_LOG}
        fi
    fi
fi
ENDSSH

echo ""
echo "=========================================="
echo "✅ 部署并启动完成！"
echo "=========================================="
echo ""
echo "📝 后续操作："
echo ""
echo "1️⃣  连接到服务器:"
echo "   ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
echo "   密码: dXI7ldI+vPec"
echo ""
echo "2️⃣  监控实验（在服务器上执行）:"
echo "   cd ${REMOTE_DIR}"
echo "   ./monitor_experiment.sh"
echo "   tail -f bandwidth_experiment_*.log"
echo ""
echo "3️⃣  停止实验:"
echo "   ./stop_experiment.sh"
echo ""
echo "4️⃣  下载结果:"
echo "   scp -P ${SERVER_PORT} -r ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/results ./results_from_autodl"
echo ""
echo "=========================================="
echo ""
echo "💡 预计实验时间: 30-38小时"
echo "⚠️  确保AutoDL实例有足够运行时长"
echo ""
