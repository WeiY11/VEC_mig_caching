#!/bin/bash
# VEC项目远程服务器部署脚本
# 用途：将项目上传到AutoDL服务器并配置环境

# ========== 服务器配置 ==========
SERVER_HOST="region-9.autodl.pro"
SERVER_PORT="19287"
SERVER_USER="root"
SERVER_PASSWORD="dfUJkmli0mHk"
REMOTE_DIR="/root/VEC_mig_caching"

echo "=========================================="
echo "VEC项目服务器部署脚本"
echo "目标服务器: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}"
echo "=========================================="

# ========== 步骤1：测试连接 ==========
echo ""
echo "[1/5] 测试服务器连接..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_HOST} "echo '连接成功！'" || {
    echo "❌ 连接失败！请检查服务器信息"
    echo "💡 如果没有安装sshpass，请运行: sudo apt install sshpass (Linux) 或手动连接"
    exit 1
}

# ========== 步骤2：创建远程目录 ==========
echo ""
echo "[2/5] 创建远程项目目录..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${REMOTE_DIR}"

# ========== 步骤3：同步项目文件 ==========
echo ""
echo "[3/5] 上传项目文件（这可能需要几分钟）..."
echo "正在同步文件..."

# 使用rsync同步，排除不必要的文件
sshpass -p "${SERVER_PASSWORD}" rsync -avz --progress \
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
    ./ ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/

# ========== 步骤4：配置服务器环境 ==========
echo ""
echo "[4/5] 配置服务器环境..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'ENDSSH'
cd /root/VEC_mig_caching

echo "检查Python和CUDA环境..."
python --version
nvcc --version 2>/dev/null || echo "⚠️  CUDA未安装或不在PATH中"

echo ""
echo "检查GPU..."
nvidia-smi || echo "⚠️  无法检测GPU"

echo ""
echo "安装Python依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "验证PyTorch和CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}');"

ENDSSH

# ========== 步骤5：创建训练启动脚本 ==========
echo ""
echo "[5/5] 创建远程训练脚本..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'ENDSSH'
cd /root/VEC_mig_caching

cat > start_training.sh << 'EOF'
#!/bin/bash
# 远程训练启动脚本

echo "=========================================="
echo "VEC项目 - 训练启动"
echo "时间: $(date)"
echo "=========================================="

# 激活环境（如果使用虚拟环境）
# source /root/venv/bin/activate

# 训练参数
ALGORITHM=${1:-TD3}
EPISODES=${2:-200}
DEVICE="cuda"

echo ""
echo "训练配置:"
echo "  算法: ${ALGORITHM}"
echo "  轮次: ${EPISODES}"
echo "  设备: ${DEVICE}"
echo ""

# 启动训练（后台运行，输出到日志）
nohup python train_single_agent.py \
    --algorithm ${ALGORITHM} \
    --episodes ${EPISODES} \
    --device ${DEVICE} \
    > training_${ALGORITHM}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "✅ 训练已在后台启动！"
echo "   进程ID: ${PID}"
echo "   日志文件: training_${ALGORITHM}_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "监控命令:"
echo "  查看日志: tail -f training_${ALGORITHM}_*.log"
echo "  查看进程: ps aux | grep train_single_agent"
echo "  停止训练: kill ${PID}"
echo ""
EOF

chmod +x start_training.sh

cat > monitor_training.sh << 'EOF'
#!/bin/bash
# 训练监控脚本

echo "=========================================="
echo "VEC训练监控"
echo "=========================================="

echo ""
echo "运行中的训练进程:"
ps aux | grep -E "(train_single_agent|train_multi_agent)" | grep -v grep

echo ""
echo "GPU使用情况:"
nvidia-smi

echo ""
echo "最新训练日志 (最后20行):"
if ls training_*.log 1> /dev/null 2>&1; then
    LATEST_LOG=$(ls -t training_*.log | head -1)
    echo "日志文件: ${LATEST_LOG}"
    echo "----------------------------------------"
    tail -20 ${LATEST_LOG}
else
    echo "未找到训练日志"
fi

EOF

chmod +x monitor_training.sh

echo "✅ 训练脚本创建完成！"
ENDSSH

echo ""
echo "=========================================="
echo "✅ 部署完成！"
echo "=========================================="
echo ""
echo "📝 下一步操作："
echo ""
echo "1️⃣  连接到服务器:"
echo "   ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
echo ""
echo "2️⃣  进入项目目录:"
echo "   cd ${REMOTE_DIR}"
echo ""
echo "3️⃣  启动训练（后台运行）:"
echo "   ./start_training.sh TD3 200        # 训练TD3算法200轮"
echo "   ./start_training.sh SAC 200        # 训练SAC算法200轮"
echo ""
echo "4️⃣  监控训练进度:"
echo "   ./monitor_training.sh              # 查看训练状态"
echo "   tail -f training_*.log             # 实时查看日志"
echo ""
echo "5️⃣  下载训练结果:"
echo "   使用以下命令将结果下载到本地："
echo "   scp -P ${SERVER_PORT} -r ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/results ./results_from_server"
echo ""
echo "=========================================="

