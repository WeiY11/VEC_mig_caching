#!/bin/bash
# AutoDL服务器部署脚本 - 带宽成本对比实验
# 用途：自动化部署并在AutoDL服务器上运行RSU计算资源对比实验

# ========== 服务器配置 ==========
SERVER_HOST="region-41.seetacloud.com"
SERVER_PORT="38597"
SERVER_USER="root"
SERVER_PASSWORD="dXI7ldI+vPec"
REMOTE_DIR="/root/VEC_mig_caching"

# ========== 实验配置 ==========
EXPERIMENT_SCRIPT="experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py"
EXPERIMENT_TYPES="rsu_compute"
RSU_COMPUTE_LEVELS="default"
EPISODES="1200"
SEED="42"
OPTIMIZE_HEURISTIC="--optimize-heuristic"

echo "=========================================="
echo "AutoDL VEC项目部署 - 带宽成本对比实验"
echo "目标服务器: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}"
echo "=========================================="
echo ""
echo "实验配置:"
echo "  实验类型: ${EXPERIMENT_TYPES}"
echo "  RSU计算资源档位: ${RSU_COMPUTE_LEVELS}"
echo "  训练轮次: ${EPISODES}"
echo "  随机种子: ${SEED}"
echo "  优化启发式: 是"
echo ""

# ========== 步骤1：测试连接 ==========
echo "[1/6] 测试服务器连接..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_HOST} "echo '✅ 连接成功！'" || {
    echo "❌ 连接失败！请检查服务器信息"
    echo ""
    echo "💡 如果没有安装sshpass:"
    echo "   Linux/WSL: sudo apt install sshpass"
    echo "   macOS: brew install hudochenkov/sshpass/sshpass"
    echo ""
    echo "💡 或者手动连接:"
    echo "   ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
    echo "   密码: ${SERVER_PASSWORD}"
    exit 1
}

# ========== 步骤2：创建远程目录 ==========
echo ""
echo "[2/6] 创建远程项目目录..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${REMOTE_DIR}"

# ========== 步骤3：同步项目文件 ==========
echo ""
echo "[3/6] 上传项目文件（这可能需要几分钟）..."
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
    --exclude '.vscode' \
    --exclude 'venv' \
    ./ ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/ || {
    echo "❌ 文件同步失败！"
    exit 1
}

# ========== 步骤4：配置服务器环境 ==========
echo ""
echo "[4/6] 配置服务器环境..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'ENDSSH'
cd /root/VEC_mig_caching

echo "检查Python和CUDA环境..."
python --version
nvcc --version 2>/dev/null || echo "⚠️  CUDA未安装或不在PATH中"

echo ""
echo "检查GPU..."
nvidia-smi || echo "⚠️  无法检测GPU"

echo ""
echo "安装Python依赖（使用清华镜像加速）..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || {
    echo "⚠️  使用清华镜像失败，尝试默认源..."
    pip install -r requirements.txt
}

echo ""
echo "验证PyTorch和CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())];"

echo ""
echo "✅ 环境配置完成！"
ENDSSH

# ========== 步骤5：创建训练启动脚本 ==========
echo ""
echo "[5/6] 创建远程训练脚本..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} bash << ENDSSH
cd /root/VEC_mig_caching

# 创建实验启动脚本
cat > start_bandwidth_experiment.sh << 'EOF'
#!/bin/bash
# 带宽成本对比实验启动脚本

echo "=========================================="
echo "VEC项目 - RSU计算资源对比实验"
echo "时间: \$(date)"
echo "=========================================="

# 实验参数
EXPERIMENT_SCRIPT="${EXPERIMENT_SCRIPT}"
EXPERIMENT_TYPES="${EXPERIMENT_TYPES}"
RSU_COMPUTE_LEVELS="${RSU_COMPUTE_LEVELS}"
EPISODES="${EPISODES}"
SEED="${SEED}"

echo ""
echo "实验配置:"
echo "  脚本: \${EXPERIMENT_SCRIPT}"
echo "  实验类型: \${EXPERIMENT_TYPES}"
echo "  RSU计算资源档位: \${RSU_COMPUTE_LEVELS}"
echo "  训练轮次: \${EPISODES}"
echo "  随机种子: \${SEED}"
echo "  优化启发式: 是"
echo ""

# 创建日志文件名
LOG_FILE="bandwidth_experiment_\$(date +%Y%m%d_%H%M%S).log"

echo "启动实验（后台运行）..."
nohup python \${EXPERIMENT_SCRIPT} \\
    --experiment-types \${EXPERIMENT_TYPES} \\
    --rsu-compute-levels \${RSU_COMPUTE_LEVELS} \\
    --episodes \${EPISODES} \\
    --seed \${SEED} \\
    ${OPTIMIZE_HEURISTIC} \\
    > \${LOG_FILE} 2>&1 &

PID=\$!
echo \$PID > bandwidth_experiment.pid

echo ""
echo "✅ 实验已在后台启动！"
echo "   进程ID: \${PID}"
echo "   日志文件: \${LOG_FILE}"
echo ""
echo "监控命令:"
echo "  查看日志: tail -f \${LOG_FILE}"
echo "  实时监控: watch -n 5 nvidia-smi"
echo "  查看进程: ps aux | grep python"
echo "  停止实验: kill \${PID}"
echo ""
echo "保存进程信息..."
echo "PID: \${PID}" > experiment_info.txt
echo "日志: \${LOG_FILE}" >> experiment_info.txt
echo "开始时间: \$(date)" >> experiment_info.txt
echo ""
EOF

chmod +x start_bandwidth_experiment.sh

# 创建监控脚本
cat > monitor_experiment.sh << 'EOF'
#!/bin/bash
# 实验监控脚本

echo "=========================================="
echo "VEC实验监控"
echo "时间: \$(date)"
echo "=========================================="

echo ""
echo "运行中的Python进程:"
ps aux | grep python | grep -v grep

echo ""
echo "GPU使用情况:"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv

echo ""
if [ -f bandwidth_experiment.pid ]; then
    PID=\$(cat bandwidth_experiment.pid)
    echo "实验进程ID: \${PID}"
    if ps -p \${PID} > /dev/null 2>&1; then
        echo "状态: ✅ 正在运行"
    else
        echo "状态: ❌ 已停止"
    fi
else
    echo "⚠️  未找到进程ID文件"
fi

echo ""
echo "最新训练日志 (最后30行):"
if ls bandwidth_experiment_*.log 1> /dev/null 2>&1; then
    LATEST_LOG=\$(ls -t bandwidth_experiment_*.log | head -1)
    echo "日志文件: \${LATEST_LOG}"
    echo "----------------------------------------"
    tail -30 \${LATEST_LOG}
else
    echo "未找到训练日志"
fi

echo ""
echo "磁盘使用情况:"
df -h | grep -E '(Filesystem|/root)'

EOF

chmod +x monitor_experiment.sh

# 创建停止脚本
cat > stop_experiment.sh << 'EOF'
#!/bin/bash
# 停止实验脚本

echo "=========================================="
echo "停止VEC实验"
echo "=========================================="

if [ -f bandwidth_experiment.pid ]; then
    PID=\$(cat bandwidth_experiment.pid)
    echo "尝试停止进程: \${PID}"
    
    if ps -p \${PID} > /dev/null 2>&1; then
        kill \${PID}
        echo "✅ 已发送停止信号"
        sleep 2
        
        if ps -p \${PID} > /dev/null 2>&1; then
            echo "⚠️  进程仍在运行，强制停止..."
            kill -9 \${PID}
        fi
        
        echo "✅ 实验已停止"
    else
        echo "⚠️  进程未运行"
    fi
else
    echo "⚠️  未找到进程ID文件，尝试查找Python进程..."
    pkill -f "run_bandwidth_cost_comparison.py"
    echo "✅ 已尝试停止相关Python进程"
fi

echo ""
echo "当前Python进程:"
ps aux | grep python | grep -v grep

EOF

chmod +x stop_experiment.sh

echo "✅ 训练脚本创建完成！"
ENDSSH

# ========== 步骤6：启动实验 ==========
echo ""
echo "[6/6] 启动实验..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} bash << 'ENDSSH'
cd /root/VEC_mig_caching
./start_bandwidth_experiment.sh

# 等待几秒确认启动
sleep 3

# 检查进程是否在运行
if [ -f bandwidth_experiment.pid ]; then
    PID=$(cat bandwidth_experiment.pid)
    if ps -p ${PID} > /dev/null 2>&1; then
        echo ""
        echo "✅ 确认：实验正在运行"
        echo ""
        echo "显示最新日志（最后20行）:"
        LATEST_LOG=$(ls -t bandwidth_experiment_*.log | head -1)
        tail -20 ${LATEST_LOG}
    else
        echo ""
        echo "❌ 警告：进程可能已终止，请检查日志"
        LATEST_LOG=$(ls -t bandwidth_experiment_*.log | head -1)
        tail -30 ${LATEST_LOG}
    fi
fi
ENDSSH

echo ""
echo "=========================================="
echo "✅ 部署并启动完成！"
echo "=========================================="
echo ""
echo "📝 监控和管理命令："
echo ""
echo "1️⃣  连接到服务器:"
echo "   ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
echo "   密码: ${SERVER_PASSWORD}"
echo ""
echo "2️⃣  监控实验进度:"
echo "   cd ${REMOTE_DIR}"
echo "   ./monitor_experiment.sh              # 查看实验状态"
echo "   tail -f bandwidth_experiment_*.log   # 实时查看日志"
echo "   watch -n 5 nvidia-smi                # 监控GPU"
echo ""
echo "3️⃣  停止实验:"
echo "   ./stop_experiment.sh                 # 安全停止"
echo ""
echo "4️⃣  下载实验结果:"
echo "   scp -P ${SERVER_PORT} -r ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/results ./results_from_autodl"
echo ""
echo "5️⃣  检查实验配置:"
echo "   cat experiment_info.txt              # 查看实验信息"
echo ""
echo "=========================================="
echo ""
echo "💡 预计实验时间："
echo "   - TD3策略 (1200轮): ~24-30小时"
echo "   - 启发式策略 (300轮): ~6-8小时"
echo "   - 总计: ~30-38小时 (使用GPU加速)"
echo ""
echo "⚠️  注意事项："
echo "   1. 确保AutoDL实例有足够的运行时长（建议至少40小时）"
echo "   2. 实验会在后台运行，可以断开SSH连接"
echo "   3. 定期检查日志确保实验正常进行"
echo "   4. 结果保存在 results/parameter_sensitivity/ 目录"
echo ""
echo "=========================================="
